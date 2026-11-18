#!/usr/bin/env python3
"""Does a cheaper lm_head keep the candidate set? An offline, CPU-only per-slot top-k fidelity check.

The drafter's `logits_top16_ids` stage scores 7 draft rows through the *target's* `lm_head`
(`[248320, 2560]` bf16, 1,271.4 MB, 6.99 ms of an 18.75 ms round). Two cheaper rankers are queued: the
same head at weight-only FP8 (e4m3, one scale a 128 x 128 block) and the head restricted to a
frequency-ranked vocabulary trim. Both are gated on one question that needs no engine, no GPU and no
training: **does the low-precision ranking preserve the exact bf16 top-16 candidate set, slot by slot?**

This tool answers it over head inputs dumped by the engine. It is personal-computer-only by construction:
pure numpy, no torch, no GPU, no server, and it never loads the model -- it reads `lm_head` alone, out of
its safetensors shard, through a JSON header read plus a memmap. That is 1.27 GB of a 102 GB checkpoint.

    python tools/dflash2/analysis/head_fidelity.py --inputs DIR --ckpt SNAPSHOT --out report.json \\
        [--vocab-ranks FILE] [--slots 7] [--top 16]

`--inputs` is a `.npy` of `[rows, 2560]` bf16-widened rows, or a directory of them concatenated in name
order. Rows are taken round-major: slot `s` of round `r` is row `r * slots + s` (a dump cut mid-round is
allowed; it is flagged in the report rather than refused). Rows are cast down to bf16 before use -- the
engine's own inputs are bf16, and an f32-widened copy carries no more information.

Cost, once: the head is read and FP8-quantized in one chunked pass (40-55 s on 20 CPU cores, 3.3 GB
peak RSS, almost all of it the float64 e4m3 rounding), then every variant is a slice of that one pass --
the trim costs no extra bytes.

`--vocab-ranks` is a rank-ordered id list -- one id a line (a trailing count column is ignored), a single
whitespace/comma separated line, a JSON array, or a `.npy` of ids; `--trim-rows` (default 65536) of it is
kept, in file order. Absent it, only the FP8 and NVFP4 variants are checked.

Reported per variant (bf16+trim, fp8, fp8+trim) and per block slot, against the exact bf16 full-vocabulary
top-16: set recall, exact-set rate, top-1 agreement, the rank each reference candidate lands at under the
cheap ranking, how many reference candidates fall out of the top-16, and the same split by the reference
top-1 probability (the buckets the survey says the damage concentrates in: p >= 0.9, 0.6-0.9, 0.3-0.6,
< 0.3). Slots are separate because per-slot acceptance is 0.740/0.662/0.592/0.621/0.389/0.429/0.333 --
an aggregate number hides the slots that decide the round.

FP8 reuses the repo's own block-FP8 arithmetic (`tools/dflash2/quantize/dflash_fp8.py`, the same
`weight_scale_inv = amax / 448` and round-to-nearest-even e4m3 codes the export and the kernels use), so a
number from here is comparable to a shipped FP8 checkpoint rather than to a private reimplementation.
Weights only: activations stay bf16, accumulated in f32, which is the W8A16 target.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

TOOL_VERSION = "1.1.0"
BLOCK = 128
E4M3_MAX = 448.0
CHUNK_ROWS = 8192  # vocab rows per weight pass: 8192 x 2560 x 4 B = 84 MB of float32
HEAD_NAMES = ("lm_head.weight", "model.language_model.lm_head.weight", "language_model.lm_head.weight")
# (label, low, high) buckets on the reference top-1 probability, the regime §6.2 of the survey splits on.
BUCKETS = (("p>=0.9", 0.9, 2.0), ("0.6<=p<0.9", 0.6, 0.9), ("0.3<=p<0.6", 0.3, 0.6), ("p<0.3", 0.0, 0.3))


# -- shared FP8 arithmetic ------------------------------------------------------------------


def fp8_module():
    """The repo's block-FP8 module (`quantize/dflash_fp8.py`), loaded by path: no package, no torch."""
    path = Path(__file__).resolve().parent.parent / "quantize" / "dflash_fp8.py"
    if not path.is_file():
        raise SystemExit(f"missing {path}: this tool borrows the repo's block-FP8 arithmetic from it")
    spec = importlib.util.spec_from_file_location("dflash_fp8", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path}: not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- NVFP4 arithmetic (ModelOpt layout: e2m1 codes, group-16 e4m3 block scales, fp32 global) ----


E2M1_MAX = 6.0  # the largest magnitude e2m1 represents
E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_GROUP = 16


def e4m3_round(x: np.ndarray) -> np.ndarray:
    """Round non-negative float32 values to the e4m3 grid (3-bit mantissa, max 448), ties to even.

    The block scale of an NVFP4 tensor is itself e4m3, so the dequantised weight carries that
    rounding on top of the 4-bit code; ignoring it would overstate fidelity.
    """
    a = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    a = np.ascontiguousarray(np.minimum(a, E4M3_MAX))
    u = a.view(np.uint32)
    exp = ((u >> 23) & 0xFF).astype(np.int32)
    mant = (u & 0x7FFFFF).astype(np.int32)
    m3 = mant >> 20
    rem = mant & 0xFFFFF
    half = 0x80000
    up = (rem > half) | ((rem == half) & ((m3 & 1) == 1))
    m3 = m3 + up.astype(np.int32)
    carry = m3 >> 3
    m3 = m3 & 7
    exp_out = exp + carry - 127 + 7
    val = (1.0 + m3.astype(np.float32) / 8.0) * np.exp2((exp_out - 7).astype(np.float32))
    # below 2^-6 e4m3 is subnormal with a 2^-9 step; above the max finite it saturates
    tiny = np.round(a / 2.0**-9) * 2.0**-9
    val = np.where(exp_out < 1, tiny, val)
    val = np.where((exp_out > 15) | ((exp_out == 15) & (m3 > 6)), E4M3_MAX, val)
    return np.minimum(np.maximum(val, 0.0), E4M3_MAX).astype(np.float32)


def e2m1_round(t: np.ndarray) -> np.ndarray:
    """Round float32 values onto the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} signed, ties to even.

    A tie is resolved to the candidate with the even index on the magnitude grid, which is the
    round-half-to-even convention of the hardware convert; it moves a vanishing fraction of codes.
    """
    a = np.abs(np.asarray(t, dtype=np.float32))
    idx = np.clip(np.searchsorted(E2M1_GRID, a, side="left"), 1, E2M1_GRID.size - 1)
    lo = E2M1_GRID[idx - 1]
    hi = E2M1_GRID[idx]
    mid = (lo + hi) * 0.5
    take_hi = (a > mid) | ((a == mid) & ((idx % 2) == 0))
    mag = np.where(take_hi, hi, lo)
    return np.copysign(mag, np.asarray(t, dtype=np.float32)).astype(np.float32)


def quantize_nvfp4(w: np.ndarray, gs: np.ndarray | float, group: int = FP4_GROUP) -> tuple[np.ndarray, float]:
    """Weight-only NVFP4 for one block of head rows: returns the dequantised weights and rel-L2.

    `gs` is the per-tensor multiplier, 448 * 6 / amax(head), computed once over the whole tensor the
    way ModelOpt does -- a per-chunk global would flatter the chunk that happens to be largest.
    """
    ws = np.asarray(w, dtype=np.float32) * np.float32(gs)
    blocks = ws.reshape(-1, group)
    bamax = np.max(np.abs(blocks), axis=1)
    scale = e4m3_round(np.maximum(bamax / E2M1_MAX, 1e-12))
    scale = np.where(scale <= 0.0, 1e-12, scale)
    code = e2m1_round(blocks / scale[:, None])
    deq = (code * scale[:, None] / np.float32(gs)).reshape(ws.shape)
    err = float(np.sum((deq.astype(np.float64) - w.astype(np.float64)) ** 2))
    ref = float(np.sum(w.astype(np.float64) ** 2))
    return deq.astype(np.float32), float(np.sqrt(err / ref)) if ref > 0 else 0.0


# -- checkpoint, header-only ----------------------------------------------------------------


def read_header(path: Path) -> tuple[dict, int]:
    """The JSON header of a safetensors file and the offset its data buffer starts at."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SystemExit(f"{path}: too short for a safetensors length prefix")
        (n,) = struct.unpack("<Q", raw)
        if n == 0 or n > 100 * 1024 * 1024:
            raise SystemExit(f"{path}: implausible header length {n}")
        header = json.loads(f.read(n))
    return header, 8 + n


def read_config(snapshot: Path) -> dict:
    """vocab_size and hidden_size, from the top level or from `text_config` (Flash-Next nests them)."""
    cfg = json.loads((snapshot / "config.json").read_text())
    for node in (cfg, cfg.get("text_config", {}), cfg.get("language_config", {})):
        if isinstance(node, dict) and "vocab_size" in node and "hidden_size" in node:
            return {"vocab_size": int(node["vocab_size"]), "hidden_size": int(node["hidden_size"])}
    raise SystemExit(f"{snapshot}/config.json: no vocab_size/hidden_size at the top level or under text_config")


def locate_head(snapshot: Path) -> tuple[Path, str]:
    """The shard holding `lm_head.weight`, by the index when there is one, else by reading headers."""
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        for name in HEAD_NAMES:
            if name in weight_map:
                return snapshot / weight_map[name], name
    for shard in sorted(snapshot.glob("*.safetensors")) + sorted((snapshot / "tensors").glob("*.safetensors")):
        if str(shard).endswith(".index.json"):
            continue
        try:
            header, _ = read_header(shard)
        except (OSError, ValueError):
            continue
        for name in HEAD_NAMES:
            if name in header:
                return shard, name
    raise SystemExit(f"{snapshot}: no lm_head tensor in the index or in any shard header")


class Head:
    """`lm_head` as a read-only bf16 memmap, plus whole-head FP8 quantization in row chunks."""

    def __init__(self, shard: Path, name: str, vocab: int, hidden: int):
        header, data_start = read_header(shard)
        if name not in header:
            raise SystemExit(f"{shard}: no {name}")
        entry = header[name]
        self.dtype = entry["dtype"]
        self.shape = tuple(entry["shape"])
        if self.shape != (vocab, hidden):
            raise SystemExit(f"{name}: shape {self.shape}, config says [{vocab}, {hidden}]")
        if self.dtype != "BF16":
            raise SystemExit(f"{name}: {self.dtype}; this check compares against a bf16 reference head")
        begin, end = entry["data_offsets"]
        want = vocab * hidden * 2
        if end - begin != want:
            raise SystemExit(f"{name}: {end - begin} bytes, expected {want}")
        self.shard = shard
        self.rows = np.memmap(shard, dtype="<u2", mode="r", offset=data_start + begin, shape=(vocab, hidden))

    def f32(self, i0: int, i1: int) -> np.ndarray:
        """Rows [i0, i1) as float32 holding bf16 values exactly (bf16 is f32's top half)."""
        u = np.asarray(self.rows[i0:i1])
        return (u.astype(np.uint32) << np.uint32(16)).view(np.float32)



# -- inputs ----------------------------------------------------------------------------------


def bf16_round(x: np.ndarray) -> np.ndarray:
    """float32 -> float32 holding the nearest bf16 (round to nearest even), which is what the head saw."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


def load_inputs(source: Path, hidden: int) -> tuple[np.ndarray, list[str]]:
    """Every row in `source` (a .npy or a directory of them) as bf16-valued float32 [rows, hidden]."""
    paths = sorted(source.glob("*.npy")) if source.is_dir() else [source]
    if not paths:
        raise SystemExit(f"{source}: no .npy files")
    blocks, names = [], []
    for path in paths:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 1 and arr.size == hidden:
            arr = arr.reshape(1, hidden)
        if arr.ndim != 2 or arr.shape[1] != hidden:
            raise SystemExit(f"{path}: shape {arr.shape}, expected [rows, {hidden}]")
        if arr.dtype == np.uint16:  # raw bf16 bits, as some dumps write them
            as_f32 = (np.asarray(arr).astype(np.uint32) << np.uint32(16)).view(np.float32)
        elif arr.dtype in (np.float32, np.float64, np.float16):
            as_f32 = bf16_round(np.asarray(arr))
        else:
            raise SystemExit(f"{path}: dtype {arr.dtype}; expected float32/float16 or uint16 bf16 bits")
        if not np.all(np.isfinite(as_f32)):
            raise SystemExit(f"{path}: non-finite head input rows")
        blocks.append(as_f32)
        names.append(str(path))
    return np.concatenate(blocks, axis=0), names


def load_vocab_ranks(path: Path, vocab: int) -> np.ndarray:
    """A rank-ordered id list, in file order: one id a line (extra columns ignored), one line, JSON, or .npy."""
    if path.suffix == ".npy":
        ids = np.load(path).astype(np.int64).ravel()
    else:
        text = path.read_text()
        lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if text.lstrip().startswith("["):
            ids = np.asarray(json.loads(text), dtype=np.int64).ravel()
        elif len(lines) == 1:
            ids = np.asarray([int(t) for t in lines[0].replace(",", " ").split()], dtype=np.int64)
        else:
            ids = np.asarray([int(line.split()[0].rstrip(",")) for line in lines], dtype=np.int64)
    if ids.size > vocab:
        raise SystemExit(f"{path}: {ids.size} ids for a {vocab}-token vocabulary")
    if np.unique(ids).size != ids.size or ids.min() < 0 or ids.max() >= vocab:
        raise SystemExit(f"{path}: ids must be unique and in [0, {vocab})")
    return ids


# -- ranking --------------------------------------------------------------------------------


def top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """The k highest logits' indices, ties to the lower index: the set the selector would carry.

    The tie rule is exact, not whatever `argpartition` happens to return: everything above the k-th value
    is in, and the remaining places go to the tied tokens in ascending id order.
    """
    if k >= logits.size:
        return np.argsort(-logits, kind="stable")
    thr = np.partition(logits, logits.size - k)[logits.size - k]
    above = np.flatnonzero(logits > thr)
    if above.size > k:  # impossible unless NaN crept in
        raise ValueError("partition threshold broke: more than k values above it")
    tied = np.flatnonzero(logits == thr)[: k - above.size]
    sel = np.concatenate([above, tied])
    return sel[np.lexsort((sel, -logits[sel]))]


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    e = np.exp(shifted)
    return e / e.sum()


def empty_buckets() -> dict:
    return {label: {"rows": 0, "recall": [], "agree": [], "rank": []} for label, _, _ in BUCKETS}


def rank_of(cand: np.ndarray, token: int, sub_mask: np.ndarray | None) -> int:
    """1 + the number of scored tokens the cheap ranking puts above `token` (positions, not percentiles)."""
    score = cand[token]
    higher = cand > score
    if sub_mask is not None:
        higher = higher & sub_mask
    return 1 + int(np.count_nonzero(higher))


def score_row(
    ref: np.ndarray,
    cand: np.ndarray,
    sub: np.ndarray | None,
    k: int,
    ref_ids: np.ndarray,
    p1: float,
) -> dict:
    """One row's fidelity numbers: reference top-k vs the cheap ranking over `sub` (all tokens if None)."""
    sub_mask = None
    if sub is None:
        cand_ids = top_k(cand, k)
        oov = 0
    else:
        sub_mask = np.zeros(ref.size, dtype=bool)
        sub_mask[sub] = True
        local = top_k(cand[sub], k)
        cand_ids = sub[local]
        oov = int(np.count_nonzero(~sub_mask[ref_ids]))
    inter = np.intersect1d(ref_ids, cand_ids, assume_unique=True)
    ranks, displaced = [], 0
    for token in ref_ids:
        if sub_mask is not None and not sub_mask[token]:
            displaced += 1
            continue
        r = rank_of(cand, int(token), sub_mask)
        ranks.append(r)
        if r > k:
            displaced += 1
    ranks_arr = np.asarray(ranks) if ranks else np.array([], dtype=np.int64)
    return {
        "recall": inter.size / k,
        "exact": float(inter.size == k and oov == 0),
        "agree": float(ref_ids[0] == cand_ids[0]),
        "rank_mean": float(ranks_arr.mean()) if ranks_arr.size else float("nan"),
        "rank_p90": float(np.percentile(ranks_arr, 90)) if ranks_arr.size else float("nan"),
        "rank_max": float(ranks_arr.max()) if ranks_arr.size else float("nan"),
        "displaced": displaced,
        "oov": oov,
        "p1": p1,
    }


def summarize(rows: list[dict]) -> dict:
    """Mean (and worst) of a slot's per-row numbers."""

    def mean(key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else None

    return {
        "rows": len(rows),
        "recall_at_k": mean("recall"),
        "set_exact": mean("exact"),
        "top1_agree": mean("agree"),
        "mean_rank": mean("rank_mean"),
        "p90_rank": mean("rank_p90"),
        "max_rank": max((r["rank_max"] for r in rows if not np.isnan(r["rank_max"])), default=None),
        "mean_p1": mean("p1"),
        "mean_displaced": mean("displaced"),
        "max_displaced": max((r["displaced"] for r in rows), default=0),
        "mean_oov": mean("oov"),
    }


def summarize_bucket(bucket: dict) -> dict:
    out = {"rows": bucket["rows"]}
    for src, key in (("recall", "recall_at_k"), ("agree", "top1_agree"), ("rank", "mean_rank")):
        vals = [v for v in bucket[src] if not np.isnan(v)]
        out[key] = float(np.mean(vals)) if vals else None
    out["flip_rate"] = None if out["top1_agree"] is None else 1.0 - out["top1_agree"]
    return out


# -- self check -----------------------------------------------------------------------------


def self_check(module, head: Head, hidden: int) -> dict:
    """Prove the fast paths on this box: tie-exact top-k, bf16 cast, and the FP8 round trip's grid."""
    rng = np.random.default_rng(1234)
    ok = {}

    logits = rng.standard_normal(20000).astype(np.float32)
    naive = np.argsort(-logits, kind="stable")[:16]
    ok["top16_matches_argsort"] = bool(np.array_equal(top_k(logits, 16), naive))

    tied = np.round(rng.standard_normal(5000).astype(np.float32) * 2.0)  # heavy ties at the boundary
    mine = top_k(tied, 16)
    ok["top16_tie_rule"] = bool(
        np.array_equal(tied[mine], np.sort(tied)[::-1][:16])
        and np.array_equal(mine, np.lexsort((np.arange(tied.size), -tied))[:16])
    )

    x = rng.standard_normal((4, hidden)).astype(np.float32) * 0.05
    back = bf16_round(x)
    ok["bf16_cast_fixed_point"] = bool(np.array_equal(bf16_round(back), back))
    ok["bf16_cast_error"] = float(np.max(np.abs(back.astype(np.float64) - x)))

    w = head.f32(0, BLOCK)
    codes, scale, _ = module.quantize_blockwise(w, BLOCK)
    nb, kb = -(-BLOCK // BLOCK), -(-hidden // BLOCK)
    blocks = w.reshape(nb, BLOCK, kb, BLOCK).transpose(0, 2, 1, 3)
    vals = module.decode_codes(codes)  # [BLOCK, hidden], back in row-major block order
    # the codes are exactly RTN of the normalized block, decode/encode is a fixed point, and the
    # dequantized weight the GEMM multiplies by is those values times their block scale
    ok["fp8_codes_are_rtn"] = bool(
        np.array_equal(module.to_e4m3(blocks / scale[:, :, None, None]).transpose(0, 2, 1, 3).reshape(vals.shape), vals)
    )
    ok["fp8_codes_round_trip"] = bool(np.array_equal(module.encode_codes(vals), codes))
    ok["fp8_dequant_matches"] = bool(
        np.array_equal(
            module.dequantize(codes, scale, BLOCK),
            (vals.reshape(BLOCK, kb, BLOCK) * scale[0][None, :, None]).reshape(BLOCK, hidden),
        )
    )
    ok["fp8_scale_is_amax_over_448"] = bool(
        np.allclose(scale, np.abs(w).reshape(BLOCK, -1, BLOCK).max(axis=(0, 2)) / E4M3_MAX)
    )
    return ok


# -- report ---------------------------------------------------------------------------------


def round_floats(node):
    if isinstance(node, dict):
        return {k: round_floats(v) for k, v in node.items()}
    if isinstance(node, list):
        return [round_floats(v) for v in node]
    if isinstance(node, float):
        return None if np.isnan(node) else round(node, 6)
    return node


def fmt(value, spec=".4f") -> str:
    """A table cell: a missing value (empty bucket, no rows) prints as a dash, not as nan."""
    return "-" if value is None else format(value, spec)


def print_table(report) -> None:
    """One line a slot, one block a variant: the shape the per-slot acceptance curve is read in."""
    k = report["reference"]["top_k"]
    print(f"rows {report['inputs']['rows']}  slots {report['inputs']['slots']}  "
          f"vocab {report['checkpoint']['shape'][0]}  fp8 rel_l2 {fmt(report['fp8']['rel_l2'])}")
    for name, variant in report["variants"].items():
        note = f", {variant['subspace']} rows" + (f", reference top-16 reachable {variant['mean_reachable']:.4f}"
                                                  if variant["trimmed"] else "")
        print(f"\n{name}  ({variant['weights']} weights{note})")
        print(f"  slot  rows  recall{k}  setexact  top1    p1   rank   p90   max  out{k}  "
              f"rec@p>=0.9  rec@p<0.3")
        for slot in variant["per_slot"]:
            b = slot["buckets"]
            print(f"  {slot['slot']:>4}  {slot['rows']:>4}  {fmt(slot['recall_at_k'])}  {fmt(slot['set_exact'])}"
                  f"  {fmt(slot['top1_agree'], '.3f')}  {fmt(slot['mean_p1'], '.3f')}"
                  f"  {fmt(slot['mean_rank'], '.3f')}  {fmt(slot['p90_rank'], '.2f')}"
                  f"  {fmt(slot['max_rank'], '.1f')}  {fmt(slot['mean_displaced'], '.2f')}"
                  f"  {fmt(b['p>=0.9']['recall_at_k'], '.4f')}  {fmt(b['p<0.3']['recall_at_k'], '.4f')}")
        o = variant["overall"]
        print(f"  all  {o['rows']:>4}  {fmt(o['recall_at_k'])}  {fmt(o['set_exact'])}"
              f"  {fmt(o['top1_agree'], '.3f')}  {fmt(o['mean_p1'], '.3f')}"
              f"  {fmt(o['mean_rank'], '.3f')}  {fmt(o['p90_rank'], '.2f')}"
              f"  {fmt(o['max_rank'], '.1f')}  {fmt(o['mean_displaced'], '.2f')}")
    print(f"\ntimings: head read {report['timings']['head_read_s']:.1f}s, fp8 quantize "
          f"{report['timings']['fp8_quantize_s']:.1f}s, gemm {report['timings']['gemm_s']:.1f}s, "
          f"total {report['timings']['total_s']:.1f}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline CPU-only top-16 fidelity of a cheaper lm_head, per block slot.")
    ap.add_argument("--inputs", required=True, help="a .npy of [rows, hidden] or a directory of them")
    ap.add_argument("--ckpt", required=True, help="the target snapshot directory (config.json + safetensors)")
    ap.add_argument("--out", required=True, help="the JSON report to write")
    ap.add_argument("--vocab-ranks", help="rank-ordered token ids: the frequency trim to also check")
    ap.add_argument("--trim-rows", type=int, default=65536, help="how many of the rank list to keep (default 65536)")
    ap.add_argument("--slots", type=int, default=7, help="block slots per round; rows are round-major (default 7)")
    ap.add_argument("--top", type=int, default=16, help="candidate set size (default 16)")
    ap.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS, help=f"vocab rows per weight pass (default {CHUNK_ROWS})")
    ap.add_argument("--max-vocab-rows", type=int, help="SMOKE TEST ONLY: truncate the head, results are not evidence")
    args = ap.parse_args(argv)

    started = time.time()
    snapshot = Path(os.path.expanduser(args.ckpt)).resolve()
    cfg = read_config(snapshot)
    vocab, hidden = cfg["vocab_size"], cfg["hidden_size"]
    shard, name = locate_head(snapshot)

    x, sources = load_inputs(Path(os.path.expanduser(args.inputs)), hidden)
    rows = x.shape[0]
    partial_round = bool(rows % args.slots)
    if partial_round:  # a dump cut mid-round is harmless; an out-of-order dump is not, but it is silent either way
        print(f"WARNING: {rows} rows is not a multiple of --slots {args.slots}: the last round is partial",
              file=sys.stderr)
    if rows * vocab * 4 * 2 > 8e9:
        raise SystemExit(f"{rows} rows would need {rows * vocab * 8 / 1e9:.1f} GB of logits; split the inputs")

    module = fp8_module()
    head = Head(shard, name, vocab, hidden)
    checks = self_check(module, head, hidden)
    if not all(v is True for k, v in checks.items() if isinstance(v, bool)):
        print(json.dumps(checks, indent=2), file=sys.stderr)
        raise SystemExit("self check failed: refusing to report numbers this build cannot stand behind")
    use_vocab = vocab if args.max_vocab_rows is None else min(vocab, args.max_vocab_rows)
    if use_vocab != vocab:
        print(f"WARNING: --max-vocab-rows truncates the head to {use_vocab}; this run is a smoke test",
              file=sys.stderr)

    # One pass over the head: exact bf16 logits, FP8 logits and NVFP4 logits, chunk by chunk.
    # NVFP4 needs the per-tensor global scale before any block can be quantized, so amax comes first.
    t0 = time.time()
    amax = 0.0
    for i0 in range(0, use_vocab, args.chunk_rows):
        i1 = min(i0 + args.chunk_rows, use_vocab)
        amax = max(amax, float(np.max(np.abs(head.f32(i0, i1)))))
    nv_gs = (E4M3_MAX * E2M1_MAX) / max(amax, 1e-12)
    logits_bf16 = np.empty((rows, use_vocab), dtype=np.float32)
    logits_fp8 = np.empty((rows, use_vocab), dtype=np.float32)
    logits_nvfp4 = np.empty((rows, use_vocab), dtype=np.float32)
    err2 = ref2 = 0.0
    nv_err2 = nv_ref2 = 0.0
    saturated = elements = 0
    gemm_s = read_s = quant_s = nv_quant_s = 0.0
    for i0 in range(0, use_vocab, args.chunk_rows):
        i1 = min(i0 + args.chunk_rows, use_vocab)
        r0 = time.time()
        w = head.f32(i0, i1)
        read_s += time.time() - r0
        q0 = time.time()
        codes, scale, stats = module.quantize_blockwise(w, BLOCK)
        deq = module.dequantize(codes, scale, BLOCK)
        quant_s += time.time() - q0
        q4 = time.time()
        deq4, nv_rel = quantize_nvfp4(w, nv_gs)
        nv_quant_s += time.time() - q4
        err2 += stats["rel_l2"] ** 2 * float(np.sum(w.astype(np.float64) ** 2))
        ref2 += float(np.sum(w.astype(np.float64) ** 2))
        nv_err2 += nv_rel**2 * float(np.sum(w.astype(np.float64) ** 2))
        nv_ref2 += float(np.sum(w.astype(np.float64) ** 2))
        saturated += stats["saturated_codes"]
        elements += w.size
        g0 = time.time()
        logits_bf16[:, i0:i1] = x @ w.T
        logits_fp8[:, i0:i1] = x @ deq.T
        logits_nvfp4[:, i0:i1] = x @ deq4.T
        gemm_s += time.time() - g0
    fp8_stats = {
        "format": "e4m3",
        "block": BLOCK,
        "scale": "weight_scale_inv = block amax / 448, dequant = code * weight_scale_inv",
        "codes": "round-to-nearest-even, saturating at 448",
        "activation": "bf16 head inputs, f32 accumulate (W8A16)",
        "rel_l2": float(np.sqrt(err2 / ref2)) if ref2 > 0 else 0.0,
        "saturated_fraction": saturated / max(1, elements),
        "rows_quantized": int(use_vocab),
    }
    nvfp4_stats = {
        "format": "e2m1",
        "group": FP4_GROUP,
        "global_scale": float(nv_gs),
        "amax": float(amax),
        "scale": "per-tensor multiplier 448*6/amax(head); block scale = e4m3(block amax / 6); "
                 "dequant = code * block_scale / global_scale",
        "codes": "round-to-nearest-even e2m1, block scale round-to-nearest-even e4m3",
        "activation": "bf16 head inputs, f32 accumulate (W4A16)",
        "rel_l2": float(np.sqrt(nv_err2 / nv_ref2)) if nv_ref2 > 0 else 0.0,
        "rows_quantized": int(use_vocab),
    }
    if args.max_vocab_rows is not None:
        print(f"smoke test: ranked over {use_vocab} of {vocab} head rows", file=sys.stderr)

    trims = {}
    if args.vocab_ranks:
        ids = load_vocab_ranks(Path(os.path.expanduser(args.vocab_ranks)), vocab)
        if ids.size < args.trim_rows:
            raise SystemExit(f"{args.vocab_ranks}: {ids.size} ids, fewer than --trim-rows {args.trim_rows}")
        trims["trim"] = ids[: args.trim_rows]
        if use_vocab < vocab:  # smoke test only: keep the trim inside the rows that were actually read
            trims["trim"] = trims["trim"][trims["trim"] < use_vocab]
    variants = {"fp8": {"weights": "fp8", "sub": None}, "nvfp4": {"weights": "nvfp4", "sub": None}}
    if "trim" in trims:
        variants["bf16_trim"] = {"weights": "bf16", "sub": trims["trim"]}
        variants["fp8_trim"] = {"weights": "fp8", "sub": trims["trim"]}
        variants["nvfp4_trim"] = {"weights": "nvfp4", "sub": trims["trim"]}

    k = args.top
    per_variant = {}
    for vname, spec in variants.items():
        sub = spec["sub"]
        logits = {"bf16": logits_bf16, "fp8": logits_fp8, "nvfp4": logits_nvfp4}[spec["weights"]]
        slots = [{"rows": []} for _ in range(args.slots)]
        buckets = [empty_buckets() for _ in range(args.slots)]
        reachable = []
        for r in range(rows):
            ref = logits_bf16[r]
            ref_ids = top_k(ref[:use_vocab], k)
            if sub is not None:
                reachable.append(float(np.count_nonzero(np.isin(sub, ref_ids))) / k)
            p1 = float(np.max(softmax_probs(ref[:use_vocab])))
            row = score_row(ref[:use_vocab], logits[r][:use_vocab], sub, k, ref_ids, p1)
            slot = r % args.slots
            slots[slot]["rows"].append(row)
            for label, lo, hi in BUCKETS:
                if lo <= p1 < hi:
                    b = buckets[slot][label]
                    b["rows"] += 1
                    b["recall"].append(row["recall"])
                    b["agree"].append(row["agree"])
                    b["rank"].append(row["rank_mean"])
                    break
        per_slot = []
        for s in range(args.slots):
            if not slots[s]["rows"]:
                continue
            entry = {"slot": s, **summarize(slots[s]["rows"])}
            entry["buckets"] = {label: summarize_bucket(buckets[s][label]) for label, _, _ in BUCKETS}
            per_slot.append(entry)
        everything = [r for s in slots for r in s["rows"]]
        per_variant[vname] = {
            "weights": spec["weights"],
            "trimmed": sub is not None,
            "subspace": int(use_vocab if sub is None else sub.size),
            "mean_reachable": float(np.mean(reachable)) if reachable else 1.0,
            "overall": summarize(everything),
            "worst_slot_recall_at_k": min(p["recall_at_k"] for p in per_slot),
            "worst_slot_top1_agree": min(p["top1_agree"] for p in per_slot),
            "per_slot": per_slot,
        }

    report = {
        "tool": {"name": "head_fidelity", "version": TOOL_VERSION,
                 "note": "offline, CPU-only, numpy only: no GPU, no torch, no model load"},
        "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint": {"snapshot": str(snapshot), "tensor": name, "shard": str(shard),
                       "dtype": head.dtype, "shape": list(head.shape)},
        "fp8": fp8_stats,
        "nvfp4": nvfp4_stats,
        "inputs": {"source": str(Path(os.path.expanduser(args.inputs))), "files": sources,
                   "rows": int(rows), "slots": args.slots, "hidden": hidden,
                   "partial_last_round": partial_round},
        "reference": {"rule": f"exact bf16 head over all {vocab} rows, top-{k}, ties to the lower id",
                      "top_k": k, "ties": "ascending token id"},
        "self_check": checks,
        "variants": per_variant,
        "timings": {"total_s": round(time.time() - started, 2), "head_read_s": round(read_s, 2),
                    "fp8_quantize_s": round(quant_s, 2), "nvfp4_quantize_s": round(nv_quant_s, 2),
                    "gemm_s": round(gemm_s, 2),
                    "pass_s": round(time.time() - t0, 2)},
        "caveats": [
            "one head, one set of captured rows: this measures the ranking, not end-to-end acceptance",
            "FP8 here is weight-only with bf16 activations and f32 accumulate; an activation-quantized "
            "kernel measures something else",
            "NVFP4 is the ModelOpt weight-only layout: e2m1 codes, a group-16 e4m3 block scale and one "
            "per-tensor fp32 global scale. A different global scale or group size moves the number",
            "NVFP4 dequantises to f32 before the GEMM. On our hardware the win is bytes read, so this "
            "matches the served kernel; a fused FP4 tensor-core path would round differently",
            "the trim variants assume the rank list comes from this model's own generations; ranks do not "
            "transfer between models (survey §9.1)",
            "rows are round-major, slot s of round r at index r*slots+s; a mis-ordered dump silently mixes slots",
        ],
    }
    report = round_floats(report)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print_table(report)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
