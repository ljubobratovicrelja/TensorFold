#!/usr/bin/env python3
"""Write the target's per-layer hyper-connection input mixes as weight-only FP8 e4m3, one scale a row.

Each of the 48 decoder layers carries four of these matrices:

  layers.{i}.attn_hyper_connection.input_mix_weight_down   bf16 [320, 10240]   6.55 MB
  layers.{i}.attn_hyper_connection.input_mix_weight_up     bf16 [10240, 320]   6.55 MB
  layers.{i}.mlp_hyper_connection.input_mix_weight_down    bf16 [320, 10240]   6.55 MB
  layers.{i}.mlp_hyper_connection.input_mix_weight_up      bf16 [10240, 320]   6.55 MB

1.26 GB of bf16 read twice a decode round -- once in the attention branch's input mix
(`conv_attn_in`) and once in the MoE's (`conv_mlp_in`), 11.4 ms together at 73-131 GB/s, the least
efficient stages of the pass. A one-byte-a-element copy is 0.63 GB of codes plus 4.1 MB of row
scales, so each stage's weight stream halves for the same tensor-core MACs (`qmmf`'s FP8G lane,
bf16 activations, f32 accumulate: `zig/src/cuda/fp8.zig`, the same lane the FP8 `lm_head` arm reads).

Unlike that head arm there is no exact rescore here: every channel of the down and up matrices
contributes to the mixed row, and the error is then followed by a normalisation and a residual path.
So the arm is judged end to end (output agreement on the served prompts), and this script's job is to
make the error as small as RTN can make it and to report it per tensor.

    python3 tools/dflash2/quantize/quantize_hc_fp8.py --src SNAP --out artifacts/hc_fp8

CPU only, numpy and the standard library (no torch, no safetensors, no GPU). Deterministic: the same
source gives the same bytes and sha256, and `manifest.json` carries no wall-clock timestamp.

Scale direction is `fp8.zig`'s: the stored f32 is the **multiplier** a row's codes are read with
(`dequant = code * scale`), and it is small (amax / 448). Quantisation is plain round-to-nearest
(RTN) with no clipping search and no calibration: `scale = amax(row) / 448`, one scale a row, and a
row of zeros takes 1.0 rather than 0 so its dequantisation is still zero (`ZERO_BLOCK_SCALE`). The
rounder is `quantize_lm_head.quantize_rows`, i.e. `dflash_fp8.to_e4m3` + `encode_codes`: the same
arithmetic the FP8 head arm was shipped with, not a second implementation.

Three facts about the artifact the loader depends on, all in `manifest.json`:

* the codes files are **row-major** `[rows, cols]` e4m3, in the engine's own orientation: `down` is
  `[low + inject, wide]` with the checkpoint's 320 mix rows first and the layer's 4
  `block_inject_weight` rows appended (that is the matrix the engine builds, `n_down`), `up` is
  `[wide, low]`. The loader permutes them into qmmf's fragment order itself
  (`cuda/fp8.zig:packCodes`), so the file stays inspectable and the one implementation of that
  permutation is the tested one.
* the scales files are `[rows]` little-endian float32, one per output row, in row order.
* the inject rows are inside the `down` table because the engine's down projection is one GEMM over
  `n_down = 320 + 4` rows; their error is reported separately below (`inject` stats) so a reader can
  see what sharing the table cost them. They are 0.08 MB of the 1.26 GB stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dflash_fp8 as F  # noqa: E402
import quantize_lm_head as H  # noqa: E402  (quantize_rows / summarize: the shipped per-row RTN)

TOOL_VERSION = "1.0.0"
FORMAT = "flashnext-hc-fp8-rtn"
MANIFEST = "manifest.json"
GEOMETRY = "geometry.json"
LOW = 320  # the mix's rank, the layer's normed row collapsed to it
WIDE = 10240  # 4 streams x 2560 hidden
INJECT = 4  # block_inject_weight rows, appended to `down` as the engine does
DEFAULT_CHUNK = 1024  # output rows a pass

# `model.language_model.layers.7.attn_hyper_connection.input_mix_weight_up.weight` in this
# checkpoint; the text prefix moves in other exports, so the pattern is on the layer index alone.
TENSOR = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(attn|mlp)_hyper_connection\."
    r"(input_mix_weight_down|input_mix_weight_up|block_inject_weight)\.weight$"
)
# The MTP head has its own `mtp.layers.0.attn_hyper_connection.*`, the same names one namespace over and
# not a decoder layer the engine's flag arms: it would collide with layer 0 under the pattern above.
MTP = re.compile(r"(?:^|\.)mtp\.")
KINDS = ("attn", "mlp")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def codes_name(layer: int, kind: str, which: str) -> str:
    return f"l{layer:02d}.{kind}.{which}.fp8.e4m3.bin"


def scales_name(layer: int, kind: str, which: str) -> str:
    return f"l{layer:02d}.{kind}.{which}.fp8.scales.f32"


def scan(src: Path) -> dict[tuple, tuple[str, str]]:
    """Every hyper-connection input-mix tensor in `src`: {(layer, kind, which): shard name}."""
    index = src / "model.safetensors.index.json"
    if not index.is_file():
        raise SystemExit(f"{src}: needs model.safetensors.index.json (a sharded checkpoint)")
    weight_map = json.loads(index.read_text())["weight_map"]
    found: dict[tuple, tuple[str, str]] = {}
    for name, shard in weight_map.items():
        if MTP.search(name):
            continue
        m = TENSOR.search(name)
        if m is None:
            continue
        key = (int(m.group(1)), m.group(2), m.group(3))
        if key in found:
            raise SystemExit(f"two tensors claim layer {key[0]} {key[1]} {key[2]}: {found[key][0]} and {name}")
        found[key] = (name, shard)
    if not found:
        raise SystemExit(f"{src}: no layers.{{i}}.{{attn,mlp}}_hyper_connection.input_mix_weight_* tensors")
    return found


def shapes_of(found: dict) -> list[int]:
    """The layer indices, checked contiguous and complete: every layer has both branches' three tensors."""
    layers = sorted({k[0] for k in found})
    if layers != list(range(layers[0], layers[0] + len(layers))):
        raise SystemExit(f"layer indices are not contiguous: {layers[:4]} .. {layers[-4:]}")
    for layer in layers:
        for kind in KINDS:
            for which in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
                if (layer, kind, which) not in found:
                    raise SystemExit(f"layer {layer} {kind} has no {which}: refusing a partial artifact")
    return layers


def quantize_tensor(w: np.ndarray, labels: np.ndarray, chunk: int) -> tuple[bytes, bytes, dict]:
    """One weight [rows, cols]: row-major e4m3 codes, f32 row scales, per-row (max, mean, rel) errors.

    `labels` says which rows are inject rows (1) or mix rows (0): the summaries are split on it, so
    the four columns the engine reads as a gate are reported apart from the 320 that carry the mix.
    """
    rows, cols = w.shape
    codes_out, scales_out = bytearray(), bytearray()
    per_row = {k: np.empty(rows, dtype=np.float32) for k in ("max_abs_err", "mean_abs_err", "rel_l2")}
    err2_total = ref2_total = 0.0
    zeros = saturated = 0
    max_code = 0
    for r0 in range(0, rows, chunk):
        n = min(chunk, rows - r0)
        block = w[r0 : r0 + n]
        codes, scale, stats = H.quantize_rows(block)
        deq = F.decode_codes(codes) * scale[:, None]
        codes_out += codes.tobytes()
        scales_out += scale.astype("<f4").tobytes()
        for k in per_row:
            per_row[k][r0 : r0 + n] = stats[k]
        e = (deq.astype(np.float64) - block.astype(np.float64)) ** 2
        r = block.astype(np.float64) ** 2
        err2_total += float(e.sum())
        ref2_total += float(r.sum())
        zeros += int((scale == np.float32(F.ZERO_BLOCK_SCALE)).sum())
        saturated += int(((codes & 0x7F) == 0x7E).sum())
        max_code = max(max_code, int(np.max(codes & 0x7F)))
    if len(codes_out) != rows * cols or len(scales_out) != rows * 4:
        raise SystemExit(f"wrote {len(codes_out)}/{len(scales_out)} bytes, expected {rows * cols}/{rows * 4}")
    inj = labels == 1
    summary: dict = {
        "rel_l2": float(np.sqrt(err2_total / ref2_total)) if ref2_total > 0 else 0.0,
        "rel_l2_per_row": H.summarize(per_row["rel_l2"]),
        "max_abs_err": H.summarize(per_row["max_abs_err"]),
        "mean_abs_err": H.summarize(per_row["mean_abs_err"]),
        "rows": rows,
        "rows_inject": int(inj.sum()),
        "rows_with_zero_scale": zeros,
        "saturated_codes": saturated,
        "max_code_magnitude": max_code,
    }
    if inj.any() and (~inj).any():
        summary["mix"] = {
            "rows": int((~inj).sum()),
            "rel_l2": float(np.sqrt(per_row["rel_l2"][~inj].astype(np.float64).__pow__(2).mean())),
            "rel_l2_per_row": H.summarize(per_row["rel_l2"][~inj]),
        }
        summary["inject"] = {
            "rows": int(inj.sum()),
            "rel_l2": float(np.sqrt(per_row["rel_l2"][inj].astype(np.float64).__pow__(2).mean())),
            "rel_l2_per_row": H.summarize(per_row["rel_l2"][inj]),
            "max_abs_err": H.summarize(per_row["max_abs_err"][inj]),
        }
    return bytes(codes_out), bytes(scales_out), summary


class Shards:
    """The shards holding the hyper-connection tensors, mapped on demand and dropped oldest first.

    A layer's three source tensors are usually in one shard but not always, and the checkpoint has two
    dozen: reading one preview at a time would re-read a 5 GB header per tensor, and mapping all of
    them at once invites page-cache thrash. A handful of live mappings covers a layer's neighbours.
    """

    def __init__(self, src: Path, cap: int = 4):
        self.src, self.cap, self.live = src, cap, {}
        self.order: list[str] = []

    def _map(self, shard: str):
        if shard in self.live:
            self.order.remove(shard)
            self.order.append(shard)
            return self.live[shard]
        path = self.src / shard
        header, data_start = F.read_header(path)
        f = open(path, "rb")
        blob = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        got = self.live[shard] = (f, blob, memoryview(blob), header, data_start)
        self.order.append(shard)
        while len(self.order) > self.cap:
            self._drop(self.order[0])
        return got

    def _drop(self, shard: str) -> None:
        f, blob, view, _, _ = self.live.pop(shard)
        self.order.remove(shard)
        view.release()  # the mapping cannot close while a view of it is exported
        blob.close()
        f.close()

    def read(self, shard: str, name: str, which: str) -> tuple[np.ndarray, dict]:
        """One bf16 tensor as float32 [rows, cols], plus the manifest record of where it came from."""
        _, _, view, header, data_start = self._map(shard)
        entry = header[name]
        if entry["dtype"] != "BF16":
            raise SystemExit(f"{name} is {entry['dtype']}, this arm quantises bf16 matrices")
        rows, cols = (int(x) for x in entry["shape"])
        want = {"input_mix_weight_down": (LOW, WIDE), "input_mix_weight_up": (WIDE, LOW),
                "block_inject_weight": (INJECT, WIDE)}[which]
        if (rows, cols) != want:
            raise SystemExit(f"{name} is {rows} x {cols}, expected {want[0]} x {want[1]}")
        if cols % 64 != 0:
            raise SystemExit(f"{name} has {cols} inputs, not a multiple of the kernel's 64")
        raw = F.tensor_bytes(view, data_start, header, name)
        digest = hashlib.sha256(raw).hexdigest()
        w = F.bf16_to_f32(raw).reshape(rows, cols)
        # a copy: the source is a view of a mapping this cache may close under the caller
        return np.array(w, dtype=np.float32, copy=True, order="C"), {
            "tensor": name, "shard": shard, "dtype": entry["dtype"], "shape": [rows, cols],
            "tensor_bytes": rows * cols * 2, "tensor_sha256": digest,
        }

    def close(self) -> None:
        while self.order:
            self._drop(self.order[0])


def self_test() -> None:
    """A down table with inject rows: the appended rows are labelled, and the shapes are the engine's."""
    w = np.zeros((LOW + INJECT, 64), dtype=np.float32)
    w[:, 0] = np.arange(1, LOW + INJECT + 1)
    w[-1, :] = 1.0
    labels = np.zeros(LOW + INJECT, dtype=np.uint8)
    labels[LOW:] = 1
    codes, scales, stats = quantize_tensor(w, labels, 128)
    assert len(codes) == (LOW + INJECT) * 64 and len(scales) == (LOW + INJECT) * 4, (len(codes), len(scales))
    assert stats["rows"] == LOW + INJECT and stats["rows_inject"] == INJECT, stats
    assert stats["mix"]["rows"] == LOW and stats["inject"]["rows"] == INJECT, stats
    # a 7-bit + sign table never writes the NaN code, and this one's error is the format's own
    assert stats["max_code_magnitude"] <= 0x7E, stats["max_code_magnitude"]
    assert stats["rel_l2"] < 0.05, stats["rel_l2"]
    print("self-test: ok", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--src", type=Path, help="the served checkpoint directory")
    ap.add_argument("--out", type=Path, help="the artifact directory to write")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help=f"output rows a pass (default {DEFAULT_CHUNK})")
    ap.add_argument("--layers", type=int, default=0, help="write only the first N layers (a smoke test: the engine refuses a layer index past the manifest's count)")
    ap.add_argument("--force", action="store_true", help="write into a non-empty --out")
    ap.add_argument("--self-test", action="store_true", help="check the rounder and the row split, then exit")
    args = ap.parse_args(argv)

    self_test()
    if args.self_test:
        return 0
    if args.src is None or args.out is None:
        ap.error("--src and --out are required")
    if args.chunk < 1:
        ap.error("--chunk must be positive")

    src, out = args.src.resolve(), args.out.resolve()
    found = scan(src)
    layers = shapes_of(found)
    if out == src:
        raise SystemExit("--out is the source checkpoint")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty; pass --force to replace its files")

    if args.layers > 0:
        if args.layers > len(layers):
            ap.error(f"--layers {args.layers} but the checkpoint has {len(layers)}")
        keep = set(layers[: args.layers])
        found = {k: v for k, v in found.items() if k[0] in keep}
        layers = layers[: args.layers]

    out.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict] = {}
    sums: dict[str, tuple[str, int]] = {}
    shards = Shards(src)
    try:
        for layer in layers:
            for kind in KINDS:
                entry = {"files": {}, "source": {}}

                def emit(key: str, w: np.ndarray, labels: np.ndarray, srcs: dict) -> None:
                    codes, scales, stats = quantize_tensor(w, labels, args.chunk)
                    cn, sn = codes_name(layer, kind, key), scales_name(layer, kind, key)
                    (out / cn).write_bytes(codes)
                    (out / sn).write_bytes(scales)
                    sums[cn] = (sha256_bytes(codes), len(codes))
                    sums[sn] = (sha256_bytes(scales), len(scales))
                    entry["files"][key] = {
                        "codes": {"file": cn, "shape": list(w.shape), "dtype": "uint8", "order": "row-major"},
                        "scales": {"file": sn, "shape": [w.shape[0]], "dtype": "<f4", "order": "one per output row"},
                    }
                    entry["source"].update(srcs)
                    if key == "down":
                        # the mix rows and the inject rows are the same table but not the same job
                        entry["source"]["input_mix_weight_down"]["errors"] = stats["mix"]
                        entry["source"]["block_inject_weight"]["errors"] = stats["inject"]
                        entry["source"]["down_table"] = {
                            "file": cn, "shape": list(w.shape), "rows_inject": INJECT,
                            "errors": {k: v for k, v in stats.items() if k not in ("mix", "inject")},
                        }
                    else:
                        entry["source"]["input_mix_weight_up"]["errors"] = stats

                # down: the 320 mix rows and then this layer's 4 inject rows, which the engine reads as one GEMM
                parts, srcs = [], {}
                for which in ("input_mix_weight_down", "block_inject_weight"):
                    name, shard = found[(layer, kind, which)]
                    w, got = shards.read(shard, name, which)
                    parts.append(w)
                    srcs[which] = got
                w_down = np.concatenate(parts, axis=0)
                labels = np.zeros(w_down.shape[0], dtype=np.uint8)
                labels[LOW:] = 1
                emit("down", w_down, labels, srcs)
                del parts, w_down

                name, shard = found[(layer, kind, "input_mix_weight_up")]
                w_up, got = shards.read(shard, name, "input_mix_weight_up")
                emit("up", w_up, np.zeros(w_up.shape[0], dtype=np.uint8), {"input_mix_weight_up": got})
                del w_up
                entries[f"layers.{layer}.{kind}"] = entry
    finally:
        shards.close()

    # the loader's copy: the eight fields it checks, so arming a branch 96 times does not parse the
    # ~0.5 MB manifest 96 times. `manifest.json` stays the audit trail (per-tensor sources and errors).
    geometry = {
        "format": FORMAT,
        "schema_version": 1,
        "layers": len(layers),
        "shape_down": [LOW + INJECT, WIDE],
        "shape_up": [WIDE, LOW],
        "rows_down": LOW,
        "rows_inject": INJECT,
        "wide": WIDE,
        "low": LOW,
    }
    (out / GEOMETRY).write_text(json.dumps(geometry, indent=1, sort_keys=True) + "\n")
    geo_sha = sha256_file(out / GEOMETRY)
    geo_bytes = (out / GEOMETRY).stat().st_size

    manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "layers": len(layers),
        "layer_indices": [layers[0], layers[-1]],
        "shape_down": [LOW + INJECT, WIDE],
        "shape_up": [WIDE, LOW],
        "rows_down": LOW,
        "rows_inject": INJECT,
        "wide": WIDE,
        "low": LOW,
        "tool": {
            "name": "tools/dflash2/quantize/quantize_hc_fp8.py",
            "version": TOOL_VERSION,
            "layout": "row-major e4m3 codes + one f32 scale an output row; the loader permutes the codes into qmmf's fragment order (zig/src/cuda/fp8.zig:packCodes) and the scales into its [npad/64][k/64][64] fp32 tile",
            "rounder": "tools/dflash2/quantize/quantize_lm_head.py:quantize_rows (dflash_fp8.to_e4m3 + encode_codes)",
        },
        "quantization": {
            "precision": "fp8_w8a16",
            "activation_dtype": "BF16",
            "weight_dtype": "F8_E4M3",
            "scale_dtype": "F32",
            "scale_rule": "per-output-row max-abs: scale[row] = amax(|w[row]|) / 448",
            "code_rule": "code = round_to_nearest_even(w / scale), saturating at 448",
            "dequant": "w ~= code * scale",
            "code_grid": "e4m3fn, 4-bit exponent bias 7, 3-bit mantissa, subnormals below 2^-6",
            "zero_row_scale": F.ZERO_BLOCK_SCALE,
            "calibration": "none (RTN, deterministic)",
            "inject_rows": "the layer's block_inject_weight rows are appended to the down table (rows 320..323) because the engine reads one [324, wide] GEMM; their error is reported separately",
        },
        "source": {"dir": str(src)},
        "tensors": entries,
        "files": {},
    }
    sums[GEOMETRY] = (geo_sha, geo_bytes)
    for name, (digest, size) in sums.items():
        manifest["files"][name] = {"sha256": digest, "bytes": size}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    manifest_sha = sha256_file(out / MANIFEST)
    manifest_bytes = (out / MANIFEST).stat().st_size
    sums[MANIFEST] = (manifest_sha, manifest_bytes)
    (out / "SHA256SUMS").write_text("".join(f"{d}  {n}\n" for n, (d, _) in sorted(sums.items())))

    # the aggregate a reader checks first, then the one matrix that is worst
    codes_bytes = sum(s[1] for n, s in sums.items() if n.endswith(".e4m3.bin"))
    scales_bytes = sum(s[1] for n, s in sums.items() if n.endswith(".scales.f32"))
    worst = (0.0, "")
    rel, inj_rel = [], []
    for key in sorted(entries):
        for which, src in sorted(entries[key]["source"].items()):
            if "errors" not in src:
                continue
            e = src["errors"]
            rel.append(e["rel_l2"])
            if which != "block_inject_weight" and e["rel_l2"] > worst[0]:
                worst = (e["rel_l2"], f"{key}.{which}")
            if which == "block_inject_weight":
                inj_rel.append(e["rel_l2"])
    rel_a = np.array(rel, dtype=np.float64)
    print(f"wrote {out}: {len(entries)} projections, {codes_bytes / 1e9:.3f} GB of codes, {scales_bytes / 1e6:.2f} MB of row scales")
    print(f"rel-L2 a matrix: mean {rel_a.mean() * 100:.4f}%, p99 {np.percentile(rel_a, 99) * 100:.4f}%, "
          f"worst {worst[0] * 100:.4f}% ({worst[1]})")
    if inj_rel:
        print(f"rel-L2 the {len(inj_rel)} inject tables: mean {np.mean(inj_rel) * 100:.4f}%, max {np.max(inj_rel) * 100:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
