#!/usr/bin/env python3
"""Write the target's `lm_head` as weight-only FP8 e4m3 with one scale an output row.

The head is 248,320 x 2,560 bf16 = 1.27 GB, read once a decode round and measured at 5.29 ms, i.e.
at the box's 240 GB/s wall. A one-byte-a-element copy is 0.61 GiB of codes plus 1.0 MB of row
scales, so the stream halves for the same tensor-core MACs (`qmmf`'s FP8G lane, bf16 activations,
f32 accumulate). The rank is not exact, so service never trusts it alone: the engine keeps the top
64 of this table's logits a row as candidates and rescores them against the exact bf16 rows
(`zig/src/families/flashnext/cuda_head_fp8.zig`).

    python3 tools/dflash2/quantize/quantize_lm_head.py --src SNAP --out artifacts/lm_head_fp8

CPU only, numpy and the standard library (no torch, no safetensors, no GPU). Deterministic: the same
source gives the same bytes and sha256, and `manifest.json` carries no wall-clock timestamp.

Scale direction is `fp8.zig`'s: the stored f32 is the **multiplier** a row's codes are read with
(`dequant = code * scale`), and it is small (amax / 448). Quantisation is plain round-to-nearest
(RTN) with no clipping search and no calibration: `scale = amax(row) / 448`, one scale a row, and a
row of zeros takes 1.0 rather than 0 so its dequantisation is still zero (`ZERO_BLOCK_SCALE`).

Two facts about the artifact the loader depends on, both in `manifest.json`:

* `head.fp8.e4m3.bin` is `[vocab, hidden]` e4m3 codes in **row-major** order — the loader permutes
  them into qmmf's fragment order itself (`cuda/fp8.zig:packCodes`), so the file is inspectable and
  the one implementation of that permutation is the tested one.
* `head.fp8.scales.f32` is `[vocab]` little-endian float32, one per output row, in row order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dflash_fp8 as F  # noqa: E402

TOOL_VERSION = "1.0.0"
FORMAT = "flashnext-lmhead-fp8-rtn"
# The head's stored name: this checkpoint keeps it at the top level, the engine's reader finds it
# under the text prefix. The artifact is the same either way, and the manifest records which.
TENSOR_NAMES = ("lm_head.weight", "language_model.lm_head.weight")
CODES = "head.fp8.e4m3.bin"
SCALES = "head.fp8.scales.f32"
ERRORS = "head.fp8.errors.f32"
MANIFEST = "manifest.json"
DEFAULT_CHUNK = 2048  # output rows a pass: 5.2 M values, ~40 MB a float64 temporary


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


def locate(src: Path) -> tuple[str, Path]:
    """The head's stored name in `src`, and the safetensors file that holds it.

    The name is one of `TENSOR_NAMES` — `lm_head.weight` here, the text-prefixed spelling in a
    checkpoint that keeps the head inside the language model — and the file is the index's map or,
    with no index, the single `model.safetensors` beside it.
    """
    index = src / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        for name in TENSOR_NAMES:
            if name in weight_map:
                return name, src / weight_map[name]
        raise SystemExit(f"{src}: the index has no {' or '.join(TENSOR_NAMES)}")
    single = src / "model.safetensors"
    if not single.is_file():
        raise SystemExit(f"{src}: needs model.safetensors or model.safetensors.index.json")
    header, _ = F.read_header(single)
    for name in TENSOR_NAMES:
        if name in header:
            return name, single
    raise SystemExit(f"{single}: no {' or '.join(TENSOR_NAMES)}")


def quantize_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """RTN e4m3 with one scale a row: codes, f32 row scales, and the per-row error stats.

    `scale = amax(row) / 448` (1.0 for an all-zero row); `code = round_even(w / scale)` saturating at
    448, which is exactly `to_e4m3(w / scale)`; `dequant = code * scale`. The stats are per row, in
    the order the caller writes them: max |error|, mean |error|, and the row's relative L2 error.
    """
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {w.shape}")
    w = np.ascontiguousarray(w, dtype=np.float32)
    amax = np.abs(w).max(axis=1)
    if not np.all(np.isfinite(amax)):
        bad = np.flatnonzero(~np.isfinite(amax))[:3].tolist()
        raise ValueError(f"non-finite row amax at rows {bad}: refusing to write a NaN scale")
    scale = (amax / np.float32(F.E4M3_MAX)).astype(np.float32)
    scale = np.where(scale == 0.0, np.float32(F.ZERO_BLOCK_SCALE), scale).astype(np.float32)

    q = w / scale[:, None]
    values = F.to_e4m3(q)
    codes = F.encode_codes(values)
    deq = (values * scale[:, None]).astype(np.float32)

    err = np.abs(deq - w)
    err2 = (deq.astype(np.float64) - w.astype(np.float64)) ** 2
    ref2 = w.astype(np.float64) ** 2
    rel = np.sqrt(err2.sum(axis=1) / np.where(ref2.sum(axis=1) > 0, ref2.sum(axis=1), np.inf))
    stats = {
        "max_abs_err": err.max(axis=1),
        "mean_abs_err": err.mean(axis=1),
        "rel_l2": np.where(np.isfinite(rel), rel, 0.0),
    }
    return np.ascontiguousarray(codes), scale, stats


def summarize(v: np.ndarray) -> dict:
    """A per-row vector's distribution, rounded: what a reader checks the whole table against."""
    v = np.asarray(v, dtype=np.float64)
    return {
        "min": float(v.min()),
        "mean": float(v.mean()),
        "p50": float(np.percentile(v, 50)),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
        "max_row": int(np.argmax(v)),
    }


def self_test() -> None:
    """RTN per row on the shapes that break a rounder: a tie, a saturation, a zero row, a zero scale."""
    w = np.array(
        [
            [1.0, 1.0625, 1.1875, 448.0],
            [1e6, -1e6, 2**-9, 2**-10],
            [0.0, -0.0, 0.0, 0.0],
            [-3.0, 0.5, 2.0, -2.0],
        ],
        dtype=np.float32,
    )
    codes, scale, stats = quantize_rows(w)
    deq = F.decode_codes(codes) * scale[:, None]
    # row 0: scale 1.0, so the codes are the values themselves: a tie goes to the even mantissa,
    # 448 saturates onto the top code
    assert scale[0] == np.float32(1.0), scale[0]
    assert float(deq[0, 0]) == 1.0 and float(deq[0, 3]) == 448.0, deq[0]
    assert float(deq[0, 1]) == 1.0 and float(deq[0, 2]) == 1.25, deq[0]
    # row 1: scale 1e6/448, so 1e6 saturates onto the top code and the subnormals round to zero
    assert float(deq[1, 0]) == np.float32(448.0) * scale[1], (deq[1], scale[1])
    assert float(codes[1, 1]) == 0xFE, hex(int(codes[1, 1]))  # -448, not the NaN code 0xFF
    assert int(codes[1, 2]) == 0 and int(codes[1, 3]) == 0, codes[1]
    # row 2: a row of zeros keeps a finite scale and writes zero codes
    assert np.isfinite(scale[2]) and scale[2] == np.float32(F.ZERO_BLOCK_SCALE), scale[2]
    assert not codes[2].any(), codes[2]
    # every row's error is below the format's half-ulp at the top of its own scale, and no NaN code
    # is written; the top binade's step is 32*s (values in [256, 448]), so half of it is 16*s
    assert np.all((codes & 0x7F) != 0x7F), "a NaN e4m3 code was written"
    for r in range(w.shape[0]):
        bound = float(scale[r]) * 16.0 if np.abs(w[r]).max() > 0 else 0.0
        if not np.all(stats["max_abs_err"][r] <= bound):
            raise SystemExit(f"self-test: row {r} max error {stats['max_abs_err'][r]} > {bound}")
    print("self-test: ok", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--src", type=Path, help="the served checkpoint directory")
    ap.add_argument("--out", type=Path, help="the artifact directory to write")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="output rows a pass (default 2048)")
    ap.add_argument("--force", action="store_true", help="write into a non-empty --out")
    ap.add_argument("--self-test", action="store_true", help="check the rounder and exit")
    args = ap.parse_args(argv)

    self_test()
    if args.self_test:
        return 0
    if args.src is None or args.out is None:
        ap.error("--src and --out are required")
    if args.chunk < 1:
        ap.error("--chunk must be positive")

    src, out = args.src.resolve(), args.out.resolve()
    tensor, shard = locate(src)
    if out == src or out == shard:
        raise SystemExit("--out is the source checkpoint")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty; pass --force to replace its files")

    header, data_start = F.read_header(shard)
    if tensor not in header:
        raise SystemExit(f"{shard}: no {tensor}")
    entry = header[tensor]
    if entry["dtype"] != "BF16":
        raise SystemExit(f"{tensor} is {entry['dtype']}, this arm quantises a bf16 head")
    vocab, hidden = (int(x) for x in entry["shape"])
    if hidden % 64 != 0:
        raise SystemExit(f"{tensor} is {vocab} x {hidden}: a width that is not a multiple of 64")

    out.mkdir(parents=True, exist_ok=True)
    per_row = {k: np.empty(vocab, dtype=np.float32) for k in ("max_abs_err", "mean_abs_err", "rel_l2")}
    codes_sha, scales_sha, errors_sha = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    codes_bytes = scales_bytes = errors_bytes = 0
    zeros = saturated = 0
    max_code = 0

    with open(shard, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as blob:
        view = memoryview(blob)
        try:
            raw = F.tensor_bytes(view, data_start, header, tensor)
            src_sha = hashlib.sha256(raw).hexdigest()
            w_all = F.bf16_to_f32(raw).reshape(vocab, hidden)
            raw.release()  # the mapping cannot close while a view of it is exported
            with open(out / CODES, "wb") as fc, open(out / SCALES, "wb") as fs, open(out / ERRORS, "wb") as fe:
                for row0 in range(0, vocab, args.chunk):
                    rows = min(args.chunk, vocab - row0)
                    codes, scale, stats = quantize_rows(w_all[row0 : row0 + rows])
                    cb, sb = codes.tobytes(), scale.astype("<f4").tobytes()
                    eb = np.stack([stats["max_abs_err"], stats["mean_abs_err"], stats["rel_l2"]], axis=1)
                    eb = eb.astype("<f4").tobytes()
                    for fh, payload in ((fc, cb), (fs, sb), (fe, eb)):
                        fh.write(payload)
                    codes_sha.update(cb)
                    scales_sha.update(sb)
                    errors_sha.update(eb)
                    codes_bytes += len(cb)
                    scales_bytes += len(sb)
                    errors_bytes += len(eb)
                    for k in per_row:
                        per_row[k][row0 : row0 + rows] = stats[k]
                    zeros += int((scale == np.float32(F.ZERO_BLOCK_SCALE)).sum())
                    saturated += int(((codes & 0x7F) == 0x7E).sum())
                    max_code = max(max_code, int(np.max(codes & 0x7F)))
        finally:
            view.release()

    if codes_bytes != vocab * hidden or scales_bytes != vocab * 4:
        raise SystemExit(f"wrote {codes_bytes} code bytes and {scales_bytes} scale bytes, expected {vocab * hidden} and {vocab * 4}")

    manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "shape": [vocab, hidden],
        "tool": {
            "name": "tools/dflash2/quantize/quantize_lm_head.py",
            "version": TOOL_VERSION,
            "layout": "row-major e4m3 codes + one f32 scale an output row; the loader permutes the codes into qmmf's fragment order (zig/src/cuda/fp8.zig:packCodes) and the scales into its [npad/64][k/64][64] fp32 tile",
        },
        "source": {
            "dir": str(src),
            "safetensors": shard.name,
            "tensor": tensor,
            "dtype": entry["dtype"],
            "shape": [vocab, hidden],
            "tensor_bytes": int(np.prod(entry["shape"], dtype=np.int64)) * 2,
            "tensor_sha256": src_sha,
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
            "rows_with_zero_scale": zeros,
            "saturated_codes": saturated,
            "max_code_magnitude": max_code,
        },
        "layout": {
            "codes": {"file": CODES, "dtype": "uint8", "shape": [vocab, hidden], "order": "row-major"},
            "scales": {"file": SCALES, "dtype": "<f4", "shape": [vocab], "order": "one per output row, row order"},
            "per_row_errors": {
                "file": ERRORS,
                "dtype": "<f4",
                "shape": [vocab, 3],
                "columns": ["max_abs_err", "mean_abs_err", "rel_l2"],
                "note": "diagnostic only: the engine does not read this file",
            },
        },
        "errors": {
            "max_abs_err": summarize(per_row["max_abs_err"]),
            "mean_abs_err": summarize(per_row["mean_abs_err"]),
            "rel_l2": summarize(per_row["rel_l2"]),
            "rows": vocab,
        },
        "files": {},
    }
    sums = {
        CODES: (codes_sha.hexdigest(), codes_bytes),
        SCALES: (scales_sha.hexdigest(), scales_bytes),
        ERRORS: (errors_sha.hexdigest(), errors_bytes),
    }
    for name, (digest, size) in sums.items():
        manifest["files"][name] = {"sha256": digest, "bytes": size}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    manifest_sha = sha256_file(out / MANIFEST)
    manifest_bytes = (out / MANIFEST).stat().st_size
    manifest["files"][MANIFEST] = {"sha256": manifest_sha, "bytes": manifest_bytes}
    sums[MANIFEST] = (manifest_sha, manifest_bytes)
    (out / "SHA256SUMS").write_text("".join(f"{d}  {n}\n" for n, (d, _) in sorted(sums.items())))

    e = manifest["errors"]
    print(f"wrote {out}: {codes_bytes / 1e9:.3f} GB of codes, {scales_bytes / 1e6:.2f} MB of row scales")
    print(f"rel-L2 per row: mean {e['rel_l2']['mean'] * 100:.4f}%, p99 {e['rel_l2']['p99'] * 100:.4f}%, "
          f"max {e['rel_l2']['max'] * 100:.4f}% (row {e['rel_l2']['max_row']})")
    print(f"max|err| per row: mean {e['max_abs_err']['mean']:.6f}, max {e['max_abs_err']['max']:.6f}; "
          f"{saturated} saturated codes, {zeros} rows on a zero scale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
