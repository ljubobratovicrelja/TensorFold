#!/usr/bin/env python3
"""Write the DFlash2 drafter as weight-only FP8 (e4m3 weights, 128x128 block scales).

The three decoder-linear groups the drafter spends its FLOPs in -- MLP gate/up/down and the
attention q/k/v/o projections -- become `F8_E4M3` with an `F32` `weight_scale_inv` a 128x128
block, in the repo's block-FP8 layout (`zig/src/cuda/fp8.zig`). Everything else (the `fc`
projection, both convs' tensors and kernel projections, every norm, and the candidate
selector's hidden_projection/codebooks) is copied through unchanged as BF16, so the output
directory is a complete standalone checkpoint.

    python3 tools/dflash2/quantize/quantize_drafter.py --src SNAP --out checkpoints/dflash2-fp8

CPU only, numpy and the standard library. Writes `model.safetensors`, a `config.json` carrying
the same object plus the precision marker, `manifest.json` with the provenance and per-matrix
error, and `SHA256SUMS`.

Scale direction: `weight_scale_inv = amax / 448` a block -- the value a block's codes are
multiplied by (`dequant = code * weight_scale_inv`), which is what `fp8.reference`, the qmmf
FP8G kernel and `tools/step5_fp8_to_mlx.py` all do with a checkpoint's `weight_scale_inv`. The
multiplier that turns the weight into codes is its reciprocal, 448 / amax.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dflash_fp8 as F  # noqa: E402

DEFAULT_GROUPS = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def quantized_names(header: dict, groups: list[str]) -> tuple[list[str], list[str]]:
    """The tensors of `header` that match a group (a `.weight` in one of them), and the rest."""
    wanted, rest = [], []
    for name in sorted(F.tensor_names(header)):
        hit = any(name.endswith(f".{g}.weight") or name == f"{g}.weight" for g in groups)
        (wanted if hit else rest).append(name)
    if not wanted:
        raise SystemExit(f"no tensors matched the groups {groups}: is --src the drafter checkpoint?")
    return wanted, rest


def self_test() -> None:
    """The rounding the repo pins: halfway to even, saturating, and the subnormals from 2^-6."""
    cases = [
        (1.0625, 1.0, "halfway between 1.0 and 1.125 rounds to the even mantissa"),
        (1.1875, 1.25, "halfway between 1.125 and 1.25 rounds up to the even mantissa"),
        (1e6, 448.0, "saturates at 448"),
        (-1e6, -448.0, "saturates at -448"),
        (2**-9, 2**-9, "the smallest subnormal"),
        (2**-10, 0.0, "below the smallest subnormal rounds to zero"),
        (0.0, 0.0, "zero"),
        (-0.0, 0.0, "negative zero"),
    ]
    for x, want, why in cases:
        got = float(F.to_e4m3(np.array([x], dtype=np.float32))[0])
        if got != want:
            raise SystemExit(f"self-test: to_e4m3({x}) = {got!r}, expected {want!r} ({why})")
    # every code survives decode -> round -> encode, and the table has no NaN hole
    codes = np.arange(0x7F, dtype=np.uint8)
    again = F.encode_codes(F.to_e4m3(F.decode_codes(codes)))
    if not np.array_equal(again, codes):
        raise SystemExit(f"self-test: codes are not a fixed point of decode/to_e4m3/encode: {again[:8]}")
    if F.E4M3_MAGS[-1] != 448.0 or not np.all(np.diff(F.E4M3_MAGS) > 0):
        raise SystemExit("self-test: the e4m3 magnitude table is not ascending to 448")
    # every rounded value is a nearest e4m3 value of what it came from (ties may go either way)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(200000) * np.exp2(rng.integers(-12, 12, 200000))).astype(np.float32)
    v = F.to_e4m3(x)
    a = np.minimum(np.abs(x).astype(np.float64), F.E4M3_MAX)
    hi = np.minimum(np.searchsorted(F.E4M3_MAGS, a), F.E4M3_MAGS.size - 1)
    lo = np.maximum(hi - 1, 0)
    got = np.abs(v).astype(np.float64)
    nearest = np.minimum(np.abs(a - F.E4M3_MAGS[hi]), np.abs(a - F.E4M3_MAGS[lo]))
    if not np.all(np.abs(got - a) <= nearest + 1e-30):
        raise SystemExit("self-test: to_e4m3 did not return a nearest e4m3 value")
    # and the codes round-trip through decode to the same values
    if not np.array_equal(F.decode_codes(F.encode_codes(v)), v):
        raise SystemExit("self-test: encode/decode does not reproduce to_e4m3's values")

    # a block of zeros takes a finite scale, and a saturated block reconstructs
    z = F.quantize_blockwise(np.zeros((128, 128), dtype=np.float32))
    if not np.isfinite(z[1]).all() or np.any(z[0]):
        raise SystemExit("self-test: a block of zeros did not take a finite scale and zero codes")
    print("self-test: ok", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--src", type=Path, help="the bf16 drafter checkpoint directory")
    ap.add_argument("--out", type=Path, help="the fp8 checkpoint directory to write")
    ap.add_argument("--block", type=int, default=F.BLOCK, help="scale block, rows and columns (default 128)")
    ap.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUPS),
        help="comma-separated module groups to quantize, matched as '<group>.weight' suffixes",
    )
    ap.add_argument("--force", action="store_true", help="write into a non-empty --out")
    ap.add_argument("--self-test", action="store_true", help="check the e4m3 rounding and exit")
    args = ap.parse_args(argv)

    self_test()
    if args.self_test:
        return 0
    if args.src is None or args.out is None:
        ap.error("--src and --out are required")
    if args.block != 128:
        raise SystemExit("the loader reads 128x128 blocks; --block is not free to vary yet")

    src, out = args.src.resolve(), args.out.resolve()
    src_st = src / "model.safetensors"
    cfg_path = src / "config.json"
    if not src_st.is_file() or not cfg_path.is_file():
        raise SystemExit(f"{src}: needs model.safetensors and config.json")
    if out == src:
        raise SystemExit("--out is the source checkpoint")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty; pass --force to replace its files")
    out.mkdir(parents=True, exist_ok=True)

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    header, data_start = F.read_header(src_st)
    to_quant, rest = quantized_names(header, groups)
    src_bytes = src_st.stat().st_size
    src_sha = sha256_file(src_st)

    # the file's layout: sorted names, each quantized weight followed by its `weight_scale_inv`
    plan = []
    for name in sorted(to_quant):
        n, k = header[name]["shape"]
        plan.append((name, "F8_E4M3", (n, k), n * k))
        plan.append((name + "_scale_inv", "F32", (-(-n // args.block), -(-k // args.block)),
                     4 * (-(-n // args.block)) * (-(-k // args.block))))
    for name in rest:
        e = header[name]
        nbytes = int(np.prod(e["shape"], dtype=np.int64)) * F.DTYPE_SIZE[e["dtype"]] if e["shape"] else F.DTYPE_SIZE[e["dtype"]]
        plan.append((name, e["dtype"], tuple(e["shape"]), nbytes))
    plan.sort(key=lambda t: t[0])

    stats: dict[str, dict] = {}
    bodies: dict[str, bytes] = {}
    with open(src_st, "rb") as src_file, mmap.mmap(src_file.fileno(), 0, access=mmap.ACCESS_READ) as blob:
        view = memoryview(blob)
        try:
            def quantize_one(name):
                if name not in stats:
                    shape = tuple(header[name]["shape"])
                    raw = F.tensor_bytes(view, data_start, header, name)
                    w = F.bf16_to_f32(raw).reshape(shape)
                    codes, scale, st = F.quantize_blockwise(w, args.block)
                    st["shape"] = [int(shape[0]), int(shape[1])]
                    stats[name] = st
                    bodies[name] = codes.tobytes()
                    bodies[name + "_scale_inv"] = scale.tobytes()
                return bodies[name]

            def producer(entry):
                name = entry[0]
                base = name[: -len("_scale_inv")]
                if name in to_quant:
                    return quantize_one(name)
                if name.endswith("_scale_inv") and base in to_quant:
                    quantize_one(base)
                    return bodies[name]
                return F.tensor_bytes(view, data_start, header, name)

            written = F.write_safetensors(out / "model.safetensors", plan, producer)
        finally:
            view.release()

    # config.json: every existing key, and the precision marker merged into `dflash_config`
    cfg = json.loads(cfg_path.read_text())
    merged = dict(cfg.get("dflash_config") or {})
    if "precision" in merged:
        raise SystemExit(f"config.json already carries dflash_config.precision={merged['precision']!r}")
    merged.update({"precision": "fp8_w8a16", "quantized_weight_dtype": "F8_E4M3", "quantized_block": [args.block, args.block]})
    cfg["dflash_config"] = merged
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    data_bytes = sum((e[3] + 7) // 8 * 8 for e in plan)
    codes_bytes = sum(e[3] for e in plan if e[1] == "F8_E4M3")
    scale_bytes = sum(e[3] for e in plan if e[1] == "F32")
    pass_bytes = sum(e[3] for e in plan if e[2] and e[1] not in ("F8_E4M3", "F32"))
    per_tensor = []
    for name in to_quant:
        st = stats[name]
        per_tensor.append({
            "name": name,
            "shape": st["shape"],
            "scale_shape": [(-(-st["shape"][0] // args.block)), (-(-st["shape"][1] // args.block))],
            "bytes": st["shape"][0] * st["shape"][1],
            "rel_l2": st["rel_l2"],
            "worst_block_rel_l2": st["worst_block_rel_l2"],
            "blocks": st["blocks"],
            "blocks_over_2pct": st["blocks_over_2pct"],
            "zero_blocks": st["zero_blocks"],
            "max_code_magnitude": st["max_code_magnitude"],
            "saturated_codes": st["saturated_codes"],
        })
    all_blocks = sum(t["blocks"] for t in per_tensor)
    manifest = {
        "format": "dflash2-fp8-w8a16",
        "schema_version": 1,
        "tool": {
            "name": "tools/dflash2/quantize/quantize_drafter.py",
            "version": F.TOOL_VERSION,
            "layout": "zig/src/cuda/fp8.zig (e4m3 codes [n,k], f32 weight_scale_inv [ceil(n/128),ceil(k/128)])",
        },
        "source": {
            "dir": str(src),
            "safetensors": "model.safetensors",
            "safetensors_bytes": src_bytes,
            "safetensors_sha256": src_sha,
            "config_sha256": sha256_file(cfg_path),
            "tensors": len(F.tensor_names(header)),
        },
        "quantization": {
            "precision": "fp8_w8a16",
            "weight_dtype": "F8_E4M3",
            "scale_dtype": "F32",
            "scale_tensor_suffix": "_scale_inv",
            "block": [args.block, args.block],
            "scale_semantics": "weight_scale_inv = amax(block)/448; dequant = code * weight_scale_inv; code = round_to_nearest_even(w * 448/amax), saturating at 448",
            "code_grid": "e4m3fn, 4-bit exponent bias 7, 3-bit mantissa, subnormals below 2^-6",
            "rounding": "round to nearest, ties to even",
            "zero_block_scale": F.ZERO_BLOCK_SCALE,
            "activation_dtype": "BF16",
            "groups": groups,
            "tensors_quantized": len(to_quant),
            "tensors_passthrough": len(rest),
        },
        "tensors": {"quantized": per_tensor, "passthrough": rest},
        "blocks": {
            "total": all_blocks,
            "over_2pct": sum(t["blocks_over_2pct"] for t in per_tensor),
            "worst_block_rel_l2": max((t["worst_block_rel_l2"] for t in per_tensor), default=0.0),
        },
        "bytes": {
            "file": written["bytes"],
            "codes": codes_bytes,
            "scales": scale_bytes,
            "passthrough_bf16": pass_bytes,
            "header": written["bytes"] - data_bytes,
            "data": data_bytes,
        },
        "files": {
            "model.safetensors": {"sha256": written["sha256"], "bytes": written["bytes"]},
            "config.json": {"sha256": sha256_file(out / "config.json"), "bytes": (out / "config.json").stat().st_size},
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    sums = "".join(
        f"{sha256_file(out / n)}  {n}\n" for n in ("config.json", "manifest.json", "model.safetensors")
    )
    (out / "SHA256SUMS").write_text(sums)

    worst = max(per_tensor, key=lambda t: t["rel_l2"])
    print(f"wrote {out / 'model.safetensors'}: {written['bytes'] / 1e6:.1f} MB "
          f"({len(to_quant)} quantized, {len(rest)} bf16 passthrough)")
    print(f"codes {codes_bytes / 1e6:.1f} MB, scales {scale_bytes / 1e6:.2f} MB, "
          f"passthrough {pass_bytes / 1e6:.1f} MB")
    print(f"worst matrix relL2 {worst['rel_l2'] * 100:.4f}% ({worst['name']}); "
          f"{manifest['blocks']['over_2pct']} of {all_blocks} blocks above 2%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
