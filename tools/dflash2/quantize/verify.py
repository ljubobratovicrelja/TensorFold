#!/usr/bin/env python3
"""Check an exported DFlash2 FP8 drafter against its bf16 source, independent of the writer.

Dequantizes every quantized matrix in numpy (`code * weight_scale_inv`), and for each prints
its relL2 against the bf16 source, the worst 128x128 block's relL2 and how many blocks are
above 2%; then asserts the round trip is idempotent (re-quantizing the dequantized weights
reproduces the same codes and scales), that every BF16 passthrough tensor is byte-identical,
and that `manifest.json`'s hashes and per-matrix errors are the ones on disk.

    python3 tools/dflash2/quantize/verify.py --fp8 checkpoints/dflash2-fp8 --src SNAP

Exit status is non-zero if any check fails.
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--fp8", type=Path, required=True, help="the fp8 checkpoint directory")
    ap.add_argument("--src", type=Path, required=True, help="the bf16 checkpoint it was made from")
    ap.add_argument("--block", type=int, default=F.BLOCK)
    ap.add_argument(
        "--fail-over-rel-l2",
        type=float,
        default=0.05,
        help="fail a matrix above this relL2 (e4m3's own floor on gaussian weights is ~2.6%%)",
    )
    args = ap.parse_args(argv)

    fp8, src = args.fp8.resolve(), args.src.resolve()
    failures: list[str] = []

    def check(ok: bool, why: str) -> None:
        if not ok:
            failures.append(why)

    manifest = json.loads((fp8 / "manifest.json").read_text())
    q_names = [t["name"] for t in manifest["tensors"]["quantized"]]
    pass_names = list(manifest["tensors"]["passthrough"])
    man_rel = {t["name"]: t for t in manifest["tensors"]["quantized"]}

    # -- the files and the header -----------------------------------------------------------
    sums = dict(
        (line.split("  ", 1)[1].strip(), line.split("  ", 1)[0].strip())
        for line in (fp8 / "SHA256SUMS").read_text().splitlines()
        if line.strip()
    )
    for name in ("config.json", "manifest.json", "model.safetensors"):
        if name not in sums:
            failures.append(f"SHA256SUMS does not list {name}")
            continue
        got = sha256_file(fp8 / name)
        check(got == sums[name], f"SHA256SUMS: {name} is {got[:16]}..., recorded {sums[name][:16]}...")
    hdr_sha = sha256_file(fp8 / "model.safetensors")
    check(hdr_sha == manifest["files"]["model.safetensors"]["sha256"],
          "manifest.json's model.safetensors sha256 does not match the file")
    cfg = json.loads((fp8 / "config.json").read_text())
    check(cfg.get("dflash_config", {}).get("precision") == "fp8_w8a16",
          "config.json lacks dflash_config.precision == fp8_w8a16")
    src_cfg = json.loads((src / "config.json").read_text())
    for k, v in src_cfg.items():
        if k != "dflash_config":
            check(cfg.get(k) == v, f"config.json changed {k!r} of the source's")
    for k, v in (src_cfg.get("dflash_config") or {}).items():
        check(cfg["dflash_config"].get(k) == v, f"config.json dropped dflash_config.{k}")

    header, data_start = F.read_header(fp8 / "model.safetensors")
    # the container itself: what any safetensors parser assumes of a header and a buffer
    size = (fp8 / "model.safetensors").stat().st_size
    offs = sorted((tuple(e["data_offsets"]), n) for n, e in header.items() if n != "__metadata__")
    cursor = 0
    for (begin, end), n in offs:
        pad = begin - cursor
        check(0 <= pad <= 7 and begin % 8 == 0, f"{n}: data starts at {begin}, {pad} bytes after the last tensor")
        check(end > begin, f"{n}: empty data range {begin}..{end}")
        cursor = end
    check(cursor <= size - data_start and size - data_start - cursor <= 7,
          f"model.safetensors is {size} bytes: {size - data_start - cursor} past the last tensor")
    for n, e in header.items():
        if n == "__metadata__":
            continue
        check(e["dtype"] in F.DTYPE_SIZE, f"{n}: dtype {e['dtype']} is not a safetensors dtype")
        check(isinstance(e["shape"], list) and len(e["shape"]) >= 1, f"{n}: shape {e.get('shape')}")

    names = set(F.tensor_names(header))
    check(names == set(q_names) | set(pass_names) | {n + "_scale_inv" for n in q_names},
          "the fp8 checkpoint's tensor names are not the manifest's")
    for name in q_names:
        e = header.get(name, {})
        check(e.get("dtype") == "F8_E4M3", f"{name}: dtype {e.get('dtype')}, expected F8_E4M3")
        sname = name + "_scale_inv"
        if sname not in header:
            continue
        n, k = e["shape"]
        want = (-(-n // args.block), -(-k // args.block))
        s = header[sname]
        check(s["dtype"] == "F32", f"{sname}: dtype {s['dtype']}, expected F32")
        check(tuple(s["shape"]) == want, f"{sname}: shape {s['shape']}, expected {list(want)}")

    src_header, src_start = F.read_header(src / "model.safetensors")
    rows, over_total, blocks_total = [], 0, 0
    with open(fp8 / "model.safetensors", "rb") as f8, mmap.mmap(f8.fileno(), 0, access=mmap.ACCESS_READ) as blob8, \
            open(src / "model.safetensors", "rb") as fs, mmap.mmap(fs.fileno(), 0, access=mmap.ACCESS_READ) as blobs:
        v8, vs = memoryview(blob8), memoryview(blobs)
        try:
            for name in q_names:
                n, k = header[name]["shape"]
                # copied out of the mapping: a view would keep the mmap exported at close
                codes = np.array(F.tensor_bytes(v8, data_start, header, name), dtype=np.uint8).reshape(n, k)
                sname = name + "_scale_inv"
                scale = np.frombuffer(bytes(F.tensor_bytes(v8, data_start, header, sname)), dtype="<f4")
                scale = scale.reshape(header[sname]["shape"])
                check(bool(np.all(np.isfinite(scale))) and bool(np.all(scale > 0)),
                      f"{sname}: non-finite or non-positive scale")
                check(int(np.max(codes & 0x7F)) < 0x7F, f"{name}: an e4m3 NaN code was written")

                w = F.bf16_to_f32(F.tensor_bytes(vs, src_start, src_header, name)).reshape(n, k)
                deq = F.dequantize(codes, scale, args.block)

                err = deq.astype(np.float64) - w.astype(np.float64)
                rel = float(np.sqrt((err**2).sum() / max((w.astype(np.float64) ** 2).sum(), 1e-300)))
                nb, kb, b = -(-n // args.block), -(-k // args.block), args.block
                e2 = (err**2).reshape(nb, b, kb, b).sum(axis=(1, 3))
                r2 = (w.astype(np.float64) ** 2).reshape(nb, b, kb, b).sum(axis=(1, 3))
                blk = np.sqrt(e2 / np.where(r2 > 0, r2, np.inf))
                over = int((blk > 0.02).sum())
                over_total += over
                blocks_total += int(blk.size)

                # idempotency: the same scale, and the same codes, out of the dequantized block
                again = F.round_trip_codes(codes, scale, args.block)
                same = again.tobytes() == codes.tobytes()
                check(same, f"{name}: re-quantizing the dequantized weight did not reproduce its codes")
                wq = F.quantize_blockwise(deq, args.block)
                check(np.array_equal(wq[0], codes), f"{name}: quantize_blockwise(dequantized) differs in codes")
                check(np.array_equal(wq[1], scale), f"{name}: quantize_blockwise(dequantized) differs in scales")

                m = man_rel.get(name, {})
                check(abs(m.get("rel_l2", -1) - rel) <= 1e-9, f"{name}: manifest rel_l2 {m.get('rel_l2')} != {rel}")
                check(abs(m.get("worst_block_rel_l2", -1) - float(blk.max())) <= 1e-9,
                      f"{name}: manifest worst_block_rel_l2 differs")
                check(m.get("blocks_over_2pct") == over, f"{name}: manifest blocks_over_2pct differs")
                check(rel <= args.fail_over_rel_l2, f"{name}: relL2 {rel * 100:.3f}% over {args.fail_over_rel_l2 * 100:.1f}%")
                rows.append((name, (n, k), rel, float(blk.max()), over, int(blk.size)))

            for name in pass_names:
                a = bytes(F.tensor_bytes(v8, data_start, header, name))
                b = bytes(F.tensor_bytes(vs, src_start, src_header, name))
                check(a == b, f"{name}: passthrough bytes differ from the source's bf16")
                check(header[name]["dtype"] == "BF16", f"{name}: passthrough dtype {header[name]['dtype']}")
        finally:
            v8.release()
            vs.release()

    check(manifest["blocks"]["over_2pct"] == over_total,
          f"manifest blocks.over_2pct {manifest['blocks']['over_2pct']} != {over_total}")
    check(len(rows) == len(q_names), "not every quantized matrix was checked")

    print(f"{'matrix':58s} {'shape':>13s} {'relL2':>8s} {'worst blk':>10s} {'>2%':>5s}")
    for name, shape, rel, worst, over, nblk in sorted(rows, key=lambda r: -r[2]):
        flag = "  <-- above 3%" if rel > 0.03 else ""
        print(f"{name:58s} {shape[0]:6d}x{shape[1]:<6d} {rel * 100:7.3f}% {worst * 100:9.3f}% {over:5d}{flag}")
    worst_m = max(rows, key=lambda r: r[2]) if rows else None
    print()
    print(f"{len(rows)} matrices, {blocks_total} blocks, {over_total} blocks above 2% "
          f"({over_total / max(blocks_total, 1) * 100:.2f}%)")
    if worst_m:
        print(f"worst matrix: {worst_m[0]} at relL2 {worst_m[2] * 100:.3f}%, "
              f"worst block {worst_m[3] * 100:.3f}%")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for why in failures[:40]:
            print(f"  {why}")
        return 1
    print("\nverified: codes, scales, passthrough bytes, round-trip idempotence and manifest hashes all hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
