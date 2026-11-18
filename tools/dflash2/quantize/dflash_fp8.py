#!/usr/bin/env python3
"""Shared pieces of the DFlash2 drafter's weight-only FP8 export: safetensors I/O, bf16, e4m3.

The FP8 layout is the repo's block-FP8 one (`zig/src/cuda/fp8.zig`, `zig/src/core/quant/fp8.zig`):
for a quantized `X.weight` of shape [n, k] the checkpoint carries

  X.weight             F8_E4M3  [n, k]                     row-major e4m3 codes
  X.weight_scale_inv   F32      [ceil(n/128), ceil(k/128)]  one scale a 128 x 128 block

with `weight_scale_inv = amax / 448` for the block's largest magnitude, `code =
round_to_nearest_even(w / weight_scale_inv)` saturating at 448, and `dequant = code *
weight_scale_inv`. That is the direction the repo's readers already take it in --
`fp8.reference` (`e4m3(code) * s`), the qmmf FP8G kernel's `acc * s`, and
`tools/step5_fp8_to_mlx.py:fp8_dequant` (`w.to(float32) * s`) -- so the stored f32 is the
multiplier a block's codes are read with, and it is small (amax / 448), not its reciprocal.

Pure numpy and the standard library: no torch, no safetensors, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np

TOOL_VERSION = "1.0.0"
BLOCK = 128
E4M3_MAX = 448.0
ZERO_BLOCK_SCALE = 1.0  # a block of zeros still gets a finite scale: dequant = 0 * 1.0

DTYPE_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1}


# -- safetensors ---------------------------------------------------------------------------


def read_header(path: Path) -> tuple[dict, int]:
    """The JSON header of a safetensors file and the byte offset its data buffer starts at."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: too short for a safetensors length prefix")
        (n,) = struct.unpack("<Q", raw)
        if n == 0 or n > 100 * 1024 * 1024:
            raise ValueError(f"{path}: implausible header length {n}")
        header = json.loads(f.read(n))
    if not isinstance(header, dict):
        raise ValueError(f"{path}: header is not an object")
    return header, 8 + n


def tensor_names(header: dict) -> list[str]:
    return [k for k in header if k != "__metadata__"]


def tensor_bytes(blob: memoryview, data_start: int, header: dict, name: str) -> memoryview:
    """The raw bytes of `name`, checked against the header's dtype and element count."""
    entry = header[name]
    begin, end = entry["data_offsets"]
    dtype, shape = entry["dtype"], entry["shape"]
    if dtype not in DTYPE_SIZE:
        raise ValueError(f"{name}: unsupported dtype {dtype}")
    want = int(np.prod(shape, dtype=np.int64)) * DTYPE_SIZE[dtype] if shape else DTYPE_SIZE[dtype]
    if end - begin != want:
        raise ValueError(f"{name}: {end - begin} bytes, expected {want} for {dtype} {shape}")
    return blob[data_start + begin : data_start + end]


def bf16_to_f32(raw: memoryview | np.ndarray) -> np.ndarray:
    """bf16 bytes (or a uint16 view of them) as float32, bit for bit: bf16 is f32's top half."""
    u = np.frombuffer(raw, dtype="<u2") if not isinstance(raw, np.ndarray) else raw.view("<u2")
    return (u.astype(np.uint32) << 16).view("<f4")


def write_safetensors(path: Path, plan: list[tuple], producer) -> dict:
    """Write a safetensors file from a plan of (name, dtype, shape, nbytes), streaming each body.

    `producer(entry)` returns the tensor's bytes. Returns the sha256 and size of the file written.
    Keys are emitted sorted, as the HF tooling writes them, and every body starts 8-byte aligned.
    """
    entries, offset = {}, 0
    for name, dtype, shape, nbytes in plan:
        entries[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes + (-nbytes % 8)  # pad to 8: alignment the GPU loaders expect
    header = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    # safetensors pads the header to a multiple of 8 with spaces so the buffer starts aligned
    header += b" " * ((-len(header)) % 8)
    digest = hashlib.sha256()
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        prefix = struct.pack("<Q", len(header))
        f.write(prefix)
        digest.update(prefix)
        f.write(header)
        digest.update(header)
        for name, dtype, shape, nbytes in plan:
            body = producer((name, dtype, shape, nbytes))
            body = bytes(body)
            if len(body) != nbytes:
                raise ValueError(f"{name}: produced {len(body)} bytes, planned {nbytes}")
            f.write(body)
            digest.update(body)
            pad = -nbytes % 8
            if pad:
                f.write(b"\0" * pad)
                digest.update(b"\0" * pad)
    os.replace(tmp, path)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


# -- e4m3 ----------------------------------------------------------------------------------


def e4m3_values() -> np.ndarray:
    """Every finite e4m3 code 0x00..0x7e's magnitude, ascending. 0x7f is NaN and never emitted."""
    b = np.arange(0x7F, dtype=np.int64)
    e = (b >> 3) & 0xF
    m = (b & 7).astype(np.float64)
    mag = np.where(e == 0, m / 8.0 * 2.0**-6, (1.0 + m / 8.0) * np.exp2(e - 7))
    return mag  # ascending: the subnormals then the binades


E4M3_MAGS = e4m3_values()


def decode_codes(codes: np.ndarray) -> np.ndarray:
    """e4m3 codes [..., ] as their values, as float32; 0x7f/0xff (NaN) refused."""
    c = codes.astype(np.int64)
    if np.any((c & 0x7F) == 0x7F):
        raise ValueError("NaN e4m3 code in the checkpoint")
    mag = E4M3_MAGS[c & 0x7F]
    return (np.where(c & 0x80, -mag, mag)).astype(np.float32)


def to_e4m3(x: np.ndarray) -> np.ndarray:
    """`x` (float32) rounded to the nearest e4m3 value, ties to even, saturating at 448.

    `core.quant.fp8.toE4m3`'s arithmetic: the step comes from the binade, floored at the
    subnormals' 2^-9, and a ratio exactly halfway between two codes steps down to the even one.
    """
    xf = np.asarray(x, dtype=np.float64)
    a = np.minimum(np.abs(xf), E4M3_MAX)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    step = np.exp2(np.maximum(e, -6.0) - 3.0)
    r = a / step
    q = np.floor(r + 0.5)  # half away from zero, then the even fix below
    frac = r - np.trunc(r)
    tie = np.abs(frac - 0.5) < 1e-12
    q = np.where(tie & (np.mod(q, 2.0) == 1.0), q - 1.0, q)
    return (np.sign(xf) * np.minimum(q * step, E4M3_MAX)).astype(np.float32)


def encode_codes(v: np.ndarray) -> np.ndarray:
    """e4m3 codes [..., ] for values that already lie on the e4m3 grid; sign-magnitude, no NaN."""
    a = np.asarray(v, dtype=np.float64).ravel()
    mag = np.abs(a)
    idx = np.searchsorted(E4M3_MAGS, mag)
    idx = np.minimum(idx, E4M3_MAGS.size - 1)
    if not np.all(E4M3_MAGS[idx] == mag):
        bad = np.flatnonzero(E4M3_MAGS[idx] != mag)[:3]
        raise ValueError(f"values off the e4m3 grid, e.g. {mag[bad].tolist()}")
    codes = idx.astype(np.uint8)
    codes[np.flatnonzero((a < 0) & (mag > 0))] |= 0x80
    return codes.reshape(np.shape(v))


# -- block quantization --------------------------------------------------------------------


def quantize_blockwise(w: np.ndarray, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray, dict]:
    """Quantize a float32 [n, k] weight: e4m3 codes, f32 block scales, and the error stats.

    `weight_scale_inv = amax / 448` over each block (1.0 for a block of zeros); `code =
    round_even(w / weight_scale_inv)`, saturating at 448. `dequant = code * weight_scale_inv`.
    """
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {w.shape}")
    w = np.ascontiguousarray(w, dtype=np.float32)
    n, k = w.shape
    nb, kb = -(-n // block), -(-k // block)
    padded = np.zeros((nb * block, kb * block), dtype=np.float32)
    padded[:n, :k] = w
    blocks = padded.reshape(nb, block, kb, block).transpose(0, 2, 1, 3)  # [nb, kb, block, block]

    amax = np.abs(blocks).max(axis=(2, 3))
    if not np.all(np.isfinite(amax)):
        bad = np.argwhere(~np.isfinite(amax))[:3].tolist()
        raise ValueError(f"non-finite block amax at {bad}: refusing to write a NaN scale")
    scale = (amax / np.float32(E4M3_MAX)).astype(np.float32)
    zeros = scale == 0.0
    scale = np.where(zeros, np.float32(ZERO_BLOCK_SCALE), scale).astype(np.float32)

    q = blocks / scale[:, :, None, None]  # float32, as the reader's amax/448 scale is applied
    values = to_e4m3(q)
    codes = encode_codes(values.reshape(nb, kb, block, block))

    deq = (values * scale[:, :, None, None]).astype(np.float32)
    deq = deq.transpose(0, 2, 1, 3).reshape(nb * block, kb * block)[:n, :k]
    err2 = ((deq - w).astype(np.float64) ** 2).reshape(nb, block, kb, block).sum(axis=(1, 3))
    ref2 = (w.astype(np.float64) ** 2).reshape(nb, block, kb, block).sum(axis=(1, 3))
    block_rel = np.sqrt(err2 / np.where(ref2 > 0, ref2, np.inf))
    total = float(np.sqrt(err2.sum() / ref2.sum())) if ref2.sum() > 0 else 0.0

    # [nb, kb, block, block] -> [n, k] in row-major order
    codes = codes.transpose(0, 2, 1, 3).reshape(nb * block, kb * block)[:n, :k]
    stats = {
        "rel_l2": total,
        "worst_block_rel_l2": float(block_rel.max()) if block_rel.size else 0.0,
        "blocks": int(block_rel.size),
        "blocks_over_2pct": int((block_rel > 0.02).sum()),
        "zero_blocks": int(zeros.sum()),
        "max_code_magnitude": int(np.max(codes & 0x7F)) if codes.size else 0,
        "saturated_codes": int((((codes & 0x7F) == 0x7E)).sum()),
    }
    return np.ascontiguousarray(codes), np.ascontiguousarray(scale), stats


def dequantize(codes: np.ndarray, scale: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """e4m3 codes [n, k] * their block scales [ceil(n/128), ceil(k/128)] as float32 [n, k]."""
    n, k = codes.shape
    nb, kb = scale.shape
    if (nb, kb) != (-(-n // block), -(-k // block)):
        raise ValueError(f"scales {scale.shape} do not match codes {codes.shape} at block {block}")
    values = decode_codes(codes)
    rep = np.repeat(np.repeat(scale, block, axis=0), block, axis=1)[:n, :k]
    return (values * rep).astype(np.float32)


def round_trip_codes(codes: np.ndarray, scale: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Re-quantize a dequantized weight: the codes a second pass writes for the same block scales."""
    w = dequantize(codes, scale, block)
    n, k = w.shape
    nb, kb = -(-n // block), -(-k // block)
    padded = np.zeros((nb * block, kb * block), dtype=np.float32)
    padded[:n, :k] = w
    blocks = padded.reshape(nb, block, kb, block).transpose(0, 2, 1, 3)
    values = to_e4m3(blocks / scale[:, :, None, None])
    again = encode_codes(values.reshape(nb, kb, block, block))
    return np.ascontiguousarray(again.transpose(0, 2, 1, 3).reshape(nb * block, kb * block)[:n, :k])
