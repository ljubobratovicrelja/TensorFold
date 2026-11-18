#!/usr/bin/env python3
"""The NVFP4 weight format, in one place: e2m1 codes, group-16 e4m3 block scales, one fp32 global scale.

Two tools in this tree quantize the same tensors with the same arithmetic and must not drift apart:
`analysis/head_fidelity.py` scores the target's LM head at this precision against its bf16 ranking, and
`quantize/nvfp4_body.py` fits and evaluates the drafter's linear layers. A number out of one is only
comparable to the other if the rounders, the group layout and the scale direction are literally the same
code, so they live here and are imported, not copied.

The layout is ModelOpt's weight-only NVFP4, as the served kernel reads it:

  dequant(w) = e2m1( w * gs / block_scale ) * block_scale / gs

with `gs = 448 * 6 / amax(tensor)` computed once over the whole tensor (a per-chunk global would flatter
the chunk that happens to be largest), `block_scale = e4m3(block amax / 6)` over each group of 16
weights along the input dim, and the block scale itself e4m3 -- so the dequantised weight carries the
scale's own rounding on top of the 4-bit code. Ignoring that would overstate fidelity.

Everything here is numpy and pure functions: no torch, no GPU, no I/O. `HEAD_FIDELITY_VERSION`-stable
behaviour is a contract: the numbers `head_fidelity.py` prints must not move when this file changes.
"""

from __future__ import annotations

import numpy as np

E4M3_MAX = 448.0  # the largest finite e4m3 value; the block scale saturates here
E2M1_MAX = 6.0  # the largest magnitude e2m1 represents
E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_GROUP = 16  # weights per block scale, along the input dim


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


def global_scale(w: np.ndarray) -> float:
    """The per-tensor multiplier ModelOpt computes once: 448 * 6 / amax(w), floored at 1e-12 amax."""
    amax = float(np.max(np.abs(np.asarray(w, dtype=np.float32))))
    return (E4M3_MAX * E2M1_MAX) / max(amax, 1e-12)


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


def quantize_nvfp4_scaled(w: np.ndarray, group: int, scale_mul: float) -> tuple[np.ndarray, float]:
    """`quantize_nvfp4` for callers that own the per-tensor global scale (`scale_mul` == gs).

    Same arithmetic, just named: a fitter that moves the global scale (or reuses the one recorded in a
    calibration file) must not have to reconstruct it from amax and hope it matches.
    """
    if scale_mul <= 0.0:
        raise ValueError(f"global scale must be positive, got {scale_mul}")
    if np.asarray(w).shape[-1] % group:
        raise ValueError(f"{np.asarray(w).shape}: last dim is not a multiple of group {group}")
    return quantize_nvfp4(w, float(scale_mul), group)


def encode_nvfp4(w: np.ndarray, block_scales: np.ndarray, gs: float, group: int = FP4_GROUP) -> np.ndarray:
    """Dequantise `w` with caller-chosen block scales (already on the e4m3 grid, scaled domain).

    `block_scales` has one entry per run of `group` consecutive weights in row-major order -- the same
    layout `quantize_nvfp4` derives from `w.reshape(-1, group)`: every output row carries its own
    scales, 16 input columns at a time, which is what NVFP4's per-16 block means for a [n, k] weight.

    This is the fitting path: a scale search or a GPTQ pass picks `block_scales` itself, and the code
    it leaves behind still has to be a legal NVFP4 tensor -- e2m1 codes, e4m3 block scales, one global
    scale -- or the number it measures is not the number the kernel would serve.
    """
    w = np.asarray(w, dtype=np.float32)
    blocks = w.reshape(-1, group) * np.float32(gs)
    scales = np.asarray(block_scales, dtype=np.float32).reshape(-1, 1)
    if blocks.shape[0] != scales.shape[0]:
        raise ValueError(f"{w.shape} has {blocks.shape[0]} runs of {group} weights, "
                         f"{scales.shape[0]} scales given")
    scales = np.where(scales <= 0.0, np.float32(1e-12), scales)
    code = e2m1_round(blocks / scales)
    return (code * scales / np.float32(gs)).reshape(w.shape)
