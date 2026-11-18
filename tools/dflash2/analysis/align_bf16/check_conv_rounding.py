#!/usr/bin/env python
"""Test one hypothesis about the per-block divergence, offline, from dumped stages.

Hypothesis: `tf_q35d_dyn_conv` rounds its accumulator to bf16 after every term
(`total = rbf(total + ...)`, three times), while the reference's
`_grouped_dynamic_convolve` accumulates in bf16 tensors with no per-step rounding. If that is the
whole story, then recomputing our conv **in numpy with bf16 rounding per step** reproduces our
`conv_attn_out_k` closely, and recomputing it **with fp32 accumulation and one rounding at the store**
(which is what the reference does, its `output + kernel * values` staying in bf16 only for the
multiply) moves us toward `conv_*_out_k`'s reference value.

The check needs the operands. `conv_attn_out_k = residual_prev + conv(x)`, where `x = attn_proj_k`
(both dumped), and `conv_mlp_out_k = conv_attn_out_k + conv(mlp_k)`. Each conv call is
`out[t] = sum_l (BASE[l,c] + DYN[t,l,c//group]) * x[t-l, c]` with `x[-1] = 0`, `branch` selecting the
first or second kernel of `base_kernel` (`[2, kernel_size, hidden]`).

Operands: BASE is a pure checkpoint tensor; DYN is `kernel_projection(hidden).view(..., 2, ks, groups)`
and is not dumped, so this script **derives** DYN from the checkpoint's `kernel_projection` weight and
the dumped input hidden — which also makes it a check on our own projection, independently of the
conv rounding question.

Usage:
  python check_conv_rounding.py <ref-dir> <ours-dir> <drafter-checkpoint-dir> [--block 0]
"""

import argparse
import json
import os
import sys

import numpy as np

R = lambda x: np.float32(x)                      # noqa: E731

def to_bf16(a):
    """Round an fp32 array to bf16 (round-half-to-even), staying in float32."""
    u = np.asarray(a, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    rounded = u + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def load(side, name):
    p = os.path.join(side, "tensors", f"{name}.npy")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    return np.load(p)


def read_safetensors_tensor(path, name):
    """A single bf16 tensor out of a safetensors file, returned as float32."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
        base = 8 + n
        meta = head[name]
        s, e = meta["data_offsets"]
        f.seek(base + s)
        raw = f.read(e - s)
    arr = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return arr.view(np.float32).reshape(meta["shape"])


def conv(x, base, dyn, ks, group):
    """`out[t] = sum_l (base[l,c] + dyn[t,l,c//group]) * x[t-l, c]`, accumulated in float32, one
    rounding at the store — the reference's arithmetic, made explicit."""
    rows, D = x.shape
    g = D // group
    out = np.zeros_like(x)
    for t in range(rows):
        acc = np.zeros(D, dtype=np.float32)
        for l in range(ks):
            if t - l < 0:
                continue
            k = base[l].astype(np.float32) + np.repeat(dyn[t, l].astype(np.float32), group)
            acc = acc + k * x[t - l]
        out[t] = to_bf16(acc)
    return out


def conv_rbf(x, base, dyn, ks, group):
    """The engine's arithmetic: round to bf16 after every addition."""
    rows, D = x.shape
    out = np.zeros_like(x)
    for t in range(rows):
        total = np.zeros(D, dtype=np.float32)
        for l in range(ks):
            if t - l < 0:
                continue
            b = base[l].astype(np.float32)
            d = np.repeat(dyn[t, l].astype(np.float32), group)
            v = x[t - l]
            total = to_bf16(to_bf16(total) + to_bf16(to_bf16(b) * to_bf16(v)))
            total = to_bf16(to_bf16(total) + to_bf16(to_bf16(d) * to_bf16(v)))
        out[t] = to_bf16(np.asarray(total))
    return out


def rel(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = a - b
    return float(np.linalg.norm(d) / np.linalg.norm(a)), float(np.abs(d).max())


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("ours_dir")
    ap.add_argument("drafter", help="the DFlash2 checkpoint dir (safetensors)")
    ap.add_argument("--block", type=int, default=0)
    args = ap.parse_args(argv[1:])

    k = args.block
    st_path = os.path.join(args.drafter, "model.safetensors")
    if not os.path.exists(st_path):
        cands = [f for f in os.listdir(args.drafter) if f.endswith(".safetensors")]
        if not os.path.exists(cands[0]):
            raise SystemExit(f"no safetensors in {args.drafter}")
        st_path = os.path.join(args.drafter, cands[0])

    # the checkpoint's conv parameters for this layer
    with open(st_path, "rb") as f:
        n_bytes = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n_bytes))
    prefix = f"layers.{k}.attention_conv"
    if f"{prefix}.base_kernel" not in header:
        raise SystemExit(f"no {prefix}.base_kernel in {st_path}")
    print(f"[conv] layer {k} conv tensors: "
          f"{[n for n in header if n.startswith(prefix)]}", flush=True)
    base = read_safetensors_tensor(st_path, f"{prefix}.base_kernel")      # [2, ks, D]
    proj = read_safetensors_tensor(st_path, f"{prefix}.kernel_projection.weight")  # [2*ks*g, D]

    ours_attn = load(args.ours_dir, f"attn_proj_{k}")
    ours_mlp = load(args.ours_dir, f"mlp_{k}")
    ours_out = load(args.ours_dir, f"conv_attn_out_{k}")
    ref_out = load(args.ref_dir, f"conv_attn_out_{k}")
    prev = load(args.ours_dir, f"block_hidden_{k - 1}") if k else None

    rows, D = ours_attn.shape
    ks = base.shape[1]
    g = proj.shape[0] // (2 * ks)
    group = D // g
    print(f"[conv] rows {rows} D {D} ks {ks} groups {g} group_size {group}", flush=True)

    # DYN = proj @ hidden, seen as (2, ks, g). done here, in fp32, from the reference's own inputs
    dyn_full = (ours_attn.astype(np.float32) @ proj.T.astype(np.float32)).reshape(rows, 2, ks, g)
    dyn = dyn_full[:, 0]

    # The residual is not known without more dumps, but it cancels: `conv_attn_out = residual + conv(x)`
    # with the *same* residual on both sides, so what the hypothesis has to explain is the difference
    # between the two conv outputs — `y_a - y_b` should reproduce `ours_out - ref_out` if rounding is
    # the only difference. Compare the two arithmetic forms against each other and against the gap.
    target_gap = ours_out.astype(np.float64) - ref_out.astype(np.float64)
    target_ours = ours_out - (prev if prev is not None else 0.0)
    target_ref = ref_out - (prev if prev is not None else 0.0)

    y_plain = conv(ours_attn, base[0], dyn, ks, group)
    y_rbf = conv_rbf(ours_attn, base[0], dyn, ks, group)
    gap = y_plain.astype(np.float64) - y_rbf.astype(np.float64)
    print(f"[conv] the two arithmetic forms differ by rel {np.linalg.norm(gap)/np.linalg.norm(target_gap):.3e} "
          f"of the observed ours-vs-ref gap (cosine "
          f"{np.dot(gap.ravel(), target_gap.ravel())/((np.linalg.norm(gap)*np.linalg.norm(target_gap)) or 1):.4f})",
          flush=True)
    print(f"[conv] ours vs ref conv_attn_out (the gap to explain): "
          f"rel {rel(ours_out, ref_out)[0]:.3e}, max {np.abs(target_gap).max():.3e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))