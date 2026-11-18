"""Teacher-forced accept length across 8 real chunks, with a bootstrap interval on every arm.

One chunk (133 positions) gave a ~9% standard error, which cannot rank arms whose gap is 1-2%. Eight
chunks of real assistant text give ~880 positions, and a per-position bootstrap turns "1.068 vs 1.083"
into a statement about whether that difference survives resampling.

Unit: one position = one draft round anchored on the true token, scored by how many of the 7 drafted
slots match the text. Mean accepted slots, P(at least one), and the full-7 rate are reported per arm.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/relja/Projects/tf-wt/dflash2")
sys.path.insert(0, str(REPO / "tools/dflash2/train"))
sys.path.insert(0, str(REPO / "tools/dflash2/quantize"))
import forward as F  # noqa: E402

spec = importlib.util.spec_from_file_location("dflash_fp8", REPO / "tools/dflash2/quantize/dflash_fp8.py")
fp8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp8)

DRAFTER = Path.home() / ".cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6"
TARGET = Path.home() / ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300"
CHUNKS = sorted(Path("/tmp/harv10/out").glob("chunk-*"))
dev = "cuda"
BODY = ("q", "k", "v", "o", "gate", "up", "down")

ARMS = [("affine4 rtn", "/tmp/nv4/aff4_10_rtn_beside_rtn.npz", "_rtn"),
        ("affine4 fitted", "/tmp/nv4/aff4_10_scale_beside_scale.npz", "_fitted")]


def slots(d):
    out = {}
    for i in range(len(d.layers)):
        for n in BODY:
            out[f"L{i}_{n}"] = d.layers[i][n]
    out["fc"] = d.top["fc"]
    return out


def positions_of(fx, per_chunk):
    end = int(fx["manifest"]["pos_end"])
    hi = end - 7                      # a 7-slot lookahead has to stay inside the chunk
    lo = max(8, hi - per_chunk)       # ... and the prefix has to exist behind it
    return list(range(lo, hi))


def measure(d, per_chunk=110):
    per_pos, per_chunk_mean = [], []
    for ch in CHUNKS:
        fx = F.load_fixture(ch, dev)
        ids = fx["ids"].cpu().numpy()
        acc = []
        for p in positions_of(fx, per_chunk):
            props = d.round(F.ctx_rows_for(fx, p), int(ids[p]), p)["proposals"]
            posterior = [int(ids[p + 1 + j]) for j in range(len(props))]
            acc.append(F.acceptance(props, posterior))
        per_pos += acc
        per_chunk_mean.append(float(np.mean(acc)))
        del fx
    return np.asarray(per_pos, dtype=np.float32), per_chunk_mean


def report(label, per_pos, per_chunk):
    """Marginal CI only -- the paired interval comes from `paired()` against the teacher.

    Per-chunk means here span 0.22..3.16, so an independent resample of one arm's mean is dominated by
    which chunk a position came from, not by the quantization. That is why the arms are compared
    position-by-position instead.
    """
    rng = np.random.default_rng(0)
    boots = [per_pos[rng.integers(0, per_pos.size, per_pos.size)].mean() for _ in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    r = {"accept": float(per_pos.mean()), "ci95_marginal": [float(lo), float(hi)],
         "p_ge_1": float((per_pos >= 1).mean()), "full7": float((per_pos >= 7).mean()),
         "positions": int(per_pos.size)}
    spread = ""
    if per_chunk is not None:
        r["per_chunk_mean"] = [round(float(c), 4) for c in per_chunk]
        r["per_chunk_spread"] = [float(np.min(per_chunk)), float(np.max(per_chunk))]
        spread = f"  chunks {min(per_chunk):.3f}..{max(per_chunk):.3f}"
    print(f"{label:16} accept {r['accept']:.3f}  [{lo:.3f}, {hi:.3f}]{spread}  "
          f"P(>=1) {r['p_ge_1']:.3f}  n {r['positions']}")
    return r


CACHE = Path("/tmp/nv4/accept10_perpos.npz")
res, perpos = {}, {}
cached = np.load(CACHE) if CACHE.exists() else {}


def run_arm(key, label, setup=None):
    """Measure once per arm, cache the per-position vector immediately so a later crash is cheap."""
    if key in cached:
        perpos[key] = (cached[key], None)
    else:
        d = F.Drafter(DRAFTER, TARGET, dev)
        if setup is not None:
            setup(d)
        perpos[key] = measure(d)
        np.savez(CACHE, **{k: v[0] for k, v in perpos.items()})
    res[key] = report(label, *perpos[key])


run_arm("bf16", "bf16 (teacher)")

for label, path, suffix in ARMS:
    W = np.load(path)

    def setup(d, W=W, suffix=suffix):
        for k, t in slots(d).items():
            t.copy_(torch.from_numpy(W[f"{k}{suffix}"]).to(t.dtype).to(dev))

    run_arm(label, label, setup)


def setup_fp8(d):
    for k, t in slots(d).items():
        w = t.detach().float().cpu().numpy()
        codes, scale, _ = fp8.quantize_blockwise(w)
        t.copy_(torch.from_numpy(fp8.dequantize(codes, scale).astype(np.float32)).to(torch.bfloat16).to(dev))


run_arm("fp8", "fp8 body", setup_fp8)

# --- paired comparison: resample positions, take the mean of (arm - teacher) -------------------
teacher = perpos["bf16"][0]
base = float(teacher.mean())
rng = np.random.default_rng(1)
idx = rng.integers(0, teacher.size, (4000, teacher.size))
print("\npaired delta vs bf16 on the same positions (95% bootstrap over positions):")
paired = {}
for k, (pp, _ch) in perpos.items():
    d_ = pp - teacher
    boots = d_[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_worse = float((boots >= 0).mean())
    res[k]["delta_vs_teacher"] = float(d_.mean())
    res[k]["delta_ci95_paired"] = [float(lo), float(hi)]
    res[k]["p_no_loss"] = p_worse
    print(f"  {k:16} {d_.mean():+.4f} slots  [{lo:+.4f}, {hi:+.4f}]  "
          f"({100*d_.mean()/base:+6.2f}%)  P(delta>=0) {p_worse:.3f}")

# a paired test against the teacher's own resampling noise, for scale
bn = teacher[idx].mean(axis=1) - base
print(f"\nnoise floor: the teacher against itself resamples to "
      f"[{np.percentile(bn,2.5):+.4f}, {np.percentile(bn,97.5):+.4f}] slots")

out = {"chunks": [c.name for c in CHUNKS], "positions_per_chunk": 110,
       "probe": "teacher-forced accept length, 7 slots, anchored on the true token, real assistant text",
       "analysis": "paired over positions (identical positions per arm), 4000 bootstrap resamples; "
                   "marginal CIs are reported but are dominated by chunk-level variance (0.22..3.16)",
       "variants": res}
Path("/tmp/nv4/accept10.json").write_text(json.dumps(out, indent=1))
print("\nwrote /tmp/nv4/accept10.json and accept10_perpos.npz")
