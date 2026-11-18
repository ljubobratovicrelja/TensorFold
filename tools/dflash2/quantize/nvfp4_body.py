#!/usr/bin/env python3
"""Can the DFlash2 drafter keep its linear layers at NVFP4 weight-only precision, and what does it cost?

`quantize/dflash_fp8.py` already ships the drafter at block-FP8. The next step down is NVFP4 (e2m1
codes, group-16 e4m3 block scales, one fp32 global scale per tensor -- `quantize/nvfp4.py`). Weight
error alone does not answer whether that is usable: what matters is the error *the drafter's own
activations see*, layer by layer, and then whether the block's proposals survive. So this file is
three stages over one calibration set:

  collect  runs the torch port (`train/forward.py`) over N consecutive fixture positions and saves the
           real per-layer inputs the linear weights see -- not synthetic noise, not the training
           distribution, the actual rows the drafter feeds its projections.
  fit      CPU, numpy only, no gradients: quantizes every body linear weight three ways (RTN, a per-group
           scale search, and the same search followed by a GPTQ-shaped residual pass), measures each
           candidate's output error on the collected inputs, and writes the dequantised weights for the
           next stage. It never needs torch: `collect` records the drafter path and `fit` re-reads the
           bf16 checkpoint through a safetensors header.
  eval     the money metric: load the real drafter on the GPU, swap in one variant's body weights at a
           time, run the port over K fixture positions and score the result against the bf16 body with
           exactly the numbers `analysis/head_fidelity.py` uses -- recall@16 of the true top-16, exact
           set rate, top-1 agreement, mean rank, foreign ids -- plus how many of the 7 selector
           proposals per position come out identical.

WHAT IS QUANTIZED, and what is deliberately not
===============================================
Quantized (the linear weights that carry the FLOPs; 7 a layer plus the fuse):

  layers.{i}.self_attn.{q,k,v,o}_proj.weight
  layers.{i}.mlp.{gate,up,down}_proj.weight
  fc.weight

Left in bf16, and why:

  candidate_selector.predecessor_codebook / successor_codebook   lookup tables, not GEMMs: a per-row
      4-bit re-encode changes the selector's arithmetic, not its precision
  candidate_selector.hidden_projection.weight   a [256, 2560] projection into rank-256 selector space
      whose output is consumed as a dot product with a codebook row; 0.3% of the bytes
  every norm vector (input_layernorm, post_attention_layernorm, q_norm, k_norm, hidden_norm, norm)
      and attention_conv/mlp_conv.{base_kernel,kernel_projection}   the convs' dynamic kernels are
      [640, 2560] and the base kernels [2, 2, 2560]: group-16 scales do not line up with a 2-element
      lag axis or a 16-wide conv group, and together they are noise next to the MLP

THE THREE VARIANTS
==================
rtn          what a plain export would do: one global scale 448*6/amax(W), then each run of 16 input
             weights in each row gets e4m3(amax/6) and its e2m1 codes.
scale        per-block scale search: NVFP4 puts one scale on every run of 16 input weights *within a
             row*, so each output row's block tries amax/(6*m) for m in a fixed list around 1 (which
             includes RTN at m=1) and keeps the candidate with the smallest measured
             ||X_g (w_g - w_hat_g)||^2 over the calibration rows. Scale only: the global scale and the
             block layout are untouched, so the result is still one legal NVFP4 tensor.
scale+comp   the same per-group scale, then columns are quantized in input order and each column's
             residual is propagated into the not-yet-quantized ones through H = 2 X^T X (damped as in
             GPTQ), so later groups compensate earlier error. Exact GPTQ over the full k x k inverse,
             at the price of O(k^2) memory per tensor.

Reported per tensor: weight rel-L2 (RTN / fitted), the REAL-input output rel-L2 of the same two, the
block-FP8 baseline's output rel-L2 on the same inputs, and the fit time. The summary means are what a
checkpoint-level decision is made on; the per-tensor rows are where a bad layer shows up. Scales and
residuals are fitted on four fifths of the calibration rows and every output number is measured on the
fifth they never saw -- otherwise a mode that only cancels error along the rows it was handed scores
itself. `output_rel_l2_fitted_train` is the same weight on the rows it was fitted to; the gap between
the two is how much a mode is fitting the calibration set instead of the layer.

    python3 tools/dflash2/quantize/nvfp4_body.py collect --fixture FIX --positions 140 --out calib.npz
    python3 tools/dflash2/quantize/nvfp4_body.py fit --calib calib.npz --out report.json --mode scale
    python3 tools/dflash2/quantize/nvfp4_body.py eval --calib calib.npz --variants rtn,fitted,fp8 \\
        --out eval.json --weights report_beside_scale.npz

`collect` and `eval` need torch and the checkpoints; `fit` needs neither (numpy and the standard
library only, so it runs on a laptop against a calibration file collected elsewhere). All three write
only the paths given on the command line.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HIDDEN = 2560
N_TAPS = 5
BLOCK = 8  # draft rows per round: one anchor plus 7 mask rows

# (tensor name in `Drafter`, checkpoint suffix template, the collected stage that feeds it)
BODY = (
    ("q", "layers.{i}.self_attn.q_proj.weight", "attn_in"),
    ("k", "layers.{i}.self_attn.k_proj.weight", "attn_in"),
    ("v", "layers.{i}.self_attn.v_proj.weight", "attn_in"),
    ("o", "layers.{i}.self_attn.o_proj.weight", "o_in"),
    ("gate", "layers.{i}.mlp.gate_proj.weight", "mlp_in"),
    ("up", "layers.{i}.mlp.up_proj.weight", "mlp_in"),
    ("down", "layers.{i}.mlp.down_proj.weight", "down_in"),
)
STAGE_OF = {key: stage for key, _, stage in BODY}
STAGE_OF["fc"] = "fc_in"
FC_NAME = "fc.weight"

# Multipliers on the RTN step: the candidate scale is amax/(6*m), so m=1 is RTN itself and the search
# can never pick something worse for that group under its own metric.
SCALE_MULTS = (0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25, 1.35, 1.45)
GPTQ_DAMP = 0.01  # 1% of the mean Hessian diagonal, GPTQ's reference value
FP8_BLOCK = 128


def _load(path: Path, name: str):
    """A sibling module by path: this tree has no packages, and one copy of each arithmetic."""
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path}: not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NV = _load(HERE / "nvfp4.py", "nvfp4_primitives")
F8 = _load(HERE / "dflash_fp8.py", "dflash_fp8")


# ---------------------------------------------------------------------------------------------
# Numbers every stage reports, defined once
# ---------------------------------------------------------------------------------------------


def tensor_key(layer: int | None, name: str) -> str:
    """`L3_down`, `fc`: the name a tensor has in the calibration file, the report and the weights npz."""
    return f"L{layer}_{name}" if layer is not None else name


def output_rel_l2(W: np.ndarray, What: np.ndarray, X: np.ndarray, chunk: int = 512) -> float:
    """||X W^T - X What^T||_F / ||X W^T||_F over the calibration rows, accumulated chunk by chunk.

    This is the number that matters: the same weight error is harmless in a tensor whose inputs are
    tiny and expensive in one whose inputs are large, and only the real rows know which it is.
    """
    err2 = ref2 = 0.0
    for r0 in range(0, X.shape[0], max(1, chunk)):
        xb = np.ascontiguousarray(X[r0 : r0 + chunk], dtype=np.float32)
        y = xb @ W.T
        yh = xb @ What.T
        err2 += float(np.sum((y.astype(np.float64) - yh.astype(np.float64)) ** 2))
        ref2 += float(np.sum(y.astype(np.float64) ** 2))
    return float(np.sqrt(err2 / ref2)) if ref2 > 0 else 0.0


def weight_rel_l2(W: np.ndarray, What: np.ndarray) -> float:
    err = float(np.sum((W.astype(np.float64) - What.astype(np.float64)) ** 2))
    ref = float(np.sum(W.astype(np.float64) ** 2))
    return float(np.sqrt(err / ref)) if ref > 0 else 0.0


# ---------------------------------------------------------------------------------------------
# The bf16 drafter checkpoint, header-only, numpy (fit must run without torch)
# ---------------------------------------------------------------------------------------------


class DrafterWeights:
    """The drafter's safetensors shard read lazily: one BF16 tensor at a time, widened to float32.

    81 tensors, ~1.9 GB on disk. Only the body weights are asked for, and each is materialised alone
    (the largest, `down`, is 19.7 M floats = 79 MB as float32).
    """

    def __init__(self, snap: str | Path) -> None:
        self.snap = Path(snap)
        self.path = self.snap / "model.safetensors"
        if not self.path.is_file():
            raise SystemExit(f"{self.path}: the drafter checkpoint is a single model.safetensors")
        self.header, self.start = F8.read_header(self.path)

    def n_layers(self) -> int:
        return len([n for n in self.header if n.endswith("self_attn.q_proj.weight")])

    def get(self, name: str) -> np.ndarray:
        if name not in self.header:
            raise SystemExit(f"{self.path}: no tensor {name}")
        entry = self.header[name]
        if entry["dtype"] != "BF16":
            raise SystemExit(f"{name}: {entry['dtype']}, expected BF16")
        shape = tuple(entry["shape"])
        begin, end = entry["data_offsets"]
        want = int(np.prod(shape, dtype=np.int64)) * 2
        if end - begin != want:
            raise SystemExit(f"{name}: {end - begin} bytes, expected {want} for {shape}")
        raw = np.memmap(self.path, dtype="<u2", mode="r", offset=self.start + begin, shape=shape)
        # bf16 is float32's top half, bit for bit
        return (np.asarray(raw).astype(np.uint32) << np.uint32(16)).view(np.float32)

    def body(self, layer: int | None, key: str) -> tuple[str, np.ndarray]:
        ckpt = FC_NAME if key == "fc" else next(s for k, s, _ in BODY if k == key).format(i=layer)
        return tensor_key(layer, key), np.array(self.get(ckpt), dtype=np.float32, copy=True)


# ---------------------------------------------------------------------------------------------
# Stage 1: collect -- the real per-layer inputs
# ---------------------------------------------------------------------------------------------


def forward_module():
    """`train/forward.py`, the torch port of the drafter round (needs torch; only collect/eval call this)."""
    return _load(HERE.parent / "train" / "forward.py", "dflash_forward")


def _ctx_sample(F, fx, position: int, want: int) -> np.ndarray:
    """`want` evenly spaced prefix tap rows at `position` (12800 wide), as bf16-valued float32."""
    import torch

    width = N_TAPS * HIDDEN
    if want <= 0:
        return np.zeros((0, width), dtype=np.float32)
    rows = F.ctx_rows_for(fx, position)
    n = int(rows.shape[0])
    idx = torch.from_numpy(np.linspace(0, n - 1, min(want, n)).round().astype(np.int64)).to(rows.device)
    return rows.index_select(0, idx).float().cpu().numpy()


def collect(args) -> int:
    """Run the port over N consecutive fixture positions and save the inputs each body weight sees."""
    import torch

    F = forward_module()
    fx = F.load_fixture(args.fixture, device=args.device)
    d = F.Drafter(args.drafter, args.target, device=args.device)
    n_layers = len(d.layers)

    if args.positions < 1:
        raise SystemExit("--positions must be at least 1")
    pos_end = int(fx["pos_end"])
    start = args.start if args.start is not None else pos_end - args.positions
    if start < 0:
        raise SystemExit(f"--positions {args.positions} from pos_end {pos_end} runs before position 0")
    positions = list(range(start, start + args.positions))

    stages = ("attn_in", "o_in", "mlp_in", "down_in")
    buf: dict[str, list[np.ndarray]] = {tensor_key(i, k): [] for i in range(n_layers) for k in stages}
    n_rows = 0
    t0 = time.time()

    for p in positions:
        trace: dict = {}
        F.run_position(d, fx, p, trace=trace)
        for i, layer in enumerate(d.layers):
            buf[tensor_key(i, "attn_in")].append(trace[f"conv_attn_in_{i}"].float().cpu().numpy())
            buf[tensor_key(i, "o_in")].append(trace[f"attn_{i}"].float().cpu().numpy())
            xin = trace[f"conv_mlp_in_{i}"]                     # bf16, as the round feeds gate/up
            buf[tensor_key(i, "mlp_in")].append(xin.float().cpu().numpy())
            # the MLP intermediate, computed exactly as the round does before down_proj
            gate = torch.matmul(xin, layer["gate"].t())
            up = torch.matmul(xin, layer["up"].t())
            act = (gate.float() * torch.sigmoid(gate.float()) * up.float()).to(torch.bfloat16)
            buf[tensor_key(i, "down_in")].append(act.float().cpu().numpy())
        n_rows += int(trace["conv_attn_in_0"].shape[0])
        del trace

    arrays = {key: np.concatenate(vals, axis=0) for key, vals in buf.items()}
    # the prefix taps: fc's input, and (per layer, identical bytes) the context k/v would see
    ctx = _ctx_sample(F, fx, positions[len(positions) // 2], args.ctx_rows)
    arrays["fc_in"] = ctx.astype(np.float32)
    for i in range(n_layers):
        arrays[f"L{i}_ctx"] = ctx

    meta = {
        "fixture": str(Path(args.fixture).resolve()),
        "positions": positions,
        "start": int(start),
        "pos_end": pos_end,
        "drafter": str(Path(args.drafter).resolve()),
        "target": str(Path(args.target).resolve()),
        "rows": int(n_rows),
        "block": int(fx["block"]),
        "window": int(fx["window"]),
        "hidden": int(fx["hidden"]),
        "taps": int(fx["n_taps"]),
        "vocab": int(d.head.shape[0]),
        "ctx_rows": int(ctx.shape[0]),
        "stage_keys": sorted(arrays),
        "note": ("L{i}_attn_in feeds q/k/v; L{i}_o_in is the attention output, o_proj's input; "
                 "L{i}_mlp_in feeds gate/up; L{i}_down_in is silu(gate)*up; L{i}_ctx and fc_in are "
                 "the prefix's 5-tap rows (12800 wide)"),
    }

    np.savez(args.out, meta=np.array(json.dumps(meta)), **arrays)
    mb = sum(a.nbytes for a in arrays.values()) / 1e6
    print(f"collected {len(positions)} positions x {BLOCK} rows = {n_rows} rows, {n_layers} layers")
    print(f"  positions {positions[0]}..{positions[-1]}  ctx rows {meta['ctx_rows']}  "
          f"{len(arrays)} arrays, {mb:.0f} MB uncompressed")
    print(f"  {time.time() - t0:.1f}s -> {args.out}")
    return 0


# ---------------------------------------------------------------------------------------------
# Stage 2: fit -- RTN, per-group scale search, GPTQ residual pass
# ---------------------------------------------------------------------------------------------


def rtn_group_scales(Wg: np.ndarray, gs: float) -> np.ndarray:
    """The scale plain RTN would give each output row's block: e4m3(amax/6) in the scaled domain."""
    if Wg.size == 0:
        return np.full(Wg.shape[0], np.float32(1e-12), dtype=np.float32)
    amax = np.max(np.abs(Wg), axis=1)
    s = np.asarray(NV.e4m3_round((amax / NV.E2M1_MAX * gs).astype(np.float32)), dtype=np.float32).ravel()
    return np.where(s <= 0.0, np.float32(1e-12), s).astype(np.float32)


def pick_group_scales(Wg: np.ndarray, G: np.ndarray, gs: float, mults=SCALE_MULTS) -> np.ndarray:
    """One block scale per output row (scaled domain, on the e4m3 grid), minimising that row's error.

    The NVFP4 block is 16 consecutive input weights *within a row*, so a [n, 16] slice of the weight
    has n independent scales and n independent errors -- and each row's error is delta^T G delta, with
    G = X_g^T X_g over the calibration rows for these 16 input columns. That is exactly the group's
    contribution to the output error, with no (rows x neurons) product ever materialised.

    Candidates are amax/(6*m); m=1 is RTN, so no row can come out worse than RTN on this metric.
    """
    n = Wg.shape[0]
    base = (np.max(np.abs(Wg), axis=1) if Wg.size else np.zeros(n, np.float32)) / NV.E2M1_MAX
    best_err = np.full(n, np.inf, dtype=np.float64)
    best = np.full(n, np.float32(1e-12), dtype=np.float32)
    for m in mults:
        s = np.asarray(NV.e4m3_round((base / m * gs).astype(np.float32)), dtype=np.float32).ravel()
        s = np.where(s <= 0.0, np.float32(1e-12), s).astype(np.float32)
        step = (s / np.float32(gs))[:, None]
        delta = Wg - NV.e2m1_round(Wg / step) * step
        err = np.sum((delta @ G).astype(np.float64) * delta.astype(np.float64), axis=1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best = np.where(better, s, best).astype(np.float32)
    return best


def fit_scales(W: np.ndarray, X: np.ndarray, gs: float, group: int) -> tuple[np.ndarray, int]:
    """Per-row, per-block scale search -> dequantised weight, and how many blocks left RTN's scale."""
    n, k = W.shape
    nb = k // group
    scales = np.empty((n, nb), dtype=np.float32)
    moved = 0
    for gi, g0 in enumerate(range(0, k, group)):
        g1 = g0 + group
        Wg = np.ascontiguousarray(W[:, g0:g1], dtype=np.float32)
        Xg = np.ascontiguousarray(X[:, g0:g1], dtype=np.float32)
        scales[:, gi] = pick_group_scales(Wg, Xg.T @ Xg, gs)
        moved += int(np.count_nonzero(scales[:, gi] != rtn_group_scales(Wg, gs)))
    return NV.encode_nvfp4(W, scales, gs, group), moved


def fit_scales_gptq(W: np.ndarray, X: np.ndarray, gs: float, group: int,
                    damp: float = GPTQ_DAMP) -> tuple[np.ndarray, int]:
    """Scale search + GPTQ: quantize input columns in order, propagate each residual into the rest.

    H = 2 X^T X, damped by `damp` x its mean diagonal, inverted once and Cholesky-factored upper so
    its rows are the elimination-order inverses the column loop needs. The per-column update is
    Frantar et al.'s, unblocked (one column at a time): a column's error divided by Hinv[j,j] is
    subtracted from every later column along Hinv[j, j:]. The factor 2 cancels in both places; it is
    kept because it is the matrix the method is defined on.

    Scale selection stays per row here too: a block's rows quantize with their own steps, and a row's
    residual enters the same k x k Hessian because the update is linear in the error.

    Cost: O(k^2) floats for H and Hinv (down: 7680^2 x 4 B = 236 MB each, fc: 655 MB each) and
    O(rows * k^2) flops. That is the honest price of the exact pass; `--mode scale` is the cheap one.
    """
    n, k = W.shape
    H = X.T.astype(np.float32) @ np.ascontiguousarray(X, dtype=np.float32)
    H *= 2.0
    mean_diag = float(np.diag(H).mean()) if k else 0.0
    H[np.diag_indices(k)] += np.float32(damp * max(mean_diag, 1e-30))
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Hinv = np.linalg.pinv(H)
    del H
    Hinv1 = None
    for extra in (0.0, 10.0):
        if extra:
            Hinv[np.diag_indices(k)] += np.float32(extra * damp * max(mean_diag, 1e-30))
        try:
            Hinv1 = np.linalg.cholesky(Hinv).T  # upper: Hinv1^T Hinv1 = Hinv
            break
        except np.linalg.LinAlgError:
            continue
    if Hinv1 is None:
        raise SystemExit("Hessian inverse is not positive definite even after damping; use --mode scale")

    nb = k // group
    scales = np.empty((n, nb), dtype=np.float32)
    moved = 0
    Wq = np.array(W, dtype=np.float32, copy=True)
    for gi, g0 in enumerate(range(0, k, group)):
        g1 = g0 + group
        Wg = np.ascontiguousarray(W[:, g0:g1], dtype=np.float32)
        Xg = np.ascontiguousarray(X[:, g0:g1], dtype=np.float32)
        scales[:, gi] = pick_group_scales(Wg, Xg.T @ Xg, gs)
        moved += int(np.count_nonzero(scales[:, gi] != rtn_group_scales(Wg, gs)))
        step = (scales[:, gi] / np.float32(gs)).astype(np.float32)
        for j in range(g0, g1):
            w = Wq[:, j].copy()
            q = NV.e2m1_round(w / step) * step
            Wq[:, j] = q
            djj = float(Hinv1[j, j])
            if djj <= 0.0 or j + 1 >= k:
                continue
            e = (w - q) / np.float32(djj)
            Wq[:, j + 1 :] -= e[:, None] * Hinv1[j, j + 1 :][None, :]
    return NV.encode_nvfp4(Wq, scales, gs, group), moved


def fp8_baseline(W: np.ndarray, X: np.ndarray) -> dict | None:
    """The repo's own block-FP8 on the same tensor: codes + 128x128 scales, dequantised, scored.

    Shape permitting only: the block layout needs both dims divisible by 128, which every body tensor
    is (2560, 4096, 1024, 7680, 12800 are multiples of 128), but a small smoke matrix may not be.
    """
    n, k = W.shape
    if n % FP8_BLOCK or k % FP8_BLOCK:
        return None
    codes, scale, stats = F8.quantize_blockwise(W, FP8_BLOCK)
    deq = F8.dequantize(codes, scale, FP8_BLOCK)
    return {
        "label": f"fp8 (block {FP8_BLOCK}x{FP8_BLOCK})",
        "output_rel_l2": output_rel_l2(W, deq, X),
        "weight_rel_l2": float(stats["rel_l2"]),
    }


def fit(args) -> int:
    """Quantize every body weight off one calibration file; write the report and the dequantised weights."""
    calib = np.load(args.calib)
    if "meta" not in calib:
        raise SystemExit(f"{args.calib}: no meta array; was this written by `collect`?")
    meta = json.loads(str(calib["meta"]))
    mode, group = args.mode, args.group
    if group <= 0 or group % 2:
        raise SystemExit(f"--group {group}: NVFP4 block scales cover whole groups of weights")

    weights = DrafterWeights(meta["drafter"])
    n_layers = weights.n_layers()
    out_path = Path(args.out)
    weights_path = out_path.with_name(f"{out_path.stem}_beside_{mode}.npz")

    todo: list[tuple[int | None, str]] = [(None, "fc")]
    for i in range(n_layers):
        todo += [(i, key) for key, _, _ in BODY]

    tensors: list[dict] = []
    store: dict[str, np.ndarray] = {}
    per_tensor_s: dict[str, float] = {}
    t_start = time.time()

    for layer, key in todo:
        name, W = weights.body(layer, key)
        calib_key = tensor_key(layer, STAGE_OF[key])
        if calib_key not in calib:
            raise SystemExit(f"{args.calib}: no {calib_key} for {name}")
        X = np.ascontiguousarray(calib[calib_key], dtype=np.float32)
        if X.shape[1] != W.shape[1]:
            raise SystemExit(f"{name}: calibration inputs are {X.shape[1]}-wide, the weight expects "
                             f"{W.shape[1]}")

        # Fit on 4/5 of the rows, measure on the fifth the fit never saw: a scale or a residual that
        # only cancels error along the calibration rows would otherwise score itself.
        hold = np.zeros(X.shape[0], dtype=bool)
        if X.shape[0] >= 10:
            hold[4::5] = True
        X_fit = X[~hold] if hold.any() else X
        X_measure = X[hold] if hold.any() else X

        t0 = time.time()
        gs = NV.global_scale(W)  # 448*6/amax, recorded in the report so the number can be reproduced
        W_rtn, rel_rtn = NV.quantize_nvfp4_scaled(W, group, gs)
        if mode == "rtn":
            W_fit, moved = W_rtn, 0
        elif mode == "scale":
            W_fit, moved = fit_scales(W, X_fit, gs, group)
        else:
            W_fit, moved = fit_scales_gptq(W, X_fit, gs, group)
        fp8 = fp8_baseline(W, X_measure)  # RTN in a different format: nothing to fit, nothing held out
        fit_s = time.time() - t0

        out_rtn = output_rel_l2(W, W_rtn, X_measure, args.chunk)
        out_fit = output_rel_l2(W, W_fit, X_measure, args.chunk)
        out_fit_train = output_rel_l2(W, W_fit, X_fit, args.chunk)
        tensors.append({
            "name": name,
            "shape": list(W.shape),
            "blocks": int(W.size // group),
            "global_scale": gs,
            "weight_rel_l2_rtn": rel_rtn,
            "weight_rel_l2_fitted": weight_rel_l2(W, W_fit),
            "output_rel_l2_rtn": out_rtn,
            "output_rel_l2_fitted": out_fit,
            "output_rel_l2_fitted_train": out_fit_train,
            "output_rel_l2_gain": out_rtn - out_fit,
            "blocks_off_rtn": int(moved),
            "fp8": fp8,
            "time_s": round(fit_s, 3),
        })
        per_tensor_s[name] = round(fit_s, 3)
        store[f"{name}_rtn"] = np.ascontiguousarray(W_rtn, dtype=np.float32)
        store[f"{name}_fitted"] = np.ascontiguousarray(W_fit, dtype=np.float32)
        fp8_out = f"{fp8['output_rel_l2']:.5f}" if fp8 else "    -"
        print(f"  {name:14s} {str(tuple(W.shape)):14s} out rtn {out_rtn:.5f}  fit {out_fit:.5f}  "
              f"(train {out_fit_train:.5f})  fp8 {fp8_out}  {fit_s:.2f}s", flush=True)

    def mean_of(fn):
        vals = [v for v in (fn(t) for t in tensors) if v is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "tensors": len(tensors),
        "mean_output_rel_l2_rtn": mean_of(lambda t: t["output_rel_l2_rtn"]),
        "mean_output_rel_l2_fitted": mean_of(lambda t: t["output_rel_l2_fitted"]),
        "mean_output_rel_l2_fp8": mean_of(lambda t: t["fp8"]["output_rel_l2"] if t["fp8"] else None),
        "mean_output_rel_l2_fitted_train": mean_of(lambda t: t["output_rel_l2_fitted_train"]),
        "mean_weight_rel_l2_rtn": mean_of(lambda t: t["weight_rel_l2_rtn"]),
        "mean_weight_rel_l2_fitted": mean_of(lambda t: t["weight_rel_l2_fitted"]),
        "max_output_rel_l2_gain": max((t["output_rel_l2_gain"] for t in tensors), default=0.0),
        "tensors_worse_than_rtn": sum(1 for t in tensors if t["output_rel_l2_gain"] < 0),
        "blocks_off_rtn": sum(t["blocks_off_rtn"] for t in tensors),
        "blocks_total": sum(t["blocks"] for t in tensors),
    }

    caveats = [
        "inputs are the bf16 body's activations: layers are fitted independently, so error "
        "accumulation down the stack is not modelled (eval measures it end to end)",
        "every output rel-L2 is measured on rows held out of the fit (every fifth calibration row); "
        "output_rel_l2_fitted_train is the same weight on the rows it was fitted to, and the gap "
        "between the two is how much a mode is fitting the calibration set rather than the layer",
        "4-bit codes on a group-16 scale are the format; these are weights only (W4A16), so the "
        "numbers say nothing about a kernel that quantizes activations too",
        "fp8 is the repo's 128x128 block layout: an FP8 with smaller blocks would score better, so "
        "this is the shipped baseline, not FP8's best case",
    ]
    if mode == "scale":
        caveats.append("the search includes m=1, i.e. RTN, so a group cannot lose on its own metric; "
                       "a tensor's output number can still tick up because neighbouring groups interact")
    if mode == "scale+comp":
        caveats.append("scale+comp is exact GPTQ over the full k x k Hessian: O(k^2) memory and "
                       "O(rows * k^2) time, dominated by down_proj and fc on the real tensors")
        caveats.append("GPTQ wants more calibration rows than input columns; with 140 positions the "
                       "calibration has ~1120 rows against 2560/7680/12800 columns, so H is rank "
                       "deficient and the residual pass can trade weight error for error along rows "
                       "it was fitted to. Read output_rel_l2_fitted against output_rel_l2_fitted_train "
                       "(the training-set number) before believing any gain; if the gap is wide, the "
                       "pass is fitting the calibration, not the layer")
    if mode == "rtn":
        caveats.append("rtn is the plain export: one global scale, e4m3(amax/6) a group, no search")

    report = {
        "tool": {"name": "nvfp4_body", "stage": "fit",
                 "note": "CPU, numpy only, no gradients; output error measured on collected "
                         "calibration inputs, not on synthetic noise"},
        "mode": mode,
        "group": group,
        "scale_mults": list(SCALE_MULTS),
        "gptq_damp": GPTQ_DAMP if mode == "scale+comp" else None,
        "calibration": {k: meta[k] for k in
                        ("fixture", "positions", "rows", "drafter", "target") if k in meta},
        "protocol": "fit on 80% of the calibration rows (every 5th held out), measure on the held-out "
                    "20%; RTN and fp8 are scored on those same held-out rows",
        "quantized_tensors": [t["name"] for t in tensors],
        "not_quantized": [
            "candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook",
            "candidate_selector.hidden_projection.weight",
            "layers.{i}.input_layernorm.weight", "layers.{i}.post_attention_layernorm.weight",
            "layers.{i}.self_attn.q_norm.weight", "layers.{i}.self_attn.k_norm.weight",
            "layers.{i}.attention_conv.base_kernel", "layers.{i}.attention_conv.kernel_projection.weight",
            "layers.{i}.mlp_conv.base_kernel", "layers.{i}.mlp_conv.kernel_projection.weight",
            "hidden_norm.weight", "norm.weight",
        ],
        "tensors": tensors,
        "summary": summary,
        "timings": {"total_s": round(time.time() - t_start, 2), "per_tensor_s": per_tensor_s},
        "weights_npz": str(weights_path),
        "caveats": caveats,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    np.savez(weights_path, **store)

    gib = sum(a.nbytes for a in store.values()) / 2**30
    print(f"\nmode {mode}  group {group}  {len(tensors)} tensors  "
          f"{summary['blocks_off_rtn']}/{summary['blocks_total']} blocks off RTN")
    fp8_mean = summary["mean_output_rel_l2_fp8"]
    fp8_txt = f"{fp8_mean:.5f}" if fp8_mean is not None else "n/a (no 128-divisible tensor)"
    print(f"  mean output rel-L2  rtn {summary['mean_output_rel_l2_rtn']:.5f}  "
          f"fitted {summary['mean_output_rel_l2_fitted']:.5f}  "
          f"(fitted, train rows {summary['mean_output_rel_l2_fitted_train']:.5f})  fp8 {fp8_txt}")
    print(f"  mean weight rel-L2  rtn {summary['mean_weight_rel_l2_rtn']:.5f}  "
          f"fitted {summary['mean_weight_rel_l2_fitted']:.5f}")
    print(f"  {summary['tensors_worse_than_rtn']} tensor(s) worse than RTN on output error; "
          f"{report['timings']['total_s']:.1f}s total")
    print(f"wrote {out_path} and {weights_path} ({gib:.2f} GiB of float32 weights)")
    return 0


# ---------------------------------------------------------------------------------------------
# Stage 3: eval -- swap the weights in, run the real round, score the proposals
# ---------------------------------------------------------------------------------------------


def head_metrics_module():
    """`analysis/head_fidelity.py` for `top_k`/`score_row`/`summarize`: the same metrics, one definition."""
    return _load(HERE.parent / "analysis" / "head_fidelity.py", "head_fidelity_metrics")


def load_variant(name: str, weights_path: Path | None, originals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The dequantised body weights of one variant, as numpy float32 keyed like the calibration file."""
    if name == "fp8":
        out = {}
        for key, w in originals.items():
            n, k = w.shape
            if n % FP8_BLOCK or k % FP8_BLOCK:
                raise SystemExit(f"{key}: {w.shape} is not a multiple of {FP8_BLOCK}; fp8 cannot express it")
            codes, scale, _ = F8.quantize_blockwise(w, FP8_BLOCK)
            out[key] = F8.dequantize(codes, scale, FP8_BLOCK)
        return out
    if name not in ("rtn", "fitted"):
        raise SystemExit(f"unknown variant {name}; expected rtn, fitted or fp8")
    if weights_path is None or not Path(weights_path).is_file():
        raise SystemExit(f"--weights: no dequantised weights at {weights_path}; run `fit` first and "
                         f"pass the report_beside_<mode>.npz it wrote")
    npz = np.load(weights_path)
    out = {}
    for key, orig in originals.items():
        npz_key = f"{key}_{name}"
        if npz_key not in npz:
            raise SystemExit(f"{weights_path}: no {npz_key}; was it written by a `fit` of this mode?")
        arr = np.asarray(npz[npz_key], dtype=np.float32)
        if arr.shape != orig.shape:
            raise SystemExit(f"{npz_key}: shape {arr.shape}, checkpoint says {orig.shape}")
        out[key] = arr
    return out


def compare_rows(ref_ids: np.ndarray, var_logits: np.ndarray, p1: float, k: int, HM) -> dict:
    """One row's fidelity: `head_fidelity.score_row`'s numbers plus the foreign ids they imply."""
    row = HM.score_row(var_logits, var_logits, None, k, ref_ids, p1)
    row["foreign"] = float(k - np.intersect1d(ref_ids, HM.top_k(var_logits, k), assume_unique=True).size)
    return row


def evaluate(args) -> int:
    """Load the real drafter, run each weight variant over the fixture, compare against bf16."""
    import torch

    F = forward_module()
    HM = head_metrics_module()
    calib = np.load(args.calib)
    meta = json.loads(str(calib["meta"]))
    fixture = args.fixture or meta["fixture"]
    fx = F.load_fixture(fixture, device=args.device)
    d = F.Drafter(meta["drafter"], meta["target"], device=args.device)

    positions = list(meta["positions"])
    if args.positions is not None:
        positions = positions[: args.positions]
    if not positions:
        raise SystemExit(f"{args.calib}: no positions to run; re-collect with --positions N")

    originals = {}
    for layer in range(len(d.layers)):
        for key, _, _ in BODY:
            originals[tensor_key(layer, key)] = d.layers[layer][key].float().cpu().numpy()
    originals["fc"] = d.top["fc"].float().cpu().numpy()

    k = 16
    ref = {"ids": {}, "p1": {}, "proposals": {}}
    for p in positions:
        out = F.run_position(d, fx, p)
        logits = d.head_logits(out["hn"])
        ref["ids"][p] = torch.topk(logits, k, dim=-1).indices.cpu().numpy()
        ref["p1"][p] = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().numpy()
        ref["proposals"][p] = list(out["proposals"])
    print(f"reference: {len(positions)} bf16 positions, {len(positions) * (BLOCK - 1)} rows")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    reports = {}
    for variant in variants:
        # only rtn/fitted need fit's companion npz: fp8 is recomputed from the bf16 weights on the fly
        weights_path = None
        if variant in ("rtn", "fitted"):
            weights_path = Path(args.weights) if args.weights else _find_weights(args.calib, args.out)
        tensors = load_variant(variant, weights_path, originals)
        _install(d, tensors, args.device)
        rows, per_slot, identical = [], {s: [] for s in range(BLOCK - 1)}, 0
        t0 = time.time()
        for p in positions:
            out = F.run_position(d, fx, p)
            logits = d.head_logits(out["hn"]).float()
            identical += int(list(out["proposals"]) == ref["proposals"][p])
            for s in range(BLOCK - 1):
                row = compare_rows(ref["ids"][p][s], logits[s].cpu().numpy(), float(ref["p1"][p][s]), k, HM)
                rows.append(row)
                per_slot[s].append(row)
        _restore(d, originals, args.device)
        o = HM.summarize(rows)
        foreign = float(np.mean([r["foreign"] for r in rows])) if rows else None
        reports[variant] = {
            "weights": "fp8 (block 128x128)" if variant == "fp8" else f"nvfp4 {variant}",
            "rows": len(rows),
            "overall": o,
            "mean_foreign": foreign,
            "proposals_identical": identical,
            "proposals_total": len(positions),
            "per_slot": [{"slot": s, **HM.summarize(per_slot[s])}
                         for s in range(BLOCK - 1) if per_slot[s]],
            "eval_s": round(time.time() - t0, 2),
        }
        print(f"  {variant:7s} recall16 {o['recall_at_k']:.4f}  set-exact {o['set_exact']:.4f}  "
              f"top1 {o['top1_agree']:.4f}  rank {o['mean_rank']:.3f}  "
              f"foreign {foreign:.2f}  proposals {identical}/{len(positions)}")

    report = {
        "tool": {"name": "nvfp4_body", "stage": "eval",
                 "note": "the torch port over fixture positions; metrics are head_fidelity's own functions"},
        "calibration": {"fixture": fixture, "positions": [positions[0], positions[-1]],
                        "count": len(positions), "rows": meta.get("rows")},
        "variants": reports,
        "caveats": [
            "one fixture: this is the drafter's proposal fidelity on that text, not end-to-end acceptance",
            "proposals_identical counts positions whose 7-token list matches bf16 exactly; a flip at "
            "slot 6 is counted the same as a flip at slot 0",
            "the head is bf16 in every variant: this isolates the body, and the head's own 4-bit "
            "fidelity is what analysis/head_fidelity.py measures",
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


def _install(d, tensors: dict[str, np.ndarray], device: str) -> None:
    """Swap the body weights in place; the embedding table and head are never touched."""
    import torch

    for key, arr in tensors.items():
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=torch.bfloat16)
        if key == "fc":
            d.top["fc"] = t
        else:
            layer, name = key.split("_", 1)
            d.layers[int(layer[1:])][name] = t


def _restore(d, originals: dict[str, np.ndarray], device: str) -> None:
    _install(d, originals, device)


def _find_weights(calib: str, out: str) -> Path:
    """Locate `fit`'s companion npz: beside the report, beside the calibration file, or in the cwd."""
    for root in (Path(out).resolve().parent, Path(calib).resolve().parent, Path.cwd()):
        hits = sorted(root.glob("*_beside_*.npz"))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise SystemExit(f"{root}: several fitted-weight files {[h.name for h in hits]}; pass --weights")
    raise SystemExit("no *_beside_*.npz found; run `fit` and pass --weights PATH")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)

    c = sub.add_parser("collect", help="dump the real per-layer inputs the body weights see")
    c.add_argument("--fixture", required=True, help="fixture directory (manifest.json + ring.bin + ids.bin)")
    c.add_argument("--positions", type=int, required=True, help="how many consecutive positions to run")
    c.add_argument("--out", required=True, help="the .npz to write")
    c.add_argument("--drafter", default=None, help="drafter snapshot (config.json + model.safetensors)")
    c.add_argument("--target", default=None, help="target snapshot: the embedding table and LM head")
    c.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    c.add_argument("--ctx-rows", type=int, default=4096,
                   help="prefix tap rows kept per layer for context-aware k/v (default 4096)")
    c.add_argument("--start", type=int, default=None, help="first position (default pos_end - positions)")

    f = sub.add_parser("fit", help="quantize the body weights off a calibration file (CPU, numpy)")
    f.add_argument("--calib", required=True, help="a .npz written by `collect`")
    f.add_argument("--out", required=True, help="the JSON report to write")
    f.add_argument("--mode", default="scale", choices=("rtn", "scale", "scale+comp"))
    f.add_argument("--group", type=int, default=NV.FP4_GROUP, help="weights per block scale (default 16)")
    f.add_argument("--chunk", type=int, default=512, help="calibration rows per output-error chunk")

    e = sub.add_parser("eval", help="run the real drafter with each weight variant and score the proposals")
    e.add_argument("--calib", required=True, help="the same .npz `fit` read")
    e.add_argument("--variants", default="rtn,fitted,fp8")
    e.add_argument("--out", required=True, help="the JSON report to write")
    e.add_argument("--fixture", default=None, help="override the fixture recorded in the calibration file")
    e.add_argument("--positions", type=int, default=None, help="how many positions (default: all collected)")
    e.add_argument("--weights", default=None, help="fit's *_beside_<mode>.npz (default: search beside --out)")
    e.add_argument("--device", default="cuda")

    args = ap.parse_args(argv)
    if args.stage == "collect":
        _default_snapshots(args)
        return collect(args)
    if args.stage == "fit":
        return fit(args)
    return evaluate(args)


def _default_snapshots(args) -> None:
    """Fill --drafter/--target from the HF cache the other tools already default to, if unset."""
    hf = Path.home() / ".cache/huggingface/hub"
    defaults = {
        "drafter": hf / "models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6",
        "target": hf / "models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300",
    }
    for name, path in defaults.items():
        if getattr(args, name) is None:
            if not path.is_dir():
                raise SystemExit(f"--{name}: no snapshot at {path}; pass it explicitly")
            setattr(args, name, str(path))


if __name__ == "__main__":
    sys.exit(main())
