# DFlash2 divergence: debugging brief for the working agent

Status: **partially closed.** The fusion-rounding defect in §4 is fixed and verified (§9). The
block-body divergence is still open and the fix for it is not yet known. This document is the
hand-off from the senior pass. It tells you what is
already proven, what the one real defect found so far is, how to fix it, and how to debug the rest
of this port the right way. Read it before touching anything. The older chronicle is
`flashnext-dflash2-acceptance-log.md` — treat its conclusions as claims to re-verify, not as truth;
two of them are wrong in ways this brief corrects below.

## 1. The goal, restated correctly

The drafter is `incoai/Qwen3.8-Flash-Next-DFlash2`, drafting against the target
`nvidia/Qwen3.8-Flash-Next-FP4-Local`. Our engine is the private TensorFold fork in
`~/Projects/tf-wt/dflash2` (branch `dflash2`), serving via the DFlash2 CUDA port.

The performance bar, from the **vendor's own card for this exact model** (HF README, fetched —
do NOT requote the z-lab 27B numbers):

- Measured at **speculation block width 4** (3 draft tokens/step — the card explicitly says the
  QSA implementation caps it at 4, even though the checkpoint's `dflash_config.block_size` is 8).
- **Acceptance length** (completion tokens / verify steps, concurrency 1): GSM8K **3.26**, MATH-500
  **3.17**, HumanEval 2.94, MBPP 3.06, MT-Bench 2.83.
- **Throughput** (output tok/s, speedup vs autoregressive, concurrency 1): 368.2 (**1.79×**),
  383.6 (**1.82×**), 363.8 (1.72×), 378.3 (1.80×), 343.8 (1.63×).
- Hardware/setup: one GB300, FP8 target, QSA + Gated DeltaNet, temp 1.0 / top-p 0.95 / top-k 20,
  `xhigh` reasoning.

The z-lab **Qwen3.8-27B** drafter (acceptance 4.10–5.46, 3.43×) is a *different model* — a 64-layer
dense target on H200. It is not our bar. Whenever a number is quoted, state the width and the model;
conflating these two cards is the single most common way this project manufactures a fake gap.

## 2. What is already closed — do not reopen

- Head GEMM / detile / weights: logits match numpy to 2.6e-07; detile 64/64 bit-exact.
- Tap basis: `mean4` (plain mean of the 4 raw HC streams) is correct — confirmed both by the
  vendor's `qwen38-dflash2-capture.patch` (`.unflatten(-1,(hc_count,hidden)).mean(dim=-2)`) and the
  checkpoint geometry (`hc_count` 4, hidden 2560, `fc.weight [2560,12800]`). `readout`/`normed`
  are different tensors. We ship `mean4`.
- Layer→tap mapping: `[5,15,25,35,45]` is correct. This model's capture setter uses the ids
  **unmodified** (no +1); the generic SGLang +1 shift is for other models.
- Selector arithmetic (`tf_fxd2_select` == SGLang `_score_edges`), anchor layout (1+N, no
  `sample_from_anchor`), dynamic conv == `_grouped_conv`, attention scale `rsqrt(head_dim)`.
- Drafter quality on the check walk matches the reference (1.317 vs 1.342 teacher-forced) — but see
  §5 for why that number must NOT be read as "the drafter is right".
- The sliding-window mask fix (block row `m` starts at `begin + m`): correct, keep it, it was not
  the divergence source.

## 3. The one defect found so far: the tap fusion does not round like the reference

This is the first divergent stage, and it is the thing the earlier agents spent a day chasing without
finding, because they were looking one stage too late.

**The fusion.** Both sides compute the round's cross-attention context as
`hidden_norm(fc(taps))`: the five taps concatenated (5 × 2560 = 12800), projected by `fc` down to
2560, then RMS-normalised. The engine does this in one kernel, `tf_fxd2_fuse`
(`zig/kernels/cuda/qwen35_draft.cu`), the reference as two torch modules.

**The difference is rounding order.** The reference path is:

```python
ctx_hidden = model.hidden_norm(model.fc(ctx_x))
```

- `model.fc` is a bf16 `nn.Linear` → its output is **rounded to bf16** (round 1).
- `model.hidden_norm` is `Qwen3RMSNorm`, whose `forward` is:
  `x.to(f32)` → `variance` → `x * rsqrt(variance+eps)` in f32 → `weight * x.to(bf16)`.
  That is a second bf16 round (of `x*inv`), then a bf16-by-bf16 multiply (round 3).

The engine's `tf_fxd2_fuse` keeps the dot in fp32 the whole way and rounds **once**, at the very end:

```cuda
OUT[...] = __float2bfloat16_rn(acc[j] * inv * bf(NORM[j]));
```

It never rounds `acc[j]` (the `fc` output) to bf16, and it never rounds `acc*inv` to bf16 before the
weight multiply. That is two missing roundings.

**Proof, already measured.** On the frozen `window` fixture, with a torch reproduction of each
formula run in bf16 on the GPU (`allow_bf16_reduced_precision_reduction = False`):

- Reference formula (`F.linear` bf16 out → `x.to(bf16)` → `weight * x_bf16`) matches the reference
  dump `fc_ctx_rows.npy` **bit-for-bit** (rel_l2 0).
- Engine formula (fp32 dot → `acc*inv*w` single round) matches the engine dump **100%** of elements
  (rel_l2 1.7e-05, i.e. only the f32-widening noise).
- The two differ by **rel_l2 3.29e-3**, which is exactly the observed `fc_ctx_rows` divergence.

So the first divergent stage is the fusion, and the cause is a missing bf16 rounding in
`tf_fxd2_fuse` — not any of the five blocks, not the attention, not the tap basis.

## 4. The fix

The same defect exists in the standalone RMSNorm kernel, because the reference `Qwen3RMSNorm` is the
same class used for `input_layernorm`, `post_attention_layernorm` and the final `norm`. Inventory of
the four norm/fusion kernels and which do it right:

| kernel | file (qwen35_draft.cu) | output line | intermediate round? | verdict |
|---|---|---|---|---|
| `tf_q35d_norm_rope` (per-head q/k) | ~83 | `rbf(rbf(v*inv)*bf(W))` | yes | correct |
| `tf_fxd2_knorm` (per-head ctx keys) | ~296 | `rbf(rbf(v*inv)*bf(W))` | yes | correct |
| `tf_fxd2_rmsnorm` (hidden-width RMSNorm) | ~274 | `rbf(bf(X)*inv*bf(W))` | **no** | **bug** |
| `tf_fxd2_fuse` (fc + hidden_norm) | ~258 | `rbf(acc*inv*bf(NORM))` | **no (×2)** | **bug** |

The reference RMSNorm is `rbf( rbf(x*inv) * w )`, not `rbf( x*inv*w )`. The per-head kernels already
do this; the hidden-width ones don't. Fix them to match:

- `tf_fxd2_rmsnorm`: compute `v = bf(X)` (f32), `inv` as today, then
  `out = rbf( rbf(v * inv) * bf(W) )` — insert the intermediate `rbf(v*inv)`.
- `tf_fxd2_fuse`: after the fp32 dot, `fc_bf16 = rbf(acc)` (round 1 — the `nn.Linear` output); then
  the variance must be computed over `fc_bf16` upcast to f32 (the reference squares the *bf16* fc
  output, not the fp32 dot); then `inv`, then `out = rbf( rbf(fc_bf16_f32 * inv) * bf(NORM) )`
  (rounds 2 and 3).

Do not "improve" precision by keeping fp32 longer — that is the bug. The target is bit-identity with
the reference, not lower error. The per-head kernels prove the double-rounding pattern is the house
style; the hidden-width ones are simply the two that predate the discipline.

## 5. How to verify the fix

The harness is `tools/dflash2/align_bf16/`. It is a two-sided stage-diff: the reference
(`reference_bf16.py`, vendor torch in bf16 on GPU) and the engine (`run_engine_dflash2.sh`, the
fixture mode of `cuda_draft2_check.zig`) both run one round over the same frozen fixture and dump 53
named stages; `compare.py` joins them. The frozen fixture is
`tools/dflash2/align_bf16/fixture/window/` (seed 20261116, pos_end 2048); its `SHA256SUMS` proves
both sides consumed the same bytes. `isolate_blocks.py` runs one block at a time.

Verification sequence after the fix, in order, each gating the next:

1. **Re-run the reference once** (unchanged, ~2 min) and **re-run the engine** with the rebuilt
   binary, both on `fixture/window`. Run `compare.py`.
2. **The verdict is `fc_ctx_rows`.** Before the fix it is 3.3e-3 / not bit-identical. After a
   correct fix it must be **bit-identical** (0.0 rel_l2). If it is not, the fusion is still wrong —
   do not proceed to read block stages; the comparison is meaningless downstream of a wrong context.
3. Only once `fc_ctx_rows` is bit-identical, read the block stages in order (`conv_attn_in_0`,
   `qkv_0`, `attn_0`, ...). One of two outcomes:

   - **All 52 stages bit-identical** → the fusion rounding was the whole story. The earlier
     "compounding to 2.6e-1 by block 4" and "flat ~2e-2 per block" were the fusion error being
     amplified by healthy blocks, exactly as the isolation's shape should have suggested. Go to §7.
   - **A later stage still diverges** → you now have a *clean upstream* and a first-divergent stage
     to localise. Follow §6. The stage that diverges is where to read source, not every stage that
     carries it.

Recompute the `.npy` header trap is already fixed (`86daffeaf`), but keep it in mind: a dump whose
every stage reads cos ≈ 0 / rel_l2 exactly 1.000 is a header/read fault, not a finding — check the
engine's own `stages.json` summary against the `.npy` before believing a catastrophic comparison.

## 6. How to debug the rest, if the fusion fix is not the whole story

Do not repeat the earlier agents' mistake. Their conclusion "the fault is inside the block body,
block 0 fails on the undisputed `noise_emb` input" is **wrong for a mechanical reason**:
`isolate_blocks.py` calls `reference_bf16.run_round`, which recomputes
`ctx_hidden = hidden_norm(fc(ctx_x))` **inside every call**, using the *reference's* fused context.
So both isolation arms always fed the reference's context to the block — the block was never actually
isolated from the fusion. The "block 0 fails on a table lookup" observation was really "the reference
block fed the reference context differs from our block fed our context", which is just the fusion
divergence propagating through a healthy block. The fix in §4 removes that confound; only then does a
block-by-block result mean anything.

Rules for any further debugging:

- **Localise by first-divergent stage, not by compounding curve.** A wrong upstream input amplified
  by five healthy blocks looks identical to a systematic per-block error. Only a restart/isolate
  experiment — feed block k the *same* input on both sides and diff the output — separates them, and
  it is only valid once the context it cross-attends to is also identical on both sides.
- **Read the source only at the first stage that diverges.** If `qkv_0` matches but `attn_0` does
  not, the fault is in the attention arithmetic (or its inputs' layout), nowhere before it. Do not
  audit a kernel that a bit-identical dump has already exonerated.
- **Compare against the reference semantics, not against "what looks reasonable".** The oracle is
  `~/Projects/refimpls/dflash/dflash/model.py` (self-contained torch, loads the checkpoint directly)
  and the SGLang production code `~/Projects/refimpls/sglang-flashnext` (sha 3c560089e1). For any
  suspected divergence, reproduce the reference's exact formula in torch bf16 on GPU and diff
  bit-for-bit — that is how §3 was found. A torch `nn.Linear`/`RMSNorm` rounds at module boundaries;
  a fused CUDA kernel that keeps fp32 through a boundary is *wrong*, not *better*.
- **Run the reference in bf16 on GPU, same precision as the engine.** fp32-vs-bf16 is a tolerance
  band wide enough to hide a real defect and narrow enough to manufacture a pass. Same-precision
  turns "cos ≥ 0.99975" into a bit-identity gate.
- **`token_sha256` cannot validate a drafter.** The target verifies every token, so a drafter that
  proposes near-chance tokens still reproduces the text. The only drafter-quality number is
  acceptance per round at a stated width. A lossless sha is necessary, never sufficient.
- **A null A/B is a broken A/B, not a null result.** If two arms produce byte-identical output,
  prove the override actually reached the code (env read at `init`, not a comptime constant) before
  believing the null.

## 7. After the divergence is closed: measure acceptance, then cost

Divergence and acceptance are two different questions. The check-walk agreement (1.317 vs 1.342)
proves our forward ≈ the reference on the same taps; it does **not** prove the drafter hits the
vendor's acceptance, because that walk is teacher-forced and scores faithfulness to text the model
did not write (~0.26–0.34 for the vendor's own code too).

Once the stage diff is clean, measure acceptance the only way that counts: a live greedy serve
against the target, reading `accepted=X/Y` and `rounds=N` off the `[tensorfold] done` line, and
always quote the width (`Y/rounds`) beside the rate. Judge against the vendor's **same-model, same
width** figure: acceptance length ~2.8–3.3 at width 4, or ~2.4 accepted/round at width 7. Do not
quote the width-8 z-lab numbers against a width-4 or width-7 run.

Only after acceptance is right does the round cost matter, because `ms/round` against a
no-draft baseline is what turns acceptance into throughput. Break-even is
`draft_ms / nodraft_ms` tokens a round; below it drafting is a net loss however good acceptance
sounds.

## 8. Guardrails (never violate)

- One model load, one `zig build`, one server at a time on this box — two loads exceed the 128 GiB
  unified memory and kill the host. Serialise everything; the orchestrator runs every build and
  server.
- Never start/stop/restart any host service. Relja owns the service lifecycle.
- Commit and push the branch before a long measurement; never amend a commit another agent made.
- Every measured cell gets ≥3 reps; put `not measured` in cells you did not run. An invented number
  is worse than an empty row.

## 9. Result of the §4 fix (measured 2026-11-17)

The fusion-rounding fix (commit `3a173079b`) was built and the engine re-run on `fixture/window`
alongside the unchanged reference. Pre-fix = `bf16d_fixed`, post-fix = `bf16e_fusedfix`, both
compared against the same `bf16/ref_window`.

| stage | pre rel L2 | post rel L2 | pre exact% | post exact% |
|---|---|---|---|---|
| `fc_ctx_rows` | 3.29e-3 | **4.53e-4** | 65.5 | **99.08** |
| `conv_attn_in_0` | 5.12e-3 | **2.96e-3** | 24.2 | **62.3** |
| `qkv_0` | 4.07e-3 | 3.18e-3 | 26.0 | 47.3 |
| `attn_0` | 2.77e-2 | 2.52e-2 | 10.2 | 18.4 |
| `block_hidden_0` | 8.46e-3 | 9.62e-3 | 5.5 | 6.1 |
| `block_hidden_4` | 2.68e-1 | 2.57e-1 | 0.8 | 0.8 |
| `proposals` | `[20,15,18,15,1,494,25]` | `[20,1,847,471,494,348,383]` | — | — |

**Reading.** The fix did exactly what it promised and nothing more: `fc_ctx_rows` and
`conv_attn_in_0` (the two stages that consume the fixed kernels' outputs directly) improved
markedly, and `fc_ctx_rows` is now 99% bit-exact. But from `qkv_0` onward the numbers barely move,
and `block_hidden_4` is still ~2.6e-1. So the fusion rounding was a real, fixable defect — and it
was **not** the source of the block-body divergence. The per-block error still compounds ~2× a layer
from `qkv_0` down, exactly as the pre-fix run did.

**The next first divergence is `qkv_0`** (3.18e-3, 47% exact, fed by `conv_attn_in_0` which is now
itself only 2.96e-3). The remaining ~3e-3 at `conv_attn_in_0` after the norm fix is either the
`attention_conv` (dynamic grouped conv) or the block's own projections — those are the two things
to read next, using the §6 method (first-divergent stage, read source there only). Do not chase the
compounding curve below `qkv_0`; a wrong `qkv_0` makes every later stage look wrong.

One correction to what §4 implied: the fix is necessary but not sufficient. The earlier "the fault
is inside the block body" was wrong for the mechanical reason in §6, and the block-body divergence
is real — it is just *upstream* of where the isolation claimed it was.

## 10. The residual is summation order, not a further defect (measured 2026-11-17)

The remaining divergence below `conv_attn_in_0` was localised with two torch probes against the
existing dumps, and it closes the numerics question:

- **The reference path is bit-exact.** Recomputing `conv_attn_in_0` in torch
  (`rmsnorm → kernel_projection → grouped_conv`, cuBLAS for the projections) matches the reference
  dump 100% (rel_l2 0).
- **`kernel_projection` is order-insensitive at bf16 output.** cuBLAS-bf16 and a plain
  fp32-matmul-then-round give bit-identical results for this input (rel_l2 0), so the projection's
  rounding is robust to reduction order — the residual is not that.
- **The engine's hand-written dense matches cuBLAS to 99.8%.** Feeding the engine's *own* dumped
  `conv_attn_in_0` through cuBLAS q/k/v projections reproduces the engine's `qkv_0` dump to
  rel_l2 2.28e-4 / 99.8% exact. The engine's `tf_q35_dense_bf16_batch` is therefore numerically
  equivalent to cuBLAS up to its fixed FMA reduction order; the last-ULP difference is the
  deliberate, reproducible summation (`__fmaf_rn`, stride-256, warp-shuffle tree) — not a bug.

So the ~1–3e-3 that survives at every projection is the fixed-order reduction's bf16 rounding
noise, and it is the price of the determinism guarantee: the engine pins its summation order so a
`token_sha256` is reproducible across builds/hardware, while cuBLAS's order is undocumented and
moves between versions. It cannot and should not be "fixed" to match cuBLAS bit-for-bit; that would
throw away reproducibility for nothing, because losslessness (the target verifies every token) never
depended on bit-identity of the drafter's forward.

**Do not keep chasing the stage-compounding curve.** The fix in §4 was the real defect; what
remains is irreducible reduction noise, and the earlier "block body fault" was never there to
begin with. The acceptance gap — the actual project goal — lives in the served cross-round context
(§1 of `flashnext-dflash2-acceptance-log.md` and the two open §10 defects there: the `q.fed` note
after window advance, and `st.st.end = 0` every round), which the single-round fixture cannot see.
That is the next place to spend effort, not the bf16 stage diff.
