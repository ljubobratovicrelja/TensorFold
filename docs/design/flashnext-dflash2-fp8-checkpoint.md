# The fp8 weight arm — a second drafter checkpoint, and what swapping to it costs

2026-11-18. The drafter's seven decoder linears now have a weight-only FP8 build (e4m3, 128×128
block scales, bf16 activations, fp32 accumulate — **W8A16**, never W8A8). It ships as a separate
checkpoint next to the published bf16 one, and the engine picks between them with a flag. This
document is the artifact, the switch, and the measured difference; it is not a claim that fp8 is
free.

## 1. The artifact

`checkpoints/dflash2-fp8/` — built once, offline, no calibration corpus.

| part | bytes | share |
|---|---:|---:|
| e4m3 codes (35 matrices) | 425,984,000 | 54.6% |
| f32 block scales | 104,000 | 0.01% |
| bf16 passthrough (46 tensors) | 354,163,200 | 45.4% |
| safetensors header | 12,800 | 0.002% |
| **total** | **780,264,000** | — |

1.55× smaller than the 1,206,140,008 B bf16 source. `model.safetensors` sha256 `d18d9ec7…c27a6c`.

**What is quantized:** `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `self_attn.q_proj`,
`self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj` — 7 groups × 5 layers.
**What is not, in either arm:** the context fuse `fc`, the two convs per layer, every norm, and the
selector (pred/succ codebooks + `hidden_projection`). A block-FP8 GEMM is only paid for where the
bytes dominate the stage; `fc` is one 65.5 MB read, the convs are latency-bound, and the selector
**is** the acceptance mechanism — quantizing it moves the thing being measured.

**The scales.** Per 128×128 block, `amax = max|w|`, stored `weight_scale_inv = amax/448` as f32, and
dequant is `w ≈ code * weight_scale_inv`. The reciprocal convention (`448/amax` with a divide) is the
same mathematics with the same codes; what is shipped is the **multiply** form, because that is what
`cuda/fp8.zig`, `qmmf`'s `.fp8g` accumulate (`acc += d * s`) and `contract_qwen_rope_yarn.zig`
(`scale ≈ amax/448`) all implement.

**Reproducible:** `tools/dflash2/quantize/` (`dflash_fp8.py`, `quantize_drafter.py`, `verify.py`),
pure numpy + stdlib. Re-running produces the identical sha256. `verify.py` checks codes and scales,
no NaN code, finite positive scales, byte-identical passthrough, a key-preserving `config.json`,
manifest hashes, and round-trip idempotence (re-quantizing the dequantized weights reproduces the
stored codes *and* scales bit for bit).

    SNAP=~/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6
    python3 tools/dflash2/quantize/quantize_drafter.py --self-test
    python3 tools/dflash2/quantize/quantize_drafter.py --src "$SNAP" --out checkpoints/dflash2-fp8
    python3 tools/dflash2/quantize/verify.py --fp8 checkpoints/dflash2-fp8 --src "$SNAP"

The 780 MB blob is **gitignored**; `manifest.json`, `config.json`, `SHA256SUMS` and the tooling are
committed. There is no git-lfs on this box, and a 780 MB blob in history is not reversible — the
manifest plus a deterministic tool is the pin, not the bytes.

## 2. The switch

    tensorfold draft-check ... --drafter-precision bf16|fp8
    tensorfold serve ...        --drafter-precision bf16|fp8

Omitted, the checkpoint's own tensors decide: `config.json`'s `precision`, else the presence of
`*_scale_inv` tensors. A checkpoint whose declared precision and contents disagree is refused at load
(`DrafterPrecisionMismatch`, naming both); families whose drafter has no quantized arm refuse `.fp8`
by name rather than reading e4m3 codes as bf16 rows. Both arms produce bf16 activations, so every
downstream stage is untouched — the flag changes what is read, not what is computed.

**Revert is a flag.** The bf16 checkpoint is never rewritten; dropping `--drafter-precision fp8` (or
pointing `--drafter` back at the published checkpoint) restores the previous behaviour exactly, and
the shipped default is `null`, i.e. the checkpoint decides.

## 3. What it measures

`fixture/sequence`, **same binary, same session**, 15 served reps each, medians.

| stage | bf16 | fp8 | delta | |
|---|---:|---:|---:|---|
| mlp (5) | 2.757 | 1.544 | −1.213 | −44.0% |
| qkv (5) | 0.834 | 0.532 | −0.302 | −36.3% |
| attn_proj (5) | 0.546 | 0.379 | −0.167 | −30.6% |
| attention (5) | 4.385 | 3.380 | −1.005 | −22.9% |
| `fc_ctx` (unchanged) | 0.709 | 0.697 | −0.012 | −1.7% |
| head `logits_top16_ids` (unchanged) | 6.986 | 6.981 | −0.005 | −0.1% |
| `proposals` (unchanged) | 1.445 | 1.465 | +0.020 | +1.4% |
| **stage sum** | **18.600** | **15.908** | **−2.692** | **−14.5%** |
| **served round** | **18.731** | **16.019** | **−2.712** | **−14.5%** |
| cold round (rep 0) | 284.167 | 216.039 | −68.128 | −24.0% |

**Only 1.69 ms of the 2.71 is weight bytes** (mlp + qkv + attn_proj). The other 1.00 ms is attention,
which the fp8 path does not touch. The most likely mechanism is cross-stage clock coupling — bf16 mlp
moves 2× the bytes, the SoC clocks down, and tiny-M attention follows — and the evidence fits without
proving it: across two runs in opposite order, fp8 attention sat at **0.675–0.681 ms/layer** both
times, while bf16 attention read **0.87 then 1.10**. A served stream gets this benefit; a
drafter-only profile taken alone would not. It is recorded as a hypothesis, not as a kernel win.

The figure: `docs/design/profiles/2026-11-18-fp8-vs-bf16-served.png`
(`tools/dflash2/plot_stages.py CAPTURE --baseline BASE`, committed as `fb871c4d5`).

**The cost: proposals move.** 48 of 112 proposal slots are identical between arms; 64 differ. Per rep
the identical count is 1, 1, 4, 7, 1, 3, 1, 0, 3, 3, 2, 5, 4, 6, 3, 4 — one rep is fully identical,
one fully differs. Hidden-value drift: median relative L2 7.4e-3, p95 4.9e-2, worst 1.4e-1 on the
last block. This is expected: §5 of the plan records that the selector amplifies sub-bf16 noise into
a different candidate set, which is exactly why bit-exactness cannot be the gate for a precision
change.

## 4. What is still open, and what would close it

**Acceptance is not measured yet.** Nothing here says fp8 drafts as well as bf16 through a served
stream; it says the weights' error is 2.6% relL2 and the round is 2.7 ms shorter. The gate is
per-prompt accept length over many prompts, and the instrument is `tools/dflash2/accept_ab.py`
(paired A/B on the served path, verdict refused when the confidence interval is wider than the effect
asked about).

Sizing matters and is the reason this is not a 16-prompt question: at a per-prompt accept-length
spread of ~0.21, resolving a 2% effect needs ~39 paired prompts, and resolving 1% needs ~150.

**Per-slot acceptance** (the vendor's width-8 shape is 0.740 / 0.662 / 0.592 / 0.621 / 0.389 / 0.429 /
0.333) needs a tap-level replay rather than the server's `done` line, because the served log reports
totals only. That is a second instrument.

**Where a collapse would show:** accept length 3.34 → ≤2.5, or degenerate proposals (repeats, all
seven identical). Neither appeared in 16 reps of the fixture, which is 16 rounds of evidence, not a
guarantee.
