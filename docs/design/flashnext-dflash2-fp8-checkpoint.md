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
(`tools/dflash2/harness/plot_stages.py CAPTURE --baseline BASE`, committed as `fb871c4d5`).

**The cost: proposals move.** 48 of 112 proposal slots are identical between arms; 64 differ. Per rep
the identical count is 1, 1, 4, 7, 1, 3, 1, 0, 3, 3, 2, 5, 4, 6, 3, 4 — one rep is fully identical,
one fully differs. Hidden-value drift: median relative L2 7.4e-3, p95 4.9e-2, worst 1.4e-1 on the
last block. This is expected: §5 of the plan records that the selector amplifies sub-bf16 noise into
a different candidate set, which is exactly why bit-exactness cannot be the gate for a precision
change.

## 4. The served A/B, run 2026-11-18

Four arms, one server at a time on `:8099`, the same 16 GSM8K prompts and seeds, numbers taken from
the server's own `done chatcmpl` lines:

| arm | tok/s | accept len | round ms |
|---|---:|---:|---:|
| AR (`--no-drafts`) | 25.0 | 1.004 | 40.2 |
| MTP (no `--drafter`) | 53.0 | 3.505 | 67.1 |
| DFlash2 bf16 | 48.1 | 3.387 | 69.6 |
| DFlash2 fp8 | 49.4 | 3.322 | 66.8 |

**Losslessness holds: all four arms produced identical completion hashes on 16 of 16 prompts.** The
fp8 checkpoint changes what the drafter proposes and nothing about what the target emits.

Paired on the same prompts (`accept_ab.py --compare`):

- **acceptance −1.94%**, 95% CI **−3.54% to −0.33%** — a real loss, not noise, and small. The
  selector is doing exactly what §3 describes: a 2.646% weight error moves 64 of 112 slots, and the
  accept length follows.
- **throughput +1.31 tok/s**, CI **+0.61 to +2.01** — real, and bigger than the acceptance loss costs,
  because the round is 2.8 ms cheaper.
- The paired per-prompt sd was 0.111 tok/round, so 16 prompts *did* resolve a ~2% effect here
  (n ≥ 11 was needed at that spread). That is luck, not a licence: at the 0.21 spread assumed when
  this harness was written, the same 16 prompts would have resolved nothing.

**Verdict: fp8 is a net gain on this box and this prompt set, and it is not free.** The honest summary
is "−1.9% accept for +2.7% throughput, both statistically real", not "no cost".

## 5. What is still open

**Per-slot acceptance** (the vendor's width-8 shape is 0.740 / 0.662 / 0.592 / 0.621 / 0.389 / 0.429 /
0.333) is not measured — the served log reports totals only, so this needs a tap-level replay rather
than the `done` line. The aggregate says acceptance falls ~2%; per-slot data would say *where*, and
that is what would tell us whether a bf16 rescue of one group (say `down_proj`) buys it back cheaply.

**One prompt set.** 16 GSM8K prompts is one distribution. Coding and long-context streams are the
other two shapes this box serves, and the survey's external evidence was for the 27B drafter, not
this one.

**Where a collapse would show:** accept length 3.34 → ≤2.5, or degenerate proposals (repeats, all
seven identical). Neither appeared, on fixture or served.

## 6. Deployment, measured on the running build (2026-11-18)

`dflash2` merged fast-forward into `running` (72 commits, pushed), rebuilt in
`~/Projects/tf-wt/running`, and the fp8 artifact staged at
`~/.cache/tensorfold/dflash2-fp8` with all three sha256 verified against its own `SHA256SUMS`.

**Blocker: any drafter plus `--spill-gib` dies at startup.**

```
info: flashnext: --drafter ...: DFlash2, taps { 5, 15, 25, 35, 45 }, 7 drafts a round
info: cuda_window_graph.HeadKey graphs: 1 captured, 0 updated, 0 replayed, 1 held
error: ProbeUnmarked
```

The same unit without `--drafter` starts in 72 s, so it is not a merge regression, and it reproduces
with `--learn-daily-gib` and `--prompt-cache-disk` removed — `--spill-gib` alone is enough.
`flashnext_disk.zig` calls `learned.probe`, which requests marks at 4096 and `toks.len - 8` and
requires both to fire (`learned.zig`: `p.marks != marks.len → ProbeUnmarked`); a speculative round
commits past a mark position, so `marked()` never runs for it. **Hypothesis, not yet confirmed by a
run with drafting forced off inside the probe** — that is the one-line test.

**And on the production config it loses to MTP.** Same binary, same four short coding prompts,
`--context 262144 --parallel 4 --kv-cache fp8 --vision --thinking`:

| arm | decode tok/s | tokens/round | drafts offered/round | accept rate |
| --- | --- | --- | --- | --- |
| MTP (no drafter) | **52.5** | **3.764** | 6.56 | 0.669 |
| DFlash2 fp8 | 39.2 | 2.430 | 3.19 | 0.669 |

At `--context 32768` the fp8 arm measured 38.9 tok/s, so context length is not the cause. The
window-cost curve is identical in both runs (37.19 ms at one row, 100.90 at 32), so it is not a
depth-rule reaction to a different cost estimate either. What differs is the traffic shape: these
are short coding answers, where MTP's head is strong — consistent with MTP winning 12/16 GSM8K
prompts, and worse than the 16-prompt GSM8K pair (49.4 vs 53.0). **n = 4 prompts**, so this is
directional, not a verdict; it is enough to say fp8 is not the default on this config.

The unit therefore ships with MTP and carries the fp8 line commented, with both findings inline.

