# FP8 allocation on the dense path: head first, then the hyper-connections

Branch `fp8-alloc`, cut from `running`. Two experiments, run strictly one after the other, each judged
against an autoregressive baseline measured on the same box in the same session.

## Why these two tensors

The verify pass is bandwidth-bound and its stages were measured against the box's sustainable read
rate (~234 GB/s, `docs/design/flashnext-layer-time.md`):

| stage | bytes per round | measured | achieved |
|---|---:|---:|---:|
| `head` (`lm_head`, bf16) | 1.27 GB | 5.29 ms | 240 GB/s — at the wall |
| `conv_attn_in` + `conv_mlp_in` (four bf16 hyper-connection mixes) | 1.26 GB | 11.5 ms | 73–131 GB/s |
| `mlp` (NVFP4 experts, top-11 of 512) | 2.6–5.5 GB | 15–32 ms | 170–185 GB/s |

The experts are already 4-bit and quantisation-aware distilled; nothing below 4 bits is available on this
part, and the published attempts cost 4–11 points of top-1 agreement. The dense path is where bytes can
still be removed: the head is the one stage running at the wall, so halving its bytes halves its time
almost exactly, and the hyper-connections are read every round in bf16 and are the least efficient
stages in the pass.

## The two arms

1. **head_fp8** — `lm_head` (248320 × 2560 bf16, 1.27 GB) stored as FP8 e4m3 with per-vocab-row scales,
   ranked in FP8 and the top-64 survivors **rescored in bf16** through the exact dequantised rows before
   top-k selection. Expected ~2.6 ms of a 5.3 ms stage, ~5 % of the pass.
2. **hc_fp8** — the four hyper-connection mixes per layer (bf16, 1.26 GB per round) at FP8 e4m3,
   per-output-row scales, fp32 accumulate. Expected 2.5–4.3 ms of an 11.5 ms pair of stages.
   Only started after arm 1 is measured and committed.

Both are RTN: round-to-nearest in the weight, no calibration, no training, the engine dequantises at
load. Anything smarter is only justified if RTN fails the gate below.

## Fidelity gates

Run in this order, each cheap before the next:

1. **Weight reconstruction** — per-tensor relative L2 of `dequant(quant(w)) - w`, and the max relative
   error per row. This is a sanity check, not a verdict.
2. **Top-1 agreement against the bf16 arm** — greedy decode on the 16 public GSM8K test prompts
   (`tools/dflash2/harness/prompts/gsm8k_16.json`), token-by-token agreement of the emitted text. The
   harness records a sha256 per output, so identical answers are visible without a rerun.
3. **Graded accuracy** — `\boxed{}` exact match on the 8 prompts that have golds in
   `prompts/gsm8k_gold.json`.
4. **Throughput** — paired per-prompt `tok/s`, mean difference with a 95 % t-interval, same session,
   same binary except the arm.

An arm ships only if it gains throughput with a CI excluding zero **and** loses no graded answer. A
throughput null with identical outputs is a failed experiment, not a win.

## The harness contract

* `ar_run.sh <arm> <out.json> [flags]` — starts this worktree's binary on `:8099` with `--no-drafts`
  and no `--drafter`, waits for `/v1/models`, runs the bench, stops the server by pidfile.
* `ar_bench.py` — greedy (`temperature 0`), 512 new tokens, `thinking` off, no drafting of any kind:
  pure autoregressive decode, which is the only thing a weight change should be judged on.
* Artifact: one JSON per arm, `config` + `aggregate` + `rows[]`, where each row carries tokens, wall,
  `tok_s`, output sha256, the extracted answer, the gold and the server's own `done chatcmpl` line.
  `ar_bench.py --compare a.json b.json` prints the paired table and the verdict.
* Results live under `runs/ar/<arm>.json` and are committed; plots are written later from those files
  and must not re-derive numbers by hand.

## Order of work

1. Baseline arm on the unmodified `running` source — `runs/ar/baseline.json`.
2. `head_fp8`: implement quantiser + loader + ranked-and-rescored head, build, smoke test, measure,
   commit.
3. `hc_fp8` on top of a passing `head_fp8`: same pipeline.
4. Consolidated plot: per-prompt throughput overlay and a stage table, from the committed JSONs.

## Results

All autoregressive numbers below are the same 16 public GSM8K prompts, 1024 new tokens, thinking off,
one `:8099` server per arm, paired per prompt against a control run on the same box.

### Arm 1 — FP8 e4m3 `lm_head`, kept

| | baseline | head_fp8 |
|---|---:|---:|
| tok/s mean | 24.74 | 26.37 |
| head stage | 5.28 ms | 3.12 ms |
| outputs | — | 16/16 byte-identical |
| graded | 8/8 | 8/8 |

Paired difference **+1.63 tok/s [+1.47, +1.79] = +6.59% [+5.96, +7.22]**. The stage halved as the bytes
halved; the residual 0.48 ms over the ideal 2.64 ms is the rank, gather, top-64 rescore and cast.
Quantisation: per-row relative L2 2.64% (p99 2.76%), 0.057% saturated codes, no NaN. Artifact pinned in
`runs/ar/head_fp8.artifact.json`, blob gitignored.

### Arm 2 — FP8 e4m3 hyper-connections, rejected

Same-session pair: 24.57 → 24.42 tok/s, −0.60% [−0.78, −0.43]. A drift control — the *unmodified*
source measured twice in different sessions — reads −0.68% with 16/16 identical outputs, so the arm's
real effect is **0.0 ± 0.2%**. Matched stage capture at rows=1: `conv_attn_in` 4.66 → 4.86,
`conv_mlp_in` 4.95 → 4.78, net +0.03 ms. Those stages are not bandwidth-bound at this row count, so
removing 42% of their bytes removed nothing. It also changes 12 of 16 outputs: fidelity spent for
nothing. Kept behind `TF_FLASHNEXT_HC_FP8`, off by default.

**The rule this pair establishes: quantisation pays only where a stage is already at the wall.** The
head was (240 GB/s of ~234 achievable); the hyper-connection stages were not (73–131 GB/s, which is a
latency number, not a traffic number).

### The head arm with the DFlash2 drafter

The point of the exercise: the drafter scores its candidates through the *target's* head, so the head is
the one stage both models pay for. With `TF_FLASHNEXT_LMHEAD_FP8` plus the affine4 body, split-KV
attention and block-FP8 drafter head:

| | t1_ar | t1_drafter |
|---|---:|---:|
| tok/s mean | 25.49 | **62.82** |
| accept / draft | — | 0.642 (5,936/9,245) |
| outputs | — | 16/16 byte-identical to AR |
| graded | 8/8 | 8/8 |

Paired **+37.33 tok/s [+33.74, +40.92] = +146% [+132, +161]**. This is with thinking off, so the answers
are short and the round count low; it is not the production number, but it is the clean statement that
the head change composes with the drafter instead of competing with it.

Two harness facts discovered on the way, both now encoded in `ar_run.sh`:

- `--no-drafts` disables an external `--drafter` as well (`cuda_drafter_open.zig`'s predicate), so a
  drafter arm must leave drafting on (`AR_DRAFTS=1`).
- The engine proposes **no drafts at all for greedy non-thinking requests** (`accepted=0/0`, one token
  per round); speculation engages only at temperature > 0. Drafter arms are therefore sampled
  (`--temperature 1.0 --top-p 0.95 --top-k 20`, seeded per prompt), and the AR control is run at the
  same sampling so the pair stays comparable.
