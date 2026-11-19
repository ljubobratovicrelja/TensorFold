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
