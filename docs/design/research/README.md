# Research notes on making this model faster

Six read-only research passes run on 2026-11-19, filed here rather than kept outside the repo. Every
number in them is attributed to a source; the compilation is ours, the measurements are not. Nothing
here has been reproduced on this box unless a document in `docs/design/` says so.

| file | what it covers |
|---|---|
| `moe-weight-manip-research.md` | non-training weight manipulations: expert merging, expert+router permutation, pruning, rotations, packing |
| `moe_decode_runtime_kernels_report.md` | runtime and kernels: W4A4 vs W4A16 at batch 1, dedup vs per-slot, persistent/fused kernels, graphs, expert caching, overlap |
| `approx-exec-and-specdec-levers.md` | approximate execution: route tail skip, layer skipping, KV eviction, sparse attention, PLE handling, draft-length theory |
| `precision_allocation_report.md` | precision on an already-4-bit QAD checkpoint: head, hyper-connections, KV, n-gram table, re-quantisation |
| `dgx-spark-qwen38-flash-next-report.md` | how other people run this model on DGX Spark: the MiaAI-Lab recipe, NVIDIA forum threads, 13 configurations and their knobs |
| `flashnext-knob-audit.md` | a source audit of this engine: every runtime switch, unused kernel and tuning constant, with file:line evidence |

## What each pass concludes, in one line

- **Weight manipulation**: expert merging costs real points at top-10-of-512 (HC-SMoE −3.4 pts at −25 %
  experts on Mixtral); permutation is semantics-exact but has no measured single-device gain; pruning is
  the only structural lever with published near-lossless results, and it must be calibrated on your own
  traffic.
- **Runtime**: W4A4 loses to bf16 at concurrency 1 on fixed per-call overhead, and the FP4 MMA tile is
  ~6 % utilised at M ≤ 8 — the FP4 tensor-core path is not the lever it looks like. Deduplicating expert
  reads is confirmed net-negative at small pass sizes twice over. The lever is weight-major kernels,
  CUDA graphs and glue fusion: glue kernels can cost as much as the MoE GEMM itself.
- **Approximate execution**: route tail skip is measured as pure downside when experts are resident;
  layer skipping is slower than autoregressive without layer-dropout training; KV eviction is fragile.
  The free knob is draft length — the optimum at acceptance ≈ 0.55 is likely shorter than what we run.
- **Precision**: the head and the hyper-connections are the two tensors left in bf16 on the hot path;
  everything below 4 bits on the experts has no kernel on this part and no published win. The LM-head
  citation that most people use for "FP8 is lossless" explicitly excludes `lm_head`.
- **Spark configurations**: the biggest single reported win across community recipes is not a kernel at
  all — it is keeping the 47.7 GiB n-gram table out of the working set, then MTP draft tuning.
- **Knob audit**: decode rounds are not CUDA-graphed at TP=1 (the predicate requires a collective);
  `nvfp4_experts_tile` is built and never selected; a per-slot expert plan is a ~20-line change from the
  identity `Plan`; the expert path is fully resident with no cache or paging on CUDA.
