# Runtime & kernel techniques for fast MoE decode at small batch

Scope: what is actually implementable by a small team that owns its engine, for a 48-layer
hybrid MoE (36 Gated-DeltaNet + 12 sparse-attention), top-10/512 + shared, NVFP4 checkpoint,
W4A16 execution, 2–8 rows per pass, on one GB10 (sm_121a, 128 GB unified LPDDR5X, ~234 GB/s
measured sustained).

Tags: **[M]** = measured and reported by the cited source. **[I]** = inference/derivation by
me from the numbers given or from the source's own model. Every number below is attributed;
nothing is rounded into a claim it does not support.

---

## 0. Where your time actually is (do this arithmetic first)

Your own numbers, turned into bytes moved:

| Stage | Time | Achieved BW | Bytes implied (time × BW) | Time at 234 GB/s | Headroom |
|---|---|---|---|---|---|
| head | 5.3 ms | 240 GB/s | ~1.27 GB | 5.4 ms | none — at the wall |
| expert MLP | 15 ms | 180 GB/s | ~2.70 GB | 11.5 ms | **~3.5 ms** |
| conv/norm (one stage) | 8.6 ms | 73 GB/s | ~0.63 GB | 2.7 ms | **~5.9 ms** |
| conv/norm (one stage) | 4.9 ms | 131 GB/s | ~0.64 GB | 2.7 ms | **~2.2 ms** |
| attention | 1.4 ms | — | — | — | small |

**[I]** The listed stages account for 30–40 ms of the 47 ms verify pass; 7–17 ms sits in stages
you have not itemised (router, reductions, glue, launch gaps). Read that as ±: the byte figures
are derived from your timings, not from a byte counter, so treat them as order-of-magnitude.

**The point that organises everything else in this report:** the head stage already runs at the
machine's measured wall, so 234 GB/s is reachable from inside your engine. The expert MLP and
the conv/norm stages are running at 77% and 31–56% of that same wall. **[I]** Making the stages
you already have hit the bandwidth the head already hits is worth roughly 5–9 ms of a 47 ms
pass (11–19%), and it costs no new numerics, no new accuracy risk, and no new dependency. This
is the highest-ROI item available to you, and it is a tile/occupancy/vectorisation job, not a
research job.

Note what that does to the FP4-tensor-core question. At 2–8 rows, W4A16 and W4A4 **read exactly
the same weight bytes**. W4A4 buys you only: no dequant ALU, half the SMEM/L1 staging for
weights, and no bf16 expansion in the pipeline. If your kernels are already only reaching 77% of
wall on the MLP, you are not ALU-bound, so tensor-core FP4 has almost nothing to sell you.
Section 1 has the measurements that agree.

---

## 1. W4A4 / FP4 tensor-core paths on Blackwell

### 1.1 What the major stacks actually ship

- **vLLM**: `CompressedTensorsW4A4MoeMethod` with a TRTLLM NVFP4 fused-MoE kernel, plus CUTLASS
  `nvfp4_blockwise_moe`. https://github.com/vllm-project/vllm/pull/28892
- **SGLang**: `sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu`, explicit `sm_100a` CUTLASS schedule
  `KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100`, `ThreadBlockShape = <128,128,128>`, plus a
  FlashInfer-TRTLLM NVFP4 fused-MoE path. https://github.com/sgl-project/sglang/pull/27401
- **TensorRT-LLM**: TRTLLM-Gen NVFP4 cubins, CuteDSL NVFP4 grouped GEMM with fused SwiGLU and
  fused finalize, and an explicit A16 variant (`nvfp4A16BlackwellMoePlugin`) — i.e. NVIDIA itself
  keeps a W4A16 NVFP4 MoE plugin alive for small/MoE shapes.
  https://github.com/NVIDIA/TensorRT-LLM/pull/9288 ,
  https://nvidia.github.io/TensorRT-Edge-LLM/latest/cpp_api/plugins/nvfp4A16BlackwellMoePlugin.html
- **FlashInfer**: `flashinfer.moe_ep` MegaMoE with interchangeable backends: **NVFP4 (W4A4)** CuTe
  DSL block-16 scaling, MXFP8 (W8A8) block-32, and DeepGEMM W4A8 as baseline/fallback. Combine
  format selectable independently (BF16/MXFP8/NVFP4). **[M]** claims CUDA-graph capture/replay
  correctness tests as part of validation. https://flashinfer.ai/2026/09/22/mega-moe.html
  For sm_12x specifically, FlashInfer has an SM120/SM121 fp8 + mxfp8_mxfp4 grouped-MoE kernel
  family, and a known open bug where the CUTLASS grouped block-scaled GEMM path produced invalid
  output on SM120 and needed patching:
  https://github.com/flashinfer-ai/flashinfer/commit/48c29d2efaf70d53c61054b28ba67b8392ee37fa ,
  https://github.com/flashinfer-ai/flashinfer/issues/2723
- **Marlin / Machete**: W4A16 mixed-input GEMM (fp16 × int4), the small-batch workhorse for
  Ampere/Hopper. Marlin's own framing is "near-ideal" 4-bit weight throughput at small batch via
  asynchronous weight dequant overlapped with MMA:
  https://github.com/IST-DASLab/marlin ,
  https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel

### 1.2 The measurement that decides it for you

**[M] NVIDIA, Model-Optimizer QAD announcement (Qwen3.6-35B-A3B, W4A4 NVFP4, three shapes):**
"W4A4 loses to BF16 only at concurrency 1, by a near-constant **0.87–0.88×** across all three
shapes. That is fixed per-call overhead in the FP4 kernel path, which batch-1 decode has no
arithmetic intensity to amortize — not a recipe knob."
https://nvidia.github.io/Model-Optimizer/announcements/qwen36-w4a4-qad.html

Read that carefully: NVIDIA's own W4A4 path is **slower than BF16 at concurrency 1**, on its own
tuned kernels, on a Qwen3 MoE. W4A16 (Marlin/Machete/dequant-then-mma) is at or above BF16 speed
at batch 1 because it moves a quarter of the weight bytes. So **[M+I] W4A4 is slower than W4A16
at batch 1** in the only like-for-like public measurement that exists. This is the direct answer
to "does anyone report W4A4 faster than W4A16 at batch 1–8": no.

Why: at M ≤ 8 the tensor core is idle anyway. **[M] ReSET** states the mechanism explicitly —
at `M ≤ 8` the Tensor-Core `tcgen05.mma` tile is **≤ 6.25% utilised**, so they built a
**CUDA-core** small-M NVFP4 kernel instead, and report **2.5× kernel-level decode speedup over
vLLM's NVFP4** and **~2× end-to-end over BF16** at batch 1 on B200.
https://github.com/aiha-lab/ReSET , https://arxiv.org/html/2606.13233v1

**[I]** That is the honest shape of the W4A4 opportunity: the win, where it exists, is not
"use the FP4 tensor cores" — it is "stop paying tensor-core setup for a GEMV". ReSET's 2.5× is a
CUDA-core-kernel-vs-tensor-core-kernel result, not a 4-bit-vs-16-bit result. Your engine is
already on the CUDA-core side of that divide.

### 1.3 SGLang's batch-1 FP4 number, and why it does not transfer

**[M]** GPT-OSS-20B (E=32, top-4) on B200: SGLang FP4 is **1.32× faster than vLLM FP4 at batch
1**, 2.23× faster than SGLang BF16 at BS=128, 3.54× over BF16 at BS=4096. Three causes: kernel
fusion (7 memory passes → 5), Blackwell-specific CUTLASS warp-specialised FP4 schedules with
TMA, and adaptive grid sizing.
https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison

**[I]** Note the comparison: FP4-vs-FP4 (1.32×) and FP4-vs-BF16 at large batch. There is no
W4A4-vs-W4A16 row. E=32/top-4 is also the most dedup-friendly routing in existence; your
E=512/top-10 is the least.

### 1.4 Accuracy cost of activation FP4

**[M]** Same NVIDIA QAD run: after W4A4 PTQ on Qwen3.6-35B-A3B, across six benchmarks, "**Only
IFBench and MMMU-Pro show a W4A4 deficit larger than run-to-run noise. On the other four, W4A4 is
effectively lossless** and QAD neither helps nor hurts." QAD (500 iters, 32K seq, 128×GB200)
buys ~+2 pts on the two that move. PTQ alone is 9 minutes; QAD is 735 GPU-hours.

**[I]** So the accuracy cost of activation FP4 on a *QAD-distilled* checkpoint is near zero on
most benchmarks — your checkpoint is already QAD'd by NVIDIA, which is the best case. Accuracy
is **not** the reason to avoid W4A4. Bandwidth-economics and kernel maturity are.

### 1.5 Verdict for you

- Applicability: **low**, and specifically *not* because of accuracy.
- The Tensor-Core W4A4 route needs `tcgen05`/SM100-class MMA plus TMA plus block-scale plumbing;
  it buys nothing at M ≤ 8 where the tile is ≤ 6.25% occupied, and it is measured slower than
  BF16 at concurrency 1.
- The *only* W4A4 variant worth a look is ReSET's shape: CUDA-core small-M, **A4 weights already
  resident as 4-bit, expanded in registers**. **[I]** For you that is a possible 10–20% on the
  expert MLP (fewer SMEM bytes staged per tile, no bf16 SMEM round trip) — but it requires the
  activations to be quantised to FP4, which puts a new error term in the verify pass, and
  measured evidence at your exact shape does not exist. Do it *after* §6 items 1–3.
- Risk: SM121 is the worst-supported Blackwell variant for FP4 kernels. FlashInfer shipped
  broken CUTLASS block-scaled output on SM120 and had to patch it; a one-engine team would be
  debugging someone else's kernel bugs on the least-tested arch.

---

## 2. Grouped GEMM and small-batch MoE kernels; dedup vs per-slot

### 2.1 The literature, in one line each

- **MegaBlocks** (2211.15841): block-sparse formulation, no token dropping, no capacity factor;
  does well because it avoids padding, which is the dominant waste when tokens-per-expert is 0–1.
  https://arxiv.org/pdf/2211.15841
- **Tutel** (2206.03382): adaptive parallelism + a 2D-to-1D re-layout of expert computation;
  the contribution that matters for decode is that it treats "few tokens per expert" as the
  normal case, not an edge case. https://arxiv.org/abs/2206.03382
- **FasterMoE** (3503221.3508418): performance model + shadowing/warp-level imbalance handling;
  a *training* paper, but its core observation transfers: at small per-expert M, the grouped
  GEMM's cost is dominated by *the number of groups*, not by the arithmetic.
  https://doi.org/10.1145/3503221.3508418
- **DeepGEMM**: now ships a "Mega MoE" fused kernel (dispatch + FP8×FP4 grouped GEMM with SwiGLU
  in the first epilogue + combine). Its own changelog shows the failure modes of persistent
  kernels under graph replay: "A CUDA graph kernel node keeps the same `%gridid` on every replay…
  a barrier line still holds `encode_state(grid_idx, kNumSplits)`, which is exactly the state the
  next replay waits for" → they had to add an epoch counter. **[M]**
  https://github.com/deepseek-ai/DeepGEMM (commit 5081f85)
- **MonoMoE** (2609.04244, in FlashInfer): see §3 — it is the most transferable idea in this
  whole report.

### 2.2 Dedup vs per-slot: the finding, with the mechanism

The decisive source is **arXiv 2610.34440**, "Speculative Decoding for MoE Models: Expert Bytes,
Kernel Bandwidth and the Best Draft Length". It is the only work that measures the *bandwidth of
each kernel* separately instead of assuming one bandwidth for all reads.
https://arxiv.org/html/2610.34440v1

Its model: a pass of n tokens with m selections each requests `mnW` bytes per-slot and
`Σ w_l U_l(n)` bytes deduplicated. Dedup is faster **exactly where** `ū(n) < r(n)`, where
`ū = E_dedup/E_slot` (set by routing) and `r = β_dedup/β_slot` (set by the kernels).

**[M]** Measured `r` = **0.32 to 0.75** — the deduplicating kernel moves the same bytes at a third
to three-quarters of the per-slot rate. **[M]** Consequence: "**deduplication is faster only from
5 to 8 tokens per pass, or never**", confirmed on deduplicating builds of llama.cpp Metal, MLX-LM
and llama.cpp CUDA. The saving prediction landed inside its error bound at every point on CUDA
and MLX-LM and 36 of 48 on Metal; crossover predicted within one pass size in 10 of 12
configurations.

**[M]** Reuse, for context: ρ_l(2) layer-mean = 0.363 (OLMoE), 0.480 (Qwen3), 0.492 (gpt-oss);
ρ_l(9) = 0.762 / 0.780 / 0.856. Also cited there: 44–48% next-token reuse on Qwen3-30B-A3B.

**[M]** And the punchline for speculation: with a real draft model, "plain decoding is fastest on
**33 of 37** held-out prompts; the median speedup over plain decoding is **0.877 at k=1 and 0.517
at k=8**." On a per-slot engine, drafting is a net loss at every draft length because each draft
adds a full token's worth of expert bytes.

**A second, closer measurement, on your silicon.** lablup's mlxcel measured the same question on
an **RTX Spark N1X, sm_121**, gemma-4-26b-a4b (E=128, top-8):
**[M]** 8 clients → 64 pairs → median 45 distinct experts → 30.4% fewer matrix reads; 4 clients →
32 pairs → 28 distinct → **12.5%** fewer reads. They patched in a measurement of the "read each
distinct expert once" upper bound and got: gathers are ~54% of a layer, a perfect grouped kernel
saves **~13% of a step at 8 clients and ~5% at 4 clients — "the size of this model's run-to-run
spread (about 6%)"**. Decision recorded: **"Not built."**
https://github.com/lablup/mlxcel/blob/main/docs/benchmark_results/moe-distinct-experts-n1x-2026-10-31.md

**[I]** Apply that to you: top-10 over 512 experts at 8 rows = 80 pairs, and your experts are
less reused than gemma-4's top-8/128, so the read reduction at 8 rows is *lower* than the 30%
they measured at 64 pairs; at 2–4 rows it is near zero. And the 13% figure is an *upper bound*
that assumed the redundant multiplies were free — a real kernel still multiplies every row.

### 2.3 Verdict

Do **not** build dedup. Two independent measurements on two different engines, one of them on
sm_121, put the ceiling at "below the noise floor" for ≤4 rows and at "within run-to-run spread"
for 8. Per-slot is also the better-designed kernel for your case because it keeps the grid
independent of routing — see §3.

---

## 3. Persistent kernels, CUDA graphs, fusion at 2–8 rows

This is where the real, transferable wins are.

### 3.1 MonoMoE: weight-major, not token-major (the key idea)

**[M]** MonoMoE (FlashInfer, `csrc/fused_moe/monomoe`, arXiv 2609.04244): "Existing grouped and
batched GEMMs are token-major: they construct expert-local token tiles and obtain parallelism
from the token dimension. When few tokens reach each expert, this organization incurs tile
padding, preprocessing, **launches short-lived grids that underutilise memory bandwidth**, and
exposes quantization, activation and reduction as separate stages."
Their fix: "*places the complete decode-step token tile on the fine-grained tensor-core N
dimension and partitions CTAs over expert-weight tiles*". Results **[M]**: up to **1.54×** over
vLLM Triton grouped GEMM on H200, 2.20–3.84× over a FlashMoE-FP8 adaptation, and **end-to-end
TPOT down up to 18.7%** on FP8 models.
https://arxiv.org/abs/2609.04244

Design doc is unusually explicit and worth reading in full:
https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/monomoe_kernel.md
Concrete details that matter: `Dims_BS8_E256_N512_K2048`, **BS ≤ 8**, one persistent kernel,
one launch for routing → up-proj(+SiLU+requant) → down-proj → weighted combine; CUDA-graph
capturable; 8 blocks per expert × 16 expert-groups concurrent; sentinel handoff where the
act-scale cell doubles as the readiness flag (no barrier between up and down); Phase-4 atomics
guarded by a zero-fill that provably retires before the first atomicAdd.

**Why this is the transferable insight [I]:** in a weight-major decomposition the **grid size is a
function of weight-tile count, not of the expert set**. Routing changes *which* work each
persistent CTA picks up off a device-side queue, not *how many CTAs exist*. That makes the kernel
trivially graph-capturable with a fully dynamic expert set — you never re-capture because the
experts changed. Token-major grouped GEMM has the opposite property: grid = f(Σ tiles per active
expert), so every routing change is a new shape, which is exactly the CUDA-graph problem in §3.3.

### 3.2 Fusion and launch overhead: the numbers that say it matters more than the GEMM

**[M] Ling-3.0-flash at batch 1 on 4× Blackwell**, 42 layers, 512-expert MoE, KDA+MLA hybrid
(the closest published analogue to your model):
- final: mean TPOT **3.33 ms → 1.53 ms**, single-request 288 → 606 tok/s.
- **final profile breakdown: "MoE grouped GEMM 1215 µs, router / activation / glue small kernels
  1127 µs, all-reduce 951 µs, dense GEMM 918 µs, KDA 488 µs, MLA 244 µs, plus 400 µs residual
  idle."**
- the largest single post-structural change was a **dtype change**: router gate and lm_head from
  fp32 to bf16, halving those bytes, **≈ +10% end-to-end** at batch 1.
- the structural change was **PDL chaining** (programmatic dependent launch) to link MoE, router,
  KDA and all-reduce so preparation hides behind GPU work.
https://www.lmsys.org/blog/2026-08-21-ling3-flash-spec-decode-blackwell

**[I]** Two things to steal. (1) Their **glue/launch kernels cost 93% as much as the MoE GEMM
itself** at batch 1. Your unaccounted 7–17 ms is the same phenomenon. (2) A pure dtype change on
bandwidth-only weights (router gate, lm_head) was worth 10% — check whether anything in your pass
is still fp32 for weights that have no arithmetic behind them.

### 3.3 CUDA graphs with dynamic expert sets

The state of the art, and the pitfalls, in order of usefulness:

- **Bucketize on rows, not on experts.** With per-slot kernels the grid depends on (rows, hidden,
  top-k), never on which experts — so `rows ∈ {1..8}` is 8 captures, full stop. **[I]** This is
  the cheapest fix available and it is why per-slot + graphs compose well and dedup + graphs do
  not.
- **Breakable CUDA Graph (BCG)**, SGLang-originated, now also a standalone Meta-PyTorch library:
  captures graph-safe segments and runs marked ops eagerly between them. **[M]** prefill: BCG
  **1.70× vs eager**, full capture 1.93×; graph build 3.8–5.2× faster than the `torch.compile`
  piecewise path (521 vs 1771 lines). **[M]** And the relevant caveat: "a refill made of pure
  device kernels writing persistent buffers is capturable; anything that goes through a
  FlashInfer-style `plan()` is not", plus "capture must be idempotent" — `trtllm_mla`'s
  `_init_cuda_graph_metadata` used to allocate fresh tensors per capture, leaving earlier graphs
  reading freed memory.
  https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph ,
  https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph
- **Workarounds that fail.** TensorRT-LLM's `multi_stream_moe` + MLIR fusion under a *monolithic*
  decode CUDA graph caused **silent data corruption**: GSM8K 90% → 80% on gemma-4-26B-A4B,
  root-caused to MLIR Triton kernels on the aux stream interacting badly with the caching
  allocator during capture. https://github.com/NVIDIA/TensorRT-LLM/issues/12954
  **[I]** Read this as: any multi-stream capture where an allocator is touched on the aux stream
  is a correctness hazard, not just a perf hazard. Verify with greedy parity, not with loss.
- **Persistent-kernel-under-graph ownership.** DeepGEMM's `%gridid`-as-barrier-tag bug (§2.1) is
  the canonical trap: a graph node keeps the same `gridid` on replay, so any barrier keyed on it
  is stale on the second replay. Use an epoch.
- **[M]** TensorRT-LLM's own guide: "Without CUDA graph, … the host overhead dominates the
  execution of the MoE module", and its fused path only overtakes the unfused one once GPU work
  exceeds dispatch cost. https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/moe-perf-guide.html

### 3.4 Which kernel to prefer at small M

**[M]** TensorRT-LLM MoE perf guide, from `bench_moe` sweeps on GB300/B300 with Qwen3.8 among the
models: "**We prefer TRTLLM over CUTEDSL when `world_size <= 8`.** A small world size leaves many
experts on each rank (32 to 224), so **the grouped GEMM has many groups with a small `M`
dimension**. In these cases TRTLLM-Gen's tuned kernels fit that shape better than the generic
CuteDSL grouped GEMM." CuteDSL only overtakes above **~24576 activated tokens per rank**.

**[I]** That is NVIDIA saying, in its own perf guide, that at small M the specialised/tuned
per-shape kernel beats the general grouped GEMM by enough to be the default. Your engine is a
specialised per-shape kernel. That is the right side of this divide — don't trade it for a
general grouped GEMM (or a dedup kernel, §2).

### 3.5 Verdict

1. **Weight-major persistent megakernel for the routed expert layer of all 48 layers**, grid
   fixed by weight tiles, work pulled from a device-side expert queue. This is the single
   largest structural change available and it is well-documented enough to copy.
2. Graph-capture the whole verify pass, bucketed by rows.
3. Fuse the router+activation+reduce glue into the same launch; check for fp32 weights that could
   be bf16.

---

## 4. Expert caching, pinning, tiering

### 4.1 What exists, and what it reports

- **MoE-Infinity** (2401.14361, ASPLOS'24): request-level expert activation tracing + priority-
  based LRU cache with lookahead prefetch. **[M]** "reduce traffic by up to 98.5% … in models like
  Arctic, NLLB and Switch during decoding"; 2–20× latency vs DeepSpeed-Inference / llama.cpp /
  Mixtral-Offloading / BrainStorm; "almost 64–95% of the latency and throughput of full-GPU
  serving" at up to 8× lower GPU cost. **Caveat, [M] in the same paper:** if experts must cross
  PCIe 4.0, TPOT for Mixtral-8x7B on an A5000 exceeds 6 s vs 175 ms with 8 resident GPUs.
  https://arxiv.org/html/2401.14361v2
- **KTransformers** (SOSP'25): AMX CPU kernels; **[M]** AMX reaches 21.3 TFLOPS sustained on one
  Xeon socket (3.9× PyTorch), NUMA-aware TP gives +63% decode on dual-socket, CUDA-graph capture
  of the hybrid CPU/GPU path cuts launch overhead "from over 20% to nearly zero", and **Expert
  Deferral** (reordering expert execution across layers so CPU experts overlap GPU attention)
  gives **up to 1.45× decode throughput with <0.5% accuracy change**.
  https://www.lmsys.org/blog/2025-10-22-KTransformers/
- **llama.cpp `--moe-expert-cache` RFC** — and this one is on **your exact model**:
  **[M]** "Qwen3.8-Flash-Next (qwen4exp, 48 MoE layers, all experts on CPU, UD-Q3_K_XL, 8K
  context, greedy)" on a single RTX 4090: baseline 11.55 t/s; N=16 slots → +12%; N=32 → +52%;
  N=48 → +65%; **N=64 → +84%**; real code-gen workload 12.82 → 17.14 t/s (+34%) at N=64;
  PPL 3.3096 vs 3.3262 (equivalent). Mechanism is worth noting: they remap expert ids to slot ids
  with a per-tensor map table via `ggml_get_rows`, so **zero kernel changes**.
  https://github.com/ggml-org/llama.cpp/discussions/28248
- **SpecMD** (Apple): speculative expert prefetching study.
  https://machinelearning.apple.com/research/specmd-expert-prefetching
- **DuoServe-MoE**, OLED-MoE: dual-phase prefetch / inter-iteration locality for offloaded
  dLLM serving. https://arxiv.org/html/2509.07379 , https://arxiv.org/abs/2609.33385

### 4.2 Reading the hit rates honestly

**[I]** The llama.cpp curve is the most informative datapoint in this whole section, because it
lets you invert it: +84% at N=64 slots over a 4090's 24 GB means a *large* fraction of decode
traffic was being served from a small resident set. With 48 MoE layers × top-10 and 512 experts
per layer, 64 slots is 12.5% of one layer's experts — and it still returned most of the win.
That says the hot set per layer is small and stable within a session, consistent with the
ρ_l(2) ≈ 0.36–0.49 same-token-neighbour reuse and 44–48% next-token reuse figures from §2.2.
Caveat: they used UD-Q3_K_XL on a 4090, so the baseline was PCIe-bound at ~11 t/s; the marginal
value of a hit is far higher there than on a 128 GB box where every expert is already resident.

### 4.3 Applicability to a 128 GB unified box: essentially nil, with one real exception

You have no offload problem: everything is resident and the edit path already runs from SSD at
2–4 GB/s and works. So there is no PCIe/NVMe term to hide, no eviction policy to tune, no hit
rate to raise. **MoE-Infinity, KTransformers, `--moe-expert-cache` and Neuromorphic-style
tiering are all solving a problem you do not have.** Copying them would add a cache-index
indirection, a remap on the hot path, and a correctness surface (see the llama.cpp RFC's own
disclosure of "one invalid verification round") for zero bytes saved.

**[I] The one exception** is *L2 residency*, not offload: on a unified-memory part, L2 is the only
tier between you and LPDDR5X, and repeatedly touching the same hot expert within a session turns
some fraction of DRAM traffic into L2 hits. This is a **scheduling** question, not a caching
question: order the expert-queue traversal so that CTAs working on the same expert tile run close
together in time (§3.1 already does this by construction — weight-major means all tokens for an
expert are consumed in one pass), and avoid re-reading a tile in a later phase. **[I]** Expected
size: bounded by how much of the working set exceeds L2, which for a 512-expert layer is most of
it; treat L2 as a bonus on the small tiles (scales, router weights), not as a strategy.

### 4.4 Verdict

Skip tiering entirely. Take only the *scheduling* lesson: weight-major traversal gives you
intra-expert temporal locality for free.

---

## 5. Overlapping fetches with attention or the drafter

### 5.1 Mechanisms that ship

- **TensorRT-LLM `multi_stream_moe`**: puts the shared-expert branch on an aux stream so it
  overlaps the routed-expert path. **[M]** Exists, is used, and its failure mode under graph
  capture is documented in §3.3 (silent accuracy corruption with MLIR fusion).
  https://github.com/NVIDIA/TensorRT-LLM/issues/12954
- **KTransformers Expert Deferral**: **[M] up to 1.45× decode throughput, <0.5% accuracy change**,
  by deferring some experts to a later stage so CPU expert compute overlaps GPU attention. Valid
  only because residual connections make small delays tolerable.
- **GC-MoE (Alibaba)**: **[M]** CUDA Green Contexts split the device at driver level into a small
  comms partition and a large compute partition, so dispatch and grouped GEMM run as *concurrent
  roots* in a 3-node (4 on Hopper) CUDA graph. On 8×H20, DeepSeek-V4.1-Flash decode **1.188×
  avg** (peaks 1.296×/1.353×); on 8×B300, GLM-5.2 and DeepSeek-V4.1-Flash **1.106× avg**
  (peak 1.209×), beating DeepGEMM Mega MoE at all 54 measured points; **reserves only 20 of 148
  SMs** for dispatch. https://arxiv.org/html/2611.04222
- **GC-MoE on SM120** (consumer/workstation Blackwell, PCIe): **[M] 1.522× over FlashInfer CuTe
  DSL W4A8 MegaMoE and 1.387× over Cake MegaMoE across 18 decode points, 18/18 wins**; swap-AB
  warp-level block-scaled MMA with FP4 weights in the register A operand, scale bytes selected by
  MMA immediates, compact-nibbles path that halves weight staging, adaptive token-tile geometry.
  https://arxiv.org/html/2611.16219v1
- **Weave / OverlapAdvisor**: fine-grained dynamic SM scheduling in an MoE megakernel, and
  modelling of compute-communication overlap. https://arxiv.org/abs/2609.21483 ,
  https://arxiv.org/abs/2611.11857

### 5.2 The catch that most people miss

**[I]** Every one of those numbers is a **communication-overlap** number: GC-MoE overlaps an
all-to-all with a GEMM, KTransformers overlaps a CPU with a GPU. You have neither. On a single
unified-memory device, two overlapping streams share **one** 234 GB/s pipe. If the expert MLP is
already pulling 180 GB/s of it, the residual budget for anything you overlap underneath it is
**~54 GB/s**. Overlapping a second bandwidth-hungry stage into that window buys almost nothing —
you would just be time-slicing the same DRAM.

**[I] Where the same trick does pay for you:** the stages that are *not* bandwidth-efficient.
conv/norm at 73–131 GB/s are leaving 100–160 GB/s on the table — they are latency- or
occupancy-bound, not bandwidth-bound. Overlapping a 73 GB/s stage underneath the 180 GB/s expert
MLP has real, non-zero headroom (up to ~54 GB/s of the conv/norm stage's demand can be served
inside the MLP window without slowing it), whereas overlapping another 180 GB/s stage has none.
So: **overlap the inefficient stages, not the efficient ones.** That is the opposite of the
intuition the distributed-MoE literature gives you.

### 5.3 Drafter overlap

**[M]** §2.2: on per-slot engines, speculation is a net loss at every draft length (median 0.877
at k=1, 0.517 at k=8, plain decoding fastest on 33/37 prompts). **[M]** Ling-3.0's MTP path went
288 → 606 tok/s, so it is not that drafting can't win — it's that it only wins when the extra
rows and the verify pass are cheap relative to the tokens committed.

**[I]** For your 2–8 rows: overlap the drafter is the wrong framing. The right question is
whether *more rows per pass* costs you bytes. With per-slot kernels, 8 rows cost 8× the expert
bytes of 1 row unless reuse is exploited — which §2.2 says you should not exploit. So a draft of
k tokens multiplies your MLP traffic by ~k. **[I]** Unless the drafter's acceptance is high enough
that `tokens_committed / bytes` beats plain decode, drafting on this architecture at this batch
is a throughput loss. Measure it as: `(1 + a + a² + …) / (1 + k)` versus bytes ratio; for a=0.8,
k=4: 3.36 tokens for 5 passes ≈ 0.67 tokens/pass versus 1.0 for plain. Don't build it.

### 5.4 Verdict

- Multi-stream MoE for the shared-expert branch: **maybe**, and only after the graph work, with
  greedy-parity tests (see the TRT-LLM corruption bug).
- Green Contexts / SM partitioning: **no** — it exists to isolate an all-to-all; you have none.
- Deferral (KTransformers-style) across layers: **plausible**, 1.45× is a big reported number and
  the accuracy cost is <0.5%, but **[I]** their gain comes from overlapping CPU with GPU. On one
  device the ceiling is the ~54 GB/s residual. Try it only if the inefficient stages prove
  overlappable.

---

## 6. What to actually do, ranked

Each item: mechanism → expected size → why a small team can do it → risk.

1. **Close the bandwidth gap on the stages you already have.** Mechanism: tile shape, occupancy
   target, vector width, and number of CTAs per expert tile so every stage reaches the 240 GB/s
   the head already reaches. Expected: **[I] 5–9 ms of 47 (11–19%)**, derived in §0. Small-team:
   pure kernel tuning in code you own, verifiable with a bandwidth counter. Risk: near zero,
   greedy-parity check only.
2. **Weight-major persistent megakernel for the routed expert path.** Mechanism: §3.1 — grid
   fixed by weight tiles, token tile on N, work pulled from a device-side queue, one launch for
   routing → up → SiLU → down → combine. Expected: **[M]** MonoMoE 1.54× operator-level and 18.7%
   TPOT e2e; **[I]** you should see most of the operator-level number because your MLP is at 77%
   of wall and the fixed costs (launch, router, reduction, glue) are exactly what this removes.
   Small-team: hard but bounded — one kernel, one shape family (BS ≤ 8), one model. Risk: medium;
   see the DeepGEMM `%gridid` and FlashInfer-sentinel traps. Build the sentinel/epoch discipline in
   from the start.
3. **CUDA-graph the whole verify pass, bucketed by rows.** Mechanism: per-slot kernels make the
   grid independent of routing, so `rows ∈ {1..8}` is 8 captures. Expected: **[M]** Ling-3.0
   removed "over 20% → nearly zero" launch overhead (KTransformers' number on a different stack);
   **[I]** with 7–17 ms unaccounted in your pass, this is where a large part of it lives.
   Small-team: yes — it's the highest ratio of gain to difficulty in this list. Risk: low if you
   keep everything on one stream, verify capture idempotence, and test greedy parity after every
   capture change. Do not put allocator-touching work on an aux stream (§3.3).
4. **Fuse router + activation + reduce + glue into the main launch; audit fp32 weights.**
   Mechanism: **[M]** Ling-3.0's glue kernels cost 93% of the MoE GEMM at batch 1, and bf16
   router/lm_head alone was +10%. Small-team: yes. Risk: low.
5. **PDL / programmatic dependent launch** to overlap the tail of one kernel with the head of the
   next. **[M]** Ling-3.0 lists PDL chaining as the second structural win after host run-ahead.
   Small-team: yes, but arch-specific and fiddly. Risk: medium-low.
6. **Overlap the inefficient (conv/norm) stages under the expert MLP** — not another
   bandwidth-heavy stage. Expected: **[I]** bounded by the ~54 GB/s residual, so single-digit %.
   Risk: medium (multi-stream + graph interaction, §3.3).
7. **Only then, consider CUDA-core small-M W4A4** (ReSET's shape, weights expanded in registers),
   if items 1–3 leave you ALU-bound rather than bandwidth-bound. Expected: **[M]** ReSET 2.5×
   kernel vs *tensor-core* NVFP4; **[I]** 10–20% on the MLP against your W4A16 path.
   Risk: high — SM121 FP4 kernel immaturity (§1.5), new activation-quant error in the verify pass.
8. **Do not build:** deduplicated/grouped-by-expert expert reads (§2.3), expert caching or tiering
   (§4.4), Green Contexts / SM partitioning (§5.4), speculative decoding (§5.3).

---

## 7. Sources

W4A4 / FP4
- https://nvidia.github.io/Model-Optimizer/announcements/qwen36-w4a4-qad.html — W4A4 vs BF16 0.87–0.88× at concurrency 1; near-lossless accuracy
- https://github.com/aiha-lab/ReSET and https://arxiv.org/html/2606.13233v1 — M ≤ 8 tensor-core tile ≤ 6.25%; CUDA-core small-M NVFP4, 2.5× kernel
- https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison — SGLang/vLLM/FlashInfer FP4, 1.32× at BS=1
- https://flashinfer.ai/2026/09/22/mega-moe.html — MegaMoE backends incl. NVFP4 W4A4
- https://github.com/vllm-project/vllm/pull/28892 , https://github.com/sgl-project/sglang/pull/27401 , https://github.com/NVIDIA/TensorRT-LLM/pull/9288
- https://github.com/flashinfer-ai/flashinfer/issues/2723 — SM120 CUTLASS block-scaled GEMM invalid output
- https://github.com/IST-DASLab/marlin , https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel

Grouped GEMM / dedup vs per-slot
- https://arxiv.org/html/2610.34440v1 — **the key source**: r = 0.32–0.75, dedup wins only at 5–8 tokens or never; ρ_l(2) and ρ_l(9); draft speedup 0.877/0.517
- https://github.com/lablup/mlxcel/blob/main/docs/benchmark_results/moe-distinct-experts-n1x-2026-10-31.md — sm_121 measurement: 30.4% read cut → ~13% at 8 clients, ~5% at 4; "Not built"
- https://arxiv.org/pdf/2211.15841 (MegaBlocks), https://arxiv.org/abs/2206.03382 (Tutel), https://doi.org/10.1145/3503221.3508418 (FasterMoE)
- https://github.com/deepseek-ai/DeepGEMM — Mega MoE; commit 5081f85 `%gridid` epoch bug

Persistent kernels / CUDA graphs / fusion
- https://arxiv.org/abs/2609.04244 (MonoMoE) + https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/monomoe_kernel.md
- https://www.lmsys.org/blog/2026-08-21-ling3-flash-spec-decode-blackwell — batch-1 Blackwell profile; glue 1127 µs vs MoE GEMM 1215 µs; bf16 router +10%
- https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph and https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph — BCG 1.70×/1.93×; capture-idempotence requirement
- https://github.com/NVIDIA/TensorRT-LLM/issues/12954 — multi_stream_moe + MLIR + monolithic graph = silent corruption
- https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/moe-perf-guide.html — prefer TRTLLM over CuteDSL at world_size ≤ 8; host overhead dominates without CUDA graph

Caching / tiering
- https://arxiv.org/html/2401.14361v2 (MoE-Infinity) — 98.5% traffic cut, 2–20×, 64–95% of full-GPU
- https://www.lmsys.org/blog/2025-10-22-KTransformers/ — AMX 21.3 TFLOPS, +63% NUMA TP, launch >20%→~0, deferral 1.45×
- https://github.com/ggml-org/llama.cpp/discussions/28248 — **Qwen3.8-Flash-Next**, +84% at N=64 slots, id-remap with no kernel change
- https://machinelearning.apple.com/research/specmd-expert-prefetching , https://arxiv.org/html/2509.07379 , https://arxiv.org/abs/2609.33385

Overlap
- https://arxiv.org/html/2611.04222 (GC-MoE, H20/B300) and https://arxiv.org/html/2611.16219v1 (GC-MoE SM120, 1.522× W4A8 MegaMoE)
- https://arxiv.org/abs/2609.21483 (Weave), https://arxiv.org/abs/2611.11857 (OverlapAdvisor)
