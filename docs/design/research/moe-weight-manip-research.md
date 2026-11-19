# Non-training, weight-level manipulations to speed up MoE decode
### A literature survey with honest quality-cost evidence
Target setting: Qwen3.8-Flash-Next class model (48 layers, 512 routed experts, top-10 + 1 shared, hybrid GDN/linear + sparse attention, NVFP4 QAD checkpoint, W4A16), served on a single GB10 (128 GB unified, ~234 GB/s measured sustained read). Decode verify pass ~47 ms at 2–8 rows, `mlp` 15–32 ms and scaling with row count because each row drags its own ~10 experts (16.6 distinct experts/layer at 2 rows, 40.5 at 8).

Everything below is restricted to **manipulations of the stored weights** (or of expert ordering) that need no gradients. I separate what papers measured from what I infer.

---

## 0. The economics that decide everything

Decode here is weight-bandwidth bound: the layer must stream the experts that its ~2–40 distinct experts touch. Two consequences drive the whole survey:

1. Anything that reduces **bytes read per layer** (fewer experts, smaller experts) buys time roughly linearly. Anything that only changes **where those bytes sit** buys time only if the current kernel is *not* already at DRAM peak.
2. **Diagnostic you should run first:** `mlp` bytes actually streamed ÷ 32 ms, compared against 234 GB/s. If you are at 85–95 % of peak, layout work (#2, #5) is dead and only structural reduction (#1, #3) can help. If you are at 50–70 %, layout work is the cheapest win available — DeepGEMM's decode path reports 98–99 % of measured pure-read peak after exactly this kind of work ([deepseek-ai/DeepGEMM PR #380](https://github.com/deepseek-ai/DeepGEMM/pull/380)).

---

## 1. Expert merging / averaging ("expert soup")

**What it is.** Cluster experts within a layer, then replace each cluster by a weighted average of its members' weights. Weights are averaged with permutation alignment (activation-matching / optimal transport) because two experts' hidden channels are not in correspondence.

**Who measured it.**
- **M-SMoE / MC-SMoE** (Li et al., ICLR 2024) introduced router-statistic-guided grouping plus neuron-permutation alignment, claiming "up to 80 % memory and 20 % FLOPs reduction with virtually no loss" on their benchmarks ([arXiv:2310.01334](https://arxiv.org/abs/2310.01334)).
- **HC-SMoE** (Chen et al., ICML 2025) showed M-SMoE's router-logit grouping is *not* task-agnostic and collapses: on Mixtral 4×7B, M-SMoE scores **0.3251** average vs 0.6748 for the original — near-random. Their own output-similarity hierarchical clustering with frequency-weighted merging does much better but still costs real accuracy ([arXiv:2410.08589](https://arxiv.org/html/2410.08589v4)):

| Model | Original | −25 % experts | −50 % experts |
|---|---|---|---|
| Mixtral 8×7B | 0.6748 | 0.6408 (−3.4) | 0.5877 (−8.7) |
| Qwen1.5-MoE 60×2.7B | 0.6008 | 0.5716 (−2.9) | 0.5223 (−7.9) |

- **REAP** (Lasby et al., ICLR 2026) supplies the theoretical objection that matters most for your architecture: merging forces several experts to share **one scalar gate per token**. With top-k renormalised routing, the layer can no longer express `w₁E₁ + w₂E₂` once `E₁` and `E₂` are fused — they claim an **irreducible error** and "functional subspace collapse". Empirically pruning beats merging on generative tasks, especially at 50 % compression ([arXiv:2510.13999](https://arxiv.org/abs/2510.13999), [repo](https://github.com/CerebrasResearch/reap)).
- **REAM** (Jha et al., 2026) pushes back: with gate-logit *and* output similarity, "pseudo-pruning" (a few large groups + many singletons), and sequential per-layer re-statistics, merging at **25 %** reduction reaches a GEN average **within ~1.1 points of the uncompressed model (69.8 vs 70.9)** at its best calibration mix, while saliency-free HC-SMoE clustering sits at 63.9–67.4 ([arXiv:2604.04356](https://arxiv.org/abs/2604.04356)). The same paper shows the method at 50 % is heavily calibration-dependent, with AIME25 swinging ~30 points across calibration mixes — so the 25 % result is a best-mix number, not a default.

**Does router-aware merging preserve outputs?** No — not exactly, and the deviation is structural rather than numerical. Router-aware grouping only picks *which* experts to fuse; it cannot restore the per-token independent gain each fused expert had. The best published framing is "preserves accuracy at ≤25 % reduction when calibrated on the deployment distribution" (REAM), not "preserves outputs".

**Applicability to an already-4-bit QAD checkpoint.** Mechanically possible (dequantise to bf16 → align → average → requantise), but three compounding problems: (a) merging error + second quantisation error stack on a model that was *distilled to tolerate one specific quantiser's error*, (b) the published recovery for stacked compression + 4-bit is distillation from the uncompressed teacher, i.e. training you don't have ([Quantization-Aware Healing, arXiv:2608.20953](https://arxiv.org/html/2608.20953v1)), (c) top-10-of-512 gives the router a *lot* of independent modulation to lose — REAP's argument is strongest exactly in this regime.

**Risk.** High. A 25 % merge on 512 experts is 128 experts' worth of function collapsed into others; REAM's 1.1-point GEN gap was measured after calibration on C4+math+code and *without* a 4-bit re-round on top.

---

## 2. Permuting expert order (+ the router's output rows) — the semantics-preserving one

**What it is.** Apply a permutation π to the expert list and the same π to the router's output channels. The layer's function is identical up to float summation order: the top-k set is the same set, softmax over the same logits, and the weighted sum is over the same terms. This is genuinely free — no accuracy question, only a bookkeeping one. (As a side note, this is the *only* item in this report with zero quality cost by construction.)

**Who measured what, and where the gain actually comes from.**
- **Cross-device communication**, not DRAM locality: a co-activation-graph placement (load-capped) cuts cross-device traffic **12–17 %** vs frequency-balanced placement at matched load balance, with the gain growing in device count and depth; the unconstrained version cuts 21 % but imbalances devices. The same write-up reports the honest negative: against an **LRU** cache, an explicit co-activation prefetcher wastes bandwidth because LRU already captures co-firing implicitly ([huggingface.co/KikoCis/moe-coactivation-placement](https://huggingface.co/KikoCis/moe-coactivation-placement)). It also documents the methodological trap that matters — clustering router co-activation over Gaussian noise gives locality-gain ≈ 1.000×, and only real hidden states reveal 1.4–2.1× structure. Any co-activation study of your model must use real prompts.
- **Offloading/cache-capacity systems** exploit the same structure: MoE-Infinity (activation-aware prefetch, [arXiv:2401.14361](https://arxiv.org/html/2401.14361v1)), CECAR (cache/expert co-aware routing, 4.5× on Qwen3-30B-A3B with 16 GB VRAM, [arXiv:2611.20119](https://arxiv.org/abs/2611.20119)), and a community repo reporting expert-cache hit rate 79.6 → 85.3 % over LRU at equal GPU memory ([moecache](https://github.com/joelvarun/moecache)).
- **Task-aware grouping for EP**: "Beyond Task-Agnostic" and CARE-MoE (correlation-aware expert placement) both target communication volume on multi-device / edge deployments, not single-device bandwidth ([arXiv:2606.01007](https://arxiv.org/html/2606.01007), [CARE-MoE](https://dl.acm.org/doi/pdf/10.1145/3832810.3832869)).

**Does anyone sort experts by co-activation or by weight similarity for cache hits?** Yes — but always where either (a) experts live on different devices, or (b) the working set exceeds fast memory. I found **no** paper reporting a decode-speedup from expert reordering when all experts are already resident in the same DRAM/HBM pool.

**Why I expect ~zero gain in your setting (my inference, testable).** You read ~17–40 experts per layer out of 512. Each expert's three matrices are hundreds of KB to a few MB — far above DRAM burst/page granularity, so each expert read is already a long sequential stream that the memory system exploits. Permuting *which* experts sit adjacent changes nothing about how many bytes you pull. The one place it could pay is if your loader issues per-expert async copies and could merge several co-selected experts into one large DMA — that's a kernel change, not a weights change, and it is exactly what a grouped-GEMM/`DENSEGEMM` formulation already does.

**The one quantisation-friendliness mechanism worth testing (my inference).** NVFP4 uses a **per-tensor FP32 scale** on top of per-16-element FP8 block scales. If your checkpoint stacks all 512 experts into one tensor per projection, that single FP32 scale is governed by the global amax, so *which* experts share the tensor changes the usable range of the block scales. Sorting experts by weight magnitude (or by amax) so that similar-scale experts share a stacked tensor could recover a little block-scale headroom. This is second-order (the FP8 E4M3 block scale has wide range) and I have seen no paper measure it — but it is cheap to test offline: compute per-expert reconstruction error before and after a magnitude-sorted re-pack, and look at how many block scales saturate. **Do not** assume a win; measure it.

**Risk.** Zero for correctness (verify with a bitwise-ish check on logits: identical top-k sets, |Δlogit| < 1e-5). Engineering risk is real: every index map (checkpoint → kernel → KV/aux metadata → any expert-parallel sharding) must agree, and a silent mismatch produces garbage rather than a crash.

---

## 3. Expert pruning by router statistics vs activation statistics

**What it is.** Delete whole experts and their router rows. Router-statistic criteria: activation frequency, mean gate value, top-k count. Activation-statistic criteria: the *magnitude* of what the expert contributes (`‖E(x)‖` weighted by its gate), i.e. REAP's `gate × ‖activation‖`.

**Reported quality cost.**
- **Frequency-only pruning is fragile and calibration-dependent.** In REAM's sweep, calibrating without code data collapses HumanEval/LiveCodeBench to near zero — a >40 point gap vs the best mix ([arXiv:2604.04356](https://arxiv.org/abs/2604.04356), Table 6).
- **Router logits vs frequency**: HC-SMoE's ablations show the two criteria behave differently with different clustering linkages and neither dominates ([arXiv:2410.08589](https://arxiv.org/html/2410.08589v4), Table 4).
- **Gate-weighted activation (REAP) is the better criterion**: mean accuracy decrease **1.9 %** with renormalised top-k logits vs **2.6 %** without, measured across ERNIE-4.5-21B-A3B, Qwen3-30B-A3B, Mixtral-8×7B-Instruct, GLM-4.5-Air, Llama-4-Scout ([repo README](https://github.com/CerebrasResearch/reap)). Headline: near-lossless at **50 %** pruning on Qwen3-Coder-480B and Kimi-K2 *for code generation and tool calling* — the "for that task mix" qualifier is doing a lot of work.
- **Activation-frequency pruning of the "Not All Experts are Equal" family** reports "satisfactory performance" while reducing size and increasing speed, but at modest ratios and with per-task calibration ([ACL 2024](https://aclanthology.org/2024.acl-long.334/)).
- **Hard pruning at aggressive ratios is catastrophic at matched latency**: in SERE's matched-TPOT comparison, HC-SMoE at 30 experts on Qwen1.5-MoE scores **13.97** average vs 48.52 for the unmodified top-4 model ([arXiv:2602.07616](https://arxiv.org/html/2602.07616), Table 1). This is the number to keep in mind whenever a paper says "50 % pruning, minimal loss": at 50 % *of a 8-expert layer* you are deleting half the model's capacity, and the honest baselines are ugly.

**Applicability to your checkpoint.** Pruning is the most attractive structural option because it is the only one that reduces bytes *and* keeps the remaining experts numerically untouched — i.e. the surviving weights stay exactly as NVIDIA's QAD distilled them, and the router keeps independent control. Cost: you must drop the corresponding router rows and renormalise, and you lose the pruned experts' contribution. 25 % (512→384) is the band where published evidence says quality is plausibly recoverable without training; 50 % is where you need either a friendly task mix or acceptance of a real hit.

**Risk.** Medium. Two concrete dangers: (a) your **private traffic** may route differently from any public calibration set — 512 experts with a heavy tail means the prunable set is workload-specific; (b) REAP's own note that renormalising top-k logits *after* pruning is required, and the gain is only ~0.7 points — get it wrong and you degrade silently. Also: 40.5 distinct experts/layer at 8 rows means pruning 25 % cuts distinct experts to maybe ~31–34, not to 30 — the decode-time saving is proportional but sublinear.

---

## 4. Channel permutation and rotations on an already-quantised checkpoint

**Two different things, keep them apart.**

**(a) Permutation of channels — cheap and semantically exact if done in pairs.** Permuting the K-dimension of `gate_up` requires the same permutation on the *input* activation (or on the producing layer's output rows); permuting the intermediate dimension requires the same permutation on `down`'s input rows. Applied consistently, the function is unchanged; and because it changes *which* 16 values share an NVFP4 block scale, it changes quantisation error. This is the mechanism behind RPTQ (assemble channels with similar min/max into the same group, [arXiv:2304.01089](https://arxiv.org/html/2304.01089v4)), per-input-channel quantisation ([arXiv:2309.15531](https://arxiv.org/abs/2309.15531)), and PerQ's permutation-before-rotation for block formats ([Brevitas/PerQ](https://xilinx.github.io/brevitas/master/papers/perq.html)). Caveat specific to your model: the MoE FFN's nonlinearity (`SiLU(gate) ⊙ up`) blocks any permutation that would have to cross the activation, so you can only permute the *input* side of `gate_up`/`down` as a shared residual-stream permutation — i.e. globally, across all layers, exactly QuaRot's `R`.

**(b) Rotation (Hadamard / QuaRot / SpinQuant) on a model that is already 4-bit.** QuaRot's premise is that rotations remove **activation** outliers so that activations can be quantised; it is a W4A4 method, and its headline is "≤0.47 WikiText-2 perplexity loss and 99 % of zero-shot at LLaMA2-70B 4-bit" ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)). SpinQuant shows random rotations vary by up to 13 points and learned ones recover most of the gap ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)). Neither is a weight-only result, and **you are W4A16** — your activations are already bf16, so the problem rotations exist to solve is not your problem.

**Is re-quantising a quantised model meaningful?** Yes, and this is better measured than I expected. AMD's online NVFP4 → MXFP4 requantisation (a *second* 4-bit rounding, on 5 production checkpoints) reports: GSM8K recovery 98.7–100.7 % of source; GPQA-Diamond-CoT and AIME25 within ±0.05 of an NVFP4-emulated reference, inside seed noise; and weight-level analysis showing the second error is zero-mean, **uncorrelated** with the first (mean pairwise correlation ≈ −0.04) so errors add in quadrature, `√(e₁²+e₂²)`, with output SQNR ~17 dB vs ~3.6 dB below NVFP4 itself ([ROCm blog, Sep 2026](https://rocm.blogs.amd.com/software-tools-optimization/nvfp4-to-mxfp4/README.html)). Startup cost 10–55 s. So "you can't requantise a 4-bit model" is false.

**But does rotation help the weights?** The honest state of the evidence is weaker than the requantisation evidence:
- Rotation demonstrably helps **weight-only** quantisation at 2–3 bits (QuIP/QuIP# incoherence processing is explicitly weight-only, and QuaRot's related-work section says so).
- At **4 bits with 16-element block scaling**, the outlier mass is already spread over blocks, and the survey "Transforms for LLM Quantization: The Great Inversion and Format Co-Design" argues the flattening prescription **weakens and can reverse on a floating-point grid** — i.e. transforms derived from uniform-INT4 arguments are not guaranteed to help FP4 microscaling formats ([arXiv:2608.25188](https://arxiv.org/html/2608.25188)). The same survey prices the online transform: fused transform ≈ 0.07× of end-to-end runtime, three separate online Hadamards ≈ 0.26× — at decode batch sizes that is not free.
- I found **no** paper that rotates an NVFP4 QAD checkpoint, requantises, and measures decode speed and quality. That is a genuine gap, and it is the specific experiment you'd have to run.

**Risk.** Medium-high, and asymmetric in a way worth naming: NVIDIA's checkpoint was produced by **quantisation-aware distillation**, so the weights are already positioned where the *NVFP4 error field* is benign. A rotation followed by requantisation creates a *new, independent* error field of similar magnitude. The AMD data says two independent 4-bit errors cost little in aggregate — but that was FP4→FP4 with the same block structure, not a rotation that also redistributes the weight distribution away from the one the distillation optimised. Second risk: the rotation must be absorbed (`W ← WR`, and the residual stream rotated globally, with norm gains folded out) or you pay an online matmul per token; for a MoE FFN you cannot rotate between `up` and `down` because the elementwise nonlinearity blocks it.

---

## 5. Weight reordering / packing that changes access pattern but not numerics

**What it is.** Pre-shuffle weights and scales into the tile layout the kernel's tensor-core instructions expect; pad expert dimensions to tile alignment; drop the padding from the *streaming* path so no padded bytes are read; lower the `BLOCK_M` floor at decode token counts.

**Measured gains, all from shipped kernels not papers:**
- **DeepGEMM SM120 decode path**: removing M-padding row stores, M-padding A-reads, and dropping `BLOCK_M` 64→32 gives 2.1–8.2 % over CUTLASS on all tested MoE shapes at M=32, and lifts streaming to **98–99 % of the device's measured pure-read DRAM peak** — after which there is nothing left to win from layout ([PR #380](https://github.com/deepseek-ai/DeepGEMM/pull/380)).
- **TensorRT-LLM DENSEGEMM** (NVFP4 MoE on Blackwell): reformulating the routed MoE as dense FC1/FC2 with per-token alpha masks reaches **1.12×** over the grouped-GEMM path in the 64–208 token regime, TP8, DeepSeek-V3-style shapes — exploiting that "the extra arithmetic has limited latency impact" while the denser GEMM shapes improve utilisation ([NVIDIA tech blog](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html)).
- **Pre-shuffled weights at load**: MXFP4 weights and block scales reshuffled once into the MFMA tile layout, expert dims padded to tile alignment, so "weight reads in every forward pass are coalesced with no need for in-kernel repacking or bounds handling" ([ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/nvfp4-to-mxfp4/README.html)). Presented as an enabler, not a measured multiplier.
- The shared premise in all three: at 2–40 rows, MoE decode is weight-bandwidth bound and padding/scheduling overhead is pure waste.

**Applicability to your checkpoint.** Fully applicable and this is the only item with **no quality question at all** — the arithmetic is identical, only the addressing changes. It is also the item most likely to already be partly implemented in a fork like yours; the DeepGEMM numbers tell you what to check for (padding read, `BLOCK_M` floor, per-expert DMA batching).

**Risk.** Low scientifically, medium operationally: layout bugs are silent numeric corruption, and rewrites that touch the checkpoint format break every downstream tool (quantiser, evaluator, other machines).

---

## 6. What I'd actually do, in order

1. **Measure the bandwidth ceiling of `mlp` first** (bytes streamed ÷ 32 ms vs 234 GB/s). This single number decides whether items 2–3 are worth anything.
2. **Packing/layout pass** (#5). If `mlp` is not at ~90 % of read peak, this is where the remaining 10–40 % is, and it costs zero quality. Check for padded reads, `BLOCK_M` floor, and whether co-selected experts can be issued as one DMA.
3. **Expert permutation** (#2) as a free enabler for (2) — group co-selected experts adjacently so a single DMA covers a group, and (separately) test magnitude-sorted re-packing for block-scale headroom. Zero quality cost; expect small or zero gain unless the loader batches DMAs.
4. **Prune 25 %, not 50 %, by gate × activation norm** (#3), with router rows dropped, top-k renormalised, and the prunable set computed on *your* traffic. This is the only option with a plausible path to a real byte reduction without training. Budget for a genuine quality measurement, not a vibe check.
5. **Do not merge** unless you need more than 25 %. REAP's irreducible-error argument plus HC-SMoE's −7.9 to −8.7 points at 50 % is the strongest negative evidence in this survey, and it is worst exactly in a top-10-of-512 regime.
6. **Rotation (#4): skip, or treat as an experiment with a stated prior of "no gain".** The motivation (activation outliers) doesn't apply to W4A16, FP4 block formats already handle the outlier problem, the survey says the flattening argument weakens on floating-point grids, and no one has published the NVFP4-QAD-rotate-requantise result. Requantisation itself, by contrast, is safe if you ever need a format change.

---

## 7. Honest gaps in the literature

- **No public measurement** of expert merging / pruning / permutation applied to a *4-bit QAD-distilled* checkpoint (NVIDIA's Qwen-family FP4 releases). Every quality number above comes from bf16 checkpoints, sometimes fine-tuned afterwards.
- **No public measurement** of expert permutation for intra-DRAM locality on a single device. The 12–17 % figure is cross-device communication.
- **No public measurement** of rotation + requantisation on an FP4 microscaling checkpoint.
- Reported "minimal loss" claims are almost always on multiple-choice suites; the generative suites (AIME, HumanEval, LiveCodeBench) are where these methods fall apart, and where the calibration mix matters most.

---

## Sources

- MC-SMoE / Merge-then-Compress — https://arxiv.org/abs/2310.01334
- HC-SMoE — https://arxiv.org/html/2410.08589v4 (ICML 2025)
- REAP — https://arxiv.org/abs/2510.13999 , https://github.com/CerebrasResearch/reap
- REAM — https://arxiv.org/abs/2604.04356
- "Not All Experts are Equal" — https://aclanthology.org/2024.acl-long.334/
- SERE (expert re-routing, matched-TPOT comparison tables) — https://arxiv.org/html/2602.07616
- Elbow-based routing (training-free dynamic top-k, 5.3 % latency) — https://arxiv.org/html/2608.04401v1
- QuaRot — https://arxiv.org/abs/2404.00456
- SpinQuant — https://arxiv.org/abs/2405.16406
- Transforms for LLM Quantization (survey; FP-grid reversal, transform cost) — https://arxiv.org/html/2608.25188
- RPTQ (channel reordering for quantisation) — https://arxiv.org/html/2304.01089v4
- per-IC quantisation / channel isolation — https://arxiv.org/abs/2309.15531
- PerQ (permutation before block rotation) — https://xilinx.github.io/brevitas/master/papers/perq.html
- AMD NVFP4→MXFP4 online requantisation (second-rounding error measurements) — https://rocm.blogs.amd.com/software-tools-optimization/nvfp4-to-mxfp4/README.html
- Quantization-Aware Healing (stacked compression + 4-bit needs distillation) — https://arxiv.org/html/2608.20953v1
- MoE-Infinity — https://arxiv.org/html/2401.14361v1
- CECAR — https://arxiv.org/abs/2611.20119
- moecache (hit-rate 79.6→85.3 % over LRU) — https://github.com/joelvarun/moecache
- Co-activation placement write-up (12–17 % comm cut; LRU makes prefetch useless) — https://huggingface.co/KikoCis/moe-coactivation-placement
- CARE-MoE — https://dl.acm.org/doi/pdf/10.1145/3832810.3832869
- DeepGEMM SM120 decode PR — https://github.com/deepseek-ai/DeepGEMM/pull/380
- TensorRT-LLM DENSEGEMM — https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html
