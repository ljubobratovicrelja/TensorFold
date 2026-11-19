# Approximate execution & speculative-decoding levers for a hybrid GDN + QSA MoE, without retraining

Target system: Qwen3.8-Flash-Next family checkpoint, NVFP4, single DGX Spark (GB10, 128 GB unified, ~234 GB/s measured), 48 layers (36 GDN / 12 QSA), 512 routed experts top-10 + 1 shared, per-layer dense hyper-connection matrices, ~28.8 GB n-gram PLE table read 16 rows/token, MTP head + external DFlash2 drafter. Decode round ≈65 ms (≈47 ms verify + drafter), 40–55 tok/s, 2–8 rows/pass, acceptance ≈0.55.

Convention: **[measured]** = number taken from a cited source's own experiments; **[inference]** = my derivation or extrapolation to this box; **[unverified]** = single-rig or self-reported, not independently replicated.

---

## 0. Bottom line, ranked by expected value per unit of risk

| Rank | Lever | Expected gain here | Quality cost | Verdict |
|---|---|---|---|---|
| 1 | Fix draft length / row count to k=2–3 at a=0.55; stop running 7–8 rows | 10–25% of decode vs. a k=6–7 setting | **none** (exact) | Do first — free |
| 2 | Draft-tree selection biased to re-use already-loaded experts (EcoSpec-style) | 5–15% of the verify pass | **none** (exact) | High value on this box |
| 3 | IndexShare-MTP-style freeze of QSA top-k across draft steps (if not already in) | draft-step cost ↓ from N indexer calls to 1 | **none** | Check first; may already ship |
| 4 | PLE: mmap/pinned-host the table, skip it in the drafter | memory + KV capacity, ~0 throughput cost | none | Likely already done; verify drafter path |
| 5 | QSA knobs: token budget K=2048 / block r=4 | none at ≤8k ctx; matters at ≥64k | small, tunable | Only above ~32k context |
| 6 | Channel-level (not expert-level) skip with low-rank recovery | 1.6–1.9× MoE block at batch 1 in the paper's kernels | 98.1% accuracy retained at 17.8% of reads | Real, but needs kernel work |
| 7 | Expert tail-skip (route tail skip, rank ≥7 of 10) | **~0%** when experts are resident | KLD 0.068→0.348 on NVFP4; ~4% of known facts | **Don't** — measured as pure downside on a big card |
| 8 | Training-free layer skipping / early-exit self-speculation | 0 to negative | catastrophic on AR models | **Don't** — see §3 |
| 9 | KV eviction / quantised KV for speed | ~0 — attention is 1.4 ms of 47 | large | **Don't** for speed |

The single most important structural fact: **on this model, at this batch size, an extra verification row is nearly free and an extra *expert* is not.** Every lever that adds rows pays; every lever that removes expert *bytes* pays only if those bytes were actually going to be re-read.

---

## 1. Skipping low-weight experts at runtime ("route tail skip")

### 1.1 The exact measurement on this model family

The claim in circulation (+4–10% decode at KL 0.056–0.086 vs. a Q8 reference) is the optimistic end of a distribution. The controlled ablation that exists for Qwen 3.8 Flash Next itself — Basalt engine, route prior (bias 1.5) + tail skip at rank ≥7 of 10, on a 5090 + 5060 Ti with 128 GB DDR5 — gives **[measured]**:

| Pack + setting | File | PPL | KLD vs Q8_0 ref | same top‑1 |
|---|---|---|---|---|
| Q8_0 exact | 178 GiB | 3.276 | 0.031 | 94.1% |
| NVFP4 exact | 122 GiB | 3.304 | **0.068** | 91.4% |
| NVFP4 + shipped flags | 122 GiB | 3.931 | **0.348** | 81.8% |
| Q8_0 + shipped flags | 178 GiB | 4.221 | 0.401 | 79.4% |
| Q2_0 exact | 65 GiB | — | 0.436 | 78.2% |
| NVFP4 + worst (bias 16, skip≥1) | 122 GiB | 5.917 | 0.823 | 71.3% |

And the speed side, which is the decisive part:

- NVFP4 **resident** on the big card: **272 → 271 tok/s** — i.e. *nothing*.
- Q8_0 **offloaded** (178 GiB > pool): **117 → 202 tok/s** (+73%).
- Task evals: Ladders −1.6 [−3.4, +0.2], Facts −1.5 [−2.5, −0.5] — roughly **4% of known facts lost** for 1 tok/s.
- On a 16 GB card mimicking small-VRAM: +56% speed for ~3× the damage.
Source: https://richland.au/blog/measured-moe-offload-speed-hacks-undo-a-quants-precision-q8-0-with-shipped-flags-lands-next-to-q2-0-on-kld (primary: https://reddit.com/r/LocalLLaMA/comments/1xkbbqi/ , repo https://github.com/jesdga95/basalt)

Note the KL numbers there (0.348, 0.401) are an order of magnitude worse than the 0.056–0.086 being circulated — that spread is the difference between skipping *one* tail slot occasionally and skipping rank ≥7 unconditionally. Treat any "KL 0.05–0.09" figure as the mild end and the 0.35 figure as what an unconditional rank-7 cut costs.

### 1.2 Why the gain vanishes when experts are resident

Route tail skip converts an *I/O* saving into time. If the 10 experts a token routes to are all in HBM/unified memory, skipping slot 10 removes ~1/10 of the routed-expert bytes for that token but leaves the kernel reading 9 experts with the same launch overhead, and — because consecutive tokens co-route heavily — much of the union is L2-warm anyway. Consecutive-token expert reuse is **[measured]** 0.363 / 0.480 / 0.492 at n=2 for OLMoE / Qwen3-30B-A3B / gpt-oss, rising to 0.76–0.86 at n=9 (https://arxiv.org/html/2610.34440). Cohere measures the union of experts over a 4-token verification pass at only **1.25–1.42×** a single token's set rather than 4× **[measured]** (https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding).

**[inference]** On your box: 10/512 of ~63 GiB routed-expert weights ≈ 1.23 GiB/token; dropping 3 tail slots ≈ 0.37 GiB ≈ 1.6 ms at 234 GB/s ≈ 3.4% of a 47 ms pass — *if* those bytes were cold. They are not: the measured answer on resident NVFP4 was 0.4%. And the 2610.34440 result explains the shape of the loss: with a per-slot kernel each extra row costs a full token's expert bytes, but L2 serves repeated reads at 0.18–0.34 of the cost on CUDA.

### 1.3 Where skipping *does* pay: the accuracy side is strongly asymmetric

The quality cost is not linear in rank — it is a cliff between the tail and the head.

- **[measured]** Omitting the two *lowest*-scoring experts raises perplexity by 0.12; dropping the *top*-2 degrades output by 93.28 (https://arxiv.org/html/2610.01265, RapidMoE).
- **[measured]** `Not All Experts are Equal` (Lu et al. 2024): expert skipping with r=6 gives 1.2–1.3× speedup at modest loss, and composes with expert pruning (https://arxiv.org/abs/2402.14800).
- **[measured]** ASET (EMNLP-findings-track style, https://aclanthology.org/2026.findings-acl.2140.pdf): static layer-wise top-k schedules yield **10–78% throughput gains with minimal degradation**, including ≥10% on DeepSeek-V3 *without measurable loss*; on DeepSeek-V2-Lite, dropping from full to 2 active experts costs ~7.5% average, while keeping 3.3 on average stays within 1%. Critically: **the best schedule is model-specific**, and sigmoid-routed models (DeepSeek-V3) are less tolerant than softmax-routed ones.
- **[measured]** A training-free, calibration-free variant (ACE, https://arxiv.org/html/2609.05228v1) reports 50% slot skipping on Qwen3.6-35B-A3B with **1.41× decode**, +4.15 pp over the best baseline, always keeping top-1.
- **[measured]** MoDES (CVPR 2026, https://arxiv.org/abs/2511.15690): 83–88% of experts skipped with >95–96% accuracy retained — but decode speedup is only **1.24–1.26×** vs. 2.03–2.16× prefill, precisely because *decode is memory-bound and fewer experts saves less than it looks*.
- **[measured]** SlimWise (https://arxiv.org/abs/2609.34117): prefill with the full model, **decode with a pruned pool reusing the prefill KV cache**; 1.81× decode at 50% pruning, 2.39× at 75%, with a short distillation stage to recover the residual loss.

### 1.4 Applicability

Route tail skip: **no**, not on this box, at these numbers. Expert-pool pruning for decode only (§1.3) is a different and more attractive proposition *if* you are willing to pay a small distillation; without retraining, SlimWise's KV-handoff still earns part of the win, and its 50% figure is 1.81× — but that is a serving-throughput number under batch, not single-stream latency, and you are single-stream.

---

## 2. Top-k reduction / top-1 routing at decode only

**Mechanism.** Drop the slot budget from 10 to m<10 per token, renormalising gate weights. Cheapest possible kernel change; reduces both bytes and expert-GEMM work.

**Measured quality cost** (all **[measured]**, from ASET unless noted):

- DeepSeek-V2-Lite (softmax, 64 experts): full → 2 experts ≈ −7.5% average accuracy; **3.3 average experts keeps degradation within 1%**; best-case at 2 experts ≈ −6%.
- DeepSeek-V3 (sigmoid, 256 experts): same curve but **unstable across reduction layouts** — because sigmoid routing does not normalise competition, retained experts contribute more uniformly and removing any single high-weight expert hurts more than one expects. Best pattern: ascending-cut for V3, descending-cut for V2-Lite.
- Qwen3.6-35B-A3B at 50% slot skipping: −7.96% WikiText-2 PPL and **+4.15 pp** downstream over the strongest baseline with ACE's screening; plain thresholding at 50% is materially worse.
- Aggressive threshold-only skipping (67–83%) costs >11% average accuracy in the MLLM baselines MoDES beats.

**Top-1 routing specifically.** No paper found that drops to top-1 without retraining and keeps quality on a fine-grained MoE. **[inference]** From the ASET curve, top-1 on a 512-expert top-10 model is far past the knee; expect double-digit loss. Top-1 is only safe where the router is already near-degenerate.

**Applicability.** A *mild* reduction (10 → 7 or 8, or a per-layer threshold that keeps top-1 always) is the only defensible version, and even that needs a per-layer search, because layer sensitivity varies by more than an order of magnitude (that is exactly what MoDES' offline α⁽ˡ⁾ calibration measures — mean KL between the full model and the model with layer l's experts skipped). Expected single-stream gain on your box, from the resident-expert argument in §1.2: **single-digit percent at best**, and less than the 10→8 count suggests because the union across rows barely moves.

**Cheaper and better-measured alternative: channel selection.** Skip *channels* inside the experts a token already selected, and recover them with a rank-32 low-rank map built from the down-projection columns. **[measured]** 98.1% of full-model accuracy at 17.8% of routed-expert reads; MoE blocks 1.60–1.90× faster than optimised vLLM at batch 1 (https://arxiv.org/html/2610.37386). This is the version of "skip low-weight stuff" that survives contact with a resident-expert box, because it cuts bytes *within* an expert rather than removing whole expert reads the L2 was serving anyway.

---

## 3. Layer skipping / early exit / self-speculative block skip

This is the category with the clearest negative result, and it matters because "skip a block and verify it" looks free.

**3.1 LayerSkip needs the training recipe.** LayerSkip (Meta, ACL 2024, https://arxiv.org/abs/2404.16710) gets 1.86× from exiting early and verifying with the remaining layers — but only on checkpoints trained with layer dropout + early-exit loss. The HuggingFace reference implementation states it plainly **[measured]**: *"For the baseline checkpoints that have not been pretrained or continually retrained with the LayerSkip training recipe, early exit self-speculative decoding is slower than autoregressive decoding. This is because during training of most LLMs, earlier layers are not motivated to learn to predict the output, and hence generating tokens using earlier layers will have a very low acceptance rate."* (https://huggingface.co/blog/layerskip)

**3.2 Training-free depth reduction: it exists, but it is a token-position trick, not a verify trick.** D3 (Position-Aware Depth Decay, ACL Findings 2025, https://aclanthology.org/2025.findings-acl.154/) exploits the observation that later tokens have lower perplexity and need fewer layers — decay the executed depth as `⌊L·α^i⌋`. Result: **1.5× speedup with <1% drop** on GSM8K and BBH across Llama 7B–70B, no retraining. Note what it is *not*: it is not self-speculative and it is not verified, so the savings are approximate, not exact.

**3.3 Skipping whole blocks is catastrophic on AR models.** A controlled 2026 study comparing diffusion and autoregressive LMs **[measured]**: Qwen2.5 loses 25–65% of benchmark retention at a **2-layer** skip (7% of FLOPs), while native diffusion LMs tolerate 6 of 32 layers at 88–102% retention. It also finds **consecutive skipping is catastrophic** — LLaDA drops GSM8K retention 91.8%→75.3% when removals are allowed to be adjacent, and Dream-7B collapses 76.8%→14.1% at a 2-layer consecutive skip (https://arxiv.org/html/2603.07475v1). Skipped layers concentrate in the first 40–60% of depth.

**3.4 Single-layer skip is often free — but not reliably.** A multilingual/multi-task study across 9 instruct models (Gemma 3, Llama 3.2, Qwen 3, OLMo 2) finds single-layer skipping frequently *improves* benchmark scores, that the best layer is in the second half of the network most often, that it varies by task/language, and that **gains concentrate in structured prediction while degradations concentrate in precise generation** (https://aclanthology.org/2026.groundlm-1.49/). Hopscotch gets more: 4 attention blocks skipped at <2% drop, up to 7 at <3% with **15% speedup** — but it requires fitting lightweight per-block scaling parameters (no pretraining data, model weights untouched) (https://arxiv.org/html/2506.03303v2).

**Applicability.** On a 3:1 GDN:QSA hybrid, the layers are *not* interchangeable: 36 of 48 carry a fixed-size recurrent state and 12 carry the KV. Skipping a GDN layer destroys a state that later layers depend on; skipping a QSA layer removes the only exact-retrieval path in that block. **[inference]** Expect the AR-model result, not the diffusion-model result. A verified "skip block k..k+3 and let the rest correct it" scheme requires the drafter to *be* that prefix, i.e. LayerSkip's trained regime — which you cannot enter. The one training-free variant worth a measurement is Hopscotch-style scaling, and even that is a fit, not a free lunch.

---

## 4. Approximate attention at long context, and QSA specifically

### 4.1 The economics on this box

Attention is **1.4 ms of a 47 ms verify pass** (≈3%). No amount of attention approximation can move decode at short-to-medium context, and the published KV-eviction wins (Quest: 7.03× self-attention speedup, 2.23× end-to-end *decode*) are end-to-end wins on dense models where attention dominates. **[inference]** Here the ceiling on any attention-side saving at 8k context is ~3%.

Above ~32k it is a different story, because the QSA *indexer* — not the attention it feeds — becomes the cost: it scores the query against every ⌈L/4⌉ compressed block, so its cost is O(n²/r) and grows quadratically.

### 4.2 What QSA actually is

From the technical report (https://arxiv.org/html/2608.30320, PDF: https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf) and the SGLang day-0 write-up (https://www.lmsys.org/blog/2026-08-26-qwen-flash-next):

- **Two paths.** A lightweight **indexer** decides *where to look*; **sparse GQA** reads the selected entries from the *original, uncompressed* K/V cache. The compressed keys are only an index — the final softmax and value aggregation use full-precision K/V.
- **Indexer shape.** MQA: 4 query heads (d=128) + 1 shared key head. Keys are average-pooled in non-overlapping blocks of **r=4**, RMSNorm'd, then **partial RoPE on 64 of 128 dims applied *after* pooling** (rotate-after-pool matters: averaging tokens that already carry different rotary phases scrambles the positional signal).
- **Score.** `s(t,b) = (1/√128) Σ_h ReLU(⟨q_t,h , k̄_b⟩)` under strict block-causal masking. ReLU (not raw dot) means a head with negative affinity abstains rather than voting against, so one strong positive head can carry a block in.
- **Selection.** Keep the best **512 blocks** (= 2,048 logical token positions), expand to tokens, and always append the 0–3 tokens of the current incomplete block → at most **2,051 positions** attended.
- **Tuning knobs that exist:** token budget K=2048, block/compress ratio r=4, indexer head count, partial-RoPE dim split, per-layer vs. shared indices. K and r are the ones exposed in serving configs.
- **Trained, not bolted on.** QSA was introduced by two-stage continued pretraining at 256K: stage 1 distils a dense teacher's *block-pooled* attention distribution into the indexer via KL (1,000 steps, ~2B tokens, lr 1e-3); stage 2 trains backbone+indexer jointly with the KL restricted to the selected blocks. The LM-loss gap vs. full attention is on the order of **1e-4** **(measured)**. This matters for you: the indexer is a trained component, so its selection quality is not something you can retune freely — but the *budget* is.

### 4.3 The lever that actually pays: IndexShare MTP

SGLang's **IndexShare** **[measured]**: the top-k selection computed by the draft-extend pass over the tokens the target just accepted is **held frozen for the whole MTP iteration**, so every draft decode step skips the indexer and reuses that selection plus the positions drafted since. Per-iteration indexer work drops from **N invocations to 1**, and the metadata kernels that exist only to feed it (compressed decode view, pending-ring, group-ring) leave the draft decode step entirely. **Accept length is unchanged** — the justification is that the query has moved by at most N positions out of L, so the reused ranking is essentially what the indexer would recompute.

Two things to check on your engine: (a) whether this is already on (it is the difference between an O(L/4) scan per draft step and one per iteration), and (b) the hard constraint it imposes: the pending index-key ring holds exactly one group of four, so engines refuse `num_draft_tokens > 4` for this architecture with `NotImplementedError: Qwen QSA requires speculative_num_draft_tokens <= the QSA compress ratio (4)`. **Your draft budget is architecturally capped at 4 by r=4.** That is a real, published ceiling (https://huggingface.co/randomllama/Qwen3.8-Flash-Next-DGX-Spark-field-notes).

### 4.4 Eviction / quantised KV: what the evidence says if you ever do need it

- **Quest** (query-aware page selection, keeps all KV): 7.03× self-attention speedup, 2.23× e2e decode latency, and — the useful part — **1K-token budget matches full cache** on LongBench, with passkey retrieval at 99–100% (https://arxiv.org/abs/2406.10774).
- **Eviction is far more fragile than its papers suggest.** Under a shared globally-capped harness, LRU/H2O/SnapKV/StreamingLLM/Ada-KV/QUEST all collapse to **F1 0.01–0.04 against a 0.315 full-cache ceiling** without structural protection, and *do not improve with a larger cache* — because once the prompt-boundary anchors are gone, quality is gone regardless of what survives. Reserving **10% of the cache at each boundary** recovers 69–90% of ceiling (https://arxiv.org/html/2605.18053). Practical rule: protect the sink and the boundary tokens; the scoring function is secondary.
- **Quantised KV** is the boring, safe version, and it is already in your serving stack (FP8 KV is the default in the Spark recipes).

**Applicability.** Skip attention approximation for latency. Use QSA's own budget only if you go past ~64k context, and then measure the indexer cost, not the attention cost. If you need *capacity* (more KV), the PLE offload in §5 buys 78% more KV tokens for free, which is a much better trade than evicting.

---

## 5. n-gram / PLE table tricks

### 5.1 What the table is

**[measured]** (SGLang): PLE is a **hash-addressed learned n-gram embedding memory** at decoder block 2 (layer index 1), **51.2 B parameters, ~95.4 GiB in BF16**. For token x_t, **eight 2-gram hash heads** over (x_{t−1}, x_t) and **eight 3-gram hash heads** over (x_{t−2}, x_{t−1}, x_t) produce **16 row IDs**; each row contributes 160 values, concatenated to a 2560-dim E_t. E_t becomes K_t (4×2560) and V_t (2560); a gate over Norm(Q_t), Norm(K_t) produces 4 scalars; the gated value is added to its own depthwise-conv output and injected into the four hyper-connection branches. **Only 16 rows are touched per token** — 2560 of 51.2 B values.

Your 28.8 GB is a quantised/sharded form of the same table (fp8 ≈ 48 GiB, NVFP4 ≈ 24 GiB at the full width; the exact figure depends on the pack).

### 5.2 The offload result, which is the whole trick

**[measured]** Sparse pinned-host offload: keep each rank's vocabulary-parallel shard in pinned host memory, gather only the 16 selected rows into a small BF16 GPU buffer with a Triton UVA kernel, and overlap the gather with the first decoder block on a dedicated stream. On H200/TP4: target weights **83.91 → 60.45 GiB (−23.46 GiB)**, KV capacity **1.84M → 3.28M tokens (+78.5%)**, and matched throughput **−0.07% geometric mean** across 1/2/4 concurrent requests, with **bit-identical output IDs**. The `mmap`-from-NVMe variant on a Spark drops the checkpoint from 122 to ~75 GiB and reports no decode cost (https://github.com/roozich/Qwen3.8-Flash-Spark).

**So: the table is already effectively free to move off the GPU.** There is nothing left to win by compressing it further — the only remaining PLE win is *not paying for it twice*.

### 5.3 The draft-model trick

**[measured]** *(a)* SGLang: **only the one-layer MTP draft model disables PLE** — the target retains it during prefill, decode, and verification. *(b)* LongCat's N-gram Embedding paper makes the same point from the training side: in speculative decoding the draft is shallower and its overhead is proportionally larger, so they (i) use a **conventional embedding layer for the draft model** to bypass n-gram lookup and (ii) **cache n-gram embeddings during the drafting phase** to eliminate the recomputation at verify (https://arxiv.org/html/2601.21204v1).

**[inference]** If your drafter is a separate DFlash2 model it may still be paying for a 16-row gather × k draft steps. If it is disabled/deduplicated per draft step the savings are on the order of one gather per token, i.e. small but free.

### 5.4 "Can it be skipped on ordinary text?"

Partially, and the mechanism is already implemented: **hit rate is a property of the n-gram order**. **[measured]** 2-gram hashing has a *gradual* vocabulary hit rate (many pairs unseen), while higher-order n-gram hashing converges quickly to hit rate ≈ 1.0 (https://arxiv.org/html/2601.21204v1, Fig. 3a). Consequence **[inference]**: on ordinary prose the 3-gram head almost always hits and carries most of the signal; the 2-gram head is the one that misses, and a miss is a wasted row read. A hit-rate-aware policy (compute the 2-gram contribution only when its hash is in a precomputed hot set) would skip a fraction of the 8 bigram rows on rare text — but this changes the model's input, so it is an approximation with **no published quality curve**. Low priority: 16 row gathers out of a 65 ms round is not where the time is.

Hash-and-collision facts worth knowing, since they are free to check in the checkpoint: collision counts **spike when the n-gram vocab is an integer multiple of the base vocab** (independent of primality), N=3..5 and K≥2 is the robust regime, and allocating >50% of parameters to n-grams is past the U-shaped optimum.

---

## 6. Speculative decoding parameters — the free-to-tune space

This is where the remaining money is, because it is **exact**: any verifier-accepted token is a token the target would have produced.

### 6.1 The theory you can compute on a napkin

Step time `V(k)`, geometric acceptance `a`, expected tokens `G_a(k) = 1 + a + … + a^k`, throughput `T_a(k) = G_a(k)/V(k)`. The best draft length `k*(a)` is a **monotone staircase** in `a` — provably non-decreasing, and each step point is exact; the results are machine-checked in Lean 4 (https://arxiv.org/html/2610.34440). Two consequences:

- Speculation beats plain decoding iff `a > V(1)/V(0) − 1`, i.e. iff the acceptance exceeds the *relative* cost of one extra row.
- On an MoE with a **per-slot** expert kernel, each draft token costs a full token's expert bytes, so `V(k)` is near-affine and long drafts never pay. On a **deduplicating** kernel, dedup is faster only where `ū(n) < r(n)` (byte share < bandwidth ratio); measured bandwidth ratios are r = 0.32–0.75, so **deduplication only wins from 5–8 tokens per pass, or never** **[measured]**. With a real draft model on llama.cpp/Metal, **plain decoding was fastest on 33 of 37 MT-Bench prompts**.

### 6.2 Your numbers

Calibrating `V(n)` to Cohere's measured `T(4)/T(1) = 1.25` (4 rows cost 1.25× one row) and charging draft cost at Cohere's measured 14.3% of a target step for 3 draft tokens **[inference, but both inputs measured]**:

| acceptance a | k*= | best speedup | k=1 | k=2 | k=3 | k=4 | k=6 | k=7 | k=8 |
|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 2 | 1.31× | 1.283 | **1.311** | 1.254 | 1.173 | 1.016 | 0.949 | 0.889 |
| **0.55** | **2** | **1.47×** | 1.371 | **1.469** | 1.451 | 1.387 | 1.228 | 1.152 | 1.083 |
| 0.65 | 3 | 1.69× | 1.460 | 1.644 | **1.687** | 1.660 | 1.525 | 1.446 | 1.370 |
| 0.75 | 4 | 2.01× | 1.548 | 1.834 | 1.966 | **2.005** | 1.945 | 1.882 | 1.811 |

(ceiling if verify were free: 1/(1−a) = 1.82 / 2.22 / 2.86 / 4.00)

**Read:** at a≈0.55 the optimum is **k=2, i.e. 3 rows**. Running 7–8 rows costs you **~22% of achievable throughput** (1.15× vs 1.47×) and buys nothing — `G_a` has saturated at 1/(1−a)=2.22 while `V` keeps climbing. Your observed "2–8 rows per pass" is a control loop hunting; replacing it with a fixed k=2–3 (or a k chosen from measured acceptance per request) should be worth 10–25% at zero quality cost. Also note the whole staircase shifts right as `V` flattens: every improvement that makes an extra row cheaper (dedup kernel, larger L2-resident union) moves k* up.

### 6.3 Published optima and where they disagree

- **[measured]** `Speculative Decoding for MoE: Expert Bytes, Kernel Bandwidth and the Best Draft Length` (https://arxiv.org/abs/2610.34440): the definitive treatment. Best k is within one of measured at 11/11 acceptance values on CUDA; drafting at predicted k* loses at most 1.6–7.7% of best throughput; a 10% bandwidth error moves the predicted speedup by ≤25%. **No deduplication benefit below 5 rows** on any of four engines.
- **[measured]** Cohere (production MoE): SD speedup is **non-monotonic in batch size** — peaks at moderate batch, unusually high at BS=1 because fixed launch overhead (2–10 µs, 2–3 µs of which is just launch) amortises over the 4-row pass. At BS=1 they see **1.95×** at accept length 2.73, with `T(4)/T(1) = 1.25` measured vs 1.46 predicted by Amdahl. Unique experts over verification: **1.25–1.42×**, stable across 13 task categories and 7 languages.
- **[measured]** `Why Speculative Decoding Works Better Than Predicted on Sparse MoE` (https://arxiv.org/html/2610.38981v1): across 10 sparse MoEs (30B–1T, 8–512 experts), fine-grained models **break** the concave speedup-vs-batch prediction and show *higher* speedup at **low** batch. At low batch, MoE layers are a much smaller share of runtime than of weights; profiling attributes this to higher MoE-kernel bandwidth utilisation vs. attention, with **L2 absorbing repeated reads of co-routed experts**. This is the best available explanation of why your 47 ms pass is not simply expert-bytes/bandwidth.
- **[measured]** EcoSpec (https://arxiv.org/html/2607.12696v1): confidence-driven draft trees cause **expert scattering** — high-probability draft tokens route to disjoint experts, inflating the verification union. Selecting draft paths that keep acceptance while **reusing already-covered experts** gives up to **1.62×** on Qwen3-235B, 1.50× on GPT-OSS-120B, 1.47× on DeepSeek-V3.1, without touching the verification rule. **This is the highest-value exact lever available to you** and it is pure tree-selection policy.
- **[measured]** Tree verification: on a fixed tree, max expected block length = 1 + target mass covered; TEV reduces verifier-stage latency 15% and ExitTrain improves block length 13% (https://arxiv.org/html/2610.11750). Applicable only if you run tree drafts.
- **[measured]** Tree shape vs. batch: at B=1 use a **wide** tree (topk=4) to exploit idle compute; at B=32 use a **narrow chain** (topk=1) because a wide tree triggers more MoE dispatches per verify (https://huggingface.co/blog/lujangusface/tw-eagle3-minimax). You are at B=1 — wide is indicated *if* your union stays small.

### 6.4 Two hard measurements from this exact model on this exact box

- **[measured]** An external vocabulary-compatible draft model (Qwen3.5-0.8B, same 248320 vocab) gave **no speedup at all** on the Spark: accepted length 2.88, decode unchanged at ~23 tok/s. Reason: on top-10-of-512, k tokens can activate up to k×10 different experts, so weight traffic scales with k and **there is nothing to amortize**. Only the MTP head (which shares the target's routing structure) and n-gram/span speculators help — `ngram-mod` reaches 50–60-token accepted spans and 3× on copy-heavy tasks, ~22 tok/s on prose (https://github.com/rumi-ali/qwen38-flash-next-spark/blob/main/docs/speculative-decoding.md).
- **[measured]** MTP 3/1/4 on SGLang/2-node: accept length **3.30–3.85** against a **hard ceiling of 4.0** (the QSA ring, §4.3), accept rate **0.66**, ~63 tok/s; throughput correlates with acceptance at **r = +0.786** — the run-to-run variance people chase thermals for is acceptance variance. Cumulative campaign figure: accept length 2.99, accept rate 0.66 (https://huggingface.co/randomllama/Qwen3.8-Flash-Next-DGX-Spark-field-notes).

**Your a=0.55 is below the 0.66 that stack reaches.** That gap is worth more than any approximation on this list: at a=0.66 the optimum moves to k=3 and the ceiling from 1.47× to ~1.7×. The levers that raise `a` here are drafter-side (MTP head fine-tuning — the one published recipe notes "the only learned change is a small LoRA on the MTP drafter, and the drafter affects speed only"), reduced draft vocabulary (a 65,536-token draft head against the 248,320 target vocab, scattered into the full vocab just before verify, took acceptance 55.7%→69.3% and decode 43→53 tok/s **[measured, single-rig]** https://huggingface.co/Vidyaganesh/Qwen3.8-Flash-Next-DGX-Spark-GB10), and sampling-parameter alignment (that stack's model default is T=1.0/top_k=20/top_p=0.95; a temperature-0 request is a different distribution than the drafter was tuned for).

---

## 7. What I would not spend a day on

1. **Route tail skip / routing bias** — measured at +1 tok/s and −4% facts on a resident NVFP4 pack. It is a fix for VRAM starvation, and you have solved VRAM starvation with the PLE mmap.
2. **Early-exit / self-speculative block skip without training** — the reference implementation says it is *slower than autoregressive* on untrained checkpoints, and the AR layer-skipping literature says 2 consecutive layers costs 25–65% of quality.
3. **KV eviction for speed** — attention is 1.4 ms of 47. For capacity, prefer PLE offload (+78% KV for −0.07% throughput).
4. **Top-1 routing** — no evidence it survives on a 512-expert top-10 model, and the ASET curve says the knee is at 3.3 experts on a 64-expert model.
5. **Compressing the PLE table further** — it is already off the GPU at zero cost; the reads you cannot remove are the 16 gathers.

## 8. What I would measure next, in order

1. Sweep k ∈ {1,2,3,4} at fixed a and log *actual* rows→time (your `V(n)` curve). Everything in §6.2 hinges on that one curve, and it is a 20-minute experiment. If `V` is flatter than 1.25 at n=4, k* moves up immediately.
2. Check whether IndexShare (frozen QSA top-k across draft steps) is active; if not, it is the largest free draft-side win and it does not change accept length.
3. Compute per-layer expert-skip KL offline (MoDES-style α⁽ˡ⁾) on ~1k calibration samples. It costs minutes and tells you which of the 48 layers would tolerate a 10→8 cut — and whether the answer is "none", which is the likely answer here.
4. Before any expert-dropping: measure the **union** of experts over your actual verification rows. If it is already 1.25–1.4× a single row (as Cohere measures), the entire expert-skipping family is capped at a few percent no matter how well it is calibrated.
