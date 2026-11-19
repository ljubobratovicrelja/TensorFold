# Precision allocation for an already-4-bit, QAD-trained MoE — what can still pay

**Scope.** TensorFold fork serving `nvidia/Qwen3.8-Flash-Next-FP4-Local` on DGX Spark (GB10, 128 GB unified, 234 GB/s measured). Experts already NVFP4 + QAD-distilled. Retraining out of scope. Question: which precision changes can still buy decode time or footprint, and which are dead ends.

**The constraint that shapes everything below.** On GB10 the only native low-precision GEMM paths are NVFP4 (E2M1, FP8 block scale per 16) and FP8 (E4M3). There is no 3-bit or 5-bit tensor-core path. Any tensor stored at 3/5/6 bits must be unpacked and dequantised to FP8/BF16 before the GEMM, so a sub-4-bit or intermediate-bit scheme buys **footprint only and usually loses bandwidth** (extra ALU, extra register pressure, activations promoted to a wider type). The realistic format ladder for this engine is therefore **{NVFP4, FP8, BF16}**, and every "mixed-precision allocation" question collapses to *which tensor sits at which of those three rungs*.

---

## 0. Where the time actually goes (from the measured ledger)

Decode step ≈ 5.3 ms head + 15–32 ms MLP + 4.9–8.6 ms hyper-connections + 9.2 ms qkv ⇒ **34–55 ms**, plus expert traffic inside the MLP term.

| Tensor | Current | Measured | Implied rate | Halving it would save |
|---|---|---|---|---|
| lm_head | BF16 1.27 GB | 5.3 ms | 240 GB/s (**at wall**) | **2.65 ms** (5–8% of step) |
| hyper-connections | BF16 1.26 GB | 4.9–8.6 ms | 73–131 GB/s | **2.5–4.3 ms** (4–13%) |
| qkv / dense attn | FP8 2.1 GB | 9.2 ms | 140 GB/s | 4.6 ms (8–13%) |
| experts (MLP) | NVFP4 | 15–32 ms | 170–185 GB/s (**73–79% of wall**) | 25% of bytes ≈ 3.8–8.0 ms, **but not achievable without a 3-bit kernel** |
| embeddings | BF16 1.27 GB | ~0 | row lookup | ~0 ms (footprint only) |
| n-gram PLE table | 28.8 GB | ~0 | row lookup | ~0 ms (**footprint only, −21.6 GB at Q4_0**) |

Two conclusions fall straight out of the ledger:

1. **The remaining bandwidth wins are in the tensors that are still BF16, not in the experts.** The experts already run at 73–79% of wall bandwidth in a 4-bit format with native tensor cores; they are close to done. The BF16 survivors (head 240 GB/s, hyper-connections 73–131 GB/s) are where bytes are still being wasted, and they are wasting *more* bytes than their share of parameters because nothing about them is compressed.
2. **Footprint and bandwidth are now separate problems.** The single largest footprint item is the 28.8 GB n-gram table (22% of the 128 GB budget), which costs essentially zero decode bandwidth because only the looked-up rows are read. Compressing it cannot make decode faster; it only buys room for KV/cache/concurrency.

---

## 1. Re-quantising an already-NVFP4/QAD checkpoint (double quantisation)

**What it is.** Dequantise the 4-bit weights exactly to FP16 (lossless — the grid is known) and fit a *new, narrower* quantiser (3-bit, or a 2-bit base + residuals) with GPTQ/AWQ/RTN, optionally with error feedback so the residual of layer *l* is pushed into layer *l+1*.

**Published evidence.**
- **QEP** (Fujitsu, NeurIPS 2025, [arxiv 2504.09629](https://arxiv.org/abs/2504.09629)) is the cleanest measurement of the mechanism: quantising only the first 10 blocks of Llama-2-7B and tracking `‖f_m(X) − f̂_m(X)‖²` shows **approximately exponential error accumulation per block, and continued growth through the *unquantised* blocks**. Layer-wise independent quantisation neither accounts for inherited error nor corrects it.
- **GPTAQ** ([2504.02692](https://arxiv.org/abs/2504.02692)) attacks exactly this: "asymmetric calibration" matches the quantised layer's output to the *full-precision* model output rather than to a shared input, explicitly absorbing error accumulated in previous layers.
- **"When Does Compensating Inherited Quantization Error Help?"** ([2610.16020](https://arxiv.org/abs/2610.16020)) is the honest counterweight: compensation is mean-square safe only under zero-mean rounding with an unbounded grid; **with a bounded codebook, exact layer-wise minimisers can give a *larger* output error**, and it already fails for two scalar linear layers. Empirically, half-strength compensation beat uncompensated rounding on two 8B models at 4 bits — but *no* rule for choosing strength generalised, and lowering loss on the calibration text did not reliably lower it on new text.
- **Error feedback**: FBQuant ([2501.16385](https://arxiv.org/abs/2501.16385)) bounds reconstructed weights and reports +1.2% zero-shot at 3-bit Llama-2-7B; error-feedback coefficient design is analysed in [2610.18139](https://arxiv.org/abs/2610.18139), which finds downstream-aware feedback helps but trace-optimality "need not improve finite-level performance".
- **Recurrent Residual Quantization** ([2608.04048](https://arxiv.org/abs/2608.04048)) is the most actionable framing: a 4-bit tensor *is* a 2-bit base plus 2-bit residuals. Storing base+residual and dropping residues gives sub-4-bit with a recoverable path back, calibration-free, "model-dependent behaviour at 4 bits".
- **Same-width re-encoding is nearly free.** AMD's online NVFP4→MXFP4 requantisation ([ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/nvfp4-to-mxfp4/README.html)) runs a *coarser* scale format (block 32, E8M0, no per-tensor scale) and reports "no meaningful accuracy loss" vs native NVFP4 — evidence that FP4 error is dominated by the 4-bit mantissa, not the scale, so a second FP4 rounding that lands mostly on the same value set is second-order. Double-quantisation *error* as a distinct phenomenon is documented for the consistent-scaling case in FP8-Flow-MoE ([MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/9cb295d4dce6db99f01e0dd512e46ba0-Abstract-Conference.html)): tensors quantised along different axes accumulate inconsistent scales and drift.

**Applicability to a QAD model — the key asymmetry.** A QAD checkpoint is *not* "BF16 weights plus noise". Distillation moved the weights to a local optimum of the **quantised** network: the residual error has already been absorbed into the weights, is structured rather than zero-mean, and there is no BF16 teacher on disk to calibrate against. Three consequences:
- Compensation terms derived assuming zero-mean rounding (GPTQ-family, QEP's closed form) are being applied outside their assumption set.
- A second quantiser **raises a loss floor that is already elevated**. QUASAR ([2608.13966](https://arxiv.org/abs/2608.13966)) documents that QAT/QAD models sit above the BF16 loss floor by construction; you are adding to a floor, not recovering from one.
- Re-running PTQ on the QAD checkpoint is what Unsloth already did — their Dynamic 3.0 notes for this model state plainly "we do NOT use QAT or QAD. Everything is done through post-training quantization" ([unsloth.ai/docs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)). So 4→~4 re-quantisation is a solved, low-risk operation; **4→3 is not**.

**Risk.** Medium-high for 4→3. No published case of dropping a QAD-trained NVFP4 MoE below 4 bits without distillation; and no 3-bit GEMM on GB10, so the win would be footprint-only while decode likely gets *slower*.

**Verdict.** At equal width, re-encoding is mechanical and safe (useful if we ever need MXFP4 for portability). Going below 4 bits on top of QAD: **footprint only, negative bandwidth, unproven quality. Dead end for the decode-time goal.**

---

## 2. Sensitivity-driven mixed-precision allocation

**What it is.** Assign bit-width per layer/tensor from a measured sensitivity score instead of uniformly. The Unsloth/qtum pattern: keep the router in F32, the shared expert in Q8_0, the sensitive dense path in BF16/Q8, and spend the freed budget on expert `gate/up`.

**Published effect.**
- **IMPQ** ([2509.15455](https://arxiv.org/abs/2509.15455)) frames allocation as a cooperative game (Shapley estimates of layer sensitivity *and inter-layer interaction*) and solves a binary quadratic program for 2-vs-4-bit. Gains: 20–80% PPL reduction vs the best isolated-metric baseline — **but the margin grows as average bits tighten**, i.e. the payoff is smallest at ~4 bits. That is precisely our regime.
- **MXSens** ([2607.17733](https://arxiv.org/abs/2607.17733)): triplet 4/6/8-bit mantissas, top-32 most-outlier columns per layer at 8-bit; at ~4.25 bits on Llama-2-13B it matches Atom on PPL *and* beats it 70.3% vs 63.5% zero-shot. Note the finding that survives translation: **the number of extreme-outlier columns per layer is typically < 32** — a small, per-layer set buys most of the quality.
- **The metric question is settled against weight error.** "Component Type, Not Reconstruction Error, Predicts Attention Quantization Sensitivity" ([2609.26173](https://arxiv.org/abs/2609.26173), 3,808 measurements, 9 models) finds relative reconstruction error explains **median R² = 0.044** of the perplexity sensitivity within a component type (and <10% of variance in 27 of 36 cases). Component type and layer identity explain more; **activation-weighted error is a moderately better within-component predictor for V projections (R² 0.20 vs 0.06)**. Conclusion: `‖W−Ŵ‖_F` is nearly useless as an allocation signal; per-tensor ΔKLD replay is the signal.
- **Real-world recipes for this exact model.** The qtum Qwen3.8-Flash-Next GGUF card ([HF](https://huggingface.co/qtum/Qwen3.8-Flash-Next-GGUF)) documents a working protection map: `ffn_gate_inp*` F32 (router), `ffn_*_shexp` Q8_0 (shared expert, "runs on every token"), GDN `ssm_*` F32, `hc_*` F32 (hyper-connections), `token_embd`/`output` Q6_K, and — critically — "the 51B PLE n-gram table tolerates heavy compression (**measured: Q8_0 → Q4_0 costs nothing**)", with the saved bytes *re-spent* on expert gate/up for ~14% better PPL at a smaller file.
- **Counter-evidence that protection maps are often wrong.** "Why Gated DeltaNet Survives 4-Bit Quantization" ([2609.04098](https://arxiv.org/abs/2609.04098)) replays one projection at a time at W4A4 on the Qwen3.8-27B hybrid (48 GDN + 16 attention layers) and finds **the two tensors every public recipe protects (the decay gate `a` and write-strength gate `b`) are the two *least* sensitive**: 11.0% and 8.5% GEMM error squash to 2.1% and 2.6% output error, while the plain GEMMs carry the error (out 12.7%, qkv 10.4%, z 9.9%). Errors are statistically independent across projections (combine in quadrature: 19.4% predicted vs 19.2% measured) and **do not grow along the sequence**.
- The complementary per-tensor KLD table for a sibling hybrid ([lyn.one recipe](https://lyn.one/unsloth-quantize-recipe), Unsloth Dynamic 2.0 port) puts `lm_head` at KLD ≈ 0.05 (safest tensor in the model), `embed_tokens` ≈ 0.15, q/k/v 1.5–2.9, `self_attn.o_proj` ≈ 1.5 but **un-correctable because no preceding norm layer**, and `linear_attn.out_proj` ≈ 6.0 — "worst tensor by far, cannot be corrected".

**Applicability.** The actionable allocation for us is *not* downward on experts (already 4-bit, 73–79% of wall) but: (a) compress the 28.8 GB n-gram table to Q4_0 — measured free, −21.6 GB footprint; (b) identify the one or two 4-bit tensors that could go to FP8/5-bit if we had the kernel — but we don't, so this is inert; (c) protect the router and shared expert (already FP8/BF16), which is cheap because with 512 experts they are a rounding error of total bytes.

**Risk.** Low for the n-gram move (measured on this model family). Medium for trusting any published protection map: the GDN paper shows intuition about "fragile" tensors is systematically wrong for hybrid/recurrent blocks, and our model's hyper-connections are protected by *intuition* in every public recipe with no measured sensitivity behind them.

**Verdict.** Allocation machinery pays little at 4 bits average, but **the diagnostic method pays immediately and cheaply**: a per-tensor KLD replay (96 replays was enough for a 27B) tells us whether hyper-connections are the 2.6%-squashed gate case or the 12.7% case. Do that before spending anything on them.

---

## 3. FP8 for the tensors still in BF16 (head, hyper-connections, embeddings)

**What it is.** Store BF16 tensors as FP8 E4M3 (per-tensor or per-row scales, FP32 accumulate) where the GEMM already runs in FP8, or dequantise to BF16 on the fly if not.

**Published evidence — and a citation trap.** The most-quoted result, "Give Me BF16 or Give Me Death" (Red Hat AI / ISTA, [ACL 2025](https://aclanthology.org/2025.acl-long.1304.pdf)), concludes **W8A8-FP is "effectively lossless"** across the whole Llama-3.1 family over 500k+ evaluations. Its Limitations section, however, states the study left open "the impact of compressing other model components such as **the KV-cache, input embeddings, and language modeling head**". So the lossless claim **does not cover the head, embeddings, or KV** — those are exactly the tensors we are considering. Anyone citing it as licence to quantise `lm_head` is misreading it.
- **lm_head specifically.** ARCHead ([2608.02703](https://arxiv.org/abs/2608.02703)) is the direct study: naive low-bit head quantisation minimises a weight-space error that "treats every hidden direction equally" while the final hidden states are strongly anisotropic; head error reaches the softmax with no norm, residual, or learned transform to absorb it. Numbers: at **25.6% of BF16 head storage it reaches 1.007 relative PPL**, whereas **storage-matched naive INT4 gives 1.14–1.16**; replacing the BF16 head with AWQ or bitsandbytes costs only **0.006–0.007 cross-entropy with <2% throughput change**. It also notes that AWQ and bnb-NF4 checkpoints routinely *leave* `lm_head` in BF16 — the head becomes one of the largest remaining dense tensors.
- **Ecosystem default.** llm-compressor's INT4 W4A16 recipe ships `GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])` ([docs](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/int4/)) — i.e. the tooling's default is *don't*, because heads are cheap relative to blocks in normal models and quantised heads have broken vLLM in practice. Both reasons are weaker here: the head is 5.3 ms of a 34–55 ms step, and we control the kernel.
- **Hyper-connections and embeddings**: no published measurement exists for hyper-connection projections. The only data is the qtum protection map (F32, "every layer") which, per §2, is intuition. Embeddings in the sibling hybrid are KLD ≈ 0.15 at Q5 — the second-safest tensor.

**Applicability.** Head: 1.27 GB BF16 → 0.64 GB FP8 halves a 5.3 ms wall-bound stage ⇒ **≈2.65 ms, 5–8% of the step**, for a change whose measured analogue (AWQ/bnb head) costs 0.006–0.007 CE. This is the best risk-adjusted item in the whole list. Hyper-connections: 1.26 GB → 0.63 GB ⇒ **2.5–4.3 ms, 4–13%**, same mechanism, unknown quality — and because the stage currently runs at 73–131 GB/s (well below wall) part of its cost may be launch/dequant-bound, in which case the win is smaller than the byte ratio suggests. Embeddings: **footprint only** (~0.6 GB), zero decode win, since only the current token's rows are read.

**Risk.** Head FP8 with per-row scales + FP32 accumulate: low. Head FP4: needs ARCHead-style activation-metric residual, otherwise 14–16% PPL. Hyper-connections: unknown shape of the risk (a per-layer multiplicative gate is the worst place for a rounding error to sit) — measure first.

**Verdict.** **Real bandwidth win, the only unambiguously real one left in the BF16 set.** Head first (best evidence, at-wall stage), hyper-connections after a KLD replay.

---

## 4. KV-cache precision

**What it is.** Store K/V in FP8 (or INT4/2-bit) instead of BF16; per-tensor, per-head, per-channel (K) or per-token (V) scaling.

**Published evidence.**
- **KIVI** ([2402.02750](https://arxiv.org/abs/2402.02750), ICML 2024) established the axis result from measured element distributions: **keys should be quantised per-channel, values per-token**, delivering 2-bit KV at "almost the same quality" with 2.6× peak-memory reduction. **KVQuant** ([2401.18079](https://arxiv.org/abs/2401.18079)) adds the mechanism: **keys show outliers in specific channels before RoPE, and the outlier channels become *less* consistent after RoPE** — which is why per-channel pre-RoPE handling and dense-and-sparse outlier isolation are needed at 3-bit; it reports <0.1 PPL degradation at 3-bit.
- **vLLM's FP8 KV validation** ([vllm.ai blog 2026-04-22](https://vllm.ai/blog/2026-04-22-fp8-kvcache)): near-baseline accuracy, **per-head scales** now supported, decode per-token KV cost down to ~54% of BF16; it also documents a hardware trap — FP8 Tensor Core accumulation lost precision at long context, dropping a 128k needle task from 91% to **13%**, fixed only by a two-level accumulation writing to a real FP32 register. Two-level accumulation then costs prefill speed at `head_dim=256`.
- **The interaction term that matters for us.** The GDN paper ([2609.04098](https://arxiv.org/abs/2609.04098), §7) measures FP8 KV on a **W4A4** model: task scores don't move, capacity grows 1.8–1.9×, but *"the one systematic cost is perplexity at 32K: +0.13 for BF16 and +0.41 for Minima — 3× larger, plausibly because Minima's K/V projections are already W4A4, leaving the values less headroom before the cache rounds them again."* Shipping **static calibrated per-tensor FP8 scales** (32 tensors on 16 attention layers) drops PPL@32K 10.84 → 10.50, **recovering 83% of the penalty**, with throughput within 0.4% — "performance-free". That is a direct measurement of double-quantisation interaction on a 4-bit-quantised model, and it says: **the second rounding is 3× more expensive than it would be on a BF16 model, and calibrated scales fix most of it.**
- Community signal for this model family: "Qwen3.8-27b q8 KV cache does seem to actually hurt model performance" ([r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1w0pscn/qwen3827b_q8_kv_cache_does_seem_to_actually_hurt/)) — anecdotal but consistent with the 3× interaction.

**Applicability.** Our model is hybrid: only the QSA attention blocks hold a KV cache, so total KV bytes are structurally smaller than a full-attention model's. At short context that makes the decode-time win ≈0, because KV traffic is a small fraction of the step. The win is at 32K+ and in concurrency/footprint. **Per-channel/per-token schemes (KIVI) buy more at 2-bit; at FP8 with calibrated scales the per-tensor result is already free**, and per-channel scaling would break the fused attention kernel.

**Risk.** Low *if* calibrated scales ship and evaluation happens at 32K, not 4K. High if uncalibrated — that is precisely the documented 3× failure mode.

**Verdict.** **Footprint/concurrency win, not steady-state decode.** Cheap and near-free at FP8 with calibrated scales; skip per-channel work unless going below 8 bits.

---

## 5. Fitting the quantiser to output error at small scale

**What it is.** Replace "minimise weight error" with "minimise the error in this layer's *output*, computed with the upstream model already quantised": `min ‖W_l X_l − Ŵ_l X̂_l‖²`. GPTQ does the Hessian-weighted version of this with a **full-precision** `X`; QEP ([2504.09629](https://arxiv.org/abs/2504.09629) Eq. 3–4) replaces `X` with `X̂` from the quantised prefix and adds a closed-form correction `W_l δ_l X̂_lᵀ Ĥ_l⁻¹` where `δ_l = X_l − X̂_l` is inherited error; GPTAQ ([2504.02692](https://arxiv.org/abs/2504.02692)) does the same thing under the name asymmetric calibration, in ~20 extra lines over GPTQ.

**Published effect.** QEP's own table (Llama-2 7B/13B/70B, Llama-3-8B, Mistral-7B, WikiText-2):

| | INT4 off | INT4 + QEP | INT2 off | INT2 + QEP |
|---|---|---|---|---|
| Llama-3-8B AWQ | 7.108 | **6.981** | — | — |
| Llama-3-8B GPTQ | 147.912 | **9.509** | — | — |
| Mistral-7B QuIP | 11.109 | **5.479** | — | — |
| Llama-2-7B GPTQ | 6.083 | **5.933** | 10.881 | **7.898** |

Read honestly: at INT4 the gain is **1–2.5% PPL**, i.e. small; the dramatic numbers are all at INT2/INT3, or pathological cases (Llama-3-8B GPTQ 147.9 → 9.5 and Mistral QuIP 11.1 → 5.5 are broken-baseline rescues, not 50% improvements). The paper says so itself: gains are "most pronounced in the extremely low-bit regime".

**Hessian-free variants** (relevant because we do not want to hold per-layer Hessians for an 180B MoE):
- Activation-weighted **diagonal** second moments: measured R² 0.20 for V projections vs 0.06 for reconstruction error ([2609.26173](https://arxiv.org/abs/2609.26173)) — cheap and clearly better than weight MSE.
- **KLD replay** per tensor, the Unsloth/imatrix signal: 150 KLD benchmarks over 121 configurations produced their allocation table ([lyn.one](https://lyn.one/unsloth-quantize-recipe)). Costs a forward pass per tensor, no Hessian.
- **ARCHead** for the head: fit a low-rank correction *after* an activation-derived metric transform, provably "the best rank-r approximation in that metric before factor quantisation" ([2608.02703](https://arxiv.org/abs/2608.02703)) — the cheapest way to get an output-aware 4-bit head.
- **Error feedback** as the sequential approximation: FBQuant ([2501.16385](https://arxiv.org/abs/2501.16385)) bounds the reconstruction and gains +1.2% zero-shot at 3-bit; the analysis in [2610.18139](https://arxiv.org/abs/2610.18139) shows downstream-aware feedback coefficients help in synthetic/MLP settings but trace-optimality doesn't transfer to finite-level performance.

**Applicability and the honest caveat.** Per-tensor sequential fitting with the rest of the model already quantised is *exactly* what QEP/GPTAQ formalise, so the method is available and cheap (one forward pass to collect `X̂`, one closed-form solve per tensor). But [2610.16020](https://arxiv.org/abs/2610.16020) is the paper that should temper expectations: with a **bounded codebook** (which FP4 and FP8 both are), even exact layer-wise minimisers can yield a *larger* output error; the guarantee needs zero-mean rounding on an unbounded grid, which a QAD checkpoint's structured residual error does not satisfy. And empirically, on two 8B models at 4 bits, improving full-network loss on the calibration data "did not reliably lower it on new text" — the calibration gain is partly overfit.

**Risk.** Low cost, uncertain payoff, real overfitting risk that must be checked on held-out text rather than the calibration set.

**Verdict.** **Worth doing only where a tensor is actually being re-quantised** (head to FP4 via ARCHead; if KV ever goes below FP8). It cannot help tensors we leave at NVFP4 — there is nothing to fit. At the 4-bit operating point expect 1–2%, not the INT2 headline numbers.

---

## Verdict — ranked by expected decode-ms per unit of risk

| # | Change | Win | Type | Evidence | Risk |
|---|---|---|---|---|---|
| 1 | **lm_head BF16 → FP8** (per-row scale, FP32 acc) | **2.65 ms (5–8%)** | bandwidth | ARCHead: AWQ/bnb head +0.006–0.007 CE; naive INT4 head 1.14–1.16 rel PPL | Low |
| 2 | **Hyper-connections BF16 → FP8**, after a KLD replay | **2.5–4.3 ms (4–13%)** | bandwidth | none published; qtum protects by intuition, GDN paper shows such intuitions invert | Medium (measure first) |
| 3 | **n-gram PLE table → Q4_0** | **−21.6 GB footprint**, 0 ms | footprint | qtum, measured on this model: "Q8_0 → Q4_0 costs nothing" | Very low |
| 4 | **FP8 KV with calibrated static scales** | long-context + concurrency; ~0 at 4K | footprint/latency | GDN §7: 3× penalty on W4A4, calibrated scales recover 83%, throughput free | Low if calibrated + evaluated at 32K |
| 5 | **Output-aware refit (ARCHead/QEP) on any tensor actually re-quantised** | 1–2% PPL at 4 bits | quality | QEP/GPTAQ gains concentrate at INT2/3; 2610.16020 shows bounded-codebook limits | Low cost, overfit risk |
| 6 | Embeddings → FP8 | ~0.6 GB, 0 ms | footprint | KLD ≈0.15 | Very low, near-worthless |
| 7 | **Experts 4 → 3 bit** | footprint only; **decode likely slower** (no 3-bit GEMM on GB10) | — | no published case below 4-bit on a QAD MoE | High — **do not** |
| 8 | More mixed-precision at 4-bit average | ~0 | — | IMPQ: gains shrink as bits tighten; recon-error R²=0.044 | — dead end |

**Bottom line.** After QAD at 4 bits the expert weights are done: 73–79% of wall bandwidth in a native format, and every step away from 4 bits costs kernel support we do not have. What is left is not *re*-quantisation of the 4-bit model but **finishing the quantisation that was never done** — the head and hyper-connections are still full BF16 and are burning 2.65 + 2.5–4.3 ms of a 34–55 ms step doing nothing with the extra precision, and the 28.8 GB n-gram table is a measured-free 21.6 GB of footprint. Realistic ceiling from precision work alone: **~8–20% decode improvement plus ~22 GB of footprint, essentially all of it from three tensors that are currently uncompressed.** Everything below 4 bits on the expert path is a footprint story with a likely latency regression.

**Two measurements to run before committing (both cheap):** (a) per-tensor KLD replay for the hyper-connection and GDN/QSA projections under the engine's own quantiser, to replace every published protection map with our numbers; (b) a held-out-text check on the head after any FP8/FP4 change, because ARCHead's cross-entropy deltas are measured on Qwen3-8B-Base, not on a QAD 4-bit MoE where the head is the last full-precision path to the softmax.
