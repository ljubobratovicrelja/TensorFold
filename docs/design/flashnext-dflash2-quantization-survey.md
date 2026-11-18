# Quantization for a bandwidth-bound DFlash2 drafter on GB10 — what is actually measured elsewhere

Status: **survey, no local experiments.** Nothing in §7 has been run on this box. Everything in §3–§6
is other people's measurement or vendor documentation, cited. §2 is our own measured baseline and is
kept separate on purpose so the two do not get confused.

Purpose: decide whether weight-precision work on the drafter body, the `fc` block, or the target head
buys more than the kernel work already queued in `flashnext-dflash2-gb10-optimisation-plan.md` §6, and
at what acceptance risk.

Evidence grades used throughout:

- **[M-GB10]** measured by the author on a DGX Spark / GB10 or a GB10 OEM box.
- **[M-other]** measured by the author on different silicon (RTX 5090, RTX PRO 6000, 4090, GB300, Apple).
- **[V]** vendor documentation or vendor claim, not an independent measurement.
- **[P]** paper, typically on hardware unlike ours.

---

## 1. Method and what the search actually returned

Searched: NVIDIA developer forums (DGX Spark / GB10 board), the `ggml-org/llama.cpp` DGX Spark
performance thread, HF model cards and community blog posts, arXiv (quantization + speculative
decoding), vendor documentation for TensorRT-Edge-LLM and vLLM, and the GB10 owner-blog cluster
(ai-muninn, gpuwerk, kubesimplify, jschmied, sergioamsilva field notes).

Two things about the evidence base are worth stating before anything else.

**The GB10 owner community is unusually good on this exact question, and unusually bad on precision.**
There are dozens of measured decode numbers per format on this box (§3), and the measurement culture is
healthy — most posts state n, the engine, the kernel actually selected, and whether CUDA graphs were on.
But almost nobody measures *acceptance length* after quantizing a drafter. Of the corpus, four sources
report acceptance: two DFlash2-specific, two on adjacent architectures (§6.1). Treat any claim of
"no quality loss" from a decode tok/s table as unmeasured.

**A large fraction of published GB10 numbers are kernel artefacts, not format effects.** The same
weights, same box, same file on disk have been measured 70× apart depending on whether the engine
picked a native FP4 kernel or fell back to emulation ([field notes](https://github.com/sergioamsilva/dgx-spark-field-notes/blob/main/docs/quantization.md),
[Avarok](https://github.com/Avarok-Cybersecurity/dgx-vllm/blob/main/NVFP4_BREAKTHROUGH_DGX_SPARK.md)).
Any format comparison that does not name the selected kernel is not evidence.

---

## 2. Our baseline (measured here, not re-derived)

Served drafter round: **22.99 ms** median (16 reps, `fixture/sequence`); a second capture reads 23.80 ms
and the ~3% is clock state. Stages:

| stage | ms | share | weight bytes moved | implied BW |
|---|---:|---:|---:|---:|
| `logits_top16_ids` — target head, 7 rows x 248,320 | 6.98 | 30.4% | 1,271.4 MB | 182 overall |
| — of which GEMM | 5.43 | 23.6% | 1,271.4 MB | 234 |
| — of which top-k | 1.39 | 6.0% | — | — |
| — of which merge | 0.19 | 0.8% | — | — |
| `fc_ctx` — taps → 12800 → `fc` | 5.10 | 22.2% | 65.5 MB | **13** |
| attention, 5 layers | 4.32 | 18.8% | activations + KV | n/a |
| mlp, 5 layers | 2.75 | 12.0% | 589.8 MB | 214 |
| selector walk (`proposals`) | 1.29 | 5.6% | 255.6 MB | 198 |
| convs | 0.87 | 3.8% | 33 MB | 38 |
| qkv, 5 layers | 0.84 | 3.7% | 157.3 MB | 187 |
| attn proj, 5 layers | 0.54 | 2.3% | 104.9 MB | 194 |
| **total** | **22.69** | | 1.21 GB drafter + 1.27 GB head | |

Acceptance: **3.34** tokens/round at block width 8 (vendor card 3.26). Per-slot
0.740 / 0.662 / 0.592 / 0.621 / 0.389 / 0.429 / 0.333.

Two structural facts that bound everything below, and that the rest of this document keeps running into:

**(a) Half the round is not weight bytes.** Weight-read stages (head 6.98, mlp 2.75, selector 1.29,
qkv 0.84, proj 0.54) total **12.40 ms**. Activation/latency-bound stages (`fc_ctx` 5.10, attention 4.32,
convs 0.87) total **10.29 ms**. No quantization scheme, at any bit width, touches the second group.
The maximum a perfect weight-compression scheme could remove is 12.40 ms, and realistically ~30% of it.

**(b) The drafter is 32% of the served round.** DFlash2 w8 measures 45.0 tok/s at a **74.2 ms** round
(`flashnext-dflash2-gb10-optimisation-plan.md` §1), of which the drafter is ~23 ms and the target verify
is ~50 ms. So 1 ms off the drafter is worth ~1.35% of end-to-end tok/s. A 5 ms drafter win is
45 -> ~48 tok/s, not 45 -> 60. This ratio, not the stage table, is the number to rank options against.

The served round treats the head as part of the drafter: it is the *target's* `lm_head` applied to 7
draft rows, so its 1.27 GB is paid inside the 23 ms we are trying to shrink.

---

## 3. What GB10 owners actually do about bandwidth

### 3.1 The box is at ~230–240 GB/s sustained, not 273

The datasheet number is 273 GB/s. Every measurement I found lands lower, and consistently:

- "128 GB unified LPDDR5x memory, **~238 GB/s measured sustained bandwidth (idle)**" — veloGB10, a
  from-scratch Rust+CUDA engine written for sm_121 ([README](https://github.com/sf-stav/veloGB10)).
- "the drafter is 15.5% of the bytes moved per step on a machine measured at **231.8 of 273 GB/s**" —
  an FP8 DFlash2 drafter writeup on GB10 ([josch15366](https://huggingface.co/josch15366/Qwen3.8-27B-DFlash2-FP8)).
- Baseline AR decode on GB10 measured at **94% of the ~12 tok/s roofline** implied by 22 GB of NVFP4
  weights and 273 GB/s ([madeye](https://madeye.github.io/qwen3.8-dflash2-gb10/)).

Our stages sit at 187–234 GB/s, i.e. we are at the same wall and marginally above the mean the other
owners report. There is nothing left in "use the bus better" for the weight-streaming stages.

### 3.2 The speed comes from bytes, not from the FP4 tensor cores

The cleanest measurement on this box, on a pure dense model with all confounds removed: FP8 W8A8
**25.65 tok/s**, NVFP4 W4A4 **38.59**, **NVFP4 W4A16 40.85** — and W4A16 wins by *dequantizing the FP4
weights to BF16 and never touching the FP4 ALU*. "The path that uses the fancy FP4 compute unit loses
to the path that ignores it and treats FP4 as a smaller file."
([ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute))

That is the single most decision-relevant measurement in this survey for us: our stages are GEMV-shaped
(7 rows for the head, 7-ish rows for the drafter), we will never be compute-bound, and therefore the
kernel we want is the one with the lowest dequant overhead per byte, not the one with the fastest MMA.
It also means W4A16 is the right activation choice for a 7-row GEMM — quantizing activations buys
nothing we need and costs accuracy.

### 3.3 SM 121 is the odd part, and the failure mode is the kernel

GB10 is `sm_121`, not `sm_120` (B200/GB200 and RTX 5090 are 12.0). It lacks the
`cvt.rn.satfinite.e2m1x2.f32` instruction that NVFP4 conversion needs; vanilla vLLM therefore fell back
to an emulation path and measured **1.1 tok/s**, 30× slower than it should have been
([Avarok](https://github.com/Avarok-Cybersecurity/dgx-vllm/blob/main/NVFP4_BREAKTHROUGH_DGX_SPARK.md)).
With a FlashInfer CUTLASS kernel the same weights run at 77 tok/s — "same silicon, same file on disk,
70x apart. The format was never the problem" ([field notes](https://github.com/sergioamsilva/dgx-spark-field-notes/blob/main/docs/quantization.md)).

The 1.1 tok/s figure is still being repeated as "NVFP4 is a trap on GB10". The corrected reading, from
the same author who published the trap claim and then retracted it, is that the missing instruction costs
*compute*, and single-stream decode is not compute-bound ([ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute)).
For a drafter this matters twice: our engine is hand-written Zig+CUDA (§`tf_fxd2_*` kernels), so we
select our own dequant path and are not exposed to the vLLM fallback — but we also would not inherit a
fixed kernel for free, and would be writing the FP4 dequant ourselves.

**Practical rule others converged on:** "FP4 speedup ∝ the weight bytes it actually shrinks"
([ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute)). A hybrid or a
partially-quantized model dilutes the win in exact proportion to the bytes left in BF16.

### 3.4 Format comparison, measured on GB10

Same model (Qwen3-30B-A3B), one box ([field notes](https://github.com/sergioamsilva/dgx-spark-field-notes/blob/main/docs/quantization.md)):

| Format | Engine | bits | single-stream |
|---|---|---:|---:|
| Q4 GGUF | Ollama | 4 | 85.2 tok/s |
| Q8 GGUF | Ollama | 8 | ~62 |
| NVFP4 | vLLM (FlashInfer CUTLASS) | 4 | 77.1 |
| FP8 | vLLM | 8 | ~53 |

"Within each engine, the 4-bit format beats the 8-bit one by ~45%. That is the bandwidth model working
as predicted." Independent cross-check on a different model and engine set: Qwen3-Coder-30B-A3B at
BF16 30.6–31.7, FP8 50.5–55.0, NVFP4 65.0–66.0, GGUF Q8_0 61.1, AWQ 4-bit 80.9
([gpuwerk](https://gpuwerk.com/dgx-spark/guides/dgx-spark-quantization-speed)). The AWQ number being
highest and the AWQ single-stream ranking being the *worst* in another harness
([field notes](https://github.com/sergioamsilva/dgx-spark-field-notes/blob/main/docs/quantization.md))
is the clearest demonstration that these tables rank kernels at least as much as formats.

The one consistent quantitative law across every source: **decode tok/s scales with 1/(bytes per step)**,
with a ~0.4–0.5 rescale because not all bytes shrink. Qwen3.8-27B, identical flags, 16 concurrent:
FP8 30.9 GB -> NVFP4 23.4 GB is a 1.32× byte ratio, and measured throughput gain was **1.29–1.34×**
([gpuwerk](https://gpuwerk.com/dgx-spark/guides/dgx-spark-quantization-speed)). That is about as close
to a validated predictive model as this literature gets, and it is the model our §7 table uses.

### 3.5 Whole-system recipes on this box

For completeness — none of these change the analysis, but they set the envelope of what a GB10 owner
gets today.

- **veloGB10** — a purpose-built sm_121 engine (Rust + CUDA), NVFP4/FP8 and EXL3 packs, its own DFlash2
  support (`--spec-source dflash2-auto`). Qwen3.8-27B NVFP4 + DFlash2: ~40 tok/s single node average,
  ~70 tok/s sustained on code; Qwen3.8-Flash-Next from EXL3 3.05 bpw runs at full 262k on one node
  ([README](https://github.com/sf-stav/veloGB10)). Its own bandwidth claim (238 GB/s) is from the same
  document.
- **NVFP4 Qwen3.8-Flash-Next on 1/2/4 Sparks**, NVIDIA's official quant, upstream vLLM: median 32.5 tok/s
  single stream (peak 43.8), 194 aggregate at 6 streams on one box. Its recipe keeps the 47.7 GB n-gram
  table on disk and preads only the rows needed
  ([NVIDIA forum](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-on-1-2-and-4-dgx-sparks-with-nvidias-official-nvfp4-quant-64-tok-s-peak-single-stream/382476)).
- **GGUF `UD-IQ1_S` at 3.28 effective bits** is the only Flash-Next build that fits one Spark
  (72.5 of 121 GiB), 34.5 tok/s single stream, at a measured perplexity cost
  ([kubesimplify](https://blog.kubesimplify.com/running-qwen3-8-flash-next-on-dgx-spark-and-rtx-pro-6000)).
- **NVIDIA's own NVFP4 checkpoints do not quantize the head.** Both official Flash-Next checkpoints
  carry a `modules_to_not_convert` list; only the routed experts are quantized, and "attention, GDN, QSA,
  shared experts, routers, `lm_head`, embeddings, the vision encoder and the MTP head all stay in BF16"
  ([kubesimplify](https://blog.kubesimplify.com/running-qwen3-8-flash-next-on-dgx-spark-and-rtx-pro-6000)).

That last point is the vendor default, and §6 is about the two independent people who went against it
and measured what happened.

---

## 4. Quantization schemes, and which are real on this part

### 4.1 The formats, ranked by fit to a 7-row weight-streaming GEMM

**W4A16 / NVFP4 weight-only (4.5 bits/weight effective).** Best byte reduction available with a
supported kernel. Block-16 scales, E4M3 block scale, FP32 second level
([NVFP4 QAD report](https://arxiv.org/html/2601.20088v3)). On GB10 the fastest measured single-stream
path is W4A16 — dequant to BF16, plain BF16 matmul, FP4 used purely as storage
([ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute)). Practically: 1.27 GB
head -> ~358 MB; 589.8 MB MLP -> ~166 MB. Cost: every one of the 7 rows pays the dequant, and on 7 rows
that dequant is not amortised — this is the one place where a 4-bit kernel can lose to fp8 despite
moving fewer bytes, and it is worth measuring rather than assuming.

**FP8 e4m3 W8A8, per-channel weights + dynamic per-token activations (4.5 bits/weight with scales, ~8
nominal).** The best-supported, lowest-risk halving. On this box, measured on the actual DFlash2 drafter
family (§6.1), it delivered **+6.1% decode at literally unchanged acceptance length**. Also the format
TensorRT-Edge-LLM defaults EAGLE drafts to. Nominal 8 bits but 4.5-bit-scale overhead is nil, so
byte saving is a clean 2×.

**INT8 W8A16 group-128.** Measured head-to-head against FP8 on GB10 with the same drafter:
43.2 vs 43.8 tok/s and acceptance 4.16 vs 4.24 — "at this sample size the two schemes are not
distinguishable… Pick either" ([josch15366](https://huggingface.co/josch15366/Qwen3.8-27B-DFlash2-FP8)).
The expectation that FP8's native Blackwell support would beat Marlin-routed INT8 was **not** confirmed.

**MXFP4 (E2M1, block 32).** One citation of note: drafts quantized by *direct cast* of the BF16 weights,
no calibration, no retraining, used as plug-and-play speculative drafts, "up to 2× over the BF16
baseline" and 2.72× when nested ([ML-SpecQD](https://arxiv.org/abs/2503.13565)). That is a paper on
server hardware, but it is the strongest statement in the corpus that a naively-cast 4-bit draft works.

**EXL3 trellis 5.0 bpw.** Module-adaptive, includes the candidate selector in the quantized set. On GB10
it cut a DFlash2 drafter 3.85 -> 1.4 GB and raised end-to-end decode **+33%** with acceptance at parity
([Mia-AiLab](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw)). Caveat: it requires an
exllamav3 fork, and 5 bpw is not a byte count we can express in our own kernels without porting a trellis
codec. Read it as an upper bound on what a well-tuned ~5-bit drafter achieves, not as a recipe.

**Q2_K GGUF (2-bit).** A 2-bit DFlash2 drafter measured at **2.81 vs 2.80** mean accepted length at K=3,
60.22% vs 60.00% acceptance ([analogalok](https://huggingface.co/analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF)).
I weight this low despite it being the most dramatic claim in the corpus: K=3 (ours is 8), one harness,
one model, and the card's framing is about 24 GB VRAM capacity rather than precision. It is nonetheless
evidence that block-diffusion drafters degrade slowly, which is the §6.1 pattern.

### 4.2 What the vendor recipe actually does

TensorRT-Edge-LLM has a first-class command for exactly our model class:

```
trtedge quantize draft \
  --base-model-dir /path/to/base_model \
  --draft-model-dir /path/to/dflash_draft \
  --output-dir /tmp/dflash_draft_nvfp4 \
  --quantization nvfp4 \
  --lm-head-quantization nvfp4
```

From the same page, the rules that matter to us ([NVIDIA docs](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html)):

- "DFlash draft quantization has been validated with NVFP4. **The target-hidden projector stays in dense
  precision**; export will reject checkpoints where that `fc` module is quantized." — that is our 65.5 MB
  `fc`, and the vendor rule is *do not quantize it*.
- `--keep-last-layer-fp16` "leaves the last layer's `o_proj` and MLP projections (`gate_proj`, `up_proj`,
  `down_proj`) unquantized… it applies to `llm` and to DFlash-family `draft` quantization". Stated reason:
  a 4-bit GEMM writes FP16 output, so a last layer whose activations exceed 65504 saturates and "accuracy
  collapses".
- For **EAGLE-4** drafts, which have "a low-rank candidate head": `fc` stays FP16, "the KV cache and the
  **candidate head factors also stay FP16**… with a candidate head, **`lm_head` stays FP16** and
  `--lm-head-quantization` is rejected too."
- Calibration uses `cnn_dailymail` or `wikitext` by name; the base model is needed at calibration time
  to produce target hidden states, and as an LM-head fallback.

The EAGLE-4 paragraph is the closest vendor statement to our architecture: a drafter with a candidate
head keeps both the candidate head and the `lm_head` at FP16, even in a tool whose whole purpose is
quantizing drafters. That is a design opinion from the people who wrote the kernels, and it is
consistent with §6.4.

Note also that for EAGLE drafts the default is **fp8** and for DFlash drafts **nvfp4** — same tool,
different default per architecture.

### 4.3 Per-component support matrix (vendor, verbatim)

| Component | Supported methods |
|---|---|
| Backbone | `fp8`, `int4_awq`, `nvfp4`, `mxfp8`, `int8_sq` |
| LM head | `fp8`, `int4_awq`, `nvfp4`, `mxfp8` |
| Token embedding | `fp8`, `nvfp4` |
| KV cache | `fp8` |
| DFlash draft | validated with `nvfp4` backbone and optional `nvfp4` LM-head |

So `lm_head` quantization is a supported, vendor-blessed operation for a DFlash draft — the question is
not "can it be done" but "does it change the top-16 set", which §6 answers.

---

## 5. Selective and mixed precision: which parts tolerate it

This is the weakest-evidenced area in the corpus and the one most relevant to us, because our real
decision is not "which format" but "which of seven stages to touch".

### 5.1 What people protect, and the stated reasons

Converging list across sources:

1. **`fc` / target-hidden projector — protected, emphatically.** TensorRT-Edge-LLM refuses to export a
   DFlash draft with a quantized `fc` ([V](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html)).
   Every published DFlash2 quant keeps it out of the quantized set: FP8 build quantizes 2-D weights but
   leaves convs/selector/norms; NVFP4 build quantizes 36 matrices *including* `fc` but keeps convs and
   selector.
2. **Candidate selector and codebooks — protected in 2 of 3 published DFlash2 quants.** FP8 build:
   "deliberately left in BF16: the two-tap convolutions, the `candidate_selector` and its codebooks, and
   all norms. Together they are 0.39 GB — the saving is small and they are the parts that make
   block-diffusion drafting work." EXL3 5.0 bpw is the exception and does include the selector.
3. **Last decoder layer — protected by option.** `--keep-last-layer-fp16`, on FP16-range grounds, not
   sensitivity grounds ([V](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html)).
4. **Embeddings — freely quantized.** `fp8` (one FP32 scale per table) or `nvfp4` (per-token, per-16-hidden
   block) both supported and consumed directly by the lookup kernel.
5. **FFN vs attention — attention is the more fragile one at short context.** A measured mixed-precision
   study on Gemma-3-1B with TensorRT-LLM found that for FFN and `lm_head` "the time costs of
   quantization/dequantization procedures are compensated by the introduction of integer arithmetic",
   but for Attention "at short context lengths the reverse is observed — the addition of another
   quantization step leads to slower model execution". Best configuration under a quality constraint was
   FFN-only + `lm_head` at 11.0% latency reduction with 98.90% top-1 agreement; the more aggressive
   all-FFN variant reached 19.1% only by accepting visible quality loss
   ([arXiv 2608.28003](https://arxiv.org/html/2608.28003v1)). That is [P], different model, different
   device — but it is the only source that separates FFN, attention and `lm_head` contributions with
   measurements, and its direction (FFN and head are the good targets, attention is not) matches what the
   DFlash2 quants chose.
6. **`down_proj` vs `gate`/`up_proj` — no evidence found that separates them.** Every source treats the
   three MLP projections as one group. I could not find a single measurement distinguishing them, in any
   model, on any hardware. Anyone claiming down_proj is the sensitive one is asserting, not measuring.

### 5.2 The one real layer-sensitivity result I found

The Gemma-3-1B study reports a per-block sensitivity profile: FFN block normalised sensitivity
Q = 0.796 (mean SQNR 29.70 dB) and the embeddings "surprisingly robust" at Q = 0.819 (45.61 dB), "with
other layers staying in between" ([arXiv 2608.28003](https://arxiv.org/html/2608.28003v1)). Two
cautionary notes: the paper's own text is internally inconsistent about which of 0.796/0.819 is "more
sensitive", and a 1B model's sensitivity profile does not transfer to a 5-layer block-diffusion drafter
with a fused context projector and a codebook selector. It does, however, support the general shape:
FFN is not the fragile part, embeddings are not the fragile part, and the parts that are fragile are the
non-standard ones.

### 5.3 What this means for a 5-layer drafter

Our drafter is 5 layers. "Protect the last layer" on a 5-layer stack is protecting 20% of the layers and
~20% of the 852 MB of decoder-linears traffic — a much bigger concession than on a 48-layer target. And
TensorRT's stated reason for the option (FP16 overflow in a 4-bit GEMM's output) is a numerical-range
argument that applies to *any* layer, not specifically the last; the last layer is just the one whose
output goes straight to the head. Whether our layers actually overflow is checkable from the captured
stage tensors already in `profiles/`, and is not something the literature can answer.

---

## 6. The head: what happens to a top-k candidate set when you quantize `lm_head`

This is the highest-value and highest-risk question for us, because the head is 6.98 ms of a 22.99 ms
round and 1.27 GB of the 2.5 GB moved per round.

### 6.1 Evidence that drafters quantize well, including the acceptance number

Four sources report acceptance length after quantizing a DFlash2-class drafter. This is the core of the
survey.

**FP8 W8A8 on GB10, same engine, same session, only the drafter swapped** ([josch15366](https://huggingface.co/josch15366/Qwen3.8-27B-DFlash2-FP8), [M-GB10], 24 replayed agent contexts, K=7):

| drafter | decode | acceptance length |
|---|---:|---:|
| BF16 | 41.3 tok/s | 4.24 |
| INT8 W8A16 | 43.2 | 4.16 |
| **FP8 W8A8** | **43.8** | **4.24** |

"+6.1% decode at an unchanged acceptance length. The quantization was predicted to buy ~7.5% from the
byte budget alone (the drafter is 15.5% of the bytes moved per step on a machine measured at 231.8 of
273 GB/s); the measured 6.1% is close." That is a byte-model prediction validated to within 1.4 points
on the exact drafter family we are running. The author's own scope note is honest: n=24, single stream,
one target.

**EXL3 5.0 bpw on GB10** ([Mia-AiLab](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw), [M-GB10]):

| metric | bf16 draft | 5.0 bpw |
|---|---|---|
| greedy acceptance (code prose) | 2.68–2.80 | 2.68–2.74 |
| HumanEval accept (T=0.6) | 4.27 | **4.43** |
| decode tok/s (HumanEval T=0.6) | 35.7 | **47.5** |

+33% end-to-end decode, acceptance at parity (one run "bit-identical output"). Note the acceptance
*range* widened slightly downward (2.68–2.80 -> 2.68–2.74), which is what a small perturbation looks like
when you measure enough: a floor that holds and a ceiling that drops a little.

**NVFP4 W4A16 on SM120 (RTX 5090, not GB10)** ([YourHighnessLA](https://huggingface.co/YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4), [M-other]): RTN, no calibration, 36 matrices including `fc`;
convs, `candidate_selector` and `hidden_projection` stay BF16; 3.6 GiB -> 1.3 GiB. The card reports
262k-context verification and a KV-pool benefit but **no acceptance number**, so it is evidence of
format support, not of acceptance.

**Q2_K on a 4090** ([analogalok](https://huggingface.co/analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF), [M-other]):
2.81 vs 2.80 accepted length at K=3. Low weight, as discussed in §4.1, but it is the only 2-bit data point
in existence for this drafter family.

The pattern across all four: **block-diffusion drafters lose very little acceptance under weight-only
quantization, and the loss that does appear is in the ceiling, not the floor.** One structural reason is
suggested by the Q2_K card: standalone `llama-imatrix` calibration fails on DFlash2
(`dflash requires ctx_other to be set`) because it is not a standalone causal LM, so importance-matrix
quants cannot be computed. Whatever survival mechanism is at work, it is not imatrix.

### 6.2 The counter-evidence: quantizing `lm_head` is not neutral

Two independent A/Bs on Qwen3.8-27B disagree about *why*, but agree on the sign.

**The FP8 case, on Flash-Next itself** ([jschmied](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/quantizing-lm-head.md), [M-GB10]):

| | `lm_head` BF16 | `lm_head` FP8 |
|---|---:|---:|
| code | 23.2 tok/s | 26.1 (+12.5%) |
| factual | 23.7 | 26.2 (+10.5%) |
| German | 23.5 | 25.4 (+8.1%) |
| NLL/token (14 chunks, 646 tokens) | 0.9687 | 0.9628 (−0.60%) |
| tasks | 10/10 | 10/10 |

"The bandwidth model predicted ~+10% from removing 0.64 GB/token; we measured +11%." Quality: nine
chunks improved, five worsened, "mixed signs, so this is noise, not damage." Same author, different model:
an **NVFP4** head measured 2.4% worse NLL with 8 of 8 chunks worse, and was declined for production,
while the FP8 head is loss-neutral. **The layer is precision-sensitive; the format decides it.** That is
the most important sentence in this survey for §7.

Also useful: `lm_head` is BF16 `[248320, 2560]` in **every published GPU checkpoint** of Flash-Next, and
three independent pieces of vLLM plumbing (head never receives `quant_config`; `config.json`'s `ignore`
list contains `lm_head`; the vocab weight loader rejects the block-scale companion tensor) each
individually prevent it. So the default is BF16 by *omission*, not by measurement — which is exactly why
the two A/Bs below were needed.

**The NVFP4 case, sampled serving** ([adrienbrault](https://github.com/adrienbrault/qwen3.8-27b-rtx5090/blob/main/bench/results/lmhead-quantization-ab.md), [M-other], RTX 5090):

| | NVFP4 lm_head | BF16 lm_head |
|---|---:|---:|
| top-1 agreement vs FP8 reference | 0.9013 | 0.9147 |
| confident flips (ref p>=0.9, 255K tokens) | 0.93% | 0.92% |
| flips at p 0.3–0.6 | **20.5%** | **17.5%** |
| decode c1 prose / code (t/s) | 121 / 164 | 98 / 121 |

"The fidelity gain is real (+1.3 pts agreement) but lives entirely in low-confidence tokens: where the
reference is confident, the quantized head flips nothing the bf16 head does not also flip (0.93% vs
0.92%). The price is 20–26% of decode." Conclusion drawn: "For temperature-sampled agentic serving, the
rule against quantizing lm_head is falsified on this stack: an NVFP4 lm_head is close to free where it
matters. Greedy-decode evals are maximally sensitive to the low-confidence tie-breaks that do move."

**This is the finding that should govern our decision, and it points the opposite way from the way it is
usually quoted.** The quantized head's damage is concentrated in the low-confidence regime. Our
per-slot acceptance is 0.740 / 0.662 / 0.592 / 0.621 / **0.389 / 0.429 / 0.333** — slots 5–7 are exactly
the low-confidence, near-tie regime where 20% of candidates flip. And DFlash2's own Recall@16 declines
from 99.5% at slot 0 to 87.8% at slot 6 ([inco.ai](https://inco.ai/blog/dflash2/)), so the tail slots have
the least headroom to absorb a reordering. A head quantization that is "close to free" for greedy prose
generation is not free for us; we are structured so that its error term lands on the slots that already
decide the round.

### 6.3 The architectural fact that makes vocab trimming possible here

DFlash2 "never carries a head of its own. It scores candidates through the target's own `lm_head`, and the
trim stacks on top" ([Avifenesh](https://huggingface.co/blog/Avifenesh/masked-mtp-drafts)). That is our
structure too — our `logits_top16_ids` stage applies the target head to 7 draft rows. So a vocabulary
trim is applicable to us in a way it would not be to a drafter with a private head.

### 6.4 The vendor precedent: with a candidate head, keep the head in FP16

Restating §4.2 because it is the decision: TensorRT-Edge-LLM, on EAGLE-4 drafts with "a low-rank candidate
head", keeps the `lm_head` in FP16 and *rejects* `--lm-head-quantization`, and also keeps the candidate
head factors in FP16 ([V](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html)).
Our selector is a codebook-based bilinear candidate head in the same functional role. The one vendor with
kernels for this hardware, writing a tool whose purpose is to quantize drafters, decided not to quantize
the head when a candidate head is present.

Meanwhile DFlash drafts in the same tool *do* allow `nvfp4` on the LM head. So the vendor's own position
is self-inconsistent across the two drafter families, which means neither line is a strong citation on
its own — but the EAGLE-4 reasoning (candidate head + head are a coupled selection mechanism) is the one
that describes our stack.

Note also that the vLLM-side blocker is gone: the `ValueError: DFlash2 requires an unquantized target LM
head for candidate TopK` guard lived in an unmerged PR and is absent from released 0.28.0, where
`LogitsProcessor._apply_head` dispatches through `lm_head.quant_method.apply(...)` and "handles a
quantized head fine" ([madeye](https://madeye.github.io/qwen3.8-dflash2-gb10/)). That is a serving-stack
statement, not an acceptance statement — it says a quantized head will *run*, not that it preserves the
candidate set.

---

## 7. Activation-aware and calibration-based methods

The question for us is narrow: does anything that needs a calibration corpus buy more than plain
round-to-nearest on a drafter whose weights we can read?

### 7.1 What each method needs, and whether a 7-row streaming GEMM benefits

| Method | Needs | Byte saving | Applies to us? |
|---|---|---|---|
| RTN weight-only | nothing | format-dependent | yes — this is what every DFlash2 quant used |
| GPTQ | calibration set, layer-wise Hessian (hours) | same as RTN at 4-bit | marginal |
| AWQ | calibration set, activation stats | same as RTN at 4-bit | marginal |
| SmoothQuant | calibration set, activation outliers | **none** (W8A8) | no |
| QuaRot / SpinQuant | calibration or learned rotations | **none** | no |
| per-block scaling (NVFP4 block-16) | nothing (RTN) | built into format | yes |

The negative results are structural, not empirical:

**SmoothQuant, QuaRot, SpinQuant and every rotation-based method exist to make *activation* quantization
survivable.** They redistribute outlier channels from activations onto weights
([arXiv 2608.28003](https://arxiv.org/html/2608.28003v1) implements SmoothQuant for `lm_head`
specifically because "direct per-channel quantization of activations incurs a large accuracy loss due to
outlier channels in x, whereas the weights W are distributed far more evenly"). If we do weight-only
quantization — which is what every GB10 measurement says is fastest, because W4A16's dequant path beats
W4A4's FP4 path single-stream ([ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute)) — then the problem these
methods solve does not arise, and they buy zero bytes. Rotation methods in particular are
accuracy-preserving transforms at *identical* byte cost: a 4-bit rotated weight is still 4 bits. They are
incompatible with "fewer bytes" as a goal.

**AWQ vs GPTQ is the wrong axis on this box.** The one head-to-head on GB10 found INT8 and FP8 drafters
within 1.4% of each other, "the same order as the ~2.5% session-to-session drift we have measured
elsewhere on this box: at this sample size the two schemes are not distinguishable"
([josch15366](https://huggingface.co/josch15366/Qwen3.8-27B-DFlash2-FP8)). If the scheme ranking is
inside the noise floor of the box, the calibration pipeline is not where the value is.

**The one measured exception, and it argues for cheapness.** The same DFlash2 FP8 card's position is that
"the drafter is the one model where quantization is risk-free by construction: under verified-lossless
speculation, drafter quality only moves the acceptance rate (speed), never output quality. So the full
mixed-precision machinery (KL-Fisher allocation, calibration) is the wrong tool here — plain RTN is the
right one" ([TechPrototyper](https://github.com/TechPrototyper/local-ai-lab/blob/master/notes/dflash2-drafter-fp8-quant.md), [M-GB10]).

That is the correct framing for the drafter body: RTN, no calibration, measure acceptance. It is *not*
the correct framing for the head, because the head's job is not "propose plausible tokens", it is
"identify the top 16 of 248,320" — a ranking task whose error mode is a set change, not a
slightly-worse distribution. Which is why §6 needed two separate A/Bs.

### 7.2 QSpec: the one scheme that is genuinely free

A complementary-quantization trick worth naming because it fits our engine exactly: run the *same*
weight-quantized model in a low-precision activation mode to draft and a high-precision mode to verify,
sharing weights and KV cache, switching at near-zero cost. "Compared to high-precision baselines, QSpec
achieves up to 1.64× speedup without quality degradation", and it "requires no training"
([QSpec](https://arxiv.org/html/2410.11305v3), [P]). Its motivation table is also the best evidence for
why we should not quantize activations: W4A4 loses ~30% average on MBPP/GSM8K while W4A16 loses ~4%,
despite both being within 0.85% on WikiText-2 perplexity.

For us the analogous move is cheap and already half-built: rank with a reduced-precision head, rescore
the surviving candidates exactly. See §8.2.

---

## 8. Post-quantization recovery

### 8.1 What is reported, and the corpus sizes

**QAD (quantization-aware distillation) is now the vendor default for NVFP4, not QAT.** The NVIDIA report
is explicit about why: QAT "reuses the same pipelines and training objectives as the original
high-precision models", which fails for models that went through SFT -> RL -> merge, and fails again when
"the original training data of open models may not be available". QAD distils a full-precision teacher
into the quantized student with a KL loss and is "robust to data quality and coverage, enabling accuracy
recovery without full training data" ([NVFP4 QAD](https://arxiv.org/html/2601.20088v3), [V]/[P]).

Measured recovery on Nemotron 3 Nano 30B-A3B (AA-LCR / AIME25 / GPQA-D / LiveCodeBench / SciCode):

| | AA-LCR | AIME25 | GPQA-D | LCB-v5 | SciCode |
|---|---:|---:|---:|---:|---:|
| BF16 | 35.9 | 89.1 | 73.0 | 72.1 | 33.0 |
| NVFP4 PTQ | 31.3 | 85.0 | 71.6 | 68.9 | 30.5 |
| QAD, SFT data only | 32.6 | 86.0 | 72.7 | 70.0 | 31.7 |
| QAD, RL-prompt generations | 34.0 | 82.7 | 73.9 | 70.4 | 33.1 |
| QAD, SFT+RL mixture | **34.3** | 87.9 | 72.7 | 68.9 | 32.3 |

So NVFP4 PTQ cost 4.6 points of AA-LCR; QAD recovered 3.0 of them. The three data sources — SFT only,
RL-prompt generations only, mixture — all land within noise, "demonstrating QAD's robustness to data
composition". A partial recovery, from a fourth to a half of the gap, with ~thousands of samples rather
than the training set.

**Full retraining of a draft model is the expensive alternative, and its cost is documented.** EDA adapts
a drafter to a shifted target using a decoupled shared/private architecture, regenerating training data
from the fine-tuned target and selecting high-value samples. On Qwen2.5-7B -> Qwen2.5-Math-7B it reaches
average acceptance length 4.79 vs 4.37 for the baseline adaptation method "while incurring only 60.8% of
the training cost of full draft model retraining", with a uniform **68,000 samples per domain**
([EDA](https://arxiv.org/html/2603.09527v1), [P]). That 68k figure is the best available answer to "what
corpus does draft adaptation need": order 10^4–10^5 samples, not 10^9 tokens.

**Acceptance is a trainable objective in its own right.** Directly optimising expected accepted length
(EAL, for greedy verification) or a window total-variation loss (WTV, for sampling) beats KL-based
training "consistently", and both can be combined with GRPO using simulated acceptance length as reward
([arXiv 2609.24150](https://arxiv.org/abs/2609.24150), [P]). Relevant only if we ever retrain; noted
because it changes what a "recovery" run should optimise. If we do run QAD on a quantized drafter, the
student loss should be acceptance-shaped, not KL-shaped — and nobody in the corpus has done that for a
block-diffusion drafter.

### 8.2 What recovery is actually needed for us, per stage

This is where the evidence gets thin and I have to reason rather than cite.

- **Drafter body (MLP/qkv/proj): none needed.** Four independent measurements show acceptance at parity
  under weight-only quantization (§6.1). Spending GPU-weeks on QAD for the body would be solving a
  problem the corpus does not show exists.
- **Head: recovery cannot fix the failure mode.** QAD/KL recovery works by matching the student's output
  *distribution* to the teacher's. Our failure mode is a change in the *argmax set* at low confidence
  (§6.2) — a discrete event that KL loss barely penalises, because the tokens that flip have low
  probability under both models. A KL-trained recovery would report near-perfect distribution match and
  still reorder the top-16. If the head is quantized, the recovery objective has to be set-level
  agreement (top-16 recall against the bf16 head on captured rows), which is a much cheaper thing to
  measure than to train: it is an offline check over captured head inputs, not a training run.
- **The cheap version of recovery is a rescore, not a retrain.** Rank candidates with the quantized head,
  then recompute exact logits for the surviving top-N only. At N=16 that is 16 rows x 2560 = 82 KB of
  head weight touched per round, i.e. free. This preserves the selector's unary `U_t(b)` values exactly;
  it does *not* recover candidates the quantized ranking dropped. That residual risk is measurable
  offline and is the thing to measure before touching the engine path.

---

## 9. Alternatives to quantizing the head at all

The head is a 248,320-row argmax over 7 rows. Three families of alternative, all with acceptance data.

### 9.1 Frequency-ranked vocabulary trimming

**FR-Spec** (ACL 2025) constrains the draft search to a frequency-prioritised token subset, reducing
LM-head computation by **75%**, and reports an average **1.12×** speedup over EAGLE-2 and 1.08× over
Medusa, with the verification distribution unchanged ([FR-Spec](https://arxiv.org/html/2502.14856v1), [P]).
Its premise: 75% of vocabulary tokens account for <5% of occurrences.

**VocabTrim** (Qualcomm) is training-free, rebuilds the drafter LM head over the most frequently sampled
target tokens, "reduces draft vocabularies by up to 97% with high coverage", giving "up to 16% latency
reduction and 20% throughput improvement" on domain tasks and "up to 6.7%" on diverse OOD tasks
([VocabTrim](https://arxiv.org/html/2506.22694v1), [P]). The OOD number is the honest one for us.

**Measured on DFlash2 and on Flash-Next, on GB10 and elsewhere** — this is the best evidence in the
survey for anything (§Avifenesh):

| setting | result |
|---|---|
| Qwen3.8-27B, MTP head, 32,768 rows (7.6× cut) | acceptance 66.7% -> **63.6%**, decode 117.1 -> 121.7 tok/s (+3.9%, K=3) |
| same, RTX PRO 6000, 5 interleaved runs | **+5.1%** (137.9 vs 131.2 decode p50) |
| same, RTX 5090 Laptop | **+6.4%** |
| 27B **with DFlash2 drafter**, RTX PRO 6000 | **+4.8% chat, +1.7% agentic** |
| 27B with DFlash2, RTX 5090 Laptop | **+3.9%** |
| **Qwen3.8-Flash-Next, FP4, vLLM, 65,536-row trim, DGX Spark** | **+4.5% to +9.7% at 1 user**, +2.2–3.0% at 8 |
| Ornith-1.5-35B-A3B | +10% short single-user, +2.9% at 8 users |

Three cautions that matter more than the gains:

1. **"Ranks don't transfer between models, I measured -12 acceptance points when I tried."** A trim list
   must be built from *this* model's own generations. Ours: 163k tokens of agentic output + 154k of prose
   was the recipe. Cheap to collect, but it is a data dependency, not a knob.
2. **The trimmed head still accepts more; it loses anyway only when the head is a big share of the step.**
   "The full head still accepts more, 0.76 vs 0.74. Loses anyway." The economics invert at long context
   and at high concurrency: "when most of a round is reading the KV cache, the head is a small part of
   it, and the acceptance cost wins."
3. **Requantizing the trimmed head to NVFP4 cost "zero measured acceptance"** in that author's runs. Note
   the tension with §6.2: a *trimmed* head at NVFP4 is a narrower problem than a full head at NVFP4,
   because the surviving rows are the high-frequency ones where the reference is confident. Coherent, but
   n=1 and not our harness.

`2603.05210` reframes the same trade as a constrained optimisation (coverage over assistant responses vs
architecture-aware FLOPs for the head, optimised with a TPE over the coverage–latency Pareto frontier) and
reports "up to 16% latency reduction and 20% throughput improvement" with up to 97% vocabulary reduction
([arXiv](https://arxiv.org/html/2603.05210v1), [P]). Useful if we ever want to choose the trim size by
method rather than by sweep; not itself evidence about our model.

### 9.2 Dynamic vocabulary selection

**SpecVocab** selects a vocabulary subset per decoding step and reports "up to an 8.1% increase in average
throughput over EAGLE-3", with higher acceptance length than EAGLE-3 rather than lower
([SpecVocab](https://arxiv.org/html/2602.13836v1), [P]). The reframing is worth keeping: the target token
being out-of-vocabulary is the cost of *static* trimming, and per-step selection avoids it.

Against it, from the SlimSpec analysis: dynamic selection "adds inference-time machinery: routing, top-k
selection, irregular indexing, and gathering context-dependent subsets of weights. These operations are
not as GPU-friendly as dense matrix multiplication and can become bottlenecks themselves"
([Nebius](https://nebius.com/blog/posts/slimspec-faster-speculative-decoding-without-cutting-the-vocabulary)).
On a 7-row GEMM with a hand-written kernel, an irregular gather per round is a real risk.

### 9.3 Low-rank head (compress the representation, not the vocabulary)

SlimSpec replaces `logits = W h` with `W_up W_down h`, `W_up` in R^(V×r), `W_down` in R^(r×d), giving
cost O(Vr + rd) instead of O(Vd), with r = d/8 proposed. Reported: **4–5× reduction in LM-head GPU time**
versus the full projection, "while keeping acceptance quality competitive", and on an end-to-end
(ν, ρτ) plane SlimSpec "achieves the largest LM-head acceleration ν while keeping ρτ close to 1"
([Nebius](https://nebius.com/blog/posts/slimspec-faster-speculative-decoding-without-cutting-the-vocabulary)).
Compare VocabTrim/SpecVocab at ~60% LM-head latency reduction in the same figure.

This is architecturally attractive for us — 4–5× beats vocabulary trimming's ~3.8–7.6× byte cut only if
the rank is small, and unlike trimming it is lossless in *support* (no token is unreachable). But it is a
**retrain**, not a conversion: `W` and `W_up W_down` are different functions, and the drafter was trained
against the full-rank head. There is no training-free version. Also note r = d/8 with d=2560 is rank 320,
so `W_up` is 248,320 x 320 — 159 MB at bf16, versus 1,271 MB. The saving would be real (4–5× is
consistent: 1271 -> ~280 MB).

Which brings us to the constraint that rules this out for the near term: **a low-rank head changes the
drafter's function and therefore needs acceptance validation across the whole slot profile**, and the
DFlash2 head is the *target's* head — shared, not ours to re-parameterise without affecting the target's
own decode path.

### 9.4 Hierarchical / two-stage candidate retrieval

No measured instance of this for a drafter was found. The nearest thing is QSpec's draft-then-verify with
different precisions (§7.2) and the FR-Spec trim. I would not cite a hierarchical sketch-retrieval scheme
as evidence-based: nobody in this corpus has published acceptance for one on a block drafter, and the
two-stage structure adds exactly the irregular-gather cost §9.2 warns about. If it is tried, it should be
tried as "quantized ranking pass + exact rescore of the survivors", which is §8.2 and already in the
optimisation plan as the head item.

---

## 10. Mapping each option onto our stages

Byte arithmetic at the 200 GB/s wall, as requested; the `local` column scales the *measured* stage time by
the byte fraction, which is the more honest estimate where the stage is known to run at its own rate.
The head's `local` column uses the measured 5.43 ms GEMM only — the 1.39 ms top-k does not scale with
bytes.

| # | option | stage shrunk | bytes before -> after | saved | ms @200 | ms local | acceptance risk | evidence |
|---|---|---|---:|---:|---:|---:|---|---|
| 1 | FP8 W8A8 on drafter linears (MLP+qkv+proj) | mlp 2.75, qkv 0.84, proj 0.54 | 852.0 -> 426.0 MB | 426.0 MB | 2.13 | **2.07** | **low** — measured at unchanged acceptance (4.24 vs 4.24) on this drafter family, this box, n=24 | [M-GB10] [josch15366](https://huggingface.co/josch15366/Qwen3.8-27B-DFlash2-FP8); [M-GB10] [TechPrototyper](https://github.com/TechPrototyper/local-ai-lab/blob/master/notes/dflash2-drafter-fp8-quant.md) |
| 2 | NVFP4 W4A16 on drafter linears | same | 852.0 -> 239.6 MB | 612.4 MB | 3.06 | ~2.9 | **low-medium** — no acceptance number published for a DFlash2 NVFP4 body; W4A4 vs W4A16 measured on a *target* suggests W4A16 is safe, and 4-bit drafts are plug-and-play in ML-SpecQD | [V] [TRT-Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html) validated for DFlash; [M-other] [YourHighnessLA](https://huggingface.co/YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4) no acceptance reported |
| 3 | INT8 W8A16 group-128 on drafter linears | same | 852.0 -> 426.0 MB | 426.0 MB | 2.13 | ~2.07 | **low** — 4.16 vs 4.24; "not distinguishable" from FP8 | [M-GB10] [josch15366](https://huggingface.co/josch15366/Qwen3.8-27B-DFlash2-FP8) |
| 4 | FP8 head (GEMM only) | head GEMM 5.43 | 1271.4 -> 635.7 MB | 635.7 MB | 3.18 | **2.71** | **medium** — loss-neutral NLL on Flash-Next itself, but top-1 agreement −1.3 pts and low-confidence flips +3 pts on the 27B; our slots 5–7 live in that regime | [M-GB10] [jschmied](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/quantizing-lm-head.md); [M-other] [adrienbrault](https://github.com/adrienbrault/qwen3.8-27b-rtx5090/blob/main/bench/results/lmhead-quantization-ab.md) |
| 5 | NVFP4 head | head GEMM 5.43 | 1271.4 -> 357.6 MB | 913.8 MB | 4.57 | 3.90 | **high** — same author measured 2.4% worse NLL, 8/8 chunks, and declined it while keeping the FP8 head in prod | [M-GB10] [jschmied](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/quantizing-lm-head.md) |
| 6 | Vocab trim to 65,536 (3.79×) | head GEMM + top-k | 1271.4 -> 335.5 MB | 935.9 MB | 4.68 | **4.00** (+~1.0 top-k) | **medium** — measured +4.5–9.7% on Flash-Next/GB10 at 65,536; needs a rank list from this model's own generations; ranks do not transfer (−12 pts) | [M-GB10] [Avifenesh](https://huggingface.co/blog/Avifenesh/masked-mtp-drafts) |
| 7 | Vocab trim to 32,768 (7.58×) | head GEMM + top-k | 1271.4 -> 167.8 MB | 1103.6 MB | 5.52 | **4.71** (+~1.2 top-k) | **medium-high** — 66.7% -> 63.6% draft acceptance on MTP at this width; DFlash2 measured only at ~65k | [M-GB10] same, + [M-other] 5090/PRO 6000 rows |
| 8 | Low-rank head (SlimSpec, r=d/8) | head GEMM | 1271.4 -> ~280 MB | ~990 MB | 4.95 | ~4.1 | **high** — requires retraining the head; changes the function, not the storage; no DFlash2 acceptance data exists | [P]/[blog] [Nebius](https://nebius.com/blog/posts/slimspec-faster-speculative-decoding-without-cutting-the-vocabulary) |
| 9 | FP8/INT8 on `fc` (65.5 MB) | `fc_ctx` 5.10 | 65.5 -> 32.8 MB | 32.8 MB | 0.16 | ~0.0 | **n/a — do not do this** | stage runs at 13 GB/s; bytes are not its problem |
| 10 | Quantize selector codebooks | proposals 1.29 | 255.6 -> 127.8 MB | 127.8 MB | 0.64 | 0.65 | **high** — the selector is the acceptance mechanism; vendor keeps it FP16 when a candidate head is present; 2 of 3 published DFlash2 quants protect it | [V] [TRT-Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html); [M-GB10] [josch15366](https://huggingface.co/josch15366/Qwen3.8-27B-DFlash2-FP8) |
| 11 | Quantize convs (33 MB) | convs 0.87 | 33 -> 16.5 MB | 16.5 MB | 0.08 | 0.43 | **high for ~0.4 ms** — convs run at 38 GB/s, i.e. latency-bound; shaving bytes mostly does not help, and they are block-diffusion machinery | [M-GB10] [josch15366](https://huggingface.co/josch15366/Qwen3.8-27B-DFlash2-FP8) |
| 12 | `--keep-last-layer-fp16` (quality guard, no bytes) | none — *adds* bytes | — | **−170 MB** | −0.85 | −0.9 | **reduces risk**; costs ~0.9 ms | [V] [TRT-Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html) |
| 13 | QAD / QLoRA recovery (enabler, no bytes) | none | — | 0 | 0 | 0 | **reduces risk**; KL loss cannot see a top-16 set change (§8.2) | [V]/[P] [NVFP4 QAD](https://arxiv.org/html/2601.20088v3); [P] [EDA](https://arxiv.org/html/2603.09527v1) |

### 10.1 What the arithmetic adds up to

- **Best low-risk package (rows 1 + 4): FP8 drafter linears + FP8 head = 4.78 ms**, i.e. 22.99 -> 18.2 ms
  of drafter, 74.2 -> 69.4 ms of round, 45.0 -> **48.1 tok/s (+6.9%)**. The whole of it rides on two
  measurements on this exact drafter family and this exact box, both reporting acceptance at parity.
- **Best aggressive package (rows 2 + 6): NVFP4 body + 65,536 trim = ~7.9 ms**, round 74.2 -> 66.3 ms,
  **50.3 tok/s (+11.8%)**. Both halves have measured precedent; neither has been measured *together*, and
  the trim's acceptance data is at 65,536 on Flash-Next while our per-slot profile is steeper.
- **Upper bound if everything worked (rows 2 + 7 + 10, no acceptance loss): ~9.1 ms**, 51.5 tok/s
  (+14.4%). The extra 1.2 ms over the previous line costs the most acceptance risk in the table, which is
  a bad trade on its face.
- **For scale:** deleting the entire drafter only gets to ~66 tok/s (+47%), and half the round (10.29 ms)
  is not addressable by any of this at all. The head-plus-`fc_ctx` pair (12.08 ms, 53% of the round) is
  where the real money is, and `fc_ctx` is a 13 GB/s kernel problem, not a precision problem.

---

## 11. What I would rule out, and why

**Rule out outright:**

1. **Quantizing `fc` (row 9).** The stage runs at 13 GB/s — 11× below what the head kernel achieves on the
   same bytes-per-launch regime. Halving its bytes at 13 GB/s saves 0.16 ms while the *kernel* fix is
   worth up to ~4.6 ms. TensorRT-Edge-LLM refuses to export a DFlash draft with a quantized `fc`, and the
   published NVFP4 drafter exceptions are on other silicon with no acceptance number. Doing this would be
   optimising the one stage whose problem is not bytes.
2. **SmoothQuant, QuaRot, SpinQuant and rotation-based quantization (§7.1).** They buy zero bytes at weight-only,
   require calibration, and exist to protect activation precision we are deliberately not using.
3. **Quantizing activations (W4A4/W8A8) in the head or body.** W4A16 is the fastest measured single-stream
   path on this box ([ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute)),
   and W4A4 costs ~30% on reasoning benchmarks vs ~4% for W4A16 ([QSpec](https://arxiv.org/html/2410.11305v3)).
   We are GEMV-shaped; activation quant buys nothing and costs the thing we care about.
4. **A GPTQ/AWQ calibration pipeline for the drafter body (row 1–3 alternatives).** Scheme differences are
   inside the box's own session-to-session drift (~2.5%), and RTN was sufficient for every measured DFlash2
   quant. Calibration is effort without a measurable return here.
5. **The 2-bit direction (Q2_K-style).** One data point, at K=3, on silicon that has 24 GB of VRAM pressure
   we do not have. Our target is speed on a 128 GB box, not capacity on a 4090.
6. **`--keep-last-layer-fp16` as a default (row 12).** On a 5-layer stack it protects 20% of the decoder
   traffic (-170 MB, ~0.9 ms) for a numerical-range argument we can test directly from captured stage
   tensors. Adopt it only if the captured activations actually exceed 65504; otherwise it is a 4%
   throughput donation to a hypothetical.

**Rule in, with conditions:**

7. **FP8 on the drafter linears** — the best-verified option in the survey. Two independent GB10
   measurements on this exact drafter family, acceptance at parity, byte-model prediction validated to
   1.4 points.
8. **FP8 on the head, gated on an offline top-16 check.** The format that was loss-neutral where NVFP4 was
   not, on the same model. The gate is not optional: compare exact-bf16 top-16 against fp8-ranked top-16
   over many captured rows, per slot, before the engine changes. That check is cheap and the plan already
   names it.
9. **Vocabulary trim, gated on collecting our own rank list.** Measured positive on Flash-Next on this
   exact box (+4.5–9.7% at 1 user), stacks on the target's head (which is our structure), and cuts the
   top-k cost too. The dependency is a rank list built from *this* model's generations, and an
   acceptance-by-slot measurement afterwards — not the aggregate tok/s.

**Not enough evidence to act on, but worth knowing:**

10. **NVFP4 on the drafter body** plausibly saves 0.8 ms more than FP8, and is the vendor-validated path for
    DFlash drafts. But nobody has published DFlash2-body-NVFP4 acceptance, the dequant cost is paid 7 rows
    at a time, and we would be writing the FP4 dequant ourselves. Worth a measurement if FP8 lands clean,
    not worth being the first thing we try.
11. **Low-rank head** is the only option that beats trimming without giving up vocabulary support, and it
    needs a retrain against the *target's* head, which is not ours. Park it. If pursued, pursue it as a
    distillation with an acceptance-shaped loss ([arXiv 2609.24150](https://arxiv.org/abs/2609.24150)),
    not as a KL recovery.

---

## 12. Ranked recommendation

Ordered by expected end-to-end tok/s per unit of risk and work, at the current 74.2 ms round.

1. **FP8 W8A8 on the three decoder-linear groups (MLP, qkv, attn proj), RTN, dynamic per-token
   activations.** ~2.07 ms, risk low, evidence is two GB10 measurements on this drafter family with
   acceptance at parity. This is the only option in the table where the risk rating rests on a
   same-box, same-architecture, same-K measurement of the metric we care about.
2. **Offline top-16 fidelity check of an FP8 head, then — if it holds — the FP8 head.** ~2.71 ms, risk
   medium. The check is the deliverable regardless: it also tells us how much headroom the
   quantized-head ranking has, which is the number that decides between options 2, 3 and 4 below. Do it
   per slot, because the whole argument in §6.2 is that the damage lands on slots 5–7.
3. **Collect a frequency-ranked token list from this model's own generations, measure the trim at 65,536
   with per-slot acceptance.** ~5.0 ms (GEMM + top-k), risk medium, and the largest single win available
   without touching the selector. Note this partially *subsumes* item 2: redundant with a head quant,
   additive with it, and strictly better than a 32,768 trim on risk.
4. **Shuffle-based top-16 merge** (already ranked in the plan, no acceptance risk, not a quantization
   item) — ~1.2 ms of the 1.39 ms top-k. Keep it in the queue because it stacks with every row above.
5. **Re-attack `fc_ctx`.** 5.10 ms at 13 GB/s is 22% of the round and the survey offers no precision
   answer for it, only confirmation that precision is the wrong axis. This is the largest single
   recoverable block in the drafter and it is a kernel problem.
6. **NVFP4 on the decoder linears**, only after FP8 is landed and measured, if the extra ~0.8 ms is worth
   writing an FP4 dequant for.
7. **Nothing on the selector, convs, or `fc`,** and no calibration pipeline, no rotation methods, no
   activation quantization, no 2-bit experiments.

If only one thing is done: item 1 plus item 4 — both are acceptance-safe, both are ~3.3 ms combined, and
both are reversible. If two: add the offline head check from item 2, even if the head itself is never
quantized, because that check is what makes items 2 and 3 decidable instead of arguable.

---

## 13. What I could not verify

Stated plainly, because these are the places where the recommendation could be wrong:

1. **No DFlash2-under-Flash-Next quantization exists in public.** Every measured DFlash2 quant is the
   **27B** drafter (5 layers, 1.92B params, 3.85 GB bf16) on a 27B target. Ours is the Flash-Next drafter
   on a Flash-Next target. Architecture and lineage are the same but the drafter is a different size
   (1.21 GB vs 3.85 GB) and the target's vocabulary path is different. The +6.1% FP8 result is the closest
   available analogue, not our measurement.
2. **No per-slot acceptance data exists for any quantized drafter, anywhere.** Every source reports
   aggregate acceptance length. Our entire risk argument in §6.2 and §10 is an inference from
   (a) adrienbrault's per-confidence-bucket head A/B and (b) inco.ai's per-slot Recall@16 curve. Both are
   sound but neither is a per-slot post-quantization measurement. This is the single largest gap, and it
   is cheap to close on our side and impossible to close from the literature.
3. **The FP8-head "no measurable cost" claim is weak statistics.** n=14 chunks / 646 tokens, nine improved
   and five worsened. The author says so. It is not a null result, it is an underpowered one, which is
   exactly why the per-slot top-16 check has to precede any engine change.
4. **The NVFP4-head rejection (2.4% worse NLL, 8/8 chunks) is one author, one model, one harness.** The
   adrienbrault A/B on a different 27B checkpoint reached the opposite practical conclusion for sampled
   serving. Both are internally coherent; they differ in metric (NLL vs top-1 agreement) and in regime
   (greedy eval vs temperature sampling). I resolved this in favour of "medium/high risk for us" because
   our selection is greedy argmax over 7 rows and our tail slots are the fragile ones — but the
   literature does not settle it.
5. **Vocab-trim numbers on Flash-Next are aggregate tok/s at 1 user, one run** ("Also one run"), not
   acceptance by slot. The +4.5–9.7% band is wide because the workload mix matters.
6. **Nobody has published acceptance for a low-rank or hierarchical draft head on a block-diffusion
   drafter.** SlimSpec's numbers are EAGLE-3 on Llama-3.1-8B. I could not find a two-stage
   sketch-retrieval scheme with published acceptance at all, so it is excluded from the table rather than
   rated.
7. **The `down_proj`/`gate`/`up` split is unmeasured in every source I found.** If a mixed-precision MLP
   is ever wanted, nobody's data supports a choice.
8. **Kernel-selection risk for our own FP4 path is unquantified.** All the "70× apart" evidence is about
   vLLM/FlashInfer/CUTLASS selections. Our kernels are ours; the analogous risk is writing a dequant that
   is slower than the byte saving, which no source can predict for us.
9. **Whether our 5-layer stack overflows FP16 in a 4-bit GEMM's output** (the `--keep-last-layer-fp16`
   rationale) is checkable from the captured stage tensors and I did not check it.

---

## 14. Sources

Primary measurements and model cards:

- ai-muninn, *NVFP4 is 1.5× FP8 on a DGX Spark — but it's compression, not the FP4 cores*: https://ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute
- sergioamsilva, *dgx-spark-field-notes — Quantization on GB10*: https://github.com/sergioamsilva/dgx-spark-field-notes/blob/main/docs/quantization.md
- gpuwerk, *DGX Spark quantization speed: NVFP4, MXFP4, FP8, AWQ, GGUF*: https://gpuwerk.com/dgx-spark/guides/dgx-spark-quantization-speed
- sf-stav, *veloGB10*: https://github.com/sf-stav/veloGB10
- Avarok, *We Unlocked NVFP4 on DGX Spark*: https://github.com/Avarok-Cybersecurity/dgx-vllm/blob/main/NVFP4_BREAKTHROUGH_DGX_SPARK.md
- madeye, *DFlash2 on GB10 — 4.70× decode speedup*: https://madeye.github.io/qwen3.8-dflash2-gb10/
- jschmied, *Quantizing `lm_head`: +11% for nothing*: https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/quantizing-lm-head.md
- adrienbrault, *lm_head quantization, a controlled A/B*: https://github.com/adrienbrault/qwen3.8-27b-rtx5090/blob/main/bench/results/lmhead-quantization-ab.md
- josch15366, *Qwen3.8-27B-DFlash2-FP8* (GB10, K=7, acceptance A/B): https://huggingface.co/josch15366/Qwen3.8-27B-DFlash2-FP8
- TechPrototyper, *Quantizing the DFlash2 drafter: acceptance-neutral*: https://github.com/TechPrototyper/local-ai-lab/blob/master/notes/dflash2-drafter-fp8-quant.md
- Mia-AiLab, *Qwen3.8-27B-DFlash2-EXL3-5.0bpw* (GB10, +33%): https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw
- YourHighnessLA, *Qwen3.8-27B-DFlash2-NVFP4* (SM120, no acceptance reported): https://huggingface.co/YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4
- analogalok, *Qwen3.8-27B-DFlash2-Q2_K-GGUF*: https://huggingface.co/analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF
- Avifenesh, *Masked MTP drafts* (FR-Spec trim incl. DFlash2 and Flash-Next on GB10): https://huggingface.co/blog/Avifenesh/masked-mtp-drafts
- kubesimplify, *Running Qwen3.8-Flash-Next on a DGX Spark and RTX PRO 6000*: https://blog.kubesimplify.com/running-qwen3-8-flash-next-on-dgx-spark-and-rtx-pro-6000
- NVIDIA developer forums, *Qwen3.8-Flash-Next on 1, 2 and 4 DGX Sparks with NVIDIA's official NVFP4 quant*: https://forums.developer.nvidia.com/t/qwen3-8-flash-next-on-1-2-and-4-dgx-sparks-with-nvidias-official-nvfp4-quant-64-tok-s-peak-single-stream/382476
- ggml-org, *Performance of llama.cpp on NVIDIA DGX Spark*: https://github.com/ggml-org/llama.cpp/discussions/16578
- inco.ai, *DFlash 2: Keep Drafting Parallel* (Recall@1/Recall@16 per slot): https://inco.ai/blog/dflash2/

Vendor documentation:

- NVIDIA, *TensorRT Edge-LLM — Quantization* (DFlash draft, EAGLE-4 head rule, keep-last-layer-fp16, SM12x MoE export): https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html
- vLLM, *per-channel fp8 quantization of an unquantized lm_head* PR #61815: https://github.com/vllm-project/vllm/pull/61815

Papers:

- *FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling*: https://arxiv.org/html/2502.14856v1
- *VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding*: https://arxiv.org/html/2506.22694v1
- *Speculative Decoding with a Speculative Vocabulary* (SpecVocab): https://arxiv.org/html/2602.13836v1
- *Balancing Coverage and Draft Latency in Vocabulary Trimming*: https://arxiv.org/html/2603.05210v1
- *ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts*: https://arxiv.org/abs/2503.13565
- *QSpec: Speculative Decoding with Complementary Quantization Schemes*: https://arxiv.org/html/2410.11305v3
- *Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery*: https://arxiv.org/html/2601.20088v3
- *Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation* (EDA): https://arxiv.org/html/2603.09527v1
- *Acceptance-Aware Draft Model Training for Speculative Decoding*: https://arxiv.org/abs/2609.24150
- *A Method for Layer Bit-Width Allocation in LLM Quantization* (FFN vs attention vs lm_head, SmoothQuant-for-head): https://arxiv.org/html/2608.28003v1
- *Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware*: https://arxiv.org/html/2607.17283v1

Blog/analysis:

- Nebius, *SlimSpec: faster speculative decoding without cutting the vocabulary*: https://nebius.com/blog/posts/slimspec-faster-speculative-decoding-without-cutting-the-vocabulary
