# Multi-lane speculative decoding — external research and work plan

Status: plan, nothing implemented. Written 2026-11-20 against `running` @ `58a1125c0`.
Scope: why this engine's aggregate throughput is flat in lane count, what other
engines do differently, and the ordered experiments that would settle it.

Companion documents: `flashnext-dflash2-experiment-ledger.md` (measurements),
`flashnext-dflash2-serving-defects.md` (fixed defects), `flashnext-dflash2-tap-ring.md`
(ring mechanics), `flashnext-layer-time.md` (per-stage timing).

---

## 1. The observation this plan is about

One round at 4 lanes costs ~4.5x a lone round and returns the same tokens per
stream. Aggregate sits at ~1.1x single-stream, so 4 lanes buy nothing.

| lanes | round time | tokens/round | aggregate tok/s | per-stream tok/s |
|-------|-----------|--------------|-----------------|------------------|
| 1     | ~60 ms    | ~3.5         | ~56             | ~56              |
| 2     | ~137 ms   | —            | ~62             | ~31              |
| 4     | ~273 ms   | ~4.38        | ~66.6           | ~16.7            |

The curve is almost exactly linear in lanes. A term that scales linearly with
lanes and dominates the round is, by definition, not being amortised by batching.

That is the whole problem, and it is not "one lane at a time" — `draftTogether`
and `verifyMany` do put all lanes in one pass (`cuda_mtp_ops.zig:219`).

---

## 2. What the research says (external, not measured on this engine)

Grades: **[GB10]** measured on GB10/sm_121 by a named source; **[other]** measured on
datacenter silicon; **[code]** read from a named source path; **[inferred]** derivation.

### 2.1 The arithmetic nobody escapes

Verification cost is proportional to `BS * K`; drafting cost to `BS`; the weight
read is constant. As batch grows, `BS*K` crosses the point where the step is
compute-bound instead of weight-read-bound, and speculation's premise — trade
spare FLOPs for fewer weight reads — inverts into a loss. **[code]** vLLM states
this directly in its dynamic-speculation doc, and **ships a schedule that encodes
it**: `[[1,16,5],[17,32,4],[33,64,3],[65,128,1],[129,512,0]]` — K drops to 1 at
batch 65 and to **zero** at batch 129. TensorRT-LLM's own doc: *"speed ups are
only observable at low batch sizes."*

### 2.2 Published curves all decay; the question is where they cross 1.0x

EAGLE-3, LLaMA-3.1-8B, SGLang, 8xH100 **[other]** — relative speedup over AR:

- batch 1/2/4/8/16/32: **2.0x / 1.9x / 1.7x / 1.5x / 1.2x / 1.0x**
- EAGLE-1 on RTX3090 crosses 1.0x at batch 32 and reaches **0.88x** at 64.

DFlash (block-diffusion drafter) holds **1.3–2.1x at concurrency 32** — better
than EAGLE — because its draft cost is ONE parallel forward per round instead of
gamma sequential ones. Decay is still monotone. **[other]**

### 2.3 MagicDec: the crossover is set by sequence length, not batch size

Measured on 8xH100, LLaMA-3.1-8B, 16k input **[other]**: parameter load is FLAT
(0.7 → 0.8 ms) across batch 1..256; KV load grows 0.4 → 21.8 ms (55x). At large
batch the step is ~95% KV traffic. Speculation was buying parameter-read
amortisation, and parameters stopped being the cost. Consequence: speculation
pays only **beyond a critical sequence length**, where KV loading dominates and
verify's compute overhead is small relative to what it amortises. Reported up to
2.51x at batch 32–256 for moderate-to-long sequences.

### 2.4 GB10 specifics — thin, community-measured, and mostly configuration

- Achieved bandwidth is not 273 GB/s. Idle 227–246 GB/s; STREAM triad 200–214;
  under a running model the residual copy bandwidth falls to ~161 → 90 GB/s
  read while vLLM decodes. **[GB10]**
- SGLang DFlash2 on GB10: verify step time **flat at 138 ± 3 ms** across accept
  lengths 1.55–7.95 at 27K–84K depth. The step does not track tokens committed —
  DRAM-bandwidth-bound signature. Arithmetic on the byte table gives ~146–185 GB/s
  (~53–68% of nominal). **[GB10]**
- vLLM MTP-4 on GB10, GLM-5.3-Flash NVFP4 TP2: MTP off→on gives +101% at C1,
  +65% at C4, +61% at C8 — relative gain decays 2.0x → 1.65x → 1.61x. **[GB10]**
- Two diagnosed "speculation stops scaling" cases on GB10 were **configuration**:
  a mamba state-slot cap that silently held concurrency at 2, and an unpinned
  graph capture list. **[GB10]**
- Practical lane ceiling on GB10: streams stall around 6–12 concurrent; Flash-Next
  measured 63 tok/s single-stream, 306 tok/s at 6 streams, capped at 8 total.
  vLLM's own Spark guidance: keep `--max-num-seqs` low. **[GB10]**

### 2.5 The per-sequence state layouts — copy the shape, not the code

**Nobody uses a gather/scatter index table as the primary layout.** Two patterns
dominate, and both are production-shipped on sm_121:

- **vLLM — padded fixed-size slots keyed by persistent request index.** Draft
  logits live in `[max_num_reqs, num_spec_steps, vocab]`, indexed by
  `req_state_idx`, not batch position. A `row_mapping`/`inverse_row_mapping` pair
  detects stale rows when a request is skipped and its slot reused. **[code]**
- **vLLM — per-token flat tap buffer.** Target aux hidden states are captured into
  `[max_num_tokens, hidden]` whose row index **is** the token's row in the packed
  batch; the drafter reads the same `query_start_loc`/`sample_indices`. No modulo,
  no ring, no aliasing. For FULL CUDA graphs the first `_num_graph_context_tokens`
  rows are staged into a fixed-address buffer at capture time. **[code]**
- **SGLang — request-block-major ring.** `local row(req,pos) = 1 + req*rows_per_req
  + (pos//dcp) % rows_per_req`, with `rows_per_req = ceil((window_left+1+block_size)/dcp)+1`.
  The `req` term is **outside** the modulo, so each request owns a contiguous block
  and lanes cannot alias. `min_pos` floors each request's window at its
  prefix-cache hit. **[code]** `python/sglang/srt/speculative/dflash_draft_ring.py`
- **SGLang — EAGLE tree mask** is a flat 1-D buffer of per-sequence prefix block +
  per-sequence tree block, with per-sequence offsets from `seq_lens`; acceptance
  walks per-sequence sibling links. Mask memset is skipped when a buffer is passed —
  worth ~100s of MB/step at 273 GB/s. **[code]** `eagle_utils.py`

### 2.6 The closest architectural analogue to our drafter

DFlash does not *record* taps into a ring. It injects the fused target feature into
the **draft model's own paged KV cache** as persistent context entries, so the
per-sequence target state inherits the KV cache's existing page table and slot
mapping. Correctness comes free from KV-cache addressing instead of a second
addressing scheme. Extra parameters: one shared projection, ~42 MB at D=2048 bf16.
Cost on GB10: padded draft KV reserves ~36.7 KB/token against ~2 KB used —
~3.3 GiB at a 2048-token batch budget, ~7.4 GiB at 7168. **[other]** architecture,
**[GB10]** memory.

### 2.7 What is genuinely unknown

- No engine publishes a spec-decode speedup-vs-batch curve measured **on GB10**.
- No source measures the cost of reading a drafter's hidden-state taps as a
  function of batch size, on any hardware. Our tap-ring question is unanswered
  in public literature.
- No DRAM-counter measurement of a speculative verify step on GB10 exists; the
  146–185 GB/s figures are arithmetic on a byte table.
- Whether acceptance genuinely falls with concurrency, or is a prompt-mix /
  window-counter artifact, is contested even in the GB10 sources.

---

## 3. What we already know from our own measurements

Established in this repo, not inferred:

1. **The tap ring write is not a bandwidth problem.** 40 KiB per tap layer per
   round at 4 lanes = 0.0002% of weight bytes. Even 8 taps at 4 lanes is 0.31 MiB.
2. **The per-row ring table attempt failed for an addressing reason, not a cost
   reason.** Built, compiled, ran: 2x slower and acceptance collapsed to 1–3%
   (vs ~46% baseline), `tokens_per_chunk` 4.4 → 1.1. Drafts were read from the
   wrong ring rows — the silent-wrong-output failure. Reverted; `running` clean at
   `58a1125c0`. The table made the addressing *legal* but not *correct*: the
   mean4 collapse and the ring's slot layout assume row-ordered contiguity.
   `pos % tap_rows` drops `req` entirely and aliases lanes.
3. **Drafter + disk tier now works.** The blocker was `covered` not being advanced
   by `note()` on a pure prefill, so every prefill-saved state was un-diskable and
   the probe hit `ProbeUnmarked`. Fixed in `58a1125c0`; verified `evicted 3`,
   336 MiB written, cache hits resuming past the prefix.
4. **MTP currently beats the drafter on production traffic** (the 2026-11-18
   39.2 vs 52.5 figure is stale; production measures ~60 tok/s on MTP). The
   drafter remains the project's goal, not a fallback.

---

## 4. Hypotheses, ranked by what would be cheapest to disprove

**H1 — the round is dominated by a term proportional to lanes.** Our arithmetic:
a ~12 ms drafter against a ~2.5 ms target weight read means the drafter is
already the per-round bottleneck; batching the target cannot recover more than
~17% of round time. *If true, no ring or verify work will move aggregate.*

**H2 — the flat curve is configuration, not architecture.** Both diagnosed GB10
cases were configuration. Ours may be: lanes queued rather than running, captured
graph shapes not covering reachable row counts, or a state-slot cap.
*If true, this is a config fix and no code change.*

**H3 — the ring addressing is wrong and multi-lane drafting is silently
misdrafting.** The 1–3% acceptance collapse is the signature of reading the wrong
rows. A request-block-major address fixes it without changing bytes.
*If true, expect correctness first and a small speed win, not a large one.*

**H4 — the drafter's own cost is the lever, not the batching.** A tap-ring
drafter drafting N tokens in N sequential steps pays N times per round; a
block-parallel drafter pays once. *If true, the answer is architectural — a DFlash
style block-parallel drafter or a draft-KV injection — not more batching.*

---

## 5. Work plan — ordered, each step gated on the previous

### Step 0 — Measure the split before changing anything (~1 h, no code change)

Instrument one round into its stages: draft-forward, tap read, verify, accept-walk,
resample. The ledger has per-layer time; extend it to per-stage on a 4-lane round.
**Gate:** we know whether the ~12 ms attribution is real. If the 12 ms is tap
*read* bandwidth, H4 stands and batching will not help. If it is the draft forward,
H1 stands and we should attack the drafter, not the lanes.

### Step 1 — Rule out configuration (~30 min, no code change)

1. Confirm 4 lanes are genuinely 4 running requests, not queued — per-request
   metrics on, not just aggregate.
2. Confirm captured graph shapes cover every reachable row count
   `S_req = { n*(k+1) }` for n in 1..slots.
3. Confirm no state-slot cap is silently limiting concurrency.
**Gate:** H2 excluded or confirmed. If confirmed, fix and re-measure before
touching the ring.

### Step 2 — Fix the ring address, request-block-major (~1 day, code)

Two routes, in preference order:

- **Route A (cheaper, vLLM's shape):** index tap writes by the token's packed-batch
  row index and have the verifier read the same rows. Request identity resolves
  through the batch's per-request offsets; no modulo to alias. This is the smaller
  change and it removes the class of bug rather than fixing one instance.
- **Route B (SGLang's shape):** keep the ring but make the address
  `base + req*rows_per_req + f(pos % rows_per_req)`, with `req` outside the modulo,
  plus a `min_pos` floor at the prefix-cache hit for the `DraftTapNotCovered` case.

**Gate:** acceptance back to ~46% at 4 lanes with bit-identical outputs to the
single-lane reference on the same 16 prompts. Re-run the A/B; if acceptance is
right and aggregate is still flat, H1/H4 are confirmed and Step 2 bought
correctness only — record that as the finding, not as a failure.

### Step 3 — Decide whether K should track batch size (~half day, config + code)

If Step 0 shows step time growing with accept length, we are compute-bound in
verify and should cut K as batch rises — implement vLLM's
`num_speculative_tokens_per_batch_size` semantics. If step time is flat in accept
length as SGLang measured on GB10, deeper drafting is nearly free and the budget
should go to acceptance, not to lanes. **Gate:** a K-vs-batch curve on our own box.

### Step 4 — Consider the architectural lever (~week+, only if 0–3 say so)

A block-parallel drafter (DFlash-style) or draft-KV injection, so drafting costs
one forward per round instead of N sequential steps. This is the lever that gets
1.3–2.1x at concurrency 32 in the published numbers. Cost on GB10: padded draft KV
memory, ~3.3 GiB at a 2048-token batch budget. **Do not start this** until Steps
0–3 have produced a measured answer to "is the round dominated by a lanes-
proportional term?"

### Step 5 — Document and stop

Whatever the outcome, write the result into the ledger with the harness, warm/cold
state and the exact commit. The GB10 wiki's own open question — "spec on vs off at
C1/2/4/8/16/32 on one lane, same boot family, forced and natural EOS labelled,
per-request metrics on" — is the experiment this repo is now in a position to
answer for its own engine. That is worth publishing.

---

## 6. What NOT to do

- **Do not restructure the ring before Step 0.** The previous attempt failed for
  an addressing reason and cost a day; the 2x regression was the wrong diagnosis
  to chase first.
- **Do not expect batching to make the drafter cheap.** The drafter runs per
  request per round. Batching the verify pass cannot amortise work the verify pass
  did not cause.
- **Do not copy vLLM's padded-slot layout wholesale.** It is correct and it is
  what production does, but it costs memory proportional to `max_num_reqs`, and on
  a 128 GB unified-memory box with a 262k context that is the scarce resource.
  Route A gets the correctness without the padding.

---

## 7. Sources

Engine source (pinned paths, [code]):
- vLLM `draft_logits.py`, `dflash/speculator.py`, `dflash2/speculator.py`,
  `acceptance_estimator.py`, `extract_hidden_states.py`, `ngram_proposer.py`
- SGLang `dflash_draft_ring.py`, `eagle_utils.py`, `eagle_info.py`
- llama.cpp `docs/speculative.md`, `ggml-org/llama.cpp`

Vendor docs:
- vLLM speculative decoding; vLLM dynamic speculative decoding (the K-vs-batch
  schedule); SGLang speculative decoding; TensorRT-LLM speculative decoding;
  NVIDIA DGX Spark playbooks (speculative-decoding README)

Papers:
- EAGLE-3 https://arxiv.org/abs/2503.01840 (Table 3, Table 5)
- DFlash https://arxiv.org/abs/2602.06036 (Table 12, draft-KV injection, memory)
- MagicDec https://arxiv.org/abs/2408.11049 (critical sequence length, KV table)
- EAGLE-2 https://arxiv.org/abs/2406.16858
- Medusa https://arxiv.org/abs/2401.10774

GB10 field evidence:
- https://github.com/sovereignbrah/gb10-inference-wiki (02-speculative-decoding,
  16-engine-dispatch-maps, 17-sm121-hardware-facts, science/03-decode-bytes-per-step)
- https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- https://gpuwerk.com/dgx-spark/guides/dgx-spark-memory-bandwidth-measured/
- https://huggingface.co/randomllama/Qwen3.8-Flash-Next-DGX-Spark-field-notes

Bugs that mirror ours:
- SGLang #32089 — prefill hidden_state_offset not advanced for non-requesting reqs
  (cross-request hidden-state leak)
- vLLM #51538 — padded speculative slot's context length read as uint32 hung the GPU
- vLLM #52816 — related padded-slot fix
- SGLang #36889 — related
