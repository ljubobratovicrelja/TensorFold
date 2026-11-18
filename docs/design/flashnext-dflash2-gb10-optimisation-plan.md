# DFlash2 on GB10 — findings and optimisation plan

Status: **executing.** Stages below are run serially, one server on :8099 at a time, results
recorded in §4 as they land. Every number in this file is measured on this box unless marked
otherwise.

## 1. Why this file exists

The vendor card's numbers were produced on one GB300 (HBM3e). We serve on one GB10 (LPDDR5X
unified). Section 3 of `flashnext-dflash2-vendor-baseline.md` establishes that our port is at
the card on the vendor's own metric — 1.72–1.76× vs 1.79× over autoregressive — and that the
reason DFlash2 loses to our MTP here is **cost per round, not correctness**. This file is about
that cost.

## 2. Findings

### 2.1 The memory system, measured by others

- GB10 theoretical peak **273 GB/s** (LPDDR5X, 8533 MT/s); ~229–238 GB/s achieved on large
  buffers on an idle node (84–87% of peak).
- **Under active decode the bus collapses.** Measured copy bandwidth falls to **~87 GB/s**
  (−55%) and triad to ~39 GB/s (−60%), with an ~80–90 GB/s contention floor for any active
  decode workload regardless of model size (NVIDIA developer forums, UMA diagnostics on GB10).
- Consequence for us: the drafter does not run on a 273 GB/s bus, it runs on whatever is left of
  a ~87 GB/s bus while the target streams its own weights. Bytes are the currency.

### 2.2 Where our round spends them

Measured, same harness, 16 GSM8K prompts, single stream:

| arm | tok/s | accept len | round ms | overhead vs AR step | extra tokens | ms per extra token |
|---|---|---|---|---|---|---|
| AR | 25.5 | — | 39.2 | — | — | — |
| MTP | 59.6 | 3.89 | 65.3 | 26.1 | 2.89 | **9.0** |
| DFlash2 w8 | 45.0 | 3.34 | 74.2 | 35.0 | 2.34 | **15.0** |

On the card's GB300 the same quantities are ~2.2 ms (MTP) and ~1.8 ms (DFlash2) per extra token.
So this box inflates MTP's overhead ~4× and DFlash2's ~8.5×. The asymmetry is the whole problem:
the drafter has a large **fixed** per-round cost that does not shrink with the target's weights.

Byte budget of that fixed cost:

- drafter weights: 1.2 GB bf16 (5 layers, hidden 2560)
- **target lm_head applied to 8 draft rows: 248,320 × 2560 bf16 ≈ 1.27 GB**
- candidate selector: two codebooks `[vocab, 256]` bf16 ≈ 254 MB, plus the `[rank, hidden]`
  projection — and we measured the selector as **inert** (+0.05 accept length), so this traffic
  buys nothing.

The head is comparable to the entire drafter, and it is the single component the
`candidate_selector` was designed to avoid.

### 2.3 What practitioners do on GB10

- **veloGB10** — a Rust+CUDA engine that runs *Qwen3.8-Flash-Next from NVFP4 packs with DFlash 2*
  on GB10. Its stated design rule: *"Optimization here means fewer bytes (NVFP4, fused
  projections, frequency-ranked draft vocabularies), not fewer launches."*
- **TensorRT-Edge-LLM** — draft quantization in `fp8`, `int4_awq`, `nvfp4`, `int8_sq`. For
  DFlash-family drafts it quantizes **only the decoder-layer linears**; `fc`, `lm_head`, the
  low-rank candidate head and the draft KV cache stay FP16. It warns quantized drafts can lose
  acceptance, so each arm must be measured.
- **Swift-1.5 recipe, Qwen3.8-Flash-Next on DGX Spark** (our model, our box) — MTP with a
  **65,536-token draft vocabulary**, ~43 tok/s single stream.

### 2.4 The inversion is not a hardware law

On the same hardware, 27B DFlash2 beats MTP single-stream: 38.0 vs 28.6 tok/s (ALLSmartLLM,
DGX Spark) and 41.2 vs 12.6 in the QAI vLLM-vs-SGLang race. MTP only wins from 4 concurrent
requests up. So a DFlash2 that loses to MTP at concurrency 1 on GB10 is a cost problem we can
attack, not a property of the machine.

## 3. Plan

Each stage is a separate measurement arm, serial, box verified idle, one server on :8099,
numbers from the server's own `done chatcmpl` line. Nothing is adopted unless it wins on
measured tok/s **and** holds acceptance.

1. **Profile the round by stage.** Drafter layers / head+top-k / selector / verify, in ms.
   Deliverable: a stage table, so stages 2–4 are aimed rather than guessed.
2. **Draft-vocabulary reduction.** Score the draft with a frequent-token shortlist instead of the
   full 248,320 head; verify against the full head so decoding stays lossless. Expected to be the
   largest byte win; both GB10 engines do this.
3. **Quantize the drafter's decoder layers (NVFP4/FP8), keep `fc`, head, candidate head and draft
   KV in FP16.** The Edge-LLM recipe. Acceptance measured per arm — the vendor literature is
   explicit that it can drop.
4. **Fused projections** — only if the profile still shows byte or launch overhead worth removing.

## 4. Results

### 4.1 Stage 1 — profile (DONE)

Harness: `tools/dflash2/align_bf16/run_engine_dflash2.sh` on the repo's `fixture/window`
fixture (2048-row context, the full served window), engine's own stage collector,
`stages.json` with 53 stages. The `fixture/block8` case is stale — `pos_end` 9 vs the check's
required `>= block + 2 = 10` — it refuses with `FixtureIdsShape`; regenerate before use.

Round total **438.0 ms**; stage sum 441 ms (the `.npy` mirror of `fc_ctx` excluded).

| stage | ms | share |
|---|---|---|
| `fc_ctx` (taps → 12800 → fc → hidden_norm, whole window) | **333.8** | **75.7%** |
| attention, 5 layers (16.5–16.9 each) | **83.7** | **19.0%** |
| head (8 rows over 248,320 vocab, `logits_top16_ids`) | 8.4 | 1.9% |
| selector walk (`proposals` + `selector_scores`) | 8.9 | 2.0% |
| qkv, mlp, attn_proj, convs, norms (all 5 layers) | 5.3 | 1.2% |

Two findings that change the plan:

1. **The head is 1.9%, not the bottleneck.** The 248,320 × 2560 lm_head over 8 rows — which the
   byte model in §2.2 flagged as "comparable to the whole drafter" — measures at 8.4 ms. Stage 2
   as written (draft-vocabulary reduction) can remove at most ~2% of a round. It is not worth the
   engineering or the acceptance risk.
2. **Attention is ~330× its bandwidth floor.** 8 query rows against ~2048 cached keys X 8 kv
   heads X 128 dim is ~4.2 MB per layer — **0.05 ms at the ~87 GB/s decode bus**. It measures
   **16.7 ms per layer**. That is implementation-bound (one warp serially sweeping the key ring),
   not hardware-bound, and it is 19% of the round before `fc_ctx` is even considered.

`fc_ctx` at 76% is the full-window recompute the fixture performs; serving runs the context path
incrementally, so a serving round is 74 ms rather than 438 ms. Its share in serving is therefore
smaller — but attention is paid **every round, at full window**, because block rows attend to the
whole cached context, so the attention finding carries over to serving unchanged.

**Revised order:** attention kernel first (19% of the round, 330× off floor), then `fc_ctx`
incremental/recompute behaviour, then quantization of the drafter's decoder layers. Draft-vocab
reduction is dropped — a 2% ceiling does not justify it.


## Correction (2026-11-18): the stage profile above is the COLD round, not the served one

The profile attributed 95% of the round to `fc_ctx` (75.7%) and attention (19.0%). Both numbers
come from a **cold** round — a fresh `State`, so the fuse and every layer's prefix k/v projection
run over the whole 2047-row window. A serving round is warm: `fuseFrom(lo, hi, kv_lo, kv_end)`
returns `max(lo, kv_end)` once `hi > kv_end`, so a round advances the position by the accepted run
and only re-fuses/re-projects those ~`accepted + 1` rows.

Measured, not assumed: two warm rounds on the same state over `fixture/window` still report
**435.8 ms against 438.0 cold**, because replaying one fixture position gives `hi == kv_end` and the
engine deliberately re-fuses a rewound prefix. So **a single-position fixture cannot measure the
served steady state**, and the 76%/19% split must not be used to rank optimisation work.

Consistent with that: the served DFlash2 round is ~74 ms at ~1600 ctx, 6x below the fixture's 438 ms.

Consequence for the plan:

- Attention is **not** the first lever. A tiled rewrite (`tf_fxd2_attn_tile`: one CTA a KV head, a
  warp a (row, head), K/V staged in shared instead of 32x from global) is bit-identical and only
  marginal on the cold path — because the `attn_i` stage window covers the layer's **prefix k/v
  projection** as well as the attention kernel (see `layer()`'s stage markers).
- Profiling the served round needs a fixture whose **positions advance** across rounds, carrying
  real taps for each. That is the next harness change, before any further kernel work.

## The served round, measured (2026-11-18) — this is the scoreboard that counts

`fixture/sequence` (new): 44 rounds on **one** `State`, positions advancing 1900 → 2047 by 3–4,
the shape a decode has. Rep 0 is the cold round, reps 1+ are the served ones.

    cold  400.1 ms      served  25.5 ms (median of 11)      16x

Stage medians over the served reps, and the same stage in the cold rep:

    logits_top16_ids     8.60 ms  30.5%    8.62      <- the head, 8 rows over a 248,320 vocab
    fc_ctx               5.14 ms  18.2%  306.68      <- fuses only the accepted rows; the weight read dominates
    attn_0..4            5.05 ms  17.9%   76.0       <- 1.01 ms/layer
    mlp_0..4             2.76 ms   9.8%    2.77
    proposals            1.32 ms   4.7%    3.57
    (selector_scores is a HOST recomputation the harness adds, not part of the round: 3.01 ms)

### What this reverses

- **The head is the first lever, not the last.** It was 1.9% of the cold round and is 30.5% of the
  served one, because the cold round's fuse swamps it. Draft-vocabulary reduction — the lever I
  dropped on the cold profile — is back on top: the kernel streams the whole 1.27 GB matrix every
  round for 8 rows, at a measured ~148 GB/s (so ~8.6 ms is already near the machine's rate; the win
  has to come from reading fewer bytes, not from a faster kernel).
- **`fc_ctx` is 5.14 ms for ~4 rows.** It streams the 65 MB `fc` weight per round at ~13 GB/s — 11x
  below what the head kernel achieves on the same box. That is a real inefficiency, and it is 18%
  of the served round.
- **Attention is done.** `tf_fxd2_attn_tile` measured in the served regime: **0.568 vs 0.823 ms a
  layer** (legacy via `TF_FLASHNEXT_ATTN_LEGACY=1`), round **26.14 vs 28.58 ms, −8.5%**, and the
  capture is bit-identical stage for stage. Its earlier "no effect" reading was the cold fixture.

### Ranked next work

1. Draft vocabulary / head bytes (30.5%).
2. `fc_ctx` weight-read efficiency (18.2%).
3. Layer block projections (mlp 9.8%, qkv/attn_proj the rest).

### The head, split (2026-11-18)

`TF_FLASHNEXT_HEAD_SPLIT=1` times the head's three launches apart. Served rounds, 2048-row window:

    GEMM (7 rows x 248,320 x 2,560)   5.43 ms   63% of the head   <- streams 1271.4 MB at ~234 GB/s
    top-k (tf_fxd2_topf32)            2.97 ms   35%               <- head_tile 1024
    merge (tf_fxd2_head_merge)        0.19 ms    2%

So "the head is the GEMM" is false: a third of it was the top-k, which is oversubscribed by
construction — `head_tile = 1024` with 256 threads means each thread inserts 4 values, then a
warp leader merges 512 entries **serially**, then thread 0 merges 128, three barriers a row.

`head_tile = 8192` (31 tiles instead of 243) removes most of that: **top-k 2.97 -> 1.39 ms**, GEMM
unchanged, capture bit-identical, cold round 439.47 -> 427.08 ms. What is left is still the serial
warp-leader merge; a shuffle-based top-16 merge is the next step, worth ~1 ms a round.

The GEMM's 5.43 ms is ~234 GB/s over 1271.4 MB — already near this box's rate, so the only way down
is fewer bytes: an fp8 (or lower) head for the ranking pass, rescoring the top-N in bf16. That is
an acceptance risk, not just a speed change, and it needs the tap-replay harness to clear it.
