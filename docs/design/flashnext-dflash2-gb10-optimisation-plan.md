# DFlash2 on GB10 — where the round goes, and what we have changed

Status: **executing.** Every number here is measured on this box (DGX Spark, GB10, 128 GB unified)
unless it is explicitly attributed to the vendor card. Measurements are serial, one process at a
time. Superseded numbers are kept in §7 with the reason they were wrong, not deleted — a stale
number that quietly disappears is one somebody re-derives later.

The one-line state of play: the port is at the vendor's card on acceptance (3.33 vs 3.26 accept
length) and behind it on cost per round, because the drafter pays a large fixed per-round cost that
this box's bus makes expensive. A served round is **23.8 ms** and its two biggest items are the
target's `lm_head` (6.98 ms) and `fc_ctx` (5.07 ms).

## 1. The bar

`flashnext-dflash2-vendor-baseline.md` holds the card at source
(`incoai/Qwen3.8-Flash-Next-DFlash2`, raw README, fetched 2026-11-17): one **GB300**, FP8 target,
SGLang, 3 drafts per step, GSM8K conc 1 — **AR 205.4 · MTP 326.0 (1.59×) accept 3.08 ·
DFlash2 368.2 (1.79×) accept 3.26**. Block width 4 there is a QSA serving cap, not the trained
width (the checkpoint declares `block_size: 8`).

Measured here, same 16 GSM8K prompts, pinned seeds, single stream:

| arm | tok/s | accept len | round ms | extra tokens | ms per extra token |
|---|---|---|---|---|---|
| AR | 25.5 | — | 39.2 | — | — |
| MTP | 59.6 | 3.89 | 65.3 | 2.89 | **9.0** |
| DFlash2 w8 | 45.0 | 3.34 | 74.2 | 2.34 | **15.0** |

On the card's GB300 those last-column figures are ~2.2 ms (MTP) and ~1.8 ms (DFlash2) per extra
token. So this box inflates MTP's overhead ~4× and DFlash2's ~8.5×, and the asymmetry is entirely
the drafter's fixed per-round cost. That is the thing this file is about.

## 2. How each number is obtained

Anything below can be reproduced with these four pieces. They are the harness, and a claim that
cannot be reproduced with them does not belong in this file.

**`tensorfold draft-check`** (`zig/src/cli/draft_check.zig`) — one drafter round over a pre-collected
fixture: fixture ring + drafter checkpoint + the target's embed table and head. **7.6 GB RSS, ~4 s a
rep, ~3.8 GiB device**, against ~60 s and 79 GiB for booting the full 48-layer target. It is
bit-identical to the serving path: 52 stages, proposals `{20, 1, 847, 471, 494, 348, 383}`, round
437.74 ms standalone vs 439.47 ms through the engine.

**`tools/dflash2/align_bf16/gen_inputs.py`** writes three fixtures: `block8` (**stale** — `pos_end`
9, the check requires `>= block + 2 = 10`, it refuses `FixtureIdsShape`), `window` (2048 rows, one
position) and **`sequence`** (2048 rows, 44 rounds, positions advancing 1900 → 2047 by 3–4).

**`fixture/sequence` is the one that matters.** A manifest may carry a `positions` list;
`draft-check` runs them **in order on one `State`**, so every round after the first reuses the span
the previous one projected. Rep 0 is the cold round, reps 1+ are the served ones. Without this,
every measurement is of the wrong regime — see §7.

**`tools/dflash2/regress.py`** — the gate. `capture <label> [--repeat N] [--via standalone|server]`,
`check <base> <cand>`, `list <run>`. Functional gate: every stage tensor must match the baseline
(`--func-tol`, default 0 = bf16-exact). Speed gate: per-stage ms and the round total over the
**median of `--repeat` reps** (default 3), because two captures of one binary measured 438 ms and
689 ms — GPU clock state, every stage ~+70% uniformly. Stages below `--min-ms` (1.0) are reported
but never gated; 43 of 53 are sub-millisecond with 20–150% spread.

**Knobs, all env, all off by default:** `TF_FLASHNEXT_ATTN_LEGACY=1` (old attention launch),
`TF_FLASHNEXT_HEAD_SPLIT=1` (time the head's three launches apart), `TF_FLASHNEXT_NO_SELECTOR=1`
(per-slot argmax instead of the lattice walk), `TF_FLASHNEXT_STAGE_CTX_TENSOR=1` (mirror `fc_ctx`
rows into the dump), `TF_FLASHNEXT_LG_DUMP=<dir>` (head input + fp32 logits, row 0).

## 3. The scoreboard — the served round

`fixture/sequence`, median of 11 served reps, after the changes in §4:

    cold  398.2 ms        served  23.80 ms        ~17x

| stage | ms | share | cold rep |
|---|---|---|---|
| `logits_top16_ids` — the target head, 7 rows over a 248,320 vocab | **6.98** | **26.2%** | 8.62 |
| `fc_ctx` — taps → 12800 → `fc` → `hidden_norm`, accepted rows only | **5.07** | **19.1%** | 306.68 |
| attention, 5 layers (`attn_0..4`) | 4.95 | 18.6% | 76.0 |
| mlp, 5 layers | 2.76 | 10.4% | 2.77 |
| `proposals` — the selector walk | 1.44 | 5.4% | 3.57 |
| qkv, attn_proj, convs, norms | the rest | | |

**The same numbers as a figure:** `docs/design/profiles/2026-11-18-served-round.png`, drawn by
`tools/dflash2/plot_stages.py` from a 16-rep `fixture/sequence` capture. Four views: the stages
ranked (A), categories served against cold on a log axis (B), the round in execution order as a
cumulative curve (C), and every rep (D). Panel D clips the cold round on purpose — at 17x a served
round it would otherwise flatten the run-to-run wobble the panel exists to show.

    .venv-plot/bin/python tools/dflash2/plot_stages.py tools/dflash2/runs/profile --out PNG

The script prints the whole table to stdout as well, so the figure is never the only record.
`selector_scores` (3.08 ms) appears in the dump but is a **host recomputation the harness adds**,
not part of the round — it is excluded from the share column.

The byte model behind the top two rows: the head streams **1,271.4 MB** (`248320 × 2560 × 2`) every
round for 7 rows; `fc_ctx` streams the 65 MB `fc` weight every round regardless of how few rows it
fuses. Both are pure weight-read costs, which is why the served round barely shrinks with the
context and why "fewer bytes" is the only lever that moves them.

## 4. What we have changed

**`tf_fxd2_attn_tile`** (commit `d42536604`). `tf_fxd2_attn` gave each `(row, head)` its own warp
CTA, so the 32 queries sharing a KV head each streamed the whole context from global one position
per iteration, with a `warp_sum` in the dependency chain at every position. The tiled kernel is one
CTA per KV head, a warp per `(row, head)`, K/V staged per 64-position tile in shared. Per-`(row,
head)` arithmetic is unchanged — same per-lane dot, same `warp_sum`, same position order — and the
capture is bit-identical stage for stage.

Paired in one session on `fixture/sequence`, legacy vs tiled: **0.568 vs 0.823 ms a layer, round
26.14 vs 28.58 ms (−8.5%)**. (Absolute per-layer cost moves with clock state between sessions —
0.99 ms/layer appears in the §3 run; the ratio is the claim, not the absolute.)

**`head_tile` 1024 → 8192** (commit `dc87eb516`). `TF_FLASHNEXT_HEAD_SPLIT=1` split the head, which
had been assumed to be its GEMM:

    GEMM (7 rows x 248,320 x 2,560)   5.43 ms   63%   <- 1271.4 MB at ~234 GB/s
    top-k (tf_fxd2_topf32)            2.97 ms   35%   <- head_tile 1024
    merge (tf_fxd2_head_merge)        0.19 ms    2%

The top-k was oversubscribed by construction: a 1024-wide tile against 256 threads leaves every
thread inserting 4 values, then a warp leader merging 512 entries **serially**, then thread 0
merging 128 — three barriers a row, 243 tiles × 7 rows. `head_tile = 8192` (31 tiles) cuts it to
**1.39 ms** with the GEMM untouched. `regress.py check base headtile`: **PASS, functionally
identical**; cold round 439.47 → 427.08 ms; served head 8.60 → 6.98 ms, served round 25.5 → 23.8 ms.

## 5. Closed — do not re-run

Each of these cost real time. The reason is recorded so it is not re-derived.

- **Block width 4.** Zero per-slot quality gain (0.720 / 0.667 / 0.542 vs 0.740 / 0.662 / 0.592),
  loses 12.5% advance, saves ~8 ms of a ~76 ms round, nets ~3% slower. Also not a config knob:
  `cuda_draft2_fwd.zig` has `pub const block = 8` and refuses any other value
  (`DraftBlockUnsupported`).
- **The candidate selector as the acceptance gap.** Engine +0.05 accept length on vs off, reference
  +0.06. The vendor's +0.34/+0.47 ablation is a Qwen3.5-4B number and does not transfer.
- **`mean4` vs a learned hyper-connection mix.** The vendor capture patch is authoritative: DFlash2
  was trained on `hidden_states.unflatten(-1, (hc_count, hidden)).mean(dim=-2)`, `hc_mean=True`,
  features 12800. Changing it moves the drafter off its training distribution.
- **Cross-round context, context semantics, 10% tap noise, engine-vs-reference ceiling**
  (1.317 vs 1.342 tokens/round) — all measured and closed in
  `flashnext-dflash2-acceptance-log.md` §5.
- **Draft-vocabulary reduction as a byte win over the full head** — see §6: the head is now the
  largest single item, but the reduction that practitioners use needs a retrained or mapped draft
  vocabulary; it is not a serving-side switch. What is live is the head's *precision*.
- **The `attn_i` stage is not just attention.** `layer()` markers put norm+rope and `prefix()`
  inside that window, so a kernel-only change can look like it did nothing.

## 6. Open, ranked

1. **The head's GEMM, 5.43 ms.** ~234 GB/s over 1,271.4 MB is already near this box's rate, so the
   only way down is fewer bytes: a block-fp8 head (the `fp8.zig` 128×128 path exists) for the
   ranking pass, with the top-N rescored in bf16 so the selector's unary values stay exact.
   **This is an acceptance risk, not just a speed change** — top-16 membership then depends on fp8
   ranking. Validate offline first: exact bf16 top-16 vs fp8-ranked top-16 over many rows, on
   captured head inputs, before the engine path is touched.
2. **The top-k's remaining 1.39 ms** — still the serial warp-leader merge. A shuffle-based top-16
   merge should take it to a few tenths. Exact, no acceptance risk.
3. **`fc_ctx`, 5.07 ms for ~4 rows**, streaming 65 MB at ~13 GB/s — 11× below what the head kernel
   achieves on the same box, and 19% of the round. This is the clearest inefficiency we have not
   explained yet.
4. Layer block projections: mlp 2.76 ms, qkv/attn_proj the remainder of the round.

## 7. Superseded measurements, and why they were wrong

**The 438 ms profile (2026-11-17) was the cold round.** It reported `fc_ctx` 75.7%, attention 19.0%,
head 1.9%, and drove a plan that put attention first and dropped the head as "a 2% ceiling". All of
it described a fresh `State`: the fuse and every layer's prefix k/v projection run over the whole
2047-row window, which happens once per stream, not once per round.

The reuse rule is `fuseFrom(lo, hi, kv_lo, kv_end) = if (lo >= fused_lo and hi > fused_end)
max(lo, fused_end) else lo`. Replaying one fixture position gives `hi == kv_end`, which the engine
treats as a rewind and deliberately re-fuses in full — so **a single-position fixture cannot
measure the served regime at all**. Measured, not assumed: two warm rounds on one state over
`fixture/window` still read **435.8 ms against 438.0 cold**.

Consequence: `fixture/sequence` was built, and on it the same stages read 30.5% (head) and 18.2%
(`fc_ctx`) instead of 1.9% and 75.7%. Two of the three conclusions in the old §4.1 were inverted by
the regime change, which is why the harness now comes before the kernel work rather than after it.

**The 25.5 ms / 8.60 ms served figures** were measured before `head_tile = 8192`; §3 supersedes
them with 23.80 / 6.98.
