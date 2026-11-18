# DFlash2 on GB10 — where the round goes, and what we have changed

Status: **executing.** Every number here is measured on this box (DGX Spark, GB10, 128 GB unified)
unless it is explicitly attributed to the vendor card. Measurements are serial, one process at a
time. Superseded numbers are kept in §7 with the reason they were wrong, not deleted — a stale
number that quietly disappears is one somebody re-derives later.

The one-line state of play: the port is at the vendor's card on acceptance (3.33 vs 3.26 accept
length) and behind it on cost per round, because the drafter pays a large fixed per-round cost that
this box's bus makes expensive. A served round on the published bf16 drafter is **18.75 ms**, and its
two biggest items are the target's `lm_head` (6.99 ms, 37.5%) and the five attention layers (4.45 ms,
23.9%). The context fuse, second at 5.10 ms a day ago, is now 0.71 ms (§4) — which moved weight-read
work from 54% of the round to 67% and made precision, not kernels, the next fork.

That fork is now taken: a second drafter checkpoint with the seven decoder linears in weight-only FP8
takes the same binary to **16.02 ms** (−14.5%) on this fixture, with the bf16 arm provably unchanged
(§4.3; artifact, flag and cost in `flashnext-dflash2-fp8-checkpoint.md`). It moves proposals — 64 of
112 slots — so it is **not a shipped default yet**: acceptance through a served stream is the gate,
and that run has not happened.

## 1. The bar

`flashnext-dflash2-vendor-baseline.md` holds the card at source
(`incoai/Qwen3.8-Flash-Next-DFlash2`, raw README, fetched 2026-11-17): one **GB300**, FP8 target,
SGLang, 3 drafts per step, GSM8K conc 1 — **AR 205.4 · MTP 326.0 (1.59×) accept 3.08 ·
DFlash2 368.2 (1.79×) accept 3.26**. Block width 4 there is a QSA serving cap, not the trained
width (the checkpoint declares `block_size: 8`).

Measured here, same 16 GSM8K prompts, pinned seeds, single stream:

| arm | tok/s | accept len | round ms | extra tokens | ms per extra token |
|---|---|---|---|---|---|
| AR | 25.0 | 1.004 | 40.2 | — | — |
| MTP | 53.0 | 3.505 | 67.1 | 2.50 | **10.7** |
| DFlash2 w8, bf16 | 48.1 | 3.387 | 69.6 | 2.38 | **12.2** |
| DFlash2 w8, fp8 | 49.4 | 3.322 | 66.8 | 2.32 | **11.5** |

Re-measured 2026-11-18 on the current binary, same 16 prompts, same seeds, one server at a time,
each arm's numbers from the server's own `done chatcmpl` lines (`~/.hermes/cache/scratch/e2e/`).
**All four arms produced identical completion hashes on 16 of 16 prompts** — verification is
lossless, so drafting cannot change the output, and the fp8 checkpoint included.

Against the previous run of this table: DFlash2 **45.0 → 48.1 tok/s** at bf16 (the fuse fix, worth
4.6 ms a round end to end) and **→ 49.4** with the fp8 checkpoint. Paired on the same prompts, fp8
against bf16 is **+1.31 tok/s (95% CI +0.61 to +2.01)** — real — while acceptance is **−1.94%
(CI −3.54% to −0.33%)** — also real, small, and in the direction the selector predicted: the round
is 2.8 ms cheaper and the candidate set is slightly worse, and the 2.8 ms wins.

**MTP reads 53.0 here against the 59.6 recorded above, at an unchanged round cost (67.1 vs 65.3 ms),
so the gap is acceptance (3.505 vs 3.89), not speed.** Nothing in the fp8 work touches the MTP path,
and the AR arm is flat (25.0 vs 25.5), so this is either a different measurement regime in the older
row or a genuine acceptance regression on the default path. **Unexplained — do not quote MTP as
unregressed until it is bisected.**

On the card's GB300 those last-column figures are ~2.2 ms (MTP) and ~1.8 ms (DFlash2) per extra
token. So this box inflates MTP's overhead ~4.9× and DFlash2's ~6.8× at bf16 (6.4× at fp8) — the
asymmetry is entirely the drafter's fixed per-round cost, and the fuse fix plus fp8 have moved it the
right way without closing it. That is the thing this file is about.

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

`fixture/sequence`, median of 15 served reps (`~/.hermes/cache/scratch/verify-fuse`, 2026-11-18):

    cold  284.00 ms        served  18.75 ms        ~15x

The same capture against the pre-fuse binary read **22.99 ms** (sd 0.95), so the 4.24 ms difference is
the change in §4, not clock state; the post-fuse spread is 0.77 ms.

| stage | ms | share | cold rep | cold÷served |
|---|---|---|---|---|
| `logits_top16_ids` — the target head, 7 rows over a 248,320 vocab | **6.99** | **37.5%** | 7.00 | 1.0× |
| attention, 5 layers (`attn_0..4`) | 4.45 | 23.9% | 76.72 | 17× |
| mlp, 5 layers | 2.75 | 14.7% | 2.86 | 1.0× |
| `proposals` — the selector walk | 1.42 | 7.6% | 4.25 | 3.0× |
| convs (10) | 0.87 | 4.7% | | 1.0× |
| qkv, 5 layers | 0.84 | 4.5% | | 1.0× |
| `fc_ctx` — taps → 12800 → `fc` → `hidden_norm` | **0.71** | **3.8%** | 190.56 | 268× |
| attn_proj, 5 layers | 0.54 | 2.9% | | 1.0× |
| norms, block residual, block hidden | 0.06 | 0.3% | | |

Stage sum 18.63 against a round of 18.75 → 0.12 ms of launch gaps.

**The round is now 67% weight bytes, measured.** Checkpoint bytes per group against the served ms:

| group | bytes a round | ms | GB/s |
|---|---:|---:|---:|
| head GEMM (7 rows × 248,320 × 2,560) | 1,271.4 MB | 5.43 | 234 |
| mlp | 589.8 MB | 2.75 | 214 |
| selector codebooks | 255.6 MB | 1.42 | 180 |
| qkv | 157.3 MB | 0.84 | 187 |
| attn_proj | 104.9 MB | 0.54 | 194 |
| `fc`, once a chunk (§4) | 65.5 MB | 0.71 | 92 |
| convs | 33.0 MB | 0.87 | 38 |

MLP, qkv, attn_proj and the head GEMM all sit in a 187–234 GB/s band: **there is no kernel rewrite
that makes them faster, only fewer bytes.** The head's top-k (1.39 ms of its 6.99) is the one
non-byte part of that group. The genuinely non-byte work is attention (4.45, K/V for 8 query rows
over ~1,900 keys — a tiny-M problem, not a bandwidth one), convs (0.87 at 38 GB/s, latency-bound)
and `fc_ctx` (0.71).

**The same numbers as a figure:** `docs/design/profiles/2026-11-18-served-round-post-fuse.png`
(`2026-11-18-served-round.png` is the pre-fuse capture, kept for the comparison), drawn by
`tools/dflash2/plot_stages.py`. Four views: the stages ranked (A), categories served against cold on
a log axis (B), the round in execution order as a cumulative curve (C), and every rep (D). Panel D
clips the cold round on purpose; panel B's cold multiplier is computed from the data, because it
moved from 60× to 268× when the fuse stopped re-reading its weight.

    .venv-plot/bin/python tools/dflash2/plot_stages.py ~/.hermes/cache/scratch/verify-fuse --out PNG

The script prints the whole table to stdout as well, so the figure is never the only record.
`selector_scores` (3.08 ms) appears in the dump but is a **host recomputation the harness adds**,
not part of the round — it is excluded from the share column.

End to end, now measured through the server rather than by arithmetic (§1): the bf16 round is
**69.6 ms, 48.1 tok/s** (from 74.2 / 45.0), and the fp8 checkpoint takes it to **66.8 ms, 49.4 tok/s**
with acceptance 3.322 against 3.387 — a real −1.94% accept for a real +2.7% throughput.

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

**The context fuse reads its weight once per chunk** (2026-11-18). `tf_fxd2_fuse` launched **one block
per row** and every block streamed the whole 65.5 MB `fc` weight, so the cost was `rows × 65.5 MB`: 4
rows = 262 MB = 5.10 ms, 1,900 rows = 124 GB = 307 ms. `glm_fuse_dot` instead runs one CTA per 64
output columns with **all rows of the chunk inside it**, staging the weight k-major into shared a
256-k segment at a time. Each dot is the same scalar `fmaf` walk over K in ascending order and the
same single `__float2bfloat16_rn`; the `rms_norm` moved out into `tf_fxd2_rmsnorm` over the bf16 rows.
It is the old arithmetic with the weight's reuse fixed, not a reassociated one — which is why it is
bit-exact where the tensor-core version is not (§5).

Verified independently of the agent that wrote it, against a baseline capture of the old binary:
proposals identical on **16/16** reps, every stage tensor bit-identical, `regress.py check base fuse`
PASS.

| | before | after |
|---|---:|---:|
| `fc_ctx`, served | 5.10 ms | **0.71 ms** |
| served round | 22.99 ms | **18.75 ms** |
| `fc_ctx`, cold | 307.03 ms | 190.56 ms |
| cold round | 399.94 ms | 284.00 ms |

Cold improves only 1.6×: the bit-exact scalar dot is instruction-bound (~11 fma/cycle/SM against 128),
so 1,900 rows still cost 190 ms. That is first-token latency, not throughput, and it is the one place
a non-exact kernel is tempting — see §5 for why it was refused.

### 4.3 A second checkpoint: weight-only FP8 on the decoder linears

The seven linears that dominate the byte bill (`mlp.gate/up/down`, `self_attn.q/k/v/o`, 852 MB of
bf16) now have an e4m3 build with 128×128 block scales, bf16 activations and fp32 accumulate —
**W8A16**, not W8A8: at 8 rows the activations are 205 KB a layer against 590 MB of mlp weights, so
quantizing them would buy 0.03% of the traffic and add risk. `fc`, the convs, every norm and the
selector stay bf16 in both arms. The artifact is a separate 780.3 MB checkpoint
(`checkpoints/dflash2-fp8/`, manifest + `SHA256SUMS` committed, the blob gitignored), selected with
`--drafter-precision bf16|fp8`; omitted, the checkpoint's own tensors decide, and mismatches are
refused at load. Full detail: `flashnext-dflash2-fp8-checkpoint.md`.

Same binary, same session, `fixture/sequence`, 15 served reps each:

| stage | bf16 | fp8 | delta | |
|---|---:|---:|---:|---|
| mlp (5) | 2.757 | 1.544 | −1.213 | −44.0% |
| qkv (5) | 0.834 | 0.532 | −0.302 | −36.3% |
| attn_proj (5) | 0.546 | 0.379 | −0.167 | −30.6% |
| attention (5) | 4.385 | 3.380 | −1.005 | −22.9% |
| head, `fc_ctx`, `proposals` (untouched) | 9.14 | 9.14 | ±0.02 | noise |
| **served round** | **18.731** | **16.019** | **−2.712** | **−14.5%** |
| cold round (rep 0) | 284.167 | 216.039 | −68.128 | −24.0% |

**1.69 ms of the 2.71 is weight bytes; 1.00 ms is attention, which the arm does not touch.** The likely
mechanism is cross-stage clock coupling — bf16 mlp moves twice the bytes, the SoC clocks down, tiny-M
attention follows — and the evidence fits without proving it: fp8 attention held 0.675–0.681 ms/layer
across two runs in opposite order while bf16 attention read 0.87 then 1.10. Recorded as a hypothesis,
not as a kernel win.

The bf16 arm is gated bit-identical (`regress.py check base base-bf16`: PASS, every stage tensor
equal, tol 0). The fp8 arm is **not** and cannot be: 48 of 112 proposal slots are identical, 64 differ,
one rep fully identical and one fully divergent, with hidden drift median 7.4e-3 relative L2. That is
the selector amplifying a 2.6% weight error, exactly as §5 predicted, and it is why acceptance — not
proposal identity — is this change's gate. It has not been measured through a served stream yet.

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
- **Draft-vocabulary *retraining* as a byte win.** A drafter with its own smaller vocabulary needs a
  new embedding/head and a fine-tune; it is not a serving switch. What is live and different is a
  **serving-side trim** — compute the head only over a frequency-ranked 65,536 subset of the same
  vocabulary — which is §6 item 2 and is measured elsewhere on this box.
- **A tensor-core GEMM for the context fuse.** `glm_mm16_64` over a tiled `fc` was built and measured
  first, and it is *faster than the shipped version*: 0.33 ms warm and 9.00 ms cold, against 0.71 and
  190.56. It was refused because it **changed proposals on 7 of 16 reps** (rep 0: `{4173,29,198,16,13,
  1228,1}` → `{91,7925,18453,…}`) on a 1.7e-7 relative-L2 shift in the fused rows, with downstream
  movement ≤4e-3. The selector amplifies sub-bf16 noise into a different candidate set. Two
  consequences worth keeping: bit-exactness is only reachable by matching the scalar summation order,
  and **acceptance length per slot over many prompts is the only honest gate for a precision change** —
  proposal identity is the gate for a refactor, not for quantization.
- **`fc_ctx` at 13 GB/s as an unexplained access-pattern problem.** It was never bandwidth: the kernel
  re-read the 65.5 MB weight once per row, so the "rate" was `rows × weight` amortised over a 4-row
  round. Fixed in §4. Do not reopen it as a layout or bus question.
- **The `attn_i` stage is not just attention.** `layer()` markers put norm+rope and `prefix()`
  inside that window, so a kernel-only change can look like it did nothing.

## 6. Open, ranked

The external evidence for every option below is in `flashnext-dflash2-quantization-survey.md`, with
its grade ([M-GB10] measured on a GB10 by someone else, [V] vendor, [P] paper) and 99 cited sources.
Its byte arithmetic was computed against the pre-fuse 22.99 ms round; the ordering below is rebased
onto §3, and two of its conclusions move because of that.

0. **The exact set, first, because it is free.** The shuffle-based top-16 merge (1.39 ms → a few
   tenths) and the convs (0.87 ms at 38 GB/s, 5× off their floor, latency-bound). **~1.5 ms, zero
   acceptance risk.** The survey agrees and ranks the merge in its own list.
1. **FP8 W8A8 on the decoder linears — mlp + qkv + attn_proj, 852 MB → 426 MB.** ~2.07 ms, and the
   survey's highest-confidence item: two independent GB10 measurements on this drafter family at K=7
   with acceptance **4.24 → 4.24 unchanged** (one reports 4.16 for INT8 as indistinguishable). Our
   stages run at 187–214 GB/s, so halving bytes should translate; the unmeasured risk is that our fp8
   path is a 128×128 block-scaled kernel and its dequant overhead may not amortise at 8 rows.
   **Built and measured (§4.3): 1.69 ms, not 2.07 — the dequant overhead does not fully amortise at
   8 rows, and mlp lands at 44% cheaper rather than 50%.** What is left of this item is not speed but
   the acceptance question, which needs the served A/B.
2. **The head's GEMM, 5.43 ms — the largest single byte block, 1,271.4 MB at 234 GB/s.** At the box
   rate, so only fewer bytes move it, and the survey has two scoped routes:
   - **FP8 head** (635.7 MB, ~2.71 ms) **after** an offline per-slot top-16 fidelity check. Its
     evidence says the damage lands on low-confidence tokens — our slots 5–7 at 0.389/0.429/0.333 —
     while NLL stays loss-neutral. The check is cheap and decides this item either way.
   - **Frequency-ranked trim to 65,536** (335.5 MB, ~5.0 ms including top-k). Largest win available
     without touching the selector, but it needs a rank list from *this* model's own generations:
     published rank lists lose ~12 points when transferred between models.
3. **Attention, 4.45 ms.** Not bandwidth: K/V for 8 query rows over ~1,900 keys is ~39 MB a round.
   It is a tiny-M occupancy problem, worth ~3 ms and the most work of anything on this list.
4. **NVFP4 on the decoder linears** — ~0.8 ms beyond FP8, only after FP8 has landed and been measured.

**Ruled out, with reasons, so they are not proposed again:** quantizing `fc` (now one 65.5 MB read at
92 GB/s — precision is not its axis), the selector codebooks (it *is* the acceptance mechanism, and
the vendor keeps it FP16 when a candidate head is present), convs by precision (latency-bound),
SmoothQuant / QuaRot / SpinQuant (zero byte saving at weight-only), activation quantization (W4A16
beats W4A4 single-stream on this box by ~26 points on MBPP/GSM8K), 2-bit, and calibration pipelines
whose whole effect is inside the 2.5% session-to-session drift.

**Where the survey's ranking changes under §3's numbers.** It put `fc_ctx` fifth on the grounds that
5.10 ms at 13 GB/s was the largest recoverable block; that is done, and its estimate was conservative.
And because the fuse fix removed 4.24 ms of *non-byte* cost from a fixed total, weight-read work is
now **67% of the round instead of 54%** — the quantization case is stronger than the survey could see,
not weaker. What has not changed is the scale correction: the drafter is 18.75 ms of a ~70 ms round,
so 2.07 ms of drafter body is ~3% end-to-end while 2.71 ms of head is ~3.9%. Per unit of risk the
body wins; per millisecond the head does.

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
them with 23.80 / 6.98, and §4 supersedes *those* with 18.75 / 6.99.

**"`fc_ctx` runs at 13 GB/s — an unexplained access-pattern inefficiency."** Superseded 2026-11-18.
The 13 GB/s was never a bandwidth measurement: the kernel read the whole 65.5 MB weight **once per
row**, so the effective rate was `rows × 65.5 MB` over a 4-row round. The stage is now 0.71 ms (§4) and
there is no layout mystery left in it.

**"Weight bytes are only 55% of our round"** (survey §2, computed on the pre-fuse 22.99 ms round).
After §4 it is **67%**: the fix removed 4.24 ms of non-byte cost from a fixed total, so the same byte
cost is a larger share. Any arithmetic built on the 55% figure understates what precision work buys.

**"The served round is ~23.0 ms"** — the state of play from the plan's own header until 2026-11-18.
Current: **18.75 ms**, acceptance unchanged at 3.34.
