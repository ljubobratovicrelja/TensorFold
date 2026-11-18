# DFlash2 experiment ledger (2026-11-18)

Status: **LIVE.** One page that answers "have we tried that?" before anyone runs it again.
Each entry is an experiment, its measured result, and its verdict. This file indexes the chronicles; it
does not replace them — every row names where the numbers live.

Verdict vocabulary:

- **FIXED** — a real defect; the fix is in and measured.
- **CORRECT** — the thing was suspected and is right; nothing to do.
- **NULL** — a real A/B whose arms did not differ enough to matter.
- **WRONG** — a hypothesis that was measured and refuted. Do not re-open.
- **SUPERSEDED** — a measurement that was taken in a regime we no longer measure in.
- **OPEN** — no decisive measurement exists yet.

Read with `docs/design/README.md`, which carries document status and the stale-claim list.

## 1. Numerics — the drafter's forward

| experiment | result | verdict | where |
|---|---|---|---|
| Head GEMM / detile / `lm_head` weights | logits match a numpy product of the same rows to 2.6e-07; detile 64/64 blocks row-for-row | CORRECT | acceptance-log §2 |
| Head top-k published one lane per warp | every candidate id ≡ 0 mod 32; 31/32 of the vocabulary unproposable, silently. Fixed; proposals now exactly the reference's `{248058,198,27,1628,21402,956,29}` | FIXED | acceptance-log §2 |
| Tap basis: `readout` vs `mean4` | 48/103 vs 49/103 accepted, one binary, basis flipped by env only. `mean4` is the training contract (`hidden_states.unflatten(-1,(hc_count,hidden)).mean(dim=-2)`, vendor patch + vLLM port). Also found `TF_FLASHNEXT_TAP_SOURCE` was inert (comptime constant) | NULL | acceptance-log §3 |
| Layer→tap mapping `[5,15,25,35,45]` vs ±1 | 1.450 / 1.433 / 1.450 mean advance; shipped set wins or ties every column | CORRECT | acceptance-log §7 |
| Selector contribution (engine + reference arms) | engine +0.05 accept length, reference +0.06 on the same dump. Vendor's +0.34/+0.47 is Qwen3.5-4B and does not transfer | NULL | vendor-baseline §4 |
| Context window shift −1 / 0 / +1 | 1.275 / 1.325 / 1.325 teacher-forced advance | WRONG (offset) | handoff §7 |
| Block attention mask ignored the sliding window | row *m* attended *m* columns it may not see. Fixed; top-1 proposal moved 220 → 20. Not the divergence source | FIXED | divergence-brief §10 |
| Fusion rounding (`tf_fxd2_fuse`, `tf_fxd2_rmsnorm` kept fp32 across two module boundaries) | `fc_ctx_rows` rel-L2 3.29e-3 → 4.53e-4, exact 65.5% → 99.08%. The one real numeric defect | FIXED | divergence-brief §3–§4, §9 |
| The residual below `conv_attn_in_0` | engine's dense matches cuBLAS to 2.28e-4 / 99.8% exact; the last ULP is the deliberate pinned FMA order that makes `token_sha256` reproducible | CORRECT (not a defect) | divergence-brief §10 |
| Torch forward parity (trainer) | 216/224 proposal slots byte-identical, per-slot acceptance matches on two fixtures | CORRECT | training-corpus §"torch forward is gated" |
| Cross-round context: `note()` span + `st.st.end = 0` | both real, both fixed (`ecf44db6b`); cold acceptance moved 3.46 → 3.21, i.e. **null on the cold path** — they bite only on skipped/warm rounds | NULL (cold) / FIXED (warm) | acceptance-log §10, serving-defects §9 |

## 2. Acceptance

| experiment | result | verdict | where |
|---|---|---|---|
| Served vs vendor card, width 8 | accept length 3.33–3.39 against the card's 3.26 (card at width 4); our MTP 69.7% per draft against the card's MTP 69.3% — the baseline is healthy | CORRECT | vendor-baseline §2–§3 |
| "DFlash2 accepts 49% vs MTP 71%" | artifact of draft count: we offer 4.49 drafts/round, the card offers 3, and the decaying tail slots (0.74/0.66/0.59/0.62/0.39/0.43/0.33) dilute the aggregate | WRONG | vendor-baseline §3 |
| Like-for-like per-draft at the card's three slots | ours 0.740/0.662/0.592 (~66.5%) against the card's 75.3% → **0.27 accept length, ~9 points** per draft | OPEN (unattributed) | vendor-baseline §4 |
| Selector as the cause of that 9 points | see numerics: +0.05/+0.06 | WRONG | vendor-baseline §4 |
| Block width 4 vs 8 | per-slot flat or worse (0.720/0.667/0.542 vs 0.740/0.662/0.592), −12.5% advance, ~3% net slower; compile-time constant anyway | WRONG | plan §5 |
| Depth pinned to 7 | 7.05 drafts/round, 2.16 tokens/round; +33% tokens per round for +27% round cost → ~5% net | NULL | acceptance-log §4 |
| Defect B fix (window costs published, depth no longer hard-capped at 3) | 2.97 → 5.19 drafts/round, count prompt 52 → 79.5 tok/s at 100% acceptance | FIXED | serving-defects §7 |
| fp8 W8A16 drafter vs bf16, 16 GSM8K prompts | 49.4 vs 48.1 tok/s at accept 3.322 vs 3.387: −1.94% accept (CI −3.54…−0.33) for +1.31 tok/s (+0.61…+2.01), both real | ACCEPTED COST | fp8-checkpoint §4 |
| Production config (`--context 262144 --parallel 4 --kv-cache fp8 --vision --thinking`) | MTP 52.5 tok/s / 3.764 tokens a round / 6.56 drafts offered; DFlash2 fp8 39.2 / 2.430 / 3.19, same accept column. n = 4 prompts | OPEN (n too small, and the draft-count gap is unexplained) | fp8-checkpoint §6 |
| Count prompt 1..400, greedy | both arms 100% accepted; MTP 106.5 tok/s (8.93/round) vs DFlash2 79.5 (6.19/round) — structural, MTP drafts inside the target's forward | CORRECT | plan §5 |
| **Real-traffic A/B** — 16 of Relja's own agentic/thinking prompts from `~/.hermes/state.db`, 384 max_tokens, `--context 262144 --parallel 3 --kv-cache fp8 --vision --thinking`, one server at a time (2026-11-18) | **MTP 39.4 tok/s, AL 2.186, 1.88 drafts offered a round, 0.630 accepted per draft, 55.5 ms a round; DFlash2 fp8 30.4 tok/s, AL 1.985, 2.83 offered, 0.347 per draft, 65.3 ms a round. 16/16 outputs bit-identical. `--parallel 4` vs 3 is not a confound (MTP 40.5 vs 40.4 at 4 and 3)** | **NEW — the deficit is per-draft acceptance on out-of-distribution text, not draft count** | below |

### 2.1 The real-traffic pair (2026-11-18)

GSM8K says the drafter is at the card (0.67 accepted per draft, AL 3.39 vs 3.26). Real traffic says
0.347 — the same drafter, the same target, the same served config, half the per-draft acceptance. The
target's own MTP head holds 0.630 on the same prompts, so this is not "the workload is hard for
speculation": it is the DFlash2 drafter being out of distribution on thinking-heavy, tool-shaped,
instructive text, which is exactly what `flashnext-dflash2-training-corpus.md` predicts given the
harvester teaches user-role text.

Consequences, in order:

- Any acceptance figure quoted from GSM8K **overstates production** by ~2×. Quote this pair instead.
- The depth rule prices this correctly (it offers 2.83 drafts a round, not 7), so widening will not
  rescue it — the drafts are being refused on merit.
- Workload adaptation (fine-tune on assistant-role traffic from this target) moves from "legitimate
  but parked" to **the only lever that addresses the measured deficit**. It is still gated on the
  harvester fix.
- Round cost is 65.3 vs 55.5 ms at the same context — a second, smaller tax.

Reproduction: `/tmp/agentic_16.json` (regenerate from `state.db`), `accept_ab.py --arm … --max-tokens 384`.

## 3. Serving and state

| experiment | result | verdict | where |
|---|---|---|---|
| Empty `family_costs` → hard 3-draft cap | `depth()` short-circuits before reading `draft_ms`; flashnext CUDA published no `window_costs`. Ported the load-time sweep (36.55 ms @ 1 row → 98.94 @ 32) | FIXED | serving-defects §3, §7 |
| Tap ring not carried through prompt-cache restore | warm drafting refused (`DraftTapNotCovered`); ring now rides the snapshot, RAM and disk image (52.8 MiB a stream). Warm 23 → 79.7 tok/s, equal to cold, lossless | FIXED | serving-defects §7 |
| `--prompt-cache-disk` + drafter → `error: ProbeUnmarked` at startup | `learned.probe` needs marks at 4096 and `toks.len-8`; a speculative round commits past a mark. Arming the disk tier is what triggers it; `--spill-gib` alone reproduces it | WORKAROUND (drop the disk tier) / OPEN (mechanism) | fp8-checkpoint §6 |
| Multi-lane drafting | refused by design (`DraftSharedSpanMismatch`) — a shared pass has one tap ring, not one per lane | CORRECT (limitation) | handoff §5 |
| MTP arm reading 53.0 today vs 59.6 on 2026-11-17 at unchanged round cost (67.1 vs 65.3 ms) | AR flat (25.0 vs 25.5), nothing in the fp8 work touches the MTP path | OPEN — bisect before calling MTP unregressed | plan §1 |

## 4. Round cost

| experiment | result | verdict | where |
|---|---|---|---|
| The 438 ms "served profile" (`fc_ctx` 75.7%, head 1.9%) | it was the **cold** round; a single-position fixture cannot measure the served regime (two warm rounds on `fixture/window` still read 435.8 vs 438.0). `fixture/sequence` built | SUPERSEDED | plan §7 |
| Context fuse read its weight once per row | `tf_fxd2_fuse` streamed 65.5 MB per block; batched `glm_fuse_dot`: 5.10 → **0.71 ms**, bit-identical, proposals 16/16 unchanged | FIXED | plan §4 |
| Tiled block attention | 0.568 vs 0.823 ms a layer, round 26.14 vs 28.58 ms (−8.5%), capture bit-identical | FIXED | plan §4 |
| `head_tile` 1024 → 8192 | top-k 2.97 → 1.39 ms, head 8.60 → 6.98 ms, GEMM untouched | FIXED | plan §4 |
| Tensor-core GEMM for the fuse | faster (0.33 ms warm vs 0.71) but changed proposals on 7/16 reps on a 1.7e-7 rel-L2 shift — the selector amplifies sub-bf16 noise | REFUSED | plan §5 |
| fp8 W8A16 on the seven decoder linears | mlp −44.0%, qkv −36.3%, attn_proj −30.6%, attention −22.9% (clock coupling, hypothesis); served round 18.75 → **16.02 ms** | FIXED (shipped as an arm) | fp8-checkpoint §3 |
| Anything left in "use the bus better" | every weight-read stage sits at 187–234 GB/s against a ~231–238 GB/s sustained box rate | WRONG (closed) | survey §3.1 |
| NVFP4 body | ~0.8 ms beyond fp8, no published DFlash2-body acceptance, requires writing our own FP4 dequant | OPEN (ranked last) | plan §6 |

## 5. Precision decisions taken from the survey

`flashnext-dflash2-quantization-survey.md` is 99 cited sources, no local experiment; §10 maps each
option onto our stages with bytes, ms and acceptance risk. Taken:

- **fp8 W8A16 on the decoder linears** — built, measured, shipped as `--drafter-precision fp8`.
- **Ruled out**: quantizing `fc` (one 65.5 MB read at 92 GB/s; vendor refuses to export it), the selector
  codebooks (it *is* the acceptance mechanism; vendor keeps a candidate head at FP16), the convs by
  precision (latency-bound at 38 GB/s), activation quantization (W4A16 beats W4A4 by ~26 points
  single-stream on this box), SmoothQuant/QuaRot/SpinQuant (zero byte saving weight-only), 2-bit, and
  calibration pipelines (scheme differences sit inside the ~2.5% session drift).
- **Open**: fp8 `lm_head` (~2.71 ms) and a frequency-ranked 65,536 vocabulary trim (~5.0 ms incl. top-k).

### 5.1 The per-slot top-16 fidelity check — RUN 2026-11-18

Instrument: `tools/dflash2/analysis/head_fidelity.py` (CPU, numpy only, reads just `lm_head` out of its shard).
Rows: 49 real served-round head inputs from `fixture/sequence` at positions 1904–1925, captured with
`TF_FLASHNEXT_LG_DUMP` through `draft-check` (7.6 GB, no target boot), 7 slots × 7 rounds.

| variant | recall@16 | set exact | top-1 | mean rank of the top-16 | foreign ids in the top-16 |
|---|---:|---:|---:|---:|---:|
| fp8 (e4m3, 128×128 blocks, bf16 activations) | **0.9541** | 0.327 | 0.939 | 8.62 | 0.73 |
| **nvfp4 (e2m1, group 16, e4m3 block scales, fp32 global; W4A16)** | **0.8673** | 0.041 | **0.653** | 9.46 | 2.12 |

Head weight rel-L2: fp8 2.65%, NVFP4 9.45% (per-tensor global scale 6522.5 over an amax of 0.412).

Per slot recall fp8 0.982 / 0.946 / 0.938 / 0.964 / 0.964 / 0.946 / 0.938 — flat, so fp8 is not
worse at depth. NVFP4 per slot 0.911 / 0.857 / 0.857 / 0.857 / 0.875 / 0.893 / 0.821, top-1 agreement
0.857 / 0.571 / 1.000 / 0.714 / **0.286** / 0.571 / 0.571 — the damage is everywhere, not at depth.

Every row here sits in the `p<0.3` confidence bucket (mid-document text), which is the regime a block
drafter actually runs in: fp8 flips 0% of those rows, NVFP4 flips 14%.

**Verdict: fp8 head alone FAILS the gate and NVFP4 fails it far worse.** fp8 drops ~4.6% of candidate
slots and churns 0.73 foreign ids into the top-16; the tensor-core fuse precedent (a 1.7e-7 rel-L2
change flipped 7 of 16 proposals) says that is a proposal flip, not noise. NVFP4 at 0.867 recall with
top-1 agreement 0.653 would lose roughly a third of the block's anchors — it would be slower than
no drafting, since the target verifies a wrong first slot and the whole round is spent. The head is the
one tensor where 4 bits is not a byte-count trade but a different distribution.

Consequence: an NVFP4 **body** stays on the table (vLLM #67006 measured a BF16→NVFP4 drafter on GB10 at
+13.1% with acceptance length unchanged, +7.6% over an FP8 drafter), but an NVFP4 **head** is dead short
of a two-stage fp8/fp4 ranking with an exact bf16 rescore of the survivors. The trim variant is still
unmeasured (no frequency rank list has been collected).

**Rescue by rescore is also ruled out (measured).** Widening the cheap ranking's top-K and asking how
much of the true top-16 it contains: fp8 reaches 0.968 at K=64 and 0.974 at K=256; NVFP4 reaches only
0.893 at K=64 and 0.909 at K=256, with true candidates sitting beyond rank 700. So the 4-bit head is not
"a top-16 with a few swaps" — it reorders the low-confidence tail wholesale, and no cheap gather-then-
rescore recovers it. FP8, by contrast, is effectively complete at K=64.

## 6. Corpus and training

| experiment | result | verdict | where |
|---|---|---|---|
| Do document-shaped fixtures measure the drafter? | no: 1.625 and 1.000 accept length against 3.387 served — the drafter only drafts assistant-role text | WRONG (as a fixture source) | training-corpus §1 |
| Does the harvester emit assistant text? | no — `harvest.py` sends `messages=[{role:user}]`; ~99% of harvested tokens teach user text | OPEN (fix before any training run) | training-corpus §2 |
| Harvest rate and the disk wall | 2,039 tok/s cold prefill at 2k chunks (2,825 at 4k, 3,149 at 8k); ring is 27.1 KB/token regardless of prompt → ~2 TB a night against 60 GB free → must stream | CORRECT (constraint) | harvest |
| Local agent history as a corpus | 15,020 assistant content rows (~3.8M tokens), 54,103 tool-call rows (~18–22M), 55,782 reasoning (~23M); the assistant text is another model's (deepseek), the tool-call rows are the valuable shape | CORRECT (usable split) | training-corpus §3 |
| Trainer forward | gated on 216/224 slot parity, acceptance matches | CORRECT | training-corpus §4 |

## 7. What is open, and what each needs

1. **The ~9-point first-three-slot deficit** (66.5% vs the card's 75.3%). Next named test is the
   two-tap conv ablation — zero the dynamic projection, output must equal identity — the same A/B shape
   as the selector one. **Not run.**
2. **Head cost**: 6.99 ms of an 18.75 ms drafter round, 1,271.4 MB at 234 GB/s, untouched. fp8 alone is
   now measured and **fails the fidelity gate** (§5.1, recall 0.954) — it needs a two-stage
   fp8-rank → bf16-rescore or the vocabulary trim, whose rank list has not been collected. **Partly done.**
3. **Zero-risk round cost**: shuffle-based top-16 merge (1.39 ms → a few tenths) + the conv kernels
   (0.87 ms at 38 GB/s, 5× off their floor). ~1.5 ms, no acceptance risk, not implemented.
4. **Per-slot acceptance in the production regime**: the aggregate is now measured on real traffic (§2.1,
   0.347 per draft), but *where* the drafts die is not — the `done` line publishes `accepted=X/Y` with no
   per-slot split, and `tap_accept.py` scores teacher-forced prefixes, not verify rounds. Getting this
   needs a per-slot counter in the echo path (engine change) or an accept-dump on the serving path. **Not run.**
5. **Corpus path**: blocked only on the harvester's role bug and on a conversation-shaped held-out set.

## 8. Rules this campaign cost us

- A drafter change is gated on **acceptance at a stated width**, never on `token_sha256` — the target
  verifies every token, so a near-chance drafter still reproduces the text.
- A **null A/B is a broken A/B** until the arms are shown to disagree (the inert `TF_FLASHNEXT_TAP_SOURCE`).
- **Proposal identity is the gate for a refactor; per-slot accept length is the gate for a precision
  change.** The fp8 arm moves 64 of 112 slots and is fine; the tensor-core fuse moved 7 of 16 and was refused.
- **Quote the width** with any per-draft rate (`Y/rounds`), and the acceptance length with any per-slot rate.
- The **cold round is not the served round**; a single-position fixture measures the wrong regime.
- **The measurement is only as good as the binary.** `tools/dflash2/engine.py` replays a contract round
  (fixture `analysis/align_bf16/fixture/window` → proposals `{20, 11, 18, 15, 1, 314, 437}`, position
  2047, anchor 77491) and warns when `zig/` is newer than the binary. If the contract moves on purpose,
  update `CONTRACT` here in the same commit as the number it changes.
- Vendor ablations are properties of the vendor's target and suite — ablate a component in *both* the
  engine and the reference before spending kernel time on it.

## 9. Weight-only 4-bit body: the closed-form fit, measured (2026-11-18)

Tools: `tools/dflash2/quantize/nvfp4.py` (format primitives, shared with `analysis/head_fidelity.py`) and
`tools/dflash2/quantize/nvfp4_body.py` (`collect` → `fit` → `eval`). Fit is numpy on CPU; no gradients,
no corpus.

**Calibration is a handfull of real positions, not a corpus.** 140 positions × 8 block rows = 1,120 rows
taken from one harvested 4,240-token chunk. Harvest cost: 2.2 s at 1,965 tok/s, 12.2 KB/token of ring.
Error is flat from ~32 rows, so this is not a data-limited fit.

| stage | mean output rel-L2 | recall@16 | set-exact | top-1 | mean rank | foreign | identical proposals |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 body | — | 1.000 | 1.000 | 1.000 | — | 0 | 140/140 |
| NVFP4 RTN | 7.54% | 0.8774 | 0.051 | 0.895 | 9.55 | 1.96 | 58/140 |
| NVFP4 fitted group scales | **5.13%** | **0.9022** | **0.115** | 0.916 | 9.26 | 1.57 | 77/140 |
| fp8 (block 128×128, the served format) | 2.10% | 0.9610 | 0.467 | 0.961 | 8.62 | 0.62 | 101/140 |

**affine4** — the group-64 signed-scale+ bias layout `core/affine4.zig` already loads, so this is the
variant that could be served without new kernels. Same calibration, same fp8 baseline:

| format | out rel-L2 RTN -> fitted | recall@16 | set-exact | identical proposals |
|---|---|---:|---:|---:|
| NVFP4 (group 16) | 7.54% -> 5.13% | 0.8711 -> 0.9022 | 0.049 -> 0.115 | 58 -> 77 / 140 |
| affine4 (group 64) | 7.41% -> **4.99%** | 0.8711 -> **0.9036** | 0.049 -> 0.109 | 52 -> 67 / 140 |
| fp8 (block 128×128) | 2.10% | 0.9610 | 0.467 | 101 / 140 |

Coarser groups win: one signed scale *and* a bias per 64 weights beats 16 symmetric blocks on output
error, which is why affine4 is the format to chase.

**Teacher-forced accept length, and the number that replaced it.** A first pass on one chunk
(133 positions) read bf16 **1.098**, affine4 RTN 1.083 (−1.4%), affine4 fitted 1.068 (−2.7%), NVFP4 RTN
1.038 (−5.5%), NVFP4 fitted 1.060 (−3.4%), fp8 1.098 (0.0%) — and that table was wrong to quote, because
each arm's mean was bootstrapped independently and chunk identity is worth far more than the quantization.

Re-run on **8 chunks** of real assistant text (28,888 tokens harvested at 2,049 tok/s, 7,040 calibration
rows, 110 positions per chunk, **880 positions**, affine4 refit on all eight):

| arm | accept | paired Δ vs bf16 (95% CI) | P(Δ≥0) |
|---|---:|---|---:|
| bf16 (teacher) | 0.752 | — | — |
| affine4 RTN | 0.757 | +0.0045 [−0.0295, +0.0409] | 0.628 |
| affine4 fitted | 0.749 | −0.0034 [−0.0318, +0.0261] | 0.423 |
| fp8 | 0.750 | −0.0023 [−0.0182, +0.0159] | 0.422 |

Every arm is within ±0.04 slots of the teacher and of every other arm. The **resampling noise floor of the
metric itself is ±0.11 slots** (the teacher against itself) — this probe at 880 positions cannot resolve an
effect below ~0.05 slots, and nothing here is that big. So: weight-only 4-bit on the body costs no
measurable teacher-forced acceptance, and neither does fp8; the earlier −3.4%/−5.5% readings were
chunk-specific noise, not fidelity.

Two method rules this paid for:

* **Compare arms paired, never as independent means.** Per-chunk acceptance in this corpus spans
  **0.22 to 3.16**; unpaired CIs come out at ±14% and hide a zero effect behind corpus composition.
  Resample *positions* and take the mean of (arm − teacher).
* **The corpus is the experiment.** The same drafter scores 0.752 here and 1.098 on the earlier
  GSM8K-flavoured single chunk, with P(≥1) 0.291 vs 0.541. Any acceptance number without its corpus and
  its chunk spread is not comparable to anything — including to itself next week.

Verdict: fidelity-wise affine4 fitted is the best 4-bit format (4.99–5.40% mean output rel-L2 against
fp8's 2.1%), and acceptance-wise **all of it is free** at this resolution. That kills the "4-bit costs
acceptance" objection and leaves exactly one open question, which this harness cannot answer: does the
byte win show up as round time in the actual speculative loop? That needs the engine to load an affine4
drafter, and `cuda_drafter2.zig` still has no q4 path.

Method rule this campaign just paid for: **calibrate and evaluate on real activations only.** The same
fit run on `analysis/align_bf16/fixture/window` (synthetic taps, `gen_inputs.py`, random token ids) read
recall@16 0.498 for the fitted NVFP4 body versus 0.902 on a real chunk, and no acceptance at all can be
measured there — bf16 itself scores 0.0 because the ids are random.
