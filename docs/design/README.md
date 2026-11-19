# docs/design — index

This worktree is a private TensorFold fork (branch `running`) implementing DFlash2 speculative decoding
for `Qwen3.8-Flash-Next-FP4-Local` on a DGX Spark (GB10). DFlash2 is a 5-layer drafter that cross-attends
to five target-layer hidden-state taps and drafts a block of 8 rows (7 proposals), which the target
verifies; verification makes output lossless, so only acceptance length and round cost matter.

Measured state of the published and fp8 arms: the drafter's round in the served regime is **18.75 ms** on the published bf16
checkpoint (`fixture/sequence`, 15 served reps; 22.99 ms before the context-fuse change of 2026-11-18)
and **16.02 ms** with the weight-only FP8 build of the seven decoder linears (−14.5%, same binary, same
session, `flashnext-dflash2-fp8-checkpoint.md`), and
acceptance is **3.387 tokens/round at width 8** (3.322 on the fp8 checkpoint). End to end, measured
through the server on the same 16 GSM8K prompts, one arm at a time: **AR 25.0 · MTP 53.0 · DFlash2 bf16
48.1 · DFlash2 fp8 49.4 tok/s**, with identical completion hashes on 16 of 16 prompts in all four arms.
DFlash2 was 45.0 tok/s before the fuse fix; MTP's row reads 53.0 here against 59.6 in the earlier
capture at an unchanged round cost, which is an open question (plan §1), not a settled regression. On the vendor's own card that is parity on
acceptance (3.26) and behind on cost — 15.0 ms per extra token here against ~1.8 ms on GB300, the whole
of it the drafter's fixed per-round weight traffic.

Since 2026-11-19 the drafter that gets deployed is the **combined arm** — affine4 body, split-KV
attention, block-FP8 head with an exact bf16 rescore — with a **9.22 ms** drafter round and two
cold-start served passes at **53.2 / 53.4 tok/s**, accept 3.287 / 3.307, 16/16 identical outputs
(`flashnext-dflash2-optimized-drafter.md`; ledger §13). The figures above are the published and fp8
arms and remain the comparison.

The files below were written over several days by successive agents. They are kept, not pruned, because
the negative results cost real time; but only one of them describes the current state.

## Status vocabulary

- **LIVE** — authoritative for the current state. Numbers here are current.
- **HISTORY** — accurate account of an earlier state. Read for the mechanism, not the scoreboard.
- **SUPERSEDED** — contains numbers or claims now known to be wrong; the specific errors are in
  "Known stale claims" below.
- **REFERENCE** — external facts (vendor cards) or static specifications (kernel geometry, ring layout)
  that are not a claim about present performance.

## The documents

| file | contents | status |
| --- | --- | --- |
| `flashnext-dflash2-gb10-optimisation-plan.md` | Current scoreboard: served-round stage breakdown (`fixture/sequence`, 15 served reps), the byte rate of every weight-read stage, the harness that produces every number (`draft-check`, `regress.py`, env knobs), what has been changed (tiled attention, `head_tile` 8192, the batched context fuse), the closed questions, and the ranked open work. §7 is the retraction log for superseded measurements. | **LIVE** |
| `flashnext-dflash2-experiment-ledger.md` | Every DFlash2 experiment, its measured result and its verdict (FIXED / CORRECT / NULL / WRONG / SUPERSEDED / OPEN), grouped by numerics, acceptance, serving state, round cost, precision and corpus, plus §7 the five things genuinely open. Start here before re-running anything. | **LIVE** |
| `flashnext-dflash2-optimized-drafter.md` | **The deployment drafter.** What affine4 body + split-KV attention + block-FP8 head each are and what each buys, the exact serve line and env flags, the per-arm numbers with their session, n and CI, the losslessness claims and their limits, a one-line revert per part, the `key_value()` sign bug and why id-level checks missed it, and what is still unmeasured. Read this to run, measure or revert the arm; the derivations stay in the ledger (§9–§13). | **LIVE** |
| `flashnext-dflash2-fp8-checkpoint.md` | The shipped FP8 weight artifact and the switch: what is quantized (7 decoder linears) and what is not (`fc`, convs, norms, selector), the scale convention, the reproducible tooling and its manifest, `--drafter-precision bf16\|fp8` with checkpoint-decides default, the measured bf16→fp8 delta on `fixture/sequence`, the 64-of-112 proposal movement and why proposal identity is the wrong gate here, and what the served acceptance A/B still has to prove. | **LIVE** |
| `flashnext-dflash2-harvest.md` | The corpus harvester: one prefill pass per chunk through `TF_FLASHNEXT_TAP_DUMP` becomes a `draft-check` fixture, the measured cold prefill rates (2,039 tok/s at 2k chunks, 3,149 at 8k), the 27.1 KB/token ring and the 2 TB-a-night disk wall that forces streaming, the prompt-cache trap, and the 2,064-slot ring limit that fixes the chunk size. | **LIVE** |
| `flashnext-dflash2-training-corpus.md` | What a drafter fine-tune needs and what this box can produce: the measured proof that document-shaped fixtures score 1.6 against 3.3 served (the drafter only drafts assistant-role text), the chat-template defect in the current harvester, the local agent-history inventory with the caveat that its assistant text is another model's, the torch-forward parity gate (216/224 slots), and the plan, sizing and why RL collapses into self-distillation here. | **LIVE** |
| `flashnext-dflash2-quantization-survey.md` | What practitioners and researchers actually do about memory bandwidth on DGX Spark / GB10, and what transfers to us: format comparison measured on this box, per-component vendor support, which layers tolerate quantization, what a quantized `lm_head` does to a top-k candidate set, activation-aware and calibration methods, post-quant recovery, and the non-precision alternatives (vocabulary trim, low-rank head, two-stage retrieval). Every claim graded [M-GB10] / [M-other] / [V] / [P] and cited; §10 maps each option onto our stages with bytes, ms and acceptance risk. **No local experiment** — nothing in its §7 has been run here. | **REFERENCE** (its §2 baseline and §10 arithmetic are pre-fuse; the plan §6 is the rebased version) |
| `flashnext-dflash2-vendor-baseline.md` | The incoai card at source (GB300, FP8, SGLang, block 4) plus the corrected like-for-like comparison against our width-8 run, and the three corrections to what the repo previously believed about MTP vs DFlash2. | **REFERENCE** (card) / its §2 measurement rows are superseded by the plan §1 |
| `flashnext-dflash2-handoff.md` | The long mid-project handoff: repo/checkpoint paths, build and serve commands, commit map, MTP baseline, the five rounds of drafter debugging as they happened, the z-lab suite, and §14's then-open list. Carries a banner; §1–§14 are the historical record. | **HISTORY** |
| `flashnext-dflash2-acceptance-log.md` | The first investigation chronicle: the head-bug localisation (`tf_fxd2_topf32` publishing only lane 0), tap-basis analysis, selector arithmetic, and the single-round stage-diff campaign that produced the flat ~2e-2 per-block gap. §5 is the closed-hypothesis table (head, tap basis, block layout, logit post-processing, depth throttle, cross-round append, GDN rollback, drafter-vs-reference quality); §10 lists the two serving defects. | **HISTORY** |
| `flashnext-dflash2-divergence-brief.md` | The senior-pass debugging brief that closed the numerics: §3 the missing bf16 rounding in `tf_fxd2_fuse`/`tf_fxd2_rmsnorm`, §4 the fix, §6 the debugging rules (localise at the first divergent stage; `token_sha256` cannot validate a drafter; a null A/B is a broken A/B), §9–§10 the result — fusion fix real, residual is pinned summation order, not a defect. §6 is still the method to use. | **HISTORY** (its open question closed in §10) |
| `flashnext-dflash2-serving-defects.md` | What a live `serve` showed that the fixture could not: cold DFlash2 accepting 100% of offered drafts, then defect A (tap ring not carried through prompt-cache restore) and defect B (no `window_costs` → the depth rule's hard 3-draft cap). §7 has both fixes and their measurements; §8–§9 chase the acceptance deficit and land on block width as the hypothesis. | **HISTORY** (§9's width hypothesis is closed and wrong) |
| `flashnext-dflash2-crossround-brief.md` | Implementation brief for the two cross-round defects: `note()` recording `[post_fed, post_fed+n)` instead of the span the verify pass wrote, and `st.st.end = 0` clearing the forward's watermark every round. Exact files, lines, the correct span derivation, and the "do not change" list. | **HISTORY** (both landed in `ecf44db6b`, measured null on cold acceptance) |
| `flashnext-dflash2-drafter.md` | The original feasibility and integration plan: 9 steps from tap measurement to two-path E2E, the step table and what each produced, the dead ends (torch reference route, OOM sizes), and the checkpoint facts read off the file rather than the card. | **HISTORY** (the step table describes a state before the port ran) |
| `flashnext-dflash2-kernels.md` | Kernel spec for the flashnext drafter: which `qwen35_draft.cu` kernels are reusable unchanged and with what arguments, and the geometry/arithmetic for the new ones (`tf_fxd2_fuse`, layer QKV step, bidirectional block attention, selector). | **REFERENCE** |
| `flashnext-dflash2-tap-ring.md` | The tap ring contract: five layer outputs, one row a layer, slot `p % rows`, 50.4 MiB a stream; where in the batched pass the write belongs; and the silent-failure table (a ring not carried by a state drafts from stale rows and only shows as an unexplainable speed number). | **REFERENCE** (the `TapSource` question it defers is settled: `mean4`) |
| `flashnext-build-and-test-traps.md` | Toolchain notes: the two build graphs and why `-Dgpu` erases the `test` step, judging a run by `TEST_RC`, tests that never get collected, stale fatbins and the `.zig-cache` recovery, Zig 0.17 std changes, module boundaries, timers around async launches, and the `libtfgrammar` build step. | **REFERENCE** |
| `fluid-subgraph-residency.md` | Repo-level design for Fluid Subgraph Residency (Tier-E exact path, MoE larger than memory, Metal on Flash Next). Not part of the DFlash2 work. | **REFERENCE** |

## Read in this order

1. `flashnext-dflash2-optimized-drafter.md` — if you are here to run, measure or revert the drafter,
   this is the entry point: the arm, the serve line, the per-arm numbers, the revert table.
2. `flashnext-dflash2-gb10-optimisation-plan.md` — the state of play, the harness, and what is open.
   Everything you measure should be comparable to §3, and §2 tells you how.
3. `flashnext-dflash2-vendor-baseline.md` — what the bar actually is, at source, with the width and the
   model named. Read it before quoting any number, ours or theirs.
4. `flashnext-dflash2-fp8-checkpoint.md` — the fp8 arm as built: the artifact, the flag, what it
   measured, and the acceptance gate that is still open.
5. `flashnext-dflash2-training-corpus.md` and `flashnext-dflash2-harvest.md` — before any drafter
   fine-tune: what the corpus must look like, what harvesting costs, and the parity gate.
6. `flashnext-dflash2-quantization-survey.md` — before proposing any further precision change: what is
   measured elsewhere, what the risk actually is, and which options are already ruled out.
7. `flashnext-dflash2-handoff.md` §1 — paths, build line, serve line, commit map. Skip §2–§14 unless you
   need the history of a specific bug.
8. `flashnext-dflash2-divergence-brief.md` §6 — the debugging discipline. This is the part that is still
   binding: first-divergent-stage localisation, same-precision reference runs, no lossless-sha arguments
   about drafter quality.
9. `flashnext-dflash2-serving-defects.md` §7 — how the served path was fixed, and the shape of the
   bugs (state that must ride a snapshot, cost tables that must be published) that recur.
10. `flashnext-dflash2-kernels.md` and `flashnext-dflash2-tap-ring.md` — geometry and state contracts, when
    you touch a kernel or the ring.

Only if you are reconstructing why something was tried: `acceptance-log.md`, `drafter.md`,
`crossround-brief.md`.

## Known stale claims

Each entry is a claim an agent may still find in the tree, and what is true instead.

- **The 438 ms "stage profile": `fc_ctx` 75.7%, attention 19.0%, head 1.9%.** This is the *cold* round
  (fresh `State`: fuse and every prefix k/v projection over the whole 2047-row window, once per stream).
  Served round is **18.75 ms**, and the shares are **head 37.5%, attention 23.9%, `fc_ctx` 3.8%**
  (`gb10-optimisation-plan.md` §3, §7). The profile survives in the plan only as a retraction; anything
  that puts attention first on the strength of it, or calls the head "a 2% ceiling", is using it.
  Corollary: a single-position fixture cannot measure the served regime at all (measured: two warm rounds
  on `fixture/window` still read 435.8 ms against 438.0 cold).
- **"`fc_ctx` is 19% of the round and runs at 13 GB/s — an unexplained inefficiency."** Closed
  2026-11-18. The 13 GB/s was never a bandwidth figure: `tf_fxd2_fuse` launched a block per row and each
  block streamed the whole 65.5 MB weight, so the cost was `rows × 65.5 MB`. The batched `glm_fuse_dot`
  makes the stage **0.71 ms** (3.8%), bit-identical, proposals unchanged on 16/16 reps.
- **"fp8 changes nothing but speed, so gate it on proposal identity."** Wrong in both halves. The fp8
  weight arm moves **64 of 112** proposal slots (48 identical) while every untouched stage is
  unchanged — the selector amplifies a 2.646% weight error, exactly as the refused tensor-core fuse did
  with 1.7e-7. Proposal identity is the gate for a refactor; per-slot accept length over many prompts is
  the gate for a precision change.
- **"All 35 quantized matrices are ~2.6% off, so the least accurate ones can be kept in bf16."** There is
  no spread to exploit: relL2 is 2.643–2.650% across all 35 and **all 26,000 blocks exceed 2%**, because
  2.6% is e4m3's own noise floor for these weights. A per-block mixed-precision fallback has nothing to
  select on here.
- **"Weight bytes are only 55% of the round."** The survey's §2 figure, computed before the fuse fix.
  Now **67%**, because the fix removed non-byte cost from a fixed total. Do not use 55% to discount
  precision work.
- **Block width 4 as a serving width.** Closed. Per-slot quality is flat or worse (0.720/0.667/0.542 vs
  0.740/0.662/0.592), it loses 12.5% advance, and it nets ~3% slower. It is not a knob:
  `zig/src/families/flashnext/cuda_draft2_fwd.zig:63` has `pub const block = 8` and
  `cuda_draft2_fwd.zig:451` returns `DraftBlockUnsupported` for any other declared block. The vendor's
  width 4 is a QSA cap in *their* stack; they have published no width-8 number. `serving-defects.md` §8.3/§9 ("width 8
  dilutes acceptance") is the stale claim.
- **The candidate selector is the acceptance gap.** Closed. Engine on-vs-off is **+0.05 accept length**
  (reference +0.06). The vendor's +0.34/+0.47 ablation is a Qwen3.5-4B number and does not transfer.
- **The z-lab Qwen3.8-27B drafter card is our bar.** It is not. The circulating figures (~4.1–5.5 accepted
  tokens/step, 3.43×) are a **64-layer dense target on H200** with a different drafter; our card is
  `incoai/…-DFlash2` on `Qwen/Qwen3.8-Flash-Next`, one GB300, block 4. Quoting the z-lab figures against a
  width-8 run on this box manufactures a gap that does not exist. (`handoff.md` §12 quotes them, correctly
  labelled as not our bar; `divergence-brief.md` §1 warns about them.)
- **"DFlash2 loses to MTP" / 46.2 tok/s, 3.46 accept, 62.7% draft acceptance** (`serving-defects.md` §8).
  Measured before defect B's cost table and before the head fix. Current: 3.34 accept/round and 45.0 tok/s
  at width 8, against MTP 3.89 / 59.6 — MTP still wins on throughput here, which is a cost story
  (15.0 ms per extra token vs 9.0), not an acceptance story.
- **The 25.5 ms served round and 8.60 ms served head** (`gb10-optimisation-plan.md` §7's own note).
  Pre-`head_tile = 8192`; current 18.75 ms / 6.99 after the fuse change.
- **43.8 tok/s, 1.72×, accept 3.33** (`vendor-baseline.md` §2). An earlier capture of the same harness;
  the plan §1 table (45.0, 3.34) is the current one.
- **"Round cost is the only remaining lever" — 84 ms against 39.6 ms with no drafts**
  (`acceptance-log.md` §9 item 3). Pre-dates the serving fixes, the tiled attention kernel and the
  batched fuse; the round is 18.75 ms and the levers are named and ranked in the plan §6.
- **The cross-round defects being the acceptance gap.** Both were real and both were fixed
  (`ecf44db6b`), and cold acceptance moved *down* (3.46 → 3.21) in that session's measurement — they are
  latent on a cold stream because a cold stream never skips a round. Do not reopen them as the cause;
  they are a warm/rollback correctness fix.
- **`st.st.end = 0` every round / `note()` after `q.fed` advanced** as *open* defects
  (`acceptance-log.md` §10). Both fixed; keep §10 only as a description of the bug shape.

## Docs outside this directory

- `docs/receipts/` — the measurement evidence, one directory per run with `README.md`, `SHA256SUMS` and
  the raw logs/JSON. `docs/evidence-index.md` is the machine-readable index over them
  (`tools/evidence_index.py`, no GPU needed).
- `tools/dflash2/README.md` — the torch reference port used for the tap-source question; not shipped,
  nothing imports it. `tools/dflash2/analysis/align_bf16/` is the two-sided stage diff; `regress.py` is the gate;
  `tools/dflash2/run/matrix-mtp-report.md` is a generated warm/cold MTP matrix.
- `docs/recipes/qwen3.8-flash-next.md`, `docs/recipes/flash-next-cuda.md` — serving recipes for the
  target family (Metal and CUDA). Not DFlash2-specific except where `--drafter` is passed.
- `docs/optimization-ledger.md` — which optimisation applies to which family and where it stands; the
  cross-family ledger, not the DFlash2 scoreboard.
- `docs/recipes/fluid-subgraph-residency.md` — the recipe-side variant of the FSR design in this
  directory; the two are not identical.

## Adding a doc here

State the status in the first three lines (`open`, `executing`, `historical`) and put any number that
supersedes an older one in a retraction section rather than deleting the old one, as the optimisation
plan §7 does. Add the file to the table above in the same change.
