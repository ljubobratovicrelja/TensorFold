# DFlash2 drafter — where this stands, and what to do next

Handoff for whoever picks this up. Written after a long session that got the drafter
*serving and lossless* but not *fast* and, more importantly, not **good**: the block
proposes near-chance tokens, so every throughput number below is the cost of hiding a
correctness problem.

Read this before touching the code, then read
`docs/design/flashnext-dflash2-drafter.md` for the plan and
`docs/design/flashnext-dflash2-kernels.md` for the kernel geometry.

---

## 1. Where things are

- Repo: `~/Projects/tf-wt/dflash2`, branch `dflash2`, based on `running` at `bb95ab741`.
- Served build path is `~/Projects/tf-wt/running` (another worktree) — **not** this one.
  Nothing here has been deployed and nothing here touches the live service.
- Target checkpoint:
  `~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300`
- Drafter checkpoint:
  `~/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6`
- Python: `~/.venvs/marker/bin/python` (torch present). Zig: `~/tools/zig-0.17.0`.
- Build: `PATH=$HOME/tools/zig-0.17.0:$PATH zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc`
- Server helper: `tools/dflash2/tfsrv.sh`; run the binary with
  `serve <ckpt> --host 127.0.0.1 --port 8099 --name qwen3.8-flash-next --context 32768 --parallel 1 --kv-cache fp8 --drafter <drafter-dir>`
- Serve with `--drafter` to enable DFlash2; without it the same binary serves MTP. MTP is
  still the default and has **not** regressed (see §4).

Commits on this branch that matter:

- `3e8531e6a` — the plan's real progress table (corrected; the earlier table claimed a
  step was done that was not)
- `6b82f7094` — the drafter serves, lossless, and is 5× slow
- `d9a46d23b` — the prefix ends before the anchor; the round's cost attributed
- `337a308c2` — the test build compiles again (detile dims as `usize`, the verify fixture
  gets `opsTapped`/`drafter`)
- `419442c0b` — span/refusal tests pin the prefix-before-anchor semantics
- `cb4c9ccb3` — the depth rule prices a fixed-forward drafter at its full round cost
  (`Facts.draft_ms`), not the per-draft marginal
- `11c05be97` — the head runs on `glm_mm` tensor cores (lm_head tiled once at open)
- `ac60d534e` — the block's seven projections run in one batched launch
  (`tf_q35_dense_bf16_batch`, bit-identical sums)
- `d9bd10563` — the CUDA test runner builds again (three MTP call sites pass the `kv8`
  flag they predate)

All of §13's commits are pushed to `origin/dflash2` (`3e8531e6a..d9bd10563`).

---

## 2. The headline: it is correct and useless

One lane, one prompt, same binary, per-request `draft` flag, 100 tokens
(`tools/dflash2/round_cost.py`):

- **no drafts**: 22–25 tok/s, 99 rounds, 39–42 ms/round
- **MTP**: 37.0 tok/s, 56 rounds, 73/136 accepted (2.3 a round), 57 ms/round
- **DFlash2 (now)**: 9.4 tok/s, 60 rounds, **40/87 accepted (1.52 a round)**, ~171 ms/round
- **DFlash2 (at first light)**: 5.47 tok/s, 63 rounds, 37/88 accepted, ~300 ms/round

Losslessness is real: with drafts on, the request's `token_sha256` equals MTP's
(`ccde9cc12327` on this prompt, `5d72e1f7006d…` on the earlier one). Drafted tokens are
only ever kept when the target agrees, so the wiring is not producing wrong text — it is
producing almost no *useful* text.

Acceptance is the defect. A round costs 7 drafts and returns ~1.5 tokens. For DFlash2 to
beat MTP at MTP's 57 ms and 2.3 tokens, a round would need ≤ ~40 ms at this acceptance;
at the vendor's claimed acceptance (several tokens a round) even a 170 ms round wins.
**Do not chase throughput first.** Fix acceptance, then cost.

Teacher-forced acceptance on a real tap dump is the same story:

- mean advance **1.275** tokens/round (0.275 of 7 accepted), zero-accept 85%
- mean advance **1.325** with the context window shifted +1 position, 1.325; 80% zero
- (an earlier 1.342 figure predates the prefix-origin fix and used the old convention)

Near chance. That is a *wrong tensor* signature, not a *wrong offset* signature.

---

## 3. Cost attribution (measured, keep it)

Per round, with `TF_DRAFT2_TIME=1` (drafter) and `TF_TAP_TIME=1` (verify + round):

- **prefix fusion + k/v: 34 ms** (the first round of a stream pays ~10× that; later
  rounds fuse only the new rows)
- **head (rmsnorm + top-k over the 248,320-token vocabulary): 93 ms**
- **selector: 0.6 ms**
- **target verify (2–8 row window): 43–44 ms**

Two of those were bugs of mine, not properties of the design:

1. `prefix()` used to fuse the whole 2048-row context **once per drafter layer, once per
   round** — 133 ms. It is now fused once per position into a per-stream fused ring
   (`fusePrefix`, `State.fused`), read back chunk-wise by `prefix()`. 133 → 34 ms.
2. The head kernel launched **7 blocks** (one per block row) onto a 20-SM GPU and read a
   1.27 GB matrix: 105 ms. A warp-per-vocabulary-entry rewrite made the reads coalesced
   and bought only 11% (→ 93 ms), because occupancy, not access pattern, is the limit.
   A two-pass tiled head (tile × row grid + a merge kernel) is **in flight, uncommitted**
   in `qwen35_draft.cu` / `cuda_draft2_fwd.zig`. Expect ~10–20 ms if it lands.

Also still unexplained by any timer: nothing — the "250 ms unaccounted" in the previous
session turned out to be the drafter's own GPU work measured at launch time rather than
at the sync. If you add a timer to this code, put the `synchronize()` inside the bracket
or it will lie to you.

---

## 4. What is proven to work

- The serving path: `--drafter <dir>` opens the drafter, drafts per round, verifies, and
  produces byte-identical text to MTP. Refusals are named and diagnosed, not silent.
- The MTP path is untouched behaviourally: identical `token_sha256` across the A/B, and
  medians 68.87 vs 70.40 tok/s (within the baseline's own spread).
- The tap ring is written by the target's own passes; taps are meaningful. Tap 45 through
  the target's `lm_head` ranks the true next token at rank 107 / 7 / 5 / **0** at
  successive positions; tap 35 has median rank ~23,768; tap 5 ~100,677. Lower layers
  degrade as expected — this is the strongest evidence that the *layer* choice is right
  and something subtler is wrong.
- The tap dump is reproducible:
  `TF_FLASHNEXT_TAP_DUMP=/tmp/tapdump-readout` + one prefill via
  `tools/dflash2/tap_dump_run.py`. Dumps exist at `/tmp/tapdump-readout`,
  `/tmp/tapdump-normed`, `/tmp/tapdump-readout-bf16`.

---

## 5. What is bothering me (ranked suspicions)

1. **The tap row may be the wrong tensor.** Each target layer uses four parallel
   hyper-connection streams that are mixed into one `mixed` row. "The tap" is whichever
   of those is the model's hidden state at that layer. That question is *unresolved* in
   this repo, and a wrong pick would suppress acceptance exactly like this — including
   slot 0 failing most of the time. `tap_accept.py` refuses `--ctx-mode mean4` because
   the collapse kernel does not exist; only `readout` and `normed` are available.
   **In flight:** an analysis of the existing dumps that checks per-slot lm_head ranking,
   position offsets, and the four streams individually.
2. **Our oracle shares our misreadings.** `tools/dflash2/reference.py` matched a SpecForge
   port to ~1e-6, which felt like validation, but it was written by the same hand that
   wrote the engine. A convention error held in common is invisible to it. **In flight:**
   a diff of every operation against the vendor's own `model.py`
   (`github.com/z-lab/dflash`), which is the only acceptable oracle.
3. **The depth rule prices a draft from a mis-measured round.** `measure()` times a full
   round and divides by `depth`, i.e. it reports the cost of one *draft* as if it were
   the marginal cost, then `depth.zig` charges `forward + step_ms × d`. Combined with
   `Facts.prior` never being set (so the MTP head's 0.85/0.75/0.7 curve prices a drafter
   with a different acceptance distribution), the rule can settle on depth 1 — which is
   what produced 2-row windows early on. Worth re-testing **with depth forced to 7 vs
   forced to 1** on the same prompt: separates "priced out" from "weak drafter".
4. **A rollback never rewinds `written`.** `note()` records a whole verified window as
   written; `keep(k)` then lowers `seq.fed` and nobody tells the drafter. `written` is a
   permanent high-water mark, so rows belonging to *rejected* drafts stay "valid". The
   context now ends one position before the anchor (see §6), which removes the most
   damaging case, but the bookkeeping is still wrong and `covered` has the same problem.
5. **`TapSource.normed` copies with the wrong stride.** `s.norm` rows are `4 × hidden`
   wide; `tapRingWrite` walks them at `hidden`. `normed` therefore stores rows stitched
   from four positions. We serve `readout`, so this is latent — but it is the kind of
   thing that silently poisons an experiment.
6. **The prompt cache carries no tap ring.** A cached prompt restores a stream with no
   ring, so drafting refuses (`DraftTapNotCovered`) — that is why all 9 DFlash2 cells of
   the matrix benchmark came back as HTTP 500. The ring is not in `cuda_snap.zig`.
   Any benchmark that warms the cache first will measure nothing.
7. **Multi-lane drafting is refused by design** (`DraftSharedSpanMismatch`): a shared
   pass has one tap ring, not one per lane. The `hold` side of that refusal looks
   spurious — `round()` is per-stream — but `Pass.tapRingWrite` genuinely cannot write
   several rings. Single lane is all that works today.

---

## 6. Fixed this session (do not regress)

- **The prefix no longer ends on the anchor's own position.** The block's row 0 is the
  anchor, embedded and rotated at `position`; a context ending there too put the same
  position in the keys twice and phased every context key one step late. `span()` and
  `prefix()` now agree on `[position - ctx_len, position)`, which is the vendor's
  `position_ids[:, start-ctx : start]`. Note: measured, this did **not** fix acceptance
  (1.275 → 1.325 at best), but it is the vendor's convention and it removed the
  duplicate position. Keep it; do not "fix" it back.
- **`ctx_len` was passed to the forward unclipped while `ctx` held the clipped span.**
  Fixing that is what made a round carry all seven drafts instead of one.
- **The per-stream ring state is created on lane open** (`Cuda.open` → `draft2.ensure`);
  the lanes path never calls the vtable's `open`, so before this there was no ring at all.
- **A plain decode round taps.** `Engine.round` now takes tap hooks for a single-lane
  pass, and `tapRingWrite` accepts a one-lane pass (it still refuses a shared one). Before
  this the ring was always one row short of the anchor.
- **The first token comes from the prefill's last row**, so the anchor's own row does not
  exist at the first round; the guards now require the *context* up to `anchor - 1` only.
- **`fusePrefix`/`State.fused`** — the incremental fused ring; reset via `State.reset`
  in `cuda_drafter2.zig` so a pooled state cannot answer with another request's rows.

---

## 7. Ruled out (with numbers) — do not redo

- **A one-position context offset is not the bug.** Sweeping the context window by
  −1/0/+1 positions moves teacher-forced mean advance 1.275 → 1.325 → 1.325 at best.
  `tap_accept.py --ctx-shift` exists for this.
- **The layer mapping is not off by one.** `[6,16,26,36,46]` scores *worse* (1.250) than
  the shipped `[5,15,25,35,45]`.
- **The head row shift is right.** Rows 1..7 carry the proposals (row 0 only re-predicts
  the anchor); the vendor's `draft_hidden[:, 1-verify_size:]` and `sample_from_anchor:
  False` agree.
- **`tf_fxd2_select` is not the problem.** It matches the vendor's `CandidateSelector`
  line for line, and its out-of-vocabulary guard cannot change an in-vocabulary round.
  (It *did* once read 511 GB past an allocation — the double-applied block offset is
  fixed and the kernel is guarded.)
- **The batched verify amortises.** MTP's 4-row verify costs 57 ms against 40 ms for one
  row (1.4×). An 8-row window costing 7× a 1-row pass means serialisation somewhere else,
  not a slow batched pass.

---

## 8. How to reproduce each number

- Throughput, drafts on/off, one process:
  `python3 tools/dflash2/round_cost.py 8099 100` against a `--drafter` server.
- Per-round cost split: start the server with `TF_DRAFT2_TIME=1` and read
  `draft2 stages: layers … head … select …` and `draft2 round: … = gather … + draft …`.
  Add `TF_TAP_TIME=1` for `round: … = verify … + rest` and `verify window: N rows, X ms`.
- Teacher-forced acceptance: `~/.venvs/marker/bin/python tools/dflash2/tap_accept.py
  --dump /tmp/tapdump-readout --block 8 --max-rounds 40 --ctx-mode prefix [--ctx-shift -1]`.
  It refuses a manifest whose layer list disagrees with the drafter's, by design.
- Full matrix (MTP only, historically): `tools/dflash2/matrix_bench.py`; rows in
  `tools/dflash2/run/matrix.jsonl`, report in `tools/dflash2/run/matrix-mtp-report.md`.
  DFlash2 cells are all `DraftTapNotCovered` today — see §5.6.

## 9. Traps that cost me hours

- `zig build test` can return 0 while a module's tests never run (they were reachable only
  through an unused `pub const`). Anchor test imports from the family root and check the
  test names appear in the cache binary.
- `zig/dense_serving_test.zig` pins the sha256 of `kernels/cuda/qwen35_draft.cu`. Any
  kernel edit needs the pin updated, or the suite fails for an unrelated-looking reason.
- A pass with `hooks.n_taps != 0` is **never CUDA-graph replayed**
  (`cuda_prompt_windows.zig`): a drafting server runs every window eager. That is a known,
  deliberate cost.
- Zig 0.17 has no `std.time.Timer` and no `std.posix.getenv`; this tree uses
  `std.c.getenv` + `std.c.clock_gettime`, and `std.Io.Clock.Timestamp.now(io, .awake)`.
- Instrumentation measured at *launch* time is meaningless for GPU work — sync inside the
  bracket. This produced a confident, wrong "the drafter costs 0.3 ms".

## 10. In flight when this was written (check before redoing)

- Two analysis children: a vendor-code oracle diff (op by op, `z-lab/dflash` →
  `reference.py`) and a tap-identity analysis over the existing dumps.
- One code child: the two-pass tiled head. Its edits are **uncommitted** in
  `qwen35_draft.cu` and `cuda_draft2_fwd.zig`; do not commit half of it blind.

## 11. What I would do next, in order

1. Settle the tap tensor question from data (§5.1). If the correct capture is not
   `readout`, everything measured so far is a measurement of the wrong input.
2. Run the vendor code itself on one round and diff proposals row by row against
   `tools/dflash2/draft_compare.py`'s output (§5.2). Agreement there is the first oracle
   in this repo that is not self-referential.
3. Force depth 7 and compare accepted-per-round against the depth the rule picks (§5.3).
4. Only then the cost work: land the tiled head, and make the fused ring survive a
   prompt-cache restore so a benchmark can actually run.

## 12. 2026-11-16 — vendor oracle, current measured state, incremental K/V

Settled this session (evidence in `tools/dflash2`, /tmp/voa_*.json, replays in
~/.hermes/cache/scratch/voracle/):

- **THE vendor code is two things.** `z-lab/dflash` (local shallow clone, commit 07ebd93) is the
  reference/bench repo; its README routes serving to the runtime forks. Production =
  **SGLang PR #35371** ("DFlash2: local convolution + candidate selector"): `srt/models/dflash.py`,
  `srt/speculative/dflash_worker_v2.py`, `spec_aux_hidden_state.py` + Triton selector (same team's
  ports: vLLM #52816, llama.cpp #27342, oMLX).
  Production numbers there are for **Qwen3.8-27B** (block 8): 4.39-5.46 accepted tokens a step,
  236.1 t/s vs 68.9 autoregressive (3.43x, GSM8K conc 1). **That is NOT our target and NOT our
  bar** — a 64-layer dense model on H200, not Flash-Next; see `flashnext-dflash2-divergence-brief.md`
  §1. Our bar is the Flash-Next card: 1.63-1.82x over autoregressive at block 4 on GB300, and
  3.26 vs 3.08 accept length against the target's own MTP head.
- **Our reference.py matches the production forward op-for-op.** z-lab torch model vs reference.py:
  stage rel_rms 2e-7..1.5e-6, proposals identical (the vendor's 8-row propose is the generate
  loop's slice `[:, 1-verify_size:, :]` applied outside). SGLang's conventions agree with ours:
  sliding-window bidirectional (is_causal:false + sliding_window 2048), selector lattice
  unary + <A[p]*proj(h), B[c]>, rows 1..7 headed, target layer ids unshifted for qwen3 targets,
  draft KV materialized from the target's verified hidden rows, dense target lm_head required.
- **Context semantics are acceptance-neutral** (200-round vendor-code replays on the dump):
  production-incremental 0.255, engine full-prefix 0.260, hybrid 0.260, no-context 0.065.
- **Tap noise does not matter**: row-relative gaussian noise 0/1/5/10% on the context taps gives
  0.260/0.260/0.260/0.275 accepted/round. fp8-KV taps (4.9% vs bf16) are not the acceptance issue.
- True-text per-slot marginal hit rates decay 0.20 (slot 1) -> 0.05 (slot 7); scoring still
  deflates vs live, but the shape is the drafter's capacity curve, not a defect.
- **Baseline A/B (binary @614aa3c2a, one server, per-request draft flag, temp 0, 160 tokens):**
  short prompt: drafting 12.87 t/s vs nodrafts 22.2-24.9; 1451-token prompt: drafting **8.89**
  vs nodrafts 20.79. Stage split at 1600-token ctx (TF_DRAFT2_TIME): layers 93.0, head 39.5,
  select 0.9, verify 44-47; round 133 draft + 47 verify. The `kv` sub-timer reads 1.0 — enqueue
  time, the known unsynchronized-timer trap — the ctx-scaled k/v work hides inside `layers`.

**Incremental K/V (this branch):** the k/v ring now carries projected rows between rounds, guarded
by span `[kv_lo, kv_end)` with the fused ring's own reuse rule (`reuseFrom`); a round projects
only the positions the span does not cover, `fusedReset` clears it. Tests: `cuda_draft2_fwd.zig`
"a round projects only the k/v positions the round before it did not" + span-reset.

**Measured A/B, one server, per-request draft flag, temp 0 (2026-11-16):**

- 614aa3c2a (baseline): short drafting 12.87 t/s (nodrafts 22.2-24.9); 1451-token 8.89 (20.79).
- 92ed00596 (+incremental K/V): short 14.01, 1451-token **12.72 (+43%)**; stages @1600 ctx:
  layers 93.0 -> 33 ms, kv 0.0, head 39.5, select 0.9, verify 44-47; sha still identical to
  nodrafts (lossless).
- e7acb89e2 (head tile pass: rows loop inside the tile CTA, weights stream once): perf-neutral.
  nsys of the drafting path: `tf_fxd2_headtile` = 39.8 ms/launch (1.31 s across 33 launches,
  ~32 GB/s effective — latency/issue-bound scalar path, not bandwidth); next drafter-side items:
  tf_q35_dense_bf16 (the block's per-element dense) 0.38 ms/launch x 2035, fuse 7.5 ms chunks.
  ncu is blocked on this box (ERR_NVGPUCTRPERM, needs root).

**Next, in order:** (1) the head over the repo's tensor-core engine: `glm_mm.cu` BITS=16 mode is
plain-bf16 weights and K=2560 fits its 64-groups; pre-tile lm_head once at open with
`glm_mm_tile16`, then `glm_mm16_64(hn, ldx, tiled, null, logits_f32, null, 1, 7, 248320, 2560, 1)`
per round + an fp32 slice-topk pass reusing `tf_fxd2_headmerge` — head 39.8 -> ~8-12 ms, long-ctx
drafting ~17-19 t/s. (2) the block's projections batched, killing the per-element `tf_q35_dense_bf16`
streams (the ~24 ms inside `layers`), for parity with the no-draft path. Then acceptance work on
top.

## 13. 2026-11-16 (afternoon) — depth pricing, tensor-core head, batched projections: landed

Visible work after §12 (all on `dflash2`, pushed to `origin/dflash2`; TDD throughout —
red test first, narrow filters, full-suite only ever as the last gate):

- **Depth-pricing fix (`cb4c9ccb3`).** `Rule.roundCost(d)` was `forward + mtp_step_ms × d`
  with `step_ms = 74/7 ≈ 10.6` — the measured fixed forward divided over 7 proposals, so a
  depth-1 round was priced ~55 ms while really costing ~119 (full forward + verify) and the
  rule settled on 2-row windows. New `draft_ms` term through `Facts → Model → Config → Rule`:
  a fixed-forward drafter pays its full round cost at any drafted depth (`step_ms = 0` — the
  block runs entirely whatever verifies). Red test recorded in `depth.zig` first
  ("a fixed-forward drafter's round is priced at its full cost, not the marginal step"),
  55.6 ≠ 119 observed, fixed, green with the whole 164-test depth suite.

- **Tensor-core head (`11c05be97`).** The tiled-head plan above, executed: `lm_head`
  [248320, 2560] bf16 tiled once at drafter open (`glm_mm_tile16`, ~1 s, +1.27 GB RAM, the
  tiling is a permutation so bytes are unchanged), a round's head = one `glm_mm16_16` matmul
  (7 rows fit one BM=16 block) into a new fp32 logits scratch buffer + a thin
  `tf_fxd2_topf32` slice top-k + the unchanged `tf_fxd2_headmerge`. **Head 39.5 → 5.9 ms**
  with the SHA exactly equal to no-drafts (every proposal identical — the bit-identity rule
  from §9/skill held via same order in the top-k, not the GEMM).

- **Batched block projections (`ac60d534e`) — delegated with a TDD brief, orchestrator
  verified.** `tf_q35_dense_bf16_batch` in `qwen35_bf16_rows.cu`: grid (n, 1), 256 threads,
  one CTA = one output column for all 8 block rows, W row streamed once, NO shared memory (the
  staged form was measured 1.1–2× *slower* at every served width — 40 KiB caps a 256-thread
  CTA at 2/SM — and the mid-flight design change that dropped staging but left the host
  passing `.shared` + tests pinning the budget was caught in orchestrator review and steered).
  Bit-identity: same stride-256 uint4 walk, `__fmaf_rn`, warp `__shfl_xor`, thread-0 warp-order
  sum. Gates: in-repo `tf-cuda-test qwen35-dense-batch` (9 shapes `sameBytes` vs the
  per-element kernel + a 5e-3-of-magnitude float64 band — the 1e-5 gate first written fails
  healthy data at ~1.8e-4, which is just bf16's store floor), the seven-projections host test
  in `cuda_draft2_fwd.zig`, and the runner itself needed 3 preexisting `MTP.Draft.init`
  call sites to carry the `kv8` flag (`d9bd10563`) plus measured fixes. **Layers 31.8 → 15.1
  ms** in the served round.

**Measured ladder** (one server, per-request draft flag, temp 0, 1451-token probe, 160
tokens, `TF_DRAFT2_TIME=1`):

| build | drafting tok/s | nodrafts | stages @1600 ctx (layers/head/select) | round |
|---|---|---|---|---|
| `614aa3c2a` baseline | 8.89 | 20.79 | 93.0/39.5/0.9 | ~180 ms |
| +incremental K/V `92ed00596` | 12.72 | 20.79 | 33/39.5/0.9 | ~123 ms |
| +tensor-core head `11c05be97` | 11.2–12.5 | 21.7 | 31.8/5.9/0.4 | ~82 ms |
| +batched projections `ac60d534e` | **13.99** | 22.63 | **15.1/6.1/0.5** | **~66 ms** |

SHA identical to nodrafts at every step (the four cost cuts never moved a token). The
dropped verify floor is untouched: ~44–47 ms a round is now the dominant term.

## 14. What I would investigate next, in order

1. **Measure real full-depth acceptance under the fixed depth rule — the one number this
   whole project is about.** Force depth 7 vs depth 1 on the same prompt (`TF_FLASHNEXT_DRAFTS`),
   one server, and read `rounds=`/`accepted=` from the logs with the fixed pricing in place.
   The honest prior for "working": ~2.4–3.3 accepted/round (vendor's block-8 class; our lane
   MTP does 2.3). At 66 ms a round that is ~45–50 tok/s drafted vs 22.6 nodrafts — i.e. the
   point of the exercise. At the measured ~1.5 it stays ~14 and the next item stops mattering.
   Teacher-forced replays still say ~0.26/round against the vendor's own code on the same
   tap dump (§12), so expect the forward to fix the price while exposing the real acceptance
   gap — that gap is the *block's proposals*, not the timing.
2. **The tap input tensor, last unsettled axis (§5.1 today).** `readout` (what we serve) is
   proven the right row and position by the logit lens (l45 median rank 6, top-10
   probability 0.572 of the next token); still unbuilt/unmeasured is the `mean4` basis (the
   four-hyperconnection-stream mixture the vendor's `spec_aux_hidden_state` commits) and the
   `normed` capture is a known stride bug (poisons experiments, not served). If (1) shows
   acceptance stuck near threshold, this is where the difference lives.
3. **The ~44–47 ms verify floor.** After (1)+(2) set the acceptance, the verify is the last
   cost that keeps a round near 60 ms instead of ~30. Profile it (the batched 8-row window
   costs ~1 pass per §7 "batched verify amortises", so the ms are elsewhere: look at
   per-round state save/restore, the custom metric, or the no-graph eager path from §9).
4. **Then the throughput enablers, in order:** prompt-cache restore carrying the tap ring
   (§5.6 — every warm-benchmark DFlash2 cell 500s), multi-lane tap rings (§5.7), and the 13
   image-lane GPU test ABRTs. A proper `tools/bench_serving.py` row exists for MTP; run it
   for DFlash2 once the matrix cells can actually serve.


