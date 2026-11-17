# DFlash2 served-path defects: what the end-to-end serve actually showed

Status: **open — two serving defects, neither in the drafter's arithmetic.** Measured 2026-11-17
with one real `serve` each on the DGX Spark (GB10), `--parallel 1 --context 32768`, greedy
(temperature 0), thinking off, the `three_paths_bench.py` count prompt (76 tokens → 768 output).
The drafter correct on a cold prompt is the headline: the "acceptance gap" the project has been
chasing is two serving bugs, not a weak drafter.

## 1. Measured: cold vs warm DFlash2, and the two neighbours

| path | tokens | rounds | accepted | tokens/round | tok/s |
|---|---|---|---:|---|---:|
| MTP (cold) | 768 | 86 | 681/682 | 8.93 | 104 |
| DFlash2 (cold, `resumed at 0`) | 768 | 193 | 574/574 | **3.97** | 52.4 |
| DFlash2 (warm, `resumed at 69`) | 1 | — | refused | — | — |
| no drafts (cold) | 768 | 767 | 0/0 | 1.00 | 25.1 |

Read the drafter quality from the *cold* DFlash2 row: it accepted **100%** of the drafts it was
allowed to offer (574/574). The drafter is not the problem. Two things are:

## 2. Defect A — warm requests refuse with `DraftTapNotCovered`

On any prompt-cache hit the request dies after one token:

```
error: draft2 round refused (DraftTapNotCovered): span [0,76) low 69 written 76 covered 76 anchor 76 rows 2064
[tensorfold] ended chatcmpl-… reason=error (DraftTapNotCovered) prompt=76 tokens=1 after=0.07s
```

Mechanism, read off the log pair:

- Cold: `prompt cache: 76 tokens, resumed at 0, kept 2` — the whole 76-token prompt prefill runs,
  every position is tapped into the ring, `begin=0`, and the round's span `[0,76)` is covered.
- Warm: `prompt cache: 76 tokens, resumed at 69, kept 0` — the cache restores the first 69 tokens,
  so the prefill only re-runs the last 7, the ring is written for positions `[69,76)` only,
  `begin=69`, and the round's span `[0,76)` is refused as `sp.lo 0 < low 69`.

The ring is per-stream tap state and it is **not carried through the prompt-cache restore**. This
is exactly the failure `docs/design/…-tap-source-and-state`'s "(state restored without its ring)"
calls out, except that the `covers()` guard turns it into a hard refusal rather than silent stale
drafts. Consequence for measurement: `three_paths_bench.py` (and any warm regime) has been scoring
refusals, not acceptance — the "warm acceptance" numbers in earlier chronicles are meaningless.

Fix direction: the tap ring (and its `begin`/`written`/`covered` marks) must be restored with the
prompt-cache state, parallel to how the KV/DeltaNet state is. The ring is deterministic given the
prompt tokens + target weights, so it can ride the same content-addressed blocks the cache already
writes; the open question is whether to store the bf16 ring rows as a new block class or rebuild the
covered prefix's taps on restore.

## 3. Defect B — `draft_ms` priced 0.00, so the depth rule throttles to ~3 drafts

The startup line says it outright:

```
info: flashnext: --drafter …/Qwen3.8-Flash-Next-DFlash2 …: DFlash2, taps {5,15,25,35,45}, 7 drafts a round, 0.00 ms a round, 50.4 MiB a stream
```

Cold round arithmetic: 193 rounds / 768 tokens = 3.97 tokens/round, and 574 accepted over 193
rounds = **2.97 drafts offered per round** — the rule, pricing a round at `0.00 ms`, settles on a
~3-row verify window rather than the 7 the block computes. `decode` 14.6 s / 193 rounds = **~76 ms
a round**, which is *faster* per round than MTP's 85.6 ms, yet DFlash2 nets half MTP's tok/s purely
because it throws away four proposals it already computed. Fixing `measure()`/`draft_ms` so a
fixed-forward round is priced at its real ~76 ms (the enqueue-vs-completion timing trap) should make
the rule offer the full 7; at the observed 100% per-draft acceptance on this prompt that is ~8
tokens/round, i.e. MTP-parity.

## 4. Harness gap (separate from the engine)

`tools/dflash2/three_paths_bench.py` reads `tensorfold.speculative.{drafted,accepted}` for its
`acceptance_rate` column — this build emits no `speculative` sub-block, so the column is always
`null`. The acceptance numbers live only in the server log line
`[tensorfold] done … rounds=N accepted=X/Y`. Fix the harness to parse that line (or have the server
emit the fields it reads), and never quote `X/Y` without `rounds` beside it: `X/Y` is a rate over
whatever width the rule offered, and a throttled server reports the same per-draft rate as a healthy
one while producing a fraction of the tokens.

## 5. What this does and does not close

- **Closes:** "the drafter's arithmetic is subtly wrong" — no; the cold serve accepted 100% of
  offered drafts, and the earlier stage-diff work (`flashnext-dflash2-divergence-brief.md` §10)
  already showed the forward is cuBLAS-equivalent to reduction noise.
- **Remains open:** the two serving defects above. They are what a live `serve` measures and what
  the single-round fixture structurally cannot see — which is the real reason the fixture-based
  hunt made no progress on the acceptance number.

The vendor bar for context (not yet re-measured on this box at width 7 vs width 4): acceptance
length ~2.8–3.3 at width 4, throughput ~1.7–1.8× at concurrency 1, on GB300/FP8. Our MTP cold row
(8.93 tokens/round, 4.2× over no-drafts) is already in that band, which is the control that says
the target and the box are healthy and DFlash2's deficit is Defect B, not physics.

## 6. Fix scope for Defect A (tap ring through prompt-cache restore)

The ring is per-stream DFlash2 state (`cuda_drafter2.zig`'s `State`), held in `Draft2.states`, and
it is **not** threaded through `cuda_lanes_snap.zig` → `cuda_snap.zig`'s `Snap` at all — that path
only carries `prompt.Seq` and the MTP head's `Draft` (`d: ?*Draft`). On a cache resume the prefill
only re-runs the tail, so the ring is written for `[resumed, end)` only and `begin=resumed`, where
`low = max(begin, written - rows) = resumed` and the round's `span [0, anchor)` is refused
(`sp.lo 0 < low resumed`).

Three fix directions, with cost:

- **A — carry the ring through the snapshot.** Add a ring piece to `Snap` (gated like the existing
  `head` flag), save/restore the ring rows plus `begin`/`written`/`covered`. Correct and complete,
  but the ring is 52.8 MiB a stream (2064 rows × 5 taps × 2560 × 2 B), which roughly doubles a
  drafting state's prompt-cache footprint (RAM and the NVMe `.st`/`.blk` image, whose on-disk format
  the change also touches) — a memory/disk budget decision, not just a code change.
- **B — degrade to MTP on a covered-span miss.** When the DFlash2 round cannot be drafted (ring not
  covered after a restore), fall through to the MTP head's ordinary verify instead of a 500. Cheap,
  lossless (the MTP fallback reproduces the no-draft text), and correct for warm prompts, at the cost
  of losing the draft speedup on exactly the warm case. Worth doing first regardless.
- **C — rebuild taps on restore.** Re-run the tapped target layers over the restored prefix. This
  defeats the whole point of the prompt cache (the target forward is what the cache skips); rejected.

Recommendation: **B now** (restores losslessness on warm immediately; small, no format change), then
**A** if warm-prompt draft throughput turns out to matter. Neither touches the drafter's arithmetic.