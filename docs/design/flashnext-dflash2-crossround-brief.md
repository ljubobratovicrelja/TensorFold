# Cross-round context fix — implementation brief (DFlash2 drafter)

Repo: `~/Projects/tf-wt/dflash2`, branch `dflash2`. **CODE ONLY** — do NOT build, run the server,
touch the GPU, commit, or push. The orchestrator builds, serves and measures. Your deliverable is a
precise code change with an exact account of what you changed and why.

Read these before editing (they are the diagnosis, already measured):

- `docs/design/flashnext-dflash2-acceptance-log.md` — §10 names the two open defects verbatim.
- `docs/design/flashnext-dflash2-divergence-brief.md` — §10: the drafter's forward math is CLOSED
  (bit-exact up to pinned summation order); the acceptance gap lives in the *served cross-round
  context*. §6 gives the debugging discipline (localise by first-divergent stage, read source there).

## The two defects to fix

Both are in the serving path that a single-round fixture cannot exercise. They are a matching pair:
together they let the drafter read stale/wrong tap rows from round 2 onward, which reads as
near-chance acceptance across slots 2–7.

### Defect (a) — `note()` records the wrong span (`cuda_lanes_verify.zig:51`)

```zig
if (self.drafter) |draft2| draft2.note(wins[wi].stream, q.fed, q.fed + @as(u64, @intCast(ids.len)));
```

This runs **after** `o.verify(q, d, ids)` (line 45), which has already advanced `q.fed` by
`ids.len`. So `note` records `[post_fed, post_fed + n)` — one window too far — instead of the
positions whose tap rows the verify pass actually wrote.

`draft2.note(stream, from, to)` records "the ring now holds tap rows for positions `[from, to)`"
(see `cuda_drafter2.zig` `note`/`State.begin`/`State.written`). Inflating `written` this way means
`absorbOne`'s gap detector (`start > st.written` → `DraftTapGap`) and `roundRefusal`'s
`anchor > written` / `sp.lo < low` checks all see a ring that holds rows it never wrote — the
"gap detector is blind to gaps up to 8 positions" symptom.

**The correct span.** The window's rows sit at `w.positions[r] == l.seq.fed + 1 + r` (checked at
`cuda_lanes.zig:600` against the pre-verify `l.seq.fed`), i.e. positions `[F+1, F+1+n)` where
`F` is the pre-verify fed. The verify pass writes tap rows for exactly those positions. The
correct note therefore records `[F+1, F+1+n)`.

**Do not trust the arithmetic above blindly** — trace it yourself: confirm exactly where `q.fed`
stands at line 51 (post-verify, = `F + n`), confirm `w.positions` against `cuda_lanes.zig:600`,
and derive the pre-verify `F` (available as `q.fed - n`, or better: capture `d.fed0` / the
pre-verify value explicitly so the intent is obvious and not arithmetic-dependent). The reference
for the correct pattern is the **plain (non-drafted) round** at `cuda_lanes.zig:629`, which records
`note(stream, seq.fed - 1, seq.fed)` — the single position that round actually computed. Mirror
that convention exactly for the n-row window.

### Defect (b) — `st.st.end = 0` kills the cross-round gap guard (`cuda_drafter2.zig:524`)

```zig
st.st.end = 0;
```

This clears the forward's cross-round high-water mark every round, on a stale rationale ("a round
rewrites its whole prefix into the k/v ring"). That rationale died when incremental K/V reuse
landed: a round now projects only `[reuseFrom(lo,hi,kv_lo,kv_end), hi)` and reuses the rest
(`cuda_draft2_fwd.zig` `prefix`/`fusePrefix`/`reuseFrom`). The forward's own comment on
`invalidate()` says it outright — *"Clearing `st.end` a round is **not** that signal, or the ring
would never carry anything."*

`st.end` is the block-end watermark: `draft()` sets `st.end = r.position + B` at the end, and its
guard is:

```zig
if (st.end != 0 and r.position > st.end) return error.DraftContextGap;
```

That guard exists to refuse a round whose position jumped past the previous block's end — the
"skipped round" case (copy round / plain round / rollback) — where the k/v ring genuinely has a
hole that would otherwise be read as stale context. Clearing `st.end` every round makes it dead.

**The fix is not simply to delete the line and hope.** Removing it makes the guard live, which is
correct, but you must verify what happens when `DraftContextGap` *can* fire in the served path:

1. Trace the drafted wrapper (`core/lanes/drafted.zig`) and confirm how a refused `hold`/round is
   handled — does it degrade to a plain round, or propagate to a 500? Read the actual call chain.
2. Confirm where a skipped round legitimately occurs mid-generation (copy round, plain round,
   rollback after partial accept) and whether that path calls `invalidate()` (→ `fusedReset()`,
   which clears `fused_lo/end` and `kv_lo/end` but NOT `st.end`) or `State.reset()` (which clears
   both). `invalidate()` alone leaves `st.end` stale, so the next draft would refuse with
   `DraftContextGap` even though the ring was just re-fused.
3. The correct end state: a round whose position is past `st.end` must either (i) be a genuine
   skip that the caller already invalidated/reset — in which case `st.end` must also be cleared
   at that same invalidation point, not every round — or (ii) be refused by the guard as it
   should be. Make `st.end` reset *coincide with* the position-drop that `invalidate`/`State.reset`
   already signals, and remove the unconditional per-round clear.

The goal is: the guard is live, and the skipped-round path re-fuses the whole prefix rather than
drafting from a hole. Keep `State.reset()`'s existing `st.end = 0` + `fusedReset()` (full stream
reuse must clear both). The question is only the per-round clear and the mid-generation skip path.

## What you must NOT change

- The forward's arithmetic (`cuda_draft2_fwd.zig` kernels, `fusePrefix`, `prefix`, `layer`) — it is
  bit-exact and closed. Do not "improve" precision.
- The tap basis (`mean4`), the layer mapping `[5,15,25,35,45]`, the selector, the head.
- Anything in `kernels/cuda/*.cu`.

## Deliverable

- The exact diff (file + line + before/after), plus a short written account of:
  1. For (a): the exact pre-verify fed you used and why it is the window's first row position.
  2. For (b): whether `DraftContextGap` can fire after your change in the served single-lane path,
     and the precise invalidation point that resets `st.end` on a genuine skip (file+line).
- Any host-only test you add or update (tests that do NOT touch the device — mirror the existing
  `cuda_drafter2.zig` test style for `note`/`span`/refusals). No GPU tests, no build.
