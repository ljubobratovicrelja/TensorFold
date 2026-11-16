# DFlash2 acceptance: investigation log (2026-11-16)

Status: **open.** The drafter is correct; it is not yet fast. This file records what was measured,
what it excluded, and what is left, so none of it is redone. Numbers here are measured on the DGX
Spark (GB10), one request, greedy (temperature 0.0), concurrency 1, FP8 KV, no grammar, unless a line
says otherwise.

## 1. Where the project stood at the start of the day

`dgx-flashnext.service` is intentionally down; development happens on `dflash2` with its own server
on :8099. At `4228bd6a2` the drafter was fully wired (81/81 tensors, config parsed, selector and conv
arithmetic matching SGLang) and produced proposals that shared **no ids at all** with the reference:
ours `[8160, 0, 864, 5568, 1024, 3296, 864]`, reference `[248058, 198, 27, 1628, 21402, 956, 29]`.

## 2. The head bug (fixed)

`tf_fxd2_topf32` (`zig/kernels/cuda/qwen35_draft.cu`) accumulates a top-k list per *thread* over the
positions `v ≡ warp*32 + lane (mod warps*32)` of its 1024-entry tile, then published only `lane == 0`'s
list to the block merge. Every candidate id was therefore ≡ 0 mod 32: thirty-one thirty-seconds of the
vocabulary could not be proposed, with no error.

The kernel was adapted from the *scalar* `tf_fxd2_head_topk`, where a warp owns one vocabulary entry
and lane zero is the correct publisher. `tf_fxd2_headmerge` already reduces correctly (every thread
publishes; thread zero merges).

How the fault was localized, in order, each step able to have ended it:

1. Dump the stage before the suspect: the head's fp32 logits are dense (0 NaN, 0 inf, 0 zeros) and
   match a numpy product of the same rows to **2.6e-07 relative**. The GEMM, `glm_mm_tile16` and the
   detile are correct; the loss is entirely after them.
2. The detiled head matched `lm_head.weight` row-for-row (64/64 blocks), exonerating the weights.
3. Computing the top-16 in numpy from *our own* `final_norm_block` with the true `lm_head` reproduced
   the **reference's** ids — so layers, weights and detile were right, and only the candidate
   selection was wrong.

After the fix the proposals are exactly the reference's: `{248058, 198, 27, 1628, 21402, 956, 29}`.

Gate note worth keeping: the commit that introduced the bug was gated on an unchanged
`token_sha256`, which is **structurally blind** to a broken drafter — the target verifies every token
either way. A drafter change needs a proposal-level check against a reference.

## 3. The tap basis (correct by recipe, irrelevant to acceptance)

The drafter consumes 5 hidden-state taps from target layers `[5,15,25,35,45]`. incoai's own capture
patch computes them as `hidden_states.unflatten(-1, (hc_count, hidden_size)).mean(dim=-2)` — the plain
mean of the four **raw** hyper-connection streams — and the independent vLLM port does the same. The
tree had been serving `readout`: `fn_hc_dense_mix`'s gated mean over hc-**normalized** streams, a
different tensor (per-stream `hc_norm` means −0.400 / −0.455 / +0.207 / +0.648, so they cannot
coincide).

`TapSource.mean4` is now the default (fp32 accumulation, one bf16 rounding, `tf_fxd2_mean4`).
**MEASURED: it changes nothing.** One binary, one request, basis flipped by environment only:

| basis | accepted / verified | rounds |
|---|---|---|
| `mean4` | 48 / 103 | 77 |
| `readout` | 49 / 103 | 76 |

One candidate in 103. The recipe settles which tensor is correct; it does not move the number.
`mean4` stays the default because it is the trained one.

Two defects surfaced while establishing that:

- **The mean4 tests had never compiled.** The ring test handed `expectEqualSlices` a `*[4]u16`
  (`&ring[0..hidden]`). `zig build` was green because test code is not compiled by that step. All 7
  ring tests pass now.
- **`TF_FLASHNEXT_TAP_SOURCE` was inert.** `cuda_draft2_fwd.zig` held `tap_source` as a comptime
  constant, so `sourceFromEnv()` only ever reached the dump tool; two "different" bases returned
  byte-identical results. The drafter now reads it once at `init` and hands it to `hooks`. Lesson:
  an A/B must first show its arms *disagreeing* before a null means anything.

## 4. Served performance, measured

Same request, same prompt, 125 generated tokens, one binary family:

| configuration | drafts/round | accepted / verified | tokens/round | ms/round | tok/s |
|---|---|---|---|---|---|
| no drafting | – | – | 1.00 | 39.6 | 24.3–25.0 |
| depth as the rule picks it | 1.34 | 48 / 103 (0.466) | 1.62 | 66.4 | 24.0 |
| depth pinned to 7 | 7.05 | 67 / 409 (0.164) | 2.16 | 84.1 | 25.4 |

Pinning depth 7 (branch `depth7`, `TF_FLASHNEXT_FORCE_DEPTH`, uncommitted) buys **+33 % tokens per
round** but costs **+27 % per round**, so throughput moves ~5 % against a no-drafts baseline of ~25
tok/s. `token_sha256` is unchanged in every arm: drafting is lossless at full width.

Per-draft acceptance decays hard with slot: roughly 0.47 at slot 1 and ~0.10 across slots 2–7.

## 5. Hypotheses raised and closed

| hypothesis | verdict | evidence |
|---|---|---|
| Head GEMM / detile / weights | **closed** | logits match numpy 2.6e-07; detile 64/64 rows bit-exact; `lm_head` fingerprint matches the checkpoint |
| Tap basis (`readout` vs `mean4`) | **closed** | 48/103 vs 49/103, one binary |
| `sample_from_anchor` block layout | **closed** | absent from `dflash_config`; the 1+N layout is correct |
| Logit post-processing (`output_multiplier`, `final_logit_softcapping`) | **closed** | neither key exists in this checkpoint's config; the reference's transform is the identity |
| H3 — the depth throttle puts the block out of distribution | **closed** | the drafter always computes the full 8-row block; only the *verify* window is truncated, so there is no geometry change. Full width measured: acceptance does not recover |
| H1 — cross-round context append shifted or miscounted | **closed** | the read span ends exclusively at the anchor (`cuda_drafter2.zig:495`), the write base is that same absolute position, count is `keep = accepted + 1`, and `roundRefusal` (`:175`) would refuse every round from the second on if it were `accepted`. Rejected rows are written but structurally unreadable (positions ≥ anchor, overwritten by the next window's row 0) |
| H2 — GDN conv/recurrent state not rolled back after partial accept | **closed** | a kept pass writes its end state to a per-stream spare and leaves `seq.state` at the *window start*, so an accepted prefix is reconstructed explicitly (`cuda_mtp_ops.zig:396-426`, `cuda_mtp_roll.zig:44-68`); rejected tokens never advance the state. The sparse-attention indexer is position-addressed and needs no rollback |
| Drafter quality vs the reference | **closed** | the multi-round check walk reproduces the reference's acceptance length almost exactly: 1.317 vs 1.342 tokens/round |

## 6. Reading the published numbers

The headline figures ("2.4–3.3", "4.3–5.5") are **acceptance length** — tokens per verification step
*including* the free bonus token — not accepted drafts per draft. Converted, the published band is
roughly 0.44–0.85 per draft. Our 2.16 tokens/round = 1.16 accepted + 1 bonus.

Any future comparison must state its cell (suite, width, sampling, concurrency, max tokens); several
published numbers were measured at a capped width of 4, which is not our width.

## 7. What is left

1. **Workload matching (in flight).** Establish which published cell, if any, corresponds to a
   greedy 8-wide run on plain prose with a short prompt. If none does, our numbers must be judged
   against the closest cell and the difference stated, not against a headline.
2. **Dtype-matched layer probe (in flight, branch `probe-layer`).** Feed identical random inputs to
   the reference drafter and ours, both bf16 on GPU, and diff every stage. The existing evidence says
   the layers agree to 0.44–2.23 % with cos ≥ 0.99975, but that comparison was bf16 against fp32 and
   cannot distinguish rounding from a real divergence. This probe is the only way to say
   "bit-identical" or name the stage where it is not.
3. **Round cost.** 84 ms against 39.6 ms with no drafts; verify rows cost roughly 6 ms each on top of
   a ~33 ms floor. This is the only remaining lever with arithmetic behind it.

## 8. Open defects noted, not yet fixed

- `cuda_lanes_verify.zig:51` notes `q.fed` *after* the window advanced it, so `written` is inflated by
  one window and `absorbOne`'s gap detector is blind to gaps up to 8 positions.
- `cuda_drafter2.zig:524` clears `st.st.end` every round on the comment "a round rewrites its whole
  prefix", which stopped being true when the forward gained incremental reuse; the forward's
  cross-round `DraftContextGap` guard is therefore dead code with a stale rationale.
- The replay path has no assert tying `keeps` to `state_out`; the late path does
  (`cuda_batch_late.zig:28`). A kept pass with `state_out == 0` would double-apply on replay.
- No unit test pins *recurrent* state after a partial keep (the roll contract covers only the conv
  window versus copy+shift).
