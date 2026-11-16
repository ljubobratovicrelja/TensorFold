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

## 7. The layer mapping is correct (measured 2026-11-16, closes the 2× intercept hypothesis)

The last structural explanation for the acceptance deficit was an off-by-one in the layer→tap mapping:
if `target_layer_ids` mean HF-style `hidden_states[i]` (where index 1 is layer 0's output) rather than
"layer i's completed output", every tap is one layer off — and *every* existing comparison would still
agree with itself, because the reference would be fed the same wrong rows.

Measured with the engine's own oracle rather than by reading. `TF_FLASHNEXT_TAP_LAYERS` overrides the
captured layer set with no rebuild; `TF_FLASHNEXT_DRAFT_CHECK=1` then walks the drafter over that ring
and scores each block against the sequence's **true continuation**. Three servers, one at a time,
identical checkpoint/prompt/env; the ring sha differs per run, so the capture really moved.

| layer set | `mean_advance` | accepted/draft | zero-accept |
| --- | ---: | ---: | ---: |
| `[5,15,25,35,45]` (shipped) | **1.450** | 0.450 | 72.5 % |
| `[4,14,24,34,44]` (down one) | 1.433 | 0.433 | 74.2 % |
| `[6,16,26,36,46]` (up one) | 1.450 | 0.450 | 75.0 % |

**Verdict: the mapping is not the bug.** Shifting down is worse; shifting up ties on `mean_advance`
and is worse on both tiebreakers (higher zero-accept, more candidates that propose nothing). The
shipped set wins or ties every column. A one-layer error would have shown a clear winner, not a
three-hundredths spread.

This was the only remaining hypothesis that could produce a 2× acceptance deficit *invisibly* — one
where our drafter agrees with the reference because both are fed the same wrong rows. With it closed,
"our drafter is the reference drafter" (1.317 vs 1.342, §5) stands as a measurement, not an artifact
of a shared error.

Harness note: `cuda_draft2_check.zig` walked the context as positions `[p+1-len, p]` with
`position = p`, so `ctx_len > position` at small `p` and the walk died on `ContextPastPosition`
before round 0. `Round.ctx` is `[position - ctx_len, position)` — the anchor's own position is not in
the keys — so the correct span is `[p-len, p-1]`, `len = min(ctx_rows, p)`. Fixed in
`zig/src/families/flashnext/cuda_draft2_check.zig` (branch `layerprobe`).

## 8. The dtype-matched layer probe: diverges from round 0, and the divergence compounds

Both sides ran the **same bytes** — a seeded `N(0,1)` tap ring and prompt, sha256'd before dispatch —
with the reference in vendor torch **bf16 on GPU** (`allow_bf16_reduced_precision_reduction = False`)
and ours through the production drafter path. 53 stages each, 52 compared. This is the first
comparison where a non-zero delta means a defect rather than a dtype difference.

| stage (window case, ctx 2047, 8-row block) | rel L2 | cosine | bit-identical |
| --- | ---: | ---: | --- |
| `fc_ctx_rows` (the fused context) | 3.3e-03 | 0.9999946 | no |
| `conv_attn_in_0` | 5.1e-03 | 0.9999869 | no |
| `block_hidden_0` | 8.5e-03 | 0.9999642 | no |
| `block_hidden_1` | 1.8e-02 | 0.9998489 | no |
| `block_hidden_2` | 4.0e-02 | 0.9992282 | no |
| `block_hidden_3` | **1.5e-01** | 0.9901629 | no |
| `block_hidden_4` | **2.6e-01** | 0.9693146 | no |
| `final_norm_block` | 2.4e-01 | 0.9720389 | no |
| `logits_full` | 2.1e-01 | 0.9783146 | no |
| `logits_top16_ids` | — | 0.0985 | no (21 % of ids agree) |
| `proposals` | — | 0.7774 | no |

**Reading.** No stage is bit-identical, so the two implementations are not computing the same
function. But the error is *monotone in depth*, not stepwise: it starts at 3e-3 on the shared input
and grows by roughly a factor of two per layer, hitting 26 % relative by layer 4. A single wrong
weight, a wrong bias or a misplaced reduction would show as a step — one stage jumping and the next
carrying it. Compounding per-step gain of ~1.5–2× on every layer is the signature of a *systematic*
difference reapplied each block: a dtype that is not really bf16 somewhere, a normalisation applied to
the wrong axis, or an attention scale/softmax temperature off by a constant.

That also makes it consistent with the acceptance deficit and with every earlier "the layers agree,
cos ≥ 0.99975" claim: those were bf16-against-fp32 and could not separate rounding from a real
per-layer gain of this size. They cannot be quoted again as agreement.

**What this did not yet say, and the isolation that answered it.** The compounding curve alone could
not name a faulty stage: it could equally be one bad input amplified by five healthy blocks. So each
block was then run *on its own* (`isolate_blocks.py`), twice — once from the reference's own previous
output, once from **ours** — with both outputs scored against our `block_hidden_k`:

| block | ours-in → ref block → vs our output | ref-in → ref block → vs our output |
| --- | ---: | ---: |
| 0 | **8.46e-03** | 8.46e-03 |
| 1 | 1.40e-02 | 1.77e-02 |
| 2 | 1.21e-02 | 3.98e-02 |
| 3 | 2.57e-02 | 1.62e-01 |
| 4 | 2.01e-02 | 2.46e-01 |

**The per-block divergence is flat at ~2e-2, and the compounding was an artifact of the experiment.**
Each block, fed a *matching* input, produces a mismatch of the same size regardless of depth. The
2.6e-1 at layer 4 in the full round is what the reference's own drift becomes when it is fed back into
itself five times — the isolation removes that by restarting each block from our state.

So there is no "layer 4 is wrong": the error is present, at constant size, from the first block. And
block 0 already fails on an input that is not in dispute (`noise_emb`, a table lookup on both sides),
with the same figure in both arms — which puts the fault inside the block body: the layer's attention,
conv or MLP arithmetic, not the context fusion and not the residual chain.

Still unexplained, and the next question: **why is a healthy bf16 block off by 2e-2 at all?** At bf16
with fp32 reduction the expectation for a correct implementation is ~1e-3 on a 2560-wide row — two
orders of magnitude smaller. The candidate that would produce exactly this: if our kernels round
*accumulations* to bf16 (accumulating in bf16 rather than fp32), every dot product carries a
relative error near bf16's unit roundoff and a 2e-2 difference is what it looks like. That is
checkable statically, in one grep across the drafter's kernels for accumulate types.

**Earlier reading, now superseded:** the paragraph above called this "a systematic difference
reapplied each block". The isolation shows it is not reapplied — it is constant per block. The
conclusion (our drafter and the reference are not computing the same function) stands.

**Harness note (fixed).** The first run of this comparison reported cosine ≈ 0 and rel-L2 exactly
1.000 on *every* stage, which is a dump fault, not a finding: `writeNpy` padded the header so
`10 + hlen` was a multiple of 64 and then stored `hlen - 10`, so numpy read every array ten bytes
late. numpy only checks the file is long enough, so it loaded silently — bf16 widenings came back as
denormal noise. Fixed in `86daffeaf` with a test that reads the declared offset back out of a written
file (fails on the old line: `expected 0, found 54`). The engine's own stage summary was correct
throughout (`fc_ctx_rows` mean 0.00017, std 1.04), which is what localised it to the writer.

Two further harness traps the isolation paid for: a dumped array is an **f32 widening** of a bf16
value, so it must be *cast* to bf16 to be fed back (`.view(torch.bfloat16)` on f32 reinterprets the
bytes and doubles the element count — it fails loudly only because a norm's weight is 2560 wide), and
`run_round` stores its stages unbatched, so indexing `[0]` takes a row, not a batch.

The `block8` case (pos_end 9) is not runnable: the check refuses it with `FixtureIdsShape`, because
`cuda_draft2_check.zig:415` requires `pos_end >= block + 2` = 10. The reference side runs it happily;
that fixture is one position too short and is a harness bug, not an engine one. Every number above
comes from `window`.

## 9. What is left

1. **Workload matching.** Establish which published cell, if any, corresponds to a
   greedy 8-wide run on plain prose with a short prompt. If none does, our numbers must be judged
   against the closest cell and the difference stated, not against a headline.
2. **Block-by-block isolation of the compounded divergence (§8).** Feed the reference our
   `block_hidden_k` in place of its own and find the first block that turns a matching input into a
   mismatched output. Until that runs, §8 localises the *shape* of the error, not the fault.
3. **Round cost.** 84 ms against 39.6 ms with no drafts; verify rows cost roughly 6 ms each on top of
   a ~33 ms floor. This is the only remaining lever with arithmetic behind it.

## 10. Open defects noted, not yet fixed

- `cuda_lanes_verify.zig:51` notes `q.fed` *after* the window advanced it, so `written` is inflated by
  one window and `absorbOne`'s gap detector is blind to gaps up to 8 positions.
- `cuda_drafter2.zig:524` clears `st.st.end` every round on the comment "a round rewrites its whole
  prefix", which stopped being true when the forward gained incremental reuse; the forward's
  cross-round `DraftContextGap` guard is therefore dead code with a stale rationale.
- The replay path has no assert tying `keeps` to `state_out`; the late path does
  (`cuda_batch_late.zig:28`). A kept pass with `state_out == 0` would double-apply on replay.
- No unit test pins *recurrent* state after a partial keep (the roll contract covers only the conv
  window versus copy+shift).
