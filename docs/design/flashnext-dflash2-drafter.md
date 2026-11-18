# Flash Next + DFlash 2 drafter: feasibility and integration plan

Status: **in progress.** Steps 1, 2 (hook only) and 8 landed; step 0 is blocked on an engine-side tap
dump; steps 3-7 not started. Target: `Qwen3.8-Flash-Next-FP4-Local` (family `flashnext`) drafting with
`incoai/Qwen3.8-Flash-Next-DFlash2` instead of — or beside — its own MTP head.

**A new agent should read §0 first: it holds the per-step status, the measured MTP baseline that any
DFlash2 run must beat, the dead ends already ruled out, and the traps.**

Written 2026-11-16 against `running` @ `bb95ab741`. Checkpoint facts below were read from the downloaded
file, not from the model card; the card's numbers are quoted as the vendor's and labelled as such.

## 0. Where the work stands (read this first)

Branch **`dflash2`**, worktree `~/Projects/tf-wt/dflash2`, pushed to `origin` at `1fffa44b0`.
`running` is untouched at `bb95ab741` and still serves the box. Nothing here has been merged.

| step | state | what exists |
| --- | --- | --- |
| 0 tap-row measurement | **MECHANISM DONE; blocked on the torch reference** | engine exposes `TF_FLASHNEXT_TAP_DUMP`, real dumps on disk; `tools/dflash2/analysis/reference.py` scores 0/7 — see §0.6 |
| 1 MTP baseline E2E | **DONE** | measured, see §0.2 |
| 2 tap capture | **RING WRITE DONE (single stream, all three bases; `mean4` default)** | `Hooks.tap_ring/tap_rows/tap_source`, `Pass.tapRingWrite`; persistence through Snap/disk still missing |
| 3 drafter forward on CUDA | not started | spec in `flashnext-dflash2-kernels.md` |
| 4 candidate selector on CUDA | not started | same spec, §2.5 |
| 5 `openDrafter` wiring | not started | `cuda_drafted.wrap` still refuses flashnext |
| 6 ring persistence (snapshot/disk tier) | not started | risk memo in `flashnext-dflash2-tap-ring.md` |
| 7 final two-path E2E | not started | harness exists (`tools/dflash2/harness/two_paths.py`) |
| 8 tensor contract | **DONE** | `cuda_draft2_weights.zig`, 6 tests |

### 0.1 Step 0 is blocked, and here is why (do not re-try the torch route)

The plan's step 0 assumed the target could be run in torch to capture taps. It cannot, on this box:

- `Qwen4ExpForCausalLM` **does** exist in transformers 5.22 (an earlier note claiming otherwise is stale).
- But a bf16 conversion is **225 GiB of routed experts alone** — 512 experts x 640 x 2560 x 48 layers —
  against ~108 GB available. Loading the FP4-Local checkpoint OOM-killed at 2% (`RC=137`).
- The official `Qwen/Qwen3.8-Flash-Next-FP8` is 172.8 GiB and the bf16 original 335.3 GiB; neither fits.
- The checkpoint needs modelopt dequant that transformers cannot do on load.

**So the tap row must be measured through the engine.** The mechanism is in place for it: `Hooks.taps_out`
is already copied per tapped layer. What is missing is a caller that allocates those buffers and writes
them to disk — a `TF_FLASHNEXT_TAP_DUMP=<dir>` env path that sets `hooks.taps`, `hooks.n_taps` and
`hooks.taps_out` for one prompt pass, then dumps the five `[rows, 4 x 2560]` bf16 tensors. Feed those
to `tools/dflash2/analysis/reference.py` (which runs the real drafter end to end) and compare acceptance length
for `TapSource.readout` / `.normed` / `.mean4`. That is the whole experiment.

### 0.2 The MTP baseline that DFlash2 must beat (measured, not inherited)

`dflash2` build, 32k window, 2 lanes, 256 greedy tokens, prompt in `tools/dflash2/harness/two_paths.py`:

| path | tok/s | TTFT | `token_sha256` |
| --- | ---: | ---: | --- |
| MTP | 78.66 | 0.235 s | `5d72e1f7006dd05be1b68f6bad045dfe8c4984a72fdc4ab9472c19a53ab5d54d` |
| no drafts | 25.36 | 0.159 s | same |
| MTP after the tap hook | 80.33 | 0.182 s | same |

Three facts a new agent needs: the two paths already agree byte for byte, 3.1x of decode rate is
speculation, and the tap hook did not perturb either. **Any DFlash2 run must reproduce that sha and
beat 78.66 tok/s**, or the integration is wrong somewhere.

### 0.3 What was built, file by file

- `zig/src/families/flashnext/cuda_draft2.zig` (288 lines) — the drafter's config parsing and refusals.
  Self-contained because families are separate build modules: flashnext imports only
  `cuda/core/lanes/qwen_vision/living/engine_api`, **not** `qwen3_5`, so its parser could not be shared.
  Adds two checks qwen3_5 cannot make: the drafter's `hidden` and `vocab` must equal the target's.
- `zig/src/families/flashnext/cuda_tap_ring.zig` (154) — ring arithmetic (`p % rows`), `Ring.covers` to
  catch a restore that wrapped early, and `TapSource { readout, normed, mean4 }` with one `default`.
- `zig/src/families/flashnext/cuda_draft2_weights.zig` (234) — the 81-tensor contract. Exact byte
  accounting: **1,206,131,200 bytes** of tensors, 8,808 less than the file (header + padding). The real
  count is **6 + 15 x layers**, not the `4 + 16 x layers + 3` an early prompt guessed.
- `cuda_batch_state.zig` / `cuda_batch.zig` — `Hooks.taps/n_taps/taps_out`, `tapSlot`, and
  `Pass.tapDump`, called after `moeAdds` while `s.h` still holds that layer's output. Inert at
  `n_taps = 0`; pinned by a test that runs at `cuda_batch_state_test` 179/288.
- `tools/dflash2/analysis/reference.py` (274) — torch port that runs the **real checkpoint** end to end
  (forward + selector), verified: batch-2 is bit-identical to batch-1.
- `tools/dflash2/harness/two_paths.py` (105) — the equivalence check above.
- `tools/dflash2/harness/tfsrv.sh` (54) — test server by pidfile. **Never `pkill -f tensorfold-native`**: the
  pattern matches the caller's own `bash -c` and killed the shell once while writing this.

### 0.4 Traps a new agent will hit

- **Subagents are unusable right now.** Every `delegate_task` in this session died at ~44s with
  "API call failed after 3 retries: Connection error" — the children inherit the qwen-next provider,
  which is served by the very service this work touches. Do the work inline.
- **The service is down by Relja's hand** (he stopped `dgx-flashnext` to free the box). Never start,
  stop or restart it. Test servers on other ports, by pidfile, are fine and were used throughout.
- **`zig build test` takes ~4-6 min and `zig build -Dgpu=spark` ~12-15 min.** Run them serially, never
  two at once. Judge by `RC`, not by `error:` lines: the suite prints hundreds on purpose.
- **Build before serving.** `zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc` from the worktree;
  the binary is `zig-out/native/bin/tensorfold-native`.

### 0.5 The one risk that could sink the payoff

The drafter scores the **full 248,320-token head** every verify round, where the MTP head only scores
its own draft vocabulary. That is ~2.0 M logits a round against ~0.3 G MACs of backbone — plausibly
more expensive than the MTP chain it replaces. `core.proposal_vocab` (used by `cuda_mtp_load.zig`)
already exists to slice a candidate vocab; **measure the head's share before optimising anything else**,
and be prepared for the honest answer that DFlash2 loses to MTP on this box.

---

### 0.6 State after the 2026-11-16 session (verified, not planned)

The ring exists in the engine and produces real rows. What landed:

- `Hooks.tap_ring / tap_rows / tap_source` (`cuda_batch_state.zig`), `Pass.tapRingWrite`
  (`cuda_batch.zig`), and `TF_FLASHNEXT_TAP_SOURCE=readout|normed|mean4` (`cuda_tap_ring.zig`,
  `mean4` the default: §4.1b).
- The tap is **deferred by one layer on purpose**: the mixed/normed row over layer L's output is
  produced by layer L+1's input mix, so `tapDump` only names the slot and `tapRingWrite` copies it
  at the top of the next layer. Writing at `tapDump` time can only give the four-stream `s.h`.
- `TF_FLASHNEXT_TAP_DUMP=<dir>` (`cuda_tap_dump.zig`, wired in `Engine.forward`) writes
  `ids.bin`/`ring.bin`/`manifest.json` for one prompt pass; `tools/dflash2/analysis/tap_dump_run.py` drives it.
- Trap paid for twice here: `copyCols` takes **byte** addresses but **value-unit** pitches and
  width. A contiguous `copy` is wrong for a ring whose slots interleave five taps.

Measured on this box, 1455-token prompt, fp8 KV, 32768 window:

| claim | measured |
| --- | --- |
| ring is the drafter's fixed cost | 2064 slots × 5 × 2560 bf16 = 52,838,400 B |
| the taps are real residuals | layer-45 tap (unweighted rms) through the target's `lm_head` puts the true next token at rank **107 / 7 / 5 / 0** of 248,320; layers 5/15/25/35 degrade as expected |
| same logits under fp8 and bf16 KV | identical `token_sha256` `499ef81c…` |
| taps bit-identical across KV precision | **no — 4.9% mean relative delta, max 4.3.** §4.4's bit-identity expectation is wrong; e4m3 KV cannot give bit-identical residuals. The gate is the sha plus a bounded delta, and the acceptance experiment must use **fp8** taps because that is the served configuration. |

`tools/dflash2/analysis/reference.py` still scores **0.000 of 7 accepted at 100% of rounds** (mean advance
1.000) on the real dump, including its leaky control — at chance, so the torch forward is
structurally wrong while taps/head/embed are proven good by the rank test above. Until it is fixed
there is no numeric ground truth to gate a Zig port against, which makes it the next task rather
than an optional one.

### 0.65 The CUDA drafter runs in the engine and reproduces the reference (2026-11-16)

`TF_FLASHNEXT_DRAFT_CHECK=1` runs the real CUDA drafter inside the engine over a captured tap
ring, prefix context and all, and reports the same metric as the torch harness:

| path | mean advance | accepted of 7 | zero-accept |
| --- | ---: | ---: | ---: |
| torch reference (`tap_accept.py`, same dump) | 1.342 | 0.342 | 80.0% |
| **engine (`draftcheck`, same prompt, fp8)** | **1.317** | 0.317 | 81.7% |

Machinery is in `cuda_draft2_check.zig` (the walk, head detile, dumps) + `cuda_draft2_fwd.zig`
(`Round.ctx/ctx_len`, `ctx_chunk` prefix → k/v → ring). Two bugs were caught on the way, both by
instrument rather than by reading: `compute-sanitizer` named `tf_fxd2_select` reading a codebook
511 GB past its allocation (a row offset applied twice — the candidate table is `[rows, topk]` with
no block base, and the address it hit was a *logit* reinterpreted as a token id), and the head
normalised block rows 0..6 instead of 1..7, which lags every proposal one slot and reads as chance
agreement (1.008 → 1.317 once fixed).

Still missing before a serving row exists: `openDrafter` for flashnext and the per-round drafting in
the batched lane path, so `--drafter` can serve. The check is a diagnostic, not a serving path.

### 0.7 The comparison the DFlash2 path must beat (measured 2026-11-16)

`tools/dflash2/harness/matrix_bench.py`, warm (one unmeasured pass of the same prompts per cell, every
measured row 100% `cached_tokens`), 3 reps a cell, fp8 KV, 131k window, 4-lane server,
`--max-tokens 256`, thinking off, request model `qwen3.8-flash-next`:

| fill | 1 agent | 2 agents | 4 agents | 4-agent aggregate |
| --- | ---: | ---: | ---: | ---: |
| 0 (76-token prompt) | 107.2 tok/s | 96.0 | 84.6 | 321.5 |
| 20k (19,995) | 106.9 | 84.6 | 67.0 | 237.6 |
| 100k (100,178) | 102.5 | 78.8 | 59.6 | 204.6 |

Single-stream decode is flat across fill (107 → 102), so the cost is per-round work at lanes, not
context length. TTFT 0.06 s at 4 short agents, 0.54 s (max 0.96) at 4×100k. Receipt:
`tools/dflash2/run/matrix.jsonl` + `matrix-mtp-report.md`.

MTP is also proven unregressed against `3e8531e6a` with the same 3-rep protocol: baseline median
70.40 tok/s vs 68.87 on the changed tree, ranges overlapping and `token_sha256` identical in all
six runs — the plan's earlier 78.66 was a single unrepresentative sample.

## 1. What the drafter is (read from the checkpoint)

`incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744e85c17364dcc54a6a420af7135d4c7f6`, one
`model.safetensors` of 1,206,140,008 bytes (1.21 GB), 81 tensors, all BF16, no index file, no
`draft_vocab.json`, no `target.embed`/`target.head` copies.

`config.json` (verbatim values that matter):

| field | value | consequence |
| --- | --- | --- |
| `architectures` | `DFlash2DraftModel` | SGLang/SpecForge draft-model class, not our block-drafter v1 |
| `model_type` | `qwen3` | the backbone is a plain Qwen3-style stack |
| `hidden_size` | 2560 | **equals the target's `hidden`** — the target's head reads the drafter's rows |
| `num_hidden_layers` | 5 | `sliding_attention` × 5, `sliding_window` 2048 |
| `num_attention_heads` / `_kv_heads` | 32 / 8, `head_dim` 128 | GQA 4:1 |
| `intermediate_size` | 7680 | SwiGLU |
| `vocab_size` | 248320 | the target's vocabulary (drafter does **not** ship a head slice) |
| `dflash_config.target_layer_ids` | `[5, 15, 25, 35, 45]` | 5 taps × 2560 = **12,800** features |
| `dflash_config.block_size` | 8 | 7 draft tokens a verify round |
| `dflash_config.mask_token_id` | 248077 | the noise row |
| `dflash_config.conv_kernel_size` / `conv_group_size` | 2 / 16 | two-tap dynamic depthwise conv, 160 groups |
| `dflash_config.selector_rank` / `selector_top_k` | 256 / 16 | low-rank re-ranker over the head's top-16 |
| `is_causal` | `false` | block attention is **bidirectional within the block** |

Weight budget by group (measured from the safetensors header):

- `candidate_selector` **255.6 MB** (1 tensor of the size: `predecessor_codebook` [248320, 256] and
  `successor_codebook` [248320, 256] at 127 MB each, plus `hidden_projection` [256, 2560] 1.3 MB)
- `fc.weight` [2560, 12800] 65.5 MB — the tap fusion
- 5 × `mlp` 118.0 MB each = 590 MB
- 5 × `self_attn` 52.4 MB each = 262 MB
- 10 × `*_conv` (attention_conv + mlp_conv, 5 layers) 3.3 MB each = 33 MB
- norms and `hidden_norm`/`norm` ≈ 0.1 MB

The selector is **21% of the file** and is a full-vocabulary pair of codebooks — it is a memory
cost that does not shrink with `--drafter-bits`.

## 2. How it runs (from the reference implementations)

Read from `sgl-project/SpecForge` `specforge/modeling/draft/dflash2.py`, `heads/candidate_selector.py`
and `dflash.py` (the base class), cross-checked against the HF card and `inco.ai/blog/dflash2/`.

Per verify round, for one stream:

1. **Anchor.** Row 0 of the block is the token the target already produced (verified or sampled).
   Rows 1..7 are `mask_token_id`.
2. **Fuse.** `target_hidden` = concat of the target's residuals after layers 5/15/25/35/45, then
   `hidden_norm(fc(x))` → a 2560-wide context row per position.
3. **Backbone.** 5 Qwen3 layers, each doing *cross*-attention against the fused context plus
   *self*-attention among the 8 block rows (bidirectional, `is_causal: false`, sliding window 2048).
   Each attention and each MLP is wrapped by a `DFlashGroupedConv`: one projection of the sublayer
   input predicts per-position, per-group kernel deltas (`base_kernel` + delta, base starts as an
   identity), applied as a 2-tap depthwise conv *within the block*. This is what stops the tail of
   the block decaying.
4. **Propose.** `target.lm_head(draft_hidden)` → unary logits; take top-16 per position.
5. **Select.** `CandidateSelector.build_lattice` scores every (predecessor, successor) pair:
   `unary + (pred_codebook[prev] * hidden_projection(h)) · succ_codebook[cand]`. `greedy_path` walks
   the lattice to one coherent chain of 7.
6. **Verify.** The target runs the 8 rows in one forward; acceptance is the longest matching prefix.
   Lossless: greedy output equals the target's, sampling preserves the distribution.

Two facts that matter for us and are easy to get wrong:

- **The taps are raw, unnormalized residuals**, concatenated in `target_layer_ids` order
  (`extract_context_feature` reads `hidden_states[layer_id + 1]`, offset 1 being the embedding).
  Normalization happens once, in the drafter's own `hidden_norm`.
- **They are the target's *completed layer output*, averaged over hyper-connection streams** — the
  SGLang capture patch asserts `hidden_states.unflatten(-1, (hc_count, hidden)).mean(dim=-2)`.
  This is the single most important port detail; see §4.

## 3. What the fork already has

More than a first reading suggests, but the reusable part is narrower than it looks.

| piece | where | usable for DFlash2 on flashnext? |
| --- | --- | --- |
| `serve --drafter DIR` / `--drafter-bits` | `server/cli.zig` | **yes**, generic |
| `pull` recognizes a DFlash2 folder by `dflash_config` | `cli/pull.zig` | **yes**, already matches this checkpoint |
| `lanes.drafted.Drafted` wrapper | `core/lanes` | **yes**, generic over any family backend |
| `cuda_drafted.wrap` + `F.openDrafter` hook | `native/cuda_drafted.zig` | **yes**, the integration point |
| qwen3_5 DFlash2 port (~1,750 LOC: `cuda_draft*.zig`, `qwen35_draft.cu`) | `families/qwen3_5/` | **partially** — 5-layer Qwen3 backbone is nearly identical, but it has **no** grouped conv and **no** candidate selector |
| tap ring (`setTaps`/`tapBytes`, `cuda_cache.zig`) | `families/qwen3_5/` | **as a pattern only** — flashnext has its own cache layout |
| hyper-connection machinery | `families/flashnext/cuda_batch_hc.zig` | **yes** — and it is exactly the tap point |
| MTP head | `families/flashnext/cuda_mtp.zig` | stays; DFlash2 is additive, selectable |

**flashnext has no DFlash2 support at all** — no `openDrafter`, no tap ring, no `dflash` reference in
the family. Today `cuda_drafted.wrap` refuses it with *"flashnext checkpoints load no draft model on
CUDA"*: the refusal is the missing `@hasDecl(F, "openDrafter")`, not an admission gate.

### The tap point exists and is well-defined

flashnext keeps 4 hyper-connection streams. `cuda_batch_hc.zig` already does, per layer:
`norm → hc down-projection → mix`, i.e. `writebackNorm` then `readoutNative`. The "completed layer
output averaged over streams" DFlash2 wants is **precisely the down-projected/mixed read-out row**,
so the tap is a read of a buffer that already exists — not a new computation. The hook is one store
per tapped layer into a per-stream ring.

## 4. The blockers, in order of how much they hurt

### 4.1 Tap semantics on a 4-stream family — **narrowed against the weights; the choice still needs one measurement**

**What is settled (measured, no experiment):** the target's hyper-connection mix is a *learned*
`[320, 10240]` projection, not a uniform mean, so the SGLang capture patch's `.mean(dim=-2)` does not
transfer, and a layer's output is genuinely 4 x 2560 wide. That leaves a small, named set of candidate
rows rather than an open field — `TapSource { readout, normed, mean4 }` in `cuda_tap_ring.zig`.

**What is not settled:** which of those three the drafter was trained against. That is decided by
acceptance length, and it is blocked on the engine-side tap dump (§0.1).

`docs/drafter-release.md` §"Not in version 1" says it outright: *"A tap that merges several residual
streams. `stream` has one value until a family needs another."* DFlash2 for Flash-Next is exactly
that case. Two candidate tap definitions:

- **(a) post-readout row** — a single 2560-wide row after the hyper-connection read-out.
- **(b) mean over the 4 raw streams** — literally what the SGLang patch does
  (`unflatten(-1, (hc_count, hidden)).mean(dim=-2)`).

**Both are wrong for flashnext, and the weights prove it.** Measured from the target checkpoint
(`Qwen3.8-Flash-Next-FP4-Local`, layer 5):

| tensor | shape | reading |
| --- | --- | --- |
| `attn_hyper_connection.input_mix_weight_down.weight` | [320, 10240] | **10240 = 4 × 2560** — the mix reads all 4 streams |
| `attn_hyper_connection.input_mix_weight_up.weight` | [10240, 320] | writes back **10240** — the layer output stays 4-stream |
| `attn_hyper_connection.hc_norm.weight` | [10240] (f32) | the norm is over the 4-stream tensor, not a merged row |

And `input_mix_weight_down` has `absmean 0.0515`, `absmax 2.03` — it is a **learned** mix, not a
uniform `mean(dim=-2)` (which would give a fixed 0.25 scale per stream). So:

1. flashnext's layer output is genuinely 4×2560 wide; the read-out that produces a single 2560 row
   is a *further* learned projection, and `fc`'s 12800 = 5 × 2560 confirms the drafter expects
   exactly 2560 per tap.
2. The SGLang patch's `mean(dim=-2)` is what *their* `Qwen4ExpModel` needed — it asserts
   `hidden_states.shape[-1] == hc_count * hidden_size` then averages. On flashnext the analogous
   quantity is whatever single 2560-wide row the family defines as "the completed layer output",
   and that choice must be made **from flashnext's own forward**, not copied from the patch.

**The experiment (§6 step 0) is therefore not "does the port work" but "which flashnext row is the
tap"** — read-out row vs down-projected row vs the normed variant. It is a cheap, bounded question
with three candidates, and it is answered by acceptance length, not by argument. Getting it wrong
does not crash: it silently produces plausible-but-wrong drafts and a low acceptance length.

### 4.1b The answer is (b), the plain mean — and it is now the default (2026-11-16)

Everything above reasons from the *target's* weights; the row the drafter was trained on is a fact
about the *drafter's capture recipe*, and that recipe is on the record. incoai's
`patches/qwen38-dflash2-capture.patch` for SGLang captures a DFlash2 tap as

```python
hidden_states.unflatten(-1, (hc_count, hidden_size)).mean(dim=-2)
```

i.e. the plain mean of the four **raw** streams — no gate, no `hc_norm` — and the independent vLLM
port of the same checkpoint does the same. So option (b) is the training-time tensor, and the
`readout` row this tree served first (`fn_hc_dense_mix`'s gated mean over *normalized* streams) is a
different tensor by construction. The measured symptom of serving the wrong one is acceptance around
1.5 tokens a round against the card's 2.4-3.3 band.

What landed: `TapSource.default = .mean4` (`cuda_tap_ring.zig`), the collapse kernel `tf_fxd2_mean4`
(`kernels/cuda/qwen35_draft.cu`: fp32 accumulation, one bf16 rounding), `Pass.tapRingWrite`'s mean4
arm writing into the pass's `tap4` scratch, and `TF_FLASHNEXT_TAP_SOURCE=readout|normed|mean4` as the
A/B so the other two bases can be re-measured without a rebuild.

**MEASURED (2026-11-16): the basis is not the acceptance limiter.** One binary, one prompt, one
request, the basis flipped by environment only — `mean4` 48/103 candidates, 77 rounds; `readout`
49/103, 76 rounds. Per-draft 0.466 against 0.476, i.e. indistinguishable (one candidate in 103).
Both bases sit far below the card's band, so the recipe fixes *which tensor is correct* and nothing
about the number: whatever costs acceptance is upstream or downstream of the tap row, not in the
choice of it. `mean4` stays the default because it is the tensor the drafter was trained on, not
because it measured better.

The same run exposed why the A/B was untrustworthy before: `cuda_draft2_fwd.zig`'s `tap_source` was a
comptime constant, so `TF_FLASHNEXT_TAP_SOURCE` reached only the dump tool and never the serving path
— two "different" bases produced byte-identical results. The drafter now reads it once at `init`
(`cuda_drafter2.zig`) and hands it to `hooks`. Any future A/B should first confirm the two arms
*disagree* before reading anything into a null.

### 4.2 DeltaNet / sparse-attention state interaction
The target's own KV is Gated DeltaNet + full attention + a Qwen Sparse Attention indexer. DFlash2
only reads residuals, so the *drafter* does not care — but **any state keep, snapshot or NVMe
imprint must carry the tap ring alongside the DeltaNet state**, or a restored stream drafts from a
truncated/misaligned ring. The existing disk tier (`--prompt-cache-disk`) writes a `.st` per state and
`--learn` rows; a tap ring is per-stream and must join `streamBytes`, `Snap`, and the kv8
precision-match check. This is the subtle, correctness-critical part and the one most likely to ship
a silent regression.

### 4.3 Memory at 4 × 262,144 with fp8 KV
Measured, not estimated (safetensors header + the shapes in §1):

| item | bf16 | 4-bit backbone |
| --- | ---: | ---: |
| drafter weights, total | 1,206 MB | ~493 MB |
| — selector codebooks (do **not** quantize) | 255.6 MB | 255.6 MB |
| — backbone (fc, 5 layers, convs, norms) | 950.5 MB | ~237.6 MB |
| tap ring, 5 × 2560 bf16 × 2056 rows | 50.2 MiB/stream → **0.20 GiB** at 4 streams | same |
| drafter KV, 5 layers × 8 KV heads × 128 × 2048 | 40 MiB/stream → **0.16 GiB** at 4 streams | same |

**Total added ≈ 0.85 GiB at `--drafter-bits 4`, ≈ 1.56 GiB at bf16.** The tap ring is the cheap part —
worth saying plainly, because the instinct is to assume the 262k window makes it expensive. It does
not: the ring is a fixed ~2k rows, independent of context.

The last journal line reported ~27.0 GiB spare after a 12 GiB reserve at 4×262k fp8 + vision, so the
fit is not in doubt at these sizes. Still prove it with a real start (§6, step 1) — an admission
line is a fact and arithmetic is a claim.

### 4.4 fp8 KV × drafter
`kv8` is the served cache format. The drafter's own attention is bf16 Qwen3 and independent of the
target's cache, so this is mostly orthogonal — but the tap rows are read out of the target's
forward, and the fp8 path (`cuda_heads.zig`, `flashnext_attn.cu`) is where the residual lives. Verify
taps are bit-identical under `--kv-cache fp8` and bf16 before any speed claim (same discipline as the
vision work's `token_sha256` gate).

### 4.5 `--parallel 4` and the batched lane path
flashnext runs a shared batched pass across lanes (`cuda_batch*`). DFlash2's drafting is per-stream
block-shaped; the wrapper must draft per stream while the target verifies in the batched pass. The
existing `Drafted` wrapper does this for other families, but flashnext's batched DeltaNet keep path
(`cuda_batch_gdn_lanes.zig`, `cuda_batch_side.zig`) is where a per-stream draft interacts with the
shared round. Expect this to be the bulk of the debugging, not the drafter maths.

## 5. Expected payoff — and why "wash at 4 lanes" is not established

Vendor table (HF card, target = official FP8 checkpoint, SGLang, **one GB300**, block 4 because their
QSA implementation capped it, temp 1.0/top-p 0.95/top-k 20, xhigh effort):

| | MTP | DFlash 2 |
| --- | ---: | ---: |
| acceptance length (GSM8K … MT-Bench) | 2.50–3.08 | 2.83–3.26 |
| c=1 speedup vs AR | 1.35–1.59× | **1.63–1.82×** |
| c=8 speedup vs AR | 1.11–1.25× | **1.31–1.48×** |
| c=32 speedup vs AR | 1.12–1.24× | **1.43–1.73×** |

What that does and does not tell us:

- It never shows DFlash2 *below* MTP at any concurrency. My earlier "wash or regression at 4 lanes"
  was an inference from the narrowing gap and was **too pessimistic**; the honest statement is
  *unknown, most likely a modest win in aggregate*.
- Their runtime caps the block at **4**; this checkpoint is **block 8** (7 drafts). More drafts per
  round than their published configuration — good for acceptance if it holds, worse if acceptance
  length per position decays. Must be measured, not assumed.
- Our own numbers argue speculation still pays at our operating point: `bench_concurrent` measured
  4-lane aggregate **183 tok/s drafted vs ~100 undrafted**, and single-stream prose ~56 tok/s against
  the ~25 tok/s no-draft floor. A drafter that adds accepted tokens per round has room to help.
- Counterweight: GB300 has far more bandwidth than GB10. The per-round cost of a 0.6B backbone
  forward + 8-row verify + selector walk is a larger fraction of a Spark round than of a B300 round.
  Our per-round cost is already the residual gap vs the v1.1.4 recipe (85 vs 71 ms/round, audit doc).
- GB10 is bandwidth-lean, which cuts both ways: verify is cheap relative to drafting, so *acceptance*
  is worth more than raw draft speed — the selector's whole job.

**Prior, stated as a prior:** single-stream +10–25% over current MTP; 4-lane aggregate +5–15%; if the
tap semantics in §4.1 are wrong or the block-8 chain decays, 0 to negative. Any number outside that
band is a finding, not a confirmation.

## 6. Plan

Ordered so every step is falsifiable and cheap before the expensive ones. Nothing here touches the
serving unit; all work is on a short branch off `running`, in `~/Projects/tf-wt/dflash2`.

The step numbering below is the plan's own. §0 maps it to what actually landed; the two differ in
places (the memory fit moved into step 1's measurement, and the original step 0 is blocked), so read
§0 for status and this section for intent.

**Step 0 — settle the tap row (½–1 day, decides the port).** **BLOCKED — see §0.1.**
Not "does the port work" — that is known. This is "which of flashnext's rows is the tap": the
read-out row, the down-projected row, or the normed variant (§4.1). The original route (a torch
target forward) is impossible on this box, so the experiment now runs through the engine's
`Hooks.taps_out` and `tools/dflash2/analysis/reference.py`. The definition that lands in the card's
acceptance-length band (2.8–3.3) wins. **If none reaches ~2.5, stop — the port is not worth starting.**

**Step 1 — prove the memory fit (1 h, needs the box).**
Start the real shape with a stub drafter allocation (weights only, no drafting):
`--context 262144 --parallel 4 --kv-cache fp8 --vision` + drafter resident. Read the startup
arithmetic line. Record the exact `Z GiB left`. No drafting yet.
Measured sizes for this (§4.3): ~0.85 GiB at `--drafter-bits 4`, so the fit is not in doubt — the
start is still the fact and the arithmetic the claim.

**Step 2 — tap ring in flashnext (the correctness work).** **HOOK DONE, persistence not.**
`Pass.tapDump` + `Hooks.taps/taps_out` exist and are inert when unset. Still missing: a caller that
allocates the ring, and carrying it through `Snap`/keep/snapshot/disk-tier with a kv8 precision
match. Gate: a recorded tap ring on a fixed prompt is byte-identical across (a) plain run,
(b) fp8 KV run, (c) snapshot-restore run. This is where the silent-bug risk lives.

**Step 3 — drafter forward on CUDA.**
Port `fc`+`hidden_norm`, 5 × (Qwen3 layer + 2 grouped convs), `norm`, then head+selector. The
qwen3_5 `cuda_draft*.zig` gives the layer shape for free; the new kernels are `grouped_conv`
(2-tap, group 16) and `candidate_selector` (lattice + greedy path). Gate: greedy acceptance on a
fixed prompt matches the PyTorch reference from step 0 token-for-token.
Kernel-by-kernel spec, including what is reusable unchanged from `qwen35_draft.cu`:
`flashnext-dflash2-kernels.md`.

**Step 4 — wire it up.**
`F.openDrafter` in `families/flashnext/cuda_native.zig`, `cost_adapter.identify(…, .dflash)`, window
timing, per-stream reserve. Gate: `serve --drafter <dir>` starts at 4×262k fp8, and a single-stream
greedy completion has the **same `token_sha256`** as the MTP build (lossless drafting — the card's
central claim, and our standard equivalence check).

**Step 5 — measure, honestly.**
`tools/bench_serving.py` for the fixed table plus `bench_concurrent` for lanes 1/2/4. Every cell
≥3 reps (floor), sampling at temp 1.0/top-p 0.95/top-k 20, greedy rows labelled greedy. Compare
against MTP on the *same build* via `--no-drafts` and the MTP path, never against a remembered number.
Report `not measured` where not run. The MTP bar to beat is in §0.2.

**Step 6 — decide.**
Ship as an option (`--drafter`) only if 4-lane aggregate wins ≥5% at equal context. Otherwise keep
MTP and write the finding up here; the fork is a serving repo, not a research repo.

### Rough effort
Step 0 ½ d · Step 1 1 h · Step 2 2–3 d · Step 3 4–6 d · Step 4 2 d · Step 5 1 d · Step 6 ½ d.
≈ **2–2.5 weeks** of focused work, with the real risk in steps 2 and 3, not in the maths.

### Cheaper alternative, if the goal is a number rather than a feature
The official route is SGLang + the capture patch on the FP8 checkpoint. That answers "is DFlash2
worth more than MTP *for our prompt mix on this hardware*" in about a day, on a box with headroom,
without touching the fork. It does **not** produce a servable Spark path (SGLang does not fit 4×262k
fp8 + vision on 128 GB unified the way this engine does). Recommendation: do step 0 first — it is
cheaper than the SGLang detour and it decides whether the port is even coherent.

## 7. Open questions

1. Does `--drafter-bits 4` on a bf16 checkpoint reproduce acceptance closely enough to matter? (The
   codebooks are 255 MB regardless; the backbone is only ~885 MB.)
2. Is block 8 actually better than block 4 here, or does the chain decay past position ~4 on our
   prompt classes? The card never publishes block 8.
3. Does the tap ring belong in the NVMe imprint tier, or is it cheap enough to recompute on restore?
   (Recompute needs a target forward — 11–14 s cold at 24.8k. That argues for storing it.)
4. Vision prompts: taps come from the text layers, but image prefill changes the residual stream. No
   evidence either way from the vendor; treat image prompts as `not measured` until run.

## 8. Sources

- `incoai/Qwen3.8-Flash-Next-DFlash2` config.json + `model.safetensors` header (read locally, revision
  `a22a1744e85c…`), size from the HF API.
- HF card evaluation tables: https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2
- Inco blog (2026-08-18): https://inco.ai/blog/dflash2/
- SGLang capture patch: `patches/qwen38-dflash2-capture.patch` in the same repo.
- Reference forward: https://github.com/sgl-project/SpecForge `specforge/modeling/draft/dflash2.py`,
  `draft/heads/candidate_selector.py`, `draft/dflash.py`.
- DFlash v1: https://github.com/z-lab/dflash (ICML 2026, Chen/Liang/Liu).
- Our own: `docs/drafter-release.md` (v1 format and its exclusions), `docs/design/` conventions,
  `references/decode-speed-audit.md` (per-round cost), `tensorfold-flashnext-ops` skill (bench protocol).
