# Flash Next + DFlash 2 drafter: feasibility and integration plan

Status: **plan, nothing implemented.** Target: `Qwen3.8-Flash-Next-FP4-Local` (family `flashnext`) drafting
with `incoai/Qwen3.8-Flash-Next-DFlash2` instead of — or beside — its own MTP head.

Written 2026-11-16 against `running` @ `bb95ab741`. Checkpoint facts below were read from the downloaded
file, not from the model card; the card's numbers are quoted as the vendor's and labelled as such.

---

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

### 4.1 Tap semantics on a 4-stream family — **RESOLVED against the weights, no experiment needed**

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

**Step 0 — settle the tap row (½–1 day, decides the port).**
Not "does the port work" — that is known. This is "which of flashnext's rows is the tap": the
read-out row, the down-projected row, or the normed variant (§4.1). Build a small offline harness
(not the server): run the target's forward on one fixed prompt capturing all three candidates at
layers 5/15/25/35/45, feed each through the real checkpoint's drafter via the SpecForge reference
port (runnable with `transformers`), and compare acceptance length on ~20 prompts. The definition
that lands in the card's acceptance-length band (2.8–3.3) wins. **If none reaches ~2.5, stop — the
port is not worth starting.**

**Step 1 — prove the memory fit (1 h, needs the box).**
Start the real shape with a stub drafter allocation (weights only, no drafting):
`--context 262144 --parallel 4 --kv-cache fp8 --vision` + drafter resident. Read the startup
arithmetic line. Record the exact `Z GiB left`. No drafting yet.

**Step 2 — tap ring in flashnext (the correctness work).**
`setTaps`-equivalent in `families/flashnext`, tap written from `cuda_batch_hc.zig`'s read-out,
ring sized into `streamBytes`, carried through `Snap`/keep/snapshot/disk-tier with a kv8 precision
match. Gate: a recorded tap ring on a fixed prompt is byte-identical across (a) plain run,
(b) fp8 KV run, (c) snapshot-restore run. This is where the silent-bug risk lives.

**Step 3 — drafter forward on CUDA.**
Port `fc`+`hidden_norm`, 5 × (Qwen3 layer + 2 grouped convs), `norm`, then head+selector. The
qwen3_5 `cuda_draft*.zig` gives the layer shape for free; the new kernels are `grouped_conv`
(2-tap, group 16) and `candidate_selector` (lattice + greedy path). Gate: greedy acceptance on a
fixed prompt matches the PyTorch reference from step 0 token-for-token.

**Step 4 — wire it up.**
`F.openDrafter` in `families/flashnext/cuda_native.zig`, `cost_adapter.identify(…, .dflash)`, window
timing, per-stream reserve. Gate: `serve --drafter <dir>` starts at 4×262k fp8, and a single-stream
greedy completion has the **same `token_sha256`** as the MTP build (lossless drafting — the card's
central claim, and our standard equivalence check).

**Step 5 — measure, honestly.**
`tools/bench_serving.py` for the fixed table plus `bench_concurrent` for lanes 1/2/4. Every cell
≥3 reps (floor), sampling at temp 1.0/top-p 0.95/top-k 20, greedy rows labelled greedy. Compare
against MTP on the *same build* via `--no-drafts` and the MTP path, never against a remembered number.
Report `not measured` where not run.

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
