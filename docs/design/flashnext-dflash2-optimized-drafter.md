# The optimized DFlash2 drafter (2026-11-19)

Status: **LIVE.** This is the drafter configuration that deployment and the coming real-traffic tests
use. It is three independent changes to the same 603M DFlash2 drafter, each behind its own switch,
each revertible on its own:

- **affine4 body** — the seven decoder linears of all five layers plus `fc` at 4-bit MLX affine
  (group 64, one signed bf16 scale and one bf16 bias a group). Selected by the checkpoint, not a flag.
- **split-KV attention** — `tf_fxd2_attn_split` + `_merge`, tensor-core, one CTA a (chunk, KV head).
- **block-FP8 head** — the target's `lm_head` ranked in FP8, its 64 survivors rescored against the
  exact bf16 rows, top-16 taken from the exact scores.

Drafter round, one session, `draft-check`, medians of 14: **16.31 ms → 9.22 ms (−43.5%)**. Served on
16 GSM8K prompts, 1024 tokens, one `:8099` server: **53.2 / 53.4 tok/s on two cold-start passes,
accept 3.287 / 3.307, round 65.4 / 65.6 ms, 16/16 identical outputs between passes** (2026-11-19,
`/tmp/baseline/rows_{1,2}.jsonl`).

Design history and the per-arm measurements in full are in the ledger — §9 (the 4-bit fit), §10 (the
4-bit body served), §11.1 (attention), §11.2 (the FP8 head offline), §12 (the five-arm session, the
sign bug, the served result). This file does not repeat those derivations; it is the operating
manual.

## 1. What each part is, and what it buys

All figures from one session, `draft-check --repeat 15` on `fixture/sequence`; the attention and head
columns are stages of the same reps (ledger §12 has this table with its plot).

| part | what changes | switch (default off) | stage | drafter round |
|---|---|---|---:|---:|
| — (datum) | fp8 W8A16 body, bf16 head | `--drafter checkpoints/dflash2-fp8` | — | 16.31 ms |
| body | affine4 on `fc` + 35 decoder linears; 458.75M weights, 229 MB of words + 28 MB of parameters | checkpoint contents (`affine4/manifest.json`) | — | 15.45 ms |
| attention | `tf_fxd2_attn_tile` → `tf_fxd2_attn_split` + merge, 40 CTAs at the served shape | `TF_FLASHNEXT_ATTN_SPLIT=1` | 3.78 → 1.00 ms | 12.60 ms |
| head | block-FP8 (128×128) rank to 64 ids + bf16 rescore of those 64 | `TF_FLASHNEXT_HEAD_FP8=1` + `TF_FLASHNEXT_HEAD_FP8_DIR` | 6.97 → 3.66 ms | 12.20 ms |
| all three | the deployment arm | all three of the above | 1.00 / 3.66 | **9.22 ms** |

The three compose additively in that session: 15.45 − 2.85 (attention) − 3.38 (head) = 9.22.

**What each is worth, stated honestly:**

- **body**: a byte win, not a speed win. 530 MB loaded against fp8's 780 MB and bf16's 1,206 MB
  (§10), −0.86 ms of the drafter round, and **nothing measurable served** (accept −0.15% CI
  [−1.00%, +0.86%], tok/s +0.19 ± 0.55). The affine4 decode is not byte-limited at this size
  (§10, last paragraph); ledger §13 names it the deployment body anyway, on bytes.
- **attention**: the largest single stage win, −2.78 ms of a 14 ms-class round. Served, unresolved:
  see §3.3.
- **head**: 6.97 → 3.66 ms, matches the roofline (1.27 GB → 636 MB of codes + 39.7 MB of scale tiles
  at the wall) and is lossless **because of the rescore**, not because of the fit (§11.2).

## 2. Turning it on

Served, exactly as measured (the two-pass baseline of §4 used this line):

    SNAP=~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300

    export TF_FLASHNEXT_ATTN_SPLIT=1
    export TF_FLASHNEXT_HEAD_FP8=1
    export TF_FLASHNEXT_HEAD_FP8_DIR=/tmp/headfit/full1/rtn

    zig-out/native/bin/tensorfold-native serve "$SNAP" \
      --host 127.0.0.1 --port 8099 --name qwen3.8-flash-next \
      --context 16384 --parallel 1 --kv-cache fp8 \
      --drafter checkpoints/dflash2-affine4

Offline, one drafter round over the fixture (no target boot, 7.6 GB RSS, ~4 s a rep):

    TF_FLASHNEXT_ATTN_SPLIT=1 TF_FLASHNEXT_HEAD_FP8=1 TF_FLASHNEXT_HEAD_FP8_DIR=/tmp/headfit/full1/rtn \
    TF_DRAFT2_TIME=1 \
      zig-out/native/bin/tensorfold-native draft-check \
        --drafter checkpoints/dflash2-affine4 --target "$SNAP" \
        --fixture /tmp/fx/sequence --out /tmp/arms/a4sh --repeat 15

`draft-check` also takes `--head-fp8 DIR`, which names the same artifact; the env flag is still
required (either one alone is refused).

Rules that bite, all read from the code:

| rule | where | consequence |
|---|---|---|
| **do not pass `--drafter-precision`** | `zig/src/server/cli.zig:112` accepts `bf16\|fp8` only, while `draft-check` also parses `affine4` | on `serve`, `--drafter-precision affine4` is a usage error; `--drafter-precision fp8` on the affine4 directory is a load refusal (`DrafterPrecisionMismatch`) |
| the body arm is named by `checkpoints/dflash2-affine4/affine4/manifest.json` plus `config.json`'s `"precision": "affine4"` | `cuda_draft2_fwd.zig`'s loader | a manifest beside fp8 `*_scale_inv` tensors is two quantizations in one checkpoint and is refused; a config that disagrees with the contents is refused |
| both env flags take **exactly `1`** | open path, once per drafter | `true`, `0`, empty are all "off", so a typo serves the shipped path rather than an experiment's |
| `TF_FLASHNEXT_HEAD_FP8` without `_DIR`, or `_DIR` without the flag | `headFp8Choice` | refused at open (`HeadFp8PathMissing` / `HeadFp8FlagUnset`) instead of silently substituting the bf16 head |
| one stream | every measurement above ran `--parallel 1`; multi-lane drafting is refused by design (ledger §3) | `--parallel 4` is not measured with this arm |

**Residency.** The affine4 body loads at ~530 MB; the quantised head adds 636 MB of codes + 39.7 MB
of scales, and **the 1.27 GiB detiled bf16 head stays resident** — the rescore reads it, so the flag
trades read bytes for residency, not for footprint. Nothing above has measured the load time or the
peak of the combined arm.

**Artifacts.** None of these blobs is committed (no git-lfs on this box):

| path | contents | committed? |
|---|---|---|
| `checkpoints/dflash2-affine4/affine4/` | 108 raw `words.u32` / `scale.bf16` / `bias.bf16` files + `manifest.json` | **no** — `.gitignore`d, regenerated by `tools/dflash2/quantize/export_affine4.py` from the fit's codes |
| `checkpoints/dflash2-affine4/config.json` | the source config with `dflash_config.precision = "affine4"` | yes |
| `checkpoints/dflash2-fp8/` | 745 MB blob + `manifest.json` + `SHA256SUMS` | manifest and sums only |
| `/tmp/headfit/full1/rtn/` | `head.fp8.safetensors` (635,854,600 B, sha256 `b20143a8…`) + `manifest.json` | **no** — `/tmp`, nothing pins it |

Rebuilding the head artifact (needs torch and the GPU; it collects real head inputs from the
fp8-body drafter):

    python3 tools/dflash2/quantize/fit_head.py collect --run /tmp/headfit/full1
    python3 tools/dflash2/quantize/fit_head.py fit     --run /tmp/headfit/full1
    python3 tools/dflash2/quantize/fit_head.py eval    --run /tmp/headfit/full1 --arms rtn

The search picked `clip0.0 mult1.0` for all 38,800 blocks — plain RTN — so `rtn/` *is* the fitted
arm; the separately fitted scales bought nothing (§11.2, item 2). `head.fp8.safetensors`'s sha256
and byte count are in its `manifest.json`.

## 3. The numbers, with their provenance

### 3.1 Drafter round — one session, five arms

`draft-check --repeat 15`, `fixture/sequence` (2048 rows, 44 positions advancing 1900 → 2047 by 3–4,
so rep 0 is cold and reps 1+ are the served regime), cold rep excluded, **medians of 14**. Session
2026-11-19 00:43–00:44; captures `/tmp/arms/{fp8,a4,a4s,a4h,a4sh}/rep{0..14}`, plot
`/tmp/prof/arms_all.png`.

| arm | dir | round | attention | head |
|---|---|---:|---:|---:|
| fp8 body (datum) | `fp8` | 16.31 | 3.76 | 6.97 |
| + affine4 body | `a4` | 15.45 | 3.78 | 6.97 |
| + split-KV attention | `a4s` | 12.60 | 1.00 | 6.97 |
| + block-FP8 head | `a4h` | 12.20 | 3.78 | 3.66 |
| + both | `a4sh` | **9.22** | 1.00 | 3.66 |

Absolute milliseconds move between sessions with clock/power state (the plan §4 makes the same
caveat); the ratios are the claim. The attention collateral slowdown seen on the fp8 body in §11.1
(attention −2.60, every other stage +6…+26%) does **not** appear on affine4 in this session.

### 3.2 Served — 16 GSM8K prompts, one `:8099` server per arm

Conditions for every row: `/tmp/gsm8k_16.json` (16 prompts), `--max-tokens 1024`, temperature 1.0 /
top-p 0.95 / top-k 20, pinned seeds 1000..1015, `--parallel 1 --context 16384 --kv-cache fp8`,
numbers from each server's own `done chatcmpl` line (`harness/accept_ab.py`). tok/s is the mean of
the 16 per-prompt rates, accept is the mean of `tokens/rounds`, CI is the paired per-prompt 95%
interval (the harness's t₁₅ unless noted).

| question | arms | tok/s | accept | paired Δ (relative) | outputs |
|---|---|---:|---:|---|---|
| does 4-bit cost acceptance? | fp8 → affine4 | 48.48 → 48.66 | 3.3205 → 3.3154 | accept **−0.15%** [−1.00, +0.86]; tok/s +0.19 ± 0.55 | 16/16 identical (`/tmp/ab`, 2026-11-18 23:01–23:04) |
| does split attention pay, on affine4? | affine4 → affine4+split | 46.86 → 49.24 | 3.2238 → 3.3041 | accept **+2.49%** [+0.22, +4.77]; tok/s **+2.38 (+5.08%)** t-CI [+1.76, +8.41] | 16/16 identical (`/tmp/abattn`, 2026-11-19 00:26–00:29) |
| the same question against fp8 | fp8 → affine4+split | 49.74 → 49.39 | 3.3413 → 3.3229 | accept −0.55% [−2.57, +1.47]; tok/s **−0.69%** [−2.32, +0.94] | 16/16 identical (`/tmp/aball`, 00:54–01:02) |
| the whole arm | fp8 → combined (fixed) | 48.21 → 51.73 | 3.2807 → 3.3053 | tok/s **+3.53 [+2.14, +4.91] = +7.25% [+4.31, +10.19]** (z; t₁₅ gives [+2.02, +5.03]); accept −0.74% [−3.58, +2.10] | 16/16 identical (`/tmp/abfix`, 06:45–06:49) |

The `abfix` pair is the one the ledger §12 records, taken after the head's sign fix. The 22.8 tok/s
run in `aball` (`rows_best.jsonl`, accept 1.050) is the same arm **before** that fix and is kept only
as the failure record.

### 3.3 Provenance notes a cold reader needs

1. **Only same-session pairs mean anything.** The unmodified fp8 arm read 48.48, 48.21 and 49.74
   tok/s in three sessions on the same day; the combined arm read 51.73 in `abfix` and 53.24 / 53.41
   in the two baseline passes ten minutes later. That is ~1.5 tok/s of session state, the same size as
   the split-attention effect. Quote deltas, never two absolute numbers from different runs.
2. **The split-attention effect is therefore unresolved.** `abattn` says +2.38 tok/s, `aball` says
   −0.35 on the same nominal comparison, and the ledger §12's second figure (0.56% CI [−1.48, 2.59])
   does not reproduce from any rows file still on disk (`/tmp/aball` gives −0.69% CI [−2.32, +0.94]).
   Its stage win is not in doubt (−2.78 ms, measured twice). Treat the served effect as **0 to +5%**
   and re-measure it paired in one session before quoting a number.
3. **The harness does not record the arm's switches.** `rows_*.jsonl` carries `arm`, a free-text
   label; `stages.json` writes `"command": ""`. Which env flags a given directory was produced with
   is known from the run order and the log, not from the artifacts. Fixing this (stamping the env in
   the row file) is the cheapest way to stop the next provenance question.
4. `fixture/sequence` is generated, not captured: `tools/dflash2/analysis/align_bf16/gen_inputs.py`
   (seed 20261116, `CASES["sequence"]`), and the `SHA256SUMS` beside it proves the bytes. It
   reproduces the served *geometry*, not real text; nothing about acceptance can be measured on it.

## 4. The served round, and where the time is now

From the two-pass baseline (`/tmp/baseline`, 2026-11-19 06:56 and 06:59, each a cold start):

| pass | tok/s | accept | round | outputs |
|---|---:|---:|---:|---|
| 1 | 53.24 | 3.2873 | 65.4 ms | 16/16 sha-equal to pass 2 |
| 2 | 53.41 | 3.3071 | 65.6 ms | |

Pass-to-pass **+0.175 tok/s, sd 0.833**: re-running this configuration inside one session repeats to
well under a tok/s. That is repeatability, not a licence to compare across sessions — §3.3 item 1.

Of that 65.5 ms round the drafter is **9.22 ms (14%)**; the other 56.3 ms is the target's verify pass
and everything else. So a *free* drafter would read 3.29 tokens / 56.3 ms ≈ **58.4–58.7 tok/s** — the
ceiling this work can reach by drafter optimisation alone. The remaining lever is acceptance
(≈0.55 accepted per draft here (Σaccepted/Σdrafted) against MTP's 0.630 on real traffic, ledger §2.1), i.e. the corpus and
fine-tune path, not another kernel.

Context for that comparison: MTP single observations are 52.5 and 59.6 tok/s (ledger §2 production
row, §3), and on
the real-traffic pair MTP read 39.4 against DFlash2-fp8 30.4 tok/s (§2.1). Those are different
corpora and sessions from the 53.2 above; **no MTP arm has been run in the same session as the
combined arm**, so "the optimized drafter beats MTP" is not a measured claim.

## 5. Losslessness: what is claimed, and where the claim stops

| claim | evidence | limit |
|---|---|---|
| The served output is bit-identical to the fp8 arm's | 16/16 completion `sha` equal in `/tmp/ab`, `/tmp/abattn`, `/tmp/aball`, `/tmp/abfix`; 16/16 between the two baseline passes | Verification makes any drafter lossless in *text*; this says the arm did not move the target's tokens on 16 prompts, not that it cannot |
| The fp8 head + rescore is exact | recall@16 1.0000 and set-exact 1.0000 on 6,160 real head-input rows (held-out 1.0000), accepted-slot vector bit-identical to the bf16 head on 880 teacher-forced positions | Teacher-forced, one corpus; the served accept delta is null, not proven zero (−0.74% CI [−3.58, +2.10] at n = 16) |
| The affine4 body costs no acceptance | paired −0.15% CI [−1.00, +0.86] served; 880-position teacher-forced probe within ±0.04 slots (§9) | The `n = 16` served interval still admits a ~1% loss |
| The split attention is **not** bit-identical | by construction: chunked softmax reassociates the sum, tensor-core fp32 accumulates replace the per-lane `fmaf` chain | Its gate is per-slot acceptance at a stated width, never proposal identity (ledger §8) |

The rule this section exists to enforce: proposal identity gates a *refactor*, acceptance length at a
stated width gates a *precision change* (ledger §8). Neither of the two kernels here is a refactor.

## 6. Reverting, one part at a time

| to go back to | do this | cost |
|---|---|---|
| fp8 body | `--drafter checkpoints/dflash2-fp8` (keep the env flags) | +0.9 ms round |
| bf16 body as published | `--drafter <incoai snapshot>`, no `--drafter-precision` needed | +3.8 ms round vs affine4 (19.16 vs 15.32, §10) |
| shipped attention | unset `TF_FLASHNEXT_ATTN_SPLIT`; the module is never loaded and no buffer of the split exists | +2.9 ms round (+2.8 attention) |
| bf16 head | unset **both** `TF_FLASHNEXT_HEAD_FP8` and `TF_FLASHNEXT_HEAD_FP8_DIR` (one alone is refused) | +3.4 ms round (+3.3 head), −676 MB residency |
| everything | no `--drafter` at all: the same binary serves MTP | — |

There is no partial-revert state that is broken: each switch is read once at open, and unsetting it
leaves no branch, buffer or launch of that path behind.

## 7. Failure modes already hit

**7.1 The head's `key_value()` sign bug — accept 1.05, 22.8 tok/s, and every offline check green.**
`key_value()` in `kernels/cuda/draft2_head_fp8.cu` returned `ordered(v)`, the order-preserving u32,
instead of inverting it. For a positive logit `ordered(v)` is the same bits with the sign bit set —
exactly `-v` — so the id list handed to the selector was right and every score was negated: the
lattice walk picked the worst candidate at every slot. Served: accept 1.050, 22.8 tok/s, and a 47.6 ms
round — faster, as one token a round costs the verify pass less. Fixed by inverting:
`o = 0xFFFFFFFF - key>>32; __uint_as_float((o & 0x80000000) ? (o & 0x7FFFFFFF) : ~o)`.

Why the offline checks missed it, and the rule that follows:

- `draft-check`'s comparisons were on `logits_top16_ids`, an **ordered id list**, which was exactly
  correct. An id list in rank order survives any monotone corruption of the values beside it, so no
  id-set check could see this.
- The proposals *did* move in 15/15 reps, but a proposal change is expected of a precision arm
  (ledger §8), so it was filed as numerical chaos rather than as a bug.
- **Dump the value table the selector consumes** — `TF_FLASHNEXT_HEAD_DUMP=<dir>` writes what the
  head hands the selector, `ids` and `val`, and what it rescores. Compare `val`, not only `ids`.
- After the fix, `draft-check` proposals are identical to the attention-only arm in 15/15 reps, and
  the served arm is the +7.25% row of §3.2.

**7.2 Flag semantics that would silently serve the wrong arm.** Both flags are exact-`1`, empty
counts as unset, and the head's two variables are validated as a pair at open. This is deliberate:
the one thing a switch must not do is substitute the bf16 head for the fp8 one while the caller
believes otherwise. Do not "clean up" these readers into presence checks.

**7.3 The 4-bit loader's exact-size guard.** Zig 0.17's `.limited(len)` refuses a file of exactly
`len` bytes (`StreamTooLong`), so the arm's per-tensor guard caps at `len + 1` and compares
afterwards; and the artifact's file names must be **checkpoint** tensor names
(`layers.0.self_attn.q_proj.weight`), not the fitter's compact keys (`L0_q`), because the loader asks
the manifest for the checkpoint's own name (§10). Both cost an afternoon once.

**7.4 The attention collateral slowdown, still unexplained.** On the fp8 body, turning the split on
made attention 2.60 ms cheaper and every other stage 6–26% dearer, reproduced in both measurement
orders to the hundredth of a millisecond; it did not appear on affine4. Prime suspect is
power/clock response after tensor-core bursts (the loss falls hardest on the streaming stages). Do
not attribute a delivered-round gain to the attention stage's own delta; read `TF_DRAFT2_TIME`'s
medians for the whole round.

## 8. What is still unmeasured

1. **Per-slot acceptance of the optimized drafter.** The served log reports totals only. This is the
   same gap as ledger §7 item 4 and now gates the one lever left (acceptance, not cost).
2. **Real traffic.** Every served number here is GSM8K, which §2.1 shows overstates production
   per-draft acceptance by ~2×. The 16 agentic prompts (`/tmp/agentic_16.json`) have not been re-run
   against this arm.
3. **Production serving config** (`--context 262144 --parallel 4 --kv-cache fp8 --vision --thinking`).
   Nothing here was measured at 262k context or with more than one stream; the head's 676 MB and the
   bf16 head it rescores against are both resident and both scale with nothing, but that is an
   argument, not a measurement.
4. **MTP in the same session.** No paired MTP-vs-combined run exists.
5. **The split-attention served effect** (§3.3 item 2): 0–5%, needs one paired session.
6. **Load time and peak residency of the combined arm** — the head artifact's load is a new 636 MB
   read at open; nothing has timed it.
7. **Durability of the head artifact.** It lives in `/tmp/headfit/full1/rtn`, pinned by nothing
   committed; the affine4 body's manifest is gitignored too. Before deployment, either re-run
   `fit_head.py fit` into a versioned path and record its sha256, or commit a manifest.
8. **Whether the head is worth its residency** at production context, where 676 MB competes with KV
   pages; unmeasured.
