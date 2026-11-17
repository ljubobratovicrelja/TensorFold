# DFlash2 vs the vendor baseline — the honest scoreboard

Status: **open.** This file exists because two days of work concluded "DFlash2 is worse than
MTP, stop" on a comparison that was not like-for-like. The corrected comparison is below, with
the surviving deficit isolated to one number. Read this before quoting any DFlash2 figure.

## 1. The bar, from the vendor card (verified at source)

`https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2` — raw README, fetched 2026-11-17.
This card is for **our** target (`Qwen/Qwen3.8-Flash-Next`), not the z-lab 27B drafter.

Evaluation setup as stated on the card:

- Target `Qwen/Qwen3.8-Flash-Next-FP8`; SGLang on **one GB300 (TP1)**; `trtllm_mha` draft attention.
- **Speculation block size 4 — "limited by the current QSA implementation."** This is a *serving*
  limitation of the vendor's stack, not a trained property. Our checkpoint's `config.json`
  declares `block_size: 8`, and we run 8. **The vendor has never published a width-8 Flash-Next
  number.**
- T=1.0 / top-p 0.95 / top-k 20, `xhigh`; 4096 max new tokens incl. reasoning; 128 samples, conc 1.
- Both speculative methods propose **three** draft tokens per verification step.

Concurrency 1, GSM8K:

| | AR | MTP | DFlash 2 |
|---|---|---|---|
| output tok/s | 205.4 | 326.0 (1.59×) | 368.2 (1.79×) |
| acceptance length | — | 3.08 | 3.26 |
| implied per-draft | — | 69.3% | **75.3%** |

## 2. What we measure, same harness, same day

16 z-lab/dflash GSM8K prompts, sampled, seeds pinned, single stream, box otherwise idle,
numbers read from the server's own `done chatcmpl` line. Harness `~/.hermes/cache/scratch/width_ab.py`.

| arm | tok/s | vs AR | accept length | drafts/round | per-draft |
|---|---|---|---|---|---|
| AR (no drafts) | 25.5 | 1.00× | — | — | — |
| DFlash2, width 8 | 43.8 | **1.72×** | 3.33 | 4.49 | 49.1% |
| MTP | 59.6 | **2.34×** | 3.89 | 3.59 | 69.7% |

## 3. Three corrections to what this repo believed

1. **"DFlash2 is broken" is not supported.** 1.72× vs the card's 1.79×, accept length 3.33 vs
   3.26. We are at the card, and we are doing it at width 8 where the vendor could not.
2. **"MTP is exceptional here" is false.** Our MTP is 69.7% per draft; the card's MTP is 69.3%.
   Identical. Our MTP *looks* better only because the Spark's target is 8× slower per token
   (25.5 vs 205.4 tok/s), so each accepted token is worth far more here — which is also why our
   MTP shows 2.34× where the card's shows 1.59×.
3. **"49% vs 71%" was an artifact.** Our depth rule offers 4.49 drafts per round; the card offers
   exactly 3. The decaying tail slots (0.389/0.429/0.333) drag our aggregate down while our
   accept *length* still exceeds the card's. Comparing those two percentages was never valid.

## 4. The deficit that actually survives

Truncated to the first three slots — the only like-for-like width against the card:

- ours: 0.740 / 0.662 / 0.592 → **accept length 2.99**, 66.5% per draft
- card: **3.26**, 75.3% per draft

**Gap: 0.27 accept length, ~9 points per draft.** That is the whole of it. It is also exactly
the sign flip against MTP: the card's DFlash2 beats MTP by +6 points, ours loses by −3.

The vendor's own ablation (`https://inco.ai/blog/dflash2/`, Qwen3.5-4B, T=0) prices the components:

- DFlash → +path selection: 4.27 → 4.61 (**+0.34**)
- +convolution: → 5.97
- selector alone at T=1: **+0.47**

**+0.34 to +0.47 against our 0.27 gap.** The selector is the prime suspect by magnitude, and it
is the one component whose *correct implementation is not the same as a correct contribution*.

> **Correction — the selector is NOT the cause. Measured, both sides.** A/B with the selector
> replaced by per-slot argmax, same 16 prompts, same harness:
>
> - our engine: 45.0 tok/s / accept 3.34 / 49.3% with it → 44.0 / 3.29 / 49.2% without. **+0.05.**
> - the torch reference on the same tap dump: 2.810 advance / 1.810 accepted → 2.750 / 1.750
>   without. **+0.06.**
>
> The port matches the reference to 0.01 accept length. The vendor's +0.34/+0.47 selector ablation
> was measured on **Qwen3.5-4B**, where the head is less confident and re-ranking has room to
> work. On Flash-Next the head's top-1 is already the selector's pick, so the component is worth
> almost nothing on this target — by design, not by bug. **Do not re-run this.**

## 5. What is closed by evidence — do not reopen

- **Selector is in the loop.** `tf_fxd2_select` walks the lattice from the anchor
  (`cuda_draft2_fwd_test.zig` pins the anchor-predecessor rule and the argmax tie rule). The
  "drafts as DFlash1, silently" failure mode does not apply to this port.
- **Block mask matches the reference.** `qwen35_draft.cu` implements `is_causal=false` with the
  window trimming oldest context columns from later rows, and its comment records it was
  measured against the reference's own mask (row 0 sees all ctx, row 7 sees ctx−7).
- **Proposal shift, draft-KV source, GDN rollback.** All three would corrupt greedy output.
  Greedy is 643/643 bit-exact on the count prompt. A wrong shift or a poisoned prefix KV cannot
  survive that.
- HC tap basis (mean4 vs readout): null, 48/103 vs 49/103. SGLang #45142 independently says the
  incoai Flash-Next drafter was trained on the **stream mean**, which is what we do.
- Block width: refuted as a lever. Width 4 buys no per-slot quality and nets ~3% slower.
- Engine-vs-reference ceiling: 1.317 vs 1.342 tok/round.

## 6. Ranked next tests

1. **Selector contribution — DONE, null.** A/B in both engine and reference: +0.05 / +0.06 accept
   length. The selector is present, correct, and worth nothing on this target. Closed.
2. **Conv contribution A/B.** Zero the dynamic projection — output must equal identity (base tap
   0 = 1.0). The card says the two-tap convs are what "keep the draft from decaying toward the end
   of the block", and our decay through slots 5–7 (0.389/0.429/0.333) is the largest unexplained
   structure left. This is the next test, and it is the same A/B shape as the selector one.
3. **Selector shortlist source.** Confirm the top-16 shortlist comes from the target `lm_head`
   applied to draft hidden with `output_multiplier` applied, and that `rank=256` / `top_k=16`
   are read from the checkpoint config rather than assumed. (Lower value now that the selector is
   known to be inert.)
4. **Per-request vs pooled accounting.** SGLang notes per-request mean and pooled ratio differ
   ~10% on the 27B. Confirm our accept length is the per-request mean the card uses.

## 7. Standing warning

The vendor's width-4 cap is a **QSA limitation in their stack**, not a property of the model.
Anyone who "fixes" our acceptance by rebuilding for width 4 is optimising the vendor's
constraint, not ours — we already run the trained width and already match their number.
