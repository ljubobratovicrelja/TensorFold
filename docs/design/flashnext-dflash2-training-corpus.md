# Training corpus for the DFlash2 drafter (2026-11-18)

What a fine-tune of the drafter needs, what this box can actually produce, and the two
measurements that changed the plan. Companion to `flashnext-dflash2-harvest.md` (the harvester and
its rates) and `flashnext-dflash2-fp8-checkpoint.md` (the arm being trained).

## The corpus is not text: it is assistant-role text

A drafter only ever drafts what the target emits next, and in serving that is the **assistant**
region. The taps are the target's own residuals, so no corpus is usable until it has been run
through the target — but that pass is prefill, not generation (45 tok/s decode is irrelevant).

Measured, same drafter, same harness, `draft-check` on harvested fixtures:

| fixture content | accept length | mean advance |
| --- | --- | --- |
| one long user message of markdown docs (chunk-0000) | 1.625 | 1.625 |
| ditto (chunk-0002) | 1.000 | 1.000 |
| served chat (four-arm e2e, bf16 arm) | 3.387 | — |

A drafter that scores 3.3 in service scores 1.6 on prose it never drafts in production. So a
fixture set built from documents is worse than useless: it has no headroom to show a fine-tune
helping, and `chunk-0002` sits at zero acceptance.

**Consequence:** harvested chunks must be whole conversations — real assistant spans, the pass
ending mid-assistant-turn — so the 8-row blocks sit where they sit in use.

## The harvester writes the wrong region today

`tools/dflash2/harvest.py` sends each chunk as `messages=[{role:user, ...}]`. Checked in
`ids.bin`: `im_start` occurs **once** per 2,000-token chunk, and the only assistant-region tokens
are the five-token priming tail (`<|im_start|>assistant\n` + two markers). ~99% of harvested tokens
teach the drafter to draft user text. Fix before the first training run.

## Rate and the disk wall

Cold prefill, GB10, fp8 KV, drafts off: **2,039 tok/s** at 2,048-token chunks, 2,825 at 4k, 3,149
at 8k (plateau). A ring is 52,838,400 B whatever the prompt length → **27.1 KB/token**, so one
night is ~73M tokens ≈ **2 TB** against 60 GB free. Harvest must stream: harvest a chunk, train on
it, delete it. A 2,048-token chunk is the operational size because the ring is `window + block` =
2,064 slots and a longer pass wraps.

Prompt-cache trap: a repeated prompt reports `prefill=0.11 s` and 18–34k "tok/s". That is a cache
hit; `manifest.pos_end` is the check.

## What the local agent history holds

`~/.hermes/state.db` (not `sessions.db`, which is vestigial): 131,315 messages, 2,448 sessions,
2026-06-01 onward, roles structured in separate columns. Verified against the live DB and then
snapshotted, because it compacts live:

| role / field | rows | tokens |
| --- | --- | --- |
| assistant `content` | 15,020 | ~3.8M (87.6% prose, 10.6% fenced code) |
| assistant `tool_calls` | 54,103 | ~18–22M |
| assistant `reasoning` | 55,782 | ~23M |
| user `content` | 7,090 | ~6.9M |
| tool output | 62,951 | ~61M (extrapolated, 400-row sample) |
| system prompts | 1,481 | ~10.6M (extrapolated) |

Two facts decide how it can be used:

1. **The assistant text is not our model's.** It is overwhelmingly `deepseek/deepseek-v4.1-flash`
   output — whatever the agent was running, not Qwen3.8-Flash-Next. Training on it teaches the
   drafter another model's continuations.
2. **The tool-call rows are the valuable part.** 54k rows of assistant-role JSON, in the real
   distribution, is exactly the shape that was missing; the ~61M tokens of tool output are
   *context*, not labels, which is what makes these sessions realistic.

So the usable split is: real **contexts** (user + tool output + system, ~78M tokens) are free, and
faithful **continuations** have to be produced by the target at 45 tok/s ≈ 1M tokens a night.
Regenerate selectively — the long, tool-dense sessions where DFlash collapses (accept 2.83–2.85 at
450–765 tokens while MTP holds 0.59–0.64) — and keep a slice of real traffic held out for eval.

## The torch forward is gated

`tools/dflash2/train/forward.py` reproduces the engine's per-slot acceptance **exactly** on two
fixtures, with 216/224 proposal slots byte-identical (96.4%). The residual is bf16 accumulation
order: 2.6e-3 at `conv_attn_in_0` compounding to ~4e-2 by `block_hidden_4`, so only 92–95/112 of
the head's top-16 *sets* agree. Acceptance is the gate that survives that, and it matches.

Non-obvious details it had to encode: block row `m` sees prefix `[lo+m, P)` (the oldest `m`
context columns drop); context keys are RoPE'd at absolute positions; the head reads block rows
1..7 only; both dynamic convs sit inside the residual with a bf16 rounding per product; the
selector's `hp` is not rounded and its argmax is a strict `>` (first index wins ties).

## Plan

1. Fix the harvester to emit conversations, not one user message.
2. Build the trainer on the gated forward: full fine-tune of the 5-layer backbone (426M), freezing
   `fc`, head and selector; low LR; bf16.
3. Stream shards (~300k tokens, ~8 GB) with checkpoint + resume, and evaluate each checkpoint with
   `draft-check` on held-out conversation-shaped fixtures. The log is acceptance vs tokens seen,
   not loss.
4. Re-quantize to fp8 and re-run the paired A/B; fp8's measured cost to stay inside is −1.94%
   accept for +1.31 tok/s.

Sizing: 10–30M tokens is the sensible band for an in-domain fine-tune of a 426M backbone; below
~5M it fits corpus style, and the published from-scratch recipes (~400M) are a different game. A
night is ~20M tokens harvested and a few epochs over them.

**Why not RL.** Verification is lossless (16/16 identical completions), so the committed prefix is
the target's own greedy continuation whatever the drafter proposes: the drafter never steers its
own training state. The reward is therefore dense, deterministic and computable offline by one
teacher-forced pass, which makes this regenerated-corpus imitation (what EAGLE does), not policy
gradient. The value of real traffic is the *context* distribution, not the RL machinery.
