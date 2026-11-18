# Harvesting drafter training fixtures (2026-11-18)

The drafter's input is the target's own residual taps, so no text corpus is usable until it has
been run through the target. `tools/dflash2/harness/harvest.py` does that: one prefill pass per chunk, and
the engine's existing `TF_FLASHNEXT_TAP_DUMP` capture becomes the fixture.

This is **prefill only** — no sampling, no decode, no rollout — so the target's 45 tok/s decode
rate is irrelevant to it. That is the whole reason a night of corpus work is affordable at all.

## The command

```
TF_FLASHNEXT_TAP_DUMP=/tmp/harv/dump TF_FLASHNEXT_TAP_SOURCE=mean4 \
  ./zig-out/native/bin/tensorfold-native serve "$TARGET" --port 8101 --context 16384 \
    --parallel 1 --kv-cache fp8 --no-drafts --no-thinking &

python3 tools/dflash2/harness/harvest.py --dump-dir /tmp/harv/dump --out /tmp/harv/out \
  --tokenizer-dir "$TARGET" --chunk-tokens 2000 --corpus FILE...
```

Each `chunk-NNNN/` is the format `draft-check` already consumes (`manifest.json`, `ids.bin`,
`ring.bin`), so a harvested directory is training data, not a new format.

## Measured on this box (GB10, fp8 KV, drafts off, cold passes)

| prompt tokens | engine prefill | tok/s |
| --- | --- | --- |
| 1,012 | 0.578 s | 1,750 |
| 2,012 | 0.976 s | 2,062 |
| 4,012 | 1.420 s | 2,825 |
| 8,012 | 2.545 s | 3,149 |
| 16,012 | 5.152 s | 3,108 |

Cold prefill plateaus near **3.1k tok/s**; it is not the 10–20k that a compute estimate suggests,
and the expert path is nowhere near the box's ~230 GB/s (an 8k pass reads ~60 GB of expert weights
in 2.5 s = 24 GB/s). That gap is a kernel question, not a harvesting one.

**Prompt-cache trap:** an identical repeat of the same prompt reports `prefill=0.11 s` and
18–34k "tok/s". Those are cache hits. Every harvested chunk must be unique text, and
`manifest.pos_end` is the check that the pass really read it.

## Bytes and the disk wall

A ring is `2064 x 5 x 2560 x 2 = 52,838,400 B` whatever the prompt length, so 8 chunks wrote
404 MB for 15,253 tokens = **27.1 KB/token**. Consequences, all measured or arithmetic:

- 60 GB free on `/` = **2.2M tokens** stored. One night at 2.0k tok/s is ~73M tokens = ~2 TB.
- Therefore harvest must **stream**: harvest a chunk, train on it, delete it. Storage mode is
  only for small experiments.
- A 2,048-token chunk is the operational size: the ring is `window + block` = 2,064 slots, so a
  longer pass wraps and only its tail survives. `pos_end` runs 2 short of the id count because the
  tap is deferred by one layer.

## Not captured yet

`ids.bin` gives the corpus's own next tokens, which is the label a teacher-forced pass needs.
The target's *argmax* per row — the thing acceptance is actually scored against — needs
`hooks.logits == .all`, which this capture does not set. Until it does, the trainer learns corpus
continuations, not target continuations; for in-domain text those coincide closely, and the
served A/B is what decides whether that was good enough.
