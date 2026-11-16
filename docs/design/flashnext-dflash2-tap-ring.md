# DFlash2 tap ring on flashnext: where it must be written, and what breaks if it is not

Companion to `flashnext-dflash2-drafter.md`. That plan says *what* to build; this memo records the
one part that fails silently, so a later change has something to check itself against.

Code: `zig/src/families/flashnext/cuda_tap_ring.zig` (ring arithmetic, `TapSource`, `Source`).
Config and refusals: `zig/src/families/flashnext/cuda_draft2.zig`.

## What the ring is

Five of the target's layer outputs, one row a layer, per stream, in a ring of `window + block + 8`
rows of `5 x 2560` bf16. Slot for position `p` is `p % rows`; tap `k` of a position sits
`k * hidden` into that slot's row.

Measured cost: 2064 rows × 5 × 2560 × 2 = 50.4 MiB a stream, **0.20 GiB at four streams**. The ring
is sized by the drafter's sliding window, not by the served context, so 262k and 32k cost the same.
That is the whole reason it is a ring and not a per-position buffer — but it also means the ring
**wraps**, and a wrap is where the silent bug lives.

## Where the tap is written

flashnext's batched pass leaves, per layer, both a mixed/down-projected read-out row and its normed
form (`cuda_batch_hc.zig`: `writebackNorm`, then `readoutNative`; `cuda_batch.zig`: `readout`,
`readoutKeep` leave normed rows at `s.norm + r0 * wide * 2`). The tap write belongs immediately after
the read-out for a tapped layer, over the same `[r0, r0 + n)` span, into `ring.slot(r0 + i)`.

Three candidate sources exist and are named in `TapSource` (`readout`, `normed`, `mean4`). Which one
is correct is **not settled** and is the subject of the plan's step 0. The default is a single
constant; nothing else in the engine should branch on the choice.

## The failure mode this memo exists for

A tap ring that is **not** carried by a state is not an error. The drafter reads whatever is in the
ring, cross-attends to it, and proposes tokens. They are wrong, and acceptance length drops — but
the output is still the target's (verification is lossless), so **the served tokens stay correct and
nothing fails**. The only symptom is a speed number nobody can explain. That is why every one of
these must carry the ring:

| path | what must happen | what breaks without it |
| --- | --- | --- |
| a keep / state snapshot | copy the ring's live span with the DeltaNet state | restored stream drafts from stale rows |
| `--prompt-cache-disk` (imprint tier) | write the ring into the `.st` file, restore it | warm prompts draft from a ring the restore never filled |
| kv8 vs bf16 | ring stays bf16 in both; assert the taps are bit-identical | a precision mismatch changes tap values silently |
| a wrap | refuse or re-fill when `!ring.covers(from, to)` | oldest taps overwritten before the drafter reads them |

`Ring.covers` exists for the last row: it is a cheap assertion that a restored ring actually spans
the positions the next round will read. Use it at restore, not at draft time.

## DeltaNet interaction

flashnext's linear-attention layers keep recurrences and conv tails; the tap ring is orthogonal to
them — it stores a *layer output*, not attention state — so nothing in `cuda_batch_gdn_lanes.zig` or
`cuda_batch_side.zig` changes. The one place to look is the **shared round**: a shared batched pass
advances several lanes at once, so the tap write must land per lane at that lane's own `r0`, not at
the round's first row. A shared round that writes every lane's taps at the same slot is the most
likely real bug in this area.

## The equivalence test that would catch it

Byte identity on a fixed prompt, three ways, comparing the recorded tap ring:

1. plain bf16 run
2. `--kv-cache fp8` run
3. snapshot, restore, then the same prompt

All three must produce the same ring bytes. Then the end-to-end gate: a single-stream greedy
completion must keep the same `token_sha256` as the MTP build, because DFlash2's verification is
lossless — if the tokens change, the integration is wrong somewhere, and if the tokens match but
throughput did not move, the taps are wrong. Those two symptoms together identify which of the two
failure classes you have.
