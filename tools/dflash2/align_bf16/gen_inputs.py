#!/usr/bin/env python
"""Deterministic random inputs for the DFlash2 drafter alignment test.

One file, read by both sides: the reference torch run and our CUDA engine's fixture mode
(`cuda_draft2_check.zig`'s `TF_FLASHNEXT_DRAFT_FIXTURE`). Nothing here is random at run time —
the bytes are a fixed function of `SEED` and the geometry, so a re-run reproduces them exactly
(and `SHA256SUMS` beside them proves it).

Layout, exactly the fixture format the check already reads:

  ring.bin      [rows, n_taps * hidden] bf16. Slot `j` holds position `j`'s five tap rows.
  ids.bin       [pos_end] u32. Position p's token id.
  manifest.json rows / taps / hidden / window / block / pos_end / row_bytes.

A round then takes the context positions `[P - ctx_len, P)` with `P = pos_end - 1` and
`ctx_len = min(window, P)`: with one slot a position (`rows == pos_end`), position `j` is row `j`
and no row is ever overwritten, so a "ring" here is just the prefix itself.

Two geometries, both written under distinct directories:

  block8   pos_end 9     -> ctx_len 8,    the literal shape of the ask (one block of 8 rows of taps)
  window   pos_end 2048  -> ctx_len 2047, a full draft window, where the attention reduction is
                                          as long as a served round's ever gets

The taps are `N(0, 1)` in fp32, rounded once to bf16: the magnitude real hyper-connection rows
carry after the target's `hidden_states.unflatten(-1, (4, hidden)).mean(-2)` (measured std ~0.5
to 1.0 a row on a real capture; the exact scale is immaterial to a bit-for-bit comparison, it
just has to be a realistic one so no kernel's overflow path or a denormal tail is what is being
tested).

  python tools/dflash2/align_bf16/gen_inputs.py [outdir]
"""

import hashlib
import json
import os
import sys

import numpy as np

SEED = 20261116
TAPS = [5, 15, 25, 35, 45]
HIDDEN = 2560
BLOCK = 8
WINDOW = 2048
MASK_TOKEN = 248077
VOCAB = 248320

# ctx_len the round will read, per geometry: min(window, pos_end - 1)
CASES = {
    "block8": dict(pos_end=9),
    "window": dict(pos_end=2048),
    # the served regime: consecutive rounds on ONE state, each ending where its block begins and
    # advancing by a plausible accepted run, so the round after the first reuses the span the one
    # before projected instead of re-fusing the whole window
    "sequence": dict(pos_end=2048, sequence_start=1900),
}

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture")


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round fp32 to bf16, as the bits a bf16 store would hold (round-half-to-even)."""
    u = x.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    bias = 0x7FFF + lsb
    u = (u + bias) & 0xFFFF0000
    return u.view(np.float32)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists; refusing to overwrite a fixture someone may be comparing against")
    rng = np.random.default_rng(SEED)

    # one anchor drawn from the real vocabulary, and a length of plausible ids behind it: the ids
    # only ever reach the embedding table, so they have to be in range and nothing more.
    anchor = int(rng.integers(0, VOCAB))
    print(f"seed {SEED}  anchor {anchor}  mask {MASK_TOKEN}")

    for name, spec in CASES.items():
        pos_end = spec["pos_end"]
        d = os.path.join(OUT, name)
        os.makedirs(d)

        ids = rng.integers(0, VOCAB, size=pos_end, dtype=np.int64).astype(np.uint32)
        ids[pos_end - 1] = anchor                      # P = pos_end - 1: the anchor's own id
        # every position but the anchor's is context only; keep them distinct from the mask id so a
        # dump that confused "an id" with "the block" is visible
        ids[ids == MASK_TOKEN] = MASK_TOKEN - 1

        tap = bf16_round(rng.standard_normal((pos_end, len(TAPS) * HIDDEN), dtype=np.float32))
        slot = (tap.view(np.uint32) >> 16).astype(np.uint16)   # bf16 bits, exactly the f32's top half

        ring_path = os.path.join(d, "ring.bin")
        slot.tofile(ring_path)
        ids_path = os.path.join(d, "ids.bin")
        ids.tofile(ids_path)

        man = {
            "rows": pos_end,
            "taps": TAPS,
            "hidden": HIDDEN,
            "window": WINDOW,
            "block": BLOCK,
            "pos_end": pos_end,
            "row_bytes": len(TAPS) * HIDDEN * 2,
            "source": f"gen_inputs.py seed={SEED} case={name}",
        }
        if "sequence_start" in spec:
            # a decode's advance is the accepted run plus the anchor; 2..4 accepted is the shape our
            # own served rounds show (accept_len ~3.3)
            positions, at = [], spec["sequence_start"]
            while at < pos_end - 1:
                positions.append(int(at))
                at += int(rng.integers(3, 5))
            positions.append(pos_end - 1)
            man["positions"] = positions
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(man, f, indent=2)
            f.write("\n")

        with open(os.path.join(d, "SHA256SUMS"), "w") as f:
            for fn in ("ring.bin", "ids.bin", "manifest.json"):
                f.write(f"{sha256(os.path.join(d, fn))}  {fn}\n")

        if "positions" in man:
            print(f"{name}: {len(man['positions'])} rounds {man['positions'][:4]}...{man['positions'][-1]}")
        print(f"{name}: {pos_end} rows, ctx_len {min(WINDOW, pos_end - 1)}, "
              f"ring {os.path.getsize(ring_path)} B, tap mean {tap.mean():+.5f} std {tap.std():.5f} "
              f"min {tap.min():+.4f}")


if __name__ == "__main__":
    main()
