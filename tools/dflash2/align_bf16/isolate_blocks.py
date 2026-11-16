#!/usr/bin/env python
"""Block-by-block isolation of the engine-vs-reference divergence.

The bf16 probe (`reference_bf16.py` vs `run_engine.sh`) showed no stage bit-identical and a
divergence that *compounds*: 3.3e-3 relative L2 on the shared `fc_ctx_rows`, doubling per layer to
2.6e-1 by `block_hidden_4`. That shape is ambiguous on its own — a systematic per-layer error and a
single bad input amplified by five healthy blocks look identical. This separates them.

Block k is run twice, both through the **reference's** block-k code:

  A. reference-in, reference-out — the reference's own `block_hidden_{k-1}` feeds block k. The
     control: it reproduces the compounding curve inside this one process.
  B. ours-in, reference-out — the **engine's** `block_hidden_{k-1}` feeds the same block k.

Each output is compared against the engine's own `block_hidden_k`:

  * B matches the engine  → block k is right; the reference diverged only because of the input it
                            was handed, so the fault is upstream of k.
  * B does not match      → this block turns a matching input into a mismatched output. The fault is
                            here, and it is the first such block.

Block 0's predecessor is the round's residual start (the block embedding), and the shared input for B
is the engine's `embed`. Nothing on the engine side is recomputed: every engine tensor is read from
the `.npy` the dump wrote.

Usage:
  python isolate_blocks.py --fixture <fixture-case-dir> <ref-dir> <ours-dir>
                           [--blocks 0,1,2,3,4] [--out report.json]

`ref-dir` is a `reference_bf16.py` output and `ours-dir` a `run_engine.sh` output for the same
fixture case; both must hold `tensors/` and `stages.json`.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_reference():
    import importlib.util
    script = os.path.join(HERE, "reference_bf16.py")
    spec = importlib.util.spec_from_file_location("reference_bf16", script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_npy(side, name):
    p = os.path.join(side, "tensors", f"{name}.npy")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    return np.load(p)


def stats(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        return {"shape_mismatch": True, "a": list(a.shape), "b": list(b.shape)}
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    d = a64 - b64
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    return {
        "max_abs": float(np.abs(d).max()),
        "rel_l2": float(np.linalg.norm(d) / na) if na else float("nan"),
        "cosine": float(np.dot(a64, b64) / (na * nb)) if na and nb else float("nan"),
        "bit_identical": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("ours_dir")
    ap.add_argument("--fixture", required=True, help="the fixture case dir both sides ran")
    ap.add_argument("--blocks", default="0,1,2,3,4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv[1:])

    ref = load_reference()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    ref_doc = json.load(open(os.path.join(args.ref_dir, "stages.json")))
    our_doc = json.load(open(os.path.join(args.ours_dir, "stages.json")))
    rr, orr = ref_doc["round"], our_doc["round"]
    if rr["position"] != orr["position"] or rr["ctx_len"] != orr["ctx_len"]:
        raise SystemExit(f"geometry differs: ref {rr['position']}/{rr['ctx_len']} "
                         f"ours {orr['position']}/{orr['ctx_len']} — not the same round")
    print(f"[iso] position {rr['position']} ctx {rr['ctx_len']} block {rr['block']} "
          f"fixture {args.fixture}", flush=True)

    # ---- the static side, assembled exactly as reference_bf16.main() assembles it ----------------
    device, dtype = "cuda", torch.bfloat16
    man, ring, ids = ref.load_fixture(args.fixture)
    P = man["pos_end"] - 1
    ctx_len = min(man["window"], P)
    lo = P - ctx_len
    ctx_pos = np.arange(lo, lo + ctx_len)
    ctx_rows = ring[ctx_pos % man["rows"]]
    ctx_x = torch.from_numpy(ctx_rows.copy()).view(torch.bfloat16).to(device)[None]

    vendor = ref.load_vendor()
    model = ref.build_model(vendor, device)
    embed_w, lmhead_w = ref.load_target_aliases(device, dtype)
    lm_head = ref.LMHead(lmhead_w).to(device)

    anchor = int(ids[P])
    block_ids = torch.tensor([[anchor] + [ref.MASK_TOKEN] * (ref.BLOCK - 1)],
                             dtype=torch.long, device=device)
    pos_ids = torch.arange(lo, lo + ctx_len + ref.BLOCK, dtype=torch.long, device=device)[None]
    noise_emb = F.embedding(block_ids, embed_w)

    # one warmup so cuBLAS has every shape it will be asked for
    ref.run_round(vendor, model, ctx_x, block_ids, pos_ids, embed_w, lm_head, anchor, {})

    def run_block(start_hidden, k):
        """Reference's block k only, from `start_hidden` [1, block, hidden]."""
        stages = {}
        ref.run_round(vendor, model, ctx_x, block_ids, pos_ids, embed_w, lm_head, anchor, stages,
                      start_hidden=start_hidden, start_layer=k, stop_layer=k + 1)
        torch.cuda.synchronize()
        return stages

    def to_bf16(arr):
        # a dump is an f32 *widening* of a bf16 value: cast to bf16 (exact), never reinterpret the
        # bytes — `.view(torch.bfloat16)` on f32 halves the element size and doubles the count
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype) \
            .reshape(1, ref.BLOCK, -1)

    rows = []
    prev_ref = noise_emb      # the reference's own residual start
    prev_our = noise_emb      # A/B both begin from the same embedding, which is not in dispute
    for k in [int(x) for x in args.blocks.split(",")]:
        out_ref = run_block(prev_ref, k)
        # block 0's predecessor is the block embedding, which the engine does not dump as a stage;
        # it is a table lookup on both sides and not in dispute, so the reference's own copy is used.
        our_in_path = os.path.join(args.ours_dir, "tensors", f"block_hidden_{k - 1}.npy") if k else None
        our_in = to_bf16(np.load(our_in_path)) if our_in_path else noise_emb
        out_our_in = run_block(our_in, k)
        ours_k = load_npy(args.ours_dir, f"block_hidden_{k}")
        ref_k = load_npy(args.ref_dir, f"block_hidden_{k}")

        # run_round already stores the stage unbatched ([block, hidden]); no [0] here
        a = out_ref[f"block_hidden_{k}"].float().cpu().numpy()
        b = out_our_in[f"block_hidden_{k}"].float().cpu().numpy()
        row = {
            "block": k,
            "control_ref_in_vs_our": stats(ours_k, a),
            "ours_in_vs_our": stats(ours_k, b),
            "ours_in_vs_ref_in": stats(a, b),
        }
        rows.append(row)
        print(f"[iso] block {k}: ours-in/our relL2 {row['ours_in_vs_our']['rel_l2']:.3e}  "
              f"ref-in/our relL2 {row['control_ref_in_vs_our']['rel_l2']:.3e}  "
              f"identical(ours-in) {row['ours_in_vs_our']['bit_identical']}", flush=True)

        prev_ref = to_bf16(a)
        prev_our = to_bf16(b)

    # the verdict: the first block whose *own* arithmetic turns a matching input into a mismatch
    first_fault = next((r["block"] for r in rows if not r["ours_in_vs_our"]["bit_identical"]), None)
    report = {
        "fixture": args.fixture,
        "ref_dir": args.ref_dir,
        "ours_dir": args.ours_dir,
        "position": rr["position"],
        "ctx_len": rr["ctx_len"],
        "blocks": rows,
        "first_block_that_fails_on_our_input": first_fault,
    }
    print(f"[iso] first block failing on our own input: {first_fault}", flush=True)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ours_dir)), "isolate.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(f"[iso] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))