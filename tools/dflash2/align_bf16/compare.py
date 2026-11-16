#!/usr/bin/env python
"""Join two stage dumps by name and report, stage by stage, how far apart they are.

Both sides write `tensors/<stage>.npy` as **f32 widenings of the tensor the stage actually holds**
(`cuda_draft2_check.zig` and `reference_bf16.py` both do), so f32 equality is bf16-bit equality for
every bf16 stage and plain f32 equality for the logits. That is the whole reason the dumps are f32
rather than a numpy `bf16` (which numpy does not have).

For each stage: shape, the dtype it was recorded at (from both `stages.json`), max abs diff, mean
abs diff, relative L2 (`||a - b|| / ||a||`), cosine, and whether the two arrays are bit-identical.
A stage only one side wrote is listed as MISSING, by name, and never silently dropped.

  python tools/dflash2/align_bf16/compare.py <reference-dir> <ours-dir> [--csv out.csv] [--md out.md]
"""

import argparse
import json
import os

import numpy as np

# the round's stages, in the order a round produces them: a join by name cannot order itself
ORDER = ["embed", "fc_ctx_rows", "fc_ctx"]
for _i in range(5):
    ORDER += [f"conv_attn_in_{_i}", f"qkv_{_i}", f"attn_{_i}", f"attn_proj_{_i}",
              f"conv_attn_out_{_i}", f"conv_mlp_in_{_i}", f"mlp_{_i}",
              f"conv_mlp_out_{_i}", f"block_hidden_{_i}"]
ORDER += ["final_norm_block", "logits_full", "logits_top16_ids", "logits_top16_vals",
          "selector_scores", "proposals"]


def load(d):
    with open(os.path.join(d, "stages.json")) as f:
        doc = json.load(f)
    meta = {s["stage"]: s for s in doc["stages"]}
    arrs = {}
    for name in meta:
        p = os.path.join(d, "tensors", f"{name}.npy")
        if os.path.exists(p):
            arrs[name] = np.load(p)
    return doc, meta, arrs


def bf16_exact(a: np.ndarray) -> bool:
    """Whether every value is exactly a bf16 (a bf16 widening has 16 zero low bits)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return bool(np.all((u & 0xFFFF) == 0))


def compare(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = a.size
    same = bool(np.array_equal(np.ascontiguousarray(np.asarray(a, dtype=np.float32)),
                               np.ascontiguousarray(np.asarray(b, dtype=np.float32))))
    d = a - b
    absd = np.abs(d)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {
        "n": n,
        "max_abs": float(absd.max()) if n else 0.0,
        "mean_abs": float(absd.mean()) if n else 0.0,
        "rel_l2": float(np.linalg.norm(d) / na) if na > 0 else 0.0,
        "cosine": float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0,
        "identical": same,
        "exact_frac": float(np.mean(np.asarray(a) == np.asarray(b))) if n else 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("ours")
    ap.add_argument("--csv")
    ap.add_argument("--md")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ref_doc, ref_meta, ref = load(args.ref)
    our_doc, our_meta, ours = load(args.ours)

    print(f"reference {args.ref}  impl={ref_doc.get('impl')} dtype={ref_doc.get('dtype')} "
          f"device={ref_doc.get('device')}")
    print(f"ours      {args.ours}  impl={our_doc.get('impl')} dtype={our_doc.get('dtype')} "
          f"device={our_doc.get('device')}")
    rr, ro = ref_doc.get("round", {}), our_doc.get("round", {})
    print(f"round     ref: pos {rr.get('position')} ctx {rr.get('ctx_len')} proposals {rr.get('proposals')}")
    print(f"          our: pos {ro.get('position')} ctx {ro.get('ctx_len')} proposals {ro.get('proposals')}")
    print(f"geometry  ref ctx_lo {rr.get('ctx_lo')} anchor {rr.get('anchor')} | "
          f"our ctx_lo {ro.get('ctx_lo')} anchor {ro.get('anchor')}")
    print()

    names = [n for n in ORDER if n in ref or n in ours]
    names += sorted(n for n in (set(ref) | set(ours)) if n not in names)

    rows, missing = [], []
    for name in names:
        if name not in ref or name not in ours:
            missing.append((name, "ours" if name in ref else "reference"))
            continue
        a, b = ref[name], ours[name]
        if a.shape != b.shape:
            missing.append((name, f"shape {a.shape} vs {b.shape}"))
            continue
        # the dtypes each side recorded, and whether a bf16 stage really is bf16 on both sides
        rsrc = ref_meta.get(name, {}).get("src_dtype", "?")
        osrc = our_meta.get(name, {}).get("src_dtype", "?")
        c = compare(a, b)
        flagged = ""
        if rsrc == "bf16" and not bf16_exact(a):
            flagged += "ref-not-bf16 "
        if osrc == "bf16" and not bf16_exact(b):
            flagged += "ours-not-bf16 "
        rows.append({"stage": name, "shape": "x".join(str(x) for x in a.shape),
                     "ref_dtype": rsrc, "our_dtype": osrc, **c, "flags": flagged.strip()})

    hdr = f"{'stage':<20} {'shape':<14} {'dtype':<10} {'max abs':>12} {'mean abs':>12} {'rel L2':>11} {'cosine':>18} {'bit-id':>7}  {'exact%':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        dt = r["ref_dtype"] if r["ref_dtype"] == r["our_dtype"] else f"{r['ref_dtype']}/{r['our_dtype']}"
        print(f"{r['stage']:<20} {r['shape']:<14} {dt:<10} {r['max_abs']:>12.6g} {r['mean_abs']:>12.6g} "
              f"{r['rel_l2']:>11.3e} {r['cosine']:>18.15f} {'yes' if r['identical'] else 'no':>7}  "
              f"{100 * r['exact_frac']:>7.3f}" + (f"  {r['flags']}" if r["flags"] else ""))

    if missing:
        print("\nnot compared:")
        for name, why in missing:
            print(f"  {name:<20} {why}")

    first = next((r["stage"] for r in rows if not r["identical"]), None)
    print(f"\ncompared {len(rows)} stages, {sum(1 for r in rows if r['identical'])} bit-identical; "
          f"first divergence: {first or '(none)'}")
    if rr.get("proposals") and ro.get("proposals"):
        print(f"proposals identical: {rr['proposals'] == ro['proposals']}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stage", "shape", "ref_dtype", "our_dtype", "max_abs", "mean_abs",
                        "rel_l2", "cosine", "bit_identical", "exact_frac", "flags"])
            for r in rows:
                w.writerow([r["stage"], r["shape"], r["ref_dtype"], r["our_dtype"],
                            f"{r['max_abs']:.9g}", f"{r['mean_abs']:.9g}", f"{r['rel_l2']:.6e}",
                            f"{r['cosine']:.17g}", "yes" if r["identical"] else "no",
                            f"{r['exact_frac']:.9f}", r["flags"]])
            for name, why in missing:
                w.writerow([name, "", "", "", "", "", "", "", "MISSING", "", why])
        print(f"wrote {args.csv}")
    if args.md:
        with open(args.md, "w") as f:
            f.write(f"# DFlash2 drafter: reference (torch, bf16, GPU) vs engine (CUDA, bf16) {args.label}\n\n")
            f.write(f"- reference: `{ref_doc.get('impl')}` {ref_doc.get('device')} {ref_doc.get('dtype')}\n")
            f.write(f"- ours: `{our_doc.get('impl')}` {our_doc.get('device')} {our_doc.get('dtype')}\n")
            f.write(f"- round: position {rr.get('position')} ctx_len {rr.get('ctx_len')} "
                    f"anchor {rr.get('anchor')}\n")
            f.write(f"- reference proposals {rr.get('proposals')}\n")
            f.write(f"- engine proposals    {ro.get('proposals')}\n\n")
            f.write("| stage | shape | dtype | max abs diff | mean abs diff | rel L2 | cosine | bit-identical |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                dt = r["ref_dtype"] if r["ref_dtype"] == r["our_dtype"] else f"{r['ref_dtype']}/{r['our_dtype']}"
                f.write(f"| `{r['stage']}` | {r['shape']} | {dt} | {r['max_abs']:.6g} | "
                        f"{r['mean_abs']:.6g} | {r['rel_l2']:.3e} | {r['cosine']:.15f} | "
                        f"{'yes' if r['identical'] else 'no'} |\n")
            if missing:
                f.write("\nNot compared:\n\n")
                for name, why in missing:
                    f.write(f"- `{name}`: {why}\n")
            f.write(f"\nFirst divergence: `{first or '(none)'}`. "
                    f"{sum(1 for r in rows if r['identical'])} of {len(rows)} stages bit-identical.\n")
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
