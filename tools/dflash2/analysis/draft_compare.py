"""Stage-by-stage comparison of the engine's CUDA DFlash2 round against reference.py.

Who reads the output: whoever has a `TF_FLASHNEXT_DRAFT_DEBUG` dump and a mean advance that does
not match the reference's. `tap_accept.py` says *whether* the CUDA path agrees with the torch
drafter over a whole walk; one number cannot say *where* it stops agreeing, and a round is a pipeline
of nine stages (fuse, embed, five layers, head top-k, selector) of which any one can leave everything
downstream at chance while the shape and the plumbing still look right.

This file replays the one round the engine dumped — same taps, same positions, same context
convention, same ctx_positions, fp32, through `reference.py` itself — and puts the engine's bytes
beside the reference's stage by stage. It never reimplements the drafter: `reference.py` traces its
own intermediates (`forward(trace=...)`, `propose(trace=...)`), so every reference tensor here is one
that the reference actually computed on the way to the tokens `tap_accept.py` scores.

The one exception is the wider fusion of the context stage: the fusion is row-wise, so fusing a few
neighbouring rows costs nothing and buys an alignment search that a same-length comparison cannot do
— it is how a gather that is off by one row shows up as `align +1` rather than as a diffuse loss.

What it reports a stage:
  rel_rms   ||engine - torch|| / ||torch|| per row, mean over rows (bf16 kernels against an fp32
            reference are not bit-identical, so read it as 1e-3 = same, 1e-1 = drifting,
            ~1 = unrelated);
  cos_min   the per-row cosine, worst row: 0.9999 = same direction, 0.9 = something upstream mixed
            or rotated the rows;
  align     which slice of the reference the engine's rows actually are, searched over every legal
            offset. +0 means row i is row i; +1 means the engine's row i is the reference's row
            i + 1 — for `hidden_normed` that is the block-row off-by-one `propose()` exists to avoid;
  top1      the decisive one: argmax over the whole 248,320-token vocabulary of
            `rms_norm(stage, norm) @ lm_head.T`, engine against torch, per row. A stage that mixed
            or mis-indexed its rows loses this to ~0/8 even where rel_rms still looks like bf16
            rounding, because the head is a much sharper instrument than the norm is.

Below the stage table it also runs the one experiment that separates "the rows are wrong" from
"the layers are wrong": the reference's own head and selector over block rows 1..7 (what `propose()`
does) and over rows 0..6, scored against the same targets. Two acceptance numbers from one forward —
if the engine's rows are the second set, the first number is what the round would score with the
right slice, and the second is what it scores now.

Anything the dump does not contain is reported as unlocalised rather than silently skipped: a
missing `layer3.bin` means the divergence is *not* known to be in or after layer 3, and the verdict
says so. Round 0's context is one row (p = 0), so a dump of rounds 0 and 1 cannot test the fusion or
the context attention at all; dump a later round with `TF_FLASHNEXT_DRAFT_DEBUG_ROUNDS=0,64`.

Run with the same torch that `tap_accept.py` uses, e.g. ~/.venvs/marker/bin/python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as ref  # noqa: E402  the drafter port lives next door
from tap_accept import (  # noqa: E402  the loader and the conventions, not a copy of them
    DRAFTER_DEFAULT,
    EMB_KEY,
    EMB_SHARD,
    HEAD_KEY,
    HEAD_SHARD,
    TARGET_DEFAULT,
    block_taps,
    fail,
    load_dump,
    read_bf16,
    to_device,
)

DTYPES = {"bf16": np.uint16, "f32": np.float32, "u32": np.uint32}

# A stage is only called diverged when its agreement is lost, not when bf16 rounding shows: bf16
# against fp32 is a few percent on a residual stream, so the numeric limit is loose and `cos_min`
# and `top1` carry the decision.
REL_LIMIT = {"ctx_fused": 0.05, "embed": 0.02, "hidden_normed": 0.05}

# Where reference.py's propose() starts: it heads `hidden[:, -block + 1:]`, i.e. block row 1 onwards.
PROPOSE_ROW = 1


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------

def round_dirs(debug_dir: Path) -> list[Path]:
    """Every round directory under `debug_dir`, lowest round first, or the directory itself."""
    if (debug_dir / "tensors.json").is_file():
        return [debug_dir]
    found = [p for p in debug_dir.glob("round*") if (p / "tensors.json").is_file()]
    if not found:
        fail(f"no tensors.json in {debug_dir} and no round*/tensors.json under it (did the run have "
             f"TF_FLASHNEXT_DRAFT_DEBUG set, and did the round it asked for run?)")
    return sorted(found, key=lambda p: int(p.name[len("round"):]))


def load_row(d: Path) -> tuple[dict, dict[str, dict]]:
    meta = json.loads((d / "tensors.json").read_text())
    table: dict[str, dict] = {}
    for t in meta["tensors"]:
        if not (d / t["file"]).is_file():
            fail(f"{d.name}/tensors.json names {t['file']} but it is not on disk")
        table[t["name"]] = t
    return meta, table


def read_tensor(d: Path, t: dict) -> torch.Tensor:
    """One dumped tensor as the reference would hold it: bf16 widened to fp32, ids as int64."""
    raw = np.fromfile(d / t["file"], dtype=DTYPES[t["dtype"]])
    want = int(np.prod(t["shape"])) if t["shape"] else 0
    if raw.size != want:
        fail(f"{t['file']}: {raw.size} elements, tensors.json says shape {t['shape']} ({want})")
    if t["dtype"] == "bf16":
        return torch.from_numpy(raw).view(torch.bfloat16).reshape(t["shape"]).to(torch.float32)
    if t["dtype"] == "u32":
        return torch.from_numpy(raw.astype(np.int64)).reshape(t["shape"])
    return torch.from_numpy(raw).reshape(t["shape"])


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------

def row_metrics(eng: torch.Tensor, ref_rows: torch.Tensor) -> dict:
    """Per-row relative rms error and cosine, against reference rows of the same shape."""
    a, b = eng.float(), ref_rows.float()
    rel = (a - b).pow(2).sum(-1).sqrt() / b.pow(2).sum(-1).sqrt().clamp_min(1e-12)
    cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
    return {
        "rel_rms": rel.mean().item(),
        "rel_rms_max": rel.max().item(),
        "cos_min": cos.min().item(),
        "cos_mean": cos.mean().item(),
    }


def best_align(eng: torch.Tensor, wide: torch.Tensor, prefer: int = 0) -> int:
    """The offset `s` that makes `eng == wide[s : s + len(eng)]`, by mean relative rms.

    `prefer` only breaks exact ties (a span can match a neighbour at rounding noise); it does not
    steer the search, so an off-by-one still shows up as an off-by-one.
    """
    r, n = eng.shape[0], wide.shape[0]
    if n <= r:
        return 0
    scored = {s: row_metrics(eng, wide[s:s + r])["rel_rms"] for s in range(0, n - r + 1)}
    best = min(scored, key=lambda s: (round(scored[s], 6), abs(s - prefer)))
    return best


def stage_logits(x: torch.Tensor, head: torch.Tensor, norm_w: torch.Tensor, eps: float) -> torch.Tensor:
    """The drafter's own head over a stage's rows, normalized as the round's final norm would.

    That is "what the head would have said had the round ended here": a stage does not have to be a
    hidden state to be turned into tokens this way, and the head's argmax is far more sensitive to a
    mixed or mis-indexed row than a relative difference is.
    """
    return ref.rms_norm(x.float(), norm_w, eps) @ head.T


def top1_agree(eng: torch.Tensor, ref_rows: torch.Tensor, head, norm_w, eps) -> tuple[int, int]:
    a = stage_logits(eng, head, norm_w, eps).argmax(-1)
    b = stage_logits(ref_rows, head, norm_w, eps).argmax(-1)
    return int((a == b).sum().item()), int(a.numel())


def propose_rows(model, cfg: dict, rows: torch.Tensor, anchor_id: int) -> torch.Tensor:
    """`reference.py`'s propose() over rows the caller picks: its head, its selector, its walk.

    Only the slice differs from propose(): that method heads `hidden[:, -block + 1:]`, and this lets
    the caller ask what the *other* row convention would propose over the same block, which is the
    one measurement that separates "the rows are wrong" from "the layers are wrong".
    """
    # the selector's lattice is a batch of sequences, so the rows keep the leading dim `forward`
    # gives them: a bare [rows, hidden] would have `greedy_path` slice the step axis instead
    if rows.dim() == 2:
        rows = rows[None]
    unary = rows @ model.lm_head.T
    top, cand = unary.topk(cfg["selector_top_k"], dim=-1)
    anchor = torch.tensor([int(anchor_id)], dtype=torch.long, device=rows.device)
    return model.selector.greedy_path(cand, top, rows, anchor)[0]


# --------------------------------------------------------------------------------------------
# The reference round
# --------------------------------------------------------------------------------------------

def replay(model, cfg: dict, ring: torch.Tensor, ids: torch.Tensor, p: int, ctx_rows: int,
           block: int) -> dict:
    """One round at `p`, exactly as `tap_accept.py` runs it, with the intermediates traced."""
    lo = max(0, p - ctx_rows + 1)
    taps = block_taps(ring, lo, p)

    block_ids = torch.full((1, block), int(cfg["mask_token"]), dtype=torch.long, device=ids.device)
    block_ids[0, 0] = int(ids[p])
    positions = torch.arange(p, p + block, dtype=torch.long, device=ids.device)
    ctx_positions = torch.arange(lo, p + 1, dtype=torch.long, device=ids.device)

    trace: dict = {}
    with torch.no_grad():
        hidden_out = model.forward(taps, block_ids, positions, ctx_positions, trace=trace)
        anchor = torch.tensor([int(ids[p])], dtype=torch.long, device=ids.device)
        proposed = model.propose(hidden_out, anchor, trace=trace)[0]
    trace["proposed"] = proposed
    trace["lo"] = lo
    trace["ctx_len"] = int(taps.shape[1])
    trace["taps"] = taps
    trace["positions"] = positions
    trace["block_ids"] = block_ids
    # what the round is scored against: the capture's own next `block - 1` tokens
    trace["targets"] = ids[p + 1: p + block].to(torch.long).cpu()
    return trace


def wide_ctx(model, ring: torch.Tensor, lo: int, p: int, pad: int) -> tuple[torch.Tensor, int]:
    """The fused context over a span `pad` rows wider than the round's, and where `lo` sits in it."""
    lo_w = max(0, lo - pad)
    hi_w = min(ring.shape[0] - 1, p + pad)
    taps = block_taps(ring, lo_w, hi_w)
    with torch.no_grad():
        ctx = ref.rms_norm(taps @ model.fc.T, model.hidden_norm, model.cfg["eps"])
    return ctx[0], lo - lo_w


# --------------------------------------------------------------------------------------------
# One round
# --------------------------------------------------------------------------------------------

def compare_round(d: Path, args, ids, ring, model, cfg, head) -> dict:
    meta, table = load_row(d)
    p = int(meta["p"])
    block = int(meta["block"])
    total_layers = int(meta["layers"])
    wanted = lambda name: table.get(name)  # noqa: E731  a name the dump may not hold

    print(f"\n=== round {meta['round']}  (p={p}, ctx_len={meta['ctx_len']} from {meta['ctx_lo']}, "
          f"anchor={meta['anchor']}, block={block}, layers={total_layers}) ===")

    traced = replay(model, cfg, ring, ids, p, args.ctx_rows, block)
    lo_ref = traced["lo"]
    notes: list[str] = []
    if lo_ref != int(meta["ctx_lo"]) or traced["ctx_len"] != int(meta["ctx_len"]):
        notes.append(
            f"GATHER: the engine's context is positions [{meta['ctx_lo']}, "
            f"{meta['ctx_lo'] + meta['ctx_len']}) but the reference convention is "
            f"[{lo_ref}, {lo_ref + traced['ctx_len']}) — every comparison below is against the "
            f"correct span, so this alone would put a difference in every stage")
    print(f"reference: ctx [{lo_ref}, {lo_ref + traced['ctx_len']})  taps {tuple(traced['taps'].shape)}")
    print(f"           proposed {traced['proposed'].tolist()}")
    print(f"           targets  {traced['targets'].tolist()}")
    cross = cross_check_tap_accept(args, p, traced)
    if cross is not None:
        print(f"replay vs tap_accept.json (p={p}): {cross}")
        if cross != "matches":
            notes.append(f"REPLAY: this file's replay of p={p} does not match tap_accept.json "
                         f"({cross}); settle the conventions here before reading any stage below")

    norm_w, eps = model.norm, cfg["eps"]
    results: dict[str, dict] = {}
    unlocalised: list[str] = []

    def engine(name: str) -> torch.Tensor | None:
        t = wanted(name)
        if t is None:
            unlocalised.append(name)
            return None
        return read_tensor(d, t).to(args.device)

    # ---- the fusion, and where in the prefix the engine's rows came from ------------------------
    ctx_eng = engine("ctx_fused")
    if ctx_eng is not None and ctx_eng.shape[0] > 0:
        ref_ctx = traced["ctx"][0]
        r = min(ctx_eng.shape[0], ref_ctx.shape[0])
        m = row_metrics(ctx_eng[:r], ref_ctx[:r])
        wide, at = wide_ctx(model, ring, lo_ref, p, args.align_pad)
        m["align"] = best_align(ctx_eng, wide, prefer=at) - at
        m["rows"] = int(ctx_eng.shape[0])
        results["ctx_fused"] = m
        if ctx_eng.shape[0] != ref_ctx.shape[0]:
            notes.append(f"CTX LEN: the dump holds {ctx_eng.shape[0]} fused rows, the replay "
                         f"{ref_ctx.shape[0]} — the engine's prefix span is a different length")

    # ---- the block's input rows, then its hidden after each layer ------------------------------
    stage_ref = {"embed": traced["embed"][0]}
    for i in range(total_layers):
        got = traced.get(f"layer{i}")
        if got is not None:
            stage_ref[f"layer{i}"] = got[0] if got.dim() == 3 else got

    for name in ["embed"] + [f"layer{i}" for i in range(total_layers)]:
        e = engine(name)
        if e is None or name not in stage_ref:
            continue
        r_ref = stage_ref[name]
        r = min(e.shape[0], r_ref.shape[0])
        m = row_metrics(e[:r], r_ref[:r])
        shift = best_align(e, r_ref)
        m["align"] = shift
        m["rows"] = int(e.shape[0])
        if args.stage_top1:
            m["top1"] = top1_agree(e[:r], r_ref[shift:shift + r], head, norm_w, eps)
        results[name] = m

    # ---- hidden_normed: the rows the head reads, and which block row they are -------------------
    normed = traced["normed"][0]  # [block, hidden]
    hn_eng = engine("hidden_normed")
    hn_from = PROPOSE_ROW
    if hn_eng is not None:
        hn_from = best_align(hn_eng, normed, prefer=PROPOSE_ROW)
        r = min(hn_eng.shape[0], normed.shape[0] - hn_from)
        m = row_metrics(hn_eng[:r], normed[hn_from: hn_from + r])
        m["align"] = hn_from
        m["rows"] = int(hn_eng.shape[0])
        if args.stage_top1:
            m["top1"] = top1_agree(hn_eng[:r], normed[hn_from: hn_from + r], head, norm_w, eps)
        results["hidden_normed"] = m

    # ---- the head: top-k ids and their logits, against the reference's own rows ----------------
    ids_eng = engine("head_ids")
    unary_eng = engine("head_unary")
    if ids_eng is not None and unary_eng is not None:
        # the reference's top-k over the rows the engine's dump actually holds content of (`hn_from`
        # is its own alignment), so a detile fault is measured without a row offset mixed into it
        ref_logits = stage_logits(normed[hn_from:], head, norm_w, eps)
        k = int(ids_eng.shape[1])
        ref_top = ref_logits.topk(k, dim=-1)
        rows = min(int(ids_eng.shape[0]), int(ref_top.indices.shape[0]))
        overlap, top1, ranks, value_rel = [], 0, [], []
        for t in range(rows):
            e_ids = [int(x) for x in ids_eng[t][:k].tolist()]
            r_ids = [int(x) for x in ref_top.indices[t][:k].tolist()]
            overlap.append(len(set(e_ids) & set(r_ids)) / k)
            top1 += int(e_ids[0] == r_ids[0])
            row = ref_logits[t]
            for tid in e_ids:
                ranks.append(int((row > row[tid]).sum().item()))
            want = unary_eng[t][:k].float()
            got = row[torch.tensor(e_ids, device=row.device)]
            value_rel.append(((got - want).abs().mean() / want.abs().mean().clamp_min(1e-9)).item())
        results["head_top16"] = {
            "rows": rows, "k": k,
            "overlap": sum(overlap) / rows,
            "top1": top1 / rows,
            "median_rank": float(np.median(ranks)) if ranks else -1.0,
            "max_rank": float(np.max(ranks)) if ranks else -1.0,
            "value_rel": float(np.mean(value_rel)) if value_rel else -1.0,
        }

    # ---- which block rows the head heads: the reference's convention against the engine's -------
    if hn_eng is not None:
        anchor_id = int(meta["anchor"])
        conv = {}
        for label, rows_at in (("rows 1..7 (reference.py's propose)", PROPOSE_ROW),
                              ("rows 0..6 (one block row early)", 0)):
            prop = propose_rows(model, cfg, normed[rows_at: rows_at + hn_eng.shape[0]], anchor_id)
            conv[label] = int(ref.acceptance(prop.cpu(), traced["targets"]))
        conv["engine_rows"] = hn_from
        results["row_convention"] = conv

    # ---- proposals: token agreement, and the acceptance both sides get -------------------------
    prop_eng = engine("proposals")
    targets = traced["targets"].tolist()
    ref_prop = [int(x) for x in traced["proposed"][: block - 1].tolist()]
    pr: dict = {"reference": ref_prop, "targets": targets,
                "reference_accept": ref.acceptance(traced["proposed"][: block - 1], traced["targets"])}
    if prop_eng is not None:
        eng_prop = [int(x) for x in prop_eng.flatten().tolist()][: block - 1]
        pr.update({
            "engine": eng_prop,
            "agree": sum(1 for a, b in zip(eng_prop, ref_prop) if a == b),
            "rows": min(len(eng_prop), len(ref_prop)),
            "engine_accept": ref.acceptance(torch.tensor(eng_prop), traced["targets"]),
        })
    results["proposals"] = pr

    report_table(results, unlocalised, meta, total_layers)
    verdict(results, unlocalised, notes, meta, total_layers)
    return {"meta": meta, "results": results, "unlocalised": unlocalised, "notes": notes,
            "reference": {"p": p, "lo": lo_ref, "ctx_len": traced["ctx_len"],
                          "proposed": ref_prop, "targets": targets}}


def cross_check_tap_accept(args, p: int, traced: dict) -> str | None:
    """Whether this file's replay of `p` is the round `tap_accept.json` scored, if that file exists."""
    if args.dump is None:
        return None
    path = Path(args.dump).expanduser() / "tap_accept.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError:
        return "tap_accept.json is not valid json"
    for r in doc.get("per_round", []):
        if int(r["p"]) == p:
            got = [int(x) for x in traced["proposed"][: len(r["proposed"])].tolist()]
            if got == [int(x) for x in r["proposed"]]:
                return "matches"
            return f"differs: replay {got} vs json {r['proposed']}"
    return f"p={p} is not in tap_accept.json's walk"


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------

def report_table(results: dict, unlocalised: list[str], meta: dict, total_layers: int) -> None:
    print(f"\n{'stage':<15}{'rows':>5}{'align':>7}{'rel_rms':>10}{'rel_max':>9}{'cos_min':>9}   top1")
    order = ["ctx_fused", "embed"] + [f"layer{i}" for i in range(total_layers)] + ["hidden_normed"]
    for name in order:
        r = results.get(name)
        if r is None:
            print(f"{name:<15}{'-':>5}{'-':>7}   not dumped — stage unlocalised")
            continue
        rows = r.get("rows", "-")
        t1 = f"{r['top1'][0]}/{r['top1'][1]}" if "top1" in r else ("-" if name == "ctx_fused" else "")
        print(f"{name:<15}{rows:>5}{r['align']:+7d}{r['rel_rms']:>10.5f}{r['rel_rms_max']:>9.4f}"
              f"{r['cos_min']:>9.5f}   {t1}")

    conv = results.get("row_convention")
    if conv:
        n_t = len(results.get("proposals", {}).get("targets", []))
        print("row convention (the reference's head and selector, over different block rows):")
        for label, got in conv.items():
            if label == "engine_rows":
                continue
            mark = "  <- what the engine's dump says it heads" if label.startswith(
                f"rows {conv['engine_rows']}..") else ""
            print(f"  {label:<36} {got}/{n_t}{mark}")

    h = results.get("head_top16")
    if h:
        print(f"{'head top-16':<15}{h['rows']:>5}   top-16 overlap {h['overlap'] * h['k']:.1f}/{h['k']}   "
              f"top-1 {h['top1'] * h['rows']:.0f}/{h['rows']}   "
              f"engine ids' median rank in the reference row {h['median_rank']:.0f} "
              f"(max {h['max_rank']:.0f})   logits at those ids differ by {100 * h['value_rel']:.1f}%")
    else:
        print(f"{'head top-16':<15}{'-':>5}   not dumped — stage unlocalised")

    pr = results.get("proposals") or {}
    if "engine" in pr:
        print(f"{'proposals':<15}{pr['rows']:>5}   token agreement {pr['agree']}/{pr['rows']}   "
              f"acceptance engine {pr['engine_accept']} vs reference {pr['reference_accept']} "
              f"of {len(pr['targets'])}")
        if pr["agree"] < pr["rows"]:
            print(f"{'':<21}engine    {pr['engine']}")
            print(f"{'':<21}reference {pr['reference']}")
            print(f"{'':<21}targets   {pr['targets']}")
    else:
        print(f"{'proposals':<15}{'-':>5}   not dumped — stage unlocalised"
              + (f" (reference acceptance {pr['reference_accept']})" if "reference_accept" in pr else ""))
    if unlocalised:
        print(f"unlocalised: {', '.join(sorted(set(unlocalised)))}")


def verdict(results: dict, unlocalised: list[str], notes: list[str], meta: dict, total_layers: int) -> None:
    """Name the first stage that fails, and say what that does and does not prove.

    Four questions, in the order a round asks them: did a tensor stage lose agreement, are the rows
    the head reads the right block rows, does the head's own top-k agree, does the walk. Each is
    reported with what it implicates, because "the numbers are wrong" is the finding that wasted the
    most time here.
    """
    for line in notes:
        print(f"! {line}")

    order = ["ctx_fused", "embed"] + [f"layer{i}" for i in range(total_layers)] + ["hidden_normed"]
    first = None
    head_named = False
    head_named = False
    for name in order:
        r = results.get(name)
        if r is None:
            continue
        frac = (r["top1"][0] / r["top1"][1]) if "top1" in r else 1.0
        if r["rel_rms"] > REL_LIMIT.get(name, 0.10) or r["cos_min"] < 0.99 or frac < 0.75:
            first = (name, f"rel_rms {r['rel_rms']:.4f}, cos_min {r['cos_min']:.5f}, top1 {frac:.2f}")
            break

    print()
    if first is not None:
        print(f"localised: the divergence begins at {first[0]}  ({first[1]}).")
        idx = order.index(first[0])
        after = [n for n in order[idx + 1:] if results.get(n) is not None]
        if after:
            print(f"           {', '.join(after[:5])} are downstream of it and cannot be read as")
            print("           separate findings until this stage agrees.")
    else:
        # every tensor stage agrees: the round's remaining ways to be wrong are the row the head
        # reads, the head's own top-k, and the selector's walk
        hn = results.get("hidden_normed")
        h = results.get("head_top16")
        pr = results.get("proposals") or {}
        if hn is not None and hn["align"] != PROPOSE_ROW:
            print(f"localised: every tensor stage agrees to bf16 rounding, but the rows are the wrong "
                  f"block rows.")
            print(f"           The engine's hidden_normed row i is the reference's block row "
                  f"{hn['align']} + i (align {hn['align']:+d}); reference.py's propose() heads block "
                  f"rows {PROPOSE_ROW}..{meta['block'] - 1} (`hidden[:, -block + 1:]`). The check's "
                  f"final rmsnorm starts at block row 0, so the head scores rows 0..{meta['block'] - 2}:")
            print(f"           one block row early, which shifts every proposal without changing one "
                  f"tensor's fidelity.")
        elif h is not None and (h["top1"] < 0.75 or h["median_rank"] > int(meta["vocab"]) // 100):
            head_named = True
            print(f"localised: the divergence begins at the head (tf_fxd2_head_topk).")
            print(f"           Its top-1 agrees {h['top1'] * h['rows']:.0f}/{h['rows']} with the "
                  f"reference over the same rows, and its own top-{h['k']} ids sit at a median rank "
                  f"{h['median_rank']:.0f}")
            print(f"           in the reference's ordering — not tokens this row scores highly. The "
                  f"first suspect is the map")
            print(f"           from the engine's head tiles to row-major rows (`detile` in "
                  f"cuda_draft2_check.zig), then the detiled")
            print(f"           head's row count: both leave the row's fp32 logits plausible and its "
                  f"top-k random.")
        elif "engine" in pr and pr["agree"] < pr["rows"]:
            print(f"localised: the divergence begins at the selector (tf_fxd2_select).")
            print(f"           The head's candidates agree with the reference's, but the walk lands "
                  f"{pr['agree']}/{pr['rows']} tokens;")
            print(f"           check the predecessor/successor orientation and the row the walk reads "
                  f"(`hidden_projection` over hn).")
        elif "engine" in pr:
            print("localised: no stage this dump covers disagrees beyond bf16 rounding, and the "
                  "round's proposals match.")
            print("           A walk whose mean advance is still low differs in a round this dump does "
                  "not hold, or in the walk itself (advance/overlap), not in a stage here.")
        else:
            print("localised: every stage this dump holds agrees, and the proposals were not dumped —")
            print("           the walk's own agreement is therefore unknown. Re-dump with the head and "
                  "the proposals in it (they are written by the same round).")

    if (not head_named and results.get("head_top16")
            and results["head_top16"]["median_rank"] > int(meta["vocab"]) // 100):
        print(f"! HEAD RANK: the engine's top-{results['head_top16']['k']} ids sit at median rank "
              f"{results['head_top16']['median_rank']:.0f} in the reference row's ordering — the "
              f"signature of a wrong tiled-to-row-major map (or a stale scale), not a token the "
              f"drafter got wrong.")
    if unlocalised:
        print(f"unlocalised: {', '.join(sorted(set(unlocalised)))} — the divergence is not known to be "
              f"before these; re-dump them with TF_FLASHNEXT_DRAFT_DEBUG_ROUNDS")


# --------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="engine vs reference.py, stage by stage, one dumped round")
    ap.add_argument("--debug-dir", required=True, type=Path,
                    help="TF_FLASHNEXT_DRAFT_DEBUG's directory (or one round<k> inside it)")
    ap.add_argument("--round", type=int, default=None,
                    help="which dumped round to read when the directory holds several (default: all)")
    ap.add_argument("--dump", type=Path, default=None,
                    help="the tap capture (manifest.json/ids.bin/ring.bin); default: the one tensors.json names")
    ap.add_argument("--drafter", type=Path, default=DRAFTER_DEFAULT)
    ap.add_argument("--target", type=Path, default=TARGET_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--ctx-rows", type=int, default=0,
                    help="context cap (0 = the capture's window, which is what the check uses)")
    ap.add_argument("--align-pad", type=int, default=4,
                    help="extra reference fusion rows either side of the context, for the gather search")
    ap.add_argument("--no-stage-top1", dest="stage_top1", action="store_false",
                    help="skip the per-stage head top-1 (7 vocab-wide argmaxes; ~seconds on a GPU)")
    ap.add_argument("--out", type=Path, default=None, help="JSON report path")
    args = ap.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("draft_compare: CUDA unavailable, falling back to cpu", file=sys.stderr)
        args.device = "cpu"

    dirs = round_dirs(args.debug_dir.expanduser().resolve())
    if args.round is not None:
        dirs = [d for d in dirs if int(load_row(d)[0]["round"]) == args.round]
        if not dirs:
            fail(f"no dumped round {args.round} under {args.debug_dir}")

    meta0 = load_row(dirs[0])[0]
    capture = args.dump
    if capture is None:
        named = meta0.get("capture") or ""
        if not named:
            fail("the dump does not name its capture; pass --dump <the TF_FLASHNEXT_TAP_DUMP directory>")
        capture = Path(named)
    args.dump = capture
    capture = capture.expanduser().resolve()
    print(f"engine dump     {args.debug_dir}  ({len(dirs)} round{'s' if len(dirs) != 1 else ''})")
    print(f"capture         {capture}")

    manifest, ids, ring = load_dump(capture, args.device)
    hidden = int(manifest["hidden"])
    head = read_bf16(args.target.expanduser() / HEAD_SHARD, HEAD_KEY, args.device)
    embed = read_bf16(args.target.expanduser() / EMB_SHARD, EMB_KEY, args.device)
    if head.shape[1] != hidden:
        fail(f"lm_head is [_, {head.shape[1]}], the capture's hidden is {hidden}")
    model, cfg = ref.load(args.drafter.expanduser(), head, embed)
    model.lm_head, model.embed = head, embed
    to_device(model, args.device)
    if cfg["taps"] != list(manifest["taps"]):
        fail(f"drafter target_layer_ids {cfg['taps']} != the capture's taps {manifest['taps']}")
    if cfg["hidden"] != hidden:
        fail(f"drafter hidden {cfg['hidden']} != the capture's {hidden}")

    # the check's own convention: min(the drafter's window, the ring's rows), not the whole capture
    if not args.ctx_rows:
        args.ctx_rows = min(int(manifest.get("window") or ring.shape[0]), int(ring.shape[0]))

    out = []
    for d in dirs:
        meta = load_row(d)[0]
        if int(meta["block"]) != args.block or args.block != int(manifest["block"]):
            fail(f"block: dump {meta['block']}, capture {manifest['block']}, --block {args.block}")
        out.append(compare_round(d, args, ids, ring, model, cfg, head))

    if args.out:
        args.out.write_text(json.dumps({"debug_dir": str(args.debug_dir), "capture": str(capture),
                                        "rounds": out}, indent=1, default=str) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
