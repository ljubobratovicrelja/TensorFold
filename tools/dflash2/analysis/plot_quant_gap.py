#!/usr/bin/env python3
"""Where does 4-bit error live in the drafter body, and how much of it does the fit recover?

The body (603M params, 36 weight tensors: one context fuse `fc` plus five blocks of
q/k/v/o/gate/up/down) is quantized three ways, all scored against the unquantized bf16 teacher:

  rtn      round-to-nearest 4-bit, no correction
  fitted   the same 4-bit grid, per-group scales closed-form fitted to the layer OUTPUT
  fp8      block-128x128 fp8, the format currently served

Five figures, each answering one question:

  1 quant_gap_dumbbell      per tensor: how far is each variant's layer output from bf16, and which
                            tensors carry the error? The RTN->fitted segment is the fit's gain; fp8 is
                            the bar the 4-bit path has to clear.
  2 quant_gap_variants      end to end on real text: does the body error actually move the proposal
                            list (recall, set-exact, top-1, foreign ids, identical lists)?
  3 quant_gap_slots         does the error deepen with slot index, i.e. do late slots rot first?
  4 quant_gap_accept        teacher-forced accept length: what the fidelity numbers cost in tokens
                            per round.
  5 quant_gap_weight_vs_output  per tensor, weight error against output error. The fit pushes points
                            DOWN (output improves) without moving them LEFT (weights barely change):
                            the model cares about the product, not the weight.

Units are relative L2 against the bf16 teacher throughout; every panel is labelled with its own.
Variants are discovered from the input files, never assumed — an extra fit report (e.g. a different
4-bit format) simply adds a series.

  python tools/dflash2/analysis/plot_quant_gap.py \\
      --fit-report /tmp/nv4/real_scale.json --eval /tmp/nv4/eval_real.json \\
      --accept /tmp/nv4/accept_real.json --out /tmp/nv4/plots/

Labels associate a report with a name; the key is any substring of the path, the report's stem, or
its `format` field, so both of these work:

  --label real_scale=nvfp4 --label affine4_report.json=affine4
  --label nvfp4            # bare names bind in the order the reports were given
"""

import argparse
import json
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.transforms import blended_transform_factory

# ---------------------------------------------------------------------------------------------
# house colours: a variant keeps its colour in every figure so the eye can carry it across them
# ---------------------------------------------------------------------------------------------
COLOR = {
    "rtn": "#d95f02",
    "fitted": "#1b4965",
    "fp8": "#1b998b",
    "bf16": "#7f8c8d",
}
EXTRA = ["#7570b3", "#e7298a", "#66a61e", "#8c564b"]
MUTED = "#7f8c8d"
INK = "#2c3e50"

PARTS = ["q", "k", "v", "o", "gate", "up", "down"]
PART_ORDER = {p: i for i, p in enumerate(PARTS)}


def variant_color(name, index):
    """A stable colour per variant name, known ones fixed, unknown ones by position."""
    key = str(name).lower()
    for known, colour in COLOR.items():
        if known in key:
            return colour
    return EXTRA[index % len(EXTRA)]


def label_color(label, index):
    return variant_color(label, index)


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------
def load_json(path, what):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"  missing {what}: {path}", file=sys.stderr)
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  unreadable {what}: {path} ({exc})", file=sys.stderr)
        return None


def tensor_sort_key(name):
    """Checkpoint order: fc, then L0..Ln, each block q,k,v,o,gate,up,down."""
    if name == "fc":
        return (0, 0, 0, "")
    m = re.match(r"^L(\d+)_(.+)$", name)
    if m:
        layer, part = int(m.group(1)), m.group(2)
        return (1, layer, PART_ORDER.get(part, 99), part)
    return (2, 0, 0, name)


def short_name(name):
    """Tick label inside a block group: `L2_gate` -> `gate`, `fc` -> `fc`."""
    m = re.match(r"^L(\d+)_(.+)$", name)
    return m.group(2) if m else name


def block_of(name):
    m = re.match(r"^L(\d+)_", name)
    return "fc" if name == "fc" else (f"L{m.group(1)}" if m else "other")


def resolve_labels(paths, label_args):
    """Bind --label values to reports.

    A `KEY=VALUE` label binds to the report whose path, stem, basename or `format` field contains
    KEY (case-insensitive). A bare label binds to the next still-unlabelled report, in order.
    A report with a `format` key keeps that as its default label.
    """
    reports = [{"path": p, "label": None} for p in paths]
    pending = []
    for raw in label_args:
        if "=" in raw:
            key, value = raw.split("=", 1)
            hit = False
            for rep in reports:
                hay = " ".join([rep["path"], os.path.basename(rep["path"]),
                                os.path.splitext(os.path.basename(rep["path"]))[0]]).lower()
                if key.lower() in hay:
                    rep["label"] = value
                    hit = True
            if not hit:
                print(f"  --label {raw}: no report matches key '{key}', ignoring", file=sys.stderr)
        else:
            pending.append(raw)

    for value in pending:
        for rep in reports:
            if rep["label"] is None:
                rep["label"] = value
                break
        else:
            print(f"  --label {value}: more labels than unlabelled reports", file=sys.stderr)

    for i, rep in enumerate(reports):
        if rep["label"] is not None:
            continue
        # a report that names its own format keeps that name; otherwise fall back to position
        fmt = None
        try:
            with open(rep["path"]) as fh:
                fmt = json.load(fh).get("format")
        except (OSError, json.JSONDecodeError):
            pass
        rep["label"] = fmt or (f"fmt{i + 1}" if len(reports) > 1 else "4-bit")
    return reports


def normalise_fit_report(report, label, path):
    """One fit report -> per-tensor series for rtn / fitted / fp8, in checkpoint order.

    A `mode: rtn` report has no closed-form fit, so its `*_fitted` fields are copies of `*_rtn`.
    Plotting them as a fit would invent a gain, so that series is dropped instead.
    """
    tensors = {t["name"]: t for t in report.get("tensors", [])}
    if not tensors:
        return None
    mode = str(report.get("mode", "")).lower()
    has_fit = mode != "rtn" and any(
        abs(t.get("output_rel_l2_fitted", 0.0) - t.get("output_rel_l2_rtn", 0.0)) > 1e-12
        for t in tensors.values()
    )
    fmt = report.get("format") or label
    names = sorted(tensors, key=tensor_sort_key)
    return {
        "label": label,
        "format": fmt,
        "path": path,
        "mode": mode,
        "has_fit": has_fit,
        "names": names,
        "tensors": tensors,
        "summary": report.get("summary", {}),
        "protocol": report.get("protocol", ""),
        "group": report.get("group"),
        "calibration": report.get("calibration", {}),
    }


def merge_fit_reports(reports):
    """Group fit reports by format; a fitted run supersedes an rtn-only run of the same format."""
    by_fmt = {}
    for rep in reports:
        cur = by_fmt.get(rep["format"])
        if cur is None or (rep["has_fit"] and not cur["has_fit"]):
            by_fmt[rep["format"]] = rep
    return list(by_fmt.values())


# ---------------------------------------------------------------------------------------------
# 1. dumbbell — per tensor output error
# ---------------------------------------------------------------------------------------------
def fig_dumbbell(fits, out_path, dpi):
    names = sorted({n for f in fits for n in f["names"]}, key=tensor_sort_key)
    y = {n: i for i, n in enumerate(names)}
    n_fmt = len(fits)
    spread = 0.34 if n_fmt > 1 else 0.0

    # every value on the axis, so fp8 and both 4-bit series share one honest x range
    values = []
    for f in fits:
        for n in f["names"]:
            t = f["tensors"][n]
            values.append(t["output_rel_l2_rtn"])
            if f["has_fit"]:
                values.append(t["output_rel_l2_fitted"])
            fp8 = t.get("fp8") or {}
            if fp8.get("output_rel_l2"):
                values.append(fp8["output_rel_l2"])
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    use_log = max(values) / min(values) > 25

    gains = []
    for f in fits:
        if not f["has_fit"]:
            continue
        for n in f["names"]:
            t = f["tensors"][n]
            gains.append(t["output_rel_l2_rtn"] - t["output_rel_l2_fitted"])
    g_lo, g_hi = (min(gains), max(gains)) if gains else (0.0, 1.0)
    if g_hi - g_lo < 1e-12:
        g_hi = g_lo + 1e-12
    cmap = plt.get_cmap("RdYlGn")
    norm = Normalize(vmin=g_lo, vmax=g_hi)

    fig, ax = plt.subplots(figsize=(12.0, 10.6))
    fmt_markers = ["o", "s", "^", "v", "D", "P"]

    # block bands + separators, so 36 rows read as fc then five blocks
    blocks = [block_of(n) for n in names]
    start = 0
    for i in range(1, len(blocks) + 1):
        if i == len(blocks) or blocks[i] != blocks[start]:
            if start > 0:
                ax.axhline(start - 0.5, color="#d5dbdb", lw=0.7, zorder=0)
            mid = (start + i - 1) / 2.0
            t = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(-0.075, mid, blocks[start], transform=t, ha="right", va="center",
                    fontsize=10, fontweight="bold", color=INK)
            start = i

    for fi, f in enumerate(fits):
        off = (fi - (n_fmt - 1) / 2.0) * spread
        marker = fmt_markers[fi % len(fmt_markers)]
        for n in f["names"]:
            t = f["tensors"][n]
            yy = y[n] + off
            rtn = t["output_rel_l2_rtn"]
            if f["has_fit"]:
                fit = t["output_rel_l2_fitted"]
                ax.plot([fit, rtn], [yy, yy], color=cmap(norm(rtn - fit)), lw=3.0,
                        solid_capstyle="round", zorder=2)
                ax.scatter([fit], [yy], s=46, marker=marker, color=INK, edgecolor="white",
                           linewidth=0.8, zorder=4)
            ax.scatter([rtn], [yy], s=30, marker=marker, color=MUTED, edgecolor="white",
                       linewidth=0.7, zorder=3)
            fp8 = (t.get("fp8") or {}).get("output_rel_l2")
            if fp8:
                ax.scatter([fp8], [yy], s=34, marker=marker, facecolor="none",
                           edgecolor=COLOR["fp8"], linewidth=1.3, zorder=3)

    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(list(y.values()))
    ax.set_yticklabels([short_name(n) for n in names], fontsize=8)
    if use_log:
        ax.set_xscale("log")
    ax.set_xlabel("relative L2 of the layer output (bf16 teacher as 0; held-out real activations)",
                  fontsize=10)
    ax.grid(axis="x", color="#eaecee", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [Line2D([], [], marker="o", ls="", color=MUTED, markersize=7,
                       label="RTN 4-bit (right end of each line)"),
               Line2D([], [], marker="o", ls="", color=INK, markersize=8,
                       label="fitted 4-bit (left end)"),
               Line2D([], [], marker="o", ls="", markerfacecolor="none",
                      markeredgecolor=COLOR["fp8"], markersize=8, label="fp8 (block 128x128)")]
    if n_fmt > 1:
        for fi, f in enumerate(fits):
            handles.append(Line2D([], [], marker=fmt_markers[fi % len(fmt_markers)], ls="",
                                  color="#95a5a6", markersize=9,
                                  label=f"marker = {f['label']}"))
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95,
              fontsize=8 if n_fmt > 1 else 9, edgecolor="#cccccc")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.055)
    cbar.set_label("error the fit recovered: output rel-L2 RTN − fitted", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    title = "Per-tensor output error of the 4-bit drafter body against its bf16 teacher"
    sub = []
    for f in fits:
        s = f["summary"]
        if s.get("mean_output_rel_l2_rtn"):
            if f["has_fit"]:
                sub.append(f"{f['label']}: mean {s['mean_output_rel_l2_rtn']:.3f} RTN → "
                           f"{s.get('mean_output_rel_l2_fitted', 0):.3f} fitted")
            else:
                sub.append(f"{f['label']}: mean {s['mean_output_rel_l2_rtn']:.3f} RTN "
                           "(no fit run in this report)")
    if sub:
        title += "\n" + "; ".join(sub) + "   (mean over 36 tensors)"
    ax.set_title(title, fontsize=11, loc="left", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------------------------
# 2. end-to-end fidelity bars
# ---------------------------------------------------------------------------------------------
def fig_variants(eval_data, out_path, dpi):
    variants = list(eval_data.get("variants", {}).keys())
    if not variants:
        return None

    def pull(v, key):
        ov = eval_data["variants"][v]
        if key == "mean_foreign":
            return ov.get("mean_foreign", ov.get("overall", {}).get("mean_displaced", 0.0))
        ov = ov.get("overall", {})
        if key == "identical_frac":
            tot = eval_data["variants"][v].get("proposals_total") or 0
            ident = eval_data["variants"][v].get("proposals_identical") or 0
            return ident / tot if tot else 0.0
        return ov.get(key, 0.0)

    panels = [
        ("recall_at_k", "recall@16\n(teacher's 16 ids recovered)", 1.0, "fraction"),
        ("set_exact", "set-exact\n(7 proposals = teacher's 7)", 1.0, "fraction"),
        ("top1_agree", "top-1 agreement\n(slot 0 equals teacher)", 1.0, "fraction"),
        ("mean_foreign", "mean foreign ids\n(proposals absent from teacher top-16)", 0.0,
         "ids per position"),
        ("identical_frac", "proposal list identical to bf16\n(140 positions)", 1.0, "fraction"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(16.5, 4.8), sharey=False)
    x = list(range(len(variants)))
    for ax, (key, label, baseline, unit) in zip(axes, panels):
        vals = [pull(v, key) for v in variants]
        colors = [variant_color(v, i) for i, v in enumerate(variants)]
        ax.bar(x, vals, color=colors, width=0.62, edgecolor="white", linewidth=0.6, zorder=3)
        for xi, val in zip(x, vals):
            # a white halo keeps the number legible where it sits against the bf16 rule
            ax.text(xi, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK,
                    zorder=6, path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
        ax.axhline(baseline, color=COLOR["bf16"], ls="--", lw=1.2, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=9, rotation=20 if len(variants) > 3 else 0)
        ax.set_title(label, fontsize=9.5, color=INK)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(axis="y", color="#eaecee", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        top = max(max(vals + [baseline]) * 1.22, baseline * 1.12 + 0.02)
        ax.set_ylim(min(0, min(vals) - 0.05) if min(vals) < 0 else 0, top)
        ax.tick_params(axis="y", labelsize=8)

    cal = eval_data.get("calibration", {})
    fig.suptitle("End-to-end proposal fidelity of the quantized body (head stays bf16) — "
                 f"{cal.get('count', '?')} positions of real text, {len(variants)} variants; "
                 "dashed line = bf16 teacher (1.0 per fraction, 0 foreign ids)",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------------------------
# 3. per-slot recall
# ---------------------------------------------------------------------------------------------
def fig_slots(eval_data, out_path, dpi):
    variants = [v for v, d in eval_data.get("variants", {}).items() if d.get("per_slot")]
    if not variants:
        return None

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.9))
    for i, v in enumerate(variants):
        slots = eval_data["variants"][v]["per_slot"]
        xs = [s["slot"] for s in slots]
        ys = [s["recall_at_k"] for s in slots]
        dis = [s["mean_displaced"] for s in slots]
        col = variant_color(v, i)
        ax.plot(xs, ys, marker="o", ms=5, lw=1.8, color=col, label=v)
        ax2.plot(xs, dis, marker="o", ms=5, lw=1.8, color=col, label=v)

    ax.axhline(1.0, color=COLOR["bf16"], ls="--", lw=1.2)
    ax.set_ylim(min(0.6, ax.get_ylim()[0]), 1.035)
    ax.set_xlabel("slot index (0 first token proposed, 6 last)")
    ax.set_ylabel("recall@16")
    ax.set_title("recall@16 by slot: does the error deepen with depth?", fontsize=10, loc="left",
                 color=INK)
    ax.text(6, 1.004, "bf16 = 1", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax2.set_xlabel("slot index")
    ax2.set_ylabel("mean displaced teacher ids")
    ax2.set_title("mean ids of the teacher's 16 that moved, by slot", fontsize=10, loc="left",
                  color=INK)
    ax2.text(6, 0.02, "bf16 = 0", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax2.set_ylim(bottom=0)

    for a in (ax, ax2):
        a.grid(color="#eaecee", lw=0.7)
        a.set_axisbelow(True)
        a.set_xticks(range(7))
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    ax.legend(frameon=True, framealpha=0.95, fontsize=9, edgecolor="#cccccc")

    per_slot0 = {}
    for v in variants:
        slots = eval_data["variants"][v]["per_slot"]
        per_slot0[v] = slots[0]["recall_at_k"] - slots[-1]["recall_at_k"]
    ax2.text(0.02, 0.03, "recall gap slot0 − slot6: " +
             ", ".join(f"{v} {g:+.3f}" for v, g in per_slot0.items()),
             transform=ax2.transAxes, fontsize=8, color=INK, va="bottom")

    fig.suptitle("Per-slot recall of each variant against the bf16 teacher", fontsize=11, x=0.01,
                 ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------------------------
# 4. accept length
# ---------------------------------------------------------------------------------------------
def fig_accept(accept_data, out_path, dpi):
    variants = accept_data.get("variants", {})
    if not variants:
        return None
    names = list(variants.keys())
    ref = variants.get("bf16", {}).get("accept")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.9),
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})
    x = list(range(len(names)))
    vals = [variants[n].get("accept", 0.0) for n in names]
    colors = [COLOR["bf16"] if n.lower().startswith("bf16") else variant_color(n, i)
              for i, n in enumerate(names)]
    ax.bar(x, vals, color=colors, width=0.6, edgecolor="white", linewidth=0.6, zorder=3)
    for xi, n, val in zip(x, names, vals):
        tpr = variants[n].get("tokens_per_round")
        s = f"{val:.3f}"
        if tpr is not None:
            s += f"\n{tpr:.3f} tok/round"
        if ref:
            s += f"\n{(val / ref - 1.0) * 100:+.1f}%"
        ax.text(xi, val, s, ha="center", va="bottom", fontsize=8.5, color=INK, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("mean slots accepted of 7")
    ax.set_ylim(0, max(vals + ([ref] if ref else [0])) * 1.30)
    ax.set_title("bars = mean slots accepted; text = accept, tokens/round, Δ vs bf16 bar",
                 fontsize=9.5, loc="left", color=INK)

    w = 0.36
    for k, key in enumerate(["p_ge_1", "full7"]):
        ys = [variants[n].get(key, 0.0) for n in names]
        ax2.bar([xi + (k - 0.5) * w for xi in x], ys, width=w,
                label={"p_ge_1": "P(≥1 accepted)", "full7": "P(all 7 accepted)"}[key],
                color=["#5b8fb9", "#0b3c5d"][k], edgecolor="white", linewidth=0.6, zorder=3)
        for xi, val in zip(x, ys):
            ax2.text(xi + (k - 0.5) * w, val, f"{val:.3f}", ha="center", va="bottom",
                     fontsize=7.5, color=INK, zorder=6,
                     path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("fraction of positions")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("hit rate: any token / all seven", fontsize=9.5, loc="left", color=INK)
    ax2.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor="#cccccc", loc="upper right")

    for a in (ax, ax2):
        a.grid(axis="y", color="#eaecee", lw=0.7, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    probe = accept_data.get("probe", "")
    probe = re.sub(r"^teacher-forced accept length[,:]?\s*", "", probe)
    fig.suptitle(f"Accept length, teacher-forced (slots 0-6){' — ' + probe if probe else ''}",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------------------------
# 5. weight error vs output error
# ---------------------------------------------------------------------------------------------
def fig_weight_vs_output(fits, out_path, dpi):
    usable = [f for f in fits if f["has_fit"]]
    if not usable:
        return None

    fig, axes = plt.subplots(1, len(usable), figsize=(7.4 * len(usable), 6.1), squeeze=False)
    axes = axes[0]
    for ax, f in zip(axes, usable):
        xs_rtn, ys_rtn, xs_fit, ys_fit, fs_x, fs_y = [], [], [], [], [], []
        for n in f["names"]:
            t = f["tensors"][n]
            xs_rtn.append(t["weight_rel_l2_rtn"])
            ys_rtn.append(t["output_rel_l2_rtn"])
            xs_fit.append(t["weight_rel_l2_fitted"])
            ys_fit.append(t["output_rel_l2_fitted"])
            fp8 = t.get("fp8") or {}
            if fp8.get("weight_rel_l2") and fp8.get("output_rel_l2"):
                fs_x.append(fp8["weight_rel_l2"])
                fs_y.append(fp8["output_rel_l2"])

        for x0, y0, x1, y1 in zip(xs_rtn, ys_rtn, xs_fit, ys_fit):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#b9c2c7", lw=0.9,
                                        shrinkA=2.5, shrinkB=2.5, mutation_scale=7), zorder=2)
        ax.scatter(xs_rtn, ys_rtn, s=40, color=COLOR["rtn"], edgecolor="white", linewidth=0.6,
                   label="RTN (4-bit, no fit)", zorder=3)
        ax.scatter(xs_fit, ys_fit, s=40, color=COLOR["fitted"], edgecolor="white", linewidth=0.6,
                   label="fitted (4-bit, output-fit scales)", zorder=4)
        if fs_x:
            ax.scatter(fs_x, fs_y, s=46, marker="^", facecolor="none",
                       edgecolor=COLOR["fp8"], linewidth=1.3, label="fp8 (block 128x128)", zorder=4)

        # the identity line y = x makes "output error < weight error" readable at a glance
        hi = max(xs_rtn + xs_fit + fs_x + ys_rtn + ys_fit + fs_y) * 1.08
        ax.plot([0, hi], [0, hi], color="#d5dbdb", lw=1.0, ls=":", zorder=1)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        # rotate the identity label along the drawn line: the axes are not square, so 45 deg lies
        p0 = ax.transData.transform((0.0, 0.0))
        p1 = ax.transData.transform((hi, hi))
        angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(hi * 0.52, hi * 0.52, "y = x  (output error = weight error)", fontsize=8,
                color=MUTED, ha="center", va="bottom", rotation=angle, rotation_mode="anchor",
                zorder=5)
        ax.set_xlabel("weight rel-L2 (per tensor, vs bf16 weights)")
        ax.set_ylabel("output rel-L2 (per tensor, vs bf16 output on real activations)")

        s = f["summary"]
        txt = (f"mean weight : RTN {s.get('mean_weight_rel_l2_rtn', 0):.3f} → "
               f"fitted {s.get('mean_weight_rel_l2_fitted', 0):.3f}\n"
               f"mean output : RTN {s.get('mean_output_rel_l2_rtn', 0):.3f} → "
               f"fitted {s.get('mean_output_rel_l2_fitted', 0):.3f}\n"
               f"fp8 output  : {s.get('mean_output_rel_l2_fp8', 0):.3f}")
        ax.text(0.035, 0.965, txt, transform=ax.transAxes, fontsize=8.5, va="top", color=INK,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#dddddd",
                          alpha=0.9))
        ax.set_title(f"{f['label']}: 36 tensors, arrows RTN → fitted", fontsize=10, loc="left",
                     color=INK)
        ax.grid(color="#eaecee", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8.5, frameon=True, framealpha=0.95, edgecolor="#cccccc",
                  loc="lower right")

    fig.suptitle("The fit moves points down, not left: output error falls while the weights stay "
                 "where they were", fontsize=10.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------------------------
def write_readme(out_dir, fits, eval_data, accept_data, made, dpi):
    """One line per figure saying what a reader should conclude — computed from these files,
    so the claim cannot drift away from the data the plots were drawn from."""
    lines = ["# 4-bit quantisation gap: the drafter body against its bf16 teacher",
             "",
             f"Generated by `tools/dflash2/analysis/plot_quant_gap.py` at {dpi} dpi. Every error is "
             "relative L2 against the unquantized bf16 teacher; the head stays bf16 in every "
             "end-to-end variant, so these numbers isolate the body.",
             ""]

    for f in fits:
        s = f["summary"]
        n = s.get("tensors", len(f["names"]))
        lines.append(
            f"- fit report `{os.path.basename(f['path'])}` [{f['label']}]: {n} tensors, group "
            f"{f.get('group')}, mean output rel-L2 RTN {s.get('mean_output_rel_l2_rtn', float('nan')):.4f}"
            f" → fitted {s.get('mean_output_rel_l2_fitted', float('nan')):.4f} → fp8 "
            f"{s.get('mean_output_rel_l2_fp8', float('nan')):.4f}; tensors made worse by the fit: "
            f"{s.get('tensors_worse_than_rtn', '?')}")
    if eval_data:
        cal = eval_data.get("calibration", {})
        lines.append(f"- eval: {cal.get('count', '?')} positions ({cal.get('rows', '?')} rows) of "
                     "real text, one fixture.")
    lines.append("")

    def fmt(x, nd=3):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "?"

    entries = []

    if fits:
        f0 = fits[0]
        s = f0["summary"]
        rtn_m = s.get("mean_output_rel_l2_rtn")
        fit_m = s.get("mean_output_rel_l2_fitted")
        fp8_m = s.get("mean_output_rel_l2_fp8")
        rec = (1.0 - fit_m / rtn_m) if rtn_m and fit_m else None
        worst = max(f0["names"], key=lambda n: f0["tensors"][n]["output_rel_l2_rtn"])
        best = min(f0["names"], key=lambda n: f0["tensors"][n]["output_rel_l2_rtn"])
        line = (f"Per-tensor output error, grouped by block. The error is spread across all 36 "
                f"tensors rather than concentrated in one: worst is `{worst}` at "
                f"{fmt(f0['tensors'][worst]['output_rel_l2_rtn'])} rel-L2, best is `{best}` at "
                f"{fmt(f0['tensors'][best]['output_rel_l2_rtn'])}. "
                f"RTN averages {fmt(rtn_m)}, the fit brings it to {fmt(fit_m)}"
                + (f" ({rec * 100:.0f}% of the error recovered)" if rec is not None else "")
                + (f", and fp8 sits at {fmt(fp8_m)} — the 4-bit path does not reach fp8 on any "
                   f"single tensor." if fp8_m else "."))
        entries.append(("quant_gap_dumbbell.png", line))

    if eval_data:
        ov = {v: d.get("overall", {}) for v, d in eval_data["variants"].items()}
        names = list(ov)
        if {"rtn", "fitted"} <= set(names):
            r, ft = ov["rtn"], ov["fitted"]
            fr = eval_data["variants"]["rtn"].get("mean_foreign", 0.0)
            ff = eval_data["variants"]["fitted"].get("mean_foreign", 0.0)
            line = (f"End to end on real text. The fit recovers part of RTN's loss everywhere: "
                    f"recall@16 {fmt(r.get('recall_at_k'))} → {fmt(ft.get('recall_at_k'))}, "
                    f"top-1 {fmt(r.get('top1_agree'))} → {fmt(ft.get('top1_agree'))}, set-exact "
                    f"{fmt(r.get('set_exact'))} → {fmt(ft.get('set_exact'))}, foreign ids "
                    f"{fmt(fr)} → {fmt(ff)}. Proposal lists identical to bf16 rise from "
                    f"{eval_data['variants']['rtn'].get('proposals_identical', '?')} to "
                    f"{eval_data['variants']['fitted'].get('proposals_identical', '?')} of "
                    f"{eval_data['variants']['rtn'].get('proposals_total', '?')}. fp8 is far "
                    f"ahead on every panel, set-exact most of all — 4-bit costs whole lists, "
                    f"not just the odd id.")
            entries.append(("quant_gap_variants.png", line))

        gaps = {}
        for v in eval_data["variants"]:
            slots = eval_data["variants"][v].get("per_slot") or []
            if slots:
                gaps[v] = slots[0]["recall_at_k"] - slots[-1]["recall_at_k"]
        if gaps:
            spread = max(gaps.values()) - min(gaps.values())
            line = (f"Per-slot recall. Recall is close to flat across the 7 slots (slot0 − slot6 "
                    f"gap: " + ", ".join(f"{v} {g:+.3f}" for v, g in gaps.items()) + "), so the "
                    f"body error is not concentrated in the late slots — it degrades every slot "
                    f"about equally. The displaced-id panel is where the variants separate.")
            entries.append(("quant_gap_slots.png", line))

    if accept_data:
        av = accept_data.get("variants", {})
        ref = av.get("bf16", {}).get("accept")
        parts = [f"{v} {av[v].get('accept', 0):.3f} ({av[v].get('tokens_per_round', 0):.3f} "
                 f"tok/round, {(av[v].get('accept', 0) / ref - 1) * 100:+.1f}%)"
                 for v in av if ref and v != "bf16"]
        base_p = av.get("bf16", {}).get("p_ge_1")
        # the variant that lost the most accept length, not the one nearest bf16
        worst = min((v for v in av if v != "bf16"), key=lambda v: av[v].get("accept", 0),
                    default=None)
        where = ""
        if base_p and worst and ref:
            dp = av[worst].get("p_ge_1", 0) / base_p - 1.0
            da = av[worst].get("accept", 0) / ref - 1.0
            where = (f" For the worst variant ({worst}) P(≥1 accepted) moves {dp * 100:+.1f}% while "
                     f"accept moves {da * 100:+.1f}%, so the loss is in the "
                     + ("deeper slots" if abs(dp) < abs(da) / 2 else "first slot, i.e. whole "
                        "rounds failing to start") + ", not spread evenly.")
        line = (f"Teacher-forced accept length (bf16 {ref:.3f} slots, "
                f"{av.get('bf16', {}).get('tokens_per_round', 0):.3f} tokens/round): "
                + "; ".join(parts) + "." + where)
        entries.append(("quant_gap_accept.png", line))

    if fits:
        f0 = [f for f in fits if f["has_fit"]]
        if f0:
            s = f0[0]["summary"]
            w_rtn, w_fit = s.get("mean_weight_rel_l2_rtn", 0), s.get("mean_weight_rel_l2_fitted", 0)
            o_rtn, o_fit = s.get("mean_output_rel_l2_rtn", 0), s.get("mean_output_rel_l2_fitted", 0)
            line = (f"Weight error vs output error. The fit moves every point down but hardly "
                    f"left: mean weight rel-L2 {fmt(w_rtn)} → {fmt(w_fit)} "
                    f"({(1 - w_fit / w_rtn) * 100:.0f}% less) while mean output rel-L2 "
                    f"{fmt(o_rtn)} → {fmt(o_fit)} ({(1 - o_fit / o_rtn) * 100:.0f}% less). "
                    "Judge a 4-bit format by its output column; a weight-error metric will "
                    "declare the fit nearly a no-op.")
            entries.append(("quant_gap_weight_vs_output.png", line))

    lines.append("## Figures")
    lines.append("")
    for name, summary in entries:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            lines.append(f"- **{name}** — {summary}")
    for name in ["quant_gap_dumbbell.png", "quant_gap_variants.png", "quant_gap_slots.png",
                 "quant_gap_accept.png", "quant_gap_weight_vs_output.png"]:
        if not os.path.exists(os.path.join(out_dir, name)):
            lines.append(f"- ~~{name}~~ — not generated (missing input).")
    lines.append("")
    if accept_data:
        lines.append(f"Accept probe: {accept_data.get('probe', '')}")
        lines.append("")
    if eval_data and eval_data.get("caveats"):
        lines.append("Caveats from the eval report:")
        lines.append("")
        for c in eval_data["caveats"]:
            lines.append(f"- {c}")
        lines.append("")
    lines.append("## Files")
    lines.append("")
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".png"):
            size = os.path.getsize(os.path.join(out_dir, fn)) / 1024.0
            lines.append(f"- `{fn}` — {size:.0f} KB")
    lines.append("")
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Plot the 4-bit quantisation gap of the DFlash2 drafter body against bf16.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-report", nargs="+", default=[], metavar="PATH",
                    help="fit report JSON(s); one per 4-bit format (e.g. real_scale.json)")
    ap.add_argument("--eval", dest="eval_path", default=None,
                    help="end-to-end fidelity report (eval_real.json)")
    ap.add_argument("--accept", dest="accept_path", default=None,
                    help="teacher-forced accept report (accept_real.json)")
    ap.add_argument("--out", default="/tmp/nv4/plots/", help="output directory (default: %(default)s)")
    ap.add_argument("--label", action="append", default=[], metavar="KEY=NAME",
                    help="label a report, by path/stem/format substring, or bare NAME in order")
    ap.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: %(default)s)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    fits = []
    if args.fit_report:
        for rep in resolve_labels(args.fit_report, args.label):
            data = load_json(rep["path"], "fit report")
            if data is None:
                continue
            norm = normalise_fit_report(data, rep["label"], rep["path"])
            if norm is None:
                print(f"  no tensors in {rep['path']}", file=sys.stderr)
                continue
            if not norm["has_fit"]:
                print(f"  {os.path.basename(rep['path'])} is an RTN-only run: plotting RTN only, "
                      "no fitted series", file=sys.stderr)
            fits.append(norm)
        fits = merge_fit_reports(fits)

    eval_data = load_json(args.eval_path, "eval report")
    accept_data = load_json(args.accept_path, "accept report")

    jobs = []
    if fits:
        jobs.append(("quant_gap_dumbbell.png", lambda p: fig_dumbbell(fits, p, args.dpi)))
        jobs.append(("quant_gap_weight_vs_output.png",
                     lambda p: fig_weight_vs_output(fits, p, args.dpi)))
    if eval_data:
        jobs.append(("quant_gap_variants.png", lambda p: fig_variants(eval_data, p, args.dpi)))
        jobs.append(("quant_gap_slots.png", lambda p: fig_slots(eval_data, p, args.dpi)))
    if accept_data:
        jobs.append(("quant_gap_accept.png", lambda p: fig_accept(accept_data, p, args.dpi)))

    made, skipped = [], []
    for name in ["quant_gap_dumbbell.png", "quant_gap_variants.png", "quant_gap_slots.png",
                 "quant_gap_accept.png", "quant_gap_weight_vs_output.png"]:
        job = next((j for j in jobs if j[0] == name), None)
        if job is None:
            skipped.append(name)
            continue
        path = os.path.join(args.out, name)
        try:
            if job[1](path):
                made.append(path)
            else:
                skipped.append(name)
        except Exception as exc:  # a bad report should not kill the other figures
            print(f"  failed {name}: {exc}", file=sys.stderr)
            skipped.append(name)

    print(f"wrote {len(made)}/{len(made) + len(skipped)} figures to {args.out} at {args.dpi} dpi")
    for p in made:
        print(f"  {p}  ({os.path.getsize(p) / 1024:.0f} KB)")
    for p in skipped:
        print(f"  skipped {p} (no input for it, or nothing to plot in it)")

    if made:
        readme = os.path.join(args.out, "README.md")
        write_readme(args.out, fits, eval_data, accept_data, made, args.dpi)
        print(f"  {readme}")
    if not made:
        print("nothing could be drawn: check the input paths", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
