#!/usr/bin/env python3
"""Plot a DFlash2 stage capture: where a served drafter round actually goes.

Reads what `tensorfold draft-check` (or the engine's fixture harness) writes — a capture root with
`rep{i}/stages.json`, or one `stages.json` — and draws four views of the same numbers:

  A  the stages, ranked            which single stages are the round
  B  categories, served vs cold    what the round is made of, and what the cold round inflates
  C  the round in execution order  the hot path as the GPU walks it, as a cumulative curve
  D  every rep                     cold vs served, and how much the numbers wobble between reps

The distinction that matters: rep 0 of a `fixture/sequence` capture is the COLD round — it fuses and
projects the whole window — and reps 1+ are the served ones. A single-position fixture only ever
produces cold rounds, so its ranking is of a regime the box serves once per stream.

`selector_scores` is a host recomputation the harness adds after the round, not GPU work; it is
excluded from every total unless `--include-host` says otherwise.

  .venv-plot/bin/python tools/dflash2/plot_stages.py CAPTURE [--out PNG] [--cold-rep 0]
                                                       [--top 20] [--include-host] [--show]
"""

import argparse
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# One colour a category, so panel A and panel B agree and the eye can carry a colour across them.
CATEGORY = [
    ("fc_ctx", "context fuse", "#c0392b"),
    ("logits_top16_ids", "head (gemm+topk)", "#d35400"),
    ("attn_proj_", "attn proj", "#f39c12"),
    ("attn_", "attention", "#e67e22"),
    ("qkv_", "qkv", "#f1c40f"),
    ("mlp_", "mlp", "#27ae60"),
    ("conv_", "convs", "#16a085"),
    ("block_hidden_", "block residual", "#2980b9"),
    ("final_norm_block", "norms", "#8e44ad"),
    ("logits_top16_vals", "head (mirror)", "#95a5a6"),
    ("logits_full", "head (mirror)", "#95a5a6"),
    ("proposals", "selector walk", "#34495e"),
    ("selector_scores", "selector (host)", "#7f8c8d"),
]
HOST = {"selector_scores"}          # computed on the CPU after the round; not part of round_ms
MIRRORS = {"logits_top16_vals", "logits_full"}   # same clock as logits_top16_ids, ~0 ms of their own


def category(stage):
    """The first rule that matches, in CATEGORY order — so `attn_proj_0` is not attention."""
    for prefix, name, colour in CATEGORY:
        if stage == prefix or stage.startswith(prefix):
            return name, colour
    return "other", "#bdc3c7"


def short(stage):
    """A tick label that fits: the waterfall's x axis has ~50 stages and no room for full names."""
    for prefix, abbrev in (("conv_attn_in_", "cai"), ("conv_attn_out_", "cao"),
                           ("conv_mlp_in_", "cmi"), ("conv_mlp_out_", "cmo"),
                           ("attn_proj_", "ap"), ("block_hidden_", "bh"),
                           ("logits_top16_ids", "head"), ("final_norm_block", "fnorm"),
                           ("block_residual", "resid"), ("context_fuse", "fuse")):
        if stage.startswith(prefix):
            return abbrev + stage[len(prefix):]
    return stage


def load(capture):
    """[(rep_index, round_dict, {stage: ms})], in rep order, from a root or a single stages.json."""
    if os.path.isfile(capture):
        roots = [(0, capture)]
    else:
        reps = [d for d in os.listdir(capture) if d.startswith("rep") and d[3:].isdigit()]
        roots = sorted(((int(d[3:]), os.path.join(capture, d, "stages.json")) for d in reps),
                       key=lambda t: t[0])
    if not roots:
        raise SystemExit(f"{capture}: no stages.json found")
    out = []
    for i, path in roots:
        with open(path) as f:
            j = json.load(f)
        out.append((i, j["round"], {s["stage"]: s["ms"] for s in j["stages"]}, j))
    return out


def medians(reps):
    """{stage: median ms} over the reps given."""
    names = {n for _, _, st, _ in reps for n in st}
    return {n: statistics.median([st.get(n, 0.0) for _, _, st, _ in reps]) for n in names}


def totals(per_stage, include_host=False):
    return {k: v for k, v in per_stage.items()
            if include_host or (k not in HOST and k not in MIRRORS)}


def rollup(per_stage, include_host=False):
    acc = {}
    for name, ms in totals(per_stage, include_host).items():
        cat, colour = category(name)
        e = acc.setdefault(cat, [0.0, colour])
        e[0] += ms
    return {k: (v[0], v[1]) for k, v in acc.items()}


def panel_ranked(ax, served, top):
    items = sorted(totals(served).items(), key=lambda kv: -kv[1])[:top][::-1]
    total = sum(totals(served).values())
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colours = [category(k)[1] for k, _ in items]
    ax.barh(range(len(items)), vals, color=colours, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=8, family="monospace")
    for i, v in enumerate(vals):
        ax.text(v + total * 0.004, i, f"{v:.2f} ms  {100 * v / total:.1f}%",
                va="center", fontsize=7.5)
    ax.set_xlabel("median ms over served reps")
    ax.set_title(f"A · stages, ranked (top {top} of {len(totals(served))})", loc="left", fontsize=11)
    ax.set_xlim(0, max(vals) * 1.30 if vals else 1)
    ax.grid(axis="x", alpha=0.25)


def panel_categories(ax, served, cold):
    served_r, cold_r = rollup(served), rollup(cold) if cold else {}
    cats = sorted(served_r, key=lambda c: -served_r[c][0])
    ys = range(len(cats))
    sv = [served_r[c][0] for c in cats]
    ax.barh([y + 0.20 for y in ys], sv, height=0.38,
            color=[served_r[c][1] for c in cats], edgecolor="black", linewidth=0.4,
            label="served round")
    cv = []
    if cold:
        cv = [cold_r.get(c, (0.0, ""))[0] for c in cats]
        ax.barh([y - 0.20 for y in ys], cv, height=0.38, color="none",
                edgecolor=[served_r[c][1] for c in cats], hatch="///", linewidth=0.8,
                label="cold round (rep 0)")
        for y, (s, c) in enumerate(zip(sv, cv)):
            if c > 0.01:
                ax.text(max(s, c) * 1.02, y + 0.20, f"{s:.2f}", va="center", fontsize=7.5)
    else:
        for y, s in enumerate(sv):
            ax.text(s * 1.02, y + 0.20, f"{s:.2f}", va="center", fontsize=7.5)
    total = sum(sv)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{c}  ({100 * served_r[c][0] / total:.1f}%)" for c in cats], fontsize=9)
    ax.set_xscale("log")
    # the multiplier in the label is measured, not asserted: it moved 60x -> 268x when the context
    # fuse stopped re-reading its weight per row, and a stale "up to 60x" would be a lie in a figure
    mult = max((c / s for s, c in zip(sv, cv) if s > 0.01), default=1.0) if cold else 1.0
    ax.set_xlabel(f"median ms (log scale — the cold round is up to {mult:.0f}x the served one)")
    ax.set_title("B · what the round is made of", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, which="both")


def panel_waterfall(ax, served, order):
    """Cumulative ms as the round is executed, with the biggest single jumps marked."""
    seq = [s for s in order if s in served and s not in HOST and s not in MIRRORS]
    vals = [served[s] for s in seq]
    cum = []
    run = 0.0
    for v in vals:
        run += v
        cum.append(run)
    ax.plot(range(len(seq)), cum, color="#2c3e50", linewidth=1.6, marker="o", markersize=2.5)
    ax.fill_between(range(len(seq)), 0, cum, color="#2c3e50", alpha=0.08)
    # the biggest steps get a label, because they are the hot path. They alternate above and below
    # the curve: two labels on one side collide whenever the steps are neighbours, which the two
    # largest ones always are.
    steps = sorted(range(len(vals)), key=lambda i: -vals[i])[:4]
    for n, i in enumerate(steps):
        up = n % 2 == 0
        ax.annotate(f"{seq[i]} +{vals[i]:.2f}", (i, cum[i]), textcoords="offset points",
                    xytext=(2, 14 if up else -20), fontsize=8, family="monospace",
                    color="#7b241c", ha="right" if i > len(seq) * 0.6 else "left",
                    arrowprops=dict(arrowstyle="-", color="#c0392b", linewidth=0.6,
                                    shrinkA=0, shrinkB=2))
    step = max(1, len(seq) // 6)
    ax.set_xticks(range(0, len(seq), step))
    ax.set_xticklabels([short(seq[i]) for i in range(0, len(seq), step)],
                       rotation=38, ha="right", fontsize=7.2, family="monospace")
    ax.set_ylim(0, cum[-1] * 1.18)
    ax.set_ylabel("cumulative ms")
    ax.set_title(f"C · the round in execution order — {cum[-1]:.2f} ms total", loc="left", fontsize=11)
    ax.grid(alpha=0.25)


def panel_reps(ax, all_reps, cold_rep):
    xs = [i for i, _, _, _ in all_reps]
    ys = [r["round_ms"] for _, r, _, _ in all_reps]
    served = [y for x, y in zip(xs, ys) if x != cold_rep]
    cold = [y for x, y in zip(xs, ys) if x == cold_rep]
    # the served reps share the axis, the cold round does not: it is 17x them, and drawing it to scale
    # would hide the very wobble this panel exists to show. It is clipped at the top and labelled.
    top = max(served) * 1.30 if served else max(ys) * 1.30
    bars = ax.bar(xs, [min(y, top) for y in ys],
                  color=["#c0392b" if x == cold_rep else "#2980b9" for x in xs],
                  edgecolor="black", linewidth=0.4, width=0.75)
    if cold:
        ax.text(cold_rep + 0.55, top * 0.93,
                f"cold rep: {cold[0]:.1f} ms\n{cold[0] / statistics.median(served):.1f}x a served round",
                ha="left", va="top", fontsize=8.5, color="#7b241c")
        bars[0].set_clip_on(True)
    if served:
        m, sd = statistics.median(served), statistics.pstdev(served)
        ax.axhline(m, color="black", linewidth=1.0, linestyle="--")
        ax.axhspan(m - sd, m + sd, color="black", alpha=0.10)
        ax.text(xs[-1], top * 0.97, f"served median {m:.2f} ms, sd {sd:.2f}",
                va="top", ha="right", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#7f8c8d", alpha=0.9, lw=0.6))
    ax.set_ylim(0, top)
    ax.set_xlabel("rep (0 = cold round on a fresh state)")
    ax.set_ylabel("round ms")
    ax.set_title("D · every rep — served axis (cold round clipped, labelled)", loc="left", fontsize=11)
    ax.grid(axis="y", alpha=0.25)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="a capture root with rep*/stages.json, or one stages.json")
    ap.add_argument("--out", default=None, help="PNG to write (default <capture>/stages.png)")
    ap.add_argument("--cold-rep", type=int, default=0, help="the rep to treat as cold (-1 = none)")
    ap.add_argument("--top", type=int, default=20, help="stages in panel A (default 20)")
    ap.add_argument("--include-host", action="store_true", help="count the host-only stages too")
    ap.add_argument("--show", action="store_true", help="also open an interactive window")
    a = ap.parse_args()

    all_reps = load(a.capture)
    cold_rep = a.cold_rep if any(i == a.cold_rep for i, _, _, _ in all_reps) else -1
    served_reps = [r for r in all_reps if r[0] != cold_rep]
    if not served_reps:                      # a single-round capture is all we have
        served_reps = all_reps
        cold_rep = -1

    served = medians(served_reps)
    cold = medians([r for r in all_reps if r[0] == cold_rep]) if cold_rep >= 0 else None
    order = [s["stage"] for s in all_reps[0][3]["stages"]]     # the collector's own order

    # ---- the same numbers, in text, so the figure is never the only record
    total = sum(totals(served).values())
    print(f"capture   {a.capture}")
    print(f"reps      {len(all_reps)}  (served {len(served_reps)}"
          + (f", cold rep {cold_rep}" if cold_rep >= 0 else "") + ")")
    print(f"served round  median {statistics.median([r['round_ms'] for _, r, _, _ in served_reps]):.2f} ms"
          + (f"   cold {[r['round_ms'] for i, r, _, _ in all_reps if i == cold_rep][0]:.2f} ms"
             if cold_rep >= 0 else ""))
    print(f"\n{'stage':24s}{'served ms':>10s}{'share':>8s}"
          + (f"{'cold ms':>10s}{'cold/served':>13s}" if cold else ""))
    for name, ms in sorted(totals(served).items(), key=lambda kv: -kv[1]):
        row = f"{name:24s}{ms:10.2f}{100 * ms / total:7.1f}%"
        if cold:
            c = cold.get(name, 0.0)
            row += f"{c:10.2f}{(c / ms if ms else 0):12.1f}x"
        print(row)
    print(f"{'TOTAL':24s}{total:10.2f}")
    if any(n in served for n in HOST):
        print(f"(excluded: {', '.join(f'{n} {served[n]:.2f} ms host' for n in HOST if n in served)})")
    print("\ncategories")
    for cat, (ms, _) in sorted(rollup(served).items(), key=lambda kv: -kv[1][0]):
        print(f"  {cat:18s}{ms:8.2f} ms{100 * ms / total:7.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5))
    panel_ranked(axes[0][0], served, a.top)
    panel_categories(axes[0][1], served, cold)
    panel_waterfall(axes[1][0], served, order)
    panel_reps(axes[1][1], all_reps, cold_rep)
    fig.suptitle("DFlash2 drafter — stage profile of a served round"
                 f"  ·  {os.path.basename(os.path.normpath(a.capture))}"
                 f"  ·  {len(served_reps)} served reps", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = a.out or os.path.join(a.capture, "stages.png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
