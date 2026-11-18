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

Two captures compare instead of profiling one — every panel then carries both arms and the delta:

  A  categories, both arms         what the round is made of, before vs after
  B  per stage, dumbbell           each stage's baseline dot against its candidate dot
  C  the round, before and after   served and cold, with the delta in ms and %
  D  cumulative delta in order     where along the round the delta is actually won

  .venv-plot/bin/python tools/dflash2/plot_stages.py CAPTURE --baseline BASE
                                                       [--label X] [--label-b Y] [--out PNG]
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


AB_A = "#7f8c8d"     # baseline arm
AB_B = "#16a085"     # candidate arm
GAIN = "#1e8449"
LOSS = "#c0392b"


def _delta_text(b, c):
    d = c - b
    return f"{b:.2f} → {c:.2f}  {d:+.2f} ({100 * d / b:+.1f}%)" if b else f"{b:.2f} → {c:.2f}"


def _bar_labels(ax, ys, vals, offset, fmt, colour=None, fontsize=7.2):
    for y, v in zip(ys, vals):
        ax.text(v + offset, y, fmt(v), va="center", fontsize=fontsize,
                color=colour or "black")


def panel_cat_ab(ax, base, cand, la, lb):
    """Categories, both arms — the same roll-up panel B does, with the delta on each row."""
    rb, rc = rollup(base), rollup(cand)
    cats = sorted(rb, key=lambda c: -rb[c][0])
    ys = range(len(cats))
    vb = [rb[c][0] for c in cats]
    vc = [rc.get(c, (0.0, ""))[0] for c in cats]
    ax.barh([y + 0.19 for y in ys], vb, height=0.36, color=AB_A,
            edgecolor="black", linewidth=0.4, label=la)
    ax.barh([y - 0.19 for y in ys], vc, height=0.36, color=AB_B,
            edgecolor="black", linewidth=0.4, label=lb)
    for y, (b, c) in enumerate(zip(vb, vc)):
        d = c - b
        ax.text(max(b, c) * 1.06, y, _delta_text(b, c), va="center", fontsize=7.4,
                color=GAIN if d < 0 else LOSS)
    tb, tc = sum(vb), sum(vc)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{c}  ({100 * rb[c][0] / tb:.1f}% → {100 * rc.get(c, (0,))[0] / tc:.1f}%)"
                        for c in cats], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("median ms over served reps (log scale)")
    ax.set_title(f"A · categories — {la} vs {lb}", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.009, max(vb + vc) * 40)      # the delta text is longer than the bars
    ax.grid(axis="x", alpha=0.25, which="both")


def panel_stage_ab(ax, base, cand, la, lb, top):
    """Per-stage dumbbell: one row a stage, a line from its baseline ms to its candidate ms."""
    tb, tc = totals(base), totals(cand)
    names = sorted(tb, key=lambda n: -tb[n])[:top]
    ys = list(range(len(names)))[::-1]
    for y, n in zip(ys, names):
        b = tb[n]
        c = tc.get(n, b)
        ax.plot([b, c], [y, y], color="#bdc3c7", linewidth=1.4, zorder=1)
        ax.scatter([b], [y], s=26, color=AB_A, zorder=3)
        ax.scatter([c], [y], s=26, color=AB_B, zorder=3)
        ax.annotate(f"{c - b:+.3f}", (max(b, c), y), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=7.2,
                    color=GAIN if c < b else LOSS)
    ax.set_yticks(ys)
    ax.set_yticklabels([short(n) for n in names], fontsize=8, family="monospace")
    ax.set_xscale("log")
    ax.set_xlim(max(0.004, min(tb[n] for n in names) * 0.5), max(tc.get(n, 0) for n in names) * 9)
    ax.set_xlabel("median ms over served reps (log scale)")
    ax.set_title(f"B · per stage, top {top} by {la} — dot is {la}, dot is {lb}", loc="left", fontsize=11)
    ax.scatter([], [], s=26, color=AB_A, label=la)
    ax.scatter([], [], s=26, color=AB_B, label=lb)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, which="both")


def panel_round_ab(ax, rb_warm, rb_cold, rc_warm, rc_cold, la, lb):
    """The two data points he asked for, at round level: served and cold, both arms, with delta.

    Log y: the cold round is ~15x the served one, so on a linear axis the served pair — the pair
    that sets throughput — would be two invisible stubs next to it.
    """
    groups = []
    if rb_cold is not None and rc_cold is not None:
        groups.append(("cold round\n(first round of a stream)", rb_cold, rc_cold))
    groups.append(("served round\n(median of reps 1+)", rb_warm, rc_warm))
    x = list(range(len(groups)))
    w = 0.30
    for n, (x0, (name, b, c)) in enumerate(zip(x, groups)):
        ax.bar(x0 - w / 2, b, width=w, color=AB_A, edgecolor="black", linewidth=0.4,
               label=la if n == 0 else None)
        ax.bar(x0 + w / 2, c, width=w, color=AB_B, edgecolor="black", linewidth=0.4,
               label=lb if n == 0 else None)
        ax.text(x0 - w / 2, b * 1.06, f"{b:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(x0 + w / 2, c * 1.06, f"{c:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(x0, max(b, c) * 1.9, f"{c - b:+.2f} ms\n{100 * (c - b) / b:+.1f}%",
                ha="center", va="bottom", fontsize=9.5,
                color=GAIN if c < b else LOSS,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#7f8c8d", lw=0.6, alpha=0.92))
    lo = min(min(g[1], g[2]) for g in groups)
    hi = max(max(g[1], g[2]) for g in groups)
    ax.set_yscale("log")
    ax.set_ylim(lo * 0.55, hi * 6)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("round ms (log scale)")
    ax.set_title("C · the round — before and after", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25, which="both")


def panel_delta_order(ax, base, cand, order, la, lb):
    """Where the saving is won, walking the round: cumulative (candidate - baseline) ms."""
    seq = [s for s in order if s in base and s not in HOST and s not in MIRRORS]
    d = [cand.get(s, base[s]) - base[s] for s in seq]
    cum = []
    run = 0.0
    for v in d:
        run += v
        cum.append(run)
    ax.step(range(len(seq)), cum, where="mid", color="#2c3e50", linewidth=1.6)
    ax.fill_between(range(len(seq)), 0, cum, step="mid", color=GAIN if cum[-1] < 0 else LOSS, alpha=0.10)
    ax.axhline(0, color="black", linewidth=0.8)
    # the four biggest single moves, labelled — they are the whole story of the delta
    for i in sorted(range(len(d)), key=lambda i: -abs(d[i]))[:4]:
        ax.annotate(f"{short(seq[i])} {d[i]:+.2f}", (i, cum[i]), textcoords="offset points",
                    xytext=(4, 10 if d[i] > 0 else -14), fontsize=7.6, family="monospace",
                    color=LOSS if d[i] > 0 else GAIN,
                    arrowprops=dict(arrowstyle="-", color="#7f8c8d", linewidth=0.6,
                                    shrinkA=0, shrinkB=2))
    # +1, so the ticks do not land on the same stage of every layer: the round is periodic, and a
    # step equal to the layer period labels the axis with five copies of block_hidden.
    step = max(1, len(seq) // 5) + 1
    ax.set_xticks(range(0, len(seq), step))
    ax.set_xticklabels([short(seq[i]) for i in range(0, len(seq), step)],
                       rotation=30, ha="right", fontsize=8, family="monospace")
    ax.set_xlim(-1, len(seq))
    ax.set_ylabel(f"cumulative {lb} − {la} (ms)")
    ax.set_title(f"D · where the delta is won, in execution order — {cum[-1]:+.2f} ms", loc="left",
                 fontsize=11)
    ax.grid(alpha=0.25)


def compare(a):
    """Two captures, one figure: `--baseline` is BEFORE, `capture` is AFTER, delta = after − before."""
    after, before = load(a.capture), load(a.baseline)
    l_after = a.label or os.path.basename(os.path.normpath(a.capture))
    l_before = a.label_b or os.path.basename(os.path.normpath(a.baseline))

    def split(reps):
        cr = a.cold_rep if any(i == a.cold_rep for i, _, _, _ in reps) else -1
        warm = [r for r in reps if r[0] != cr] or reps
        cold = [r for r in reps if r[0] == cr]
        return warm, cold

    bw, bc = split(before)
    aw, ac = split(after)
    sb, sa = medians(bw), medians(aw)
    cb = medians(bc) if bc else None
    ca = medians(ac) if ac else None
    order = [s["stage"] for s in after[0][3]["stages"]]

    tb, ta = sum(totals(sb).values()), sum(totals(sa).values())
    mb = statistics.median([r["round_ms"] for _, r, _, _ in bw])
    ma = statistics.median([r["round_ms"] for _, r, _, _ in aw])
    print(f"{'before':>12s}  {l_before}   {a.baseline}  ({len(bw)} served reps)")
    print(f"{'after':>12s}  {l_after}   {a.capture}  ({len(aw)} served reps)")
    print(f"\n{'stage':26s}{l_before:>10s}{l_after:>10s}{'delta':>10s}{'pct':>9s}")
    for n in sorted(totals(sb), key=lambda n: -sb[n]):
        x, y = sb[n], sa.get(n, 0.0)
        print(f"{n:26s}{x:10.3f}{y:10.3f}{y - x:+10.3f}{100 * (y - x) / x:+8.1f}%")
    print(f"{'TOTAL':26s}{tb:10.3f}{ta:10.3f}{ta - tb:+10.3f}{100 * (ta - tb) / tb:+8.1f}%")
    print(f"\n{'round':26s}{l_before:>10s}{l_after:>10s}{'delta':>10s}{'pct':>9s}")
    print(f"{'served median':26s}{mb:10.3f}{ma:10.3f}{ma - mb:+10.3f}{100 * (ma - mb) / mb:+8.1f}%")
    if bc and ac:
        x, y = bc[0][1]["round_ms"], ac[0][1]["round_ms"]
        print(f"{'cold (rep 0)':26s}{x:10.3f}{y:10.3f}{y - x:+10.3f}{100 * (y - x) / x:+8.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5))
    panel_cat_ab(axes[0][0], sb, sa, l_before, l_after)
    panel_stage_ab(axes[0][1], sb, sa, l_before, l_after, a.top)
    panel_round_ab(axes[1][0], mb, (bc[0][1]["round_ms"] if bc else None),
                   ma, (ac[0][1]["round_ms"] if ac else None), l_before, l_after)
    panel_delta_order(axes[1][1], sb, sa, order, l_before, l_after)
    fig.suptitle(f"DFlash2 drafter — {l_before} → {l_after} on the served fixture"
                 f"  ·  round {mb:.2f} → {ma:.2f} ms ({100 * (ma - mb) / mb:+.1f}%)"
                 f"  ·  stage sum {tb:.2f} → {ta:.2f} ms", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = a.out or "compare.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    if a.show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="a capture root with rep*/stages.json, or one stages.json")
    ap.add_argument("--baseline", default=None,
                    help="a second capture to compare against; the figure then shows both arms "
                         "and the delta instead of a single-arm profile")
    ap.add_argument("--label", default=None, help="name for CAPTURE in the comparison")
    ap.add_argument("--label-b", default=None, help="name for --baseline in the comparison")
    ap.add_argument("--out", default=None, help="PNG to write (default <capture>/stages.png)")
    ap.add_argument("--cold-rep", type=int, default=0, help="the rep to treat as cold (-1 = none)")
    ap.add_argument("--top", type=int, default=20, help="stages in panel A (default 20)")
    ap.add_argument("--include-host", action="store_true", help="count the host-only stages too")
    ap.add_argument("--show", action="store_true", help="also open an interactive window")
    a = ap.parse_args()

    if a.baseline:
        return compare(a)

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
