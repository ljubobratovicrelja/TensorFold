#!/usr/bin/env python3
"""Per-prompt arm comparison: how each path stacks up task by task, not on the mean.

The mean in the plan's §1 table hides the thing that decides whether a drafter is worth serving:
which prompts it wins. Speculative decoding pays per round and gets paid per accepted token, so an
arm can be ahead on average and behind on most prompts, or ahead everywhere by a little.

Three views of the same 16 measurements per arm, plus who wins each prompt:

  A  decode rate per prompt          tokens/s, one line an arm
  B  accept length per prompt        tokens committed per round (what speculation bought)
  C  slot acceptance per prompt      accepted / drafted — the drafter's hit rate on what it offered
  D  wins per prompt                 how many of the 16 prompts each arm is fastest on

Every prompt is the same text and the same seed in every arm, and verification is lossless, so the
completion token count is identical across arms: the x axis is a prompt index, and `tokens` is
printed per prompt so a reader can weight prompts by length instead of treating a 109-token answer
the same as a 900-token one.

  .venv-plot/bin/python tools/dflash2/harness/plot_arms.py mpl=rows-mtp.jsonl bf16=rows-bf16.jsonl \
      fp8=rows-fp8.jsonl --out arms.png
"""

import argparse
import json
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# one colour an arm, used by every panel so the eye can carry it across
COLOUR = {"mtp": "#2980b9", "bf16": "#7f8c8d", "fp8": "#16a085",
          "ar": "#c0392b", "df2-bf16": "#7f8c8d", "df2-fp8": "#16a085"}


def load(spec):
    name, path = spec.split("=", 1)
    rows = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("accept_len") and r.get("drafted"):
            rows[r["i"]] = r
    if not rows:
        sys.exit(f"{path}: no usable rows (need accept_len and drafted)")
    return name, rows


def series(arm_rows, key, idx):
    """The metric for every prompt, in prompt order; None where an arm has no row."""
    return [arm_rows[i][key] if i in arm_rows else None for i in idx]


def panel(ax, idx, arms, key, label, title, fmt="{:.2f}", zero=None):
    for name, rows in arms.items():
        ys = series(rows, key, idx)
        xs = [i for i, y in zip(idx, ys) if y is not None]
        vy = [y for y in ys if y is not None]
        ax.plot(xs, vy, marker="o", markersize=3.2, linewidth=1.1,
                color=COLOUR.get(name, None), label=name)
    if zero is not None:
        ax.axhline(zero, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("prompt index (same text and seed in every arm)")
    ax.set_ylabel(label)
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=len(arms))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="name=rows.jsonl, one per arm")
    ap.add_argument("--out", default="arms.png")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    pairs = [load(s) for s in a.arms]
    arms = dict(pairs)
    idx = sorted({i for _, rows in pairs for i in rows})

    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5))
    panel(axes[0][0], idx, arms, "tps", "tokens/s", "A · decode rate per prompt")
    panel(axes[0][1], idx, arms, "accept_len", "tokens per round",
          "B · accept length per prompt")
    for name, rows in arms.items():
        for i in rows:
            rows[i]["accept_rate"] = rows[i]["accepted"] / rows[i]["drafted"]
    panel(axes[1][0], idx, arms, "accept_rate", "accepted / drafted",
          "C · slot acceptance per prompt", fmt="{:.3f}")

    # D: who is fastest, prompt by prompt — the question "does it stack up across tasks"
    ax = axes[1][1]
    wins = {name: 0 for name, _ in pairs}
    ties = 0
    for i in idx:
        best = [(name, rows[i]["tps"]) for name, rows in arms.items() if i in rows]
        if not best:
            continue
        top = max(v for _, v in best)
        leaders = [n for n, v in best if abs(v - top) < 1e-9]
        if len(leaders) > 1:
            ties += 1
        for n in leaders:
            wins[n] += 1
    names = list(wins)
    vals = [wins[n] for n in names]
    bars = ax.barh(names, vals, color=[COLOUR.get(n, "#bdc3c7") for n in names],
                   edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v}/{len(idx)}",
                va="center", fontsize=9)
    ax.set_xlim(0, len(idx) * 1.15)
    ax.set_xlabel(f"prompts where the arm was fastest (ties counted for each: {ties})")
    ax.set_title("D · who wins each prompt", loc="left", fontsize=11)
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle("ARM A/B/C — " + ", ".join(f"{n} ({len(r)})" for n, r in pairs)
                 + "  ·  per-prompt, same text and seed", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(a.out, dpi=130)

    # ---- the same data as text: the figure must never be the only record
    hdr = f"{'i':>3}{'tokens':>8}" + "".join(f"{n + ' tok/s':>14s}" for n, _ in pairs)
    hdr += "".join(f"{n + ' acc':>12s}" for n, _ in pairs)
    hdr += "".join(f"{n + ' rate':>13s}" for n, _ in pairs)
    print(hdr)
    for i in idx:
        tok = next((arms[n][i]["tokens"] for n, _ in pairs if i in arms[n]), None)
        line = f"{i:3d}{tok if tok else 0:8d}"
        for key in ("tps", "accept_len", "accept_rate"):
            for n, _ in pairs:
                r = arms[n].get(i)
                v = r[key] if r else None
                line += f"{v:14.3f}" if key != "tps" and v is not None else \
                        (f"{v:14.1f}" if v is not None else f"{'-':>14s}")
        print(line)
    print()
    for n, rows in pairs:
        t = statistics.mean(r["tps"] for r in rows.values())
        al = statistics.mean(r["accept_len"] for r in rows.values())
        ar = statistics.mean(r["accepted"] / r["drafted"] for r in rows.values())
        print(f"{n:6s} mean tok/s {t:6.2f}   accept_len {al:.3f}   "
              f"slot acceptance {ar:.3f}   prompts {len(rows)}")
    print(f"\nwrote {a.out}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
