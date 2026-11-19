#!/usr/bin/env python3
"""Overlay the autoregressive arms: per-prompt throughput, and what the change is worth.

    plot_ar.py --arm baseline=runs/ar/baseline.json --arm head_fp8=runs/ar/head_fp8.json \
               --out runs/ar/plot_head_fp8.png

Left panel: per-prompt tok/s for every arm, paired by prompt index, with the arm means drawn across.
Right panel: the paired difference against the first arm, prompt by prompt, with the 95% t-interval of
the mean. Numbers come only from the committed JSON artifacts; nothing is typed in by hand.
"""
import argparse
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def t975(n):
    return {8: 2.365, 12: 2.201, 16: 2.131, 20: 2.093, 24: 2.069, 32: 2.040}.get(max(n - 1, 1), 1.96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=path.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    arms = []
    for spec in a.arm:
        name, path = spec.split("=", 1)
        d = json.load(open(path))
        arms.append((name, d, {r["i"]: r for r in d["rows"]}))
    base_name, _, base_rows = arms[0]
    idx = sorted(set.intersection(*(set(r) for _, _, r in arms)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.35, 1]})

    colours = ["#4c72b0", "#d62728", "#2ca02c", "#9467bd"]
    for k, (name, d, rows) in enumerate(arms):
        ys = [rows[i]["tok_s"] for i in idx]
        ax1.plot(range(len(idx)), ys, marker="o", ms=4, lw=1.4, color=colours[k % 4],
                 label=f"{name} — {statistics.mean(ys):.2f} tok/s")
        ax1.axhline(statistics.mean(ys), color=colours[k % 4], lw=0.9, ls="--", alpha=0.5)
    ax1.set_xlabel("prompt (16 public GSM8K test prompts, greedy, autoregressive only)")
    ax1.set_ylabel("tok/s")
    ax1.set_title("Per-prompt throughput")
    ax1.legend(fontsize=9, loc="lower right")

    if len(arms) > 1:
        span = []
        for k in range(1, len(arms)):
            name, _, rows = arms[k]
            delta = [rows[i]["tok_s"] - base_rows[i]["tok_s"] for i in idx]
            m = statistics.mean(delta)
            sd = statistics.stdev(delta)
            se = sd / len(delta) ** 0.5
            h = t975(len(delta)) * se
            ax2.bar([k - 1], [m], width=0.5, color=colours[k % 4], yerr=[h], capsize=6)
            span += [m - h, m + h]
            ax2.text(k - 1, m / 2, f"{m:+.2f} tok/s\n{100*m/statistics.mean([base_rows[i]['tok_s'] for i in idx]):+.2f}%",
                     ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        ax2.set_xlim(-0.7, max(0.7, len(arms) - 1.4))
        ax2.set_xticks(range(len(arms) - 1))
        ax2.set_xticklabels([f"{n}\nvs {base_name}" for n, _, _ in arms[1:]])
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_ylabel("paired difference, tok/s")
        lo, hi = min(span + [0.0]), max(span + [0.0])
        pad = max(0.15 * (hi - lo), 0.02)
        ax2.set_ylim(lo - pad, hi + pad)
        ax2.set_title("Paired difference (95% t-interval)")

    if a.note:
        fig.suptitle(a.note, fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
