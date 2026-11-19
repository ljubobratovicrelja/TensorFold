#!/usr/bin/env python3
"""Read the engine's TF_FLASHNEXT_EXPERT_DUMP / _HIDDEN_DUMP captures and plot what they hold.

    python3 plot_activations.py --data data --out plots [--summary summary.json]

matplotlib + numpy only. Every folder under --data that has an `experts/` directory is one category.

What is read (see README.md for the byte layouts):

  experts/p<NNNN>.bin   u32 magic 'TFEX', u32 rows, u32 pos0, u32 layers, u32 slots,
                        ids  [layers][rows][slots] u32 LE, then wts [layers][rows][slots] f32 LE
  experts/router_logits.bin  u32 'TFRT', u32 layers, u32 rows, u32 width, [layers][rows][width] f32
  hidden/hidden.bin     u32 'TFHD', layers, keep, dims, rows_of_pass, k, pos0, [layers][keep][dims] bf16

The prompt pass is the widest pass of the run (a server first measures its window costs on passes of
1..32 rows, so "the first pass" is not the prompt); every pass written after it is generation.
"""
import argparse
import glob
import json
import os
import struct

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------------------- readers

EX_MAGIC = 0x58454654
RT_MAGIC = 0x54524654
HD_MAGIC = 0x44484654


def _u32(b, at):
    return struct.unpack_from("<5I", b, at)


def read_expert_dir(path):
    """Every pass in a directory: header fields, ids [layers, rows, slots], wts [layers, rows, slots]."""
    passes = []
    for f in sorted(glob.glob(os.path.join(path, "p*.bin"))):
        blob = open(f, "rb").read()
        magic, rows, pos0, layers, slots = _u32(blob, 0)
        if magic != EX_MAGIC:
            raise ValueError(f"{f}: bad magic {magic:#x}")
        n = layers * rows * slots
        ids = np.frombuffer(blob, dtype="<u4", count=n, offset=20).reshape(layers, rows, slots)
        off = 20 + 4 * n
        wts = np.frombuffer(blob, dtype="<f4", count=n, offset=off).reshape(layers, rows, slots)
        passes.append(
            {"file": os.path.basename(f), "rows": rows, "pos0": pos0, "ids": ids, "wts": wts}
        )
    return passes


def read_router_logits(path):
    """[layers, rows, experts+1] f32, or None."""
    f = os.path.join(path, "router_logits.bin")
    if not os.path.exists(f):
        return None
    blob = open(f, "rb").read()
    magic = struct.unpack_from("<I", blob, 0)[0]
    if magic != RT_MAGIC:
        raise ValueError(f"{f}: bad magic {magic:#x}")
    layers, rows, width = struct.unpack_from("<3I", blob, 4)
    a = np.frombuffer(blob, dtype="<f4", count=layers * rows * width, offset=16)
    return a.reshape(layers, rows, width)


def read_hidden(path):
    """dict with [layers, keep, dims] float32 (bf16 widened), plus k and rows_of_pass."""
    f = os.path.join(path, "hidden.bin")
    if not os.path.exists(f):
        return None
    blob = open(f, "rb").read()
    magic = struct.unpack_from("<I", blob, 0)[0]
    if magic != HD_MAGIC:
        raise ValueError(f"{f}: bad magic {magic:#x}")
    layers, keep, dims, rows_of_pass, k, pos0 = struct.unpack_from("<6I", blob, 4)
    raw = np.frombuffer(blob, dtype="<u2", count=layers * keep * dims, offset=28)
    bf = (raw.astype(np.uint32) << 16).view(np.float32).reshape(layers, keep, dims)
    return {
        "x": bf[:, :k, :],  # only the k rows the pass wrote
        "k": k,
        "rows_of_pass": rows_of_pass,
        "pos0": pos0,
    }


def split_passes(passes):
    """(prompt pass index, generation passes): the widest pass is the prompt fill."""
    if not passes:
        return None, []
    idx = int(np.argmax([p["rows"] for p in passes]))
    return idx, passes[idx + 1 :]


# --------------------------------------------------------------------------------------- measures


def expert_counts(passes, experts=512):
    """[layers, experts] routed counts over every (row, slot) of these passes.

    Slot 10 of every row is the always-on shared expert (id 512, the router's last logit row): it is
    counted out here, so a layer's mass is 10 routed experts a token, and reported separately as
    `shared_w` in load_metrics.
    """
    layers = passes[0]["ids"].shape[0]
    counts = np.zeros((layers, experts), dtype=np.int64)
    for p in passes:
        for l in range(layers):
            ids = p["ids"][l].ravel()
            ids = ids[ids < experts]
            counts[l] += np.bincount(ids, minlength=experts)
    return counts


def shared_weight(passes):
    """[layers] mean gate weight of the shared expert (slot 10), and of the 11 slots' total."""
    layers, slots = passes[0]["wts"].shape[0], passes[0]["wts"].shape[2]
    tot = np.zeros(layers)
    n = 0
    for p in passes:
        tot += p["wts"][:, :, slots - 1].sum(1)
        n += p["wts"].shape[1]
    return tot / max(n, 1)


def entropy_norm(prob):
    """Entropy of a distribution over axis -1, divided by log(n) (1.0 = perfectly even)."""
    p = np.clip(prob, 1e-12, None)
    h = -(p * np.log(p)).sum(-1)
    return h / np.log(prob.shape[-1])


def load_metrics(counts, shared=None):
    """Per layer: normalised entropy of the load, max/mean and the share of experts never chosen."""
    n_used = counts.sum(1)
    frac = counts / np.maximum(n_used, 1)[:, None]
    ent = entropy_norm(frac)
    mean = counts.sum(1) / counts.shape[1]
    peak = counts.max(1) / np.maximum(mean, 1e-9)
    dead = (counts == 0).mean(1)
    return {"frac": frac, "entropy": ent, "peak_over_mean": peak, "dead_frac": dead, "shared_w": shared}


def cka_linear(x, y):
    """Linear CKA between two [n, d] row sets (Kornblith et al. 2019, unbiased-free form)."""
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    xy = np.linalg.norm(x @ y.T) ** 2
    xx = np.linalg.norm(x @ x.T)
    yy = np.linalg.norm(y @ y.T)
    return float(xy / max(xx * yy, 1e-30))


def layer_cka(x):
    """[layers, layers] linear CKA over the rows of each layer."""
    n = x.shape[0]
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = cka_linear(x[i], x[j])
            m[i, j] = m[j, i] = v
    return m


def js_divergence(p, q):
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        keep = a > 0
        return float((a[keep] * np.log(a[keep] / np.maximum(b[keep], 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ------------------------------------------------------------------------------------------ plots


def over_selection_z(counts):
    """(count - expected) / sqrt(expected) per layer: how far above a uniform load a cell sits.

    A layer routes 10 experts a token, so with ~1.5k tokens a cell's uniform expectation is ~30 and a
    raw cell is dominated by counting noise; the z view is the same data against that noise floor.
    """
    exp = np.maximum(counts.sum(1, keepdims=True) / counts.shape[1], 1.0)
    return (counts - exp) / np.sqrt(exp)


def plot_layer_expert_heatmaps(per_cat, out):
    """Experts are ordered by their total load (mean over categories) so a hot column is a hot column."""
    cats = list(per_cat)
    order = np.argsort(-np.mean([per_cat[c]["gen"]["counts"] for c in cats], axis=0).sum(0))
    fig, axes = plt.subplots(len(cats), 1, figsize=(15, 2.3 * len(cats) + 1.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cats):
        m = over_selection_z(per_cat[c]["gen"]["counts"])[:, order]
        im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6, interpolation="nearest")
        ax.set_ylabel("layer", fontsize=8)
        ax.set_title(f"{c}  (generation rows, {per_cat[c]['gen']['rows']} token-slots per layer)", fontsize=9, loc="left")
        ax.set_yticks([0, 11, 23, 35, 47])
    axes[-1].set_xlabel("expert id, columns sorted by total load (512)")
    cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.01)
    cb.set_label("(count - expected) / sqrt(expected)")
    fig.suptitle("MoE expert activation: layer x expert, over-selection against a uniform load", fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_expert_totals(per_cat, out, top=40):
    """The 512 experts' total share of the routed mass, over all layers: the global hot set."""
    cats = list(per_cat)
    frac = np.mean(
        [per_cat[c]["gen"]["counts"].sum(0) / max(per_cat[c]["gen"]["counts"].sum(), 1) for c in cats], axis=0
    )
    order = np.argsort(-frac)[:top]
    x = np.arange(top)
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.bar(x, frac[order] * 100, color="#444", width=0.75, label="mean over categories")
    for c in cats:
        cc = per_cat[c]["gen"]["counts"]
        f = cc.sum(0) / max(cc.sum(), 1)
        ax.scatter(x, f[order] * 100, s=18, label=c, zorder=3)
    ax.axhline(100 / 512, color="crimson", ls="--", lw=1, label="uniform share (0.195%)")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([int(i) for i in order[::2]], fontsize=7)
    ax.set_xlabel("expert id (top 40 by mean share of all routed tokens)")
    ax.set_ylabel("% of routed tokens")
    ax.set_title("Global expert load: the hot set, and how much each category disagrees with it", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_load_balance(per_cat, out):
    cats = list(per_cat)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for c in cats:
        m = per_cat[c]["gen"]["load"]
        layers = np.arange(m["entropy"].size)
        axes[0].plot(layers, m["entropy"], label=c, lw=1.4)
        axes[1].plot(layers, m["peak_over_mean"], label=c, lw=1.4)
        axes[2].plot(layers, 100 * m["dead_frac"], label=c, lw=1.4)
        if m["shared_w"] is not None:
            axes[3].plot(layers, m["shared_w"], label=c, lw=1.4)
    axes[0].set_title("load entropy / log(512)")
    axes[1].set_title("hottest expert / mean load")
    axes[2].set_title("experts never routed (%)")
    axes[3].set_title("shared expert's gate weight")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=1)
    fig.suptitle("Expert load balance per layer (generation)", fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_router_entropy(per_cat, out):
    cats = [c for c in per_cat if per_cat[c]["logits"] is not None]
    if not cats:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for c in cats:
        lg = per_cat[c]["logits"]  # [layers, rows, 513]
        z = lg[..., :512]
        z = z - z.max(-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(-1, keepdims=True)
        h = entropy_norm(p).mean(1)
        top11 = np.sort(p, axis=-1)[..., -11:].sum(-1).mean(1)
        axes[0].plot(h, label=c, lw=1.4)
        axes[1].plot(top11, label=c, lw=1.4)
    axes[0].set_title("router softmax entropy / log(512)")
    axes[1].set_title("mass on the 11 chosen experts")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Full-512 router distribution (prompt pass, first rows)", fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_hidden(per_cat, out):
    cats = [c for c in per_cat if per_cat[c]["hidden"] is not None]
    if not cats:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for c in cats:
        x = per_cat[c]["hidden"]["x"].astype(np.float32)  # [layers, k, dims]
        norm = np.linalg.norm(x, axis=-1).mean(1)
        cos_next = np.array(
            [
                float(
                    (x[l].mean(0) @ x[l + 1].mean(0))
                    / max(np.linalg.norm(x[l].mean(0)) * np.linalg.norm(x[l + 1].mean(0)), 1e-12)
                )
                for l in range(x.shape[0] - 1)
            ]
        )
        cos_last = np.array(
            [
                float(
                    (x[l].mean(0) @ x[-1].mean(0))
                    / max(np.linalg.norm(x[l].mean(0)) * np.linalg.norm(x[-1].mean(0)), 1e-12)
                )
                for l in range(x.shape[0])
            ]
        )
        axes[0].plot(norm, label=c, lw=1.5)
        axes[1].plot(np.arange(len(cos_next)), cos_next, label=c, lw=1.4)
        axes[2].plot(cos_last, label=c, lw=1.4)
    axes[0].set_title("mean L2 norm of the layer's hidden rows")
    axes[1].set_title("cosine(layer L, layer L+1), row means")
    axes[2].set_title("cosine(layer L, layer 47)")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
    axes[1].set_ylim(min(0.0, axes[1].get_ylim()[0]), 1.02)
    axes[0].legend(fontsize=7)
    fig.suptitle("Post-norm MoE-input hidden state across depth (prompt pass)", fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cka(per_cat, out):
    cats = [c for c in per_cat if per_cat[c]["hidden"] is not None]
    if not cats:
        return
    fig, axes = plt.subplots(1, len(cats), figsize=(3.4 * len(cats), 3.6))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cats):
        m = per_cat[c]["cka"]
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
    cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.01)
    cb.set_label("linear CKA")
    fig.suptitle("Representation similarity between layers (linear CKA over 32 prompt rows)", fontsize=11)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_channels(per_cat, out):
    cats = [c for c in per_cat if per_cat[c]["hidden"] is not None]
    if not cats:
        return
    fig, axes = plt.subplots(len(cats), 1, figsize=(15, 1.5 * len(cats) + 1.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cats):
        x = per_cat[c]["hidden"]["x"].astype(np.float32)
        a = np.abs(x).mean(1)  # [layers, dims]
        m = np.log2(a / np.maximum(a.mean(1, keepdims=True), 1e-9))  # each layer against its own mean channel
        im = ax.imshow(m, aspect="auto", cmap="inferno", vmin=-1.5, vmax=2.5, interpolation="nearest")
        ax.set_ylabel("layer", fontsize=8)
        ax.set_title(c, fontsize=9, loc="left")
        ax.set_yticks([0, 11, 23, 35, 47])
    axes[-1].set_xlabel("channel (2560)")
    cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.01)
    cb.set_label("log2(channel |x| / layer mean)")
    fig.suptitle("Outlier channels: each layer's per-channel magnitude against its own mean", fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_separation(per_cat, out):
    cats = list(per_cat)
    n = len(cats)
    jsd = np.zeros((n, n))
    cos = np.zeros((n, n))
    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            pa = per_cat[a]["gen"]["counts"].ravel().astype(np.float64)
            pb = per_cat[b]["gen"]["counts"].ravel().astype(np.float64)
            jsd[i, j] = js_divergence(pa, pb)
            cos[i, j] = float(
                (pa @ pb) / max(np.linalg.norm(pa) * np.linalg.norm(pb), 1e-12)
            )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.0))
    for ax, (m, name) in zip(axes[:2], [(jsd, "Jensen-Shannon divergence"), (cos, "cosine similarity")]):
        im = ax.imshow(m, cmap="magma" if name.startswith("J") else "viridis", interpolation="nearest")
        ax.set_xticks(range(n))
        ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(cats, fontsize=8)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    # per-layer divergence against the mean of the other four
    for c in cats:
        counts = per_cat[c]["gen"]["counts"]
        others = np.mean([per_cat[o]["gen"]["counts"] for o in cats if o != c], axis=0)
        d = [js_divergence(counts[l].astype(np.float64), others[l].astype(np.float64)) for l in range(counts.shape[0])]
        axes[2].plot(d, label=c, lw=1.3)
    axes[2].set_title("JSD of each layer's load vs the other categories")
    axes[2].set_xlabel("layer")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7)
    fig.suptitle("Do the five categories separate in their routing?", fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="directory holding one folder per category")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--summary", default=None, help="write the measured numbers as JSON here")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    per_cat = {}
    for cat in sorted(os.listdir(args.data)):
        d = os.path.join(args.data, cat)
        exp, hid = os.path.join(d, "experts"), os.path.join(d, "hidden")
        if not os.path.isdir(exp):
            continue
        passes = read_expert_dir(exp)
        if not passes:
            continue
        pi, gen = split_passes(passes)
        prompt = passes[pi]
        per_cat[cat] = {
            "prompt": prompt,
            "gen": {
                "rows": int(sum(p["rows"] for p in gen)),
                "counts": expert_counts(gen if gen else [prompt]),
            },
            "prompt_counts": expert_counts([prompt]),
            "logits": read_router_logits(exp),
            "hidden": read_hidden(hid),
            "n_passes": len(passes),
        }
        per_cat[cat]["gen"]["load"] = load_metrics(
            per_cat[cat]["gen"]["counts"], shared_weight(gen if gen else [prompt])
        )
        if per_cat[cat]["hidden"] is not None:
            per_cat[cat]["cka"] = layer_cka(per_cat[cat]["hidden"]["x"].astype(np.float32))
        print(f"{cat}: {len(passes)} passes, prompt {prompt['rows']} rows, {per_cat[cat]['gen']['rows']} generation rows")

    if not per_cat:
        raise SystemExit(f"no expert dumps under {args.data}")

    plot_layer_expert_heatmaps(per_cat, os.path.join(args.out, "01_layer_expert_heatmap.png"))
    plot_expert_totals(per_cat, os.path.join(args.out, "01b_expert_totals.png"))
    plot_load_balance(per_cat, os.path.join(args.out, "02_expert_load_balance.png"))
    plot_router_entropy(per_cat, os.path.join(args.out, "03_router_entropy.png"))
    plot_hidden(per_cat, os.path.join(args.out, "04_hidden_norm_drift.png"))
    plot_cka(per_cat, os.path.join(args.out, "05_layer_cka.png"))
    plot_channels(per_cat, os.path.join(args.out, "06_channel_magnitudes.png"))
    plot_separation(per_cat, os.path.join(args.out, "07_category_separation.png"))

    summary = {}
    for cat, v in per_cat.items():
        load = v["gen"]["load"]
        c = v["gen"]["counts"]
        prompt_c = v["prompt_counts"]
        top = np.argsort(-c.sum(1))[:5]
        entry = {
            "passes": v["n_passes"],
            "prompt_rows": int(v["prompt"]["rows"]),
            "generation_rows": v["gen"]["rows"],
            "gen_entropy_mean": float(load["entropy"].mean()),
            "gen_entropy_min_layer": int(np.argmin(load["entropy"])),
            "gen_entropy_min": float(load["entropy"].min()),
            "gen_dead_experts_frac_mean": float(load["dead_frac"].mean()),
            "gen_peak_over_mean_max": float(load["peak_over_mean"].max()),
            "gen_peak_over_mean_max_layer": int(np.argmax(load["peak_over_mean"])),
            "hot_layers_by_concentration": [int(i) for i in np.argsort(-load["peak_over_mean"])[:6]],
            "experts_used_prompt": int((prompt_c.sum(0) > 0).sum()),
            "experts_used_generation": int((c.sum(0) > 0).sum()),
            "prompt_tokens_per_layer": int(prompt_c.sum(1)[0]),
            "top_experts_global": [int(i) for i in np.argsort(-c.sum(0))[:10]],
            "gen_expert_mass_top10": float(np.sort(c.sum(0))[::-1][:10].sum() / max(c.sum(), 1)),
        }
        if v["hidden"] is not None:
            x = v["hidden"]["x"].astype(np.float32)
            norm = np.linalg.norm(x, axis=-1).mean(1)
            entry["hidden_rows"] = int(v["hidden"]["k"])
            entry["norm_layer0"] = float(norm[0])
            entry["norm_max_layer"] = int(np.argmax(norm))
            entry["norm_max"] = float(norm.max())
            entry["norm_min_layer"] = int(np.argmin(norm))
            entry["norm_min"] = float(norm.min())
            cons = [
                float(
                    (x[l].mean(0) @ x[l + 1].mean(0))
                    / max(np.linalg.norm(x[l].mean(0)) * np.linalg.norm(x[l + 1].mean(0)), 1e-12)
                )
                for l in range(47)
            ]
            entry["cos_next_min"] = float(min(cons))
            entry["cos_next_min_layer"] = int(np.argmin(cons))
            entry["cos_next_mean"] = float(np.mean(cons))
            entry["cka_adjacent_mean"] = float(
                np.mean([v["cka"][l, l + 1] for l in range(47)])
            )
            entry["cka_adjacent_min"] = float(min(v["cka"][l, l + 1] for l in range(47)))
            entry["cka_adjacent_min_layer"] = int(np.argmin([v["cka"][l, l + 1] for l in range(47)]))
        if v["logits"] is not None:
            z = v["logits"][..., :512]
            z = z - z.max(-1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(-1, keepdims=True)
            h = entropy_norm(p).mean(1)
            e = {}
            e["entropy_layer0"] = float(h[0])
            e["entropy_max"] = float(h.max())
            e["entropy_max_layer"] = int(np.argmax(h))
            e["entropy_min"] = float(h.min())
            e["entropy_min_layer"] = int(np.argmin(h))
            e["top11_mass_mean"] = float(np.sort(p, -1)[..., -11:].sum(-1).mean())
            entry["router_logits"] = e
        summary[cat] = entry

    if args.summary:
        json.dump(summary, open(args.summary, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:4000])
    print(f"plots written to {args.out}")


if __name__ == "__main__":
    main()
