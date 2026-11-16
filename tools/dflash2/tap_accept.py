"""DFlash2 drafter block acceptance over one engine tap dump.

Who reads the output: whoever asked "is this tap source good enough to be worth
porting into Zig", plus the write-up in docs/design/flashnext-dflash2-drafter.md.
The JSON beside the dump (tap_accept.json) is the artifact; stdout is the same
numbers in a form a human can skim.

THE METRIC IS **agreement on teacher-forced text**, not the vendor's acceptance
length. The engine prefilled a fixed sequence and we replay it: at every round the
drafter sees the true anchor and is scored against the true next 7 tokens. A real
verify round draws the anchor from the target's own sampling, so the distribution
of anchors is different, and a drafter that merely learned "copy the context" can
score higher here than it would in production. Treat the number as a relative
ranking between tap sources, never as an absolute speedup claim.

The drafter maths is NOT reimplemented here: reference.py does the forward, the
candidate selector and the prefix match. This file is loader + block loop + report.

Context convention: the draft block's cross-attention reads the fused taps for the
*prefix*, i.e. positions [p - ctx_rows + 1 .. p] -- everything the target had
already computed when the block was proposed. `--ctx-mode block` instead feeds
taps_for(p .. p + 7), which is the literal reading of "taps for the block" but
leaks the next 7 positions' hidden states into the context and will inflate
acceptance. It exists for comparison only and is flagged `leaky` in the JSON.

Run with a CUDA torch, e.g. ~/.venvs/marker/bin/python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as ref  # noqa: E402  the drafter port lives next door

DRAFTER_DEFAULT = Path(
    "~/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2"
    "/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6"
)
TARGET_DEFAULT = Path(
    "~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local"
    "/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300"
)
HEAD_SHARD, HEAD_KEY = "tensors/model-00014-of-00021.safetensors", "lm_head.weight"
EMB_SHARD = "tensors/model-00015-of-00021.safetensors"
EMB_KEY = "model.language_model.embed_tokens.weight"


def fail(msg: str) -> None:
    """Refuse loudly. A wrong tap source silently scored is worse than no score."""
    print(f"tap_accept: REFUSING: {msg}", file=sys.stderr)
    raise SystemExit(2)


def read_bf16(path: Path, key: str, device: str) -> torch.Tensor:
    if not path.is_file():
        fail(f"missing shard {path}")
    with safe_open(str(path), "pt") as f:
        if key not in f.keys():
            fail(f"{key} not in {path.name}")
        t = f.get_tensor(key)
    # reference.py works in fp32 throughout; both these tensors are plain BF16 on disk.
    return t.to(device=device, dtype=torch.float32)


def load_dump(dump: Path, device: str):
    mpath = dump / "manifest.json"
    if not mpath.is_file():
        fail(f"no manifest.json in {dump}")
    m = json.loads(mpath.read_text())

    source = str(m.get("source", ""))
    if source == "mean4":
        fail(
            "manifest source is 'mean4' (mean over the 4 hyper-connection streams). "
            "That capture path is not built and the average is not the tensor the "
            "drafter was trained on; there is nothing to score."
        )
    taps = list(m.get("taps", []))
    if len(taps) != 5:
        fail(f"manifest lists {len(taps)} taps {taps}; the drafter's fc expects exactly 5")
    if int(m.get("block", 0)) <= 0:
        fail("manifest has no usable 'block'")
    rows, hidden, window = int(m["rows"]), int(m["hidden"]), int(m.get("window", 0))

    ids_path = dump / str(m.get("ids_file", "ids.bin"))
    ring_path = dump / str(m.get("ring_file", "ring.bin"))
    for p in (ids_path, ring_path):
        if not p.is_file():
            fail(f"missing {p}")

    ids = torch.from_numpy(np.fromfile(ids_path, dtype=np.uint32).astype(np.int64)).to(device)
    expect_ring_bytes = rows * 5 * hidden * 2
    got = ring_path.stat().st_size
    if got != expect_ring_bytes:
        fail(f"ring.bin is {got} bytes, manifest implies {expect_ring_bytes} ({rows}x5x{hidden} bf16)")
    buf = np.memmap(ring_path, dtype=np.uint16, mode="c", shape=(rows, 5 * hidden))
    ring = torch.from_numpy(np.asarray(buf)).view(torch.bfloat16).reshape(rows, 5, hidden).to(device)

    # row_bytes is informational in the spec (12800 = 5*2560 elements, not bytes); the file
    # size check above is the one that matters.
    row_bytes = int(m.get("row_bytes", 0))
    if row_bytes not in (0, 5 * hidden, 5 * hidden * 2):
        print(f"tap_accept: warning: manifest row_bytes {row_bytes} is neither 5*hidden "
              f"({5 * hidden}) nor 5*hidden*2 ({5 * hidden * 2}); trusting the file size",
              file=sys.stderr)
    return m, ids, ring


def to_device(obj, device: str):
    """reference.py's model is a plain object graph, not an nn.Module: walk it."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = to_device(v, device)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = to_device(v, device)
        return obj
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        for k, v in list(vars(obj).items()):
            setattr(obj, k, to_device(v, device))
        return obj
    return obj


def block_taps(ring: torch.Tensor, p_from: int, p_to: int) -> torch.Tensor:
    """[1, n, 5*hidden] fp32 for positions p_from..p_to, wrapped through the ring."""
    rows = ring.shape[0]
    idx = torch.arange(p_from, p_to + 1, device=ring.device) % rows
    return ring[idx].reshape(1, p_to - p_from + 1, -1).to(torch.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DFlash2 drafter acceptance over one tap dump")
    ap.add_argument("--dump", required=True, type=Path, help="directory holding manifest.json/ids.bin/ring.bin")
    ap.add_argument("--max-rounds", type=int, default=64, help="cap on verify rounds (default 64)")
    ap.add_argument("--block", type=int, default=8, help="block rows (default 8, must match the drafter)")
    ap.add_argument("--device", default="cuda", help="torch device (default cuda)")
    ap.add_argument("--drafter", type=Path, default=DRAFTER_DEFAULT, help="drafter checkpoint dir")
    ap.add_argument("--target", type=Path, default=TARGET_DEFAULT, help="target checkpoint dir (head + embed)")
    ap.add_argument("--ctx-mode", choices=["prefix", "block"], default="prefix",
                    help="prefix = taps of the known prefix (default); block = taps_for(p..p+7), leaky")
    ap.add_argument("--ctx-rows", type=int, default=0, help="cap the context window (0 = manifest window)")
    ap.add_argument("--start-p", type=int, default=0,
                    help="first anchor position (default 0 = right after the ring's valid start). "
                         "Raise it to score against a real prefix instead of an empty one.")
    ap.add_argument("--out", type=Path, default=None, help="JSON path (default <dump>/tap_accept.json)")
    args = ap.parse_args(argv)

    dump = args.dump.expanduser().resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("tap_accept: CUDA unavailable, falling back to cpu", file=sys.stderr)
        args.device = "cpu"

    manifest, ids, ring = load_dump(dump, args.device)
    rows, hidden, window = int(manifest["rows"]), int(manifest["hidden"]), int(manifest.get("window", 0))

    lm_head = read_bf16(args.target.expanduser() / HEAD_SHARD, HEAD_KEY, args.device)
    embed = read_bf16(args.target.expanduser() / EMB_SHARD, EMB_KEY, args.device)
    if lm_head.shape[1] != hidden:
        fail(f"lm_head is [_, {lm_head.shape[1]}], manifest hidden is {hidden}")
    if embed.shape[1] != hidden:
        fail(f"embed is [_, {embed.shape[1]}], manifest hidden is {hidden}")

    model, cfg = ref.load(args.drafter.expanduser(), lm_head, embed)
    model.lm_head, model.embed = lm_head, embed
    to_device(model, args.device)
    if cfg["hidden"] != hidden:
        fail(f"drafter hidden {cfg['hidden']} != manifest hidden {hidden}")
    if cfg["vocab"] != lm_head.shape[0]:
        fail(f"drafter vocab {cfg['vocab']} != head rows {lm_head.shape[0]}")
    if cfg["taps"] != list(manifest["taps"]):
        fail(f"drafter target_layer_ids {cfg['taps']} != manifest taps {manifest['taps']} (fc column order)")
    if cfg["block"] != args.block or args.block != int(manifest["block"]):
        fail(f"block: drafter {cfg['block']}, manifest {manifest['block']}, --block {args.block}")

    N = int(ids.numel())
    ctx_rows = args.ctx_rows or (window or rows)
    ctx_rows = min(ctx_rows, rows)
    if N < args.block + 2:
        fail(f"ids.bin holds {N} tokens; need at least {args.block + 2} for one round")

    # Positions older than N - rows have been overwritten in the ring.
    p = max(0, N - rows, args.start_p)
    leaky = args.ctx_mode == "block"
    rounds, total_accepted, bonus_hits, bonus_seen = [], 0, 0, 0

    while len(rounds) < args.max_rounds and p + args.block < N:
        block_ids = torch.full((1, args.block), int(cfg["mask_token"]), dtype=torch.long, device=args.device)
        block_ids[0, 0] = int(ids[p])
        positions = torch.arange(p, p + args.block, dtype=torch.long, device=args.device)

        lo = p if leaky else max(0, p - ctx_rows + 1)
        hi = p + args.block - 1 if leaky else p
        taps = block_taps(ring, lo, hi)
        # The context rows are rotary'd at the positions they were captured at, so
        # pass them explicitly: the prefix (and the leaky variant) do not end at p-1.
        ctx_positions = torch.arange(lo, hi + 1, dtype=torch.long, device=args.device)

        with torch.no_grad():
            hidden_out = model.forward(taps, block_ids, positions, ctx_positions)
            anchor = torch.tensor([int(ids[p])], dtype=torch.long, device=args.device)  # propose wants [B]
            proposed = model.propose(hidden_out, anchor)[0]

        # Block row r sits at position p+r and reads the token *at* p+r, so the 7
        # proposals are rows 1..7 for positions p+1..p+7 -- propose() already drops
        # row 0 (the anchor re-prediction, exactly as dflash.py's
        # _sample_draft_tokens does). There is no leftover "tail" row: all 8 block
        # rows are spent, 1 anchor + 7 drafts.
        targets = ids[p + 1: p + args.block].to(torch.long).cpu()
        drafts = proposed[: args.block - 1].to(torch.long).cpu()
        accepted = ref.acceptance(drafts, targets)
        total_accepted += accepted
        bonus = None
        rounds.append({
            "p": p, "ctx_rows": int(taps.shape[1]),
            "accepted": accepted, "advance": 1 + accepted,
            "proposed": drafts.tolist(), "targets": targets.tolist(),
            "bonus_tail_hit": bonus,
        })
        p += 1 + accepted

    if not rounds:
        fail("no round could run: not enough tokens after the ring's valid start")

    n = len(rounds)
    adv = [r["advance"] for r in rounds]
    acc = [r["accepted"] for r in rounds]
    summary = {
        "dump": str(dump),
        "source": manifest.get("source"),
        "tap_rows": rows, "taps": manifest["taps"], "window": window, "ctx_rows": ctx_rows,
        "sequence_tokens": N, "start_p": rounds[0]["p"], "rounds": n, "max_rounds": args.max_rounds,
        "metric": "agreement on teacher-forced text",
        "metric_caveat": "engine-given sequence, teacher-forced anchors; not the vendor's acceptance length",
        "context": args.ctx_mode, "leaky": leaky,
        "block": args.block,
        "mean_advance": sum(adv) / n,
        "mean_accepted": sum(acc) / n,
        "zero_accept_rate": sum(1 for a in acc if a == 0) / n,
        "full_block_rate": sum(1 for a in acc if a == args.block - 1) / n,
        "total_accepted": total_accepted,
        "total_advance": sum(adv),
        "tokens_scored": n + total_accepted,  # anchors + landed tokens, i.e. sum(advance)
        "bonus_tail_hit_rate": (bonus_hits / bonus_seen) if bonus_seen else None,
        "peak_vram_gib": (torch.cuda.max_memory_allocated() / 2**30) if args.device.startswith("cuda") else None,
        "per_round": rounds,
    }

    out = args.out or (dump / "tap_accept.json")
    out.write_text(json.dumps(summary, indent=1) + "\n")

    print(f"dump            {dump}")
    print(f"source          {summary['source']}   taps {summary['taps']}  window {window}")
    print(f"context         {args.ctx_mode}{'  (LEAKY: sees p..p+7)' if leaky else ''}"
          f"  ctx_rows {rounds[0]['ctx_rows']}")
    print(f"rounds          {n} (start p={rounds[0]['p']}, seq {N} tokens, cap {args.max_rounds})")
    print(f"mean advance    {summary['mean_advance']:.3f} tokens/round   [agreement on teacher-forced text]")
    print(f"accepted/round  {summary['mean_accepted']:.3f} of {args.block - 1}"
          f"   zero-accept {100 * summary['zero_accept_rate']:.1f}%"
          f"   full {100 * summary['full_block_rate']:.1f}%")
    if summary["bonus_tail_hit_rate"] is not None:
        print(f"bonus tail hit  {100 * summary['bonus_tail_hit_rate']:.1f}% (row {args.block - 1} == id[p+{args.block}])")
    if summary["peak_vram_gib"]:
        print(f"peak vram       {summary['peak_vram_gib']:.2f} GiB")
    print(f"wrote           {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
