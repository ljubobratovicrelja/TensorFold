#!/usr/bin/env python3
"""Parity: the torch forward (`forward.py`) against the engine's own `draft-check`, on the same
harvested fixtures, round for round.

    /usr/bin/python3 tools/dflash2/train/parity.py                  # chunk-0000, chunk-0002
    /usr/bin/python3 tools/dflash2/train/parity.py --rounds 24 --fixtures chunk-0000

Both sides run the SAME rounds: a sequence of positions ending at `len(ids) - BLOCK - 1` (so every
round has `block-1` real targets after its anchor), stepping by one. The engine is handed a fixture
directory whose `manifest.json` carries that `positions` list — ring.bin/ids.bin are symlinked, not
copied — and runs one round per entry on one warm state (`draft-check --repeat N`), exactly the
served reuse regime. The torch side replays the same positions with `forward.run_position`.

Reported per fixture: per-slot acceptance rate, mean advance, and how many of the `rounds * 7`
proposal slots agree exactly between the two implementations, plus a stage-by-stage diff of round 0
(so a divergence is named by tensor, not by acceptance).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forward as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/dflash2, for engine.py
import engine  # noqa: E402

HOME = Path(os.path.expanduser("~"))
HF = HOME / ".cache/huggingface/hub"
DEFAULT_DRAFTER = HF / "models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6"
DEFAULT_TARGET = HF / "models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300"
DEFAULT_BINARY = engine.BIN
HARVEST = Path("/tmp/harv/out")

STAGE_NAMES = (
    [f"{s}_{i}" for i in range(5) for s in
     ("conv_attn_in", "qkv", "attn", "attn_proj", "conv_attn_out",
      "conv_mlp_in", "mlp", "conv_mlp_out", "block_hidden")]
    + ["final_norm_block", "logits_top16_ids", "logits_top16_vals"]
)


def seq_fixture(src: Path, dst: Path, positions: list[int]) -> Path:
    """A fixture directory whose manifest lists `positions`, with ring/ids symlinked from `src`."""
    man = json.loads((src / "manifest.json").read_text())
    man["positions"] = positions
    dst.mkdir(parents=True, exist_ok=True)
    for name in (man.get("ring_file", "ring.bin"), man.get("ids_file", "ids.bin")):
        link = dst / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((src / name).resolve())
    (dst / "manifest.json").write_text(json.dumps(man))
    return dst


def run_engine(binary: Path, drafter: Path, target: Path, fixture: Path, out: Path, reps: int) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cmd = [str(binary), "draft-check", "--drafter", str(drafter), "--target", str(target),
           "--fixture", str(fixture), "--out", str(out), "--repeat", str(reps)]
    print("  engine:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if "draftfix" in line or "error" in line.lower():
            print("   ", line)
    if r.returncode != 0:
        raise SystemExit(f"draft-check failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")


def read_rep(out: Path, i: int, reps: int, keep: bool) -> dict:
    d = out if reps == 1 else out / f"rep{i}"
    js = json.loads((d / "stages.json").read_text())
    tensors = {}
    for name in STAGE_NAMES:
        p = d / "tensors" / f"{name}.npy"
        tensors[name] = np.load(p) if p.exists() else None
    full = d / "tensors" / "logits_full.npy"
    if full.exists() and not keep:
        full.unlink()
    return {"round": js["round"], "proposals": js["round"]["proposals"],
            "selector_agrees": js["round"]["selector_argmax_agrees"], "tensors": tensors,
            "ctx": [s for s in js["stages"] if s["stage"] == "fc_ctx"][0]}


def slot_table(proposals: list[list[int]], ids: torch.Tensor, positions: list[int]) -> tuple[list[float], float]:
    """Per-slot acceptance rates: slot k counts only if slots 0..k-1 all matched."""
    n = len(positions)
    hit = [0] * (F.BLOCK - 1)
    accepted = 0
    for props, p in zip(proposals, positions):
        tgt = ids[p + 1:p + F.BLOCK].tolist()
        run = F.acceptance(props, tgt)
        accepted += run
        for k in range(run):
            hit[k] += 1
    return [h / n for h in hit], 1.0 + accepted / n


def stage_report(name: str, eng: np.ndarray, torch_t, eps: float = 1e-9) -> str:
    e = eng.astype(np.float32)
    t = torch_t.detach().float().cpu().numpy()
    if e.shape != t.shape:
        return f"{name:22s} SHAPE engine {e.shape} vs torch {t.shape}"
    d = np.abs(e - t)
    denom = max(float(np.abs(e).max()), eps)
    exact = 100.0 * float((d == 0).mean())
    return (f"{name:22s} max|d| {d.max():9.3e}  rel {d.max() / denom:8.2e}  "
            f"mean|d| {d.mean():9.3e}  bit-equal {exact:6.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", nargs="+", default=["chunk-0000", "chunk-0002"])
    ap.add_argument("--harvest", default=str(HARVEST))
    ap.add_argument("--drafter", default=str(DEFAULT_DRAFTER))
    ap.add_argument("--target", default=str(DEFAULT_TARGET))
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--work", default="/tmp/dp2/parity")
    ap.add_argument("--keep", action="store_true", help="keep the big logits_full.npy dumps")
    ap.add_argument("--show-mismatches", action="store_true", help="print rounds whose proposals differ")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    print("loading drafter + target embed/head ...", flush=True)
    drafter = F.Drafter(args.drafter, args.target)

    rows = []
    for name in args.fixtures:
        src = Path(args.harvest) / name
        man = json.loads((src / "manifest.json").read_text())
        n_ids = (src / man.get("ids_file", "ids.bin")).stat().st_size // 4
        last = n_ids - F.BLOCK - 1                  # p + block < n_ids: real targets after p
        positions = list(range(last - args.rounds + 1, last + 1))
        fx = F.load_fixture(src)

        print(f"\n=== {name}: rows {man['rows']} ids {n_ids} pos_end {man['pos_end']} "
              f"window {man['window']}  positions {positions[0]}..{positions[-1]}", flush=True)

        fx_dir = seq_fixture(src, work / f"{name}-seq", positions)
        out = work / f"{name}-eng"
        run_engine(Path(args.binary), Path(args.drafter), Path(args.target), fx_dir, out, len(positions))
        reps = [read_rep(out, i, len(positions), args.keep) for i in range(len(positions))]

        eng_props = [r["proposals"] for r in reps]
        torch_props, trace0 = [], None
        for i, p in enumerate(positions):
            tr = {} if i == 0 else None
            r = F.run_position(drafter, fx, p, trace=tr)
            torch_props.append(r["proposals"])
            if tr is not None:
                trace0 = tr

        eng_slots, eng_adv = slot_table(eng_props, fx["ids"], positions)
        tor_slots, tor_adv = slot_table(torch_props, fx["ids"], positions)
        agree = sum(1 for a, b in zip(eng_props, torch_props) for x, y in zip(a, b) if x == y)
        total = len(positions) * (F.BLOCK - 1)
        full_rounds = sum(1 for a, b in zip(eng_props, torch_props) if list(a) == list(b))
        rows.append((name, eng_slots, eng_adv, tor_slots, tor_adv, agree, total, full_rounds,
                     len(positions), reps[0]["selector_agrees"], sum(r["selector_agrees"] for r in reps)))

        # ---- stage diff of round 0 ---------------------------------------------------------
        rep = reps[0]
        if args.show_mismatches:
            for p, a, b in zip(positions, eng_props, torch_props):
                if list(a) != list(b):
                    tgt = fx["ids"][p + 1:p + F.BLOCK].tolist()
                    print(f"  MISMATCH pos {p}: engine {list(a)} | torch {list(b)} | target {tgt}")
        print(f"  stage diff, round 0 (position {positions[0]}, ctx_len {rep['round']['ctx_len']}):")
        for s in STAGE_NAMES:
            e = rep["tensors"].get(s)
            t = trace0.get(s)
            if e is None or t is None:
                continue
            if s in ("logits_top16_ids",):
                print(f"{s:22s} ids: {int((np.asarray(e).astype(np.int64) == t.cpu().numpy().astype(np.int64)).sum())}"
                      f"/{np.asarray(e).size} slots equal")
                continue
            print("  " + stage_report(s, e, t))
        c = rep["ctx"]
        tctx = trace0["fc_ctx_rows"].float()
        print(f"  fc_ctx summary: engine mean {c['mean']:+.6f} std {c['std']:.6f} | "
              f"torch mean {tctx.mean().item():+.6f} std {tctx.std().item():.6f}")

    # ---- the table -------------------------------------------------------------------------
    print("\n================ PARITY TABLE ================")
    hdr = f"{'fixture':12s} {'s0':>6s} {'s1':>6s} {'s2':>6s} {'s3':>6s} {'s4':>6s} {'s5':>6s} {'s6':>6s} {'meanAdv':>8s}"
    for name, es, ea, ts, ta, *_ in rows:
        print(f"{name:12s} (engine) " + " ".join(f"{v:6.3f}" for v in es) + f" {ea:8.3f}")
        print(f"{'':12s} (torch ) " + " ".join(f"{v:6.3f}" for v in ts) + f" {ta:8.3f}")
    print()
    print(f"{'fixture':12s} {'rounds':>6s} {'exact slots':>12s} {'full rounds':>12s} {'sel.agree(eng)':>15s}")
    for name, es, ea, ts, ta, agree, total, full_rounds, n, sa0, satot in rows:
        print(f"{name:12s} {n:6d} {agree:5d}/{total:<6d} {full_rounds:5d}/{n:<6d} {satot:6d}/{n * 7:<6d}")
    tot_agree = sum(r[5] for r in rows)
    tot_slots = sum(r[6] for r in rows)
    print(f"\nTOTAL exact proposal slots: {tot_agree}/{tot_slots} "
          f"({100.0 * tot_agree / tot_slots:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
