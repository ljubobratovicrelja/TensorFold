#!/usr/bin/env python3
"""DFlash2 regression scaffold — functional and speed gates for a drafter change.

One fixture round through the production drafter path, captured stage by stage, then compared
against a baseline. Two independent gates:

  functional  every stage tensor must match the baseline (`--func-tol`, default 0 = bf16-exact)
  speed       per-stage and round ms must not regress (`--ms-tol`, default 5% on the median)

**A single run cannot be a speed gate on this box.** Two captures of the same binary, back to
back, measured 438 ms and 689 ms — every stage ~+70%, uniformly. That is the GPU's clock state,
not the code. So `capture` takes `--repeat` (default 3) rounds and `check` compares **medians**;
it also prints the baseline's own run-to-run spread and refuses to trust a speed verdict when
that spread is wider than the tolerance.

Usage:
    regress.py capture base --repeat 3      # runs/<label>/rep0..repN-1
    regress.py capture cand --repeat 3
    regress.py check base cand              # both gates; exit 1 on any failure
    regress.py list base                    # median per-stage ms, slowest first

Notes that cost a debugging session already:
  * The fixture is only usable when `pos_end >= block + 2`; `fixture/block8` ships with 9 and is
    refused by the engine as FixtureIdsShape. `capture` checks this before starting a server.
  * `fc_ctx_rows` is the host mirror of `fc_ctx` (same bytes, `.npy` copy, `measured: host`). It is
    excluded from every sum or the round looks twice as expensive as it is.
  * The harness stops only its own server pid and uses :8099. It never touches a host service.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALIGN = HERE / "align_bf16"
RUNNER = ALIGN / "run_engine_dflash2.sh"
DEFAULT_FIXTURE = ALIGN / "fixture" / "window"
RUNS = HERE / "runs"

# a host-side mirror of another stage's bytes; counting both doubles the round
MIRRORS = {"fc_ctx_rows"}


def fixture_problem(path: Path) -> str | None:
    """Why this fixture cannot be used, or None when it can."""
    man = path / "manifest.json"
    if not man.exists():
        return f"{path}: no manifest.json"
    m = json.loads(man.read_text())
    block = int(m.get("block", 8))
    pos_end = int(m.get("pos_end", 0))
    if pos_end < block + 2:
        return (f"{path}: pos_end {pos_end} < block+2 = {block + 2} "
                f"— the engine refuses this with FixtureIdsShape; regenerate with gen_inputs.py")
    if not (path / "ids.bin").exists() or not (path / "ring.bin").exists():
        return f"{path}: ids.bin/ring.bin missing"
    return None


def capture(label: str, fixture: Path, repeat: int) -> Path:
    problem = fixture_problem(fixture)
    if problem:
        sys.exit(f"refusing: {problem}")
    root = RUNS / label
    for i in range(repeat):
        out = root / f"rep{i}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"capture {label} rep{i}/{repeat} -> {out}")
        r = subprocess.run([str(RUNNER), str(fixture), str(out)], capture_output=True, text=True)
        sys.stdout.write("".join(f"  {line}\n" for line in r.stdout.strip().splitlines()[-2:]))
        if r.returncode != 0 or not (out / "stages.json").exists():
            sys.stderr.write(r.stderr[-2000:])
            sys.exit(f"capture failed (exit {r.returncode}); see {out}/logs/server.log")
    return root


def reps(root: Path) -> list[Path]:
    """The capture directories of a label, newest layout first (plain dir = one rep)."""
    found = sorted(p for p in root.glob("rep*") if (p / "stages.json").exists())
    if found:
        return found
    if (root / "stages.json").exists():
        return [root]
    sys.exit(f"{root}: no captures found (run `regress.py capture {root.name}`)")


def load(run: Path):
    """(doc, per-stage meta, per-stage arrays) for one capture."""
    doc = json.loads((run / "stages.json").read_text())
    meta = {s["stage"]: s for s in doc["stages"]}
    arrs = {}
    import numpy as np
    for name in meta:
        p = run / "tensors" / f"{name}.npy"
        if p.exists():
            arrs[name] = np.load(p)
    return doc, meta, arrs


def medians(root: Path) -> tuple[dict[str, float], float, list[Path]]:
    """Median ms per stage and for the round, over every rep."""
    dirs = reps(root)
    per: dict[str, list[float]] = {}
    rounds: list[float] = []
    for d in dirs:
        doc, meta, _ = load(d)
        for name, s in meta.items():
            if name in MIRRORS:
                continue
            per.setdefault(name, []).append(float(s["ms"]))
        rounds.append(float(doc["round"]["round_ms"]))
    return ({k: statistics.median(v) for k, v in per.items()},
            statistics.median(rounds), dirs)


def spread(values: list[float]) -> float:
    """Run-to-run spread as a percentage of the median (0 for a single rep)."""
    m = statistics.median(values)
    return 0.0 if m == 0 or len(values) < 2 else (max(values) - min(values)) / m * 100


def check(base: Path, cand: Path, func_tol: float, ms_tol: float, min_ms: float = 1.0) -> int:
    import numpy as np

    bdirs, cdirs = reps(base), reps(cand)
    bms, br, _ = medians(base)
    cms, cr, _ = medians(cand)

    # functional: every rep of each side must agree with the baseline's first rep
    _, _, barrs0 = load(bdirs[0])
    fails: list[str] = []
    print(f"functional — {len(bdirs)} base rep(s) vs {len(cdirs)} cand rep(s), tol {func_tol:g}")
    for d in cdirs:
        _, _, carrs = load(d)
        for name, a in barrs0.items():
            c = carrs.get(name)
            if c is None:
                fails.append(f"{name}: missing in {d.name}")
                continue
            if a.shape != c.shape:
                fails.append(f"{name}: shape {a.shape} -> {c.shape} in {d.name}")
                continue
            den = float(np.linalg.norm(a)) or 1.0
            rel = float(np.linalg.norm(a.astype(np.float64) - c.astype(np.float64)) / den)
            if rel > func_tol:
                fails.append(f"{name}: relL2 {rel:.3g} > {func_tol:g} in {d.name}")
    print(f"  {'PASS — every stage bit-identical' if not fails else f'{len(fails)} mismatch(es)'}")

    # speed: medians, with the baseline's own noise printed so a verdict can be judged
    print(f"\nspeed — medians over {len(bdirs)}/{len(cdirs)} reps, tol {ms_tol:g}%")
    print(f"{'stage':22s} {'base ms':>8s} {'cand ms':>8s} {'delta':>8s} {'base spread':>12s}  verdict")
    print("-" * 78)
    noisy = []
    for name, bv in sorted(bms.items(), key=lambda kv: -kv[1]):
        cv = cms.get(name)
        if cv is None:
            fails.append(f"{name}: missing on candidate")
            print(f"{name:22s} {'MISSING':>8s}")
            continue
        delta = (cv - bv) / bv * 100 if bv else 0.0
        bs = spread([float(load(d)[1][name]["ms"]) for d in bdirs])
        # a stage smaller than min_ms is noise: report it, never fail on it
        gated = bv >= min_ms and cv >= min_ms
        if not gated:
            noisy.append(name)
        slow = gated and delta > ms_tol
        if slow:
            fails.append(f"{name}: speed {delta:+.1f}% > {ms_tol:g}%")
        print(f"{name:22s} {bv:8.2f} {cv:8.2f} {delta:+7.1f}% {bs:11.1f}%  "
              f"{'FAIL' if slow else ('~' if not gated else 'ok')}")

    rdelta = (cr - br) / br * 100 if br else 0.0
    if rdelta > ms_tol:
        fails.append(f"round: {rdelta:+.1f}% > {ms_tol:g}%")
    print("-" * 78)
    print(f"{'round_ms':22s} {br:8.2f} {cr:8.2f} {rdelta:+7.1f}%  {'':11s}  "
          f"{'FAIL' if rdelta > ms_tol else 'ok'}")

    if noisy:
        print(f"\nNOTE — {len(noisy)} stage(s) below {min_ms:g} ms are reported but not gated "
              f"(run-to-run noise dominates them): {', '.join(noisy[:6])}"
              f"{' ...' if len(noisy) > 6 else ''}")

    if fails:
        print(f"\nFAIL — {len(fails)} regression(s):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS — functionally identical, no stage slower beyond tolerance.")
    return 0


def listing(root: Path) -> int:
    ms, rnd, dirs = medians(root)
    total = sum(ms.values())
    print(f"{root.name}: round {rnd:.1f} ms (median of {len(dirs)}), stages {total:.1f} ms "
          f"({len(ms)} stages, mirrors excluded)")
    for name, v in sorted(ms.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {v:9.2f} ms  {100 * v / total:5.1f}%")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DFlash2 functional + speed regression scaffold")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="run N fixture rounds into runs/<label>")
    c.add_argument("label")
    c.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    c.add_argument("--repeat", type=int, default=3, help="rounds to capture (default 3)")

    k = sub.add_parser("check", help="compare two captures; exit 1 on regression")
    k.add_argument("base")
    k.add_argument("cand")
    k.add_argument("--func-tol", type=float, default=0.0,
                   help="allowed relative L2 per stage (default 0 = bf16-exact)")
    k.add_argument("--ms-tol", type=float, default=5.0,
                   help="allowed slowdown on the median, per stage and round, in %% (default 5)")
    k.add_argument("--min-ms", type=float, default=1.0,
                   help="stages smaller than this are reported but not gated (default 1.0)")

    l = sub.add_parser("list", help="median per-stage ms, slowest first")
    l.add_argument("run")

    a = ap.parse_args(argv)
    if a.cmd == "capture":
        capture(a.label, a.fixture, a.repeat)
        return 0
    if a.cmd == "check":
        return check(RUNS / a.base, RUNS / a.cand, a.func_tol, a.ms_tol, a.min_ms)
    return listing(RUNS / a.run)


if __name__ == "__main__":
    sys.exit(main())
