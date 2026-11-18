#!/usr/bin/env python3
"""The one place in tools/dflash2/ that runs the TensorFold binary.

Every script here that needs the engine — a `draft-check` round, a `serve`, a stage capture — goes
through this module, so the dependency on the Zig side is one file wide and one file deep. Nothing
else under tools/dflash2/ may name `tensorfold-native`; `check_boundary.py` enforces that.

Two things it adds over a bare subprocess call:

1. **A contract replay.** A Python tool that measures the engine is only meaningful if the engine is
   the one it was written against. `verify()` runs `draft-check` over a fixture that ships in this
   repo and compares the drafter's proposals against a recorded set. Stage names, tap convention,
   selector behaviour or the fp8 rounding changing all move that line; when it moves, the measurement
   tools are lying and we want to hear it here rather than in a benchmark table.

2. **A staleness warning.** If any file under `zig/` was modified after the binary was built, the
   binary is not the source tree. That is a warning, not an error: a stale binary is sometimes exactly
   what you want to compare against.

Nothing here decides policy — it does not start, stop or restart a service, and it does not pick which
checkpoint to serve. It only answers "which binary, and is it the engine we think it is".

Use as a library:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import engine
    engine.ensure()                      # version + contract, cached, raises ContractError on drift
    cp = engine.run(["draft-check", ...], cwd=REPO)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
BIN = REPO / "zig-out" / "native" / "bin" / "tensorfold-native"
FIXTURE = TOOLS / "analysis" / "align_bf16" / "fixture" / "window"
STAMP = Path.home() / ".cache" / "tensorfold" / "engine-contract.json"

HF = Path.home() / ".cache" / "huggingface" / "hub"

# The engine's own answer for `draft-check` on `analysis/align_bf16/fixture/window`, one round.
# Recorded 2026-11-18 against tensorfold-native 1.2.9 and the fp8 drafter; refresh it deliberately
# (and say so in the ledger) when the drafter's numerics change on purpose.
CONTRACT = {
    "version": "1.2.9",
    "fixture": "tools/dflash2/analysis/align_bf16/fixture/window",
    "position": 2047,
    "anchor": 77491,
    "proposals": [20, 11, 18, 15, 1, 314, 437],
}


class ContractError(RuntimeError):
    """The engine does not behave the way the Python tools assume."""


def drafter() -> Path:
    """The drafter checkpoint the contract and most captures use (override: TF_DFLASH2_DRAFTER)."""
    p = Path(os.path.expanduser(os.environ.get("TF_DFLASH2_DRAFTER", "~/.cache/tensorfold/dflash2-fp8")))
    if not p.is_dir():
        raise ContractError(f"drafter checkpoint not found: {p} (set TF_DFLASH2_DRAFTER)")
    return p


def target() -> Path:
    """The target snapshot (override: TF_TARGET).

    Only the embed table and the head are read by `draft-check`, so any snapshot of the same
    architecture works; the FP4-Local one is the served target.
    """
    env = os.environ.get("TF_TARGET")
    if env:
        p = Path(os.path.expanduser(env))
        return p
    snaps = sorted((HF / "models--nvidia--Qwen3.8-Flash-Next-FP4-Local" / "snapshots").glob("*"))
    if not snaps:
        raise ContractError("no Qwen3.8-Flash-Next-FP4-Local snapshot under ~/.cache/huggingface/hub")
    return snaps[-1]


def binary() -> Path:
    if not BIN.is_file():
        raise ContractError(
            f"no engine binary at {BIN}\n"
            f"build it with:  cd {REPO} && zig build -Dgpu=spark   (or -Drelease=true for the served build)")
    return BIN


def run(args, **kw) -> subprocess.CompletedProcess:
    """Run the binary with `args`. Text output, checked by default, no shell."""
    cmd = [str(binary())] + [str(a) for a in args]
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, check=kw.pop("check", True), **kw)


def version() -> str:
    """`tensorfold-native --version` without the program name, e.g. '1.2.9'."""
    cp = run(["--version"], check=False)
    out = (cp.stdout or "").strip() or (cp.stderr or "").strip()
    m = re.search(r"(\d+\.\d+\.\d+\S*)", out)
    return m.group(1) if m else (out or "unknown")


def zig_newer_than_binary() -> list[str]:
    """Zig sources touched after the binary was built (empty when the binary is current)."""
    b = binary()
    built = b.stat().st_mtime
    newer = []
    for root, _dirs, files in os.walk(REPO / "zig"):
        for f in files:
            if f.endswith((".zig", ".cu", ".cuh", ".h")):
                p = Path(root) / f
                if p.stat().st_mtime > built:
                    newer.append(str(p.relative_to(REPO)))
    return sorted(newer)[:8]


def _stamp_key() -> str:
    b = binary()
    st = b.stat()
    return f"{b}:{st.st_size}:{int(st.st_mtime)}:{version()}"


def _parse_round(text: str) -> dict | None:
    """Read the `draftfix: rep N position P ... anchor A proposals { ... }` line."""
    for line in reversed(text.splitlines()):
        if "proposals" not in line:
            continue
        m = re.search(r"position (\d+) .*anchor (\d+) proposals \{([^}]*)\}", line)
        if not m:
            continue
        ids = [int(x) for x in re.findall(r"-?\d+", m.group(3))]
        return {"position": int(m.group(1)), "anchor": int(m.group(2)), "proposals": ids}
    return None


def verify(work: Path | None = None) -> dict:
    """Replay the contract round and compare. Raises ContractError on any difference."""
    work = work or Path(os.environ.get("TF_CONTRACT_WORK", "/tmp/dflash2-contract"))
    work.mkdir(parents=True, exist_ok=True)
    out = work / "round"
    out.mkdir(parents=True, exist_ok=True)
    cp = run(["draft-check",
              "--drafter", str(drafter()), "--drafter-precision", "fp8",
              "--target", str(target()),
              "--fixture", str(REPO / CONTRACT["fixture"]),
              "--out", str(out), "--repeat", "1"], check=False)
    text = (cp.stdout or "") + (cp.stderr or "")
    if cp.returncode != 0:
        raise ContractError(f"contract replay failed (exit {cp.returncode}):\n{text.strip()[-1500:]}")
    got = _parse_round(text)
    if got is None:
        raise ContractError(f"contract replay produced no proposal line:\n{text.strip()[-1500:]}")
    diffs = [k for k in ("position", "anchor", "proposals") if got[k] != CONTRACT[k]]
    if diffs:
        raise ContractError(
            "engine contract drifted: " + ", ".join(f"{k} {CONTRACT[k]} -> {got[k]}" for k in diffs) +
            "\nIf the change was intended, update CONTRACT in tools/dflash2/engine.py and note it in "
            "docs/design/flashnext-dflash2-experiment-ledger.md; if not, every measurement in this "
            "tree is suspect.")
    return got


def ensure(verify_contract: bool = True, quiet: bool = True) -> dict:
    """Version, staleness and (once per binary) the contract replay. Returns a small report dict."""
    rep: dict = {"binary": str(binary()), "version": version()}
    stale = zig_newer_than_binary()
    if stale:
        rep["stale"] = stale
        if not quiet:
            print(f"engine: warning, {len(stale)} zig source(s) newer than the binary, e.g. {stale[0]}",
                  file=sys.stderr)
    if not verify_contract or os.environ.get("TF_SKIP_CONTRACT") == "1":
        return rep
    key = _stamp_key()
    cached = {}
    try:
        cached = json.loads(STAMP.read_text())
    except (OSError, ValueError):
        pass
    if cached.get("key") == key and cached.get("ok"):
        rep["contract"] = "cached"
        return rep
    got = verify()
    rep["contract"] = got
    try:
        STAMP.parent.mkdir(parents=True, exist_ok=True)
        STAMP.write_text(json.dumps({"key": key, "ok": True, "at": time.strftime("%FT%T"), **got}))
    except OSError:
        pass
    if not quiet:
        print(f"engine: contract ok on {got['proposals']}", file=sys.stderr)
    return rep


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Show the engine this tree measures, and verify it.")
    ap.add_argument("--verify", action="store_true", help="replay the contract round even if cached")
    ap.add_argument("--probe", metavar="ARG", nargs=argparse.REMAINDER,
                    help="pass the remaining arguments to the binary unchanged")
    a = ap.parse_args(argv)
    if a.probe:
        return run(a.probe, check=False).returncode
    rep = ensure(verify_contract=not a.verify, quiet=False)
    if a.verify:
        rep["contract"] = verify()
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
