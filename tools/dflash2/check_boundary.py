#!/usr/bin/env python3
"""Static guard for the tools/dflash2 layout: the Python side must stay a *client* of the engine.

Three rules, each of which has already been broken at least once in this tree:

  layout      tools/dflash2/ holds code only at engine.py and check_boundary.py; everything else
              lives in harness/, analysis/, quantize/, train/ or the two data dirs.
  chokepoint  no Python file except engine.py mentions the engine binary. If a tool needs a round,
              a capture or a serve, it imports engine and calls engine.run().
  no-zig      no Python imports the Zig side. The engine is reached through a process boundary or
              not at all; an import means the two halves are welded together.

Shell scripts are exempt (they cannot import), but they are listed so the exemption stays visible.

    python3 tools/dflash2/check_boundary.py [--verbose]

Exit status: 0 clean, 1 violations.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ALLOWED_TOP = {"engine.py", "check_boundary.py", "README.md"}
ALLOWED_DIRS = {"harness", "analysis", "quantize", "train", "run", "runs", "__pycache__"}
ALLOWED_CODE = {TOOLS / "engine.py", TOOLS / "check_boundary.py"}
BINARY_TOKENS = ("tensorfold-native",)
SHELLS = (".sh",)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """id() of Constant nodes that are docstrings, so quoting the path in prose is not a violation."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


def check_file(path: Path) -> list[str]:
    bad: list[str] = []
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as e:
        return [f"{path.relative_to(TOOLS)}: does not parse: {e}"]
    docstrings = _docstring_nodes(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "zig":
                    bad.append(f"{path.relative_to(TOOLS)}:{node.lineno}: imports the Zig side ({a.name})")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "zig":
                bad.append(f"{path.relative_to(TOOLS)}:{node.lineno}: imports the Zig side ({node.module})")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings or path in ALLOWED_CODE:
                continue
            for token in BINARY_TOKENS:
                if token in node.value:
                    bad.append(
                        f"{path.relative_to(TOOLS)}:{node.lineno}: names {token!r} outside engine.py "
                        f"— route it through engine.run()")
                    break
    return bad


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=(__doc__ or "boundary check").splitlines()[0])
    ap.add_argument("--verbose", action="store_true", help="list every file checked")
    a = ap.parse_args(argv)

    problems: list[str] = []

    for entry in sorted(TOOLS.iterdir()):
        if entry.is_dir():
            if entry.name not in ALLOWED_DIRS:
                problems.append(f"layout: unexpected directory tools/dflash2/{entry.name}/")
        elif entry.name not in ALLOWED_TOP:
            problems.append(f"layout: unexpected file at tools/dflash2/{entry.name} "
                            f"— code belongs in a subpackage, with engine.py as the only exception")

    files = sorted(p for p in TOOLS.rglob("*.py")
                   if not ({"__pycache__", ".venv"} & set(p.parts)))
    for p in files:
        if a.verbose:
            print(f"checking {p.relative_to(TOOLS)}")
        problems += check_file(p)

    shells = sorted(p.relative_to(TOOLS) for p in TOOLS.rglob("*.sh")
                    if ".venv" not in p.parts)
    for s in shells:
        if a.verbose:
            print(f"exempt (shell): {s}")

    if problems:
        print(f"{len(problems)} boundary violation(s):\n")
        for p in problems:
            print("  " + p)
        return 1
    print(f"ok: {len(files)} python files, {len(shells)} shell script(s) exempt, "
          f"engine reached only through engine.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
