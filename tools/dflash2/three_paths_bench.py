#!/usr/bin/env python3
"""Three-path serving benchmark on one build: MTP, the DFlash2 drafter, and no drafts.

Who reads the JSONL, and why
----------------------------
The gate in `docs/design/flashnext-dflash2-drafter.md` section 5 is not "does the drafter run" but
"does it beat MTP on this box at lanes 1/2/4, and does it return the same tokens". Both questions
need numbers from the *same build*, so this script writes one JSON line per measurement into
`--out` and never overwrites: a run that dies in rep 2 of the 4-lane cell leaves a usable file, and
the next invocation skips the cells already in it (`--out` resume). The plan's step-5 table, a PR
comment, or a receipt reader takes the per-cell medians and the sha gate from the report; the JSONL
is the evidence behind them. A row with a non-null `error` is not a measurement and is not counted:
it says what failed, not what was measured.

Protocol
--------
* One *cell* is (path, lanes, sampling). Before its first measured rep the cell gets one unmeasured
  warm pass of the same prompts, so kernels are hot and the prefix cache holds the same text; later
  reps inherit that warmth (`--warm-each-rep` re-warms before every rep). Every measured row carries
  `warm` explicitly; `--cold` skips the warm pass and labels the rows false. A failed warm pass is
  never papered over with a cold measurement: the cell is skipped and the failure is a row.
* `--reps` batches per cell, default 3, and never below 3 (`REPS_FLOOR`): the plan's floor, not a
  preference. `decode_tps` is the engine's own `tensorfold.tokens_per_second` per request, reported
  as the median over the cell's lanes; `aggregate_tps` is the batch's completion tokens over the
  batch's wall clock, so it includes prefill.
* Greedy (temperature 0) is the equivalence row: every path must return the same
  `tensorfold.token_sha256`, because DFlash2's verification is lossless. `--sampling both` adds a
  sampled row at temperature 1.0 / top_p 0.95 / top_k 20, which is the card's operating point and
  is not gated on sha.
* Thinking is off (`chat_template_kwargs.enable_thinking`) and the prompt asks for the integers
  1..400 comma separated, so the model does not stop at 25 tokens and call it an answer.
* The `seed` is sent and fixed across reps, so MTP and DFlash2 see the same request twice; reps
  therefore measure speed variance, not sampling variance.

The JSONL row
-------------
One object per line, in this order: `label`, `path`, `lanes`, `rep`, `warm`, `sampling`,
`temperature`, `max_tokens`, `seed`, `base_url`, `model`, `ttft_s`, `decode_tps`, `aggregate_tps`,
`prefill_s`, `prompt_tokens` (per request, median over lanes), `completion_tokens` (batch total),
`cached_tokens` (batch total), `token_sha256` (lane 0's; all lanes are in `streams`), `finish_reason`,
`drafts` (the server's own boolean), `drafted`, `accepted`, `acceptance_rate`, `requests`,
`requests_ok`, `wall_s`, `streams` (per-lane metrics and shas), `ts`, `error`.
`rep` is an integer for a measurement and `null` for a cell-level failure such as a failed warm
pass. `ttft_s` and `decode_tps` are null exactly when the row is not a measurement.

How a path is selected on this build (read from `zig/src/server`, 3e8531e6a)
--------------------------------------------------------------------------
    mtp       `tensorfold-native serve ...` with no --drafter. Requests draft by default.
    dflash2   the same binary with `--drafter <incoai/Qwen3.8-Flash-Next-DFlash2>`; requests draft.
    nodrafts  the request field `"draft": false` (docs/api.md), honoured by any drafted server.

There is no per-request field that picks MTP against a loaded DFlash2 drafter: the choice is the
serve flag, one process at a time. So mtp and dflash2 are the same *build* started twice, and one
`--out` collects both; nodrafts is measured on whichever server is running, which is why an mtp or
dflash2 invocation measures two of the three paths on the same process. If a per-request switch
lands, name it in `PATH_SWITCH` below; nothing else here changes.

This script never starts, stops or kills a server, and refuses to talk to the production port.

Usage
-----
    BASE_URL=http://127.0.0.1:8099/v1 MODEL=qwen3.8-flash-next \\
    python3 tools/dflash2/three_paths_bench.py --label df2-a --path-mode mtp \\
        --out tools/dflash2/run/three_paths.jsonl

    # same build, restarted with --drafter <dir>, into the same file; the report then compares both
    python3 tools/dflash2/three_paths_bench.py --label df2-a --path-mode dflash2 \\
        --out tools/dflash2/run/three_paths.jsonl --sampling both

    # re-read the file without touching a server
    python3 tools/dflash2/three_paths_bench.py --report-only --out tools/dflash2/run/three_paths.jsonl

Python 3 stdlib only; no requests, no torch, no build.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# --- what the environment supplies --------------------------------------------------------------

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8099/v1")
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
LABEL = os.environ.get("LABEL") or "unlabelled"
OUT = os.environ.get("OUT", "tools/dflash2/run/three_paths.jsonl")

# The production service. A typo must not send benchmark traffic at it.
REFUSED_PORTS = {8005}

REPS_FLOOR = 3  # the plan's floor: --reps below this is a usage error, not a warning
LANES_DEFAULT = "1,2,4"
MIN_COMPLETION_TOKENS = 64  # shorter replies make decode rate meaningless
PATHS = ("mtp", "dflash2", "nodrafts")

# Request fields per path. mtp and dflash2 differ only in the server's `--drafter` flag, so both
# carry nothing but the explicit drafts-on field; see the module docstring for the selection table.
#
# TODO(path switch): if the build grows a per-request MTP/DFlash2 selector, put the field here
# (e.g. `{"drafter": "dflash2"}`) and the server-flag caveat above stops applying. Until then the
# two drafted paths cannot be measured in one run and `--path-mode` names the server's own path.
PATH_SWITCH = {
    "mtp": {"draft": True},
    "dflash2": {"draft": True},
    "nodrafts": {"draft": False},
}

# temperature 0 is the equivalence row; the sampled row is the HF card's operating point.
SAMPLING = {
    "greedy": {"temperature": 0},
    "sampled": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
}

PROMPT_TEMPLATE = (
    "Write the integers from {lo} to {hi}, inclusive, in increasing order, each separated by a "
    "comma and a single space, on one line. Output the numbers and the separators only: no "
    "preamble, no closing sentence, no ellipsis, no thousands grouping. Do not stop before {hi}."
)


class RequestError(Exception):
    """A request the harness refuses to measure: HTTP, missing block, missing field."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def prompt_for(index: int, numbers: int) -> str:
    """Lane `index` gets its own range so concurrent lanes do not share one cached prompt."""
    lo = 1 + index * numbers
    return PROMPT_TEMPLATE.format(lo=lo, hi=lo + numbers - 1)


def median(values) -> float | None:
    xs = [v for v in values if isinstance(v, (int, float))]
    return float(statistics.median(xs)) if xs else None


def num(value, digits: int = 1, suffix: str = "") -> str:
    """Report a cell without inventing one: a number that was never returned prints n/a."""
    return f"{value:.{digits}f}{suffix}" if isinstance(value, (int, float)) else "n/a"


def post_chat(base_url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300].replace("\n", " ")
        raise RequestError(f"HTTP {e.code}: {detail}") from None
    except Exception as e:  # noqa: BLE001 - timeouts, resets, a dead port
        raise RequestError(f"{type(e).__name__}: {e}") from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RequestError(f"reply is not JSON: {e}") from None


def one_request(
    base_url: str,
    model: str,
    prompt: str,
    path: str,
    sampling: str,
    max_tokens: int,
    seed: int,
    timeout: float,
) -> dict:
    """One greedy or sampled completion, reduced to the fields the table needs."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "seed": seed,
    }
    body.update(SAMPLING[sampling])
    body.update(PATH_SWITCH[path])
    reply = post_chat(base_url, body, timeout)

    block = reply.get("tensorfold")
    if not isinstance(block, dict):
        raise RequestError("reply has no tensorfold block: the engine reported no runtime stats")
    for key in ("tokens_per_second", "time_to_first_token", "token_sha256"):
        if block.get(key) in (None, ""):
            raise RequestError(f"tensorfold.{key} is missing, so this path cannot be compared")

    usage = reply.get("usage") or {}
    speculative = block.get("speculative") or {}
    choice = (reply.get("choices") or [{}])[0]
    return {
        "ttft_s": block["time_to_first_token"],
        "decode_tps": block["tokens_per_second"],
        "prefill_s": block.get("prefill_seconds"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
        "token_sha256": block["token_sha256"],
        "finish_reason": choice.get("finish_reason"),
        "drafts": block.get("drafts"),
        "drafted": speculative.get("drafted"),
        "accepted": speculative.get("accepted"),
    }


def run_batch(base_url: str, prompts: list[str], path: str, sampling: str, args) -> tuple[list, float]:
    """`lanes` requests released together; wall clock spans prefill and decode of the whole batch."""
    results: list = [None] * len(prompts)
    go = threading.Event()

    def worker(index: int) -> None:
        go.wait()
        try:
            results[index] = one_request(
                base_url,
                args.model,
                prompts[index],
                path,
                sampling,
                args.max_tokens,
                args.seed,
                args.timeout,
            )
        except RequestError as e:
            results[index] = {"error": str(e)}
        except Exception as e:  # noqa: BLE001 - a broken lane must not hang the batch
            results[index] = {"error": f"{type(e).__name__}: {e}"}

    workers = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(prompts))]
    for w in workers:
        w.start()
    started = time.perf_counter()
    go.set()
    for w in workers:
        w.join()
    return results, time.perf_counter() - started


def build_row(
    label: str, path: str, lanes: int, rep: int | None, warm: bool, sampling: str, results, wall, args
) -> dict:
    ok = [r for r in results if "error" not in r]
    failures = [r["error"] for r in results if "error" in r]
    drafted = [r.get("drafted") for r in ok if isinstance(r.get("drafted"), (int, float))]
    accepted = [r.get("accepted") for r in ok if isinstance(r.get("accepted"), (int, float))]
    return {
        "label": label,
        "path": path,
        "lanes": lanes,
        "rep": rep,
        "warm": warm,
        "sampling": sampling,
        "temperature": SAMPLING[sampling]["temperature"],
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "base_url": args.base_url,
        "model": args.model,
        "ttft_s": median([r["ttft_s"] for r in ok]),
        "decode_tps": median([r["decode_tps"] for r in ok]),
        "aggregate_tps": (
            sum(r["completion_tokens"] or 0 for r in ok) / wall if ok and wall > 0 else None
        ),
        "prefill_s": median([r.get("prefill_s") for r in ok]),
        "prompt_tokens": median([r.get("prompt_tokens") for r in ok]),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in ok) if ok else None,
        "cached_tokens": sum(r.get("cached_tokens") or 0 for r in ok) if ok else None,
        "token_sha256": ok[0]["token_sha256"] if ok else None,
        "finish_reason": ok[0].get("finish_reason") if ok else None,
        "drafts": ok[0].get("drafts") if ok else None,
        "drafted": sum(drafted) if drafted else None,
        "accepted": sum(accepted) if accepted else None,
        "acceptance_rate": (sum(accepted) / sum(drafted)) if drafted and sum(drafted) else None,
        "requests": lanes,
        "requests_ok": len(ok),
        "wall_s": round(wall, 3),
        # per-lane shas, so the equivalence gate compares the same prompt across paths
        "streams": [
            {
                "ttft_s": r.get("ttft_s"),
                "decode_tps": r.get("decode_tps"),
                "completion_tokens": r.get("completion_tokens"),
                "cached_tokens": r.get("cached_tokens"),
                "token_sha256": r.get("token_sha256"),
                "finish_reason": r.get("finish_reason"),
                "error": r.get("error"),
            }
            for r in results
        ],
        "ts": now(),
        "error": "; ".join(failures)[:600] if failures else None,
    }


def append_row(out: Path, row: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def cell_key(label: str, path: str, lanes: int, rep: int, sampling: str) -> tuple:
    return (label, path, int(lanes), int(rep), sampling)


def load_rows(out: Path) -> list[dict]:
    if not out.exists():
        return []
    rows: list[dict] = []
    for number, line in enumerate(out.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"# warning: {out}:{number} is not JSON, ignored", file=sys.stderr)
    return rows


def done_keys(rows: list[dict]) -> set[tuple]:
    """Only a row with a number in it counts as done; a failed cell is retried on resume."""
    keys = set()
    for r in rows:
        if r.get("error") is not None or r.get("rep") is None or r.get("ttft_s") is None:
            continue
        keys.add(cell_key(r["label"], r["path"], r["lanes"], r["rep"], r.get("sampling", "greedy")))
    return keys


def warm_cell(args, path: str, lanes: int, sampling: str, prompts: list[str], out: Path) -> bool:
    """The unmeasured pass; a failure is recorded and the cell is not measured cold in its place."""
    results, wall = run_batch(args.base_url, prompts, path, sampling, args)
    if not any("error" in r for r in results):
        return True
    row = build_row(args.label, path, lanes, None, False, sampling, results, wall, args)
    append_row(out, row)
    print(f"# warm pass failed, cell skipped: {path} lanes={lanes} {sampling}", file=sys.stderr)
    return False


def measure_cell(args, path: str, lanes: int, sampling: str, missing: list[int], out: Path) -> None:
    prompts = [prompt_for(i, args.numbers) for i in range(lanes)]
    warmed = not args.cold  # --cold: no warm pass, and the rows say so
    if warmed and not warm_cell(args, path, lanes, sampling, prompts, out):
        return
    for rep in missing:
        if warmed and args.warm_each_rep and not warm_cell(args, path, lanes, sampling, prompts, out):
            return
        results, wall = run_batch(args.base_url, prompts, path, sampling, args)
        row = build_row(args.label, path, lanes, rep, warmed, sampling, results, wall, args)
        append_row(out, row)
        rate = f"{row['decode_tps']:.1f} tok/s" if row["decode_tps"] is not None else "no rate"
        aggregate = (
            f"{row['aggregate_tps']:.1f} tok/s aggregate"
            if row["aggregate_tps"] is not None
            else "no aggregate"
        )
        state = "warm" if warmed else "cold"
        print(f"# {path} lanes={lanes} {sampling} rep={rep} {state}: {rate}, {aggregate}", file=sys.stderr)


# --- report ------------------------------------------------------------------------------------


def measured(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("error") is None and r.get("ttft_s") is not None and r.get("rep") is not None]


def cell_lines(rows: list[dict]) -> list[str]:
    """One bullet per cell, median over its reps, floor called out when it is not met."""
    cells: dict[tuple, list[dict]] = {}
    for r in measured(rows):
        cells.setdefault((r["path"], r["lanes"], r.get("sampling", "greedy")), []).append(r)
    order = {p: i for i, p in enumerate(PATHS)}
    lines = []
    for (path, lanes, sampling), group in sorted(
        cells.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1], kv[0][2])
    ):
        n = len(group)
        floor = "" if n >= REPS_FLOOR else " (below floor)"
        cold = "" if all(r.get("warm") for r in group) else " (cold)"
        parts = [f"`{path}` lanes={lanes} {sampling}"]
        parts.append(f"decode {num(median(r['decode_tps'] for r in group), suffix=' tok/s')}")
        if median(r["aggregate_tps"] for r in group) is not None:
            parts.append(f"aggregate {num(median(r['aggregate_tps'] for r in group), suffix=' tok/s')}")
        parts.append(f"ttft {num(median(r['ttft_s'] for r in group), 3, ' s')}")
        if median(r.get("prefill_s") for r in group) is not None:
            parts.append(f"prefill {num(median(r['prefill_s'] for r in group), 3, ' s')}")
        if median(r.get("cached_tokens") for r in group) is not None:
            parts.append(f"cached {num(median(r['cached_tokens'] for r in group), 0, ' tok')}")
        parts.append(f"n={n}{floor}{cold}")
        lines.append(" | ".join(parts))
    return lines


def equivalence_lines(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Greedy shas grouped by (lanes, rep, lane index): every path must agree, byte for byte."""
    groups: dict[tuple, dict[str, str]] = {}
    for r in measured(rows):
        if r.get("sampling", "greedy") != "greedy":
            continue
        for index, stream in enumerate(r.get("streams") or []):
            sha = stream.get("token_sha256")
            if sha:
                groups.setdefault((r["lanes"], r["rep"], index), {})[r["path"]] = sha
    tally: dict[tuple, list[int]] = {}
    problems: list[str] = []
    for (lanes, rep, index), shas in groups.items():
        if len(shas) < 2:
            continue
        bucket = tally.setdefault((lanes, tuple(sorted(shas))), [0, 0])
        bucket[0] += 1
        if len(set(shas.values())) > 1:
            bucket[1] += 1
            pairs = " != ".join(f"{p} {s[:8]}..." for p, s in sorted(shas.items()))
            problems.append(f"lanes={lanes} rep={rep} prompt {index}: {pairs}")
    lines = []
    for (lanes, paths), (total, bad) in sorted(tally.items()):
        verdict = f"{total} of {total} identical" if bad == 0 else f"DIFFER in {bad} of {total}"
        lines.append(" vs ".join(f"`{p}`" for p in paths) + f" | lanes={lanes} | {verdict}")
    return lines, problems


def protocol_errors(rows: list[dict]) -> list[str]:
    """Rows that ran but cannot be the path they claim, or are too short to mean anything."""
    problems = []
    for r in measured(rows):
        where = f"{r['label']} {r['path']} lanes={r['lanes']} rep={r['rep']}"
        if r["path"] in ("mtp", "dflash2") and r.get("drafts") is False:
            problems.append(
                f"{where}: server reported tensorfold.drafts=false, so this is not the drafted path "
                "(was it started with --no-drafts?)"
            )
        if r["path"] == "nodrafts" and r.get("drafts") is True:
            problems.append(f"{where}: server reported drafts=true, so it ignored \"draft\": false")
        tokens = r.get("completion_tokens")
        if isinstance(tokens, (int, float)) and tokens < MIN_COMPLETION_TOKENS:
            problems.append(f"{where}: only {int(tokens)} completion tokens, decode rate is not meaningful")
    return problems


def report(rows: list[dict], requested: list[tuple], args, out: Path) -> int:
    good = measured(rows)
    bad = [r for r in rows if r.get("error") is not None]
    print(f"## three_paths_bench: {out}")
    print(f"- measured rows: {len(good)}; error rows: {len(bad)}; floor: {REPS_FLOOR} reps per cell")
    if not good:
        print("- not measured: the file holds no measurement yet")

    labels = sorted({r["label"] for r in rows}) or [args.label]
    for label in labels:
        own = [r for r in rows if r["label"] == label]
        base = next((r["base_url"] for r in own if r.get("base_url")), args.base_url)
        model = next((r["model"] for r in own if r.get("model")), args.model)
        print(f"\n### {label} | {base} | model={model}")
        print("medians over reps, thinking off:")
        lines = cell_lines(own)
        for line in lines:
            print(f"- {line}")
        if not lines:
            print("- not measured: no cell for this label")

        if requested and label == args.label:  # cells this invocation asked for, not another build's
            absent = [
                (path, lanes, sampling)
                for path, lanes, sampling in requested
                if not any(
                    r["path"] == path
                    and r["lanes"] == lanes
                    and r.get("sampling", "greedy") == sampling
                    for r in measured(own)
                )
            ]
            if absent:
                print("\nrequested cells with no measurement in this file:")
                for path, lanes, sampling in absent:
                    print(f"- `{path}` lanes={lanes} {sampling}: not measured")

        print("\nequivalence, greedy `tensorfold.token_sha256` on the same prompt:")
        same, problems = equivalence_lines(own)
        if same:
            for line in same:
                print(f"- {line}")
        else:
            print("- not measured: fewer than two paths with greedy rows")
        for problem in problems[:6]:
            print(f"- ERROR: token_sha256 mismatch, {problem}")
        if len(problems) > 6:
            print(f"- ERROR: token_sha256 mismatch in {len(problems) - 6} more prompt-reps")

        errors = protocol_errors(own)
        failed = [r for r in own if r.get("error") is not None]
        if errors or failed:
            print("\nfailures:")
            for e in errors[:10]:
                print(f"- ERROR: {e}")
            for r in failed[:10]:
                rep = r.get("rep")
                where = f"{r['path']} lanes={r['lanes']} rep={rep if rep is not None else 'warm'}"
                print(f"- ERROR: {where}: {r['error']}")

    coverage = []
    for path in PATHS:
        n = sum(1 for r in good if r["path"] == path)
        coverage.append(f"`{path}`: {f'{n} rows' if n else 'not measured'}")
    print("\ncoverage: " + "; ".join(coverage))
    return 1 if bad or any(protocol_errors(rows)) else 0


# --- driver ------------------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Measure MTP, DFlash2 and no-drafts on one build, one JSONL line per measurement."
    )
    p.add_argument("--label", default=LABEL, help="names the build and its serve flags; share it across paths")
    p.add_argument("--base-url", default=BASE_URL, help="e.g. http://127.0.0.1:8099/v1")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default=OUT, help="JSONL receipt; appended to and resumed from")
    p.add_argument(
        "--path-mode",
        choices=PATHS,
        default="mtp",
        help="the path this server is serving: mtp (no --drafter) or dflash2 (--drafter); both also "
        "measure the no-drafts floor on the same process. nodrafts measures only the floor.",
    )
    p.add_argument("--lanes", default=LANES_DEFAULT, help="comma separated concurrency levels")
    p.add_argument("--reps", type=int, default=3, help=f"measured batches per cell, at least {REPS_FLOOR}")
    p.add_argument("--sampling", choices=("greedy", "both"), default="greedy", help="both adds the sampled row")
    p.add_argument("--numbers", type=int, default=400, help="how many integers the prompt asks for")
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--seed", type=int, default=1234, help="fixed so both paths see the same request")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--cold", action="store_true", help="skip the unmeasured warm pass and label rows warm=false")
    p.add_argument(
        "--warm-each-rep",
        action="store_true",
        help="warm before every rep, not only before the cell's first one",
    )
    p.add_argument("--report-only", action="store_true", help="read --out and print the report; no requests")
    a = p.parse_args(argv)
    if a.reps < REPS_FLOOR:
        p.error(f"--reps {a.reps} is below the floor of {REPS_FLOOR}: a cell needs at least that many")
    try:
        a.lane_counts = sorted({int(x) for x in a.lanes.split(",") if x.strip()})
    except ValueError:
        p.error(f"--lanes '{a.lanes}' is not a comma separated list of integers")
    if not a.lane_counts or any(n < 1 for n in a.lane_counts):
        p.error("--lanes needs at least one positive lane count")
    if a.numbers < 1:
        p.error("--numbers must ask for at least one integer")
    return a


def refuse_production(base_url: str) -> None:
    port = None
    try:
        port = urllib.parse.urlparse(base_url).port
    except Exception:  # noqa: BLE001 - a malformed URL fails below with a clearer message
        port = None
    if port in REFUSED_PORTS or any(f":{p}" in base_url for p in REFUSED_PORTS):
        raise SystemExit(
            f"refusing {base_url}: port {sorted(REFUSED_PORTS)} serves production from this worktree"
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    refuse_production(args.base_url)
    out = Path(args.out)
    rows = load_rows(out)
    sampling = ["greedy"] if args.sampling == "greedy" else ["greedy", "sampled"]

    if args.report_only:
        return report(rows, [], args, out)

    paths = ["nodrafts"] if args.path_mode == "nodrafts" else [args.path_mode, "nodrafts"]
    requested = [(path, lanes, s) for path in paths for lanes in args.lane_counts for s in sampling]
    done = done_keys(rows)
    print(
        f"# {args.label} | {args.base_url} | model={args.model} | paths={paths} "
        f"lanes={args.lane_counts} reps={args.reps} sampling={sampling}",
        file=sys.stderr,
    )

    for path in paths:
        for lanes in args.lane_counts:
            for s in sampling:
                missing = [
                    rep
                    for rep in range(args.reps)
                    if cell_key(args.label, path, lanes, rep, s) not in done
                ]
                if not missing:
                    print(f"# skip {path} lanes={lanes} {s}: all reps already in {out}", file=sys.stderr)
                    continue
                measure_cell(args, path, lanes, s, missing, out)

    rows = load_rows(out)
    return report(rows, requested, args, out)


if __name__ == "__main__":
    raise SystemExit(main())
