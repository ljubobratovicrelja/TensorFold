#!/usr/bin/env python3
"""The context-fill matrix: lanes 1/2/4 against fills 0, 20k and 100k prompt tokens, one build.

Why a second benchmark
----------------------
`three_paths_bench.py` answers "does DFlash2 beat MTP on the same tokens". It asks that with one
short prompt, so its lanes=4 cell measures batch decode on a warm, cheap prefix. That is not the
shape of a serving day: agents carry long conversations, and the question the operator actually has
is what four concurrent 100k-token streams cost. That question has two halves, and they move in
opposite directions. Prefill is paid once per new prefix (a cold fill is ~10 s at 20k and ~50 s at
100k on this box, serially when several agents start together, per `docs/cache-benchmarks.md`); a
repeat turn hits the imprint and costs nothing. So a table that mixes cold fills into its decode
medians quotes first-turn cost as a serving rate, and one that measures only 0-token prompts never
sees the fill at all. This script measures nine cells per build, names the fill of every row, and
refuses to average a row whose prefix was not actually in the cache.

One path per run
----------------
MTP against DFlash2 is a `--drafter` serve flag (`docs/design/flashnext-dflash2-drafter.md`,
section 2). Nothing in a request can pick between them, so one invocation measures one path, and
`--path-mode` is a claim about the server the operator started, not a request field. The report
restates that claim in every block, with the `/v1/models` id and the exact `--base-url` it was
measured on, so a table cannot be read as another build. Run the two paths into one `--out` and the
report prints one section per (label, path-mode); nothing is compared across them here.

Protocol
--------
* A *fill* is a prompt of approximately F tokens: a deterministic word stream, then a fixed
  instruction, so decoding does not stop at 25 tokens and call it an answer. F is not a guess:
  `calibrate()` posts one small probe, reads `usage.prompt_tokens` from the server's own tokenizer,
  and rescales the word count until the real fill is within +-5% (`FILL_TOLERANCE`) of the target.
  Text and its measured count are cached in the run directory as `fill-<F>.txt` / `fill-<F>.json`,
  so a resumed run reuses them and pays no second probe (a 100k probe is 50 s). The cache is
  invalidated when the served model id changes, because the tokenizer may have.
* Every cell is measured WARM and every row says so (`warm: true`). Before its reps the cell runs
  one unmeasured pass of the exact same prompts: same text, same lane count, same seeds, same
  user strings. That pass puts the prefix in the prompt cache and the kernels in their steady
  state; the measured reps then hit both. A failed warm pass is recorded as a row (rep null) and
  the cell is skipped -- it is never quietly measured cold in its place.
* Each measured row carries `cached_tokens`. A row with fill > 0 and zero cached prompt tokens did
  not measure a warm stream, whatever it says about speed: it is listed under protocol failures,
  left out of the cell's medians, and makes the run exit 1. Native Zig reports zero here until the
  prompt cache is enabled (`docs/api.md`), so a server started without `--prompt-cache-gib` fails
  every fill row by construction. That is a serve-flag fact, not a rep to retry, so such a row
  still counts as done on resume.
* Lanes post one request each, released together, with the *same* prompt on purpose (a shared warm
  prefix is the thing being measured) and a distinct `user` and `seed` per lane so a build that
  keys a stream or cache row on either does not fold the lanes into one. This build ignores `user`
  (unknown chat fields are not refused, and no route reads it); the seed is what it honours.
* `decode_tps` is the engine's own `tensorfold.tokens_per_second` per request, reported as the
  median over the cell's lanes; `aggregate_tps` is the batch's completion tokens over the batch's
  wall clock, so it includes prefill and is the number to quote for "four agents at 100k".
  `max_tokens` defaults to 256 and `temperature` is 0, so reps measure speed variance, not sampling.

The server is started by the operator, never here. It needs `--context` above the fill plus
`--max-tokens`, `--parallel` at least the largest lane count, and a prompt cache large enough to
hold lanes x fill, or the largest cells lose their warm pass by design. This script refuses the
production port and never spawns or kills anything.

The JSONL row
-------------
One object per line, in this order: `label`, `path_mode`, `lanes`, `fill`, `rep`, `warm`,
`base_url`, `model` (the request field), `model_id` (what `/v1/models` said), `max_tokens`, `seed`,
`temperature`, `ttft_s` (median over lanes), `ttft_max_s` (worst lane), `decode_tps`,
`aggregate_tps`, `prefill_s`, `prompt_tokens` (median, the achieved fill), `completion_tokens`
(batch total), `cached_tokens` (batch total), `token_sha256` (lane 0; all lanes are in `streams`),
`finish_reason`, `requests`, `requests_ok`, `wall_s`, `streams` (per-lane metrics), `ts`, `error`.
`rep` is 0-based in the row and 1-based in the progress line. A line with a non-null `error` is not
a measurement and is not counted; `rep` is null for a cell-level failure such as a failed warm pass.

Usage
-----
    # one server, one path; the operator started it with the --drafter flag or without
    BASE_URL=http://127.0.0.1:8099/v1 MODEL=qwen3.8-flash-next \\
    python3 tools/dflash2/matrix_bench.py --label df2-a --path-mode mtp

    # the same build restarted with --drafter <dir>, into the same file; the report keeps them apart
    python3 tools/dflash2/matrix_bench.py --label df2-a --path-mode dflash2 --reps 3

    # a half matrix after a crash, or one lane count to re-check
    python3 tools/dflash2/matrix_bench.py --label df2-a --path-mode mtp --only-fills 0,20000
    python3 tools/dflash2/matrix_bench.py --label df2-a --path-mode mtp --only-lanes 4

    # re-read the file without touching a server
    python3 tools/dflash2/matrix_bench.py --report-only --out tools/dflash2/run/matrix.jsonl

Python 3 stdlib only (urllib, no requests, no torch, no build).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
OUT = os.environ.get("OUT", "tools/dflash2/run/matrix.jsonl")

# The production service. A typo must not send benchmark traffic at it.
REFUSED_PORTS = {8005}

REPS_FLOOR = 3  # the plan's floor: --reps below this is a usage error, not a warning
REPS_DEFAULT = 3
LANES = (1, 2, 4)
FILLS = (0, 20000, 100000)
PATH_MODES = ("mtp", "dflash2")

FILL_TOLERANCE = 0.05  # the calibrated prompt must land within +-5% of the target fill
CALIBRATION_TRIES = 4  # probe, rescale, repeat; the count is nearly linear so 2 is usual
PROBE_MAX_TOKENS = 8  # the probe exists to be prefilled, not to be read
DEFAULT_MAX_TOKENS = 256
SHORT_COMPLETION = 64  # below this the decode rate is not meaningful

# A first guess only: calibrate() replaces it with the server's own count on the first probe.
WORDS_PER_TOKEN = 1.3665

# The same word list `tools/bench_serving.py` uses, so a fill here is the same kind of text as the
# published DGX rows.
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray "
    "yankee zulu scheduler kernel latency throughput quantize tensor gradient cache "
    "pointer buffer register pipeline parallel entropy manifold"
).split()

FILL_SEED = 20261116  # the word stream is seeded, not sampled per run

FILL_HEADER = (
    "Below is an excerpt from a machine's log. Read it, then follow the instruction at the end.\n\n"
)
INSTRUCTION = (
    "List the integers from 1 to 400, inclusive, in increasing order, each separated by a comma "
    "and a single space, on one line. Output the numbers and the separators only: no preamble, no "
    "closing sentence, no ellipsis, no thousands grouping. Do not stop before 400."
)


class RequestError(Exception):
    """A request the harness refuses to measure: HTTP, missing block, missing field."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def median(values) -> float | None:
    xs = [v for v in values if isinstance(v, (int, float))]
    return float(statistics.median(xs)) if xs else None


def num(value, digits: int = 1, suffix: str = "") -> str:
    """Report a cell without inventing one: a number that was never returned prints n/a."""
    return f"{value:.{digits}f}{suffix}" if isinstance(value, (int, float)) else "n/a"


def say(line: str) -> None:
    """The progress line: unbuffered, so a 50 s prefill still looks like a running job."""
    print(line, file=sys.stderr, flush=True)


# --- prompts -----------------------------------------------------------------------------------

_WORDS: list[str] = []
_RNG: random.Random | None = None


def word_prefix(count: int) -> str:
    """The first `count` words of one seeded stream, so the 20k fill is a prefix of the 100k one."""
    global _RNG
    if _RNG is None:
        _RNG = random.Random(FILL_SEED)
    while len(_WORDS) < count:
        _WORDS.append(_RNG.choice(WORDS))
    return " ".join(_WORDS[:count])


def prompt_for(count: int) -> str:
    """A fill of approximately `count` words, then the instruction. Deterministic in `count`."""
    if count <= 0:
        return INSTRUCTION
    return FILL_HEADER + word_prefix(count) + "\n\n" + INSTRUCTION


def lane_user(label: str, fill: int, lanes: int, index: int) -> str:
    """Stable across reps on purpose: a rep must reuse the warm pass's key, not mint a new one."""
    return f"matrix-{label}-l{lanes}-f{fill}-lane{index}"


# --- HTTP --------------------------------------------------------------------------------------


def get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


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


def server_facts(base_url: str, timeout: float = 10.0) -> dict:
    """The served id, its prompt-cache capability and the batch size. Best effort: never fatal.

    The report names the id so a table cannot be read as another build; the capability decides
    whether fill rows can be warm at all; the batch size decides whether a lanes=4 cell was a batch
    or a queue.
    """
    facts: dict = {"model_id": None, "prompt_cache": None, "max_batch": None}
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        listing = get_json(base_url.rstrip("/") + "/models", timeout)
        entries = listing.get("data") or []
        if entries and isinstance(entries[0], dict):
            facts["model_id"] = entries[0].get("id")
            capabilities = entries[0].get("capabilities") or {}
            facts["prompt_cache"] = capabilities.get("prompt_cache")
    except Exception:  # noqa: BLE001 - a report without the id is weaker, not wrong
        pass
    try:
        health = get_json(root + "/health", timeout)
        facts["max_batch"] = health.get("max_batch_size")
    except Exception:  # noqa: BLE001
        pass
    return facts


def chat_body(args, prompt: str, seed: int, user: str, max_tokens: int | None = None) -> dict:
    """One chat body. `max_tokens` overrides the cell's limit for a calibration probe."""
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
        "temperature": 0,
        "stream": False,
        "seed": seed,
        "user": user,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def one_request(base_url: str, body: dict, timeout: float, require_rate: bool = True) -> dict:
    """One completion, reduced to the fields the table needs.

    `require_rate` is off for a calibration probe, which needs `usage.prompt_tokens` and nothing
    else: a probe that stops after one token may legitimately report no decode rate.
    """
    reply = post_chat(base_url, body, timeout)
    block = reply.get("tensorfold")
    if not isinstance(block, dict):
        raise RequestError("reply has no tensorfold block: the engine reported no runtime stats")
    if require_rate:
        for key in ("tokens_per_second", "time_to_first_token"):
            if block.get(key) in (None, ""):
                raise RequestError(f"tensorfold.{key} is missing, so this path cannot be compared")
    usage = reply.get("usage") or {}
    choice = (reply.get("choices") or [{}])[0]
    return {
        "ttft_s": block.get("time_to_first_token"),
        "decode_tps": block.get("tokens_per_second"),
        "prefill_s": block.get("prefill_seconds"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
        "token_sha256": block.get("token_sha256"),
        "finish_reason": choice.get("finish_reason"),
    }


def run_batch(base_url: str, bodies: list[dict], timeout: float) -> tuple[list, float]:
    """`lanes` requests released together; wall clock spans prefill and decode of the whole batch."""
    results: list = [None] * len(bodies)
    go = threading.Event()

    def worker(index: int) -> None:
        go.wait()
        try:
            results[index] = one_request(base_url, bodies[index], timeout)
        except RequestError as e:
            results[index] = {"error": str(e)}
        except Exception as e:  # noqa: BLE001 - a broken lane must not hang the batch
            results[index] = {"error": f"{type(e).__name__}: {e}"}

    workers = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(bodies))]
    for w in workers:
        w.start()
    started = time.perf_counter()
    go.set()
    for w in workers:
        w.join()
    return results, time.perf_counter() - started


# --- calibration -------------------------------------------------------------------------------


def fill_paths(run_dir: Path, target: int) -> tuple[Path, Path]:
    return run_dir / f"fill-{target}.txt", run_dir / f"fill-{target}.json"


def load_calibration(run_dir: Path, target: int, facts: dict) -> dict | None:
    """A cached calibration that still fills the target, or None to measure one."""
    text_path, meta_path = fill_paths(run_dir, target)
    if not text_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    words = meta.get("words")
    heading = text_path.read_text(encoding="utf-8")
    if not isinstance(words, int) or words < 1 or not heading:
        return None
    if meta.get("sha256") != hashlib.sha256(heading.encode()).hexdigest():
        return None  # the text was edited, truncated or swapped under the same name
    prompt_tokens = meta.get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or prompt_tokens < 1:
        return None
    if abs(prompt_tokens / target - 1) > FILL_TOLERANCE:
        return None
    cached_model = meta.get("model_id")
    if cached_model and facts.get("model_id") and cached_model != facts["model_id"]:
        say(f"# recalibrating fill={target}: cached for model {cached_model!r}, serving {facts['model_id']!r}")
        return None
    return {"target": target, "words": words, "prompt_tokens": prompt_tokens, "text": heading}


def save_calibration(run_dir: Path, calibration: dict, facts: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    text_path, meta_path = fill_paths(run_dir, calibration["target"])
    text_path.write_text(calibration["text"], encoding="utf-8")
    meta = {
        "target": calibration["target"],
        "words": calibration["words"],
        "sha256": hashlib.sha256(calibration["text"].encode()).hexdigest(),
        "prompt_tokens": calibration["prompt_tokens"],
        "error": round(calibration["prompt_tokens"] / calibration["target"] - 1, 5),
        "tolerance": FILL_TOLERANCE,
        "model_id": facts.get("model_id"),
        "ts": now(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def calibrate(args, target: int, run_dir: Path, facts: dict) -> dict:
    """Scale the word count until the server counts `target` prompt tokens, within FILL_TOLERANCE.

    The token count is what matters and only the server knows it, so small probes pay for the
    answer: seconds at 20k of prefill, ~50 s at 100k on this box, and the scale is near linear so
    two probes usually land inside the tolerance. That cost is why the result is cached on disk and
    reused by a resumed run.
    """
    cached = load_calibration(run_dir, target, facts)
    if cached:
        say(
            f"# fill={target}: reusing {cached['words']} words, {cached['prompt_tokens']} prompt tokens "
            f"({cached['prompt_tokens'] / target - 1:+.2%}) from {fill_paths(run_dir, target)[1]}"
        )
        return cached

    words = max(1, int(round(target / WORDS_PER_TOKEN)))
    best: dict | None = None
    for attempt in range(1, CALIBRATION_TRIES + 1):
        text = prompt_for(words)
        say(f"# fill={target}: calibration probe {attempt}/{CALIBRATION_TRIES} with {words} words")
        result = one_request(
            args.base_url,
            chat_body(
                args, text, args.seed, f"matrix-{args.label}-calibrate-{target}", PROBE_MAX_TOKENS
            ),
            args.timeout,
            require_rate=False,
        )
        got = result.get("prompt_tokens")
        if not isinstance(got, int) or got < 1:
            raise RequestError(
                f"calibration probe for fill={target} reported prompt_tokens={got!r}: the server "
                "did not count the prompt, so the fill cannot be calibrated"
            )
        error = got / target - 1
        say(f"# fill={target}: probe {attempt} -> {got} tokens ({error:+.2%}) in {words} words")
        candidate = {"target": target, "words": words, "prompt_tokens": got, "text": text}
        if best is None or abs(error) < abs(best["prompt_tokens"] / target - 1):
            best = candidate
        if abs(error) <= FILL_TOLERANCE:
            break
        words = max(1, int(round(words * target / got)))
    assert best is not None
    achieved = best["prompt_tokens"] / target - 1
    if abs(achieved) > FILL_TOLERANCE:
        say(
            f"# WARNING fill={target}: best calibration is {best['prompt_tokens']} tokens "
            f"({achieved:+.2%}), outside +-{FILL_TOLERANCE:.0%}; rows carry the real count"
        )
    save_calibration(run_dir, best, facts)
    return best


# --- rows --------------------------------------------------------------------------------------


def build_row(args, fill: int, lanes: int, rep: int | None, warm: bool, results, wall, facts) -> dict:
    ok = [r for r in results if "error" not in r]
    failures = [r["error"] for r in results if "error" in r]
    return {
        "label": args.label,
        "path_mode": args.path_mode,
        "lanes": lanes,
        "fill": fill,
        "rep": rep,
        "warm": warm,
        "base_url": args.base_url,
        "model": args.model,
        "model_id": facts.get("model_id"),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "temperature": 0,
        "ttft_s": median([r["ttft_s"] for r in ok]),
        "ttft_max_s": max((r["ttft_s"] for r in ok), default=None),
        "decode_tps": median([r["decode_tps"] for r in ok]),
        "aggregate_tps": (
            sum(r["completion_tokens"] or 0 for r in ok) / wall if ok and wall > 0 else None
        ),
        "prefill_s": median([r.get("prefill_s") for r in ok]),
        "prompt_tokens": median([r.get("prompt_tokens") for r in ok]),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in ok) if ok else None,
        "cached_tokens": sum(r.get("cached_tokens") or 0 for r in ok) if ok else None,
        "token_sha256": ok[0].get("token_sha256") if ok else None,
        "finish_reason": ok[0].get("finish_reason") if ok else None,
        "requests": lanes,
        "requests_ok": len(ok),
        "wall_s": round(wall, 3),
        "streams": [
            {
                "ttft_s": r.get("ttft_s"),
                "decode_tps": r.get("decode_tps"),
                "prefill_s": r.get("prefill_s"),
                "prompt_tokens": r.get("prompt_tokens"),
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


def cell_key(label: str, path_mode: str, lanes: int, fill: int, rep: int) -> tuple:
    return (label, path_mode, int(lanes), int(fill), int(rep))


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
            say(f"# warning: {out}:{number} is not JSON, ignored")
    return rows


def done_keys(rows: list[dict]) -> set[tuple]:
    """Only a row with a number in it counts as done; a failed cell is retried on resume.

    A warm-cache miss still counts: the fix is a serve flag, not another rep, and re-running would
    produce the same cold row.
    """
    keys = set()
    for r in rows:
        if r.get("error") is not None or r.get("rep") is None or r.get("ttft_s") is None:
            continue
        keys.add(cell_key(r["label"], r.get("path_mode", ""), r["lanes"], r["fill"], r["rep"]))
    return keys


# --- cells -------------------------------------------------------------------------------------


def bodies_for(args, prompt: str, fill: int, lanes: int) -> list[dict]:
    return [
        chat_body(args, prompt, args.seed + index, lane_user(args.label, fill, lanes, index))
        for index in range(lanes)
    ]


def measure_cell(
    args, prompt: str, fill: int, lanes: int, missing: list[int], out: Path, reps: int, facts
) -> None:
    """One unmeasured warm pass, then the missing reps. A failed warm pass skips the cell."""
    warm_bodies = bodies_for(args, prompt, fill, lanes)
    say(f"cell lanes={lanes} fill={fill} warm running")
    warm_results, warm_wall = run_batch(args.base_url, warm_bodies, args.timeout)
    if any("error" in r for r in warm_results):
        row = build_row(args, fill, lanes, None, False, warm_results, warm_wall, facts)
        append_row(out, row)
        say(f"cell lanes={lanes} fill={fill} warm FAILED: {row['error']}")
        return
    cached = sum(r.get("cached_tokens") or 0 for r in warm_results)
    say(
        f"cell lanes={lanes} fill={fill} warm ok {warm_wall:.1f}s "
        f"cached={cached} tok ttft={num(median(r['ttft_s'] for r in warm_results), 3, ' s')}"
    )

    for rep in missing:
        say(f"cell lanes={lanes} fill={fill} rep={rep + 1}/{reps} running")
        results, wall = run_batch(args.base_url, bodies_for(args, prompt, fill, lanes), args.timeout)
        row = build_row(args, fill, lanes, rep, True, results, wall, facts)
        append_row(out, row)
        # The rep's own cache count, not the warm pass's: an eviction between reps is visible here.
        cold_now = fill > 0 and not (row.get("cached_tokens") or 0) > 0
        state = "COLD PREFIX" if cold_now else "warm"
        say(
            f"cell lanes={lanes} fill={fill} rep={rep + 1}/{reps} {state} "
            f"decode={num(row['decode_tps'], suffix=' tok/s')} "
            f"aggregate={num(row['aggregate_tps'], suffix=' tok/s')} "
            f"ttft={num(row['ttft_s'], 3, ' s')} max={num(row['ttft_max_s'], 3, ' s')} "
            f"prompt={num(row['prompt_tokens'], 0, ' tok')} "
            f"cached={num(row['cached_tokens'], 0, ' tok')} {wall:.1f}s"
        )


# --- report ------------------------------------------------------------------------------------


def measured(rows: list[dict]) -> list[dict]:
    return [
        r
        for r in rows
        if r.get("error") is None and r.get("rep") is not None and r.get("ttft_s") is not None
    ]


def cold_prefix_rows(rows: list[dict]) -> list[dict]:
    """Measured fills that reported no cached prompt tokens: a cold stream, not a warm one."""
    return [r for r in measured(rows) if (r.get("fill") or 0) > 0 and not (r.get("cached_tokens") or 0) > 0]


def short_rows(rows: list[dict]) -> list[dict]:
    short = []
    for r in measured(rows):
        tokens = r.get("completion_tokens")
        if isinstance(tokens, (int, float)) and tokens < SHORT_COMPLETION:
            short.append(r)
    return short


def cell_lines(rows: list[dict], fill: int) -> list[str]:
    """One bullet per lane count at this fill: medians over the cell's clean reps only."""
    cold = [r for r in cold_prefix_rows(rows) if r["fill"] == fill]
    dropped = {(r["lanes"], r["rep"]) for r in cold}
    cells: dict[int, list[dict]] = {}
    for r in measured(rows):
        if r["fill"] != fill or (r["lanes"], r["rep"]) in dropped:
            continue
        cells.setdefault(r["lanes"], []).append(r)
    lines = []
    for lanes in LANES:
        group = cells.get(lanes, [])
        excluded = sum(1 for r in cold if r["lanes"] == lanes)
        if not group:
            if excluded:
                lines.append(f"- lanes={lanes} | not measured ({excluded} reps excluded: cold prefix)")
            else:
                lines.append(f"- lanes={lanes} | not measured")
            continue
        n = len(group)
        floor = "" if n >= REPS_FLOOR else " (below floor)"
        parts = [f"lanes={lanes}"]
        parts.append(f"decode {num(median(r['decode_tps'] for r in group), suffix=' tok/s')}")
        parts.append(f"aggregate {num(median(r['aggregate_tps'] for r in group), suffix=' tok/s')}")
        parts.append(f"ttft {num(median(r['ttft_s'] for r in group), 3, ' s')}")
        parts.append(f"ttft max {num(median(r.get('ttft_max_s') for r in group), 3, ' s')}")
        if median(r.get("prefill_s") for r in group) is not None:
            parts.append(f"prefill {num(median(r['prefill_s'] for r in group), 3, ' s')}")
        prompt_tokens = median(r.get("prompt_tokens") for r in group)
        parts.append(f"prompt {num(prompt_tokens, 0, ' tok')}")
        cached = median(r.get("cached_tokens") for r in group)
        # cached_tokens is the batch total, prompt_tokens the per-request median: compare the batch
        # against every lane's prompt, or lanes=4 would print 400% cached.
        batch_prompt = prompt_tokens * lanes if isinstance(prompt_tokens, (int, float)) else None
        if isinstance(cached, (int, float)) and batch_prompt:
            parts.append(f"cached {num(cached, 0, ' tok')} ({cached / batch_prompt:.0%})")
        else:
            parts.append(f"cached {num(cached, 0, ' tok')}")
        count = f"n={n} reps{floor}"
        if excluded:
            count += f", {excluded} of {n + excluded} reps excluded (cold prefix)"
        parts.append(count)
        lines.append("- " + " | ".join(parts))
    return lines


def failure_lines(rows: list[dict], limit: int = 10) -> list[str]:
    """Rows that failed, with the cell they were meant for. A failed warm pass has rep null."""
    failed = [r for r in rows if r.get("error") is not None]
    lines = []
    for r in failed[:limit]:
        rep = r.get("rep")
        where = f"lanes={r.get('lanes')} fill={r.get('fill')} rep={rep if rep is not None else 'warm'}"
        lines.append(f"- ERROR: {where}: {r['error']}")
    if len(failed) > limit:
        lines.append(f"- ERROR: and {len(failed) - limit} more")
    return lines


def calibration_note(rows: list[dict]) -> str:
    """What the server actually counted for this fill, and whether the file is one fill at all."""
    counts = [r["prompt_tokens"] for r in measured(rows) if isinstance(r.get("prompt_tokens"), (int, float))]
    if not counts:
        return "prompt tokens: not measured"
    lo, hi = min(counts), max(counts)
    note = f"server counted {num(median(counts), 0)} prompt tokens"
    spread = (hi - lo) / hi if hi else 0
    if spread > 0.02:
        note += f"; WARNING rows disagree ({num(lo, 0)} to {num(hi, 0)}): recalibrated mid-file"
    return note


def coverage_bits(rows: list[dict], fill: int) -> tuple[str, int]:
    """What this fill actually holds, in words: clean lanes, missing lanes, excluded lanes.

    Returns the phrase and how many clean cells it counted, so the caller can total them.
    """
    block = measured([r for r in rows if r.get("fill") == fill])
    if not block:
        return f"fill {fill}: not measured", 0
    touched = {r["lanes"] for r in block}
    excluded = sorted(r["lanes"] for r in cold_prefix_rows(block))
    clean = sorted(touched - set(excluded))
    never = [n for n in LANES if n not in touched]
    bits = []
    if clean:
        bits.append(f"{len(clean)}/{len(LANES)} lanes")
    if never:
        bits.append(f"lanes={','.join(str(n) for n in never)} not measured")
    if excluded:
        bits.append(f"lanes={','.join(str(n) for n in sorted(set(excluded)))} cold-prefix excluded")
    return f"fill {fill}: " + (", ".join(bits) or "not measured"), len(clean)


def report(rows: list[dict], requested_fills: list[int] | None, args, out: Path) -> int:
    good = measured(rows)
    failed = [r for r in rows if r.get("error") is not None]
    cold = cold_prefix_rows(rows)
    print(f"## matrix_bench: {out}")
    print(
        f"- measured rows: {len(good)} (all warm: one unmeasured pass of the same prompts before each "
        f"cell); error rows: {len(failed)}; cold-prefix rows: {len(cold)}; "
        f"floor: {REPS_FLOOR} reps per cell"
    )
    if not good:
        print("- not measured: the file holds no measurement yet")

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r.get("label", ""), r.get("path_mode", "")), []).append(r)

    for (label, path_mode), own in sorted(groups.items()):
        base = next((r["base_url"] for r in own if r.get("base_url")), args.base_url)
        model = next((r["model"] for r in own if r.get("model")), args.model)
        model_id = next((r["model_id"] for r in own if r.get("model_id")), None)
        own_cold = cold_prefix_rows(own)
        own_short = short_rows(own)
        print()
        print(f"### {label} | path `{path_mode}` | {base}")
        print(
            f"- server: /v1/models id `{model_id if model_id else 'unknown'}` at {base}; "
            f"request model `{model}`; one path per run, so every row below is the server as started"
        )
        mine = measured(own)
        if not mine:
            print("- not measured: no cell for this label and path")
            for line in failure_lines(own):
                print(line)
            continue

        fills = sorted({r["fill"] for r in mine} | set(requested_fills or []))
        for fill in fills:
            block = [r for r in own if r.get("fill") == fill]
            print()
            print(f"#### fill {fill} tokens")
            print(f"- {calibration_note(block)}")
            for line in cell_lines(block, fill):
                print(line)

        phrases = [coverage_bits(own, fill) for fill in fills]
        cells_done = sum(count for _, count in phrases)
        print()
        print(
            f"- coverage: {'; '.join(phrase for phrase, _ in phrases)}; "
            f"{cells_done} of {len(LANES) * len(fills)} cells measured"
        )

        if own_cold:
            print()
            print("protocol failures: measured rows whose warm prefix was not in the cache")
            for r in own_cold[:10]:
                print(
                    f"- lanes={r['lanes']} fill={r['fill']} rep={r['rep']}: cached_tokens 0 of "
                    f"{num(r.get('prompt_tokens'), 0)} prompt tokens; the warm pass did not fit the "
                    "prompt cache (raise --prompt-cache-gib / --context, or lower the fill)"
                )
            if len(own_cold) > 10:
                print(f"- and {len(own_cold) - 10} more cold-prefix rows; all excluded from the medians")
            print("- these rows stay in the JSONL as evidence and are not averaged into their cells")

        if own_short:
            print()
            print("short replies (decode rate is not meaningful, kept in the JSONL):")
            for r in own_short[:6]:
                print(
                    f"- lanes={r['lanes']} fill={r['fill']} rep={r['rep']}: "
                    f"{int(r['completion_tokens'])} completion tokens"
                )

        if any(r.get("error") is not None for r in own):
            print()
            print("failures:")
            for line in failure_lines(own):
                print(line)

    return 1 if failed or cold else 0


# --- driver ------------------------------------------------------------------------------------


def parse_int_list(text: str, name: str, parser) -> list[int]:
    values: list[int] = []
    try:
        values = sorted({int(x) for x in text.split(",") if x.strip()})
    except ValueError:
        parser.error(f"{name} '{text}' is not a comma separated list of integers")
    if not values or any(v < 0 for v in values):
        parser.error(f"{name} needs at least one non-negative integer")
    return values


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Context-fill matrix (lanes 1/2/4 x fills 0/20k/100k) on one server path; "
        "one flushed JSONL line per measurement, resumed by key, reported per fill."
    )
    p.add_argument("--label", default=LABEL, help="names the build and its serve flags; share it across paths")
    p.add_argument("--base-url", default=BASE_URL, help="e.g. http://127.0.0.1:8099/v1")
    p.add_argument("--model", default=MODEL, help="the request's model field")
    p.add_argument("--out", default=OUT, help="JSONL receipt; appended to and resumed from")
    p.add_argument(
        "--path-mode",
        choices=PATH_MODES,
        default="mtp",
        help="the path this server serves: mtp (no --drafter) or dflash2 (--drafter). One per run.",
    )
    p.add_argument("--lanes", default=",".join(str(n) for n in LANES), help="comma separated concurrency levels")
    p.add_argument(
        "--fills",
        default=",".join(str(f) for f in FILLS),
        help="comma separated prompt-token fills; 0 is the instruction alone",
    )
    p.add_argument("--only-lanes", default=None, help="run just these lane counts, e.g. 1,4")
    p.add_argument("--only-fills", default=None, help="run just these fills, e.g. 20000")
    p.add_argument("--reps", type=int, default=REPS_DEFAULT, help=f"measured batches per cell, at least {REPS_FLOOR}")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--seed", type=int, default=1234, help="lane i sends seed + i; fixed across reps")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--report-only", action="store_true", help="read --out and print the report; no requests")
    a = p.parse_args(argv)

    if a.reps < REPS_FLOOR:
        p.error(f"--reps {a.reps} is below the floor of {REPS_FLOOR}: a cell needs at least that many")
    if a.max_tokens < 1:
        p.error("--max-tokens must be at least 1")
    a.lane_counts = parse_int_list(a.lanes, "--lanes", p)
    if any(n < 1 for n in a.lane_counts):
        p.error("--lanes needs at least one positive lane count")
    a.fill_targets = parse_int_list(a.fills, "--fills", p)
    if a.only_lanes:
        keep = set(parse_int_list(a.only_lanes, "--only-lanes", p))
        a.lane_counts = [n for n in a.lane_counts if n in keep]
        if not a.lane_counts:
            p.error(f"--only-lanes {a.only_lanes} names no lane count from --lanes {a.lanes}")
    if a.only_fills:
        keep = set(parse_int_list(a.only_fills, "--only-fills", p))
        a.fill_targets = [f for f in a.fill_targets if f in keep]
        if not a.fill_targets:
            p.error(f"--only-fills {a.only_fills} names no fill from --fills {a.fills}")
    return a


def refuse_production(base_url: str) -> None:
    port = None
    try:
        port = urllib.parse.urlparse(base_url).port
    except Exception:  # noqa: BLE001 - a malformed URL fails later with a clearer message
        port = None
    if port in REFUSED_PORTS or any(f":{p}" in base_url for p in REFUSED_PORTS):
        raise SystemExit(
            f"refusing {base_url}: port {sorted(REFUSED_PORTS)} serves production from this worktree"
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out)

    # Reading a receipt sends nothing, so the production-port guard is for the measuring path only.
    if args.report_only:
        return report(load_rows(out), None, args, out)
    refuse_production(args.base_url)

    facts = server_facts(args.base_url)
    if facts["prompt_cache"] is False:
        say(
            "# WARNING /v1/models reports capabilities.prompt_cache=false: every fill>0 row will "
            "report cached_tokens 0 and be flagged as a protocol failure. Serve with a prompt cache."
        )
    over = [n for n in args.lane_counts if isinstance(facts.get("max_batch"), int) and n > facts["max_batch"]]
    if over:
        say(
            f"# WARNING lanes={over} exceed the server's max_batch_size={facts['max_batch']}: those "
            "requests queue instead of batching, so aggregate_tps is a queue rate"
        )
    say(
        f"# {args.label} | {args.base_url} | id={facts.get('model_id')} | path={args.path_mode} "
        f"lanes={args.lane_counts} fills={args.fill_targets} reps={args.reps}"
    )

    run_dir = out.parent
    rows = load_rows(out)
    done = done_keys(rows)
    prompts: dict[int, str] = {}
    for fill in args.fill_targets:
        if fill > 0:
            try:
                prompts[fill] = calibrate(args, fill, run_dir, facts)["text"]
            except RequestError as e:
                # Nothing to measure without a calibrated fill, and a traceback would hide why.
                say(f"# calibration failed for fill={fill}: {e}")
                say(f"# {out} was not written; is the server up at {args.base_url}?")
                return 1
        else:
            prompts[fill] = prompt_for(0)

    for fill in args.fill_targets:
        for lanes in args.lane_counts:
            missing = [
                rep
                for rep in range(args.reps)
                if cell_key(args.label, args.path_mode, lanes, fill, rep) not in done
            ]
            if not missing:
                say(f"# skip lanes={lanes} fill={fill}: all reps already in {out}")
                continue
            measure_cell(args, prompts[fill], fill, lanes, missing, out, args.reps, facts)

    rows = load_rows(out)
    return report(rows, args.fill_targets, args, out)


if __name__ == "__main__":
    raise SystemExit(main())
