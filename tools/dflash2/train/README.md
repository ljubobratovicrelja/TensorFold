# tools/dflash2/train — torch side of the drafter

`forward.py` is the DFlash2 drafter forward written in torch against the CUDA round, and
`parity.py` replays the same rounds through that forward and through the engine binary's
`draft-check`, then diffs the two stage by stage.

## Environment

CUDA 13.2, sm_121 (GB10 / DGX Spark), Linux arm64, Python 3.12. `torch==2.15.0` is the
PyPI `manylinux_2_28_aarch64` wheel, which bundles CUDA 13.2 — no cu126/cpu wheel, no system
CUDA toolkit needed at runtime. `numpy` and `safetensors` are the only other dependencies.
Nothing is installed as a package; `uv` makes the `.venv/` and `uv.lock` pins it.

```sh
# one-off: create .venv from uv.lock
uv sync --directory tools/dflash2/train
```

## Run

```sh
# ... from the repo root

# forward.py alone: no checkpoint reachable, so only the elementary ops run
uv run --directory tools/dflash2/train python forward.py --selftest
uv run --directory tools/dflash2/train python forward.py --help

# parity against the engine: needs zig-out/native/bin/tensorfold-native, the two HF snapshots,
# and harvested fixtures under /tmp/harv/out
uv run --directory tools/dflash2/train python parity.py --rounds 16 --fixtures chunk-0000 chunk-0002
uv run --directory tools/dflash2/train python parity.py --help
```

`parity.py` shells out to `<repo>/zig-out/native/bin/tensorfold-native draft-check` and reads
`~/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2` plus the FP4 target's
`embed_tokens`/`lm_head`; `--drafter`, `--target`, `--binary`, `--harvest` override each. It writes
its scratch fixtures and dumps to `--work` (default `/tmp/dp2/parity`).

## Re-sync after a dependency change

Edit `dependencies` in `pyproject.toml`, then:

```sh
uv lock --directory tools/dflash2/train    # re-resolve; commit the new uv.lock
uv sync --directory tools/dflash2/train    # update .venv to the new lock
```

`uv run` after a `uv.lock` edit also re-syncs on its own; `uv lock` is what pins the change.
