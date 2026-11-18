# DFlash2 drafter harness (research, not shipped)

## Layout

| dir | what lives there |
| --- | --- |
| `harness/` | things that drive the engine: serving benchmarks and their plots, the server helper, and the corpus harvester. `accept_ab.py`, `matrix_bench.py`, `three_paths_bench.py`, `two_paths.py`, `round_cost.py`, `plot_arms.py`, `plot_stages.py`, `harvest.py`, `tfsrv.sh`. |
| `analysis/` | offline comparison against a reference: `reference.py` (the torch port), `tap_accept.py`, `tap_dump_run.py`, `draft_compare.py`, `head_fidelity.py`, `regress.py`, and `align_bf16/` (the two-sided stage diff with its fixtures). |
| `quantize/` | the block-FP8 drafter quantizer and its verifier. |
| `train/` | the training-side forward and parity check. |
| `run/`, `runs/` | data: benchmark output and captured stage runs. |

Nothing at this level is code: the directory holds this README and the five subdirectories.

## Reference port

`analysis/reference.py` is a torch-only port of the DFlash2 drafter, used to answer one question
before any engine work: which flashnext row is the tap that
`incoai/Qwen3.8-Flash-Next-DFlash2` expects.

Run it with a CUDA torch (the box has one at `~/.venvs/marker/bin/python`):

    ~/.venvs/marker/bin/python tools/dflash2/analysis/reference.py --check

It loads the real 81-tensor checkpoint and runs its forward and candidate selector end to end on
synthetic taps, which is enough to prove the port's shapes and ordering. The acceptance-length
experiment needs the target's forward too and is step 0 in
`docs/design/flashnext-dflash2-drafter.md`.

Not part of the engine build; nothing imports it.
