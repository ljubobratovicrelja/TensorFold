# DFlash2 drafter harness (research, not shipped)

`reference.py` is a torch-only port of the DFlash2 drafter, used to answer one question before any
engine work: which flashnext row is the tap that `incoai/Qwen3.8-Flash-Next-DFlash2` expects.

Run it with a CUDA torch (the box has one at `~/.venvs/marker/bin/python`):

    ~/.venvs/marker/bin/python tools/dflash2/reference.py --check

It loads the real 81-tensor checkpoint and runs its forward and candidate selector end to end on
synthetic taps, which is enough to prove the port's shapes and ordering. The acceptance-length
experiment needs the target's forward too and is step 0 in
`docs/design/flashnext-dflash2-drafter.md`.

Not part of the engine build; nothing imports it.
