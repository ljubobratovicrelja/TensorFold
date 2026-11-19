# Activation dumps: what the pass routed, and where its hidden rows go

`zig/src/families/flashnext/cuda_probe.zig`, two flags, both off unless they name a directory:

| flag | what it captures | cost per pass |
| --- | --- | --- |
| `TF_FLASHNEXT_EXPERT_DUMP=<dir>` | every layer's top-k expert ids and gate weights (`Scratch.picks` / `.wts`) | a 44 B×rows copy a layer, one read-back |
| `TF_FLASHNEXT_HIDDEN_DUMP=<dir>` | every layer's post-norm MoE input row (`Scratch.mixed`, 2560 bf16) | one 5 KB×rows copy a layer, on one pass |
| `TF_FLASHNEXT_ROUTER_LOGITS=<rows>` | with the expert dump: the full 512+1 router logits of the widest pass, first `<rows>` rows | 12.6 MB of device buffer, one copy a layer |
| `TF_FLASHNEXT_HIDDEN_ROWS=<n>` | hidden rows a layer keeps (default 32) | — |

## Where it hooks

`Pass.chunk`'s layer loop, immediately after `moeBlockKeep` returns: `s.picks`, `s.wts` and
`s.router_l` hold this layer's routing and `s.mixed` its MoE input, and the next layer's
`readoutKeep` overwrites all of them. The end of `chunk`, after `head`, does the single
`cuStreamSynchronize` and the downloads; a decode round has already synchronised for its logits at
that point, so the extra wait is a few microseconds.

`Scratch.picks` is `[rows, 11]`: ten routed experts and the always-on shared expert (id 512, the
router's last logit row, gate weight ~0.47 at layer 0 and ~0.1 in the middle of the stack). The
dump keeps all eleven; a consumer that wants load balance over the routed set drops id 512.

## Safety rules, the same as `flashnext-layer-time.md`

* Unset is free: no allocation, no launch, no synchronise; every site is a null check on
  `Pass.probe`, which the engine leaves null unless a flag names a directory.
* A capture cannot hold the read-back, so `chunk` refuses (`error.ProbeInCapture`) a pass whose
  stream is already capturing. Serve with `TF_FLASHNEXT_WINDOW_GRAPH=0` when dumping.
* One capture object per Engine, made at `init` (so a bad directory fails boot, not a reply), and
  a failed write or read-back prints once and switches the probes off rather than failing the pass.

## Files

```
p0000.bin              u32 magic 'TFEX', u32 rows, u32 pos0, u32 layers, u32 slots,
                       ids [layers][rows][slots] u32 LE, wts [layers][rows][slots] f32 LE
router_logits.bin      u32 magic 'TFRT', u32 layers, u32 rows, u32 width, [layers][rows][width] f32 LE
hidden.bin             u32 magic 'TFHD', u32 layers, u32 keep, u32 dims, u32 rows_of_pass, u32 k, u32 pos0,
                       [layers][keep][dims] bf16 LE  (rows [0, k) of each layer's block are real)
manifest.json          geometry, both layouts, passes written
hidden_manifest.json   hidden geometry and the pass it came from
```

One file a pass, so a killed server loses at most the pass in flight and never leaves a truncated
one. A server measures its window costs on passes of 1..32 rows before it serves anything; those are
recorded too, and both the hidden and the logits captures keep the *widest* pass instead of the
first, so the file holds the prompt fill rather than a warm-up.

## Reading them

`tools/` has no consumer yet; the reader that produced the plots lives beside the dumps, in
`docs/design/activation-probing/` — `plot_activations.py` (numpy + matplotlib, `--data/--out/--summary`),
with `collect_task.py` and `run_category.sh` regenerating the dumps (`data/` and `logs/` are gitignored).
