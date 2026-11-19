# Flash Next: per-layer time for the target pass (`TF_FLASHNEXT_LAYER_TIME`)

The served decode round's dominant term is the 48-layer target verify pass (one `Pass.chunk`), and it
has had no per-layer visibility: `TF_FLASHNEXT_PROFILE` prints one wall clock for the whole `chunk` and
synchronises twice doing it. This adds an event-based per-stage clock inside that one loop, gated by an
environment variable, with the drafter's own stage names and `stages.json` schema so
`tools/dflash2/harness/plot_stages.py` reads it unchanged.

## Flags

| Flag | Meaning |
| --- | --- |
| `TF_FLASHNEXT_LAYER_TIME=1` | Print the per-layer report to stderr at the end of every profiled pass. Registered in the strict table (`platform/environment.zig`): `0` or `1`, default `0`, anything else refused at boot. |
| `TF_FLASHNEXT_LAYER_DUMP=<dir>` | Also write `<dir>/rep<N>/stages.json` for each profiled pass (`N` counts from 0 within the process). Read as a path, like `TF_FLASHNEXT_TAP_DUMP`, and unset or blank is off. Either flag alone turns the profiler on; the console report is printed only when `TF_FLASHNEXT_LAYER_TIME=1`. |

Neither flag changes a kernel, a launch or a buffer: the same bits come out with the profiler on, and
unsetting both restores the previous behaviour exactly.

## What is measured

One CUDA timing event (`cuEventRecord`, no synchronise) at each stage boundary, and **one**
`cuStreamSynchronize` after the pass's last launch. Boundaries are contiguous, so the stages of a pass
sum to its GPU span from the embedding's first launch to the head projection's end.

| Stage (JSON `stage`) | Boundary site |
| --- | --- |
| `embed` | `cuda_batch.zig` `chunk`, immediately before the embedding kernel |
| `conv_attn_in_{d}` | the loop's `layer(d)` call: the n-gram block, the input mix and read-out, the tap ring's copy |
| `qkv_{d}` | top of `attnBlockKeep` / `gdnBlock`, before the projection lane |
| `attn_{d}` | before `attnRows` (attention) / before the fused DeltaNet chain |
| `attn_proj_{d}` | before `attnGate` (attention) / before the gated norm and out projection (DeltaNet) |
| `conv_mlp_in_{d}` | `chunk`, before the MoE input mix |
| `mlp_{d}` | top of `moeBlockKeep`: the router and the experts |
| `moe_out_{d}` | `chunk`, before `moeAdds`: the slots summed and written back |
| `block_hidden_{d}` | `chunk`, before `tapDump`: the tapped row's copy out and the handover |
| `final_norm_block` | `chunk`, before `Pass.head` |
| `head` | inside `Pass.head`, after the read-out and before `projectHead` |

48 layers, so a pass records `8*48 + 4 = 388` events and 387 stage entries. `block_hidden_{d}`'s end is
`conv_attn_in_{d+1}`'s start: one event serves both, which is why a layer costs 8 boundaries and not 9.

## Artifact

`<dir>/rep<N>/stages.json`, the drafter's schema (`cuda_draft2_check.zig`'s `writeStages`):

```json
{"impl":"tensorfold-cuda","device":"cuda","dtype":"bf16","n_stages":387,
 "round":{"mode":"decode","rows":16,"layers":48,"chunk":2048,"position":4096,"round_ms":53.214},
 "stages":[{"stage":"conv_attn_in_0","shape":[16],"dtype":"bf16","src_dtype":"bf16","ms":0.4123,"measured":"gpu","stored":"mark"}, ...],
 "sum_stage_ms":53.214}
```

`mode` is `decode` when the pass's rows are at or below `mtp.max_rows` (32), else `prompt`. `round_ms`
is the sum of the stages (the GPU span described above), not a wall clock of the whole `chunk`: the
id/position uploads, the host-side n-gram gather and the head's host download happen outside the first
and last events. `stored: "mark"` says no tensor was stored, so `shape` is just the pass's rows.

```sh
TF_FLASHNEXT_LAYER_TIME=1 TF_FLASHNEXT_LAYER_DUMP=/tmp/lt TF_FLASHNEXT_WINDOW_GRAPH=0 <server>
python3 tools/dflash2/harness/plot_stages.py /tmp/lt --out /tmp/lt.png
```

## Refusals (no silent graph disabling)

A capture cannot hold this: an event record inside a capture becomes a node that replays a stale
timestamp, and `cuStreamSynchronize` inside one is illegal.

* `Clock.begin` refuses (`error.LayerTimeInCapture`) if the pass's stream is already capturing.
* `Engine.forward` refuses **before** a window capture begins (`error.LayerTimeInGraph`) — but only when
  that pass is one `cuda_prompt_windows.captures` says would be recorded, the same predicate
  `forwardGraph` uses, so a tapped decode verify (which never captures) is not refused. A refusal
  therefore never leaves the graph caches switched off behind it. Set `TF_FLASHNEXT_WINDOW_GRAPH=0` to
  profile prompt passes.
* `Engine.round` refuses a ranked (TP) round (`error.LayerTimeRanked`) and `chunk` refuses a ranked
  pass: TP routes the verify through `cuda_mtp_ops.verify`'s `main.verify`, not this loop, so the
  profiler is single-rank by construction and says so instead of timing nothing.
* One pool, one pass: a second pass in flight on the engine's pool is refused
  (`error.LayerTimeBusy`). A caller that needs two passes in flight gives one of them its own
  `Pass.clock`; the engine's pool is only installed on the pass that holds it. `chunk` releases the
  claim it took on any error path (`Clock.abandon`), so a failed pass never wedges the pool.
* A pass of another layer count than the pool was built for (the draft head's one layer) is not
  profiled, and says so once.

Not profiled: the multi-lane shared verify (`cuda_prompt_windows.windows`), the draft head's pieces and
the rollback replays. They take no `Pass.clock`, so their instrumentation is a null check.

## Cost

Unset: `Engine.layer_clock` is null, `Pass.clock` is null, and every boundary site is
`if (self.clock) |c| ...` — one branch on a pointer per boundary, no allocation, no event, no
synchronise, no extra launch. Nothing on the unset path touches the driver.

Set: 388 timing events made once per engine, 388 records per pass (~0.3-1 us each on the queue), one
extra synchronise at the pass's end, and the report's host time. `block_hidden`, `moe_out` and `qkv`-sided
stages of a few microseconds are at the 0.5 us event resolution and each stage's number carries that
plus the record's own overhead; treat differences under ~2 us as noise.

## Assumptions not verified without a build or a run

1. **Compilation.** No `zig build` was run (the machine cannot afford a model load). The files pass
   `zig fmt --check` and `zig ast-check` (0.17.0, the tree's formatter/analyser), which is syntax and
   unused-binding level only: no type check, no codegen.
2. `cuda.graph.captureStatus` is called at `Clock.begin` with the pass's stream; on a driver error it is
   treated as capturing and the profiler refuses (fail closed).
3. `cuEventElapsedTime` between two events recorded on the same stream, both completed by the single
   `synchronize`, is the elapsed GPU time between them — the same contract `cuda_segments.zig` relies on.
4. Boundary placement is by reading the code, not by running it. `attn_proj_{d}` opens at `attnGate`
   (the gated norm for DeltaNet) and therefore contains the out projection's `sumSlices` that `chunk`
   runs after `attnBlockKeep` returns, plus `zs.slices`; `conv_mlp_in_{d}` opens immediately after those.
   If a reader wants the mix-branch sum counted with the MoE input instead, the `.conv_mlp` record moves
   one statement earlier (before `if (keep.mix_branch != 0)`), and nothing else changes.
5. `decode_rows = 32` mirrors `mtp.max_rows` (`cuda_mtp.zig`) and is not imported, because `cuda_mtp.zig`
   imports `cuda_batch.zig`, the file the clock rides in — a real import cycle. If `mtp.max_rows` changes,
   this constant must follow; `cuda_mtp.zig`'s own test pins it at 32 today.
6. `Path`/dump: `std.Io.Dir.cwd().createDirPath` + `writeFile` as `cuda_draft2_check.zig` uses them; a
   dump failure is printed and otherwise ignored, never returned to the caller of `chunk`.
7. A `cuEventRecord` that fails (a driver-level fault) propagates out of `chunk` like any launch error.

## Files touched (final line numbers)

| File | Lines | What |
| --- | --- | --- |
| `zig/src/families/flashnext/cuda_layer_time.zig` | 1-284 (new) | `Clock`: the pool, the boundary records, one synchronise, the console report, the `stages.json` dump, the refusals, `stageName`, and the arithmetic test at 271 |
| `zig/src/families/flashnext/cuda_batch.zig` | 24 | import |
| | 116 | `Pass.clock: ?*layer_time.Clock = null` |
| | 341, 352, 388 | `gdnBlock`: `.qkv`, `.attn`, `.attn_proj` |
| | 441, 448, 467 | `attnBlockKeep`: `.qkv`, `.attn`, `.attn_proj` |
| | 537 | `moeBlockKeep`: `.mlp` |
| | 600-614 | `chunk`: the TP refusal, `Clock.begin`, the pass's own claim on the pool and its release |
| | 623 | `chunk`: `.embed`, before the embedding kernel |
| | 632 | `chunk`: `Clock.layer`, the loop's `conv_attn_in_{d}` |
| | 665, 674, 678 | `chunk`: `.conv_mlp`, `.moe_out`, `.hidden` |
| | 682, 685 | `chunk`: `finalBlock`, then `finish` after `head` |
| `zig/src/families/flashnext/cuda_batch_head.zig` | 23, 35, 70 | `head`: `headStart` in the three branches (`logits == .none`, `.all`, and the single-row path) |
| `zig/src/families/flashnext/cuda_prompt.zig` | 14 | import |
| | 137-140 | `Engine.layer_clock` |
| | 165, 233, 238 | null at init, created when a flag asks, freed at deinit |
| | 427, 429-436 | `forward`: install the clock, refuse a would-be capture before it begins |
| | 525, 529-534 | `round`: install the clock, refuse a ranked round |
| `zig/src/families/flashnext/cuda_prompt_windows.zig` | 11-22 | `captures`, `forwardGraph`'s own predicate, now shared |
| | 26-27 | `forwardGraph` starts from it |
| `zig/src/platform/environment.zig` | 190 | the `TF_FLASHNEXT_LAYER_TIME` row |
| `zig/src/platform/environment_flashnext_test.zig` | 38, 46 | 29 -> 30 settings, 20 -> 21 switches |
| `docs/environment.md` | 91, 115, 128-130 | the count, the row, and the note that the dump path is not in the table |
| `tools/dflash2/harness/plot_stages.py` | 58-61 | three `CATEGORY` rows (`moe_out_`, `embed`, `head`) |
| | 81 | `short()`: `moe_out_` -> `mo`, `embed` -> `emb` |

Nothing else changed: no kernel, no launch, no buffer, and `TF_FLASHNEXT_PROFILE`'s path is untouched.

## Checked without a build

`zig fmt --check` and `zig ast-check` (`/home/relja/tools/zig-0.17.0/zig`) pass on all seven Zig files.
That is parse and binding level only. No `zig build`, no run, no model load: every signature this module
calls was read in the tree (`cuda.Event.init/record/elapsedMs`, `Stream.synchronize`,
`cuda.graph.captureStatus`, `sys.environment.read`, `sys.getenv`, `std.Io.Writer.Allocating`,
`std.Io.Dir.cwd().createDirPath/writeFile`), but nothing has been type-checked or executed.
