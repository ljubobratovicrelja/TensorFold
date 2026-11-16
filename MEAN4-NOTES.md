# mean4: the tap basis the DFlash2 drafter was trained on

Uncommitted working-tree diff in `/home/relja/Projects/tf-wt/mean4` (branch `mean4`, off `dflash2` @
`4228bd6a2`). Nothing was built, run, served, profiled or tested — see **What I could not verify** at
the end for the exact commands that remain, and the honest reason they were not run.

## What changed, file by file

| file | +/− | what |
| --- | --- | --- |
| `zig/kernels/cuda/qwen35_draft.cu` | +42/−0 | the collapse kernel `tf_fxd2_mean4`, appended after `tf_fxd2_silu_mul` |
| `zig/src/families/flashnext/cuda_tap_ring.zig` | +186/−16 | `TapSource.default = .mean4`, the docs that argued for `readout` rewritten, `sourceFromName`, `slotOffset`, the host reference `Mean4`, four new tests |
| `zig/src/families/flashnext/cuda_batch.zig` | +27/−6 | `Pass.tapRingWrite`'s `.mean4` arm (collapse then the same strided ring copy), `tapring` import |
| `zig/src/families/flashnext/cuda_batch_kernels.zig` | +10/−0 | the `qwen35_draft` module and the `mean4` entry in the pass's `Kernels` table, plus its unload |
| `zig/src/families/flashnext/cuda_batch_state.zig` | +8/−1 | `Scratch.tap4`, its `bytes()` term, its `alloc`, the `Hooks.tap_source` doc |
| `zig/src/families/flashnext/cuda_draft2_fwd.zig` | +5/−1 | the test that pinned the default to the read-out pointer |
| `zig/dense_serving_test.zig` | +5/−1 | the reviewed-bytes sha256 of `qwen35_draft.cu`, and the comment recording the new entry |
| `docs/design/flashnext-dflash2-drafter.md` | +25/−2 | §4.1b (the settled basis and its evidence), §0.6 and the step table's `mean4` state |

`git diff --numstat`: 8 tracked files, 308 insertions, 27 deletions (plus the new untracked
`MEAN4-NOTES.md`).

## The formula and the addressing

```
out[r, d] = bf16( (1 / SS) * sum_{s<SS} src[(r * SS + s) * D + d] )
```

`SS = streams = 4` (`cuda_prompt_wt.zig`'s `streams`), `D = dims = 2560`, `r < n` the pass's rows.
That is the same `(r * SS + stream) * D + d` addressing `fn_hc_dense_mix` uses
(`flashnext_hc_dense.cuh` line 44), so the raw rows are read exactly as the target laid them down —
stream-major `[s0 | s1 | s2 | s3]`, `wide = 10240` bf16 a row.

The input is `s.h` **after** the pass's `writeback(0, n)`, which is the completed output of the tapped
layer (`writeback` writes `h` in place: `hcWriteback(h, h, ...)`), flushed early by `tapDump`. That is
the tensor the vendor's capture calls `hidden_states`:

```python
hidden_states.unflatten(-1, (hc_count, hidden_size)).mean(dim=-2)   # incoai's SGLang capture patch
```

No gate, no `hc_norm`. `readout` (our old default) is `fn_hc_dense_mix`: a sigmoid-gated mean over
hc-**normalized** streams — a different tensor, which is why acceptance sat at ~1.5 a round against
the 2.4-3.3 band.

### Kernel signature and launch

```cuda
extern "C" __global__ void tf_fxd2_mean4(const __nv_bfloat16* __restrict__ SRC,
                                         __nv_bfloat16* __restrict__ OUT, int D, int SS);
// grid(n, 1, 1) — one CTA a row; block(256, 1, 1); no dynamic shared memory.
```

Per thread it walks the row in 8-bf16 (16-byte `uint4`) units, loading each stream's vector and
accumulating into `float acc[8]`; 2560 channels = 320 vectors over 256 threads. A tail loop covers a
`D` not divisible by 8 (and switches the vector path off entirely in that case, since only `D % 8 == 0`
keeps each stream's start 16-byte aligned).

### The bf16 rounding choice, and its basis

**fp32 accumulation, `__fadd_rn` per term, one `__fdiv_rn` by the count, one bf16 rounding at the
store via `__float2bfloat16_rn` / `__floats2bfloat162_rn` in channel-pair order.** No `rbf` per add —
deliberately, and this is the one place I depart from `fn_hc_dense_mix`'s style.

Basis: torch's `.mean(dim=-2)` over a bf16 tensor reduces through `at::opmath_type` (float for
bf16) and applies the `1/n` factor once, at the end of the reduction; it does not round each partial
sum to bf16. So a per-add bf16 rounding would produce a *different* row than the one the drafter was
trained on, and would be a worse approximation, not a more faithful one. `__fdiv_rn` is exact here
(`1/4` is a power of two), and is the correctly-rounded form for any other `SS`.

Two caveats I am stating rather than hiding:

- The *summation order* over the four streams is my choice (s = 0,1,2,3 in sequence). torch's
  reduction over a 4-element strided dim is a sequential accumulate per output thread in the kernels
  I know of, but I did not read its source for this shape. A different fp32 order differs by ~2^-24
  relative, and the store rounds to bf16 (8-bit mantissa), so it disappears except at exact ties.
- I could not run torch to confirm either claim (no execution allowed). The alternative — emulating
  bf16 tensor arithmetic per add — is *not* what torch does and would deviate further, so I did not
  take it.

## The scratch buffer

`Scratch.tap4` (new field, `cuda_batch_state.zig`): `cap * dims * 2` bytes, one `dims`-wide bf16 row a
pass row. 7.9 MB at `default_chunk` (1536 rows), 84 MB at `max_chunk` (16384).

Why a new buffer and not an existing one: every `cap * dims` bf16 scratch already in `Scratch`
(`mixed`, `branch`, `rank_output`, `ple_vals`, `ple_emb`) holds live data at the moment
`tapRingWrite` runs — `s.mixed` in particular is the input of the very layer whose read-out just
produced it — and `s.h`, the source, cannot be collapsed in place. `s.ple_vals` is written later in
the same layer for `ple_layer`, so reusing it would be a trap for the next reader even where it is
briefly free.

Accounting: the field is allocated in `Scratch.alloc` (`s.tap4 = try s.add(gpa, d, cap * dims * 2)`)
and its bytes are added in `Scratch.bytes` as a separate `n += cap * dims * 2;` term with a comment,
so construction and preflight stay in step. The only test that reads that number is `cuda_batch.zig`'s
"`Scratch.bytes(default_chunk, …) < 2 << 30`" and its two monotonicity checks, which the added term
does not threaten (7.9 MB against a 2 GiB bound).

## Kernel-contract / launch-record tables

Checked `zig/src/core/contract/*.zig`: `cuda_lane_pending.zig` (records for `tf_q35d_*`),
`flashnext_cuda_records.zig`, `nvcc_133_records.zig`, and `registry.zig`'s `checkCuda` /
`contract_limits.zig`'s `live`/`fatbins`. **Those tables gate the launches they name; they do not
require every kernel in an image to appear.** `checkCuda` walks the record list, so an unrecorded
kernel is simply unchecked — nothing rejects the build or a test for its absence.

I therefore added **no record**, on purpose: a record needs register/static/bound numbers per SM that
only nvcc can produce, and a placeholder (`usage = none`, a missing `later_pending` on a `u.regs == 0`
entry) would be reported by `checkCuda` as "no compiled usage recorded" — i.e. inventing one would
turn a gap into a failure. The follow-up is a one-liner once the image exists: `tf-cuda-test
kernel-contract-limits <dir>` prints usage for the recorded launches, and `tf-cuda-test
kernel-contract` gate 5 does the same live; the new kernel's numbers then go into
`flashnext_cuda_records.zig` (or `cuda_lane_pending.zig` beside the other `qwen35_draft` entries).

## The pinned sha256

```
f7ec6e8a39e9d8fbb2d869d4049174a2d8ab4a7ee4c97fc781557ed3739ac570  zig/kernels/cuda/qwen35_draft.cu
```

Pinned at `zig/dense_serving_test.zig:34` (test "dense serving: owned CUDA images retain their
reviewed lane and decode bytes"), replacing
`88588b06f82e1dc1d067bb6e8825a6fbf5000599d68472524b6db897272afa1f`. Computed with `sha256sum` on the
final file; the comment above the pin now records that `tf_fxd2_mean4` is the appended entry and that
no existing kernel body, launch or argument list moved.

## Tests added (and the one that had to change)

All in `zig/src/families/flashnext/cuda_tap_ring.zig`, which is already reachable from the family root
(`cuda_launch.zig`'s `pub const tap_ring`, and `cuda_batch.zig`/`cuda_batch_state.zig`'s real use of
it), so they run in the same test binary as the file's existing tests. No new file, no new anchor
needed.

1. `test "the tap source default is one flip, and every source names a pointer"` — changed: asserts
   `TapSource.default == .mean4` (was `.readout`); the three `Source.ptr` expectations are untouched.
2. `test "the override still names all three bases, and an unknown name is the default"` — new: the
   `TF_FLASHNEXT_TAP_SOURCE` names still parse to `readout`/`normed`/`mean4`, and two near-misses
   (`"Readout"`, `"normd"`) do not become the other two bases.
3. `test "mean4: a tap row is the plain mean of the four raw streams, in exact bf16 bits"` — new:
   synthetic `hidden = 4`, four streams, two rows with exactly-representable means
   (`3.75, 5.0, 6.25, 7.5` and `-1, 0.5, 1.5, -2`), asserted against hard-coded bf16 bit patterns
   (`0x4070, 0x40A0, 0x40C8, 0x40F0, …`).
4. `test "mean4: the total divides in fp32 and rounds once, to nearest even"` — new: a sum whose fp32
   mean lands exactly on a bf16 tie (`1.001953125`) and must round to `0x3F80` by round-half-to-even —
   a per-add bf16 rounding or a half-up rule both fail it.
5. `test "mean4: a tap's rows land in its own slot, at the ring's wrap, and nowhere else"` — new:
   2 taps × 3 ring rows × hidden 4, two collapsed rows at position 2 (ring row 2, then wrap
   to row 0), a sentinel in every other cell, tap 1 checked to sit exactly `hidden` further into each
   slot, and `slotOffset` checked against `Ring.offset` for every (row, tap).

`zig/src/families/flashnext/cuda_draft2_fwd.zig`'s `test "the tap source is one constant, read once"`
also had to change: it asserted the default resolves to the read-out pointer (`0x1000`); it now asserts
`0x3000` (the raw streams) and that `readout`/`normed` still name `0x1000`/`0x2000`.

### Commands

```sh
cd /home/relja/Projects/tf-wt/mean4
zig build test-cuda-host          # the flashnext module's host tests, which include all five above
zig build test-qwen-serving-host  # the sha256 pin in dense_serving_test.zig
```

`test-cuda-host` is the step at `zig/build/cuda.zig:195`; `test-qwen-serving-host` is
`zig/build/living_dense.zig:14`. Both are host-only (no GPU, no fatbin). A failure prints the test
name; there is no build-level test filter for these two steps, so the names above are how you find a
specific one in the output. For a GPU build, the fatbins must be rebuilt from source — `zig build
-Dnvcc=<path to nvcc> …` or the dedicated `zig build fatbins` step — before serving; see the first
caveat under "What I could not verify".

## What I could not verify (nothing was executed)

I ran only read-only inspection (`read_file`, `grep`/`rg`, `git status|log|diff`, `ls`, `wc`, and one
`sha256sum` for the pin). I did **not** run and could not verify:

1. **That any of this compiles** — not the Zig, not the CUDA. `zig build`, `zig build test-cuda-host`,
   `zig build test-qwen-serving-host`, `zig fmt --check` were all forbidden. The likeliest places for a
   compile error I could not catch: the labeled `switch` arm with its `break :collapse` in
   `tapRingWrite`; `@splat` on a `const` `u16` for the test arrays; the `for (0..rows) |p| for (0..n_taps) |k|`
   chain in test 5.
2. **That the kernel's bits equal the host reference.** No test runs `tf_fxd2_mean4` — there is no GPU
   in my sandbox and no kernel test was added (a GPU contract test comparing it to `Mean4.collapse`
   is the missing piece; it belongs in `zig/tests/cuda/` beside `dspine_table_tests.zig` and needs a
   dispatch word in `zig/tests/cuda/main.zig`, which I did not add).
3. **That `Mean4.collapse` itself matches torch.** The fp32-accumulate claim is from ATen's opmath
   behaviour, not from running torch on this shape.
4. **Acceptance.** Whether mean4 moves acceptance into 2.4-3.3 is the whole point and is unmeasured;
   the A/B is `TF_FLASHNEXT_TAP_SOURCE=readout|normed|mean4` on a served run.
5. **That the fatbin is rebuilt.** `Kernels.load` now resolves `tf_fxd2_mean4` eagerly (the file's
   convention: every entry is a `try`). If a build embeds a *prebuilt* fatbin from before this change
   (`-Dfatbins=<dir>`), the flashnext engine refuses to open — a loud startup failure, not a silent
   wrong row, but it is one. Build the images from source (`-Dnvcc=<path>` or `zig build fatbins`)
   before serving.
6. **No launch-record gate.** Confirmed by reading `registry.checkCuda`/`contract_limits.zig` that no
   record is *required*; the new kernel is simply not gated by gates 1-5 yet.
7. **The `normed` stride bug is untouched** (out of scope, as instructed), and so are the depth rule,
   the head, the selector, prompt-cache/snapshot restore and multi-lane drafting.

Behavior with no tap ring present is unchanged by construction: `tapRingWrite` returns at
`self.tap_slot orelse return` before any new code, so a non-drafting pass launches nothing new and
touches no new bytes; the only cost it pays is the `tap4` allocation (7.9 MB at the default chunk),
which `Scratch.bytes` accounts for.

## Proof of the boundary

- `git -C /home/relja/Projects/tf-wt/mean4 status --short` → the eight ` M` files above, nothing else,
  no branch, no commit, no stash.
- `git -C /home/relja/Projects/tf-wt/dflash2 status --short` → empty (same as when I started).
- `git -C /home/relja/Projects/tf-wt/align status --short` → ` M cuda_draft2_check.zig`,
  ` M cuda_draft2_fwd.zig`, `?? cuda_draft2_stages.zig` — byte-for-byte what it showed when I started
  (that worktree belongs to another agent; I only read the status).
- Nothing outside `/home/relja/Projects/tf-wt/mean4` was written. No `zig build`, no `nvcc`, no
  `systemctl`, no kill, no dashboard call, no GPU touch.
