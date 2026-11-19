# Building and testing the flashnext CUDA tree — toolchain traps

Operational notes for `zig build` / `zig build test` / the GPU numerics runner on this fork.
They are not a claim about DFlash2's numbers; they are the places where a green build, a clean log
or an exit code lies. Status: **REFERENCE**.

## The two build graphs

- `zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc` (12-15 min, binary
  `zig-out/native/bin/tensorfold-native`) returns early into the **product** graph. The `test` step
  does not exist there — `-Dgpu=spark` plus `zig build test` fails with `error: no such step: test` —
  and no `test` block is compiled at all, so a green product build says nothing about whether test
  code type-checks.
- `zig build test` (~9.5k tests) runs without `-Dgpu`, under `umask 0022`. Under `umask 0002` the
  `cli_config` `--config` test fails on its own world-writable check — environment, not code.
- The GPU numerics runner `tf-cuda-test` (`zig/tests/cuda/main.zig`) is in neither of those: it
  builds only via `zig build install -Dnvcc=…` (installArtifact). `compile-cuda-tests` is a
  no-nvcc branch step and answers "no such step" once `-Dnvcc` is passed.
- Because the runner is unbuilt in daily flows its call sites drift: when `mtp.Draft.init` gained a
  `kv8` param, three `zig/tests/cuda` files still passed 7 args and the runner only broke when built.
  Keep fixtures at pre-flag semantics (`false`) unless the test is kv8-specific.

## Reading a test run

- Judge by `TEST_RC`, never by the log: the suite contains tests that simulate build-driver failures
  and print `failed command: … test --cache-dir …` on purpose. A clean full run may print no
  `Build Summary` and no `tests passed` line at all.
- **`zig build test` returning 0 is not evidence that new tests ran.** A module reached only through
  an unused `pub const` (`pub const x = @import("…_test.zig")`) is type-checked but its `test` blocks
  are never collected. Anchor them from the root with `test { _ = @import("<module>_test.zig"); }`,
  then prove it: `grep -rl '<test name>' .zig-cache/o` must find a binary. Run that binary with no
  arguments to see per-test `OK` lines (`All N tests passed.`).
- Iterate with narrow filters; the full suite is the final regression gate only. Extract the family's
  `zig test …` invocation from the last full-suite log (it carries all `-M` module manifests), drop
  the trailing `--listen=-`, append `--test-filter <substring>`. Pure-std modules run cheaper still
  via `zig test -lc <file>`.

## Test-code traps

- **Fixtures drift from their owners.** When a sibling adds a member to a real owner struct, a fake
  `Owner`/`Ops` stops compiling with `no field named 'X'` — mirror it with a faithful no-op (pointer
  const-ness matters: `note(_: *const @This(), …)` vs `*@This()`), don't delete the call site.
- Keep test-only dims as `usize` constants: a bare `const k = 128` (comptime_int) flowing into slice
  arithmetic fails as `type u7 cannot represent integer value 128` with no obvious source.
- `@hasDecl`/`@hasField` on an optional-pointer field: `@typeInfo(P).optional.child` is the
  *pointer*, and `@hasDecl` rejects pointer types — unwrap both (`optional.child` then
  `pointer.child`) before probing.
- Assembling a call from an `@embedFile("../families/…")` inside `zig/src/native/*_test.zig` does not
  compile (`embed of file outside package path`). Call the function and assert on its result.
- A host test cannot prove where a staged row comes from: the source is live at the write site, the
  row layout matches the vendor's wording, the copy pitch is byte-vs-element correct, and the helper
  has a production caller. Grep for the symbol — a `pub fn` reached only from tests means the real
  path is elsewhere.
- `Pass.copyCols` takes byte addresses but pitches and width in 16-bit values; advancing by `dims`
  instead of `dims * 2` writes interleaved garbage that still looks like data. A ring whose slots
  interleave several taps cannot be filled with one contiguous copy.

## Stale artifacts — when a fix appears perf-neutral

- nvcc fatbin steps for `kernels/cuda/*` are zig-cache-keyed. After editing a `.cu`, a cache hit
  re-embeds the old cubin and the change reads as "no effect". The build log is not a reliable
  witness: a full clean build printed 6 of 115 `nvcc (<name>.fatbin)` lines.
- Settle it with timestamps: `find .zig-cache -name '<image>.fatbin' -exec ls -la --time-style=+%H:%M {} \;`
  against the binary's mtime.
- The `*.fatbin` files under `.zig-cache/o/<hash>/` are small GENERATED ZIG MODULES that re-embed the
  real bytes; hand-deleting or hand-copying them mixes hash dirs and breaks module manifests in ways
  later builds cannot see. Recovery that works: `rm -rf .zig-cache`, one product build, `zig build test`.
- To verify which bytes a binary embeds without `ncu` (blocked here: `ERR_NVGPUCTRPERM`): compile the
  `.cu` by hand (`nvcc -fatbin`, `zig/build/cuda_kernels.zig`'s `torch_flags`,
  `-gencode arch=compute_121,code=sm_121`) and sha256-compare against the fresh cache fatbin.
- A build log that never mentions your kernel proves nothing either way; a fatbin newer than your
  edit is the proof it compiled.
- Chain `zig build … ; echo BUILD_RC=$? >> log` and grep `BUILD_RC=0` before believing any number the
  run produced — a stale `zig-out` binary serves happily and reports plausible rows.
- Disk: one worktree's `.zig-cache` reaches 14 GB and the NVMe runs above 90%. Do not create a second
  worktree just to build in.

## Zig 0.17 specifics

- No `std.time.Timer`, no `std.posix.getenv`. Use `std.c.getenv("X")` (returns `[*:0]u8`, so
  `std.mem.span`) and `std.c.clock_gettime(.MONOTONIC, &ts)`.
- `std.Io.Clock`'s members are `real|awake|boot|cpu_process|cpu_thread` — there is no `.monotonic`.
- A container-level `pub const x = f()` is comptime-evaluated and can never carry a `getenv`. Read the
  env once in `init` and store it on the struct; otherwise an "env override" is inert and the two
  A/B arms return byte-identical results (which is a wiring failure, not a null result).

## Module boundaries

- A new source dir needs its own module in `zig/build/cuda.zig`: a module cannot import a file that
  lives in another module's tree (`import of file outside module path`), and two instances over one
  file fail with `file exists in modules 'x' and 'y'`. Re-expose through the family's existing module
  (`cuda_launch.zig`) instead.
- A kernel symbol added to an always-loaded `Kernels` table is a serving-wide dependency: every
  server runs `Kernels.load` at start, so a missing/stale fatbin fails the *open* of non-drafting
  servers too.

## Timers and staged measurement

- A timer around an async launch measures the enqueue. Stage timers must bracket a
  `stream.synchronize()`; without one every stage of a 265 ms round read 0.1-0.3 ms.
- A stage named after one kernel may be several launches: `attn_i` covers the layer's prefix k/v
  projection too, and `TF_FLASHNEXT_HEAD_SPLIT=1` splits the head into GEMM / top-k / merge. Any
  stage whose ms disagree with its own byte model by >30% should get a split timer, not a hypothesis.
- A speed gate comparing single fixture runs measures GPU clock state: two captures of the same
  binary read 438 and 689 ms with every stage +70% uniformly. Median of ≥3 reps; gate the round and
  stages above ~1 ms. `tools/dflash2/regress.py` is the scaffold.
- A single-position fixture can only ever measure the cold round: `fuseFrom(lo, hi, kv_lo, kv_end)`
  reuses only when `hi > kv_end`, and replaying one position gives `hi == kv_end`.

## Structured output / libtfgrammar

`zig build -Dgpu=spark` never produces `zig-out/native/lib/`. `zig build grammar -Dxgrammar=DIR`
does (xgrammar pinned v0.2.8, checkout at `~/tools/xgrammar`, ~2 min). Without the lib every
structured-output / required-tool-call request gets HTTP 400 (`optional_libraries.zig`) — which
silently kills agent tool calls when the server backs the agent itself. Search order
(`grammar.zig`): `<exe>/../lib/`, beside the exe, loader path. The loader latches `lib_failed` after
the first miss, so installing the lib later needs a server restart.
