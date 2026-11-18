//! TF_FLASHNEXT_DRAFT_CHECK=1: score the CUDA DFlash2 drafter on a capture the engine just made.
//!
//! Research only, and off unless the variable is `1`. It answers, inside the engine rather than in a
//! torch harness, the one question `tools/dflash2/tap_accept.py` answers outside it: does
//! `cuda_draft2_fwd.zig`'s round — the real kernels over the real 81-tensor checkpoint — land the
//! same tokens as `tools/dflash2/reference.py` on the same tap ring? What it is not: a serving path.
//! It drafts nothing for a reply, holds no lane, touches no cache; it reads one capture's ring, runs
//! rounds over it and logs a line.
//!
//! The metric is **agreement on teacher-forced text**, not the vendor's acceptance length. The
//! engine prefilled a fixed sequence and this replays it: at every round the drafter is handed the
//! true anchor and is scored against the true next `block - 1` tokens. A real verify round draws its
//! anchor from the target's own sampling, so a drafter that merely learned "copy the context" scores
//! higher here than it would in production. Treat the number as a tape of the drafter against its
//! reference, never as a speed claim.
//!
//! The loop is `tap_accept.py`'s, in the same order:
//!
//! * `p` starts at the ring's oldest row, `max(0, ids.len - rows)` (older positions are already
//!   overwritten, so a round there would read another position's taps);
//! * the context is the *prefix*, positions `[max(0, p - window), p - 1]` (the anchor's own position
//!   is not in the keys: `Round.ctx` is `[position - ctx_len, position)`), gathered out of the ring
//!   (`slot = pos % rows`) into one contiguous `[ctx_len, n_taps * hidden]` span, because the
//!   drafter's cross-attention reads exactly that;
//! * the block is `ids[p]` then `block - 1` mask tokens, at positions `p .. p + block - 1`;
//! * `accepted` is the longest common prefix of the proposals and `ids[p + 1 .. p + block)`, and the
//!   walk advances `1 + accepted`.
//!
//! Ground truth, measured on a real capture (fp8 KV, 1455 tokens, `--ctx-mode prefix`): mean advance
//! 1.342, 0.342 of 7 accepted, 80% of rounds with nothing accepted. The gate is per-round token
//! identity — bf16 kernels against an fp32 reference need not be bit-identical, but a mean advance
//! near 1.34 is what "reproduces the reference" means here.
//!
//! Two things the engine holds in a form the drafter's kernels cannot read directly, and what this
//! file does about them:
//!
//! * the head. `Draft` scores the target's whole vocabulary with `tf_fxd2_head_topk`, which reads
//!   row-major bf16 rows; the engine keeps that same tensor repacked into bf16 matrix-core tiles
//!   (`cuda/qmmf.zig`'s `packBf16`). `detile` inverts the packing a 64-row block at a time, on the
//!   host, into one row-major `[vocab, hidden]` device buffer for the check's lifetime. No kernel is
//!   added for it: `kernels/cuda/qwen35_draft.cu`'s bytes are pinned by `dense_serving_test.zig`.
//! * the taps. `cuda_tap_dump.zig` leaves the ring on the host, already fused five taps a row, which
//!   is the span `Round.ctx` wants; `Engine.forward` hands it over as it stands.
//!
//! It runs on the prefill the capture opens on, so both variables are needed: the ring only exists
//! because `TF_FLASHNEXT_TAP_DUMP` asked for it, and the pass has to fit one chunk, or
//! `cuda_tap_dump.begin` refuses the capture and there is nothing to score.
//!
//! ```sh
//! TF_FLASHNEXT_TAP_DUMP=/tmp/tapdump-draftcheck TF_FLASHNEXT_DRAFT_CHECK=1 \
//! TF_FLASHNEXT_DRAFT_CHECK_ROUNDS=120 TF_FLASHNEXT_CHUNK=2048 \
//! TF_FLASHNEXT_DRAFTER_DIR=$HOME/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6 \
//!   <the server binary> --model <the served flashnext folder> ...
//! python3 tools/dflash2/tap_dump_run.py --dump-dir /tmp/tapdump-draftcheck   # one 1455-token prefill
//! ```
//!
//! then one line in the engine's log: `draftcheck: rounds 120 accepted 0.342 mean_advance 1.342
//! zero_accept 80.0%`. The per-round `p`/`ctx`/`proposed`/`targets` lines are `std.log.debug`, so
//! they only reach the log in a build whose log level is debug.
//!
//! **When the number is wrong, one line cannot say where.** `TF_FLASHNEXT_DRAFT_DEBUG=<dir>` (with
//! `TF_FLASHNEXT_DRAFT_DEBUG_ROUNDS=k[,k]`, default `0,1`) leaves one round's internals in
//! `<dir>/round<k>/` as raw `.bin` plus a small `tensors.json`: the fused context rows, the block's
//! input embeddings, the block hidden after each of the five layers, the head's top-16 ids and
//! logits, and the proposals (`cuda_draft2_debug.zig` has the layout). Each file isolates one
//! stage — `tf_fxd2_fuse`, embed/anchor/mask, the layers one at a time, `tf_fxd2_head_topk` (which
//! also settles the engine-head detile), `tf_fxd2_select` — and
//! `python3 tools/dflash2/draft_compare.py --debug-dir <dir> --dump <the capture>` reads them
//! against `reference.py` stage by stage, reporting relative difference, row alignment and top-1
//! agreement a stage, then token agreement and acceptance for the round.
//!
//! ```sh
//! TF_FLASHNEXT_TAP_DUMP=/tmp/tapdump-draftcheck TF_FLASHNEXT_DRAFT_CHECK=1 \
//! TF_FLASHNEXT_DRAFT_DEBUG=/tmp/draftdbg TF_FLASHNEXT_DRAFT_DEBUG_ROUNDS=0,1,64 \
//!   <the same server invocation as above>
//! python3 tools/dflash2/draft_compare.py --debug-dir /tmp/draftdbg --dump /tmp/tapdump-draftcheck
//! ```
//!
//! Round 0's context is one row and round 1's two, so a dump that is meant to test the fusion or the
//! context attention should name a later round by hand: `TF_FLASHNEXT_DRAFT_DEBUG_ROUNDS=0,64`.

const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const debug = @import("cuda_draft2_debug.zig");
const stage_dump = @import("cuda_draft2_stages.zig");
const draft2 = @import("cuda_draft2.zig");
const fwd = @import("cuda_draft2_fwd.zig");

/// The drafter's weight precision, re-exported so a caller (`cli/draft_check.zig`) names one type.
pub const Precision = fwd.Precision;
const weights = @import("cuda_prompt_wt.zig");

/// The geometry a capture and the drafter must agree on, from the drafter's own fixed widths.
pub const n_taps: usize = fwd.n_taps;
pub const hidden: usize = fwd.hidden;
pub const vocab: usize = fwd.vocab;
pub const block: usize = fwd.block;

/// Rounds a check runs when TF_FLASHNEXT_DRAFT_CHECK_ROUNDS says nothing: the reference run's cap.
pub const default_rounds: usize = 120;

/// Whether TF_FLASHNEXT_DRAFT_CHECK asks for the check: exactly `1` turns it on, anything else
/// (including unset and empty) leaves the engine completely inert.
pub fn enabled() bool {
    const v = std.c.getenv("TF_FLASHNEXT_DRAFT_CHECK") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// The directory TF_FLASHNEXT_DRAFTER_DIR names, or null. An empty value names no directory.
pub fn dirFromEnv() ?[]const u8 {
    const v = std.c.getenv("TF_FLASHNEXT_DRAFTER_DIR") orelse return null;
    const text = std.mem.span(v);
    return if (text.len == 0) null else text;
}

/// TF_FLASHNEXT_DRAFT_CHECK_ROUNDS, or `default_rounds`. A malformed or zero value is the default
/// rather than a silent cap of zero rounds, which would look like a passing run with no rounds.
pub fn roundsFromEnv() usize {
    const v = std.c.getenv("TF_FLASHNEXT_DRAFT_CHECK_ROUNDS") orelse return default_rounds;
    const n = std.fmt.parseInt(usize, std.mem.trim(u8, std.mem.span(v), " \t"), 10) catch return default_rounds;
    return if (n == 0) default_rounds else n;
}

/// The target's two tensors, as the engine holds them.
pub const Model = struct {
    /// The embedding table's rows, bf16 `[vocab, hidden]`, row-major and contiguous (the engine's
    /// dense table). 0 when the checkpoint's table is quantized, which this check refuses.
    embed: u64 = 0,
    /// Its row count: a table that is not the whole vocabulary would embed the wrong rows quietly.
    embed_rows: usize = 0,
    /// The head's bf16 weights as `cuda/qmmf.zig` packs them into matrix-core tiles, `[npad, hidden]`.
    head: u64 = 0,
    /// The head's logical rows and width: a sharded head is not the whole vocabulary, and the
    /// drafter's selector is ranked against all of it.
    head_rows: usize = 0,
    head_cols: usize = 0,
    /// Whether those tiles hold bf16 weights (the dense head) rather than 4-bit codes.
    head_bf16: bool = false,
};

/// One engine capture: the ring as `cuda_tap_dump.zig` left it on the host, and the tokens it read.
pub const Capture = struct {
    /// `rows * n_taps * hidden * 2` host bytes; the row at position `p` is slot `p % rows`, and tap
    /// `k` is the `hidden`-wide slice at `k * hidden` inside that slot.
    ring: []const u8,
    rows: usize,
    /// The tokens the pass actually read, in order: position `i` is `ids[i]`.
    ids: []const u32,
    /// Where the capture left `ring.bin`/`ids.bin`: recorded in a debug dump so the comparator can
    /// find the same capture without being told twice. Empty when the caller has no directory.
    dir: []const u8 = "",
};

/// Walk the captured sequence with the CUDA drafter and log the block acceptance. Refuses, by name,
/// everything that would make the measurement mean nothing.
pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *cuda.Driver,
    stream: cuda.Stream,
    dir: []const u8,
    m: Model,
    cap: Capture,
    max_rounds: usize,
) !void {
    if (dir.len == 0) return error.DraftCheckNoDrafterDir;
    // TF_FLASHNEXT_DRAFT_FIXTURE=<dir>: the same check, but over a ring on disk instead of the one
    // this pass just captured — one round, at the position the fixture's manifest names. Before the
    // capture's own ring is even looked at, because the capture is then only the way the engine got
    // here.
    if (fixtureDirFromEnv()) |fx| return runFixture(gpa, io, driver, stream, dir, m, fx);
    // refused before the head is detiled (a second of host work): a directory that is not a DFlash2
    // drafter is not one, whatever the rest of the environment says
    if (!draft2.declared(gpa, io, dir)) return error.DraftCheckNotADrafter;
    try checkModel(m);
    const row_bytes = n_taps * hidden * 2;
    if (cap.rows < block or cap.ring.len != try std.math.mul(usize, cap.rows, row_bytes)) return error.DraftCheckNoRing;
    if (cap.ids.len < block + 2) return error.DraftCheckSequenceTooShort;
    if (max_rounds == 0) return error.DraftCheckNoRounds;

    // 1. the head the drafter's top-k reads, out of the tiles the engine keeps it in
    var head = try detile(gpa, io, driver, m.head, vocab, hidden);
    defer head.free();

    // 2. the drafter itself: the served checkpoint, its 81 tensors, its five layers' rings
    const d = try fwd.Draft.open(gpa, io, driver, stream, dir, targetConfig(), .{ .embed = m.embed, .head = head.ptr }, null);
    defer d.deinit();
    var st = try d.create();
    defer st.deinit();

    // 3. the context span: a prefix of the capture's own rows, re-gathered and uploaded every round
    const ctx_rows = @min(d.c.window, cap.rows);
    var ctx = try cuda.DeviceBuffer.alloc(driver, try std.math.mul(usize, ctx_rows, row_bytes));
    defer ctx.free();

    // 4. the debug sink (TF_FLASHNEXT_DRAFT_DEBUG): the two spans a traced round copies into, and the
    // host staging it is read back through. Null, and a round is exactly what it was.
    const spec: ?debug.Spec = debug.specFromEnv();
    var sink: ?Sink = if (spec != null) try Sink.alloc(gpa, driver, d.c, ctx_rows) else null;
    defer if (sink) |*k| k.deinit(gpa);
    const gather = try gpa.alloc(u8, try std.math.mul(usize, ctx_rows, row_bytes));
    defer gpa.free(gather);

    const n = cap.ids.len;
    var p: usize = if (n > cap.rows) n - cap.rows else 0;
    var rounds: usize = 0;
    var accepted_total: usize = 0;
    var advance_total: usize = 0;
    var zero: usize = 0;
    var prop: [block - 1]u32 = @splat(0);
    while (rounds < max_rounds and p + block < n) : (rounds += 1) {
        // the prefix: what the target had computed when the block was proposed, capped by the window
        const len = @min(ctx_rows, p);
        const lo = p - len;
        for (0..len) |i| {
            const slot = (lo + i) % cap.rows;
            @memcpy(gather[i * row_bytes ..][0..row_bytes], cap.ring[slot * row_bytes ..][0..row_bytes]);
        }
        try ctx.upload(0, gather[0 .. len * row_bytes]);

        // a traced round copies its internals into the sink as it runs; every other round is untouched
        const want = if (spec) |sp| sp.wants(rounds) else false;
        if (sink) |*k| d.trace(if (want) k.debug else null);

        try d.draft(&st, .{ .ctx = ctx.ptr, .ctx_len = len, .anchor = cap.ids[p], .position = p }, &prop);

        // after the round, not during it: `draft` synchronises the stream, so every copy a traced
        // round queued has landed and the scratch still holds the round's own tables
        if (want) try dumpRound(gpa, io, spec.?, &sink.?, d, &st, .{
            .round = rounds,
            .p = p,
            .ctx_len = len,
            .ctx_lo = lo,
            .anchor = cap.ids[p],
        }, cap);

        const accepted = acceptedOf(&prop, cap.ids[p + 1 ..][0 .. block - 1]);
        accepted_total += accepted;
        advance_total += 1 + accepted;
        if (accepted == 0) zero += 1;
        std.log.debug("draftcheck: p {d} ctx {d} accepted {d} proposed {any} targets {any}", .{
            p, len, accepted, prop, cap.ids[p + 1 ..][0 .. block - 1],
        });
        p += 1 + accepted;
    }
    if (rounds == 0) return error.DraftCheckNoRound;

    const total: f64 = @floatFromInt(rounds);
    std.log.info("draftcheck: rounds {d} accepted {d:.3} mean_advance {d:.3} zero_accept {d:.1}%", .{
        rounds,
        @as(f64, @floatFromInt(accepted_total)) / total,
        @as(f64, @floatFromInt(advance_total)) / total,
        100.0 * @as(f64, @floatFromInt(zero)) / total,
    });
}

// -----------------------------------------------------------------------------------------------
// Fixture mode: one round, at one position, over a ring on disk
// -----------------------------------------------------------------------------------------------
//
// TF_FLASHNEXT_DRAFT_CHECK=1 with TF_FLASHNEXT_DRAFT_FIXTURE=<dir> runs exactly one round of the
// CUDA drafter over the ring `<dir>` holds, at the position its `manifest.json` names, and writes
// every intermediate to TF_FLASHNEXT_DRAFT_STAGES=<out>: a `.npy` a tensor and a `stages.json`.
//
// Why it exists: the walk above scores the drafter on a capture this engine just made, which is the
// right question for "does the drafter predict" and the wrong one for "does this stage agree with
// another implementation" — a PyTorch side of an alignment test can be fed bytes, not a prompt.
// Fixture mode is the same production round (`fwd.Draft.draft`, the real kernels, the real 81-tensor
// checkpoint) over frozen bytes, so a stage-by-stage diff compares arithmetic and not tokenizers.
//
// The round, exactly:
//
//   * block 8, window 2048, anchor position `P = pos_end - 1` from the fixture;
//   * `ctx_len = min(window, P)`, context positions `[P - ctx_len, P)`, the row for absolute
//     position `j` being ring slot `j % rows` — the context ends one position before the anchor and
//     never contains it;
//   * the block's ids are the anchor then `block - 1` mask tokens, at positions `P .. P + block`;
//   * the proposals are the selector's walk over the head's rows 1..7.
//
// Inert unless both variables are set: no serving path reads either one.

/// The fixture's mask token: `mask_token_id` in the drafter's config, pinned here so a checkpoint
/// that disagrees is refused rather than silently drafting a different block.
pub const mask_token: u32 = 248_077;

/// The directory TF_FLASHNEXT_DRAFT_FIXTURE names, or null.
pub fn fixtureDirFromEnv() ?[]const u8 {
    const v = std.c.getenv("TF_FLASHNEXT_DRAFT_FIXTURE") orelse return null;
    const text = std.mem.span(v);
    return if (text.len == 0) null else text;
}

/// The directory TF_FLASHNEXT_DRAFT_STAGES names, or null.
pub fn stageDirFromEnv() ?[]const u8 {
    const v = std.c.getenv("TF_FLASHNEXT_DRAFT_STAGES") orelse return null;
    const text = std.mem.span(v);
    return if (text.len == 0) null else text;
}

/// TF_FLASHNEXT_STAGE_CTX_TENSOR=1: write the fused context rows as a `.npy` as well as a summary.
/// Off by default, because the rows are a window wide (`ctx_len x hidden` bf16, 20 MB of f32 `.npy`
/// at the served window) and a serving-sized dump does not need them.
pub fn ctxTensorWanted() bool {
    const v = std.c.getenv("TF_FLASHNEXT_STAGE_CTX_TENSOR") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn nowNs(io: std.Io) u64 {
    const t = std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
    return @intCast(@max(0, t));
}

/// The target's two tensors, as every path in this file needs them.
fn checkModel(m: Model) !void {
    if (m.embed == 0 or m.embed_rows != vocab) return error.DraftCheckNoEmbedTable;
    if (m.head == 0) return error.DraftCheckNoHead;
    if (!m.head_bf16) return error.DraftCheckHeadNotBf16;
    if (m.head_rows != vocab or m.head_cols != hidden) return error.DraftCheckHeadNotWholeVocab;
}

/// The fixture's manifest, as much of it as a round needs.
const Man = struct {
    rows: usize,
    taps: []u32,
    hidden: usize,
    window: usize,
    block: usize,
    pos_end: usize,
    row_bytes: usize,
    source: []const u8 = "",
    /// optional: one round per entry, run in order on one state. Positions that advance by the
    /// accepted run give the served (reuse) path; a single position is the cold round.
    positions: []const u32 = &.{},
    ring_file: []const u8 = "ring.bin",
    ids_file: []const u8 = "ids.bin",
};

/// A fixture on the host: the ring, the ids it was produced from, and what they hash to.
const Fixture = struct {
    ring: []u8,
    ids: []u32,
    rows: usize,
    row_bytes: usize,
    hidden: usize,
    block: usize,
    n_taps: usize,
    window: usize,
    pos_end: usize,
    /// the rounds to run, in order, when the manifest carries `positions`
    positions: []usize = &.{},
    sha: [3][64]u8 = .{ @splat('0'), @splat('0'), @splat('0') },

    fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Fixture {
        var path_buf: [1024]u8 = undefined;
        const man_path = try std.fmt.bufPrint(&path_buf, "{s}/manifest.json", .{dir});
        const text = try std.Io.Dir.cwd().readFileAlloc(io, man_path, gpa, .limited(1 << 20));
        defer gpa.free(text);
        var parsed = try std.json.parseFromSlice(Man, gpa, text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const man = parsed.value;
        if (man.taps.len != n_taps) return error.FixtureTapCount;

        const ring_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, man.ring_file });
        const ring = try std.Io.Dir.cwd().readFileAlloc(io, ring_path, gpa, .limited(1 << 30));
        errdefer gpa.free(ring);

        const ids_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, man.ids_file });
        const ids_bytes = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, gpa, .limited(1 << 28));
        defer gpa.free(ids_bytes);
        if (ids_bytes.len % 4 != 0) return error.FixtureIdsShape;
        const ids = try gpa.alloc(u32, ids_bytes.len / 4);
        errdefer gpa.free(ids);
        for (ids, 0..) |*id, i| id.* = std.mem.readInt(u32, ids_bytes[i * 4 ..][0..4], .little);

        var f: Fixture = .{
            .ring = ring,
            .ids = ids,
            .rows = man.rows,
            .row_bytes = man.row_bytes,
            .hidden = man.hidden,
            .block = man.block,
            .n_taps = man.taps.len,
            .window = man.window,
            .pos_end = man.pos_end,
        };
        if (man.positions.len != 0) {
            const pos = try gpa.alloc(usize, man.positions.len);
            errdefer gpa.free(pos);
            for (man.positions, 0..) |v, idx| pos[idx] = v;
            f.positions = pos;
        }
        f.sha[0] = sha256Hex(ring);
        f.sha[1] = sha256Hex(ids_bytes);
        f.sha[2] = sha256Hex(text);
        return f;
    }

    fn deinit(f: *Fixture, gpa: std.mem.Allocator) void {
        gpa.free(f.ring);
        gpa.free(f.ids);
        if (f.positions.len != 0) gpa.free(f.positions);
    }
};

fn sha256Hex(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

/// One round of the production drafter over the fixture, dumped stage by stage.
fn runFixture(
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *cuda.Driver,
    stream: cuda.Stream,
    dir: []const u8,
    m: Model,
    fx_dir: []const u8,
) !void {
    return runFixtureAt(gpa, io, driver, stream, dir, m, fx_dir, stageDirFromEnv() orelse return error.DraftCheckNoStageDir, 0, null);
}

/// `runFixture` with the destination named by the caller rather than `TF_FLASHNEXT_DRAFT_STAGES`,
/// so a standalone runner (`draft_check.zig`) can write several reps without touching the process
/// environment the serving engine reads.
pub fn runFixtureAt(
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *cuda.Driver,
    stream: cuda.Stream,
    dir: []const u8,
    m: Model,
    fx_dir: []const u8,
    out_dir_: []const u8,
    reps: usize,
    /// The weights' precision, or null for the checkpoint's own (see `fwd.Precision`).
    want: ?fwd.Precision,
) !void {
    if (!draft2.declared(gpa, io, dir)) return error.DraftCheckNotADrafter;
    try checkModel(m);

    var fx = try Fixture.open(gpa, io, fx_dir);
    defer fx.deinit(gpa);
    if (fx.hidden != hidden or fx.block != block or fx.n_taps != n_taps) return error.FixtureGeometry;
    if (fx.row_bytes != n_taps * hidden * 2) return error.FixtureRowBytes;
    if (fx.rows < block or fx.ring.len != try std.math.mul(usize, fx.rows, fx.row_bytes)) return error.FixtureRingShape;
    if (fx.ids.len != fx.pos_end or fx.pos_end < block + 2) return error.FixtureIdsShape;
    // the fixture is a ring, not a pass: a position past `pos_end` would read a slot another
    // position overwrote, and the round would be quietly over the wrong rows
    if (fx.pos_end > fx.rows + 1) return error.FixturePastRing;
    const P = fx.pos_end - 1;

    // 1. the head the drafter's top-k reads, out of the tiles the engine keeps it in
    var head = try detile(gpa, io, driver, m.head, vocab, hidden);
    defer head.free();

    // 2. the drafter itself, exactly as a serving engine opens it
    const d = try fwd.Draft.open(gpa, io, driver, stream, dir, targetConfig(), .{ .embed = m.embed, .head = head.ptr }, want);
    defer d.deinit();
    if (d.c.mask != mask_token) return error.FixtureMaskMismatch;

    // 3. the rounds. A manifest may carry `positions`: one round per entry, run in order on the SAME
    // state, each ending where its block begins. Consecutive positions that advance by the accepted
    // run are the SERVED regime: `fuseFrom` reuses the span the round before projected, so only the
    // accepted rows are fused and re-projected. Replaying one position is a rewind, which the engine
    // deliberately re-fuses in full, so a one-position fixture can only measure the cold round
    // (measured: two warm rounds on `fixture/window` still read 435.8 ms against 438.0 cold).
    const win: usize = d.c.window;
    var one: [1]usize = .{P};
    const seq: []const usize = if (fx.positions.len != 0) fx.positions else one[0..];

    // a window wide whatever a round's own span is, so a sequence can grow its span round to round
    // without reallocating and no round can outrun the buffer
    var ctx = try cuda.DeviceBuffer.alloc(driver, try std.math.mul(usize, win, fx.row_bytes));
    defer ctx.free();
    const gather = try gpa.alloc(u8, win * fx.row_bytes);
    defer gpa.free(gather);
    const ctx_host = try gpa.alloc(u8, win * hidden * 2);
    defer gpa.free(ctx_host);

    var st = try d.create();
    defer st.deinit();

    const n_reps: usize = if (reps == 0) seq.len else reps;
    for (0..n_reps) |rep| {
        const pos = seq[rep % seq.len];
        if (pos >= fx.pos_end) return error.FixturePositionPastIds;
        // the span is `min(window, pos)` positions ending where the block begins: the ring's own
        // length, never the sequence's
        const ctx_len = @min(win, pos);
        const lo = pos - ctx_len;
        for (0..ctx_len) |idx| {
            const slot = (lo + idx) % fx.rows;
            @memcpy(gather[idx * fx.row_bytes ..][0..fx.row_bytes], fx.ring[slot * fx.row_bytes ..][0..fx.row_bytes]);
        }
        try ctx.upload(0, gather[0 .. ctx_len * fx.row_bytes]);

        var rep_buf: [4096]u8 = undefined;
        const out_dir: []const u8 = if (n_reps == 1) out_dir_ else try std.fmt.bufPrint(&rep_buf, "{s}/rep{d}", .{ out_dir_, rep });
        if (n_reps > 1) try std.Io.Dir.cwd().createDirPath(io, out_dir);

        // 4. one round, stage by stage. The collector synchronises at every boundary, so its numbers
        // are the work's and not the enqueue's.
        var k = stage_dump.Collector.init(gpa, io, driver, stream);
        defer k.deinit();
        d.traceStages(&k);
        var prop: [block - 1]u32 = @splat(0);
        const t0 = nowNs(io);
        try d.draft(&st, .{ .ctx = ctx.ptr, .ctx_len = ctx_len, .anchor = fx.ids[pos], .position = pos }, &prop);
        const t1 = nowNs(io);
        d.traceStages(null);
        const round_ms = @as(f64, @floatFromInt(t1 - t0)) / 1e6;

        // 5. the context: a window wide, and (TF_FLASHNEXT_STAGE_CTX_TENSOR=1) the one tensor a full
        // round would otherwise only summarise. Off by default so a serving-sized window does not put
        // 20 MB of `.npy` beside every dump; a harness that wants the fuse's own rows asks for them.
        var done: usize = 0;
        while (done < ctx_len) {
            const n = @min(fwd.ctx_chunk, ctx_len - done);
            const at = lo + done;
            const slot = at % win;
            const head_n = @min(n, win - slot);
            try cuda.memory.download(driver, st.fused.ptr + slot * hidden * 2, ctx_host[done * hidden * 2 ..][0 .. head_n * hidden * 2]);
            if (n > head_n) try cuda.memory.download(driver, st.fused.ptr, ctx_host[(done + head_n) * hidden * 2 ..][0 .. (n - head_n) * hidden * 2]);
            done += n;
        }
        if (ctxTensorWanted()) {
            const e = k.find("fc_ctx") orelse return error.StageMissing;
            try k.addHost("fc_ctx_rows", ctx_host, ctx_len, hidden, .bf16, e.ms);
        }

        // 6. the selector's lattice scores, on the host, from the round's own tables
        const sel = try selectorScores(gpa, driver, d, &k, fx.ids[pos], &prop);
        defer gpa.free(sel.scores);
        try k.addHost("selector_scores", std.mem.sliceAsBytes(sel.scores), sel.rows, sel.topk, .f32, sel.ms);

        // 7. the dumps
        try writeStages(gpa, io, out_dir, .{
            .drafter = dir,
            .pos = pos,
            .ctx_len = ctx_len,
            .ctx_lo = lo,
            .window = win,
            .anchor = fx.ids[pos],
            .round_ms = round_ms,
            .ctx_stats = stage_dump.stats(ctx_host, ctx_len, hidden, .bf16),
            .sel = sel,
            .sha = fx.sha,
        }, &k, &prop);

        std.log.info("draftfix: rep {d} position {d} ctx {d} from {d} anchor {d} proposals {any} round {d:.2} ms -> {s}", .{
            rep, pos, ctx_len, lo, fx.ids[pos], prop, round_ms, out_dir,
        });
        for (prop) |id| {
            if (@as(usize, id) >= vocab) return error.FixtureProposalOutOfVocab;
        }
        var distinct: usize = 0;
        for (0..prop.len) |idx| {
            var seen = false;
            for (0..idx) |q| {
                if (prop[q] == prop[idx]) seen = true;
            }
            if (!seen) distinct += 1;
        }
        if (distinct < 2) std.log.warn("draftfix: the walk is degenerate ({d} distinct ids)", .{distinct});
    }
}

/// The selector's scores for the predecessors the walk actually took, recomputed on the host from
/// the round's own dumped tables.
///
/// `tf_fxd2_select` keeps `score[]` in shared memory — only the proposals leave the kernel — so a
/// reference implementation's lattice has no device buffer to be downloaded from. The recomputation
/// is the kernel's arithmetic, element for element: `hp[j] = x[t] . wproj[j]`, then
/// `ctx[r] = pred[prev][r] * hp[r]` and `score[c] = unary[c] + ctx . succ[cand[c]]`, all in fp32 with
/// the same fused multiply-add and the same ascending order. `agree` is how many steps' argmax
/// reproduces the kernel's own proposal, which is the only check that matters here: a recomputation
/// that disagrees with the proposals is a different walk, and the JSON says so rather than hiding it.
const Sel = struct {
    scores: []f32,
    rows: usize,
    topk: usize,
    ms: f64,
    agree: usize,
};

fn selectorScores(
    gpa: std.mem.Allocator,
    driver: *const cuda.Driver,
    d: *fwd.Draft,
    k: *stage_dump.Collector,
    anchor: u32,
    prop: []const u32,
) !Sel {
    const hn = k.find("final_norm_block") orelse return error.StageMissingFinalNorm;
    const cand = k.find("logits_top16_ids") orelse return error.StageMissingHeadIds;
    const unary = k.find("logits_top16_vals") orelse return error.StageMissingHeadVals;
    const rows = hn.rows;
    const H = hn.cols;
    const topk = cand.cols;
    if (cand.rows != rows or unary.rows != rows or unary.cols != topk) return error.StageShapeMismatch;
    if (prop.len != rows) return error.StageShapeMismatch;
    const rank = d.c.rank;

    const t0 = nowNs(k.io);
    // the rank projection, [rank, H] bf16, and one row of each codebook at a time (both are
    // [vocab, rank]: a step reads at most `topk` successors and one predecessor)
    const wproj = try gpa.alloc(u8, try std.math.mul(usize, rank * H, 2));
    defer gpa.free(wproj);
    try cuda.memory.download(driver, d.w.hidden_proj, wproj);
    const pred_row = try gpa.alloc(u8, rank * 2);
    defer gpa.free(pred_row);
    const succ_row = try gpa.alloc(u8, rank * 2);
    defer gpa.free(succ_row);
    const hp = try gpa.alloc(f32, rank);
    defer gpa.free(hp);
    const ctx = try gpa.alloc(f32, rank);
    defer gpa.free(ctx);
    const scores = try gpa.alloc(f32, rows * topk);
    errdefer gpa.free(scores);

    var agree: usize = 0;
    for (0..rows) |t| {
        // the predecessor this step walked from: the anchor at t = 0, else the previous proposal
        const prev_token: u32 = if (t == 0) anchor else prop[t - 1];
        if (@as(usize, prev_token) >= vocab) return error.StagePrevPastVocab;
        for (0..rank) |j| {
            const w = wproj[j * H * 2 ..][0 .. H * 2];
            var s: f32 = 0;
            for (0..H) |c| s = @mulAdd(f32, bf16At(hn.host, t * H + c), bf16At(w, c), s);
            hp[j] = s;
        }
        try cuda.memory.download(driver, d.w.pred + @as(u64, prev_token) * @as(u64, rank) * 2, pred_row);
        for (0..rank) |r| ctx[r] = bf16At(pred_row, r) * hp[r];
        for (0..topk) |c| {
            const cid = std.mem.readInt(u32, cand.host[(t * topk + c) * 4 ..][0..4], .little);
            if (@as(usize, cid) >= vocab) return error.StageCandPastVocab;
            try cuda.memory.download(driver, d.w.succ + @as(u64, cid) * @as(u64, rank) * 2, succ_row);
            var s: f32 = 0;
            for (0..rank) |r| s = @mulAdd(f32, ctx[r], bf16At(succ_row, r), s);
            const unary_bits = std.mem.readInt(u32, unary.host[(t * topk + c) * 4 ..][0..4], .little);
            scores[t * topk + c] = @as(f32, @bitCast(unary_bits)) + s;
        }
        var best: usize = 0;
        for (1..topk) |c| {
            if (scores[t * topk + c] > scores[t * topk + best]) best = c;
        }
        const picked = std.mem.readInt(u32, cand.host[(t * topk + best) * 4 ..][0..4], .little);
        if (picked == prop[t]) agree += 1;
    }
    const t1 = nowNs(k.io);
    return .{ .scores = scores, .rows = rows, .topk = topk, .ms = @as(f64, @floatFromInt(t1 - t0)) / 1e6, .agree = agree };
}

fn bf16At(bytes: []const u8, i: usize) f32 {
    const bits = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
    return @bitCast(@as(u32, bits) << 16);
}

/// What one fixture round was, beside the stages it produced.
const RoundInfo = struct {
    drafter: []const u8,
    pos: usize,
    ctx_len: usize,
    ctx_lo: usize,
    window: usize,
    anchor: u32,
    round_ms: f64,
    ctx_stats: stage_dump.Stats,
    sel: Sel,
    sha: [3][64]u8,
};

/// Write `tensors/<stage>.npy` a stage and the `stages.json` beside them. Every stage is looked up
/// by name in the order a round produces them, so a round that skipped one is an error here rather
/// than a shorter list a reference implementation would diff against the wrong rows.
fn writeStages(gpa: std.mem.Allocator, io: std.Io, out_dir: []const u8, info: RoundInfo, k: *stage_dump.Collector, prop: []const u32) !void {
    var dir_buf: [1024]u8 = undefined;
    var path_buf: [1200]u8 = undefined;
    const tensors_dir = try std.fmt.bufPrint(&dir_buf, "{s}/tensors", .{out_dir});
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, out_dir);
    try cwd.createDirPath(io, tensors_dir);

    var text: std.Io.Writer.Allocating = .init(gpa);
    defer text.deinit();
    const w = &text.writer;
    var cmd_buf: [4096]u8 = undefined;
    const cmd = commandText(io, gpa, &cmd_buf) catch "(unknown)";
    var ck_buf: [1024]u8 = undefined;
    const checkpoint = checkpointPath(io, gpa, &ck_buf) catch "unknown";

    try w.print(
        "{{\"impl\":\"tensorfold-cuda\",\"device\":\"cuda\",\"dtype\":\"bf16\",\"checkpoint\":\"{s}\",\"drafter\":\"{s}\"," ++
            "\"fixture_sha256\":{{\"ring.bin\":\"{s}\",\"ids.bin\":\"{s}\",\"manifest.json\":\"{s}\"}}," ++
            "\"command\":\"{s}\",\"n_stages\":{d}," ++
            "\"round\":{{\"position\":{d},\"anchor\":{d},\"ctx_len\":{d},\"ctx_lo\":{d},\"window\":{d},\"block\":{d}," ++
            "\"mask\":{d},\"round_ms\":{d:.3},\"selector_argmax_agrees\":{d},\"proposals\":[{d},{d},{d},{d},{d},{d},{d}]}},\"stages\":[",
        .{
            checkpoint,                                         info.drafter,
            info.sha[0][0..],                                   info.sha[1][0..],
            info.sha[2][0..],                                   cmd,
            countStages(),                                      info.pos,
            info.anchor,                                        info.ctx_len,
            info.ctx_lo,                                        info.window,
            block,                                              mask_token,
            info.round_ms,                                      info.sel.agree,
            prop[0],                                            prop[1],
            prop[2],                                            prop[3],
            prop[4],                                            prop[5],
            prop[6],
        },
    );

    var name_buf: [stage_dump.name_max]u8 = undefined;
    var n: usize = 0;
    var sum_ms: f64 = 0;
    // the context first: a window-wide tensor, summarised rather than written
    {
        const s = info.ctx_stats;
        const e = k.find("fc_ctx") orelse return error.StageMissing;
        try w.print(
            "{{\"stage\":\"fc_ctx\",\"shape\":[{d}, {d}],\"dtype\":\"bf16\",\"src_dtype\":\"bf16\",\"mean\":{d:.8},\"std\":{d:.8}," ++
                "\"l2norm\":{d:.8},\"sample\":[{d:.8},{d:.8},{d:.8},{d:.8},{d:.8},{d:.8},{d:.8},{d:.8}],\"ms\":{d:.3},\"measured\":\"gpu\",\"stored\":\"summary\"}}",
            .{
                info.ctx_len, hidden, s.mean, s.std, s.l2norm,
                s.sample[0], s.sample[1], s.sample[2], s.sample[3], s.sample[4], s.sample[5], s.sample[6], s.sample[7],
                e.ms,
            },
        );
        n += 1;
        sum_ms += e.ms;
    }
    // ... and, when TF_FLASHNEXT_STAGE_CTX_TENSOR=1, the same rows as a tensor
    if (k.find("fc_ctx_rows") != null) try writeOne(gpa, io, w, k, "fc_ctx_rows", tensors_dir, &path_buf, &n, &sum_ms);

    const per_layer = [_][]const u8{ "conv_attn_in_{d}", "qkv_{d}", "attn_{d}", "attn_proj_{d}", "conv_attn_out_{d}", "conv_mlp_in_{d}", "mlp_{d}", "conv_mlp_out_{d}", "block_hidden_{d}" };
    for (0..5) |i| {
        inline for (per_layer) |fmt| {
            const name = try std.fmt.bufPrint(&name_buf, fmt, .{i});
            try writeOne(gpa, io, w, k, name, tensors_dir, &path_buf, &n, &sum_ms);
        }
    }
    try writeOne(gpa, io, w, k, "final_norm_block", tensors_dir, &path_buf, &n, &sum_ms);
    try writeOne(gpa, io, w, k, "logits_top16_ids", tensors_dir, &path_buf, &n, &sum_ms);
    try writeOne(gpa, io, w, k, "logits_top16_vals", tensors_dir, &path_buf, &n, &sum_ms);
    try writeOne(gpa, io, w, k, "logits_full", tensors_dir, &path_buf, &n, &sum_ms);
    try writeOne(gpa, io, w, k, "selector_scores", tensors_dir, &path_buf, &n, &sum_ms);
    try writeOne(gpa, io, w, k, "proposals", tensors_dir, &path_buf, &n, &sum_ms);

    try w.print("],\"sum_stage_ms\":{d:.3}}}\n", .{sum_ms});
    const path = try std.fmt.bufPrint(&path_buf, "{s}/stages.json", .{out_dir});
    try cwd.writeFile(io, .{ .sub_path = path, .data = text.written() });
    std.log.info("draftfix: {d} stages, {d:.2} ms summed, written to {s}/stages.json", .{ n, sum_ms, out_dir });
}

/// The stages a dump has: the context, the block's embedding, nine a layer, the final norm, the
/// head's three tables, the selector's scores, the context's own rows and the proposals.
pub fn countStages() usize {
    return 1 + 1 + 5 * 9 + 1 + 3 + 2 + 1;
}

fn writeOne(
    gpa: std.mem.Allocator,
    io: std.Io,
    w: *std.Io.Writer,
    k: *stage_dump.Collector,
    name: []const u8,
    tensors_dir: []const u8,
    path_buf: *[1200]u8,
    n: *usize,
    sum_ms: *f64,
) !void {
    const e = k.find(name) orelse return error.StageMissing;
    if (n.* != 0) try w.writeAll(",");
    const rows = e.rows;
    const cols = e.cols;
    const flat = cols == 1 or cols == 0;
    const total = rows * cols;
    const st = stage_dump.stats(e.host, rows, cols, e.dtype);
    const f32s = try stage_dump.widenAll(gpa, e.host, total, e.dtype);
    defer gpa.free(f32s);
    const path = try std.fmt.bufPrint(path_buf, "{s}/{s}.npy", .{ tensors_dir, name });
    try stage_dump.writeNpy(io, path, rows, if (flat) 0 else cols, f32s);
    var shape_buf: [48]u8 = undefined;
    const shape = if (flat)
        try std.fmt.bufPrint(&shape_buf, "[{d}]", .{rows})
    else
        try std.fmt.bufPrint(&shape_buf, "[{d}, {d}]", .{ rows, cols });
    try w.print(
        "{{\"stage\":\"{s}\",\"shape\":{s},\"dtype\":\"f32\",\"src_dtype\":\"{s}\",\"mean\":{d:.8},\"std\":{d:.8}," ++
            "\"l2norm\":{d:.8},\"sample\":[{d:.8},{d:.8},{d:.8},{d:.8},{d:.8},{d:.8},{d:.8},{d:.8}],\"ms\":{d:.3},\"measured\":\"{s}\",\"file\":\"tensors/{s}.npy\"}}",
        .{
            name, shape, e.dtype.text(), st.mean, st.std, st.l2norm,
            st.sample[0], st.sample[1], st.sample[2], st.sample[3], st.sample[4], st.sample[5], st.sample[6], st.sample[7],
            e.ms,
            if (e.measured_gpu) @as([]const u8, "gpu") else @as([]const u8, "host"),
            name,
        },
    );
    n.* += 1;
    sum_ms.* += e.ms;
}

/// The served checkpoint's directory, for the record: TF_ALIGN_CHECKPOINT, else the engine's own
/// first argument (the `serve <model>` the round ran under). Copied into `buf`, because the argv it
/// came from is freed before the JSON is written.
fn checkpointPath(io: std.Io, gpa: std.mem.Allocator, buf: []u8) ![]const u8 {
    if (std.c.getenv("TF_ALIGN_CHECKPOINT")) |v| {
        const text = std.mem.span(v);
        const n = @min(text.len, buf.len);
        @memcpy(buf[0..n], text[0..n]);
        return buf[0..n];
    }
    const arg = commandArg(io, gpa, 2) catch return error.NoCmdlineArg;
    defer gpa.free(arg);
    const n = @min(arg.len, buf.len);
    @memcpy(buf[0..n], arg[0..n]);
    return buf[0..n];
}

/// One space-joined line of this process's argv: the exact command the round ran under.
fn commandText(io: std.Io, gpa: std.mem.Allocator, buf: []u8) ![]const u8 {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, "/proc/self/cmdline", gpa, .limited(1 << 16));
    defer gpa.free(raw);
    var out: usize = 0;
    for (raw) |ch| {
        if (out == buf.len) break;
        buf[out] = if (ch == 0) ' ' else ch;
        out += 1;
    }
    while (out > 0 and buf[out - 1] == ' ') out -= 1;
    // procfs reports a size of 0, so a size-driven read can come back empty; TF_ALIGN_COMMAND is
    // then the only way the exact invocation reaches the JSON
    if (out == 0) {
        if (std.c.getenv("TF_ALIGN_COMMAND")) |v| {
            const text = std.mem.span(v);
            const n = @min(text.len, buf.len);
            @memcpy(buf[0..n], text[0..n]);
            out = n;
        }
    }
    return buf[0..out];
}

fn commandArg(io: std.Io, gpa: std.mem.Allocator, index: usize) ![]const u8 {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, "/proc/self/cmdline", gpa, .limited(1 << 16));
    defer gpa.free(raw);
    var it = std.mem.splitScalar(u8, raw, 0);
    var i: usize = 0;
    while (it.next()) |part| : (i += 1) {
        // the caller owns the copy: `raw` is freed when this returns
        if (i == index) return try gpa.dupe(u8, part);
    }
    return error.NoCmdlineArg;
}

// -----------------------------------------------------------------------------------------------
// The debug sink: one round's internals, onto disk
// -----------------------------------------------------------------------------------------------

/// A traced round's own numbers, as the check's loop knows them.
const RoundMeta = struct {
    round: usize,
    p: usize,
    ctx_len: usize,
    ctx_lo: usize,
    anchor: u32,
};

/// The debug sink's two device spans and the host staging a dumped round is read back through.
///
/// Both spans are the round's own bytes, so the host side is the only cost: one prefix of fused rows
/// (`ctx_rows * hidden * 2`, 10.5 MB at the served window) and one block a layer either side
/// (`(layers + 1) * block * hidden * 2`, 235 KiB).
const Sink = struct {
    debug: fwd.Debug,
    ctx: cuda.DeviceBuffer,
    layers: cuda.DeviceBuffer,
    host_ctx: []u8,
    host_layers: []u8,

    fn alloc(gpa: std.mem.Allocator, driver: *cuda.Driver, c: fwd.Config, ctx_rows: usize) !Sink {
        const ctx_bytes = try std.math.mul(usize, ctx_rows, try std.math.mul(usize, c.hidden, 2));
        const block_bytes = try std.math.mul(usize, c.block, c.hidden * 2);
        const layer_bytes = try std.math.mul(usize, c.layers + 1, block_bytes);
        var ctx = try cuda.DeviceBuffer.alloc(driver, ctx_bytes);
        errdefer ctx.free();
        var layers = try cuda.DeviceBuffer.alloc(driver, layer_bytes);
        errdefer layers.free();
        const host_ctx = try gpa.alloc(u8, ctx_bytes);
        errdefer gpa.free(host_ctx);
        const host_layers = try gpa.alloc(u8, layer_bytes);
        return .{
            .debug = .{ .ctx = ctx.ptr, .ctx_rows = ctx_rows, .layers = layers.ptr },
            .ctx = ctx,
            .layers = layers,
            .host_ctx = host_ctx,
            .host_layers = host_layers,
        };
    }

    fn deinit(k: *Sink, gpa: std.mem.Allocator) void {
        k.ctx.free();
        k.layers.free();
        gpa.free(k.host_ctx);
        gpa.free(k.host_layers);
    }
};

/// Write one traced round out (`cuda_draft2_debug.zig`'s layout): the sink's two spans, and the three
/// tables a round leaves in its own scratch — the normalized rows the head read, its top-k, and the
/// selector's walk. Nothing is recomputed here; every byte is the round's.
fn dumpRound(
    gpa: std.mem.Allocator,
    io: std.Io,
    spec: debug.Spec,
    k: *Sink,
    d: *fwd.Draft,
    st: *fwd.State,
    meta: RoundMeta,
    cap: Capture,
) !void {
    const c = d.c;
    const B = c.block;
    const R = B - 1;
    const H = c.hidden;
    const topk = c.topk;
    var b: fwd.Bufs = undefined;
    _ = try fwd.place(c, st.scratch.ptr, &b);

    // the scratch tables, by offset into the block's own buffer
    const hn = try gpa.alloc(u8, R * H * 2);
    defer gpa.free(hn);
    const cand = try gpa.alloc(u8, R * topk * 4);
    defer gpa.free(cand);
    const unary = try gpa.alloc(u8, R * topk * 4);
    defer gpa.free(unary);
    const prop = try gpa.alloc(u8, R * 4);
    defer gpa.free(prop);
    try st.scratch.download(@intCast(b.hn - st.scratch.ptr), hn);
    try st.scratch.download(@intCast(b.cand - st.scratch.ptr), cand);
    try st.scratch.download(@intCast(b.unary - st.scratch.ptr), unary);
    try st.scratch.download(@intCast(b.prop - st.scratch.ptr), prop);

    // the two traced spans: the prefix's fusion and the block's rows before and after each layer
    const ctx_bytes = meta.ctx_len * H * 2;
    if (ctx_bytes > k.host_ctx.len) return error.DebugCtxTooSmall;
    try k.ctx.download(0, k.host_ctx[0..ctx_bytes]);
    const block_bytes = B * H * 2;
    try k.layers.download(0, k.host_layers[0 .. (c.layers + 1) * block_bytes]);

    var items: [max_items]debug.Item = undefined;
    var n: usize = 0;
    items[n] = .{ .name = "ctx_fused", .file = "ctx_fused.bin", .rows = meta.ctx_len, .cols = H, .dtype = "bf16", .bytes = k.host_ctx[0..ctx_bytes] };
    n += 1;
    items[n] = .{ .name = "embed", .file = "embed.bin", .rows = B, .cols = H, .dtype = "bf16", .bytes = k.host_layers[0..block_bytes] };
    n += 1;
    var name_buf: [max_layers][16]u8 = undefined;
    var file_buf: [max_layers][16]u8 = undefined;
    for (0..c.layers) |i| {
        const name = try std.fmt.bufPrint(&name_buf[i], "layer{d}", .{i});
        const file = try std.fmt.bufPrint(&file_buf[i], "layer{d}.bin", .{i});
        const at = (i + 1) * block_bytes;
        items[n] = .{ .name = name, .file = file, .rows = B, .cols = H, .dtype = "bf16", .bytes = k.host_layers[at..][0..block_bytes] };
        n += 1;
    }
    items[n] = .{ .name = "hidden_normed", .file = "hidden_normed.bin", .rows = R, .cols = H, .dtype = "bf16", .bytes = hn };
    n += 1;
    items[n] = .{ .name = "head_ids", .file = "head_ids.bin", .rows = R, .cols = topk, .dtype = "u32", .bytes = cand };
    n += 1;
    items[n] = .{ .name = "head_unary", .file = "head_unary.bin", .rows = R, .cols = topk, .dtype = "f32", .bytes = unary };
    n += 1;
    items[n] = .{ .name = "proposals", .file = "proposals.bin", .rows = R, .cols = 1, .dtype = "u32", .bytes = prop };
    n += 1;

    try debug.writeRound(gpa, io, spec, .{
        .round = meta.round,
        .p = meta.p,
        .ctx_len = meta.ctx_len,
        .ctx_lo = meta.ctx_lo,
        .position = meta.p,
        .anchor = meta.anchor,
        .block = B,
        .hidden = H,
        .layers = c.layers,
        .topk = topk,
        .vocab = c.vocab,
        .capture = cap.dir,
        .ids_len = cap.ids.len,
    }, items[0..n]);
    std.log.info("draft debug: round {d} (p {d}, ctx {d} from {d}) written to {s}/round{d}", .{
        meta.round, meta.p, meta.ctx_len, meta.ctx_lo, spec.dir, meta.round,
    });
}

/// Tensors a round's dump holds at most: the context, the input rows, one a layer, then the head's
/// two tables, the normalized rows and the proposals.
const max_items = 2 + max_layers + 4;
/// Layers a dump can name: `cuda_draft2.zig` refuses a drafter with more than eight.
const max_layers = 8;

/// The longest prefix of `proposed` that is `posterior`'s own tokens (`reference.py`'s
/// `acceptance`): the length of the match, not a count of matches anywhere in the block.
pub fn acceptedOf(proposed: []const u32, posterior: []const u32) usize {
    var n: usize = 0;
    while (n < proposed.len and n < posterior.len) : (n += 1) {
        if (proposed[n] != posterior[n]) break;
    }
    return n;
}

/// The target geometry the drafter's config is validated against.
///
/// `cuda_draft2.parse` reads exactly three fields of it — `hidden` and `vocab` (the drafter's head
/// IS the target's) and `layers` (a tap past the last layer is refused there) — and this check reads
/// none of the rest. They are still set rather than left undefined, at the values
/// `cuda_draft2_fwd.zig`'s own test target uses, so a future reader that grows a use for one of them
/// does not silently read a garbage stack slot.
pub fn targetConfig() config.Config {
    var c: config.Config = undefined;
    c.hidden = hidden;
    c.vocab = vocab;
    c.layers = weights.layers_n;
    c.heads = 24;
    c.kv_heads = 2;
    c.value_heads = 2;
    c.eps = 1e-6;
    return c;
}

// -----------------------------------------------------------------------------------------------
// The head, out of the engine's bf16 tiles
// -----------------------------------------------------------------------------------------------

/// Rows a block of the packing covers (`qmmf.packBf16`'s row tile: 8 n-tiles of 8 rows).
pub const tile_rows = 64;

/// One 64-row block of packed bf16 tiles, written row-major into `dst` (`rows` rows of `k`).
///
/// This is `qmmf.packBf16`'s map run the other way, in its own enumeration order — `colblk` is its
/// `tg % (k / 64)`, then its `jj`, `kt`, `lane` and `e` — so the walk over `src` is sequential and
/// the writes land inside one row's 64-column block. `rows` may be short of `tile_rows` at the end of
/// a head: the padding rows the packing zeroes are dropped, never written.
pub fn detileRows(src: []const u16, dst: []u16, rows: usize, k: usize) void {
    std.debug.assert(k % tile_rows == 0);
    std.debug.assert(src.len == tile_rows * k and dst.len == rows * k and rows <= tile_rows);
    var at: usize = 0;
    for (0..k / tile_rows) |colblk| {
        for (0..8) |jj| {
            for (0..4) |kt| {
                for (0..32) |lane| {
                    const row = jj * 8 + lane / 4;
                    const base = row * k + colblk * tile_rows;
                    const pair = 2 * (lane % 4);
                    for (0..4) |e| {
                        const col = kt * 16 + pair + (e & 1) + 8 * (e >> 1);
                        const v = src[at];
                        at += 1;
                        if (row < rows) dst[base + col] = v;
                    }
                }
            }
        }
    }
}

/// The engine's head as one row-major `[n, k]` bf16 device buffer, a 64-row block at a time: 320 KiB
/// a block at this checkpoint, so the host side of it is two small buffers rather than 1.2 GiB twice.
pub fn detile(gpa: std.mem.Allocator, io: std.Io, driver: *const cuda.Driver, tiles: u64, n: usize, k: usize) !cuda.DeviceBuffer {
    if (tiles == 0 or n == 0 or k == 0 or k % tile_rows != 0) return error.DraftCheckHeadShape;
    var out = try cuda.DeviceBuffer.alloc(driver, try std.math.mul(usize, try std.math.mul(usize, n, k), 2));
    errdefer out.free();
    const src = try gpa.alloc(u16, tile_rows * k);
    defer gpa.free(src);
    const dst = try gpa.alloc(u16, tile_rows * k);
    defer gpa.free(dst);
    var row0: usize = 0;
    while (row0 < n) : (row0 += tile_rows) {
        const rows = @min(tile_rows, n - row0);
        // `packBf16` writes row blocks in ascending order, `k / 64` tiles of 4096 bf16 each: the
        // block at row `row0` starts `(row0 / 64) * (k / 64) * 4096` elements in, not `row0` of them.
        // Blocked here so the device side is one download and one upload a block, in order — the
        // copies are synchronous, so nothing below can see a half-written head.
        const at_row_block = row0 / tile_rows;
        try cuda.memory.download(driver, tiles + at_row_block * (k / tile_rows) * 4096 * 2, std.mem.sliceAsBytes(src));
        detileRows(src, dst[0 .. rows * k], rows, k);
        // TF_FLASHNEXT_DETILE_DUMP=<dir>: write each detiled 64-row block (bf16 bits) so it can be
        // checked against the checkpoint's own `lm_head.weight` outside the engine. Off unless set.
        if (std.c.getenv("TF_FLASHNEXT_DETILE_DUMP")) |dump| {
            var name: [512]u8 = undefined;
            const path = try std.fmt.bufPrint(&name, "{s}/block{d}.bf16", .{ std.mem.span(dump), at_row_block });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(dst[0 .. rows * k]) });
        }
        try out.upload(row0 * k * 2, std.mem.sliceAsBytes(dst[0 .. rows * k]));
    }
    return out;
}

// -----------------------------------------------------------------------------------------------
// Tests: the packing inverse and the acceptance rule. Nothing here touches a device.
// -----------------------------------------------------------------------------------------------

test "a detiled block is the weight the engine packed" {
    const a = std.testing.allocator;
    // k = 128 is two 64-column blocks; n = 70 runs past the packing's 64-row tile, so the second
    // block is short and its padding rows must not be written.
    const n: usize = 70;
    const k: usize = 128;
    const npad = (n + 127) / 128 * 128;
    const w = try a.alloc(u16, n * k);
    defer a.free(w);
    for (w, 0..) |*x, i| x.* = @truncate(i * 7 + 3);
    const tiled = try a.alloc(u16, npad * k);
    defer a.free(tiled);
    cuda.qmmf.packBf16(tiled, w, n, k);

    const back = try a.alloc(u16, n * k);
    defer a.free(back);
    @memset(back, 0xdead);
    const per_block = tile_rows * k;
    var row0: usize = 0;
    while (row0 < n) : (row0 += tile_rows) {
        const rows = @min(tile_rows, n - row0);
        // row block `row0` starts at `(row0 / 64) * (k / 64)` tiles of 4096 bf16 (`packBf16`'s `tg`)
        const at_row_block = row0 / tile_rows;
        detileRows(tiled[at_row_block * (k / tile_rows) * 4096 ..][0..per_block], back[row0 * k ..][0 .. rows * k], rows, k);
    }
    try std.testing.expectEqualSlices(u16, w, back);
}

test "acceptance is the block's matching prefix, not its matching tokens" {
    // one match then a miss, a match after the miss, and a full block: only the leading run counts
    try std.testing.expectEqual(@as(usize, 2), acceptedOf(&.{ 5, 6, 9, 1 }, &.{ 5, 6, 7, 1 }));
    try std.testing.expectEqual(@as(usize, 0), acceptedOf(&.{ 9, 6, 7 }, &.{ 5, 6, 7 }));
    try std.testing.expectEqual(@as(usize, 7), acceptedOf(&.{ 1, 2, 3, 4, 5, 6, 7 }, &.{ 1, 2, 3, 4, 5, 6, 7 }));
    // a propose list and a target list are the same length in a round; a short one is not overrun
    try std.testing.expectEqual(@as(usize, 1), acceptedOf(&.{ 1, 2 }, &.{1}));
}

test "the check's geometry is the drafter's, and its defaults are the reference run's" {
    try std.testing.expectEqual(@as(usize, 5), n_taps);
    try std.testing.expectEqual(@as(usize, 2560), hidden);
    try std.testing.expectEqual(@as(usize, 248_320), vocab);
    try std.testing.expectEqual(@as(usize, 8), block);
    try std.testing.expectEqual(@as(usize, 120), default_rounds);
    // a round reads `block - 1` proposals against `block - 1` targets
    try std.testing.expectEqual(@as(usize, 7), block - 1);
}
