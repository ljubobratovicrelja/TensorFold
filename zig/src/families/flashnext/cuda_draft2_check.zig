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
const draft2 = @import("cuda_draft2.zig");
const fwd = @import("cuda_draft2_fwd.zig");
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
    // refused before the head is detiled (a second of host work): a directory that is not a DFlash2
    // drafter is not one, whatever the rest of the environment says
    if (!draft2.declared(gpa, io, dir)) return error.DraftCheckNotADrafter;
    if (m.embed == 0 or m.embed_rows != vocab) return error.DraftCheckNoEmbedTable;
    if (m.head == 0) return error.DraftCheckNoHead;
    if (!m.head_bf16) return error.DraftCheckHeadNotBf16;
    if (m.head_rows != vocab or m.head_cols != hidden) return error.DraftCheckHeadNotWholeVocab;
    const row_bytes = n_taps * hidden * 2;
    if (cap.rows < block or cap.ring.len != try std.math.mul(usize, cap.rows, row_bytes)) return error.DraftCheckNoRing;
    if (cap.ids.len < block + 2) return error.DraftCheckSequenceTooShort;
    if (max_rounds == 0) return error.DraftCheckNoRounds;

    // 1. the head the drafter's top-k reads, out of the tiles the engine keeps it in
    var head = try detile(gpa, driver, m.head, vocab, hidden);
    defer head.free();

    // 2. the drafter itself: the served checkpoint, its 81 tensors, its five layers' rings
    const d = try fwd.Draft.open(gpa, io, driver, stream, dir, targetConfig(), .{ .embed = m.embed, .head = head.ptr });
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
pub fn detile(gpa: std.mem.Allocator, driver: *const cuda.Driver, tiles: u64, n: usize, k: usize) !cuda.DeviceBuffer {
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
