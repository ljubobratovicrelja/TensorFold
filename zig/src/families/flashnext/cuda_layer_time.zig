//! TF_FLASHNEXT_LAYER_TIME=1 / TF_FLASHNEXT_LAYER_DUMP=<dir>: the target's 48-layer pass, stage by stage.
//!
//! A served decode round is dominated by the verify pass (one `Pass.chunk`: 48 layers and a head) and
//! that pass is opaque. This records a CUDA timing event at every stage boundary inside the loop with
//! `cuEventRecord` — no synchronise, so the launches queue exactly as they would — reads the whole set
//! back after ONE `cuStreamSynchronize` at the pass's end, prints per-layer milliseconds, and, when a
//! dump directory is named, writes `<dir>/rep<N>/stages.json` in the drafter's own schema so
//! `tools/dflash2/harness/plot_stages.py` reads it unchanged.
//!
//! Unset, it costs nothing: no events are made, nothing is allocated, no stream is synchronised, and
//! every boundary site in the pass is a null check on `Pass.clock` (see `cuda_batch.zig`).
//!
//! A capture cannot hold this. An event record inside a capture becomes a graph node that replays a
//! stale timestamp, and `cuStreamSynchronize` inside one is illegal. `Clock.begin` therefore refuses
//! when the pass's stream is already capturing, and `Engine.forward` refuses before a window capture
//! begins — a refusal, never a silently disabled graph. Profile with TF_FLASHNEXT_WINDOW_GRAPH=0.
//!
//! The pool is made once per Engine (`Engine.layer_clock`, `cuda_prompt.zig`), on the engine's own
//! context, and is single-pass: a second pass in flight on the same pool is refused
//! (`error.LayerTimeBusy`) rather than interleaving its records into the first pass's events. A caller
//! that must keep two passes in flight gives each pass its own `Pass.clock`; the engine's is then simply
//! left uninstalled for one of them.
//!
//! The JSON's `round_ms` is the sum of the stages, which are contiguous boundaries from the embed's
//! first launch to the head's last: it is that GPU span, not a wall clock of the whole `chunk` (the
//! uploads and the host-side n-gram gather happen before the first event and are outside it).
const std = @import("std");
const cuda = @import("cuda");
const sys = @import("sys");

/// Boundaries a layer records after its first: the start of each of its eight stages. A layer's last
/// (block_hidden's end) is the next layer's first, so 8 boundaries a layer, not 9.
pub const per_layer = 8;

/// The widest decode window, `mtp.max_rows` (cuda_mtp.zig): a pass at or below it is a decode round.
/// Kept here rather than imported, because cuda_mtp.zig imports cuda_batch.zig, the pass this rides in.
pub const decode_rows: usize = 32;

/// The longest name written: `conv_attn_in_47`.
pub const name_max = 24;

const time_env: [:0]const u8 = "TF_FLASHNEXT_LAYER_TIME";
const dump_env: [:0]const u8 = "TF_FLASHNEXT_LAYER_DUMP";

/// The stages of one layer, in the order it runs them; the name a record carries is this plus `_{d}`.
const stage_names = [per_layer][]const u8{
    "conv_attn_in", // the input mix and read-out, the n-gram block, the tap ring's copy
    "qkv", //           the layers' projections (attention's q|k|v|indexer, DeltaNet's q|k|v|z|b|a)
    "attn", //          attention rows, or the DeltaNet conv window, recurrence and gated norm
    "attn_proj", //     the attention gate and out projection (DeltaNet's norm and out projection)
    "conv_mlp_in", //   the MoE's input mix and read-out
    "mlp", //           the router, the ten experts a row and the shared one
    "moe_out", //       the experts' slots summed and written back into the streams
    "block_hidden", //  the tapped row's copy out, the handover to the next layer
};

/// The bounds recorded inside a layer's own functions, each naming the stage that starts there.
/// `conv_attn_in`'s start is `Clock.layer` (the loop's own boundary), so it has no member here.
pub const Boundary = enum(u3) {
    qkv = 1,
    attn = 2,
    attn_proj = 3,
    conv_mlp = 4,
    mlp = 5,
    moe_out = 6,
    hidden = 7,
};

/// Whether TF_FLASHNEXT_LAYER_TIME asks for the console report (0 or 1; anything else refused at boot).
pub fn enabled() bool {
    return sys.environment.read(bool, time_env, false) catch false;
}

/// The directory TF_FLASHNEXT_LAYER_DUMP names, or null: unset or blank is off, never an error. Read the
/// way TF_FLASHNEXT_TAP_DUMP is (cuda_tap_dump.zig): a research path boot does not refuse, so it has no
/// row in the strict table either.
pub fn dirFromEnv() ?[]const u8 {
    const v = sys.getenv(dump_env) orelse return null;
    const text = std.mem.trim(u8, v, " \t");
    return if (text.len == 0) null else text;
}

/// Either flag: the dump alone is a reason to record.
pub fn wanted() bool {
    return enabled() or dirFromEnv() != null;
}

/// One engine's event pool and the pass currently recording into it.
pub const Clock = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    /// The layers this pool was built for: a pass of another count is not profiled (and says so).
    layers: usize,
    /// `per_layer * layers + 4`: embed's start, the layer boundaries, the final block, the head, the end.
    ev: []cuda.Event,
    stream: cuda.Stream = undefined,
    /// A pass is recording: every boundary site is a no-op while this is false.
    busy: bool = false,
    /// The layer the nested boundary calls belong to (the loop's own last `layer` call).
    cur: usize = 0,
    /// Boundaries recorded in this pass, so a pass that skipped one reports nothing rather than a shift.
    next: usize = 0,
    rows: usize = 0,
    chunk_rows: usize = 0,
    pos0: usize = 0,
    /// Dumps written: the `<dir>/rep<N>` the next one takes.
    reps: usize = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, layers: usize) !Clock {
        if (layers == 0) return error.LayerTimeNoLayers;
        const ev = try gpa.alloc(cuda.Event, per_layer * layers + 4);
        errdefer gpa.free(ev);
        var made: usize = 0;
        errdefer for (ev[0..made]) |*v| v.deinit();
        while (made < ev.len) : (made += 1) ev[made] = try cuda.Event.init(d, true);
        return .{ .gpa = gpa, .io = io, .d = d, .layers = layers, .ev = ev };
    }

    pub fn deinit(c: *Clock) void {
        for (c.ev) |*v| v.deinit();
        c.gpa.free(c.ev);
        c.* = undefined;
    }

    fn rec(c: *Clock, i: usize) !void {
        if (i >= c.ev.len) return error.LayerTimeIndex;
        try c.ev[i].record(c.stream);
        c.next = i + 1;
    }

    /// Start timing the pass of `rows` on `stream`; the pool stays busy until `finish` or `abandon`.
    /// `layers` is the pass's own count: a pass of another shape (the draft head's one layer) is skipped.
    pub fn begin(c: *Clock, stream: cuda.Stream, rows: usize, chunk_rows: usize, pos0: usize, layers: usize) !void {
        if (c.busy) return error.LayerTimeBusy;
        if (layers != c.layers) {
            std.debug.print("layer_time: a pass of {d} layers, this engine's clock holds {d}: not profiling it\n", .{ layers, c.layers });
            return;
        }
        // An event record inside a capture is a graph node replaying a stale timestamp, and the one
        // synchronise below would be illegal there: refuse loudly rather than record a lie.
        if ((cuda.graph.captureStatus(stream) catch .active) != .none) {
            std.debug.print("layer_time: refused: the pass's stream is inside a CUDA graph capture; set TF_FLASHNEXT_WINDOW_GRAPH=0 (and TF_FLASHNEXT_DECODE_GRAPH=0)\n", .{});
            return error.LayerTimeInCapture;
        }
        c.stream = stream;
        c.rows = rows;
        c.chunk_rows = chunk_rows;
        c.pos0 = pos0;
        c.cur = 0;
        c.next = 0;
        c.busy = true;
    }

    /// The pass's first launch (the embedding, or the media inject ahead of it): the zero of everything after.
    pub fn embed(c: *Clock) !void {
        if (!c.busy) return;
        try c.rec(0);
    }

    /// Layer `l`'s conv_attn_in begins; every nested `stage` call names this layer until the next one.
    pub fn layer(c: *Clock, l: usize) !void {
        if (!c.busy) return;
        if (l >= c.layers) return error.LayerTimeIndex;
        c.cur = l;
        try c.rec(1 + per_layer * l);
    }

    /// A stage of the current layer begins, recorded where that stage's own function starts it.
    pub fn stage(c: *Clock, b: Boundary) !void {
        if (!c.busy) return;
        try c.rec(1 + per_layer * c.cur + @backingInt(b));
    }

    /// The final block (the last layer's write-back, read-out and norm) begins: the end of block_hidden.
    pub fn finalBlock(c: *Clock) !void {
        if (!c.busy) return;
        try c.rec(per_layer * c.layers + 1);
    }

    /// The head's projection begins: the end of `final_norm_block`, inside `Pass.head`.
    pub fn headStart(c: *Clock) !void {
        if (!c.busy) return;
        try c.rec(per_layer * c.layers + 2);
    }

    /// The pass is over: record its end, synchronise ONCE, and report. A pass that skipped a boundary
    /// (an error path, a shape this pool was not built for) reports nothing rather than shifted times.
    pub fn finish(c: *Clock) !void {
        if (!c.busy) return;
        defer c.busy = false;
        try c.rec(per_layer * c.layers + 3);
        try c.stream.synchronize();
        if (c.next != c.ev.len) {
            std.debug.print("layer_time: {d} of {d} boundaries were recorded, so no timings are reported\n", .{ c.next, c.ev.len });
            return;
        }
        const ms = try c.gpa.alloc(f32, c.ev.len - 1);
        defer c.gpa.free(ms);
        for (ms, 0..) |*v, i| v.* = try cuda.Event.elapsedMs(c.ev[i], c.ev[i + 1]);
        c.report(ms) catch |err| std.debug.print("layer_time: the report failed ({s})\n", .{@errorName(err)});
    }

    /// A pass that did not finish (an error out of `chunk`): the pool is free for the next one.
    pub fn abandon(c: *Clock) void {
        c.busy = false;
        c.next = 0;
    }

    /// The stage a stage index `i` of `ms` is: `embed`, `{stage}_{layer}`, `final_norm_block`, `head`.
    fn stageName(i: usize, layers: usize, buf: []u8) ![]const u8 {
        if (i == 0) return "embed";
        const stages = per_layer * layers;
        if (i <= stages) return std.fmt.bufPrint(buf, "{s}_{d}", .{ stage_names[(i - 1) % per_layer], (i - 1) / per_layer });
        return if (i == stages + 1) "final_norm_block" else "head";
    }

    fn report(c: *Clock, ms: []const f32) !void {
        const n_l = c.layers;
        var total: f64 = 0;
        for (ms) |v| total += v;
        const mode: []const u8 = if (c.rows <= decode_rows) "decode" else "prompt";
        std.debug.print("layer_time: {d} rows at {d}, {d} layers, {s} pass: {d:.2} ms of stages\n", .{ c.rows, c.pos0, n_l, mode, total });
        for (0..n_l) |l| {
            const at = 1 + per_layer * l;
            var sum: f64 = 0;
            for (ms[at..][0..per_layer]) |v| sum += v;
            std.debug.print(
                "  L{d:2}  cai {d:6.3}  qkv {d:6.3}  attn {d:6.3}  ap {d:6.3}  cmi {d:6.3}  mlp {d:6.3}  moe {d:6.3}  bh {d:6.3} | {d:7.3} {d:5.1}%\n",
                .{ l, ms[at], ms[at + 1], ms[at + 2], ms[at + 3], ms[at + 4], ms[at + 5], ms[at + 6], ms[at + 7], sum, if (total > 0) 100 * sum / total else 0 },
            );
        }
        std.debug.print("  embed {d:.3}  final_norm_block {d:.3}  head {d:.3}\n", .{ ms[0], ms[per_layer * n_l + 1], ms[per_layer * n_l + 2] });
        try c.dump(ms, total, mode);
    }

    /// `<dir>/rep<N>/stages.json`: the drafter's schema (cuda_draft2_check.zig's `writeStages`), a stage a
    /// record. No bytes are stored — `stored: "mark"` says so — so `shape` is the pass's row count.
    fn dump(c: *Clock, ms: []const f32, total: f64, mode: []const u8) !void {
        const dir = dirFromEnv() orelse return;
        var text: std.Io.Writer.Allocating = .init(c.gpa);
        defer text.deinit();
        const w = &text.writer;
        try w.print(
            "{{\"impl\":\"tensorfold-cuda\",\"device\":\"cuda\",\"dtype\":\"bf16\",\"n_stages\":{d},\"round\":" ++
                "{{\"mode\":\"{s}\",\"rows\":{d},\"layers\":{d},\"chunk\":{d},\"position\":{d},\"round_ms\":{d:.3}}},\"stages\":[",
            .{ ms.len, mode, c.rows, c.layers, c.chunk_rows, c.pos0, total },
        );
        var name_buf: [name_max]u8 = undefined;
        for (ms, 0..) |v, i| {
            const name = try stageName(i, c.layers, &name_buf);
            try w.print(
                "{s}{{\"stage\":\"{s}\",\"shape\":[{d}],\"dtype\":\"bf16\",\"src_dtype\":\"bf16\",\"ms\":{d:.4},\"measured\":\"gpu\",\"stored\":\"mark\"}}",
                .{ if (i == 0) @as([]const u8, "") else ",", name, c.rows, v },
            );
        }
        try w.print("],\"sum_stage_ms\":{d:.3}}}\n", .{total});

        var dbuf: [1024]u8 = undefined;
        const sub = try std.fmt.bufPrint(&dbuf, "{s}/rep{d}", .{ dir, c.reps });
        const cwd = std.Io.Dir.cwd();
        try cwd.createDirPath(c.io, sub);
        var pbuf: [1100]u8 = undefined;
        const path = try std.fmt.bufPrint(&pbuf, "{s}/stages.json", .{sub});
        try cwd.writeFile(c.io, .{ .sub_path = path, .data = text.written() });
        c.reps += 1;
        std.log.info("layer_time: {d} stages, {d:.2} ms summed, written to {s}", .{ ms.len, total, path });
    }
};

test "a clock's pool is the pass's boundaries: eight a layer, three stages around them" {
    // no driver here: the arithmetic is the whole claim, checked against a 48-layer pass
    const L = 48;
    try std.testing.expectEqual(@as(usize, 8 * L + 4), per_layer * L + 4);
    var buf: [name_max]u8 = undefined;
    try std.testing.expectEqualStrings("embed", try Clock.stageName(0, L, &buf));
    try std.testing.expectEqualStrings("conv_attn_in_0", try Clock.stageName(1, L, &buf));
    try std.testing.expectEqualStrings("block_hidden_0", try Clock.stageName(8, L, &buf));
    try std.testing.expectEqualStrings("conv_attn_in_1", try Clock.stageName(9, L, &buf));
    try std.testing.expectEqualStrings("mlp_47", try Clock.stageName(8 * L - 2, L, &buf));
    try std.testing.expectEqualStrings("block_hidden_47", try Clock.stageName(8 * L, L, &buf));
    try std.testing.expectEqualStrings("final_norm_block", try Clock.stageName(8 * L + 1, L, &buf));
    try std.testing.expectEqualStrings("head", try Clock.stageName(8 * L + 2, L, &buf));
}
