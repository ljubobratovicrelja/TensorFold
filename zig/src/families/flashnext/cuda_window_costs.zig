//! The target's verify forward priced at load: one stream's window of 1..window_rows rows, the table
//! the depth rule prices a drafted round by (`lanes.Model.window_costs`).
//!
//! Why it exists. `core/lanes/depth.zig`'s `depth()` answers `max(1, min(most, want))` from the
//! acceptance estimate alone whenever `family_costs` is empty — "at most three offers a round, whatever
//! the drafter's acceptance" — and `family_costs` is built in `core/lanes/config.zig` from the target's
//! `Model.window_costs`, "forward ms by width, timed at load". This family never reported one, so a
//! DFlash2 stream that accepted 100% of what it offered was still pinned to three proposals a round
//! (docs/design/flashnext-dflash2-serving-defects.md §3). The rule is right; the table was missing.
//! The sibling qwen3_5 CUDA backend measures the same sweep at load (`cuda_warm.zig`) and publishes it
//! from its drafter's `facts` (`cuda_draft.zig`).
//!
//! What is timed. `prompt.Engine.forward` over `rows` tokens of one scratch sequence: the exact call
//! `cuda_mtp_ops.zig`'s `verify` issues for a drafted window (`o.e.forward(Tri, o.tri, seq, ids, …)`,
//! every row's scores into one device buffer), less the draft-state settle / rollback copies and the
//! tap hook — kilobytes of copies beside a whole forward. The other candidates do not fit:
//! `Engine.round` is one token per stream and downloads its scores inside the call, so it is not an
//! n-row forward to time; `Ops.verify` itself needs an MTP `Draft` bound to the sequence and rewinds
//! that draft between repeats, which would put bookkeeping inside the bracket.
//!
//! How it is timed. Each width runs `warm_passes` untimed passes (the first pass at a width may grow
//! the engine's scratch and score buffers, which allocate and synchronize), then `reps` timed ones and
//! keeps the fastest — one slow run cannot set the table. Each bracket opens the awake clock before the
//! launch and `synchronize`s the stream before reading it again: a wall clock read next to an async
//! launch measures the enqueue, not the work.
//!
//! What it does not price. A repeat runs its rows from the start of a cold, short sequence, as the
//! sibling's sweep does: these numbers price the forward's work by width, not the attention term a
//! served round pays for its context (which is roughly the same at every width, and is why the depth
//! rule's choice turns on the difference between widths rather than their absolute cost). The tap
//! ring's per-row write and the draft-state copies are left out for the same reason.
const std = @import("std");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const kvl = @import("cuda_kv_lanes.zig");

/// The widest window the sweep prices: a drafted window's verify is `WindowTooWide` past this
/// (`kvl.window_rows`, which is `mtp.max_rows`).
pub const most = kvl.window_rows;

comptime {
    // a drafted window wider than this is refused (`cuda_mtp_ops.zig`'s WindowTooWide), so a table past
    // it would price widths no round can verify
    std.debug.assert(@as(usize, most) == mtp.max_rows);
}

/// Untimed passes each width runs first; two are enough to settle a scratch that only grows.
pub const warm_passes = 2;
/// Timed passes each width runs; the fastest of them stands.
pub const reps = 3;

/// Widths a sweep of `context` tokens' engine prices: 1..window_rows, never more than the context holds.
pub fn widths(context: usize) usize {
    return @min(@as(usize, most), context);
}

/// Times widths `1..=widths(c.context)` and leaves them in `c.costs[0..c.n_costs]`, returning the
/// count. Nothing is stored unless every width measured: a sweep that fails halfway leaves the table
/// empty (the depth rule's unmeasured cap) rather than a table whose gaps price some windows as free.
pub fn measure(c: anytype) !usize {
    const e = c.engine;
    const w = widths(c.context);
    if (w == 0) return error.NoWindowToPrice;
    if (c.ranks != null) {
        // a rank's forward is half of a collective: timing it here alone would wait on the peer's
        // launches, and two ranks sweeping at once is not a served round either. The table stays empty,
        // which leaves the depth rule its unmeasured cap — this line is the diagnosis, not a silent
        // fallback.
        std.log.warn("flashnext: window costs unmeasured on a rank (tp>1); the depth rule keeps its three-offer cap", .{});
        return 0;
    }
    var seq = try e.newSeq(w + 1);
    defer seq.deinit(c.gpa);
    // the rows a verify carries: real token ids, distinct enough that no n-gram exit keeps a pass short
    var ids: [most]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(1 + i % 997);
    const Tri = @TypeOf(c.tri());
    var timed: [most]f64 = @splat(0);
    for (1..w + 1) |rows| {
        for (0..warm_passes) |_| try pass(Tri, c, &seq, ids[0..rows]);
        var quickest: f64 = std.math.inf(f64);
        for (0..reps) |_| quickest = @min(quickest, try bracket(Tri, c, &seq, ids[0..rows]));
        if (!std.math.isFinite(quickest) or quickest <= 0) return error.InvalidWindowMeasurement;
        timed[rows - 1] = quickest;
    }
    for (timed[0..w], 0..) |ms, i| c.costs[i] = lanes.config.Cost{ .width = @intCast(i + 1), .ms = ms };
    c.n_costs = w;
    for (timed[0..w], 1..) |ms, rows| std.log.debug("flashnext: window cost {d} rows {d:.2} ms", .{ rows, ms });
    std.log.info("flashnext: window costs measured, widths 1..{d}: {d:.2} ms at one row, {d:.2} at {d} ({d} warm, {d} timed a width)", .{ w, timed[0], timed[w - 1], w, warm_passes, reps });
    return w;
}

/// One verify-shape pass over `ids` on the scratch sequence, from a fixed position: every repeat runs
/// the same rows at the same place, so no repeat is priced by what the one before it wrote. The hooks
/// are `Ops.verify`'s own (all rows' scores to the engine's round buffer, none downloaded).
fn pass(comptime Tri: type, c: anytype, seq: *prompt.Seq, ids: []const u32) !void {
    seq.fed = 0;
    try c.engine.forward(Tri, c.tri(), seq, ids, .{ .logits = .all, .logits_out = c.logits.ptr, .host_logits = false });
}

/// One timed pass: the clock opens before the launch and closes after the stream has run it, so the
/// bracket holds the work and not the enqueue.
fn bracket(comptime Tri: type, c: anytype, seq: *prompt.Seq, ids: []const u32) !f64 {
    const io = c.engine.io;
    const t0 = std.Io.Clock.awake.now(io);
    try pass(Tri, c, seq, ids);
    try c.engine.stream.synchronize();
    const ns: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
    return ns / 1e6;
}

/// Whether the sweep runs. `TF_FLASHNEXT_WINDOW_COSTS=0` holds a server on the depth rule's unmeasured
/// cap in the same build, so the table's effect can be measured against its absence. Read through
/// `std.c.getenv` like the family's other throwaway instruments (`TF_TAP_TIME`): an unregistered name
/// needs no row in `platform/environment.zig`, and a malformed value means "on".
pub fn wanted() bool {
    const raw = std.c.getenv("TF_FLASHNEXT_WINDOW_COSTS") orelse return true;
    return !(raw[0] == '0' and raw[1] == 0);
}

test "the sweep prices 1..window_rows, and never more widths than the context holds" {
    try std.testing.expectEqual(@as(usize, most), widths(4096));
    try std.testing.expectEqual(@as(usize, 8), widths(8));
    try std.testing.expectEqual(@as(usize, 0), widths(0));
}
