//! Shared verify dispatch with a recording owner: TP=2 groups go to the ranks, a lone window verifies alone.
const std = @import("std");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const verify = @import("cuda_lanes_verify.zig");
const be = lanes.backend;
const t = std.testing;

/// Each call a pass made: 0 a lone verify, else a shared pass of that many windows (100 + n on the ranks).
const Log = struct {
    calls: [16]usize = undefined,
    n: usize = 0,
    draw_groups: usize = 0,
    fn add(l: *Log, v: usize) void {
        l.calls[l.n] = v;
        l.n += 1;
    }
};

/// A note the drafter was handed: the rows a pass says its stream's ring holds.
const Note = struct { stream: *lanes.Stream, from: u64, to: u64 };

const Owner = struct {
    const Lane = struct { seq: prompt.Seq = undefined, draft: ?mtp.Draft = @as(mtp.Draft, undefined) }; // a set optional: windows() unwraps it
    const Ranks = struct {
        log: *Log,
        pub fn verifyMany(r: *Ranks, comptime Tri: type, _: Tri, seqs: []const *prompt.Seq, ds: []const *mtp.Draft, idss: []const []const u32, _: u64) !void {
            try t.expect(seqs.len == ds.len and seqs.len == idss.len);
            r.log.add(100 + seqs.len);
        }
    };
    const Ops = struct {
        const S = struct {
            pub fn synchronize(_: *@This()) !void {}
        };
        log: *Log,
        e: struct { stream: *S = undefined } = .{},
        /// A lone window: one pass over the stream's own rows, which is what moves the stream's fed.
        pub fn verify(o: Ops, seq: *prompt.Seq, _: *mtp.Draft, ids: []const u32) !void {
            o.log.add(0);
            seq.fed += ids.len;
        }
        pub fn verifyMany(o: Ops, seqs: []const *prompt.Seq, _: []const *mtp.Draft, _: []const []const u32, _: u64) !void {
            o.log.add(seqs.len);
        }
        pub fn drawsMany(o: Ops, _: u64, _: []const *mtp.Draft, _: []const ?mtp.Sampling, _: []const []const u32, _: []const []u32) !void {
            o.log.draw_groups += 1;
        }
        pub fn draws(_: Ops, _: *mtp.Draft, _: anytype, _: []const u32, _: []u32) !void {}
    };
    /// The drafter `windows()` notes through, recording each span it is handed.
    const Drafter = struct {
        notes: *[8]Note,
        n: *usize,
        pub fn note(d: Drafter, s: *lanes.Stream, from: u64, to: u64) void {
            d.notes[d.n.*] = .{ .stream = s, .from = from, .to = to };
            d.n.* += 1;
        }
    };
    log: Log = .{},
    drafter: ?Drafter = null,
    lanes_: [4]Lane = .{ .{}, .{}, .{}, .{} },
    streams: [4]lanes.Stream = undefined,
    ranks: ?*Ranks = null,
    logits: struct { ptr: u64 = 0x1000 } = .{},

    pub fn lane(o: *Owner, s: *lanes.Stream) !*Lane {
        for (&o.streams, 0..) |*x, i| if (x == s) return &o.lanes_[i];
        return error.UnknownStream;
    }
    pub fn tri(_: *Owner) u8 {
        return 0;
    }
    pub fn ops(o: *Owner) Ops {
        return .{ .log = &o.log };
    }
    pub fn opsTapped(o: *Owner, s: *lanes.Stream) Ops {
        _ = s; // the fixture's windows hold no tap ring: the lone path records the same
        return o.ops();
    }
};

fn run(o: *Owner, rows: []const usize, cap: usize, fed: usize) !void {
    var wins: [4]be.Window = undefined;
    var out: [4]be.Verified = undefined;
    var sampled: [4][mtp.max_rows]u32 = undefined;
    var drafted: [4]usize = undefined;
    var row_ids: [4][mtp.max_rows]u32 = undefined;
    for (rows, 0..) |r, i| {
        o.streams[i].sampling = null;
        // the stream stands at `fed`: a window's rows are the ones its pass writes from there
        o.lanes_[i].seq.fed = fed;
        wins[i] = .{ .stream = &o.streams[i], .pending = 0, .held = @intCast(r - 1), .tokens = &.{}, .parents = null, .positions = &.{} };
        out[i] = .{ .sampled = &sampled[i], .drafts = &.{} };
        drafted[i] = i;
        row_ids[i] = @splat(1);
    }
    o.log = .{};
    try verify.windows(o, wins[0..rows.len], out[0..rows.len], drafted[0..rows.len], row_ids[0..rows.len], cap);
}

test "shared verify dispatch: at TP=2 a group goes to the ranks, a lone window and a split group verify alone" {
    var log: Log = .{};
    var ranks: Owner.Ranks = .{ .log = &log };
    var o: Owner = .{};
    o.ranks = &ranks;
    try run(&o, &.{ 2, 3, 2 }, 64, 0);
    try t.expectEqual(@as(usize, 1), log.n);
    try t.expectEqual(@as(usize, 103), log.calls[0]);
    try t.expectEqual(@as(usize, 0), o.log.n);
    try t.expectEqual(@as(usize, 0), o.log.draw_groups);
    log = .{};
    try run(&o, &.{4}, 64, 0);
    try t.expectEqual(@as(usize, 0), log.n);
    try t.expectEqualSlices(usize, &.{0}, o.log.calls[0..o.log.n]);
    // a cap of 4 rows splits two 3-row windows into two lone verifies
    try run(&o, &.{ 3, 3 }, 4, 0);
    try t.expectEqual(@as(usize, 0), log.n);
    try t.expectEqualSlices(usize, &.{ 0, 0 }, o.log.calls[0..o.log.n]);
    // one rank: the shared pass stays on this engine
    o.ranks = null;
    try run(&o, &.{ 2, 3, 2 }, 64, 0);
    try t.expectEqualSlices(usize, &.{3}, o.log.calls[0..o.log.n]);
    try t.expectEqual(@as(usize, 1), o.log.draw_groups);
}

test "a drafted window notes the rows its pass wrote, not the window after it" {
    var notes: [8]Note = undefined;
    var note_n: usize = 0;
    var o: Owner = .{};
    o.drafter = .{ .notes = &notes, .n = &note_n };
    // A lone 4-row window at fed 2048 writes ring slots [2048, 2052): the window's own rows, which
    // the position check names `fed + 1 + r` (one-based) and the pass writes zero-based from the fed
    // it starts at. Reading the fed after the pass had moved it would note [2052, 2056) — the next
    // window's slots, which this pass never wrote, and the inflated `written` that hides a tap gap.
    try run(&o, &.{4}, 1, 2048);
    try t.expectEqual(@as(usize, 1), note_n);
    try t.expectEqual(@as(u64, 2048), notes[0].from);
    try t.expectEqual(@as(u64, 2052), notes[0].to);
    try t.expectEqual(@as(usize, 2052), o.lanes_[0].seq.fed); // the pass moved the fed itself
    // the next window starts exactly where this one ended: the two spans tile the ring, no hole
    try run(&o, &.{1}, 1, 2052);
    try t.expectEqual(@as(u64, 2052), notes[1].from);
    try t.expectEqual(@as(u64, 2053), notes[1].to);
    // a shared pass writes no tap ring, so it notes nothing at all
    note_n = 0;
    try run(&o, &.{ 4, 4 }, 64, 0);
    try t.expectEqual(@as(usize, 0), note_n);
}
