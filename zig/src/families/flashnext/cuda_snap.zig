//! Flash Next prompt states for the shared prompt cache: one stream's caches after `at` tokens, copied on the device.

const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const Draft = @import("cuda_mtp_draft.zig").Draft;
const pages = @import("cuda_pages.zig");
const sizes = @import("cuda_seq_bytes.zig");
const Heads = @import("cuda_heads.zig").Heads;

const index_row = wt.index_dim * 2;
const tail_bytes = wt.ple_tail_n * wt.wide * 2;
/// The MTP head's cache: its own layer, whole on every rank.
const head_heads: Heads = .full;

/// The prompt cache's ladder: tokens between rungs, and the most it keeps under one history mark.
pub const ladder = .{ .tokens = 8192, .most = 4 };

/// A state's shape for its disk image (cuda_disk.zig, one rank, the whole layer): layer counts and bytes a row.
/// Keys and values have different rows at fp8 (`kv8`), so each has its own (`key_row`, `value_row`).
pub const Shape = struct { linear: usize, attn: usize, conv: usize, state: usize, key_row: usize, value_row: usize, index_row: usize, ratio: usize, tail: usize };

/// The image's shape at the cache's precision: fp8 keys keep their two scales a row, values need none (cuda_heads.zig).
pub fn shapeOf(kv8: bool) Shape {
    return .{ .linear = wt.linear_n, .attn = wt.attn_n, .conv = sizes.convBytes(.full), .state = sizes.stateBytes(.full), .key_row = Heads.full.keyRowOf(kv8), .value_row = Heads.full.valueRowOf(kv8), .index_row = index_row, .ratio = wt.index_ratio, .tail = tail_bytes };
}

/// One attention cache's first `at` key, value and indexer-key rows at heads `h`, and the pooled blocks they complete.
fn attnBytes(at: usize, h: Heads, kv8: bool) usize {
    return at * (h.keyRowOf(kv8) + h.valueRowOf(kv8) + index_row) + at / wt.index_ratio * index_row;
}

/// Device bytes a state at the model's heads `h` after `at` tokens takes: fixed DeltaNet pieces, attention by token.
pub fn bytes(at: usize, head: bool, h: Heads, kv8: bool) usize {
    const own = wt.linear_n * (sizes.convBytes(h) + sizes.stateBytes(h)) + wt.attn_n * attnBytes(at, h, kv8) + tail_bytes;
    return own + if (head) attnBytes(at, head_heads, kv8) else 0;
}

/// The DFlash2 drafter's tap ring a state may carry (cuda_drafter2.zig's `State`): the rows a stream's
/// ring holds, position `p` at slot `p % rows`, and the marks a round checks before it drafts.
pub const Ring = struct {
    /// The stream's ring on the device: `rows` slots of `row` bytes each.
    ptr: u64,
    rows: usize,
    row: usize,
    /// The oldest position the ring holds (its `low`), and the one past its newest.
    lo: u64,
    written: u64,
    /// The end of the tokens the drafter was handed (`State.covered`).
    covered: u64,
};

/// Where a restore puts a carried ring: the stream's ring (its slots and row width) and the state's three marks.
pub const RingInto = struct { ptr: u64, rows: usize, row: usize, begin: *u64, written: *u64, covered: *u64 };

/// A ring's shape without its rows: what a state prepared for a later mark has to reserve.
pub const RingShape = struct { rows: usize, row: usize };

/// The ring bytes a snapshot keeps a state at `at` in a ring of `rows` slots: the positions the ring
/// can hold (`at`, or the whole ring once the prompt is longer than it).
pub fn ringArea(at: usize, rows: usize, row: usize) usize {
    if (rows == 0) return 0;
    return @min(at, rows) * row;
}

/// One kept state; `head`: it holds the MTP head's cache over the same rows, which a drafting stream needs;
/// `ring`: it also holds the DFlash2 drafter's tap ring in `buf`, after the caches.
pub const Snap = struct {
    rank_id: u64 = 0,
    buf: cuda.DeviceBuffer,
    at: usize,
    history: [2]i64,
    head: bool,
    /// The model's heads its pieces hold (the saved sequence's): a sequence of other heads cannot take it.
    heads: Heads = .full,
    /// The cache's precision its rows were laid out at (cuda_heads.zig): a sequence of the other precision cannot take it.
    kv8: bool = false,
    /// The drafter's ring this state carries: rows [ring_lo, ring_written) of a ring of `ring_rows` slots
    /// of `ring_row` bytes, at slot `p % ring_rows` in `buf` after the caches, and the marks a restore sets.
    ring: bool = false,
    ring_rows: usize = 0,
    ring_row: usize = 0,
    ring_lo: u64 = 0,
    ring_written: u64 = 0,
    ring_covered: u64 = 0,

    /// Where this state's ring rows start in `buf`: every cache's bytes come first.
    fn ringBase(s: *const Snap) u64 {
        return @intCast(bytes(s.at, s.head, s.heads, s.kv8));
    }

    /// Whether this state's ring holds every row an image at `at` names (the last `min(at, ring_rows)`
    /// positions) and the drafter was handed the whole prefix: a state restored mid-prompt holds only
    /// [ring_lo, at) of them, and stays in RAM.
    pub fn ringWhole(s: *const Snap, at: usize) bool {
        if (!s.ring or s.at != at) return false;
        const m = @min(at, s.ring_rows);
        return s.ring_written == at and s.ring_lo == at - m and s.ring_covered >= @as(u64, @intCast(at));
    }
};

/// Where a kept state's pieces sit in its buffer (move's order) for a pass's marks (Hooks.marks): whole heads only.
pub const layout = struct {
    const conv_bytes = sizes.convBytes(.full);
    const state_bytes = sizes.stateBytes(.full);
    /// A linear layer's conv window and the n-gram tail, in bytes.
    pub const conv_size = conv_bytes;
    pub const tail_size = tail_bytes;
    /// Linear layer li's conv window.
    pub fn conv(li: usize) u64 {
        return li * conv_bytes;
    }
    /// Linear layer li's DeltaNet state.
    pub fn state(li: usize) u64 {
        return wt.linear_n * conv_bytes + li * state_bytes;
    }
    /// The n-gram tail of a state after `n` tokens (after every attention cache's rows).
    pub fn tail(n: usize, kv8: bool) u64 {
        return wt.linear_n * (conv_bytes + state_bytes) + wt.attn_n * attnBytes(n, .full, kv8);
    }
};

const Way = enum { save, restore, rows };

/// A piece `move` copies: `layers` buffers of `n` bytes, rows of `row` bytes (0: whole) a `per`-position block each.
pub const Piece = struct { layers: usize, n: usize, row: usize = 0, per: usize = 1 };

/// `move`'s pieces in order at `at` and the model's heads `h` (the disk image's at full): DeltaNet, attention, tail.
pub fn order(at: usize, head: bool, h: Heads, kv8: bool) struct { p: [11]Piece, len: usize } {
    var o: [11]Piece = undefined;
    o[0] = .{ .layers = wt.linear_n, .n = sizes.convBytes(h) };
    o[1] = .{ .layers = wt.linear_n, .n = sizes.stateBytes(h) };
    o[2..6].* = attnOrder(wt.attn_n, at, h, kv8);
    o[6] = .{ .layers = 1, .n = tail_bytes };
    if (head) o[7..11].* = attnOrder(1, at, head_heads, kv8);
    return .{ .p = o, .len = if (head) 11 else 7 };
}

/// An attention cache's pieces for `at` rows: complete pooled blocks only (the rest pool again as later rows land).
/// Keys come first, then values: at fp8 a key row carries its two scales, a value row none, so the rows differ.
fn attnOrder(layers: usize, at: usize, h: Heads, kv8: bool) [4]Piece {
    const key_row = h.keyRowOf(kv8);
    const value_row = h.valueRowOf(kv8);
    return .{
        .{ .layers = layers, .n = at * key_row, .row = key_row },
        .{ .layers = layers, .n = at * value_row, .row = value_row },
        .{ .layers = layers, .n = at * index_row, .row = index_row },
        .{ .layers = layers, .n = at / wt.index_ratio * index_row, .row = index_row, .per = wt.index_ratio },
    };
}

/// Each piece of the sequence (and the head's cache) to or from the buffer at `base`; .rows: the attention rows only.
fn move(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, base: u64, at: usize, way: Way) !void {
    const dr = stream.d;
    var off: u64 = 0;
    // an fp8 cache's rows are not a page's: the paged path below strides bf16 rows only
    if (seq.kv8 and seq.paged != null) return error.SnapShape;
    const o = order(at, d != null, seq.heads, seq.kv8);
    var bufs: [11][]cuda.DeviceBuffer = undefined;
    bufs[0..7].* = .{ seq.conv, seq.state, seq.kc, seq.vc, seq.ikc, seq.pooled, (&seq.tail)[0..1] };
    if (d) |h| bufs[7..11].* = .{ (&h.kc)[0..1], (&h.vc)[0..1], (&h.ikc)[0..1], (&h.pooled)[0..1] };
    for (o.p[0..o.len], bufs[0..o.len], 0..) |p, layers, pi| {
        if (layers.len != p.layers) return error.SnapShape;
        for (layers, 0..) |b, layer| {
            if (p.n == 0) continue;
            if (way == .rows and (pi < 2 or pi == 6)) {
                off += p.n;
                continue;
            }
            // the model's caches on the stream's pages; the head's too in a pool lane (the layer after the model's)
            const head = pi >= 7 and d != null and d.?.pg.pt != 0;
            if ((pi >= 2 and pi < 6) or head) if (seq.paged) |*pg| {
                // the same bytes in the same order as a contiguous cache's, a run a page
                const part: pages.Part = @fromBackingInt(@intCast(if (head) pi - 7 else pi - 2));
                const at_layer = if (head) wt.attn_n else layer;
                const row = pages.rowBytes(part);
                if (row != p.row) return error.SnapShape;
                const per = pages.rows(part, pg.tokens);
                const n = p.n / row;
                var r: usize = 0;
                while (r < n) {
                    const run = @min(per - r % per, n - r);
                    const at_row = pg.rowAt(at_layer, part, r);
                    const dst, const src = if (way != .restore) .{ base + off, at_row } else .{ at_row, base + off };
                    try dr.check(dr.api.cuMemcpyDtoDAsync_v2(dst, src, run * row, stream.handle), "cuMemcpyDtoDAsync");
                    off += run * row;
                    r += run;
                }
                continue;
            };
            const dst, const src = if (way != .restore) .{ base + off, b.ptr } else .{ b.ptr, base + off };
            try dr.check(dr.api.cuMemcpyDtoDAsync_v2(dst, src, p.n, stream.handle), "cuMemcpyDtoDAsync");
            off += p.n;
        }
    }
    std.debug.assert(off == bytes(at, d != null, seq.heads, seq.kv8));
}

/// A ring's rows for positions [lo, end) into or out of `base + area`, `row` bytes a slot, position `p` at
/// slot `p % rows`: one run when the span does not wrap the ring, two when it does (the second at slot 0).
fn ringMove(stream: cuda.Stream, ptr: u64, rows: usize, row: usize, lo: u64, end: u64, base: u64, area: u64, way: Way) !void {
    if (rows == 0 or row == 0) return error.SnapShape;
    if (end <= lo) return;
    const n: usize = @intCast(end - lo);
    if (n > rows) return error.SnapShape;
    const dr = stream.d;
    var at: usize = @intCast(lo % @as(u64, @intCast(rows)));
    var left = n;
    while (left > 0) {
        const run = @min(left, rows - at);
        const dev = ptr + @as(u64, at) * @as(u64, row);
        const buf = base + area + @as(u64, at) * @as(u64, row);
        const dst, const src = if (way != .restore) .{ buf, dev } else .{ dev, buf };
        try dr.check(dr.api.cuMemcpyDtoDAsync_v2(dst, src, run * row, stream.handle), "cuMemcpyDtoDAsync");
        left -= run;
        at = 0;
    }
}

/// A copy of `seq` (and of the head's cache, with `d`) as it stands after `seq.fed` tokens.
pub fn save(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft) !*Snap {
    return saveWith(gpa, driver, stream, seq, d, null);
}

/// `save`, and the drafter's ring `ring` (its rows [lo, written) and marks) with the state, when a stream drafts.
pub fn saveWith(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, ring: ?Ring) !*Snap {
    const at = seq.fed;
    if (d) |h| if (h.len != at or h.drafted != 0) return error.NotAtMark;
    // the ring's rows past the mark are the prompt's own tokens again when a resume re-runs them: carry
    // only rows through `at`, and the marks as far as `at` reaches
    var r: ?Ring = ring;
    if (r) |*x| {
        x.written = @min(x.written, @as(u64, @intCast(at)));
        x.covered = @min(x.covered, @as(u64, @intCast(at)));
        // a ring nothing has tapped yet carries no rows: the state is a plain one
        if (x.written <= x.lo) {
            r = null;
        } else if (x.written - x.lo > @as(u64, @intCast(@min(at, x.rows)))) return error.SnapShape;
    }
    const own = bytes(at, d != null, seq.heads, seq.kv8);
    const area = if (r) |x| ringArea(at, x.rows, x.row) else 0;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .buf = try cuda.DeviceBuffer.alloc(driver, own + area), .at = at, .history = seq.history, .head = d != null, .heads = seq.heads, .kv8 = seq.kv8 };
    errdefer snap.buf.free();
    errdefer cuda.completion.drain(driver, stream, "a failed snapshot save did not drain");
    try move(stream, seq, d, snap.buf.ptr, at, .save);
    if (r) |x| {
        snap.ring = true;
        snap.ring_rows = x.rows;
        snap.ring_row = x.row;
        snap.ring_lo = x.lo;
        snap.ring_written = x.written;
        snap.ring_covered = x.covered;
        try ringMove(stream, x.ptr, x.rows, x.row, x.lo, x.written, snap.buf.ptr, own, .save);
    }
    try stream.synchronize();
    return snap;
}

/// A kept state for position `at` that a pass running past it fills (Hooks.marks); `finish` adds the attention rows.
pub fn prepare(gpa: std.mem.Allocator, driver: *const cuda.Driver, at: usize, history: [2]i64, head: bool, heads: Heads, kv8: bool) !*Snap {
    return prepareWith(gpa, driver, at, history, head, heads, kv8, null);
}

/// `prepare`, with room for a ring of `ring`'s shape (a drafting stream's state): `finish` fills its rows.
pub fn prepareWith(gpa: std.mem.Allocator, driver: *const cuda.Driver, at: usize, history: [2]i64, head: bool, heads: Heads, kv8: bool, ring: ?RingShape) !*Snap {
    const area = if (ring) |x| ringArea(at, x.rows, x.row) else 0;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .buf = try cuda.DeviceBuffer.alloc(driver, bytes(at, head, heads, kv8) + area), .at = at, .history = history, .head = head, .heads = heads, .kv8 = kv8 };
    if (ring) |x| {
        snap.ring = true;
        snap.ring_rows = x.rows;
        snap.ring_row = x.row;
    }
    return snap;
}

/// After the pass ran past `snap.at`: the attention caches' first rows (and the head's) into `snap`, queued.
pub fn finish(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *Snap) !void {
    return finishWith(stream, seq, d, snap, null);
}

/// `finish`, and the drafter's ring's rows [lo, at) with the attention rows when the state carries a ring.
pub fn finishWith(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *Snap, ring: ?Ring) !void {
    if (seq.fed < snap.at) return error.NotAtMark;
    if (d) |h| if (!snap.head or h.len < snap.at) return error.NotAtMark;
    try move(stream, seq, d, snap.buf.ptr, snap.at, .rows);
    if (!snap.ring) return;
    // no ring to fill the rows it reserved: the state keeps them unwritten and stays in RAM (`ringWhole`)
    const x = ring orelse {
        snap.ring = false;
        snap.ring_rows = 0;
        snap.ring_row = 0;
        return;
    };
    if (x.rows != snap.ring_rows or x.row != snap.ring_row) return error.SnapShape;
    const at: u64 = @intCast(snap.at);
    const written = @min(x.written, at);
    if (x.lo > written) return error.SnapShape;
    snap.ring_lo = x.lo;
    snap.ring_written = written;
    snap.ring_covered = @min(x.covered, at);
    try ringMove(stream, x.ptr, x.rows, x.row, x.lo, written, snap.buf.ptr, snap.ringBase(), .save);
}

/// `seq` (and the head's cache, with `d`) becomes `snap`'s state; refused before any copy when it cannot hold it.
pub fn restore(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *const Snap) !void {
    return restoreWith(stream, seq, d, snap, null);
}

/// `restore`, and the ring a state carries into `into` (a drafting stream's own ring and marks).
pub fn restoreWith(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *const Snap, into: ?RingInto) !void {
    if (snap.at > seq.cap) return error.SequenceTooShort;
    if (!snap.heads.eql(seq.heads)) return error.SnapShape;
    // a bf16-shaped state must not land in fp8 rows, or the reverse: every row after the first would stride wrong
    if (snap.kv8 != seq.kv8) return error.SnapShape;
    if (d) |h| if (!snap.head or snap.at > h.cap) return error.NoHeadState;
    // another drafter's ring (another window, another row) cannot take these rows: refused before any copy
    if (snap.ring and into != null and (into.?.rows != snap.ring_rows or into.?.row != snap.ring_row)) return error.SnapShape;
    // a state with the head's rows serves a stream without drafts too: those rows stay in the buffer
    try move(stream, seq, d, snap.buf.ptr, snap.at, .restore);
    // its ring's rows go back to their own slots and its marks to the state: a round then finds the prefix
    // covered exactly as the saving stream left it
    if (snap.ring) if (into) |x| {
        try ringMove(stream, x.ptr, x.rows, x.row, snap.ring_lo, snap.ring_written, snap.buf.ptr, snap.ringBase(), .restore);
        x.begin.* = snap.ring_lo;
        x.written.* = snap.ring_written;
        x.covered.* = snap.ring_covered;
    };
    try stream.synchronize();
    seq.history = snap.history;
    seq.fed = snap.at;
    if (d) |h| {
        h.reset();
        h.len = snap.at;
    }
}

pub fn drop(gpa: std.mem.Allocator, stream: cuda.Stream, snap: *Snap) void {
    cuda.completion.drain(stream.d, stream, "a snapshot drop did not drain");
    snap.buf.free();
    gpa.destroy(snap);
}

test "a state's bytes: about 110 MiB of DeltaNet states and the n-gram tail, then 27 KiB a token" {
    try std.testing.expectEqual(@as(usize, 12 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, false, .full, false) - bytes(0, false, .full, false));
    // a complete block of 4 adds its pooled row
    try std.testing.expectEqual(@as(usize, 12 * (4 * (2 * 2 * 256 * 2 + 128 * 2) + 128 * 2)), bytes(4, false, .full, false) - bytes(0, false, .full, false));
    // the head's cache is one more attention layer's rows
    try std.testing.expectEqual(@as(usize, 13 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, true, .full, false) - bytes(0, true, .full, false));
    try std.testing.expect(bytes(0, false, .full, false) > 100 << 20 and bytes(0, false, .full, false) < 120 << 20);
}

test "a state at a rank's half of the heads: half the conv windows, states and KV rows, the indexer and the head's whole" {
    const h = Heads.half;
    try std.testing.expectEqual(@as(usize, 36 * (3 * 5120 * 2 + 24 * 128 * 128 * 4) + 9 * 4 * 2560 * 2), bytes(0, false, h, false));
    try std.testing.expectEqual(@as(usize, 12 * (2 * 256 * 2 + 128 * 2)), bytes(1, false, h, false) - bytes(0, false, h, false));
    // the head's cache keeps its two KV heads beside the model's one
    try std.testing.expectEqual(@as(usize, 12 * (2 * 256 * 2 + 128 * 2) + (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, true, h, false) - bytes(0, true, h, false));
    const o = order(8, true, h, false);
    try std.testing.expectEqual(@as(usize, 3 * 5120 * 2), o.p[0].n);
    try std.testing.expectEqual(@as(usize, 24 * 128 * 128 * 4), o.p[1].n);
    for (o.p[2..4]) |p| try std.testing.expectEqual(@as(usize, 512), p.row);
    for (o.p[7..9]) |p| try std.testing.expectEqual(@as(usize, 1024), p.row);
    var sum: usize = 0;
    for (o.p[0..o.len]) |p| sum += p.layers * p.n;
    try std.testing.expectEqual(bytes(8, true, h, false), sum);
}

test "an fp8 state: bytes() is the exact sum of its pieces, keys a row longer than values" {
    for ([_]bool{ false, true }) |head| for ([_]usize{ 0, 1, 3, 4, 255, 256, 4137 }) |at| {
        const o = order(at, head, .full, true);
        var sum: usize = 0;
        for (o.p[0..o.len]) |p| sum += p.layers * p.n;
        try std.testing.expectEqual(bytes(at, head, .full, true), sum);
        // a key row keeps both scales beside its codes; a value row is codes only
        try std.testing.expectEqual(Heads.full.kv8KeyRow(), o.p[2].row);
        try std.testing.expectEqual(Heads.full.kv8ValueRow(), o.p[3].row);
        try std.testing.expect(o.p[2].row > o.p[3].row);
        // the head's cache is at the whole layer's heads either way
        if (head) try std.testing.expectEqual(Heads.full.kv8KeyRow(), o.p[7].row);
    };
}

test "an fp8 state of the same length is smaller by each attention layer's two KV heads' scale blocks" {
    // bf16 stores a key and a value row of head_dim * 2 bytes a KV head; fp8 a key row of head_dim + 16 and a value
    // row of head_dim: 2 * head_dim - 16 bytes fewer a KV head, so `kv` times that a layer, a position
    for ([_]usize{ 1, 4, 1000, 4137 }) |at| for ([_]bool{ false, true }) |head| {
        const layers = wt.attn_n + @intFromBool(head);
        const saved = bytes(at, head, .full, false) - bytes(at, head, .full, true);
        try std.testing.expectEqual(layers * at * Heads.full.kv * (2 * wt.head_dim - 16), saved);
        try std.testing.expect(saved > 0);
    };
}

test "a state of the whole layer restores into no sequence of half the heads, and the reverse" {
    var driver: cuda.Driver = undefined;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    for ([_][2]Heads{ .{ .full, .half }, .{ .half, .full } }) |pair| {
        const kept: Snap = .{ .buf = .{ .d = &driver, .ptr = 0x100, .len = 1, .borrowed = true }, .at = 1, .history = .{ 0, 0 }, .head = false, .heads = pair[0] };
        var seq: prompt.Seq = .{ .heads = pair[1], .conv = &.{}, .state = &.{}, .kc = &.{}, .vc = &.{}, .ikc = &.{}, .pooled = &.{}, .tail = .{ .d = &driver, .ptr = 0, .len = 0 }, .history = .{ 0, 0 }, .fed = 0, .logits = &.{}, .cap = 8 };
        // refused before any copy: no driver call is made
        try std.testing.expectError(error.SnapShape, restore(stream, &seq, null, &kept));
        try std.testing.expectEqual(@as(usize, 0), seq.fed);
    }
}

test "a bf16 state restores into no fp8 sequence, and the reverse, before any copy" {
    var driver: cuda.Driver = undefined;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    for ([_]bool{ false, true }) |seq_kv8| {
        const kept: Snap = .{ .buf = .{ .d = &driver, .ptr = 0x100, .len = 1, .borrowed = true }, .at = 1, .history = .{ 0, 0 }, .head = false, .heads = .full, .kv8 = !seq_kv8 };
        var seq: prompt.Seq = .{ .kv8 = seq_kv8, .heads = .full, .conv = &.{}, .state = &.{}, .kc = &.{}, .vc = &.{}, .ikc = &.{}, .pooled = &.{}, .tail = .{ .d = &driver, .ptr = 0, .len = 0 }, .history = .{ 0, 0 }, .fed = 0, .logits = &.{}, .cap = 8 };
        // refused before any copy: no driver call is made
        try std.testing.expectError(error.SnapShape, restore(stream, &seq, null, &kept));
        try std.testing.expectEqual(@as(usize, 0), seq.fed);
    }
}

test {
    _ = @import("cuda_paging_snap_test.zig");
}
