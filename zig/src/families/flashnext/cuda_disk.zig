//! Flash Next's CUDA kept states on disk (--learn-conversations): their image, device copies out and in, the probe.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const snap = @import("cuda_snap.zig");
const rows = api.prompt_cache.rows;
const Allocator = std.mem.Allocator;

const words = 80; // the n-gram history, the position, whether the MTP head's rows and the drafter's ring follow, the ring's shape and its marks
const Words = extern struct { history: [2]i64, at: u64, head: u64, ring: u64, ring_rows: u64, ring_row: u64, low: u64, written: u64, covered: u64 };

/// The image's parts at `at`, the cache's precision `kv8` and the drafter's ring (`ring`: a server that drafts
/// keeps it): words, DeltaNet tails and states, each attention piece by layer, the tail, the head's, and the ring's.
pub fn partsOf(at: u32, head: bool, ring: ?snap.RingShape, kv8: bool, out: *std.ArrayList(rows.image.Part), a: Allocator) !void {
    const sh = snap.shapeOf(kv8);
    try out.append(a, .{ .bytes = words });
    try out.append(a, .{ .bytes = @as(u64, sh.linear) * @as(u64, sh.conv) });
    try out.append(a, .{ .bytes = @as(u64, sh.linear) * @as(u64, sh.state) });
    try attention(at, sh.attn, sh, out, a);
    try out.append(a, .{ .bytes = @as(u64, sh.tail) });
    if (head) try attention(at, 1, sh, out, a);
    if (ring) |r| try drafterRing(at, r, out, a);
}

/// The DFlash2 drafter's tap ring as cuda_snap.zig carries it: the last `min(at, rows)` positions, a row each,
/// position `p` at row `p % rows` (the device ring's own slot) so a state's blocks are per-position content.
fn drafterRing(at: u32, r: snap.RingShape, out: *std.ArrayList(rows.image.Part), a: Allocator) !void {
    if (r.rows == 0 or r.row == 0 or r.rows > std.math.maxInt(u32) or r.row > std.math.maxInt(u32)) return error.SnapShape;
    if (at == 0) return; // nothing tapped yet: no positions for a part to hold
    const slots: u32 = @intCast(r.rows);
    try out.append(a, .{ .bytes = @as(u64, @min(at, slots)) * @as(u64, r.row), .row = @intCast(r.row), .window = slots });
}

/// As cuda_snap.zig's move lays them: every layer's keys, then every layer's values, indexer keys, and pooled rows of
/// `ratio`. At fp8 a key row keeps its two scales and a value row none, so the two groups have different row sizes.
fn attention(at: u32, layers: usize, sh: snap.Shape, out: *std.ArrayList(rows.image.Part), a: Allocator) !void {
    const key_row: u64 = @intCast(sh.key_row);
    const value_row: u64 = @intCast(sh.value_row);
    const index_row: u64 = @intCast(sh.index_row);
    for (0..layers) |_| try out.append(a, .{ .bytes = @as(u64, at) * key_row, .row = @intCast(key_row) });
    for (0..layers) |_| try out.append(a, .{ .bytes = @as(u64, at) * value_row, .row = @intCast(value_row) });
    for (0..layers) |_| try out.append(a, .{ .bytes = @as(u64, at) * index_row, .row = @intCast(index_row) });
    for (0..layers) |_| try out.append(a, .{ .bytes = (@as(u64, at) / @as(u64, sh.ratio)) * index_row, .row = @intCast(index_row), .per = @intCast(sh.ratio) });
}

/// Whether `C` keeps a drafter whose states carry a ring (a server opened with --drafter).
fn hasRing(comptime C: type) bool {
    if (@hasField(C, "drafter")) {
        const P = @FieldType(C, "drafter");
        if (@typeInfo(P) == .optional) {
            const T = @typeInfo(P).optional.child; // *d2.Draft2 (the field is an optional pointer)
            if (@typeInfo(T) != .pointer) return false;
            const D = @typeInfo(T).pointer.child; // d2.Draft2
            return @hasDecl(D, "ringRows") and @hasDecl(D, "rowBytes");
        }
    }
    return false;
}

/// The disk tier of a Flash Next namespace `F` (its Cuda backend is the prompt cache's target).
pub fn Kept(comptime F: type) type {
    return struct {
        pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole };
        /// The states are one tensor rank's: --tp refuses --learn-conversations (native/cuda_refusal.zig).
        pub const one_rank = true;

        fn of(ptr: *anyopaque) *F.Cuda {
            return @ptrCast(@alignCast(ptr));
        }

        /// The drafter's ring shape a server that drafts keeps in its states, or null (and then no image part).
        fn ringOf(c: *F.Cuda) ?snap.RingShape {
            if (comptime hasRing(F.Cuda)) {
                const d = c.drafter orelse return null;
                return .{ .rows = d.ringRows(), .row = d.rowBytes() };
            } else return null;
        }

        fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
            const c = of(ptr);
            try partsOf(at, c.head != null, ringOf(c), c.engine.kv8, out, a);
        }

        fn copy(c: *F.Cuda, st: *snap.Snap, ranges: []const rows.image.Range, host: []u8, out: bool) !void {
            const d = c.engine.driver;
            // a stored state is this image's only at its precision: another's rows would land at the wrong stride
            if (st.kv8 != c.engine.kv8) return error.KeptPrecisionDiffer;
            // a later range's failure leaves earlier copies queued on `host`, which its caller then frees or reuses
            errdefer cuda.completion.drain(d, c.stream, "a failed kept-state copy did not drain");
            var w: Words = .{ .history = st.history, .at = st.at, .head = @intFromBool(st.head), .ring = @intFromBool(st.ring), .ring_rows = @intCast(st.ring_rows), .ring_row = @intCast(st.ring_row), .low = st.ring_lo, .written = st.ring_written, .covered = st.ring_covered };
            var o: usize = 0;
            for (ranges) |g| {
                var off: usize = @intCast(g.off);
                var len: usize = @intCast(g.len);
                if (off + len > words + st.buf.len) return error.BadKeptRange;
                if (off < words) {
                    const n = @min(len, words - off);
                    if (out) @memcpy(host[o..][0..n], std.mem.asBytes(&w)[off..][0..n]) else @memcpy(std.mem.asBytes(&w)[off..][0..n], host[o..][0..n]);
                    o += n;
                    off += n;
                    len -= n;
                }
                if (len == 0) continue;
                const dev = st.buf.ptr + off - words;
                if (out) {
                    try d.check(d.api.cuMemcpyDtoHAsync_v2(host[o..].ptr, dev, len, c.stream.handle), "cuMemcpyDtoHAsync");
                } else try d.check(d.api.cuMemcpyHtoDAsync_v2(dev, host[o..].ptr, len, c.stream.handle), "cuMemcpyHtoDAsync");
                o += len;
            }
            try c.stream.synchronize(); // `host` is the store's staging, reused by its next piece
            if (out) return;
            // a state read back stands where it was kept, with the head's rows and the ring's as kept: else it restores nowhere
            if (w.at != st.at or w.head != @intFromBool(st.head) or w.ring != @intFromBool(st.ring)) return error.KeptWordsDiffer;
            // an image of another drafter's ring (another window, another row) cannot fill this state's blanks
            if (st.ring and (w.ring_rows != st.ring_rows or w.ring_row != st.ring_row)) return error.KeptWordsDiffer;
            // the ring's marks: rows [low, written) all before its position, and nothing handed past them
            if (w.low > w.written or w.written > w.at or w.covered < w.written) return error.KeptWordsDiffer;
            st.history = w.history;
            st.ring_lo = w.low;
            st.ring_written = w.written;
            st.ring_covered = w.covered;
        }

        fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
            try copy(of(ptr), @ptrCast(@alignCast(saved)), ranges, out, true);
        }

        fn scatter(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
            try copy(of(ptr), @ptrCast(@alignCast(saved)), ranges, @constCast(in), false);
        }

        /// A device buffer for `at` tokens with the head's rows if this server drafts and the drafter's ring if it
        /// keeps one, filled by `scatter`.
        fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
            const c = of(ptr);
            const head = c.head != null;
            const ring = ringOf(c);
            const area = if (ring) |r| snap.ringArea(at, r.rows, r.row) else 0;
            const st = try c.gpa.create(snap.Snap);
            errdefer c.gpa.destroy(st);
            st.* = .{ .buf = try cuda.DeviceBuffer.alloc(c.engine.driver, snap.bytes(at, head, .full, c.engine.kv8) + area), .at = at, .history = .{ 0, 0 }, .head = head, .kv8 = c.engine.kv8 };
            if (ring) |r| {
                st.ring = true;
                st.ring_rows = r.rows;
                st.ring_row = r.row;
            }
            return @ptrCast(st);
        }

        /// A state from a stream that did not draft lacks the head's rows (or the ring's) this server's parts name:
        /// it stays off disk. So does a state of the other cache precision, or one whose ring covers only the
        /// positions after a resume (the image names the whole ring).
        fn whole(ptr: *anyopaque, saved: api.prompt_cache.Saved) bool {
            const st: *const snap.Snap = @ptrCast(@alignCast(saved));
            const c = of(ptr);
            if (ringOf(c)) |r| {
                // the image at `at` names min(at, rows) ring rows: fewer held rows mean the state is RAM's alone
                if (!st.ringWhole(st.at) or st.ring_rows != r.rows or st.ring_row != r.row) return false;
            } else if (st.ring) return false; // a ring this server's image has no parts for
            return st.head == (c.head != null) and st.rank_id == 0 and st.heads.eql(.full) and st.kv8 == c.engine.kv8;
        }

        /// A fixed prompt's state at a mark inside its last chunk, hashed, and the prompt rate it took.
        pub fn probe(a: Allocator, io: std.Io, backend: lanes.backend.Backend, target: *anyopaque) !struct { hash: u64, rate: f64 } {
            const c = of(target);
            var toks: [4096 + 40 + 7]u32 = undefined;
            for (&toks, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000);
            const Hook = struct {
                c: *F.Cuda,
                saved: ?*anyopaque = null,
                failed: ?anyerror = null,
                fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
                    const p: *@This() = @ptrCast(@alignCast(ptr));
                    p.saved = F.snap_save(p.c, s, mark) catch |e| {
                        p.failed = e;
                        return;
                    };
                }
            };
            var hook: Hook = .{ .c = c };
            defer if (hook.saved) |saved| F.snap_drop(c, saved);
            const marks = [_]u32{4136};
            var s = try lanes.Stream.init(a, .{ .id = "probe", .prompt = &toks, .max_new = 1, .drafts = c.head != null, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
            defer s.deinit(a);
            const ns = pass: {
                defer backend.vtable.release(backend.ptr, &s); // once, whether the pass, its wait or nothing failed
                const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
                try backend.vtable.prefill(backend.ptr, &s);
                try c.stream.synchronize();
                break :pass std.Io.Clock.awake.now(io).toNanoseconds() - t0;
            };
            const saved = hook.saved orelse return hook.failed orelse error.ProbeUnmarked;
            if (!whole(c, saved)) return error.ProbeUnmarked;
            const st: *snap.Snap = @ptrCast(@alignCast(saved));
            const bytes = try a.alloc(u8, words + st.buf.len);
            try copy(c, st, &.{.{ .off = 0, .len = bytes.len }}, bytes, true);
            return .{ .hash = std.hash.Wyhash.hash(0x666e, bytes), .rate = @as(f64, @floatFromInt(toks.len)) / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
        }

        /// What else the states depend on: whether the MTP head's rows and the drafter's tap ring are part of them,
        /// and the caches' precision, whose rows are a different length (fp8 keys keep their scales): each gets its
        /// own directory under the root, so a state kept without a ring is never read as one with it.
        pub fn facts(target: *anyopaque) [3]u64 {
            const c = of(target);
            return .{ @intFromBool(c.head != null), @intFromBool(c.engine.kv8), @intFromBool(ringOf(c) != null) };
        }

        /// The native key extends with policyOf under this domain (native/cuda_disk.zig keptKey).
        /// v2: the image's words carry the drafter's ring marks and its parts a ring, so no v1 state is read as one.
        pub const policy_domain = "flash-cuda-policy-v2";
        pub fn policy(target: *anyopaque) [14]u64 {
            return policyOf(of(target));
        }
    };
}

/// What a prompt pass that writes a kept state was resolved with at load: its rows, kernel forms and expert grids.
pub fn policyOf(c: anytype) [14]u64 {
    const e = c.engine;
    const k = &e.batch_k;
    const f = k.forms;
    var grids = std.hash.Wyhash.init(0); // resident blocks a grid takes, which pick the experts' decode forms
    grids.update(std.mem.asBytes(&k.nvfp4.resident));
    grids.update(std.mem.asBytes(&k.fp8.resident));
    if (k.expert_tile) |x| grids.update(std.mem.asBytes(&x.resident));
    grids.update(std.mem.asBytes(&k.native_lane.prompt_sms)); // a BF16 prompt tile runs over half these SMs' worth
    return .{
        // TF_FLASHNEXT_CHUNK_SOLO and _CHUNK as resolved: a pass's rows alone, and beside a decoding stream
        e.chunk,
        e.busy_chunk,
        @intFromBool(f.gdn_fused),
        @intFromBool(f.hc_fuse),
        @intFromBool(f.in_pass),
        f.readout_rows,
        f.fact_rows,
        // TF_FLASHNEXT_PROMPT_EXPERT_TILE, _PROMPT_BF16 and TF_NVFP4_DECODE_WARPS as the kernels loaded
        if (k.expert_tile) |x| 1 + @as(u64, @backingInt(x.selection)) else 0,
        @intFromBool(k.native_lane.prompt != null),
        // TF_QMMF_HALF_BELOW: a pass of 16 rows or fewer takes the lane's 32-column BF16 and MXFP8 tile below it
        k.native_lane.half_below,
        k.nvfp4.warps,
        grids.final(),
        // a captured set in use, and the entries TF_FLASHNEXT_OWNED runs on our glue beside it
        if (c.set != null) 1 + @as(u64, c.glue.mixed) else 0,
        // --drafter: every state this server keeps carries its tap ring (another image's states do not)
        if (comptime hasRing(@TypeOf(c.*))) @intFromBool(c.drafter != null) else 0,
    };
}

test "the image covers a state's device bytes once, its rows split into 256-token blocks, with and without the head" {
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |kv8| for ([_]bool{ false, true }) |head| for ([_]u32{ 1, 3, 256, 1000, 4137 }) |at| {
        var parts: std.ArrayList(rows.image.Part) = .empty;
        defer parts.deinit(a);
        try partsOf(at, head, null, kv8, &parts, a);
        try std.testing.expect(rows.image.valid(parts.items, at, rows.block_tokens));
        try std.testing.expectEqual(words + snap.bytes(at, head, .full, kv8), rows.image.imageBytes(parts.items));
    };
}

test "the image's parts are cuda_snap.move's pieces in its order: sizes, offsets and rows, at either precision" {
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |kv8| for ([_]bool{ false, true }) |head| for ([_]u32{ 0, 1, 3, 4, 255, 256, 1000, 4136, 4137 }) |at| {
        var parts: std.ArrayList(rows.image.Part) = .empty;
        defer parts.deinit(a);
        try partsOf(at, head, null, kv8, &parts, a);
        try std.testing.expectEqual(@as(u64, words), parts.items[0].bytes);
        const o = snap.order(at, head, .full, kv8);
        var i: usize = 1;
        var off: u64 = words; // the image's offset; move's buffer offset is this less the words
        var moved: u64 = 0;
        for (o.p[0..o.len]) |p| {
            if (p.row == 0) { // the DeltaNet's pieces and the tail: one part of every layer's bytes
                try std.testing.expectEqual(@as(u64, p.layers * p.n), parts.items[i].bytes);
                try std.testing.expectEqual(@as(u32, 0), parts.items[i].row);
                off += parts.items[i].bytes;
                moved += p.layers * p.n;
                i += 1;
                continue;
            }
            for (0..p.layers) |_| { // each layer's rows: keys, values, indexer keys, pooled blocks
                const x = parts.items[i];
                try std.testing.expectEqual(@as(u64, p.n), x.bytes);
                try std.testing.expectEqual(@as(u64, p.row), x.row);
                try std.testing.expectEqual(@as(u64, p.per), x.per);
                try std.testing.expectEqual(off - words, moved);
                off += x.bytes;
                moved += p.n;
                i += 1;
            }
        }
        try std.testing.expectEqual(parts.items.len, i);
        try std.testing.expectEqual(@as(u64, snap.bytes(at, head, .full, kv8)), moved);
        try std.testing.expectEqual(rows.image.imageBytes(parts.items), off);
    };
}

test "a drafting state's image carries the drafter's ring: the last min(at, rows) positions, a row each, windowed" {
    const a = std.testing.allocator;
    const shape: snap.RingShape = .{ .rows = 2064, .row = 25_600 }; // 2,048 + an 8-row block + slack, 5 taps of 2,560 bf16
    for ([_]bool{ false, true }) |kv8| for ([_]bool{ false, true }) |head| for ([_]u32{ 0, 1, 255, 256, 1000, 2063, 2064, 2065, 4137, 5000 }) |at| {
        var parts: std.ArrayList(rows.image.Part) = .empty;
        defer parts.deinit(a);
        try partsOf(at, head, shape, kv8, &parts, a);
        try std.testing.expect(rows.image.valid(parts.items, at, rows.block_tokens));
        // the ring follows every cache in the image, as many rows as the ring can hold of the prompt
        try std.testing.expectEqual(words + snap.bytes(at, head, .full, kv8) + snap.ringArea(at, shape.rows, shape.row), rows.image.imageBytes(parts.items));
        if (at == 0) {
            // nothing tapped yet: the ring adds no part at all
            var bare: std.ArrayList(rows.image.Part) = .empty;
            defer bare.deinit(a);
            try partsOf(at, head, null, kv8, &bare, a);
            try std.testing.expectEqual(bare.items.len, parts.items.len);
            continue;
        }
        const ring = parts.items[parts.items.len - 1];
        try std.testing.expectEqual(@as(u64, @min(at, @as(u32, @intCast(shape.rows)))) * @as(u64, shape.row), ring.bytes);
        try std.testing.expectEqual(@as(u32, @intCast(shape.row)), ring.row);
        try std.testing.expectEqual(@as(u32, @intCast(shape.rows)), ring.window);
        try std.testing.expectEqual(@as(u32, 1), ring.per);
        try std.testing.expectEqual(@as(u64, ring.bytes), parts.items[parts.items.len - 1].bytes);
        // the ring is a group of its own: a 256-token block is 256 of its rows, and only a long prompt's blocks
        // inside the window are shared (a short prompt's rows stay in the state's own file)
        var gbuf: [8]u32 = undefined;
        const groups = try rows.image.groups(parts.items, &gbuf);
        try std.testing.expectEqual(@as(usize, 2), groups.len);
        try std.testing.expectEqual(@as(u32, @intCast(shape.rows)), groups[1]);
        try std.testing.expectEqual(@as(u64, rows.block_tokens) * @as(u64, shape.row), rows.image.blockBytes(parts.items, rows.block_tokens, groups[1]));
        const w = rows.image.wholeBlocks(at, rows.block_tokens, groups[1]);
        try std.testing.expectEqual(at / rows.block_tokens, w.end);
        try std.testing.expectEqual(((at -| @as(u32, @intCast(shape.rows))) + rows.block_tokens - 1) / rows.block_tokens, w.first);
        // each block inside the window is that block's positions at their own slots: slot p % rows is the device's
        // ring slot, so a block's bytes are the device's rows in the device's order, and a block that wraps slot 0
        // comes back as two runs
        const ring_off = rows.image.imageBytes(parts.items) - ring.bytes;
        var list: std.ArrayList(rows.image.Range) = .empty;
        defer list.deinit(a);
        var wrapped = false;
        var k = w.first;
        while (k < w.end) : (k += 1) {
            list.clearRetainingCapacity();
            try rows.image.blockRanges(parts.items, rows.block_tokens, groups[1], k, &list, a);
            try std.testing.expectEqual(rows.image.blockBytes(parts.items, rows.block_tokens, groups[1]), rows.image.total(list.items));
            if (list.items.len == 2) {
                wrapped = true;
                const slot = (k * rows.block_tokens) % @as(u32, @intCast(shape.rows));
                try std.testing.expectEqual(ring_off + @as(u64, slot) * @as(u64, shape.row), list.items[0].off);
                try std.testing.expectEqual(@as(u64, @as(u32, @intCast(shape.rows)) - slot) * @as(u64, shape.row), list.items[0].len);
                try std.testing.expectEqual(ring_off, list.items[1].off);
            }
        }
        // a prompt long enough to have a block straddle the ring's slot 0 (block 16 at 4,352 tokens) must show the wrap
        if (at >= 4352) try std.testing.expect(wrapped);
    };
}

test "a state's ring fills the image's rows only when it holds the whole ring the image names" {
    var driver: cuda.Driver = undefined;
    const at: usize = 4096;
    const shape: snap.RingShape = .{ .rows = 2064, .row = 25_600 };
    var st: snap.Snap = .{ .buf = .{ .d = &driver, .ptr = 0, .len = 0, .borrowed = true }, .at = at, .history = .{ 0, 0 }, .head = false, .ring = true, .ring_rows = shape.rows, .ring_row = shape.row };
    // a cold prefill wrote every row the ring holds: the state may go to disk
    st.ring_lo = at - shape.rows;
    st.ring_written = at;
    st.ring_covered = at;
    try std.testing.expect(st.ringWhole(at));
    // the rows are all there but the drafter was handed less than the prefix: a round at `at` has no token to start from
    st.ring_covered = at - 1;
    try std.testing.expect(!st.ringWhole(at));
    st.ring_covered = at;
    // a state resumed at 3,000 holds only [3000, 4096): the image's oldest rows are missing, so it stays in RAM
    st.ring_lo = 3000;
    try std.testing.expect(!st.ringWhole(at));
    // a shorter prompt than the ring: every position from 0 is held
    st.at = 1000;
    st.ring_lo = 0;
    st.ring_written = 1000;
    st.ring_covered = 1000;
    try std.testing.expect(st.ringWhole(1000));
    st.ring_lo = 1;
    try std.testing.expect(!st.ringWhole(1000));
    // a state saved without a ring never fills a ring's image
    st.ring = false;
    try std.testing.expect(!st.ringWhole(1000));
}

test "the image's offsets total a state's bytes at either precision: the invariant its blocks are cut at" {
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |kv8| for ([_]bool{ false, true }) |head| for ([_]u32{ 1, 4, 256, 1000, 4137 }) |at| {
        var parts: std.ArrayList(rows.image.Part) = .empty;
        defer parts.deinit(a);
        try partsOf(at, head, null, kv8, &parts, a);
        try std.testing.expectEqual(words + snap.bytes(at, head, .full, kv8), rows.image.imageBytes(parts.items));
        if (kv8) { // the same length at fp8 is smaller: its keys keep 12 bytes of scales a position a layer's KV head
            try std.testing.expect(words + snap.bytes(at, head, .full, false) > rows.image.imageBytes(parts.items));
        }
    };
}

/// A driver whose device copies, out or in, queue the first and refuse the next; `pending`: queued, not yet waited for.
const OneCopy = struct {
    var queued: usize = 0;
    var pending: usize = 0;
    fn queue() cuda.abi.Result {
        if (queued == 1) return cuda.abi.error_not_ready; // check() returns NotReady for it, logging nothing
        queued += 1;
        pending += 1;
        return cuda.abi.success;
    }
    fn toHost(_: ?*anyopaque, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return queue();
    }
    fn toDevice(_: u64, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return queue();
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pending = 0;
        return cuda.abi.success;
    }
};

/// A Flash Next namespace whose Cuda has just the fields its image reads, and no MTP head.
const Bare = struct {
    pub const Cuda = struct { gpa: Allocator, engine: struct { driver: *const cuda.Driver, kv8: bool = false }, head: ?*anyopaque = null, stream: cuda.Stream };
};

test "Flash Next completion: a kept state's copy out or in that fails part way drains the copies it queued" {
    var d: cuda.Driver = undefined;
    d.api.cuMemcpyDtoHAsync_v2 = OneCopy.toHost;
    d.api.cuMemcpyHtoDAsync_v2 = OneCopy.toDevice;
    d.api.cuStreamSynchronize = OneCopy.sync;
    var c: Bare.Cuda = .{ .gpa = std.testing.allocator, .engine = .{ .driver = &d }, .stream = .{ .d = &d, .handle = @ptrFromInt(331) } };
    var st: snap.Snap = .{ .buf = .{ .d = &d, .ptr = 1 << 20, .len = 1024 }, .at = 0, .history = .{ 0, 0 }, .head = false };
    var host: [words + 1024]u8 = @splat(0);
    const K = Kept(Bare);
    // the words and half the state queue a copy; the driver refuses the other half, or that range runs past the state
    for ([_]bool{ true, false }) |refused| for ([_]bool{ true, false }) |out| {
        OneCopy.queued = 0;
        OneCopy.pending = 0;
        const ranges = [_]rows.image.Range{ .{ .off = 0, .len = words + 512 }, .{ .off = words + 512, .len = if (refused) 512 else 1024 } };
        const got = if (out) K.image.gather(&c, &st, &ranges, &host) else K.image.scatter(&c, &st, &ranges, &host);
        try std.testing.expectError(if (refused) error.NotReady else error.BadKeptRange, got);
        try std.testing.expectEqual(@as(usize, 1), OneCopy.queued);
        try std.testing.expectEqual(@as(usize, 0), OneCopy.pending); // the caller's `host` may go: nothing copies on it
    };
}
