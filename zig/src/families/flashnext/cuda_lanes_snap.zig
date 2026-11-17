//! The prompt cache's copies of a Flash Next lane's state (cuda_snap.zig), called with the lane namespace's target.
const std = @import("std");
const lanes = @import("lanes");
const snap = @import("cuda_snap.zig");
const parts = @import("cuda_batch_state.zig");

/// Marks inside a chunk kept without cutting the pass (TF_FLASHNEXT_KEEP_INPASS=0: cut there), read once at load.
pub fn keepInPass() bool {
    // the startup check refused every malformed value already, so a plain default stands here
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_KEEP_INPASS", true) catch true;
}

/// The drafter ring a state can restore into for stream `s`: the stream's own tap ring and its state's marks,
/// or null when this server keeps none (no drafter, or a stream without a state).
pub fn ringTarget(c: anytype, s: *lanes.Stream) ?snap.RingInto {
    const d = c.drafter orelse return null;
    const x = d.state(s) orelse return null;
    return .{ .ptr = x.ring.ptr, .rows = d.ringRows(), .row = d.rowBytes(), .begin = &x.begin, .written = &x.written, .covered = &x.covered };
}

pub fn Snaps(comptime Cuda: type) type {
    return struct {
        /// The n-gram history (the last two tokens) at position `to` of a pass from `from` that began at `old`.
        pub fn historyAt(old: [2]i64, ids: []const u32, from: usize, to: usize) [2]i64 {
            if (to - from >= 2) return .{ ids[to - 2], ids[to - 1] };
            if (to - from == 1) return .{ old[1], ids[to - 1] };
            return old;
        }

        /// The chunk's stop: the first cache mark in [at, stop0) it cannot keep on the way; kept ones into `marks`.
        pub fn cut(self: *Cuda, s: *lanes.Stream, l: anytype, ids: []const u32, at: usize, stop0: usize, marks: *[parts.max_marks]parts.Mark) struct { stop: usize, kept: usize } {
            var n: usize = 0;
            const forms = self.engine.batch_k.forms; // TF_FLASHNEXT_KEEP_INPASS and _GDN_FUSED as the load read them
            // A state kept on the way holds a mark the pass then runs past, and a tap ring's oldest rows are
            // overwritten by the pass's own later positions: such a state carries only [stop0 - rows, mark) of
            // its ring and could never be written whole (cuda_disk.zig's `whole`). A drafting server therefore
            // stops the pass at its marks, where the ring holds exactly the mark's window.
            const inpass = forms.in_pass and self.drafter == null and self.ranks == null and l.seq.media == null and forms.gdn_fused;
            for (s.reuse.marks) |m| if (m > at and m < stop0) {
                if (inpass and n < parts.max_marks) {
                    if (snap.prepareWith(self.gpa, self.engine.driver, m, historyAt(l.seq.history, ids, at, m), l.draft != null, l.seq.heads, l.seq.kv8, ringShape(self, s))) |p| {
                        l.prepared[n] = p;
                        marks[n] = .{ .at = m, .base = p.buf.ptr };
                        n += 1;
                        continue;
                    } else |_| {} // no room for it now: the pass stops at the mark as before
                }
                return .{ .stop = m, .kept = n };
            };
            return .{ .stop = stop0, .kept = n };
        }

        /// After the pass: each kept state finished (its attention rows, and the drafter's ring when it drafts), then offered to the cache (save takes it).
        pub fn handOver(self: *Cuda, s: *lanes.Stream, l: anytype, marks: []const parts.Mark) !void {
            for (l.prepared[0..marks.len]) |p| try snap.finishWith(self.stream, &l.seq, if (l.draft) |*d| d else null, p.?, ringOf(self, s));
            try self.stream.synchronize();
            if (s.reuse.hook) |k| for (marks) |m| k.at(k.ptr, s, @intCast(m.at));
        }

        /// The kept states a chunk prepared that the cache did not take.
        pub fn dropPrepared(self: *Cuda, l: anytype) void {
            for (&l.prepared) |*p| if (p.*) |x| {
                snap.drop(self.gpa, self.stream, x);
                p.* = null;
            };
        }

        fn of(ptr: *anyopaque) *Cuda {
            return @ptrCast(@alignCast(ptr));
        }

        /// The ring a state carries for `s`: the stream's tap ring and the marks a round checks, or null
        /// when this server drafts with no ring (an MTP-only stream, or no drafter at all).
        fn ringOf(self: *Cuda, s: *lanes.Stream) ?snap.Ring {
            const d = self.drafter orelse return null;
            const st = d.state(s) orelse return null;
            const rows = d.ringRows();
            return .{ .ptr = st.ring.ptr, .rows = rows, .row = d.rowBytes(), .lo = st.low(rows), .written = st.written, .covered = st.covered };
        }

        /// The shape of that ring, for a state prepared before its rows exist: null without one.
        fn ringShape(self: *Cuda, s: *lanes.Stream) ?snap.RingShape {
            const d = self.drafter orelse return null;
            if (d.state(s) == null) return null;
            return .{ .rows = d.ringRows(), .row = d.rowBytes() };
        }

        pub fn bytes(ptr: *anyopaque, at: u32) u64 {
            const self = of(ptr);
            const own = snap.bytes(at, self.head != null, self.engine.heads, self.engine.kv8);
            // a drafting stream's state also carries its drafter's ring: the RAM store prices it too
            const d = self.drafter orelse return own;
            return own + snap.ringArea(at, d.ringRows(), d.rowBytes());
        }

        /// The owner stream's state, its prompt pass standing at `at` (a drafting stream's with the head's rows and its ring).
        pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = try self.lane(s);
            // a state its chunk kept on the way (TF_FLASHNEXT_KEEP_INPASS): the cache takes it as it is
            for (&l.prepared) |*p| if (p.*) |x| if (x.at == at) {
                p.* = null;
                return x;
            };
            if (l.seq.fed != at) return error.NotAtMark;
            // pending DeltaNet rows into the states first: a snapshot holds them as they stand
            if (l.draft) |*d| try self.ops().flushLate(&l.seq, d);
            if (self.ranks) |r| return try r.save(&l.seq, if (l.draft) |*d| d else null);
            return try snap.saveWith(self.gpa, self.engine.driver, self.stream, &l.seq, if (l.draft) |*d| d else null, ringOf(self, s));
        }

        pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = try self.lane(s);
            if (self.ranks) |r| return r.restore(&l.seq, if (l.draft) |*d| d else null, @ptrCast(@alignCast(saved)));
            // a state that carries the drafter's ring needs a ring to put it in: a plain MTP stream has none
            const st: *snap.Snap = @ptrCast(@alignCast(saved));
            try snap.restoreWith(self.stream, &l.seq, if (l.draft) |*d| d else null, st, if (st.ring) ringTarget(self, s) else null);
        }

        pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
            const self = of(ptr);
            if (self.ranks) |r| return r.dropSaved(@ptrCast(@alignCast(saved)));
            snap.drop(self.gpa, self.stream, @ptrCast(@alignCast(saved)));
        }
    };
}
