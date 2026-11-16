//! DFlash2 in flashnext's serving path: a tap ring a stream, and one block round a stream.
//!
//! flashnext's batched pass taps five of the target's layers into a per-stream ring
//! (`cuda_tap_ring.zig`, written by `Pass.tapRingWrite`), and this module turns that ring into the
//! prefix `cuda_draft2_fwd.zig`'s round cross-attends to. It is the flashnext half of the contract in
//! `core/lanes/drafter.zig`: `taps`, `facts`, `open`, `absorb`, `hold`, `held`, `release`.
//!
//! Three things about the seam are worth stating, because each one is a way to draft plausible
//! tokens from the wrong rows:
//!
//! * **The ring is per stream, and the pass writes one ring.** `Hooks.tap_ring` is a single device
//!   pointer, so a pass that carries lanes (a shared verify, a shared round) has no per-lane ring to
//!   write into and `tapRingWrite` refuses it (`error.TapRingSharedRound`). A stream's rows therefore
//!   arrive from its own single-stream passes; `absorb` refuses a buffer that is not the lane's own
//!   ring (`error.TapRingNotPerLane`), and `hold` refuses a prefix whose rows that ring never got
//!   (`error.DraftTapNotWritten`).
//!
//! * **A round's prefix is a span, not a buffer.** A round at anchor `a` reads positions
//!   `[max(0, a + 1 - window), a]` — the same span `cuda_draft2_check.zig` builds out of a capture.
//!   The ring is circular (`slot = position % rows`), so that span is at most two contiguous runs;
//!   when it is one, the ring's own rows are handed to the round in place, and when it is two they
//!   are copied into the stream's prefix buffer, `52.4 MB` a round at the served window (about 20 µs
//!   of device copy against a round's milliseconds).
//!
//! * **Every round rewrites the drafter's own k/v ring.** `cuda_draft2_fwd.zig`'s `prefix` folds,
//!   projects, rotates and stores the whole prefix every round, so a round depends on nothing but the
//!   tap rows its span names — that is what makes a skipped or a rolled-back round harmless here.
//!
//! A stream's ring is 50.4 MiB at the served window and is declared in `facts.stream_bytes`, so
//! admission prices it before a request is admitted, exactly as it prices the target's caches.

const std = @import("std");

/// A monotonic reading for the throwaway `TF_TAP_TIME` bisect.
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @intCast(@max(0, ts.sec) * std.time.ns_per_s + ts.nsec);
}
const cuda = @import("cuda");
const lanes = @import("lanes");
const batch = @import("cuda_batch_state.zig");
const tapring = @import("cuda_tap_ring.zig");
const config = @import("config.zig");
const fwd = @import("cuda_draft2_fwd.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;

/// The drafter's block: the anchor's row and `block - 1` proposals beside it.
pub const block = fwd.block;
/// Proposals a round holds, and the rows a verify round carries beside its anchor.
pub const depth = fwd.block - 1;
/// The window a served DFlash2 checkpoint declares (`sliding_window`), for the ring arithmetic before
/// a drafter is open. A loaded drafter's own `c.window` is what its ring is allocated to.
pub const default_window: usize = 2048;
/// Lanes one `hold` takes at most: a round's streams (`cuda_lanes.zig`'s `round_streams`).
pub const max_round: usize = 32;

/// A round's prefix: positions [lo, lo + len), ending on the position *before* the anchor's own row.
pub const Span = struct {
    lo: u64,
    len: usize,

    pub fn eql(a: Span, b: Span) bool {
        return a.lo == b.lo and a.len == b.len;
    }
};

/// The prefix a round whose anchor sits at `anchor` reads: the drafter's window of positions, or
/// everything up to the anchor when the anchor is younger than the window.
pub fn span(anchor: u64, window: usize) Span {
    // The block's row 0 is the anchor, and it is embedded and rotated at the anchor's own position.
    // A context that ended there too would put the same position in the keys twice and phase every
    // context key one step late, which is what the vendor's `position_ids[:, start-ctx : start]`
    // avoids: the context is the committed tokens up to the one before the anchor.
    const have: u64 = anchor;
    const len: usize = @intCast(@min(@as(u64, @intCast(window)), have));
    return .{ .lo = anchor - @as(u64, @intCast(len)), .len = len };
}

/// One contiguous run of a span inside a ring: `n` rows from slot `at`.
pub const Run = struct {
    at: usize,
    n: usize,

    pub fn eql(a: Run, b: Run) bool {
        return a.at == b.at and a.n == b.n;
    }
};

/// The ring runs a span [lo, lo + len) occupies, in position order: one when the span does not wrap,
/// two when it does (the second starts at slot 0). A wrap is the common case at the served window —
/// a 2,064-row ring and a 2,048-position span — so the copy that joins the two runs is the case a
/// round pays for, not the exception.
pub fn runs(lo: u64, len: usize, rows: usize) [2]Run {
    const first: usize = @intCast(lo % rows);
    const head = @min(len, rows - first);
    return .{ .{ .at = first, .n = head }, .{ .at = 0, .n = len - head } };
}

/// The longest prefix of `proposed` that is `posterior`'s own tokens: the length of the match, not a
/// count of matches anywhere in the block (`cuda_draft2_check.zig`'s rule, `reference.py`'s before it).
pub fn acceptance(proposed: []const u32, posterior: []const u32) usize {
    var n: usize = 0;
    while (n < proposed.len and n < posterior.len) : (n += 1) {
        if (proposed[n] != posterior[n]) break;
    }
    return n;
}

/// Positions a round moves the stream: the anchor's own token plus every draft the target accepted.
/// A round that lands nothing still advances by the anchor, which is why a drafter's mean advance can
/// never read below 1.
pub fn advance(proposed: []const u32, posterior: []const u32) u32 {
    return 1 + @as(u32, @intCast(acceptance(proposed, posterior)));
}

// -----------------------------------------------------------------------------------------------
// What a round cannot be served, named once so a test can pin the list
// -----------------------------------------------------------------------------------------------

/// What the engine is, as far as attaching a drafter goes.
pub const Opening = struct {
    /// --drafter-bits: the drafter's quantization, 4 only (as qwen3_5's DFlash2).
    bits: u8,
    /// --tp: flashnext's ranks split every pass, and a tap ring is one stream's rows on one device.
    tp: usize = 1,
    /// TF_FLASHNEXT_KV_PAGES: a paged lane's rows are runs of a shared pool, not a ring of its own.
    paged: bool = false,
    /// Streams already open: their rings would have to exist before their first pass.
    streams: usize = 0,
    /// The MTP head. flashnext's window verify rewinds through `mtp.Draft`, so a drafted window of
    /// more than one row needs it; DFlash2 asks for seven.
    head: bool = true,
    /// The head's bf16 tiles: the drafter's own `tf_fxd2_head_topk` reads row-major rows.
    head_bf16: bool = true,
    /// The embedding table, dense: a round's block starts from the target's rows.
    embed_dense: bool = true,
};

/// The reason a drafter cannot attach to `o`, or null.
pub fn refusal(o: Opening) ?anyerror {
    if (o.bits != 4) return error.DrafterBitsUnsupported;
    if (o.tp != 1) return error.Draft2TpUnsupported;
    if (o.paged) return error.Draft2PagedKvUnsupported;
    if (o.streams != 0) return error.TapsAfterStreams;
    if (!o.head) return error.Draft2NeedsMtpHead;
    if (!o.head_bf16) return error.Draft2HeadNotBf16;
    if (!o.embed_dense) return error.Draft2EmbedNotDense;
    return null;
}

/// Why the drafter's taps cannot be the target's: a layer past the checkpoint's last one.
pub fn tapsRefusal(taps: []const u32, layers: usize) ?anyerror {
    if (taps.len == 0) return error.Draft2NoTaps;
    if (taps.len > 8) return error.Draft2TapCountUnsupported;
    for (taps) |t| if (t >= layers) return error.Draft2TapPastLayers;
    return null;
}

/// Why a lane's round cannot be drafted from its ring: the span, the ring's coverage, and the tokens
/// the drafter was handed. `low`/`written` are the positions the ring holds; `covered` is the end of
/// what `absorb` was given.
pub fn roundRefusal(sp: Span, low: u64, written: u64, covered: u64, anchor: u64, rows: usize) ?anyerror {
    if (sp.len == 0) return error.DraftNoContext;
    if (sp.len > rows) return error.DraftWindowPastRing;
    // The anchor's own row may be missing and that is the normal first case: the first token is
    // sampled from the prefill's last row, so position `anchor` has not been through a pass yet. What
    // must exist is the *context* up to `anchor - 1`; the round clips its span to `written`.
    if (anchor > written) return error.DraftTapNotWritten;
    if (sp.lo < low) return error.DraftTapNotCovered;
    // same as the ring: the context runs to `anchor - 1`, so a first round's absorb legitimately
    // stops one row short of the anchor's own position.
    if (anchor > covered) return error.DraftAbsorbMissing;
    return null;
}

/// Why a round's lanes cannot share one `hold`: their prefixes must be the same span. A lane at
/// another position read another stream's rows out of a pass that tapped one ring, and drafting from
/// those would be wrong in a way no acceptance rate would name.
pub fn sharedRefusal(spans: []const Span) ?anyerror {
    if (spans.len == 0) return error.DraftNoLanes;
    for (spans[1..]) |sp| if (sp.lo != spans[0].lo or sp.len != spans[0].len) return error.DraftSharedSpanMismatch;
    return null;
}

// -----------------------------------------------------------------------------------------------
// A stream's state
// -----------------------------------------------------------------------------------------------

/// One stream's drafter: its tap ring, one round's prefix, and the drafter's own k/v ring.
pub const State = struct {
    /// `rows * n_taps * hidden` bf16: the target's tapped rows, slot = position % rows.
    ring: cuda.DeviceBuffer,
    /// `window * n_taps * hidden` bf16: one round's prefix, gathered out of `ring` when it wraps.
    ctx: cuda.DeviceBuffer,
    /// The drafter's five layers' k/v ring and the block's scratch.
    st: fwd.State,
    /// The proposals of the last round, `block - 1` of them.
    held: [depth]u32 = @splat(0),
    held_n: usize = 0,
    /// The positions `ring` holds: [low(), written). `begin` is the oldest position written in this
    /// stream's lifetime — a restored prompt writes from its restore point, so nothing behind it is in
    /// the ring — and `written` the one past its newest.
    begin: u64 = 0,
    written: u64 = 0,
    /// The end of the tokens the drafter was handed: an anchor at or past it has no token to start
    /// from, whatever the ring holds.
    covered: u64 = 0,

    pub fn deinit(s: *State) void {
        s.ring.free();
        s.ctx.free();
        s.st.deinit();
    }

    fn reset(s: *State) void {
        s.held_n = 0;
        s.begin = 0;
        s.written = 0;
        s.covered = 0;
        s.st.end = 0;
    }

    /// The oldest position the ring still holds: `rows` writes back from its newest, or the stream's
    /// first written position when the stream is younger than its ring.
    pub fn low(s: *const State, rows: usize) u64 {
        const floor = s.written -| @as(u64, @intCast(rows));
        return @max(s.begin, floor);
    }
};

/// The engine a drafter hangs off, so closing it can drop the engine's pointer to it.
pub const Host = struct { ctx: *anyopaque, detach: *const fn (*anyopaque) void };

// -----------------------------------------------------------------------------------------------
// The drafter
// -----------------------------------------------------------------------------------------------

pub const Draft2 = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    /// The engine's own stream: a round's kernels, its copies and the target's passes are one queue,
    /// so a round needs no fence of its own.
    stream: cuda.Stream,
    fwd: *fwd.Draft,
    /// The target's head, detiled out of the engine's bf16 tiles (`cuda_draft2_check.zig`). Owned
    /// here: the round's `tf_fxd2_head_topk` reads row-major rows and the engine keeps tiles.
    head: cuda.DeviceBuffer,
    host: ?Host = null,
    states: std.AutoHashMapUnmanaged(*Stream, *State) = .empty,
    /// States between streams: allocated and touched once, so no request pays the driver's stall.
    pool: std.ArrayList(*State) = .empty,
    /// One more draft's cost in ms (the round's own round, divided over its proposals), which the
    /// round loop's depth rule prices as `forward(rows) + step_ms x drafts`.
    step_ms: f64 = 1.0,

    const Self = @This();

    /// `dir`'s drafter against `cfg` (the served target's own config, which the checkpoint is validated
    /// against) and `target`'s two tensors; `head` is the detiled table, whose ownership this takes.
    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        d: *const cuda.Driver,
        stream: cuda.Stream,
        dir: []const u8,
        cfg: config.Config,
        target: fwd.Target,
        head: cuda.DeviceBuffer,
    ) !*Self {
        // the detiled head is this drafter's from the first line, whatever refuses below: the caller
        // hands it over and never frees it again
        var owned = head;
        errdefer owned.free();
        if (target.embed == 0 or target.head == 0) return error.TargetTensorsMissing;
        const self = try gpa.create(Self);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = io, .d = d, .stream = stream, .fwd = undefined, .head = owned };
        self.fwd = try fwd.Draft.open(gpa, io, d, stream, dir, cfg, target);
        errdefer self.fwd.deinit();
        self.fwd.target.head = self.head.ptr;
        try self.measure();
        errdefer {
            while (self.pool.pop()) |st| {
                st.deinit();
                gpa.destroy(st);
            }
        }
        try self.pool.append(gpa, try self.create());
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.host) |h| h.detach(h.ctx);
        var it = self.states.valueIterator();
        while (it.next()) |st| {
            st.*.deinit();
            self.gpa.destroy(st.*);
        }
        self.states.deinit(self.gpa);
        for (self.pool.items) |st| {
            st.deinit();
            self.gpa.destroy(st);
        }
        self.pool.deinit(self.gpa);
        self.fwd.deinit();
        self.head.free();
        self.gpa.destroy(self);
    }

    // ---- the dimensions a ring and a round are sized by -----------------------------------------

    /// The drafter's window: its config's `sliding_window`, the longest prefix a round reads.
    pub fn window(self: *const Self) usize {
        return self.fwd.c.window;
    }

    /// Rows a stream's tap ring holds (`cuda_tap_ring.zig`): the window, a verify block and a slack.
    pub fn ringRows(self: *const Self) usize {
        return tapring.rowsFor(self.fwd.c.window, self.fwd.c.block);
    }

    /// Bytes one tap row takes: the drafter's five taps of the target's hidden width, bf16.
    pub fn rowBytes(self: *const Self) usize {
        return self.fwd.c.n_taps * fwd.hidden * 2;
    }

    pub fn ringBytes(self: *const Self) usize {
        return tapring.bytes(1, self.fwd.c.window, self.fwd.c.block, self.fwd.c.n_taps, fwd.hidden);
    }

    fn ctxBytes(self: *const Self) usize {
        return self.fwd.c.window * self.rowBytes();
    }

    /// Device bytes a stream takes: its ring, its prefix and the drafter's own k/v ring and scratch.
    pub fn streamBytes(self: *const Self) !usize {
        return std.math.add(usize, try std.math.add(usize, self.ringBytes(), self.ctxBytes()), try fwd.Draft.stateBytes(self.fwd.c));
    }

    /// The target's layers this drafter reads, in its own tap order.
    pub fn taps(self: *const Self) []const u32 {
        return self.fwd.c.taps[0..self.fwd.c.n_taps];
    }

    // ---- the Drafter contract --------------------------------------------------------------------

    pub fn drafter(self: *Self) dr.Drafter {
        return .{ .kind = "DFlash2", .ptr = self, .vtable = &.{
            .taps = tapsFn,
            .facts = factsFn,
            .open = openFn,
            .absorb = absorbFn,
            .hold = holdFn,
            .held = heldFn,
            .release = releaseFn,
        } };
    }

    fn cast(p: *anyopaque) *Self {
        return @ptrCast(@alignCast(p));
    }

    fn tapsFn(p: *anyopaque) []const u32 {
        return cast(p).taps();
    }

    fn factsFn(p: *anyopaque) dr.Facts {
        const self = cast(p);
        // batched is false on purpose: a round's lanes each draft on their own state, one round at a
        // time, because a round rewrites its stream's whole prefix — batching them would be a second
        // frame, not a wider one.
        return .{
            .depth = @intCast(depth),
            .step_ms = self.step_ms,
            .plain_guard = true,
            .batched = false,
            // saturating rather than erroring: a facts() has no error to give, and an overflow here is
            // a dimension no GPU could hold (the four buffers are each a window's rows)
            .stream_bytes = self.ringBytes() +| self.ctxBytes() +| (fwd.Draft.stateBytes(self.fwd.c) catch 0),
            .trees = false,
            .copy_match = 0,
        };
    }

    /// The stream's own ring and scratch, created once: the lanes wrapper does not call the vtable's
    /// `open`, so a lane's first pass would otherwise find no state, write no taps, and be refused at
    /// its first round with `DraftTapNotWritten`. Idempotent by design.
    pub fn ensure(self: *Self, s: *Stream) anyerror!void {
        if (self.states.contains(s)) return;
        return openFn(@ptrCast(self), s);
    }

    fn openFn(p: *anyopaque, s: *Stream) anyerror!void {
        const self = cast(p);
        if (self.states.contains(s)) return error.StreamOpen;
        const st = self.pool.pop() orelse try self.create();
        errdefer self.pool.append(self.gpa, st) catch {};
        st.reset();
        try self.states.put(self.gpa, s, st);
    }

    /// The rows a stream's ring holds, told to the drafter by the pass that wrote them. Called by the
    /// engine after a pass (`cuda_lanes.zig`), before the round loop asks for a draft.
    pub fn note(self: *Self, s: *Stream, from: u64, to: u64) void {
        const st = self.states.get(s) orelse return;
        if (to <= from) return;
        if (st.written == 0) {
            st.begin = from;
            st.written = to;
            return;
        }
        // a later pass extends the span it wrote; a restored stream's first pass starts past 0, and
        // `begin` keeps the rows behind it out of reach (see State.low)
        st.begin = @min(st.begin, from);
        st.written = @max(st.written, to);
    }

    /// The tap hooks a pass writes a stream's ring through, or `hk` untouched when the stream is not
    /// drafting. `n_taps == 0` is what the pass reads as "no taps", so a stream without a ring adds
    /// nothing to its pass.
    pub fn hooks(self: *Self, hk: batch.Hooks, s: *Stream) batch.Hooks {
        const st = self.states.get(s) orelse return hk;
        const c = self.fwd.c;
        var out = hk;
        out.n_taps = c.n_taps;
        for (c.taps[0..c.n_taps], 0..) |t, i| out.taps[i] = t;
        out.tap_ring = st.ring.ptr;
        out.tap_rows = self.ringRows();
        out.tap_source = fwd.tap_source;
        return out;
    }

    /// A stream's ring and the positions it holds, for the backend's `features`.
    pub fn state(self: *Self, s: *Stream) ?*State {
        return self.states.get(s);
    }

    fn absorbFn(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const self = cast(p);
        for (items) |it| try self.absorbOne(it);
    }

    /// One stream's rows: the ring already holds them (the pass wrote them), so this only records how
    /// far the ring and the drafter's tokens reach, and refuses a buffer that is not this lane's ring.
    fn absorbOne(self: *Self, it: dr.Absorb) !void {
        const st = self.states.get(it.stream) orelse return error.UnknownStream;
        if (it.features.space != .device or it.features.dtype != .bf16) return error.UnsupportedFeatures;
        // the tap ring a pass writes is the lane's own; another buffer means a shared pass's rows or a
        // forgotten ring, and drafting from either would read another stream's taps
        if (it.features.buffer != st.ring.ptr) return error.TapRingNotPerLane;
        if (@as(usize, it.features.row_bytes) != self.rowBytes()) return error.TapRowWidthMismatch;
        const start = it.features.offset;
        const n: u64 = it.features.rows;
        if (n > 0) {
            const end = start + n;
            if (st.written == 0) {
                st.begin = start;
                st.written = end;
            } else {
                // a gap is a stream whose rows came from a pass that tapped no ring (a shared verify,
                // a restored prompt): the newest rows would be drafted from as if they were its own
                if (start > st.written) return error.DraftTapGap;
                st.begin = @min(st.begin, start);
                st.written = @max(st.written, end);
            }
            // the last row's follow token is the round's anchor
            st.covered = @max(st.covered, end);
        }
    }

    fn holdFn(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const self = cast(p);
        if (items.len == 0) return;
        if (items.len > max_round) return error.DraftRoundTooWide;
        var spans: [max_round]Span = undefined;
        for (items, 0..) |it, i| {
            if (it.position == 0) return error.DraftPositionTooSmall;
            if (it.lanes != null) return error.DraftTreesUnsupported;
            spans[i] = span(it.position - 1, self.fwd.c.window);
        }
        if (sharedRefusal(spans[0..items.len])) |e| return e;
        for (items, spans[0..items.len]) |it, sp| try self.round(it, sp);
    }

    /// One stream's block: the prefix from its ring, then the drafter's round.
    fn round(self: *Self, it: dr.Hold, sp: Span) !void {
        const st = self.states.get(it.stream) orelse return error.UnknownStream;
        const anchor = it.position - 1;
        const rows = self.ringRows();
        // the context never includes the anchor's row, so an unwritten anchor is not a reason to clip;
        // a ring that holds less than the span is refused by the checks below, not silently shortened
        const ctx_span = sp;
        if (roundRefusal(ctx_span, st.low(rows), st.written, st.covered, anchor, rows)) |e| {
            std.log.err("draft2 round refused ({s}): span [{d},{d}) low {d} written {d} covered {d} anchor {d} rows {d}", .{
                @errorName(e), sp.lo, sp.lo + sp.len, st.low(rows), st.written, st.covered, anchor, rows,
            });
            return e;
        }
        var t0: u64 = 0;
        if (std.c.getenv("TF_TAP_TIME") != null) t0 = nowNs();
        const ctx = try self.gather(st, ctx_span);
        var t1 = t0;
        if (t0 != 0) t1 = nowNs();
        // A round rewrites its whole prefix into the drafter's k/v ring (`cuda_draft2_fwd.zig`'s
        // `prefix`), so the ring is contiguous by construction and the drafter's own "no gap" guard —
        // which is about a ring carried between rounds — must not see the rounds this stream skipped
        // (a copy round, a plain round, a rollback): `roundRefusal` above is the check that matters.
        st.st.end = 0;
        var out: [depth]u32 = @splat(0);
        try self.fwd.draft(&st.st, .{ .ctx = ctx, .ctx_len = ctx_span.len, .anchor = it.pending, .position = anchor }, out[0..depth]);
        if (t0 != 0) {
            const t2 = nowNs();
            std.log.info("draft2 round: {d:.1} ms = gather {d:.1} (ctx {d} rows) + draft {d:.1} (held {d})", .{
                @as(f64, @floatFromInt(t2 - t0)) / 1e6,
                @as(f64, @floatFromInt(t1 - t0)) / 1e6,
                ctx_span.len,
                @as(f64, @floatFromInt(t2 - t1)) / 1e6,
                depth,
            });
        }
        st.held = out;
        st.held_n = depth;
    }

    /// The round's prefix as one contiguous `[len, n_taps * hidden]` span: the ring's own rows when the
    /// span does not wrap, two copies into the stream's buffer when it does.
    fn gather(self: *Self, st: *State, sp: Span) !u64 {
        const row_bytes = self.rowBytes();
        const r = runs(sp.lo, sp.len, self.ringRows());
        if (r[1].n == 0) return st.ring.ptr + r[0].at * row_bytes;
        const handle = self.stream.handle;
        try st.ctx.copyFrom(0, st.ring.ptr + r[0].at * row_bytes, r[0].n * row_bytes, handle);
        try st.ctx.copyFrom(r[0].n * row_bytes, st.ring.ptr, r[1].n * row_bytes, handle);
        return st.ctx.ptr;
    }

    fn heldFn(p: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const self = cast(p);
        for (streams, out) |s, o| {
            const st = self.states.get(s) orelse return error.UnknownStream;
            if (o.len > st.held_n) return error.TooFewDrafts;
            @memcpy(o, st.held[0..o.len]);
        }
    }

    fn releaseFn(p: *anyopaque, s: *Stream) void {
        const self = cast(p);
        if (self.states.fetchRemove(s)) |kv| {
            kv.value.reset();
            self.pool.append(self.gpa, kv.value) catch {
                kv.value.deinit();
                self.gpa.destroy(kv.value);
            };
        }
    }

    // ---- allocation ------------------------------------------------------------------------------

    /// `n` states ready for the streams the server admits.
    pub fn reserve(self: *Self, n: u32) !void {
        const want: usize = n;
        try self.pool.ensureTotalCapacity(self.gpa, want);
        while (self.pool.items.len < want) self.pool.appendAssumeCapacity(try self.create());
    }

    fn create(self: *Self) !*State {
        const st = try self.gpa.create(State);
        errdefer self.gpa.destroy(st);
        var ring = try cuda.DeviceBuffer.alloc(self.d, self.ringBytes());
        errdefer ring.free();
        var ctx = try cuda.DeviceBuffer.alloc(self.d, self.ctxBytes());
        errdefer ctx.free();
        var inner = try self.fwd.create();
        errdefer inner.deinit();
        // zeroed: a slot a pass has not written yet reads as a key of 0 rather than the last stream's
        try ring.fill8(0, self.stream.handle);
        try ctx.fill8(0, self.stream.handle);
        try self.stream.synchronize();
        st.* = .{ .ring = ring, .ctx = ctx, .st = inner };
        return st;
    }

    /// Time one real round — a full window's prefix against zeroed taps — so the depth rule prices a
    /// draft instead of a guess. The tokens are not read; nothing here is a correctness check.
    fn measure(self: *Self) !void {
        const st = try self.create();
        defer {
            st.deinit();
            self.gpa.destroy(st);
        }
        const w = self.fwd.c.window;
        var out: [depth]u32 = @splat(0);
        var quickest: f64 = std.math.inf(f64);
        for (0..8) |i| {
            st.st.end = 0;
            const t0 = std.Io.Clock.awake.now(self.io);
            // a full window of context needs a block at least that far into the sequence: the prefix is
            // [position - ctx_len, position), so a block at position `w` is the youngest one that can
            // carry the whole window
            try self.fwd.draft(&st.st, .{ .ctx = st.ctx.ptr, .ctx_len = w, .anchor = 0, .position = @intCast(w) }, out[0..depth]);
            const ns: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds());
            // the first rounds pay the driver's stalls, which no later round does
            if (i >= 2) quickest = @min(quickest, ns / 1e6);
        }
        if (std.math.isFinite(quickest)) self.step_ms = quickest / @as(f64, @floatFromInt(depth));
    }
};

// -----------------------------------------------------------------------------------------------
// Tests: the span arithmetic, the refusal list, and what a round advances. Nothing here touches a
// device — a round's GPU half is `cuda_draft2_fwd.zig`'s, measured by `cuda_draft2_check.zig`.
// -----------------------------------------------------------------------------------------------

test "a round's prefix span is the drafter's window of positions, ending on the anchor" {
    // a stream's first token: one row, the anchor itself
    try std.testing.expect(Span.eql(.{ .lo = 0, .len = 1 }, span(0, 2048)));
    try std.testing.expect(Span.eql(.{ .lo = 0, .len = 2 }, span(1, 2048)));
    // a prefix shorter than the window reads all of it
    try std.testing.expect(Span.eql(.{ .lo = 0, .len = 2048 }, span(2047, 2048)));
    // past the window, the span slides one position a token
    try std.testing.expect(Span.eql(.{ .lo = 1, .len = 2048 }, span(2048, 2048)));
    try std.testing.expect(Span.eql(.{ .lo = 4097, .len = 2048 }, span(6144, 2048)));
    // the span always ends on the anchor, and never outruns the window
    for ([_]u64{ 0, 1, 7, 2047, 2048, 2049, 100_000 }) |anchor| {
        const sp = span(anchor, 2048);
        try std.testing.expectEqual(anchor, sp.lo + sp.len - 1);
        try std.testing.expect(sp.len >= 1 and sp.len <= 2048);
    }
}

test "a span is one ring run when it does not wrap and two when it does, and the runs tile it" {
    const rows = tapring.rowsFor(2048, 8);
    try std.testing.expectEqual(@as(usize, 2064), rows);
    // the first positions: no wrap, one run at the span's own slots
    const early = runs(0, 2048, rows);
    try std.testing.expect(Run.eql(.{ .at = 0, .n = 2048 }, early[0]));
    try std.testing.expectEqual(@as(usize, 0), early[1].n);
    // a span that starts past `rows - len` wraps: the tail, then the ring's head
    const sp = span(3000, 2048);
    try std.testing.expectEqual(@as(u64, 953), sp.lo);
    const wrapped = runs(sp.lo, sp.len, rows);
    try std.testing.expect(Run.eql(.{ .at = 953, .n = 1111 }, wrapped[0]));
    try std.testing.expect(Run.eql(.{ .at = 0, .n = 937 }, wrapped[1]));
    try std.testing.expectEqual(sp.len, wrapped[0].n + wrapped[1].n);
    // a span exactly filling the ring is one run and lands on the slot the next position uses
    const full = runs(0, rows, rows);
    try std.testing.expectEqual(@as(usize, rows), full[0].n);
    try std.testing.expectEqual(@as(usize, 0), full[1].n);
    // every position's span tiles the ring, and its first run starts at the span's own slot
    var anchor: u64 = 0;
    while (anchor < 20_000) : (anchor += 37) {
        const s = span(anchor, 2048);
        const r = runs(s.lo, s.len, rows);
        try std.testing.expectEqual(@as(usize, @intCast(s.lo % rows)), r[0].at);
        try std.testing.expectEqual(s.len, r[0].n + r[1].n);
        try std.testing.expect(r[0].at + r[0].n <= rows);
        try std.testing.expect(r[1].n == 0 or r[1].at == 0);
    }
}

test "a drafter is refused, by name, wherever flashnext cannot serve it" {
    const ok: Opening = .{ .bits = 4 };
    try std.testing.expect(refusal(ok) == null);
    try std.testing.expectEqual(@as(?anyerror, error.DrafterBitsUnsupported), refusal(.{ .bits = 0 }));
    try std.testing.expectEqual(@as(?anyerror, error.DrafterBitsUnsupported), refusal(.{ .bits = 8 }));
    // --tp 2 splits every pass across two devices; a tap ring is one stream's rows on one
    try std.testing.expectEqual(@as(?anyerror, error.Draft2TpUnsupported), refusal(.{ .bits = 4, .tp = 2 }));
    // a paged lane's rows are runs of a shared pool, not a ring of its own
    try std.testing.expectEqual(@as(?anyerror, error.Draft2PagedKvUnsupported), refusal(.{ .bits = 4, .paged = true }));
    // a ring has to exist before the stream's first pass
    try std.testing.expectEqual(@as(?anyerror, error.TapsAfterStreams), refusal(.{ .bits = 4, .streams = 1 }));
    // seven drafted rows need the MTP head's rewindable window verify
    try std.testing.expectEqual(@as(?anyerror, error.Draft2NeedsMtpHead), refusal(.{ .bits = 4, .head = false }));
    try std.testing.expectEqual(@as(?anyerror, error.Draft2HeadNotBf16), refusal(.{ .bits = 4, .head_bf16 = false }));
    try std.testing.expectEqual(@as(?anyerror, error.Draft2EmbedNotDense), refusal(.{ .bits = 4, .embed_dense = false }));

    // the taps are the target's: none, too many, or one past its last layer
    const taps = [_]u32{ 5, 15, 25, 35, 45 };
    try std.testing.expect(tapsRefusal(&taps, 48) == null);
    try std.testing.expectEqual(@as(?anyerror, error.Draft2NoTaps), tapsRefusal(&.{}, 48));
    try std.testing.expectEqual(@as(?anyerror, error.Draft2TapPastLayers), tapsRefusal(&taps, 45));
    const nine = [_]u32{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    try std.testing.expectEqual(@as(?anyerror, error.Draft2TapCountUnsupported), tapsRefusal(&nine, 48));
}

test "a stream's ring floor is its newest rows, and never behind the first row it wrote" {
    var st: State = undefined;
    st.begin = 0;
    st.written = 0;
    try std.testing.expectEqual(@as(u64, 0), st.low(2064)); // nothing written yet
    st.written = 100;
    try std.testing.expectEqual(@as(u64, 0), st.low(2064)); // younger than its ring: every row it wrote
    st.written = 5000;
    try std.testing.expectEqual(@as(u64, 5000 - 2064), st.low(2064)); // the newest ring's worth
    // a prompt restored at 2,800 wrote from there: the rows behind it are not in the ring, whatever
    // the ring's own size would reach
    st.begin = 2800;
    st.written = 3000;
    try std.testing.expectEqual(@as(u64, 2800), st.low(2064));
}

test "a round is refused when its prefix is not in its own ring" {
    const rows = tapring.rowsFor(2048, 8);
    const sp = span(3000, 2048); // lo 953, len 2048
    // the ring holds everything from the stream's first pass to its newest
    try std.testing.expect(roundRefusal(sp, 0, 3001, 3001, 3000, rows) == null);
    // the anchor's row is past what any pass wrote: a shared verify, or none yet
    try std.testing.expectEqual(@as(?anyerror, error.DraftTapNotWritten), roundRefusal(sp, 0, 3000, 3001, 3000, rows));
    // the ring wrapped, so the oldest rows of the span are another position's
    try std.testing.expectEqual(@as(?anyerror, error.DraftTapNotCovered), roundRefusal(sp, 1000, 3001, 3001, 3000, rows));
    // the ring holds the rows but the drafter was never handed the anchor's token
    try std.testing.expectEqual(@as(?anyerror, error.DraftAbsorbMissing), roundRefusal(sp, 0, 3001, 3000, 3000, rows));
    // a span longer than the ring cannot be served at any position
    try std.testing.expectEqual(@as(?anyerror, error.DraftWindowPastRing), roundRefusal(.{ .lo = 0, .len = rows + 1 }, 0, 1 << 20, 1 << 20, 1 << 20, rows));
}

test "a shared round's lanes must read the same span, and a tree is not grown here" {
    const one = [_]Span{span(3000, 2048)};
    try std.testing.expect(sharedRefusal(&one) == null);
    const same = [_]Span{ span(3000, 2048), span(3000, 2048) };
    try std.testing.expect(sharedRefusal(&same) == null);
    // two lanes at different positions: the pass that tapped their rows wrote one ring
    const differ = [_]Span{ span(3000, 2048), span(3001, 2048) };
    try std.testing.expectEqual(@as(?anyerror, error.DraftSharedSpanMismatch), sharedRefusal(&differ));
    // two lanes of the same span, only one of them past the window: a length mismatch all the same
    const lengths = [_]Span{ span(2047, 2048), span(2048, 2048) };
    try std.testing.expectEqual(@as(?anyerror, error.DraftSharedSpanMismatch), sharedRefusal(&lengths));
    try std.testing.expectEqual(@as(?anyerror, error.DraftNoLanes), sharedRefusal(&.{}));
}

test "a round's proposals advance exactly 1 + accepted, and never past the block" {
    // nothing accepted: the anchor alone moves the stream
    try std.testing.expectEqual(@as(usize, 0), acceptance(&.{ 9, 8, 7 }, &.{ 1, 2, 3 }));
    try std.testing.expectEqual(@as(u32, 1), advance(&.{ 9, 8, 7 }, &.{ 1, 2, 3 }));
    // a matching prefix, then a miss: the match, not the count of matches
    try std.testing.expectEqual(@as(usize, 2), acceptance(&.{ 1, 2, 9, 4 }, &.{ 1, 2, 3, 4 }));
    try std.testing.expectEqual(@as(u32, 3), advance(&.{ 1, 2, 9, 4 }, &.{ 1, 2, 3, 4 }));
    // a whole block: the anchor's token and every draft
    var proposed: [depth]u32 = undefined;
    var posterior: [depth]u32 = undefined;
    for (&proposed, &posterior, 0..) |*p, *q, i| {
        p.* = @intCast(1000 + i);
        q.* = @intCast(1000 + i);
    }
    try std.testing.expectEqual(depth, acceptance(&proposed, &posterior));
    try std.testing.expectEqual(@as(u32, block), advance(&proposed, &posterior));
    // the round holds `block - 1` proposals, so a round advances by at most one block
    try std.testing.expectEqual(@as(usize, 7), depth);
    try std.testing.expectEqual(@as(usize, 8), block);
    try std.testing.expect(advance(&proposed, &posterior) <= block);
    // a proposal list and a posterior list of different lengths do not overrun each other
    try std.testing.expectEqual(@as(usize, 1), acceptance(&.{ 5, 6 }, &.{5}));
}

test "the tap ring's bytes are one stream's fixed cost, and the drafter's are accounted for" {
    const rows = tapring.rowsFor(default_window, block);
    try std.testing.expectEqual(@as(usize, 2064), rows);
    // 2064 x 5 x 2560 x 2 a ring, and a window of the same rows for the prefix a round gathers
    try std.testing.expectEqual(@as(usize, 52_838_400), tapring.bytes(1, default_window, block, 5, fwd.hidden));
    try std.testing.expectEqual(@as(usize, 52_428_800), default_window * 5 * fwd.hidden * 2);
    // the span a full ring serves is the window's, never the ring's
    try std.testing.expect(span(1_000_000, default_window).len <= rows);
}
