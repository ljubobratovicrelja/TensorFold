//! The drafter's tap ring on CUDA: a per-stream ring of the target's tapped residuals.
//!
//! A DFlash2 drafter reads five of the target's layer outputs (its config's `target_layer_ids`)
//! as its cross-attention context. Those rows are written once per forward and read by the next
//! verify round, so a stream keeps them in a ring rather than re-running the target.
//!
//! Ring semantics are qwen3_5's (`cuda_cache.zig`): the row at position p lives in slot
//! `p % rows`. A ring is `rows` rows of `n_taps * hidden` bf16 — a tap's row at tap index k is
//! the `hidden`-wide slice at `k * hidden` within the row, in the config's tap order.
//!
//! Why a fixed ring and not a per-position buffer: a verify round only ever needs the last
//! `window + block` positions, so the ring is independent of the served context. At 5 taps x
//! 2560 x 2056 rows that is 50.2 MiB a stream, 0.20 GiB at four streams, against a 262k fp8
//! stream of 4.80 GiB.
//!
//! The subtlety that costs a silent bug if missed: a kept state, a snapshot and the NVMe imprint
//! tier must carry this ring, or a restored stream drafts from a truncated ring and produces
//! plausible-but-wrong drafts at a low acceptance length with no error anywhere.

const std = @import("std");

/// Rows a ring holds a stream: the drafter's window and a verify block more.
pub fn rowsFor(window: usize, block: usize) usize {
    return window + block + 8;
}

/// A stream's ring: `rows` rows of `n_taps * hidden` bf16, and where it starts in the arena.
///
/// The buffer is a type parameter so this module's arithmetic is testable without the CUDA build
/// graph; the engine instantiates it with `cuda.DeviceBuffer`.
pub fn Ring(comptime Buf: type) type {
    return struct {
        /// Rows of `width` bf16 each; a row is `n_taps * hidden`.
        buf: Buf,
        rows: usize,
        n_taps: usize,
        hidden: usize,
        /// The tap ids this ring was built for, in slot order.
        taps: [8]u32 = @splat(0),

        const Self = @This();

        pub fn width(r: Self) usize {
            return r.n_taps * r.hidden;
        }
        pub fn bytes(r: Self) usize {
            return r.rows * r.width() * 2;
        }

        /// The slot a position's row occupies: the ring wraps, so a state that keeps p rows needs
        /// `rows >= p` or the oldest taps are overwritten before the drafter reads them.
        pub fn slot(r: Self, pos: usize) usize {
            return pos % r.rows;
        }

        /// Where tap `k`'s row for `pos` starts, in elements of the row buffer.
        pub fn offset(r: Self, pos: usize, k: usize) usize {
            return r.slot(pos) * r.width() + k * r.hidden;
        }

        /// Whether a restored state's ring covers every position the drafter will read: a ring
        /// that wrapped early drafts from the wrong rows, and there is no error to catch it later.
        pub fn covers(r: Self, from: usize, to: usize) bool {
            return to - from <= r.rows;
        }
    };
}

/// The tap source a run records: TF_FLASHNEXT_TAP_SOURCE=readout|normed|mean4, the default otherwise.
/// Read once at a ring's allocation so nothing in the pass branches on the choice beyond naming the
/// rows; the environment override is the A/B, so a run can be compared against another basis without
/// a rebuild.
pub fn sourceFromEnv() TapSource {
    const v = std.c.getenv("TF_FLASHNEXT_TAP_SOURCE") orelse return TapSource.default;
    return sourceFromName(std.mem.span(v));
}

/// The basis a name stands for; an unknown name is the default rather than an error (the override is
/// a diagnostic, and a typo must not silently become a fourth basis).
pub fn sourceFromName(text: []const u8) TapSource {
    if (std.mem.eql(u8, text, "readout")) return .readout;
    if (std.mem.eql(u8, text, "normed")) return .normed;
    if (std.mem.eql(u8, text, "mean4")) return .mean4;
    return TapSource.default;
}

/// The three candidate sources for a tap row, from flashnext's four-stream hyper-connection layer.
///
/// SETTLED, by the vendor's own capture rather than by argument: DFlash2 for Flash-Next was trained
/// on the PLAIN MEAN OF THE FOUR RAW STREAMS. incoai's `patches/qwen38-dflash2-capture.patch` for
/// SGLang captures a tap as
/// `hidden_states.unflatten(-1, (hc_count, hidden_size)).mean(dim=-2)` — no gate, no `hc_norm` — and
/// the independent vLLM port of the same checkpoint does the same. That is `mean4` below, and it is
/// the default.
///
/// The alternative this tree served first, `readout`, is a different tensor: the kernel
/// `fn_hc_dense_mix` gives `mixed[d] = (1/SS) * sum_s sigmoid(UP[r, s, d]) * NORMED[r, s, d]`, a
/// gated mean over hc-NORMALIZED streams, and `normed` is that row's input (4 x hidden wide). Both
/// are functions of flashnext's own learned hyper-connection path, which the drafter never saw:
/// `attn_hyper_connection.input_mix_weight_down` is [320, 10240] and `hc_norm` is [10240] f32, so
/// the mix is a learned projection over 4 x 2560. Feeding it to DFlash2's `fc` [2560, 12800] costs
/// acceptance — the measured symptom is ~1.5 tokens a round against the card's 2.4-3.3 band.
///
/// The recipe settles which tensor is the tap; whether serving it moves acceptance into that band is
/// still a measurement (`docs/design/flashnext-dflash2-drafter.md` §4.1b, §6 step 0), and the
/// override above is how the other two bases are re-measured without a rebuild.
pub const TapSource = enum {
    /// The read-out's mixed row: down-projected, before the layer's norm. (The old default; kept for
    /// the A/B.)
    readout,
    /// That row after the norm — what the next layer's input norm reads.
    normed,
    /// The plain mean over the four raw streams: the vendor's capture recipe, and what the drafter's
    /// `fc` was trained against.
    mean4,

    /// One flip, here. `mean4` because it is the tensor the checkpoint was trained on.
    pub const default: TapSource = .mean4;
};

/// Where the ring's rows come from in a batched pass, named once so the choice above reaches it.
pub const Source = struct {
    /// The mixed/down-projected rows the read-out leaves for [r0, r0 + n).
    readout: u64 = 0,
    /// The normed rows of the same span.
    normed: u64 = 0,
    /// The four raw streams, `hidden * 4` wide a row, when `mean4` is selected.
    streams: u64 = 0,
    /// Rows a buffer holds, for the stride between them.
    stride: usize = 0,

    pub fn ptr(s: Source, which: TapSource) ?u64 {
        return switch (which) {
            .readout => if (s.readout != 0) s.readout else null,
            .normed => if (s.normed != 0) s.normed else null,
            .mean4 => if (s.streams != 0) s.streams else null,
        };
    }
};

/// The ring's device bytes for `streams` lanes at `window` and `block`.
pub fn bytes(streams: usize, window: usize, block: usize, n_taps: usize, hidden: usize) usize {
    return streams * rowsFor(window, block) * n_taps * hidden * 2;
}

/// Where a run of a tap's rows starts in a ring, in elements: ring row `first`, tap `slot`, of a row
/// `n_taps * hidden` wide. Named once because the pass's strided `copyCols` is the one place a wrong
/// multiplier is silent: it takes BYTE addresses but its pitch and width in 16-bit values, so the
/// destination is this offset in elements, doubled once at the call, and its pitch is `n_taps * hidden`
/// values (never bytes).
pub fn slotOffset(first: usize, n_taps: usize, slot: usize, hidden: usize) usize {
    return (first * n_taps + slot) * hidden;
}

/// A bf16 value's bits, rounded to nearest even from f32: `__float2bfloat16_rn`.
fn toBf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    const lsb: u32 = (bits >> 16) & 1;
    return @truncate((bits +% 0x7fff +% lsb) >> 16);
}

/// The f32 a bf16 value's bits stands for (exact: bf16 is f32's top 16 bits).
fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// `TapSource.mean4` as arithmetic over host rows: the reference the CUDA kernel `tf_fxd2_mean4`
/// (qwen35_draft.cu) and the pass's ring copy must reproduce, and the only form of them a host test
/// can execute. Both are plain scalar work — no CUDA types, no allocation — so a test can hand them
/// synthetic four-stream rows and read the exact bf16 bits the ring would hold.
///
/// The arithmetic is the kernel's, rounding included: fp32 accumulation in stream order, one division
/// by the stream count, one round to bf16. That is what torch's `.mean(dim=-2)` does to a bf16 tensor
/// (it reduces through `at::opmath_type`, float, and scales the total), and it is why neither here
/// nor in the kernel is a term rounded to bf16 as it is added.
pub const Mean4 = struct {
    /// `dst[r, d] = bf16((1 / streams) * sum_s src[(r * streams + s) * hidden + d])`, bf16 bits in and
    /// out, `n` rows. Every row of `dst` is `hidden` wide.
    pub fn collapse(src: []const u16, dst: []u16, streams: usize, hidden: usize, n: usize) void {
        std.debug.assert(streams != 0);
        std.debug.assert(src.len >= n * streams * hidden);
        std.debug.assert(dst.len >= n * hidden);
        const count: f32 = @floatFromInt(streams);
        for (0..n) |r| {
            for (0..hidden) |d| {
                var acc: f32 = fromBf16(src[(r * streams) * hidden + d]);
                for (1..streams) |s| acc += fromBf16(src[(r * streams + s) * hidden + d]);
                dst[r * hidden + d] = toBf16(acc / count);
            }
        }
    }

    /// The pass's ring copy over host rows: `n` collapsed rows starting at position `pos` land tap
    /// `slot` of a ring of `rows` rows, wrapping as the pass does (a run stops where the ring does).
    pub fn place(src: []const u16, ring_buf: []u16, rows: usize, n_taps: usize, hidden: usize, slot: usize, pos: usize, n: usize) void {
        var at: usize = 0;
        while (at < n) {
            const first = (pos + at) % rows;
            const run = @min(n - at, rows - first);
            for (0..run) |r| {
                const from = (at + r) * hidden;
                const to = slotOffset(first + r, n_taps, slot, hidden);
                @memcpy(ring_buf[to .. to + hidden], src[from .. from + hidden]);
            }
            at += run;
        }
    }
};

test "the ring is a fixed cost and its slot arithmetic wraps exactly once" {
    // 5 taps x 2560, a 2048 window and an 8-row block: the block and a small slack on top.
    const rows = rowsFor(2048, 8);
    try std.testing.expectEqual(@as(usize, 2064), rows);
    const per_stream = rows * 5 * 2560 * 2;
    // 50.4 MiB a stream, 0.20 GiB at four: independent of the served context.
    try std.testing.expectEqual(@as(usize, 52_838_400), per_stream);
    try std.testing.expectEqual(4 * per_stream, bytes(4, 2048, 8, 5, 2560));

    const R = Ring(void);
    const r: R = .{ .buf = {}, .rows = rows, .n_taps = 5, .hidden = 2560 };
    try std.testing.expectEqual(@as(usize, 0), r.slot(0));
    try std.testing.expectEqual(@as(usize, rows - 1), r.slot(rows - 1));
    try std.testing.expectEqual(@as(usize, 0), r.slot(rows));
    // tap k of position p sits k*hidden into its slot's row
    try std.testing.expectEqual(@as(usize, 0 * 12800 + 3 * 2560), r.offset(0, 3));
    try std.testing.expectEqual(@as(usize, 7 * 12800 + 0), r.offset(7, 0));
}

test "a restored ring that wrapped early is caught, not drafted from" {
    const R = Ring(void);
    const r: R = .{ .buf = {}, .rows = 2064, .n_taps = 5, .hidden = 2560 };
    try std.testing.expect(r.covers(0, 2064));
    try std.testing.expect(!r.covers(0, 2065));
    try std.testing.expect(r.covers(10_000, 12_064));
    try std.testing.expect(!r.covers(10_000, 12_065));
}

test "the tap source default is one flip, and every source names a pointer" {
    // The default moved from `readout` to `mean4` (the vendor's capture recipe): one flip, here.
    try std.testing.expectEqual(TapSource.mean4, TapSource.default);
    const s: Source = .{ .readout = 0x1000, .normed = 0x2000, .streams = 0x3000, .stride = 12800 };
    try std.testing.expectEqual(@as(u64, 0x1000), s.ptr(.readout).?);
    try std.testing.expectEqual(@as(u64, 0x2000), s.ptr(.normed).?);
    try std.testing.expectEqual(@as(u64, 0x3000), s.ptr(.mean4).?);
    // an absent source is null rather than a zero pointer the kernels would read
    const none: Source = .{};
    try std.testing.expect(none.ptr(.readout) == null);
}

test "the override still names all three bases, and an unknown name is the default" {
    // TF_FLASHNEXT_TAP_SOURCE is the A/B and must survive the flip: readout and normed still select
    // the mixed and normed rows, mean4 the raw streams. An unmatched name is the default — which is
    // `mean4` now, so the near-misses below are checked for NOT becoming the other two bases.
    try std.testing.expectEqual(TapSource.readout, sourceFromName("readout"));
    try std.testing.expectEqual(TapSource.normed, sourceFromName("normed"));
    try std.testing.expectEqual(TapSource.mean4, sourceFromName("mean4"));
    try std.testing.expectEqual(TapSource.default, sourceFromName("mean4 "));
    try std.testing.expectEqual(TapSource.default, sourceFromName(""));
    try std.testing.expect(sourceFromName("Readout") != .readout);
    try std.testing.expect(sourceFromName("normd") != .normed);
}

test "mean4: a tap row is the plain mean of the four raw streams, in exact bf16 bits" {
    // Hidden 4, four streams, two rows. Every mean below is exact in bf16, so the expected bytes are
    // the arithmetic's and not a rounding's: the test fails if the collapse averages the wrong
    // elements, in the wrong order, or through a bf16 step.
    const hidden = 4;
    const streams = 4;
    // row 0: 1 2 3 4 | 2 4 6 8 | 4 6 8 10 | 8 8 8 8   -> 3.75 5.0 6.25 7.5
    // row 1: -1 .5 1.5 -2, the same in all four     -> -1 .5 1.5 -2
    const src = [_]u16{
        0x3F80, 0x4000, 0x4040, 0x4080,
        0x4000, 0x4080, 0x40C0, 0x4100,
        0x4080, 0x40C0, 0x4100, 0x4120,
        0x4100, 0x4100, 0x4100, 0x4100,
        0xBF80, 0x3F00, 0x3FC0, 0xC000,
        0xBF80, 0x3F00, 0x3FC0, 0xC000,
        0xBF80, 0x3F00, 0x3FC0, 0xC000,
        0xBF80, 0x3F00, 0x3FC0, 0xC000,
    };
    var mean: [2 * hidden]u16 = @splat(0);
    Mean4.collapse(&src, &mean, streams, hidden, 2);
    const want = [_]u16{ 0x4070, 0x40A0, 0x40C8, 0x40F0, 0xBF80, 0x3F00, 0x3FC0, 0xC000 };
    try std.testing.expectEqualSlices(u16, &want, &mean);
}

test "mean4: the total divides in fp32 and rounds once, to nearest even" {
    // 1.0 + 1.0 + 1.0 + 1.0078125 = 4.0078125, / 4 = 1.001953125: exactly halfway between 1.0 and
    // 1.00390625, so round-to-nearest-EVEN gives 1.0 (0x3F80), not the neighbour. A bf16 add step
    // would land 1.0078125's term first and could not produce this bit; a round-half-up would give
    // 0x3F81.
    const src = [_]u16{ 0x3F80, 0x3F80, 0x3F80, 0x3F81 };
    var mean: [1]u16 = @splat(0);
    Mean4.collapse(&src, &mean, 4, 1, 1);
    try std.testing.expectEqual(@as(u16, 0x3F80), mean[0]);
}

test "mean4: a tap's rows land in its own slot, at the ring's wrap, and nowhere else" {
    // Two taps, three ring rows, hidden 4: a row of the ring is 8 values, tap 0 at 0..3 and tap 1 at
    // 4..7. Two collapsed rows at position 2 fill ring row 2 and then wrap to row 0. Every other
    // value of the ring keeps its sentinel, which is what catches a contiguous copy (it would land
    // two rows' worth of taps in one slot) or a byte/element mix-up in the pitch.
    const hidden = 4;
    const n_taps = 2;
    const rows = 3;
    const sentinel: u16 = 0x7FFF;
    const src = [_]u16{ 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x4000, 0x4000, 0x4000, 0x4000 };
    var ring: [rows * n_taps * hidden]u16 = @splat(sentinel);
    Mean4.place(&src, &ring, rows, n_taps, hidden, 0, 2, 2);

    for (0..ring.len) |i| {
        const row = i / (n_taps * hidden);
        const in_row = i % (n_taps * hidden);
        // position 2 lands in ring row 2; position 3 wraps to ring row 0; every other cell is untouched
        const want: u16 = if (in_row >= hidden) sentinel else if (row == 2) 0x3F80 else if (row == 0) 0x4000 else sentinel;
        if (ring[i] != want) return error.TestUnexpectedResult;
    }

    // The same rows into tap 1 sit one `hidden` further into each slot: tap 1 of ring row 0 holds
    // what tap 0 of ring row 0 holds, and tap 1 of ring row 2 what tap 0 of ring row 2 holds.
    var other: [rows * n_taps * hidden]u16 = @splat(sentinel);
    Mean4.place(&src, &other, rows, n_taps, hidden, 1, 2, 2);
    try std.testing.expectEqualSlices(u16, ring[0..hidden], other[hidden .. 2 * hidden]);
    const at2 = 2 * n_taps * hidden;
    try std.testing.expectEqualSlices(u16, ring[at2 .. at2 + hidden], other[at2 + hidden .. at2 + 2 * hidden]);
    // Ring row 1 holds neither position's row: one tap's slot never reaches into the other's, and the
    // unused rows are untouched.
    for (0..n_taps * hidden) |i| try std.testing.expectEqual(@as(u16, sentinel), other[1 * n_taps * hidden + i]);

    // And `slotOffset` is the expression the pass's copyCols call uses, in 16-bit values: the ring
    // row of tap k of ring row r is r * n_taps + k rows into the buffer.
    const R = Ring(void);
    const r: R = .{ .buf = {}, .rows = rows, .n_taps = n_taps, .hidden = hidden };
    for (0..rows) |p| for (0..n_taps) |k|
        try std.testing.expectEqual(r.offset(p, k), slotOffset(p, n_taps, k, hidden));
}
