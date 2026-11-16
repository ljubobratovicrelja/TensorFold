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
/// Read once at a ring's allocation so nothing in the pass branches on the choice; `mean4` needs the
/// four-stream collapse this tree does not have yet and is refused where it would be written.
pub fn sourceFromEnv() TapSource {
    const v = std.c.getenv("TF_FLASHNEXT_TAP_SOURCE") orelse return TapSource.default;
    const text = std.mem.span(v);
    if (std.mem.eql(u8, text, "readout")) return .readout;
    if (std.mem.eql(u8, text, "normed")) return .normed;
    if (std.mem.eql(u8, text, "mean4")) return .mean4;
    return TapSource.default;
}

/// The three candidate sources for a tap row, from flashnext's four-stream hyper-connection layer.
///
/// MEASURED against the served target's weights, and the reason this is an enum rather than a
/// constant: `attn_hyper_connection.input_mix_weight_down` is [320, 10240] and `hc_norm` is
/// [10240] f32, so the stream mix is a *learned* projection over 4 x 2560 and the layer output
/// stays 4-stream wide. DFlash2's `fc` is [2560, 12800] = 5 taps x 2560, so it wants one
/// 2560-wide row a layer. SGLang's capture patch averages the four streams (`.mean(dim=-2)`);
/// that is what their target needed, and the mix here is not a uniform mean.
///
/// Which row is right is settled by acceptance length against the card's 2.8-3.3 band, not by
/// argument — see `docs/design/flashnext-dflash2-drafter.md` §4.1 and §6 step 0.
pub const TapSource = enum {
    /// The read-out's mixed row: down-projected, before the layer's norm.
    readout,
    /// That row after the norm — what the next layer's input norm reads.
    normed,
    /// The control: an unweighted mean over the four raw streams.
    mean4,

    /// Flipped in one place once step 0 has a number.
    pub const default: TapSource = .readout;
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
    try std.testing.expectEqual(TapSource.readout, TapSource.default);
    const s: Source = .{ .readout = 0x1000, .normed = 0x2000, .streams = 0x3000, .stride = 12800 };
    try std.testing.expectEqual(@as(u64, 0x1000), s.ptr(.readout).?);
    try std.testing.expectEqual(@as(u64, 0x2000), s.ptr(.normed).?);
    try std.testing.expectEqual(@as(u64, 0x3000), s.ptr(.mean4).?);
    // an absent source is null rather than a zero pointer the kernels would read
    const none: Source = .{};
    try std.testing.expect(none.ptr(.readout) == null);
}
