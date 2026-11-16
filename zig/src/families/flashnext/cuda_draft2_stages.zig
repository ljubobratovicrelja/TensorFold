//! The stage-by-stage capture of ONE draft round: every intermediate a DFlash2 forward produces,
//! read back to the host with its own timestamp, plus the `.npy` / JSON format an outside reference
//! implementation is diffed against.
//!
//! Off unless a caller installs a collector (`Draft.stages`), and inert otherwise: a serving round
//! branches on a null pointer at each stage boundary and copies nothing.
//!
//! The rule the timings live by: **a stage's bytes are read after the stream has been synchronised,
//! and its timestamp is taken at that moment**. A dump recorded on enqueue would measure the
//! launches, not the work, and would put a 93 ms head at 0.1 ms. So each `take` is
//! `synchronize -> timestamp -> download`, and the downloaded bytes are the round's own — the
//! buffers `cuda_draft2_fwd.zig` wrote, never a recomputation.
//!
//! One exception, marked in the JSON as `"measured":"host"`: the selector's lattice scores are not
//! a buffer the round leaves behind (`tf_fxd2_select` keeps `score[]` in shared memory and writes
//! only the proposals), so `cuda_draft2_check.zig` recomputes them on the host from this module's
//! own dumped rows, in the kernel's arithmetic, and hands them back through `addHost`.
const std = @import("std");
const cuda = @import("cuda");

/// A tensor's element type, as the round holds it on the device. `.npy` files are written as f32 —
/// every value here is exactly representable (bf16 widened, and u32 ids below 2^24) — and each
/// entry records the device-side type as `src_dtype`.
pub const Dtype = enum {
    bf16,
    f32,
    u32,

    pub fn bytes(t: Dtype) usize {
        return switch (t) {
            .bf16 => 2,
            .f32 => 4,
            .u32 => 4,
        };
    }

    pub fn text(t: Dtype) []const u8 {
        return switch (t) {
            .bf16 => "bf16",
            .f32 => "f32",
            .u32 => "u32",
        };
    }
};

/// Stages one round can record: the context, the block's embedding, nine a layer, the final norm,
/// the head's three tables, the selector's scores, the proposals and the context's own rows.
pub const max_entries = 2 + 5 * 9 + 1 + 3 + 2 + 1;

/// The longest stage name: `conv_attn_out_4` and its neighbours.
pub const name_max = 32;

/// One device row block of a stage: a stage is one buffer, or several concatenated along its
/// columns (`qkv_i` is q, k and v side by side).
pub const Part = struct {
    ptr: u64,
    cols: usize,
};

/// One recorded stage.
pub const Entry = struct {
    name: [name_max]u8 = @splat(0),
    name_len: usize = 0,
    rows: usize = 0,
    cols: usize = 0,
    /// The device-side element type; the `.npy` beside it is always f32.
    dtype: Dtype = .f32,
    /// Wall-clock milliseconds between the previous stage's synchronisation and this one's.
    ms: f64 = 0,
    /// The stage's bytes on the host, `rows * cols` elements of `dtype`.
    host: []u8 = &.{},
    /// `false` for a stage the host recomputed (`selector_scores`): its `ms` is host time.
    measured_gpu: bool = true,

    pub fn nameOf(e: *const Entry) []const u8 {
        return e.name[0..e.name_len];
    }
};

fn nowNs(io: std.Io) u64 {
    const t = std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
    return @intCast(@max(0, t));
}

/// A round's stage capture in flight.
pub const Collector = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    e: [max_entries]Entry = @splat(Entry{}),
    n: usize = 0,
    last_ns: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream) Collector {
        return .{ .gpa = gpa, .io = io, .d = d, .stream = stream };
    }

    pub fn deinit(k: *Collector) void {
        for (k.e[0..k.n]) |*x| if (x.host.len != 0) k.gpa.free(x.host);
        k.n = 0;
    }

    /// The zero of the timing: the round's first launch is timed from here.
    pub fn start(k: *Collector) void {
        k.last_ns = nowNs(k.io);
    }

    /// Record a stage with no bytes: the timestamp only (the context fusion, whose rows the check
    /// summarises itself because a window-wide tensor has no business in a `.npy`).
    pub fn mark(k: *Collector, name: []const u8) !void {
        try k.stream.synchronize();
        const t = nowNs(k.io);
        const x = try k.slot(name);
        x.ms = @as(f64, @floatFromInt(t -| k.last_ns)) / 1e6;
        x.measured_gpu = true;
        k.last_ns = t;
    }

    /// Record a stage: synchronise (so the bytes are the work's, not the enqueue's), timestamp, then
    /// read `parts` back into one `[rows, cols]` host tensor.
    pub fn take(k: *Collector, name: []const u8, parts: []const Part, rows: usize, cols: usize, dtype: Dtype) !void {
        if (parts.len == 0) return error.StageNoParts;
        var sum: usize = 0;
        for (parts) |p| sum += p.cols;
        if (sum != cols or rows == 0 or cols == 0) return error.StageShapeMismatch;

        try k.stream.synchronize();
        const t = nowNs(k.io);
        const x = try k.slot(name);
        x.ms = @as(f64, @floatFromInt(t -| k.last_ns)) / 1e6;
        x.measured_gpu = true;
        k.last_ns = t;
        x.rows = rows;
        x.cols = cols;
        x.dtype = dtype;

        const eb = dtype.bytes();
        x.host = try k.gpa.alloc(u8, try std.math.mul(usize, rows * cols, eb));
        var at: usize = 0;
        for (parts) |p| {
            // a part is `rows` contiguous device rows; the staging buffer is the part's own bytes
            const tmp = try k.gpa.alloc(u8, try std.math.mul(usize, rows * p.cols, eb));
            defer k.gpa.free(tmp);
            try cuda.memory.download(k.d, @intCast(p.ptr), tmp);
            for (0..rows) |r| {
                const dst = (r * cols + at) * eb;
                const src = r * p.cols * eb;
                @memcpy(x.host[dst..][0 .. p.cols * eb], tmp[src..][0 .. p.cols * eb]);
            }
            at += p.cols;
        }
    }

    /// A stage the host recomputed: the bytes and the host's own time for them.
    pub fn addHost(k: *Collector, name: []const u8, bytes: []const u8, rows: usize, cols: usize, dtype: Dtype, ms: f64) !void {
        const x = try k.slot(name);
        x.ms = ms;
        x.measured_gpu = false;
        x.rows = rows;
        x.cols = cols;
        x.dtype = dtype;
        x.host = try k.gpa.dupe(u8, bytes);
    }

    fn slot(k: *Collector, name: []const u8) !*Entry {
        if (k.n == max_entries) return error.StageTooMany;
        if (name.len > name_max) return error.StageNameTooLong;
        const x = &k.e[k.n];
        k.n += 1;
        x.* = .{};
        @memcpy(x.name[0..name.len], name);
        x.name_len = name.len;
        return x;
    }

    pub fn find(k: *Collector, name: []const u8) ?*Entry {
        for (k.e[0..k.n]) |*x| if (std.mem.eql(u8, x.nameOf(), name)) return x;
        return null;
    }
};

/// One tensor's numbers: the summary a stage is compared on, and the eight values of its second
/// row (`sample`, the first row for a one-row tensor) so a whole block can be eyeballed, not only
/// its statistics.
pub const Stats = struct {
    mean: f64 = 0,
    std: f64 = 0,
    l2norm: f64 = 0,
    sample: [8]f32 = @splat(0),
    n_sample: usize = 0,
};

/// Summary statistics over the f32 widenings of `rows * cols` elements of `dtype`.
pub fn stats(host: []const u8, rows: usize, cols: usize, dtype: Dtype) Stats {
    const total: usize = rows * cols;
    var sum: f64 = 0;
    var sum2: f64 = 0;
    var l2: f64 = 0;
    var i: usize = 0;
    while (i < total) : (i += 1) {
        const v: f64 = value(host, i, dtype);
        sum += v;
        sum2 += v * v;
        l2 += v * v;
    }
    const n: f64 = @floatFromInt(total);
    const mean = sum / n;
    const var_ = @max(0.0, sum2 / n - mean * mean);
    var s: Stats = .{ .mean = mean, .std = @sqrt(var_), .l2norm = @sqrt(l2) };
    // row 1 of a block: the block's first *proposal* row, matching what the head and the selector
    // read (`block - 1` rows counted from row 1); a one-row tensor samples its only row
    const row: usize = if (rows > 1) 1 else 0;
    const take_n = @min(@as(usize, 8), cols);
    var j: usize = 0;
    while (j < take_n) : (j += 1) {
        s.sample[j] = @floatCast(value(host, row * cols + j, dtype));
    }
    s.n_sample = take_n;
    return s;
}

fn value(host: []const u8, i: usize, dtype: Dtype) f64 {
    return switch (dtype) {
        .bf16 => @floatCast(widen(bf16At(host, i))),
        .f32 => @floatCast(f32At(host, i)),
        .u32 => @floatFromInt(u32At(host, i)),
    };
}

fn u16At(b: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, b[i * 2 ..][0..2], .little);
}
fn u32At(b: []const u8, i: usize) u32 {
    return std.mem.readInt(u32, b[i * 4 ..][0..4], .little);
}
fn f32At(b: []const u8, i: usize) f32 {
    return @bitCast(u32At(b, i));
}
/// bf16 is the top 16 bits of an f32.
fn bf16At(b: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, u16At(b, i)) << 16);
}

fn widen(v: f32) f32 {
    return v;
}

/// The stage's bytes as f32, which is what its `.npy` holds.
pub fn widenAll(gpa: std.mem.Allocator, host: []const u8, n: usize, dtype: Dtype) ![]f32 {
    const out = try gpa.alloc(f32, n);
    if (dtype == .f32) {
        for (out, 0..) |*v, i| v.* = f32At(host, i);
    } else if (dtype == .bf16) {
        for (out, 0..) |*v, i| v.* = bf16At(host, i);
    } else {
        for (out, 0..) |*v, i| v.* = @floatFromInt(u32At(host, i));
    }
    return out;
}

// -----------------------------------------------------------------------------------------------
// `.npy`, written by hand: a 128-byte-aligned header and the raw little-endian f32 body.
// -----------------------------------------------------------------------------------------------

/// Write `rows x cols` (one dimension when `cols == 0`) f32 values as a `.npy` at `path`.
pub fn writeNpy(io: std.Io, path: []const u8, rows: usize, cols: usize, data: []const f32) !void {
    var header_buf: [128]u8 = undefined;
    const shape = if (cols == 0)
        try std.fmt.bufPrint(&header_buf, "({d},)", .{rows})
    else
        try std.fmt.bufPrint(&header_buf, "({d}, {d})", .{ rows, cols });
    var dict_buf: [256]u8 = undefined;
    const dict = try std.fmt.bufPrint(&dict_buf, "{{'descr': '<f4', 'fortran_order': False, 'shape': {s}, }}", .{shape});
    // the header, then spaces, then a newline: the total must be a multiple of 64 from the magic
    var head: [512]u8 = undefined;
    const magic = "\x93NUMPY";
    @memcpy(head[0..6], magic);
    head[6] = 1; // major
    head[7] = 0; // minor
    // numpy's header-length field counts the dict, its padding and the newline, measured from the
    // 10-byte preamble: `10 + hlen` is what the data offset must be, and must be a multiple of 64.
    var hlen: usize = dict.len + 1;
    const pad = (64 - ((10 + hlen) % 64)) % 64;
    hlen += pad;
    // v1.0 stores `hlen` itself (dict + padding + newline), and numpy reads data at `10 + hlen`.
    // Storing `hlen - 10` shifts every element ten bytes and numpy still loads it silently —
    // it only checks the file is big enough.
    std.mem.writeInt(u16, head[8..10], @intCast(hlen), .little);
    @memcpy(head[10..][0..dict.len], dict);
    var i: usize = 10 + dict.len;
    while (i < 10 + hlen - 1) : (i += 1) head[i] = ' ';
    head[10 + hlen - 1] = '\n';

    const body = std.mem.sliceAsBytes(data);
    // one write: the header and the body are one file, and a body written separately could be a
    // truncated array if the second write failed
    var file_buf: [4096]u8 = undefined;
    const total = 10 + hlen + body.len;
    if (total <= file_buf.len) {
        @memcpy(file_buf[0 .. 10 + hlen], head[0 .. 10 + hlen]);
        @memcpy(file_buf[10 + hlen ..][0..body.len], body);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = file_buf[0..total] });
        return;
    }
    // a block tensor is at most 8 x 7680 f32 (245 KiB): the streaming path is for the context
    var out: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer out.deinit();
    try out.writer.writeAll(head[0 .. 10 + hlen]);
    try out.writer.writeAll(body);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.written() });
}

test "a stage's statistics are over its own dtype's values" {
    // two bf16 rows of two: (1, 2), (3, 4)
    var bytes: [8]u8 = undefined;
    const vals = [_]f32{ 1, 2, 3, 4 };
    for (vals, 0..) |v, i| {
        const bits: u32 = @bitCast(v);
        std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @truncate(bits >> 16), .little);
    }
    const s = stats(&bytes, 2, 2, .bf16);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), s.mean, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 1.1180339887), s.std, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), s.sample[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), s.sample[1], 1e-6);
    // u32 ids widen exactly and are not read as floats
    var ids: [8]u8 = undefined;
    std.mem.writeInt(u32, ids[0..4], 248_319, .little);
    std.mem.writeInt(u32, ids[4..8], 7, .little);
    const t = stats(&ids, 1, 2, .u32);
    try std.testing.expectApproxEqAbs(@as(f64, 124_163.0), t.mean, 1e-6);
}

test "a stage's f32 widening is its dtype's values" {
    const a = std.testing.allocator;
    var bytes: [4]u8 = undefined;
    const bits: u32 = @bitCast(@as(f32, 0.5));
    std.mem.writeInt(u16, bytes[0..2], @truncate(bits >> 16), .little);
    std.mem.writeInt(u16, bytes[2..4], @truncate((@as(u32, @bitCast(@as(f32, -2.0)))) >> 16), .little);
    const out = try widenAll(a, &bytes, 2, .bf16);
    defer a.free(out);
    try std.testing.expectEqual(@as(f32, 0.5), out[0]);
    try std.testing.expectEqual(@as(f32, -2.0), out[1]);
}

test "writeNpy declares the data offset numpy actually reads at" {
    // v1.0: numpy reads the body at `10 + hlen` where `hlen` is the value in bytes 8..10, and the
    // whole preamble must be a multiple of 64. Writing `hlen - 10` shifts every element ten bytes
    // and numpy loads it without complaining, so the only way to catch it is to read the field.
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const vals = [_]f32{ 1.5, -2.25, 3.0, 4.75 };
    var pbuf: [512]u8 = undefined;
    const n = try tmp.dir.realPath(io, &pbuf);
    const base = pbuf[0..n];
    var fbuf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&fbuf, "{s}/t.npy", .{base});
    try writeNpy(io, path, 2, 2, &vals);

    // the whole file, then the two facts numpy relies on
    const all = try tmp.dir.readFileAlloc(io, "t.npy", std.testing.allocator, .limited(1 << 16));
    defer std.testing.allocator.free(all);
    try std.testing.expectEqualSlices(u8, "\x93NUMPY", all[0..6]);
    const hlen = std.mem.readInt(u16, all[8..10], .little);
    // numpy reads the body at `10 + hlen`, and that offset must be a multiple of 64
    try std.testing.expectEqual(@as(usize, 0), (10 + @as(usize, hlen)) % 64);
    // and the first body value at that offset is the first value written
    const off: usize = 10 + hlen;
    const first = std.mem.readInt(u32, all[off..][0..4], .little);
    try std.testing.expectEqual(@as(f32, 1.5), @as(f32, @bitCast(first)));
}
