//! The block's projection in one launch against the per-element kernel it replaces: same bytes out,
//! for the shapes a draft round asks for (a served 8-row block over the 2560-, 4096- and 7680-wide
//! inputs) and for the edges (one row, sixteen, one output column, the smallest whole word).
//!
//! `tf_q35_dense_bf16_batch` is only worth a launch per projection if it is `tf_q35_dense_bf16` to the
//! bit: the summing order is one thread's stride-256 walk over `k / 8` uint4 words, a warp shuffle,
//! then thread 0's ascending sum of the eight warps' partials, and a launch that produced a different
//! float here would move every row the rest of the round reads. So the check is `check.sameBytes`
//! against the per-element launch, plus a float64 band that says the batched answer is the projection
//! and not just the same mistake twice.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// `x` as the bf16 bit pattern a round-to-nearest cast gives (the drawing values are representable,
/// so this is a cast and not a reference rounding).
fn bf(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

fn f(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

const Case = struct {
    rows: usize,
    n: usize,
    k: usize,
};

pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var rng = std.Random.DefaultPrng.init(0x9f35);
    const r = rng.random();
    var kx = try cuda.qwen35_bf16_rows.Kernels.load(gpu.d);
    defer kx.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();

    // 2560 is the served hidden width (q, k, v, gate, up), 4096 the attention output's input width and
    // 7680 the MLP's; the rest are the edges of the launch: one row, the kernel's maximum of sixteen,
    // one output column, the smallest whole uint4 word, and a width smaller than the block.
    const cases = [_]Case{
        .{ .rows = 8, .n = 64, .k = 2560 },
        .{ .rows = 8, .n = 32, .k = 4096 },
        .{ .rows = 8, .n = 16, .k = 7680 },
        .{ .rows = 1, .n = 48, .k = 1024 },
        .{ .rows = 16, .n = 24, .k = 1024 },
        .{ .rows = 8, .n = 1, .k = 256 },
        .{ .rows = 5, .n = 40, .k = 8 },
        .{ .rows = 3, .n = 96, .k = 512 },
        .{ .rows = 2, .n = 7, .k = 32 },
    };

    for (cases) |cs| {
        const xh = try a.alloc(u16, cs.rows * cs.k);
        defer a.free(xh);
        for (xh) |*v| v.* = bf((r.float(f32) - 0.5) * 0.5);
        const wh = try a.alloc(u16, cs.n * cs.k);
        defer a.free(wh);
        for (wh) |*v| v.* = bf((r.float(f32) - 0.5) * 0.25);

        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
        defer dx.free();
        var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wh));
        defer dw.free();
        var one = try cuda.DeviceBuffer.alloc(gpu.d, cs.rows * cs.n * 2);
        defer one.free();
        var batched = try cuda.DeviceBuffer.alloc(gpu.d, cs.rows * cs.n * 2);
        defer batched.free();
        // a launch that writes nothing must fail the comparison rather than leave stale bytes that pass
        try one.fill8(0xff, s.handle);
        try batched.fill8(0xff, s.handle);

        try kx.denseRows(s, dx.ptr, dw.ptr, one.ptr, cs.rows, cs.n, cs.k);
        try kx.denseBatch(s, dx.ptr, dw.ptr, batched.ptr, cs.rows, cs.n, cs.k);
        try s.synchronize();

        const got = try check.download(gpu, batched);
        defer a.free(got);
        const want = try check.download(gpu, one);
        defer a.free(want);
        try check.sameBytes("the block's dense projection", got, want);

        // and the answer is the projection: bf16 against a float64 dot, its terms' magnitude the floor
        const got_h = std.mem.bytesAsSlice(u16, got);
        var worst: f64 = 0;
        var mag: f64 = 0;
        for (0..cs.rows) |row| {
            for (0..cs.n) |j| {
                var acc: f64 = 0;
                var m: f64 = 0;
                for (0..cs.k) |i| {
                    const t = f(xh[row * cs.k + i]) * f(wh[j * cs.k + i]);
                    acc += t;
                    m += @abs(t);
                }
                worst = @max(worst, @abs(f(got_h[row * cs.n + j]) - acc) / @max(m, 1e-6));
                mag = @max(mag, m);
            }
        }
        // bf16's round-to-nearest on the stored output is 2^-9 of it, and the terms' magnitude is an upper
        // bound on the output, so the band's floor is ~2e-3 of it: 5e-3 is the honest gate (measured 1.8e-4)
        try check.expect(worst <= 5e-3, "the block's dense projection ({d} rows, n {d}, k {d}): worst {e} of its terms' magnitude", .{ cs.rows, cs.n, cs.k, worst });
        try check.expect(mag > 0, "the case {d}x{d}x{d} has no magnitude to compare against", .{ cs.rows, cs.n, cs.k });
    }
    check.pass("qwen35 dense batch: {d} shapes, one launch a shape, the per-element kernel's bytes and within 5e-3 of the float64 dot", .{cases.len});
}
