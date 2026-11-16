//! qwen35_bf16_rows.cu: bf16 token rows and small bf16 dense projections (GDN in_proj_a and in_proj_b).
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const kernels = @import("kernels.zig");
const launch = @import("launch.zig");

const threads = 256;

/// Rows `denseBatch` takes at most: `tf_q35_dense_bf16_batch`'s `tf_q35_batch_rows`.
pub const batch_rows = 16;

pub const Kernels = struct {
    mod: Module,
    embed_fn: Function,
    dense_fn: Function,
    dense_batch_fn: Function,

    pub fn resolve(mod: Module) !Kernels {
        return .{
            .mod = mod,
            .embed_fn = try mod.function("tf_q35_embed_bf16"),
            .dense_fn = try mod.function("tf_q35_dense_bf16"),
            .dense_batch_fn = try mod.function("tf_q35_dense_bf16_batch"),
        };
    }

    /// Load the embedded module (owned: call unload) and resolve its functions.
    pub fn load(d: *const Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.qwen35_bf16_rows);
        errdefer mod.unload();
        return resolve(mod);
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// out[r] = table[ids[r]] (ids i32 [rows], table bf16 [vocab][hidden]); params as tf_q35_embed's, or 0.
    pub fn embedRows(k: *const Kernels, s: Stream, ids: u64, table: u64, out: u64, rows: usize, hidden: usize, params: u64) !void {
        if (rows == 0) return;
        if (hidden == 0 or hidden > std.math.maxInt(i32) or rows > std.math.maxInt(u32)) return error.RowShape;
        var a: launch.Args = .{};
        for ([_]u64{ ids, table, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(hidden)));
        a.add(params);
        try launch.launch(k.embed_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// out[rows][n] = x[rows][k] @ w[n][k]^T in bf16 with fixed-order fp32 sums (rows <= 128, k % 8 == 0, aligned).
    pub fn denseRows(kn: *const Kernels, s: Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        if (rows == 0 or n == 0) return;
        if (rows > 128 or k == 0 or k % 8 != 0 or k > std.math.maxInt(i32) or n > std.math.maxInt(i32) or x % 16 != 0 or w % 16 != 0) return error.RowShape;
        var a: launch.Args = .{};
        for ([_]u64{ x, w, out }) |v| a.add(v);
        for ([_]usize{ n, k }) |v| a.add(@as(i32, @intCast(v)));
        try launch.launch(kn.dense_fn, .{ .grid = .{ .x = @intCast(n), .y = @intCast(rows), .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// out[rows][n] = x[rows][k] @ w[n][k]^T for every row at once: grid (n, 1), one CTA an output
    /// column, a weight row read once for the whole block instead of once a row (the served block's
    /// projections are 8x less weight traffic; measured 3-8x at the served shapes, 7-8x on the wide ones).
    ///
    /// The bits are `denseRows`'s bit for bit — the same per-thread stride-256 order over k / 8 uint4
    /// words, the same warp shuffle, the same thread-0 sum — so a caller can swap one launch for `rows`
    /// and read the same bytes downstream. rows <= 16, k % 8 == 0, aligned; no shared memory.
    pub fn denseBatch(kn: *const Kernels, s: Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        if (rows == 0 or n == 0) return;
        if (rows > batch_rows or k == 0 or k % 8 != 0 or k > std.math.maxInt(i32) or n > std.math.maxInt(i32) or x % 16 != 0 or w % 16 != 0) return error.RowShape;
        var a: launch.Args = .{};
        for ([_]u64{ x, w, out }) |v| a.add(v);
        for ([_]usize{ n, k, rows }) |v| a.add(@as(i32, @intCast(v)));
        try launch.launch(kn.dense_batch_fn, .{ .grid = .{ .x = @intCast(n), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};
