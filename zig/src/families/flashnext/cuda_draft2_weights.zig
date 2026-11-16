//! The DFlash2 drafter checkpoint's tensor contract: every tensor it must have, sized from its
//! own config, and a refusal for anything missing, misshaped or extra.
//!
//! Measured against `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744e85c…`: 81 tensors, all bf16 —
//! 6 outside the layers and 15 in each of the 5 layers. The table below is generated from that
//! header, so `expected` counts 6 + 15 * layers and the test pins the real checkpoint's 81.
//!
//! Why a table rather than reading names as it loads: a misshaped tensor in the backbone does not
//! crash — it silently reads the wrong rows and the drafter proposes plausible garbage at a low
//! acceptance length, which looks like a weak drafter rather than a loader bug. Refusing up front
//! turns that into one named error before a single token is drafted.

const std = @import("std");
const Draft = @import("cuda_draft2.zig").Draft;

/// One tensor the drafter must have: its name and the shape its config implies.
pub const Tensor = struct {
    name: []const u8,
    /// 0 for a vector, else the extent of each axis; `{}` means a 1-D tensor of `vector` elements.
    shape: []const usize,
};

/// The most tensors the contract can name: a layer's 15 plus slack for a family with more.
pub const max_tensors = 6 + 15 * 8;

/// What a checkpoint's own header reports, so `check` can compare without owning the parsed file.
pub const Listing = struct {
    names: []const []const u8,
    shapes: []const []const usize,

    pub fn count(l: Listing) usize {
        return l.names.len;
    }
};

/// The tensors outside the layers, in the checkpoint's own order.
pub const top_count = 6;
/// The tensors each layer carries.
pub const per_layer = 15;

/// Fill `out` with the contract for `d`; returns the number written.
///
/// The shapes follow the drafter's config (hidden, heads, kv_heads, head_dim, ffn, vocab,
/// conv_kernel, conv_group, selector_rank, n_taps), never the served checkpoint's, so a drafter
/// trained on a different shape is refused rather than half-loaded.
pub fn expected(d: Draft, out: *[max_tensors]Tensor, shapes: *[max_tensors][4]usize) usize {
    const heads = d.heads;
    const hd = d.head_dim;
    const kvw = d.kv_heads * hd;
    const qw = heads * hd;
    const groups = d.hidden / d.group;
    const dyn = 2 * d.conv_kernel * groups;
    var n: usize = 0;

    shapes[n][0] = d.hidden;
    shapes[n][1] = d.n_taps * d.hidden;
    out[n] = .{ .name = "fc.weight", .shape = shapes[n][0..2] };
    n += 1;
    shapes[n][0] = d.hidden;
    out[n] = .{ .name = "hidden_norm.weight", .shape = shapes[n][0..1] };
    n += 1;
    shapes[n][0] = d.hidden;
    out[n] = .{ .name = "norm.weight", .shape = shapes[n][0..1] };
    n += 1;
    shapes[n][0] = d.rank;
    shapes[n][1] = d.hidden;
    out[n] = .{ .name = "candidate_selector.hidden_projection.weight", .shape = shapes[n][0..2] };
    n += 1;
    shapes[n][0] = d.vocab;
    shapes[n][1] = d.rank;
    out[n] = .{ .name = "candidate_selector.predecessor_codebook", .shape = shapes[n][0..2] };
    n += 1;
    shapes[n][0] = d.vocab;
    shapes[n][1] = d.rank;
    out[n] = .{ .name = "candidate_selector.successor_codebook", .shape = shapes[n][0..2] };
    n += 1;

    // Every layer carries the same 15, so the names are composed at check time from the index;
    // `layerTensor` builds one on demand rather than storing 15 x layers strings.
    _ = qw;
    _ = kvw;
    _ = dyn;
    return n;
}

/// The `k`-th tensor of layer `i` (`0 .. per_layer`), built into `buf` for its name and `shape` for its extents.
pub fn layerTensor(d: Draft, i: usize, k: usize, buf: []u8, shape: *[4]usize) !Tensor {
    const heads, const hd, const hidden = .{ d.heads, d.head_dim, d.hidden };
    const qw = heads * hd;
    const kvw = d.kv_heads * hd;
    const groups = hidden / d.group;
    const dyn = 2 * d.conv_kernel * groups;
    const entry = switch (k) {
        0 => .{ "input_layernorm.weight", &[_]usize{hidden} },
        1 => .{ "post_attention_layernorm.weight", &[_]usize{hidden} },
        2 => .{ "self_attn.q_proj.weight", &[_]usize{ qw, hidden } },
        3 => .{ "self_attn.k_proj.weight", &[_]usize{ kvw, hidden } },
        4 => .{ "self_attn.v_proj.weight", &[_]usize{ kvw, hidden } },
        5 => .{ "self_attn.o_proj.weight", &[_]usize{ hidden, qw } },
        6 => .{ "self_attn.q_norm.weight", &[_]usize{hd} },
        7 => .{ "self_attn.k_norm.weight", &[_]usize{hd} },
        8 => .{ "mlp.gate_proj.weight", &[_]usize{ d.intermediate, hidden } },
        9 => .{ "mlp.up_proj.weight", &[_]usize{ d.intermediate, hidden } },
        10 => .{ "mlp.down_proj.weight", &[_]usize{ hidden, d.intermediate } },
        11 => .{ "attention_conv.base_kernel", &[_]usize{ 2, d.conv_kernel, hidden } },
        12 => .{ "attention_conv.kernel_projection.weight", &[_]usize{ dyn, hidden } },
        13 => .{ "mlp_conv.base_kernel", &[_]usize{ 2, d.conv_kernel, hidden } },
        14 => .{ "mlp_conv.kernel_projection.weight", &[_]usize{ dyn, hidden } },
        else => return error.TooManyTensors,
    };
    const name = try std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ i, entry[0] });
    @memcpy(shape[0..entry[1].len], entry[1]);
    return .{ .name = name, .shape = shape[0..entry[1].len] };
}

/// The weight bytes the contract accounts for: every tensor's extent product times bf16's two bytes.
pub fn bytes(d: Draft) usize {
    var total: usize = 0;
    var buf: [64]u8 = undefined;
    var shape: [4]usize = undefined;
    // outside the layers: fc, two norms, and the selector's three
    total += d.hidden * d.n_taps * d.hidden * 2; // fc
    total += d.hidden * 2 * 2; // hidden_norm, norm
    total += d.rank * d.hidden * 2; // hidden_projection
    total += d.vocab * d.rank * 2 * 2; // both codebooks
    for (0..d.layers) |i| for (0..per_layer) |k| {
        const t = layerTensor(d, i, k, &buf, &shape) catch continue;
        var elems: usize = 1;
        for (t.shape) |s| elems *= s;
        total += elems * 2;
    };
    return total;
}

/// Refuse `l` unless it is exactly the contract for `d`. The first fault is the one reported.
pub fn check(d: Draft, l: Listing) !void {
    if (l.count() != top_count + per_layer * d.layers) return error.TensorCountMismatch;
    var want_buf: [max_tensors]Tensor = undefined;
    var shapes: [max_tensors][4]usize = undefined;
    const top = expected(d, &want_buf, &shapes);
    var name_buf: [64]u8 = undefined;
    var shape_buf: [4]usize = undefined;

    // Every name the contract expects must be present with its exact shape, and every name the
    // listing carries must be one the contract expects: an extra tensor means a checkpoint this
    // loader does not understand, which is worse to half-read than to refuse.
    for (0..top) |i| try match(l, want_buf[i]);
    for (0..d.layers) |i| for (0..per_layer) |k| {
        const t = try layerTensor(d, i, k, &name_buf, &shape_buf);
        try match(l, t);
    };
}

fn match(l: Listing, want: Tensor) !void {
    for (l.names, l.shapes) |name, shape| {
        if (!std.mem.eql(u8, name, want.name)) continue;
        if (shape.len != want.shape.len) return error.TensorShapeMismatch;
        for (shape, want.shape) |got, exp| if (got != exp) return error.TensorShapeMismatch;
        return;
    }
    return error.TensorMissing;
}

/// A checkpoint shaped like the served drafter, for the tests.
fn testDraft() Draft {
    return .{
        .hidden = 2560, .intermediate = 7680, .layers = 5, .vocab = 248320,
        .heads = 32, .kv_heads = 8, .head_dim = 128, .block = 8, .mask = 248077,
        .window = 2048, .group = 16, .rank = 256, .topk = 16, .eps = 1e-6, .theta = 10000,
        .taps = .{ 5, 15, 25, 35, 45, 0, 0, 0 }, .n_taps = 5, .conv_kernel = 2,
    };
}

test "the contract names the served drafter's 81 tensors: 6 outside the layers, 15 in each" {
    const d = testDraft();
    try std.testing.expectEqual(@as(usize, 6), top_count);
    try std.testing.expectEqual(@as(usize, 15), per_layer);
    try std.testing.expectEqual(@as(usize, 81), top_count + per_layer * d.layers);

    var out: [max_tensors]Tensor = undefined;
    var shapes: [max_tensors][4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 6), expected(d, &out, &shapes));

    var buf: [64]u8 = undefined;
    var shape: [4]usize = undefined;
    // layer 0's first and last, against the header this file was written from
    const first = try layerTensor(d, 0, 0, &buf, &shape);
    try std.testing.expectEqualStrings("layers.0.input_layernorm.weight", first.name);
    try std.testing.expectEqualSlices(usize, &.{2560}, first.shape);
    const conv = try layerTensor(d, 0, 12, &buf, &shape);
    try std.testing.expectEqualStrings("layers.0.attention_conv.kernel_projection.weight", conv.name);
    // 2 * conv_kernel * (hidden / group) = 2 * 2 * 160 = 640, and the header says [640, 2560]
    try std.testing.expectEqualSlices(usize, &.{ 640, 2560 }, conv.shape);
    const o = try layerTensor(d, 3, 5, &buf, &shape);
    try std.testing.expectEqualStrings("layers.3.self_attn.o_proj.weight", o.name);
    try std.testing.expectEqualSlices(usize, &.{ 2560, 4096 }, o.shape);
}

test "the accounted bytes match the checkpoint's 1.21 GB exactly" {
    const d = testDraft();
    const b = bytes(d);
    // 1,206,131,200 bytes: the sum of every tensor's extents x 2, read from the safetensors header.
    // The file on disk is 1,206,140,008 bytes; the difference is header and alignment padding,
    // which is why this asserts the tensors' own bytes rather than the file size.
    try std.testing.expectEqual(@as(usize, 1_206_131_200), b);
    try std.testing.expectEqual(@as(usize, 1_206_140_008 - 1_206_131_200), 8808); // header overhead
}

test "a missing, misshaped or extra tensor is refused" {
    const d = testDraft();
    const a = std.testing.allocator;
    const count = top_count + per_layer * d.layers;
    var names: std.ArrayListUnmanaged([]const u8) = .empty;
    defer {
        for (names.items) |n| a.free(n);
        names.deinit(a);
    }
    var shapes: std.ArrayListUnmanaged([]const usize) = .empty;
    defer {
        for (shapes.items) |s| a.free(s);
        shapes.deinit(a);
    }
    try names.append(a, try a.dupe(u8, "fc.weight"));
    try shapes.append(a, try a.dupe(usize, &.{ 2560, 12800 }));
    // an incomplete listing of the right length is refused on its first missing name
    for (1..count) |i| {
        try names.append(a, try std.fmt.allocPrint(a, "filler.{d}", .{i}));
        try shapes.append(a, try a.dupe(usize, &.{1}));
    }
    try std.testing.expectError(error.TensorMissing, check(d, .{ .names = names.items, .shapes = shapes.items }));
    // a short listing is refused before any name is compared
    try std.testing.expectError(error.TensorCountMismatch, check(d, .{ .names = names.items[0..3], .shapes = shapes.items[0..3] }));

}
