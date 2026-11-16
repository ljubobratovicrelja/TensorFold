//! The DFlash2 drafter a flashnext checkpoint can serve (`serve --drafter DIR`).
//!
//! The folder is incoai/Qwen3.8-Flash-Next-DFlash2 or a re-export: a 5-layer Qwen3 backbone over
//! the target's taps, a two-tap dynamic grouped convolution around every attention and MLP, and a
//! low-rank candidate selector over the head's top-k. Its config.json is DFlash2's own format.
//!
//! The parser lives here rather than being shared with `families/qwen3_5`: families are separate
//! build modules and flashnext does not import qwen3_5 (see zig/build/cuda_engines.zig). The
//! field names and the kernel-width refusals are deliberately identical to
//! `qwen3_5/cuda_draft_config.zig`, so a drafter that opens there opens here.
//!
//! What flashnext adds that qwen3_5 does not need: the target is a 4-hyper-connection-stream
//! family, so a tap is a *choice* of row (see `TapSource`), and a drafter trained against a
//! different target must be refused before it drafts a token, not after it drafts garbage.

const std = @import("std");
const config = @import("config.zig");

/// The drafter's own shape, validated.
pub const Draft = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Drafts a verify round proposes beside its anchor (`block_size - 1` are actually drafted).
    block: usize,
    mask: u32,
    window: usize,
    /// Channels a dynamic convolution group spans (`conv_group_size`).
    group: usize,
    /// The candidate selector's state rank and the head's top-k it re-ranks.
    rank: usize,
    topk: usize,
    eps: f32,
    theta: f32,
    taps: [max_taps]u32 = @splat(0),
    n_taps: usize = 0,

    pub fn qWidth(d: Draft) usize {
        return d.heads * d.head_dim;
    }
    pub fn kvWidth(d: Draft) usize {
        return d.kv_heads * d.head_dim;
    }
    /// The dynamic-kernel projection's width: input and output side, two taps, a group each.
    pub fn dynamic(d: Draft) usize {
        return 4 * (d.hidden / d.group);
    }
    /// The drafter's own KV ring rows: its sliding window and a block more.
    pub fn ringRows(d: Draft) usize {
        return d.window + 16;
    }
    /// The drafter's KV rings, every layer, keys and values, bf16.
    pub fn ringBytes(d: Draft) usize {
        return d.layers * 2 * d.ringRows() * d.kvWidth() * 2;
    }
    /// The blocks a stream holds, at most.
    pub fn maxBlocks(d: Draft) usize {
        return d.block;
    }
};

/// Which flashnext row stands in for "the completed layer output" a tap reads.
///
/// MEASURED, and the reason this is a switch rather than a constant: flashnext keeps four
/// hyper-connection streams, so a layer's output is 4 x hidden wide (`input_mix_weight_down` is
/// [320, 10240], `hc_norm` is [10240] f32, and the mix is a *learned* projection — its absmean is
/// 0.0515, not the 0.25 a uniform mean would give). The drafter's `fc` is [hidden, 5 x hidden], so
/// it wants exactly one `hidden`-wide row per tapped layer. SGLang's capture patch averages the
/// four streams; that is what *their* target needed and it is not this tensor.
///
/// Which row is correct is settled by measuring acceptance length against the card's band
/// (2.8-3.3), not by argument. Until then the default is a single flip of `default`.
pub const TapSource = enum {
    /// The hyper-connection read-out's mixed row: down-projected, before the norm.
    readout,
    /// The same row after the layer's norm.
    normed,
    /// The SGLang patch's control: the unweighted mean over the four raw streams.
    mean4,

    pub const default: TapSource = .readout;
};

pub const max_taps = 8;

/// Read `<dir>/config.json` against `target`; every malformed field is a named refusal.
pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, target: config.Config) !Draft {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    return parse(gpa, bytes, target);
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, target: config.Config) !Draft {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadDraftConfig;
    const o = doc.value.object;
    const d = o.get("dflash_config") orelse return error.BadDraftConfig;
    if (d != .object) return error.BadDraftConfig;

    var c: Draft = .{
        .hidden = try int(o, "hidden_size"),
        .intermediate = try int(o, "intermediate_size"),
        .layers = try int(o, "num_hidden_layers"),
        .vocab = try int(o, "vocab_size"),
        .heads = try int(o, "num_attention_heads"),
        .kv_heads = try int(o, "num_key_value_heads"),
        .head_dim = try int(o, "head_dim"),
        .block = try int(d.object, "block_size"),
        .mask = @intCast(try whole(d.object, "mask_token_id")),
        .window = try int(o, "sliding_window"),
        .group = try int(d.object, "conv_group_size"),
        .rank = try int(d.object, "selector_rank"),
        .topk = try int(d.object, "selector_top_k"),
        .eps = @floatCast(try num(o, "rms_norm_eps")),
        .theta = @floatCast(try num(try plainRope(o), "rope_theta")),
    };

    try slidingLayers(o, c.layers);

    const taps = d.object.get("target_layer_ids") orelse return error.BadDraftConfig;
    if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > max_taps) return error.BadDraftConfig;
    for (taps.array.items, 0..) |t, i| {
        if (t != .integer or t.integer < 0 or t.integer >= target.layers) return error.BadDraftConfig;
        c.taps[i] = @intCast(t.integer);
        // one layer tapped twice would write one residual into two ring slots
        if (std.mem.indexOfScalar(u32, c.taps[0..i], c.taps[i]) != null) return error.BadDraftConfig;
    }
    c.n_taps = taps.array.items.len;

    // The drafter's head IS the target's head, so its rows must be the target's width. A
    // mismatch here is silent garbage rather than a crash, so it is refused at admission.
    if (c.hidden != target.hidden) return error.DraftHiddenMismatch;
    if (c.vocab != target.vocab) return error.DraftVocabMismatch;

    // What the kernels fix: head width 128, two-tap convolutions, the selector's top-k within a row.
    if (c.head_dim != 128 or try int(d.object, "conv_kernel_size") != 2 or c.topk > lp_top or c.mask >= c.vocab) return error.UnsupportedDraftConfig;
    if (c.hidden % 256 != 0 or c.hidden % c.group != 0 or c.heads % c.kv_heads != 0 or c.layers > 8) return error.UnsupportedDraftConfig;
    if (c.block < 2 or c.block > max_block) return error.UnsupportedDraftConfig;
    if (c.window < 16) return error.UnsupportedDraftConfig;
    return c;
}

/// The selector's held candidates a row, as the reference's `lp_top`.
pub const lp_top = 20;
/// The verify width the batched pass can draw: 16 rows a round at most.
pub const max_block = 16;

/// CUDA's drafter runs every layer over a sliding ring; no declaration may silently select full attention.
fn slidingLayers(o: std.json.ObjectMap, layers: usize) !void {
    const enabled = o.get("use_sliding_window") orelse return error.BadDraftConfig;
    if (enabled != .bool or !enabled.bool) return error.UnsupportedDraftConfig;
    const types = o.get("layer_types") orelse return error.BadDraftConfig;
    if (types != .array or types.array.items.len != layers) return error.BadDraftConfig;
    for (types.array.items) |kind| {
        if (kind != .string or !std.mem.eql(u8, kind.string, "sliding_attention")) return error.UnsupportedDraftConfig;
    }
}

/// The draft's rope_parameters, refused unless unscaled: its kernels rope with theta alone.
fn plainRope(o: std.json.ObjectMap) !std.json.ObjectMap {
    const rope = o.get("rope_parameters") orelse return error.BadDraftConfig;
    if (rope != .object) return error.BadDraftConfig;
    for ([_][]const u8{ "rope_type", "type" }) |key| if (rope.object.get(key)) |v| {
        if (v != .string or !std.mem.eql(u8, v.string, "default")) return error.UnsupportedDraftConfig;
    };
    if (o.get("rope_scaling")) |v| if (v != .null) return error.UnsupportedDraftConfig;
    return rope.object;
}

/// Whether `dir` declares a DFlash2 drafter (a top-level `dflash_config`); no readable object: not one.
pub fn declared(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, "config.json" }) catch return false;
    defer gpa.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20)) catch return false;
    defer gpa.free(bytes);
    const doc = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch return false;
    defer doc.deinit();
    return doc.value == .object and doc.value.object.get("dflash_config") != null;
}

/// The per-stream bytes the drafter adds beyond its weights: its KV rings plus the target's tap ring.
///
/// The ring is a fixed `tap_rows` regardless of context: at 5 taps x 2560 x 2056 rows bf16 it is
/// 50.2 MiB a stream (0.20 GiB at four), against a 262k fp8 stream of 4.80 GiB — it is not what
/// decides whether the served shape fits, which is why it is worth stating rather than assuming.
pub fn streamBytes(d: Draft, tap_rows: usize, target_hidden: usize) usize {
    return d.ringBytes() + tap_rows * d.n_taps * target_hidden * 2;
}

fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try whole(o, key);
    if (v == 0) return error.BadDraftConfig;
    return v;
}

fn whole(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer < 0) return error.BadDraftConfig;
    return @intCast(v.integer);
}

fn num(o: std.json.ObjectMap, key: []const u8) !f64 {
    return switch (o.get(key) orelse return error.BadDraftConfig) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

/// A flashnext target shaped like the served checkpoint, for the tests below.
fn testTarget() config.Config {
    var c: config.Config = undefined;
    c.hidden = 2560;
    c.vocab = 248320;
    c.layers = 48;
    c.heads = 24;
    c.kv_heads = 2;
    c.value_heads = 2;
    c.eps = 1e-6;
    return c;
}

const real_json =
    \\{"architectures":["DFlash2DraftModel"],"hidden_size":2560,"intermediate_size":7680,
    \\ "num_hidden_layers":5,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,
    \\ "vocab_size":248320,"rms_norm_eps":1e-06,"max_position_embeddings":262144,
    \\ "sliding_window":2048,"use_sliding_window":true,"layer_types":["sliding_attention",
    \\ "sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
    \\ "rope_parameters":{"rope_theta":10000,"rope_type":"default"},
    \\ "dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,
    \\ "mask_token_id":248077,"selector_rank":256,"selector_top_k":16,
    \\ "target_layer_ids":[5,15,25,35,45]}}
;

test "the served drafter's config parses against a flashnext-shaped target" {
    const d = try parse(std.testing.allocator, real_json, testTarget());
    try std.testing.expectEqual(@as(usize, 2560), d.hidden);
    try std.testing.expectEqual(@as(usize, 5), d.layers);
    try std.testing.expectEqual(@as(usize, 8), d.block);
    try std.testing.expectEqual(@as(usize, 5), d.n_taps);
    try std.testing.expectEqualSlices(u32, &.{ 5, 15, 25, 35, 45 }, d.taps[0..5]);
    try std.testing.expectEqual(@as(usize, 256), d.rank);
    try std.testing.expectEqual(@as(usize, 16), d.topk);
    try std.testing.expectEqual(@as(usize, 16), d.group);
    try std.testing.expectEqual(@as(usize, 32), d.heads);
    try std.testing.expectEqual(@as(usize, 8), d.kv_heads);
}

test "a drafter trained on another target is refused before it drafts" {
    const a = std.testing.allocator;
    // A tap past the target's last layer never reaches a ring slot.
    const far = try std.mem.replaceOwned(u8, a, real_json, "[5,15,25,35,45]", "[5,15,25,35,99]");
    defer a.free(far);
    try std.testing.expectError(error.BadDraftConfig, parse(a, far, testTarget()));
    // A drafter whose rows are not the target's width is silent garbage, not a crash.
    const wide = try std.mem.replaceOwned(u8, a, real_json, "\"hidden_size\":2560", "\"hidden_size\":2688");
    defer a.free(wide);
    try std.testing.expectError(error.DraftHiddenMismatch, parse(a, wide, testTarget()));
    // A different vocabulary means its head is not the target's head.
    const vocab = try std.mem.replaceOwned(u8, a, real_json, "\"vocab_size\":248320", "\"vocab_size\":151936");
    defer a.free(vocab);
    try std.testing.expectError(error.DraftVocabMismatch, parse(a, vocab, testTarget()));
    // One layer tapped twice would write one residual into two slots.
    const dup = try std.mem.replaceOwned(u8, a, real_json, "[5,15,25,35,45]", "[5,15,15,35,45]");
    defer a.free(dup);
    try std.testing.expectError(error.BadDraftConfig, parse(a, dup, testTarget()));
}

test "the tap ring is a fixed cost, independent of the context window" {
    const d = try parse(std.testing.allocator, real_json, testTarget());
    const tap_rows = 2048 + 8;
    const ring = tap_rows * d.n_taps * d.hidden * 2;
    // 5 taps x 2560 x 2056 rows of bf16: 50.2 MiB a stream, whatever the window.
    try std.testing.expectEqual(@as(usize, 52_633_600), ring);
    const kv = d.ringBytes();
    try std.testing.expectEqual(@as(usize, 5 * 2 * (2048 + 16) * (8 * 128) * 2), kv);
    try std.testing.expectEqual(ring + kv, streamBytes(d, tap_rows, 2560));
}
