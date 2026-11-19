//! `TF_FLASHNEXT_HC_FP8=<dir>`: the hyper-connection input mixes read from row-scaled FP8 e4m3 tables.
//!
//! Every decoder layer's attention branch and MoE branch begin with the same shape of matrix pair:
//! `input_mix_weight_down` [320, 10240] bf16 and `input_mix_weight_up` [10240, 320] bf16 -- the
//! layer's four-stream row normalised, collapsed to 320 channels, activated, and mixed back out to
//! 10240. Ninety-six of those pairs (2 branches x 48 layers) are 1.26 GB of bf16 read once a decode
//! round, in the two stages the layer clock calls `conv_attn_in` and `conv_mlp_in`: 11.4 ms
//! together at 73-131 GB/s, the least efficient reads of the pass.
//!
//! The artifact this module loads is the same matrices in e4m3 with one f32 scale an output row:
//! 0.63 GB of codes and 4.1 MB of scales, so each stage's weight stream halves for the same
//! tensor-core MACs (`qmmf`'s FP8G lane -- bf16 activations, f32 accumulate, the lane the FP8
//! `lm_head` arm already ships on). There is no rescore here and none is possible: every channel of
//! both matrices feeds the mixed row, and the error is then followed by normalisation and the
//! residual path, so the arm is judged by end-to-end output agreement rather than by a recovered
//! top-k.
//!
//! Off unless `TF_FLASHNEXT_HC_FP8` names a directory (`"0"` and an empty value read as unset, so a
//! stale export cannot arm it): `HcW.fp8` is then null, the read-out takes its shipped branch, this
//! module allocates nothing, opens no file and loads no image.
//!
//! The artifact is written by `tools/dflash2/quantize/quantize_hc_fp8.py`. Its `geometry.json` is
//! checked against the layer being loaded -- a table fitted for another rank, width or row count is
//! refused at load rather than read at the wrong stride and producing a plausible, wrong mixed row.
const std = @import("std");
const cuda = @import("cuda");
const fp8 = cuda.fp8;
const qmmf = cuda.qmmf;
const head_fp8 = @import("cuda_head_fp8.zig");

/// The lane reads each table in one K slice.
///
/// The bf16 path lets `qmmf.splitK` cut the down projection into eight slices and reduce them; the
/// FP8 tables are read in one block a row tile here, which is the cheapest bookkeeping (no partials
/// buffer, no reduce launch) and one ascending accumulation -- the comparison an arm this sensitive
/// should be made on. `pinned` refuses the pin if a shape cannot take it, and every shape here can:
/// `wide / 64 = 160` groups and `low / 64 = 5`.
pub const slices: u32 = 1;

pub const Kind = enum { attn, mlp };

/// One layer branch's two tables: `down` is `[low + inject, wide]` (the engine's `n_down`, the
/// checkpoint's 320 mix rows and then its 4 inject rows), `up` is `[wide, low]`.
pub const Pair = struct {
    down: qmmf.Weight,
    up: qmmf.Weight,
    /// Both codes tables and both scale tiles, 256-byte aligned.
    arena: cuda.DeviceBuffer,
    /// `arena`'s length, for the load log.
    bytes: usize,

    pub fn free(p: *Pair) void {
        p.arena.free();
    }
};

/// `TF_FLASHNEXT_HC_FP8`'s text: the artifact's directory, or null for today's bf16 matrices. Only the
/// path -- whether an arm is taken is `spec`'s and the loader's, so `"0"` and an empty value are off
/// and a path that is not an artifact is refused at load rather than ignored.
pub fn dirFromEnv() ?[]const u8 {
    const p = std.c.getenv("TF_FLASHNEXT_HC_FP8") orelse return null;
    return choice(std.mem.span(p));
}

/// `dirFromEnv`'s parse, on the text alone: unset, empty and `"0"` are off, anything else is a path.
pub fn choice(text: []const u8) ?[]const u8 {
    if (text.len == 0 or std.mem.eql(u8, text, "0")) return null;
    return text;
}

pub const Spec = struct { layer: usize, kind: Kind };

/// The layer and branch a hyper-connection prefix names, or null if the flag must not arm it.
///
/// `cuda_prompt_wt.loadHc` is called for the per-layer branches (`language_model.model.layers.{d}
/// .attn_hyper_connection`, `.mlp_hyper_connection`) and also for the two read-out mixers -- the
/// final `language_model.model.hyper_connection_mixer` and the MTP head's
/// `language_model.mtp.hyper_connection_mixer` -- and for the MTP layer's own branches. Those are
/// not the 48 x 2 matrices this artifact holds, so they are matched out by prefix: arming them with
/// layer 0's tables would be plausible, silent and wrong.
pub fn spec(prefix: []const u8) ?Spec {
    const head = "language_model.model.layers.";
    if (!std.mem.startsWith(u8, prefix, head)) return null;
    const rest = prefix[head.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
    const tail = rest[dot + 1 ..];
    if (std.mem.eql(u8, tail, "attn_hyper_connection")) return .{ .layer = layer, .kind = .attn };
    if (std.mem.eql(u8, tail, "mlp_hyper_connection")) return .{ .layer = layer, .kind = .mlp };
    return null;
}

/// `d`'s file name for one side of one pair, in the artifact's own spelling
/// (`l07.attn.down.fp8.e4m3.bin`, `l07.attn.down.fp8.scales.f32`).
pub fn fileName(buf: []u8, layer: usize, kind: Kind, side: []const u8, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "l{d:0>2}.{s}.{s}.{s}", .{ layer, @tagName(kind), side, suffix });
}

const name_geometry = "geometry.json";
pub const manifest_format = "flashnext-hc-fp8-rtn";

/// Load layer `spec.layer`'s `spec.kind` pair out of `dir`: both codes tables and both row-scale
/// tiles packed for the FP8G lane, in one device buffer.
///
/// Every refusal is one a load should not survive: the geometry against the served shape, a table's
/// length against its rows and width, and the lane's plan for the pinned slice count.
pub fn load(
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *cuda.Driver,
    dir: []const u8,
    spec_: Spec,
    n_down: usize,
    wide: usize,
    low: usize,
) !Pair {
    if (!cuda.kernels.available) return error.BuiltWithoutKernels;
    if (n_down == 0 or wide == 0 or low == 0 or wide % qmmf.group != 0 or low % qmmf.group != 0) return error.HcFp8Shape;
    try checkGeometry(gpa, io, dir, spec_, n_down, wide, low);

    var name_buf: [96]u8 = undefined;
    const dn_codes = try readExact(gpa, io, dir, try fileName(&name_buf, spec_.layer, spec_.kind, "down", "fp8.e4m3.bin"), n_down * wide);
    defer gpa.free(dn_codes);
    const dn_scales = try readExact(gpa, io, dir, try fileName(&name_buf, spec_.layer, spec_.kind, "down", "fp8.scales.f32"), n_down * 4);
    defer gpa.free(dn_scales);
    const up_codes = try readExact(gpa, io, dir, try fileName(&name_buf, spec_.layer, spec_.kind, "up", "fp8.e4m3.bin"), wide * low);
    defer gpa.free(up_codes);
    const up_scales = try readExact(gpa, io, dir, try fileName(&name_buf, spec_.layer, spec_.kind, "up", "fp8.scales.f32"), wide * 4);
    defer gpa.free(up_scales);

    // the lane's fragment order, from the row-major codes the artifact stores
    const dn_frag = try gpa.alloc(u8, fp8.padded(n_down) * wide);
    defer gpa.free(dn_frag);
    fp8.packCodes(dn_frag, dn_codes, n_down, wide);
    const up_frag = try gpa.alloc(u8, fp8.padded(wide) * low);
    defer gpa.free(up_frag);
    fp8.packCodes(up_frag, up_codes, wide, low);

    // and the per-row scales as the kernel's [npad/64][k/64][64] tile: a row's own scale over its
    // `k / 64` input groups, so the FP8G lane applies it once a group after the group's MACs
    const dn_row = try gpa.alloc(f32, n_down);
    defer gpa.free(dn_row);
    decodeScales(dn_row, dn_scales);
    const dn_tile = try gpa.alloc(f32, fp8.padded(n_down) * (wide / qmmf.group));
    defer gpa.free(dn_tile);
    head_fp8.packRowScales(dn_tile, dn_row, n_down, wide);
    const up_row = try gpa.alloc(f32, wide);
    defer gpa.free(up_row);
    decodeScales(up_row, up_scales);
    const up_tile = try gpa.alloc(f32, fp8.padded(wide) * (low / qmmf.group));
    defer gpa.free(up_tile);
    head_fp8.packRowScales(up_tile, up_row, wide, low);

    const parts = .{
        std.mem.sliceAsBytes(dn_frag),
        std.mem.sliceAsBytes(dn_tile),
        std.mem.sliceAsBytes(up_frag),
        std.mem.sliceAsBytes(up_tile),
    };
    var offsets: [4]usize = .{ 0, 0, 0, 0 };
    var total: usize = 0;
    inline for (parts, 0..) |bytes, i| {
        const at = (total + 255) & ~@as(usize, 255);
        offsets[i] = at;
        total = try std.math.add(usize, at, bytes.len);
    }

    var pair: Pair = .{ .down = undefined, .up = undefined, .arena = undefined, .bytes = total };
    pair.arena = try cuda.DeviceBuffer.alloc(driver, total);
    errdefer pair.arena.free();
    inline for (parts, 0..) |bytes, i| try pair.arena.upload(offsets[i], bytes);
    pair.down = qmmf.pinned(fp8.weight(pair.arena.ptr + offsets[0], pair.arena.ptr + offsets[1], n_down, wide), slices) catch return error.HcFp8Plan;
    pair.up = qmmf.pinned(fp8.weight(pair.arena.ptr + offsets[2], pair.arena.ptr + offsets[3], wide, low), slices) catch return error.HcFp8Plan;
    return pair;
}

fn decodeScales(out: []f32, raw: []const u8) void {
    std.debug.assert(raw.len == out.len * 4);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
}

fn path(gpa: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    return std.fs.path.join(gpa, &.{ dir, name });
}

/// `dir`'s `name`, exactly `len` bytes: a short or long file is a different table, not one to read at
/// a guessed stride.
fn readExact(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, len: usize) ![]u8 {
    const p = try path(gpa, dir, name);
    defer gpa.free(p);
    const data = std.Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(len + 1)) catch |e| {
        std.log.err("flashnext: the quantised hyper-connection table {s} is missing or unreadable ({s}) at {s}", .{ name, @errorName(e), p });
        return e;
    };
    if (data.len != len) {
        std.log.err("flashnext: {s} is {d} bytes, this layer's shape needs {d}", .{ name, data.len, len });
        gpa.free(data);
        return error.HcFp8ArtifactShape;
    }
    return data;
}

fn field(v: std.json.Value, name: []const u8) !std.json.Value {
    return switch (v) {
        .object => |o| o.get(name) orelse error.HcFp8Geometry,
        else => error.HcFp8Geometry,
    };
}

fn int(v: std.json.Value) !usize {
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else error.HcFp8Geometry,
        else => error.HcFp8Geometry,
    };
}

/// `pair` of two ints at `name`, checked against `want`.
fn pairOf(v: std.json.Value, name: []const u8, want: [2]usize) !void {
    const arr = switch (try field(v, name)) {
        .array => |a| a,
        else => return error.HcFp8Geometry,
    };
    if (arr.items.len != 2) return error.HcFp8Geometry;
    if (try int(arr.items[0]) != want[0] or try int(arr.items[1]) != want[1]) {
        return error.HcFp8ArtifactShape;
    }
}

/// The artifact's `geometry.json`, checked for the shape this engine serves and for the layer being
/// loaded: a table fitted for another rank, width or layer count would be read at the wrong stride
/// and produce a plausible, wrong mixed row.
///
/// The geometry is its own small file rather than the manifest's first keys because the manifest is
/// the audit trail (~0.5 MB of per-tensor error statistics) and this is read once a branch, 96 times a
/// load; parsing a megabyte of JSON 96 times would cost more than the tables.
fn checkGeometry(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, spec_: Spec, n_down: usize, wide: usize, low: usize) !void {
    const p = try path(gpa, dir, name_geometry);
    defer gpa.free(p);
    const text = std.Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(1 << 16)) catch |e| {
        std.log.err("flashnext: TF_FLASHNEXT_HC_FP8 needs {s} in {s} ({s})", .{ name_geometry, dir, @errorName(e) });
        return e;
    };
    defer gpa.free(text);
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return error.HcFp8Geometry;
    defer parsed.deinit();
    const root = parsed.value;

    const format = switch (try field(root, "format")) {
        .string => |s| s,
        else => return error.HcFp8Geometry,
    };
    if (!std.mem.eql(u8, format, manifest_format)) {
        std.log.err("flashnext: {s} is a {s} artifact, this arm wants {s}", .{ dir, format, manifest_format });
        return error.HcFp8Geometry;
    }
    const layers = try int(try field(root, "layers"));
    if (spec_.layer >= layers) {
        std.log.err("flashnext: TF_FLASHNEXT_HC_FP8 holds {d} layers, layer {d} has no table", .{ layers, spec_.layer });
        return error.HcFp8ArtifactLayers;
    }
    pairOf(root, "shape_down", .{ n_down, wide }) catch |e| {
        std.log.err("flashnext: the quantised hyper-connection down table is not {d} x {d}", .{ n_down, wide });
        return e;
    };
    pairOf(root, "shape_up", .{ wide, low }) catch |e| {
        std.log.err("flashnext: the quantised hyper-connection up table is not {d} x {d}", .{ wide, low });
        return e;
    };
}

test "Flash Next hc fp8: the flag is off unless a path is named, and 0 is off" {
    try std.testing.expect(choice("") == null);
    try std.testing.expect(choice("0") == null);
    try std.testing.expectEqualStrings("/tmp/hc", choice("/tmp/hc").?);
}

test "Flash Next hc fp8: only the 48 x 2 per-layer branches arm, not the read-out mixers or the MTP" {
    const attn = spec("language_model.model.layers.7.attn_hyper_connection").?;
    try std.testing.expectEqual(@as(usize, 7), attn.layer);
    try std.testing.expect(attn.kind == .attn);
    const mlp = spec("language_model.model.layers.47.mlp_hyper_connection").?;
    try std.testing.expectEqual(@as(usize, 47), mlp.layer);
    try std.testing.expect(mlp.kind == .mlp);
    // the final read-out mixer, the MTP's mixer and the MTP layer's own branches are not in the artifact
    try std.testing.expect(spec("language_model.model.hyper_connection_mixer") == null);
    try std.testing.expect(spec("language_model.mtp.hyper_connection_mixer") == null);
    try std.testing.expect(spec("language_model.mtp.layers.0.attn_hyper_connection") == null);
    // and a prefix that is not a layer at all
    try std.testing.expect(spec("language_model.model.layers.x.attn_hyper_connection") == null);
    try std.testing.expect(spec("language_model.model.layers.3.attn_hyper_connection.extra") == null);
}

test "Flash Next hc fp8: a pair's file names are the quantiser's, per layer and branch" {
    var buf: [96]u8 = undefined;
    try std.testing.expectEqualStrings("l07.attn.down.fp8.e4m3.bin", try fileName(&buf, 7, .attn, "down", "fp8.e4m3.bin"));
    try std.testing.expectEqualStrings("l47.mlp.up.fp8.scales.f32", try fileName(&buf, 47, .mlp, "up", "fp8.scales.f32"));
}
