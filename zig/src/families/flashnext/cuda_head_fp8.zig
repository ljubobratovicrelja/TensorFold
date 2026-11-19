//! `TF_FLASHNEXT_LMHEAD_FP8=<dir>`: the target's output head ranked through a row-scaled FP8 copy of
//! `lm_head` and rescored with exact bf16 rows (`kernels/cuda/flashnext_head_fp8.cu`).
//!
//! The head is the target round's one stage at the box's memory wall: 248,320 rows of 2,560 bf16
//! halves, 1.27 GB read a round, 5.29 ms at 240 GB/s. The artifact this module loads is the same
//! table in e4m3 — 0.64 GB of codes and 0.99 MB of one f32 scale an output row — so the stream
//! halves for the same tensor-core MACs. That rank is not exact, so it is only a *candidate* source:
//! `kcov = 64` ids a row are nominated from the FP8 table, then scored against the exact bf16 rows
//! the engine already holds resident (320 KB a row, gathered out of qmmf's tiled layout) and written
//! back over the FP8 logits before the row is rounded to bf16 for the sampler.
//!
//! Off unless `TF_FLASHNEXT_LMHEAD_FP8` names a directory (`"0"` and an empty value read as unset,
//! so a stale export cannot arm it): `Model.head_fp8` is then null, `projectHead` takes its shipped
//! branch on one null check, nothing here allocates, and no image is loaded.
//!
//! The artifact is written by `tools/dflash2/quantize/quantize_lm_head.py`, and its manifest is read
//! here for the geometry: a table fitted for another vocabulary or width is refused at load rather
//! than read at the wrong stride.
const std = @import("std");
const cuda = @import("cuda");
const fp8 = cuda.fp8;
const qmmf = cuda.qmmf;

/// The candidates a row is ranked down to before they are rescored exactly. 64 is the width the
/// drafter's head was measured at (`cuda_draft2_fwd.zig:head_kcov`): on real head-input rows the
/// FP8 head's top-64 held the bf16 head's top-16 for 1.0000 of them, and the exact rescore then made
/// the top-16 identical.
pub const kcov: usize = 64;

/// The entries one rank CTA sorts: a `tile`-wide slice of a row's vocabulary, 64 KiB of shared.
pub const tile: usize = 8192;

/// The rows one call may take, matching the pass scratch's `head_rows_one` (`cuda_scratch_sizes.zig`):
/// the staging below is sized once at load for this many. A call with more rows is refused by name
/// rather than silently ranked without its rescore.
pub const rows_max: usize = 32;

/// The first pass's CTAs a row: the tiles that cover the vocabulary, the last one fewer entries.
pub fn tilesOf(vocab: usize) usize {
    return (vocab + tile - 1) / tile;
}

/// `tf_fxlh_rank_merge`'s slot count: the smallest power of two at least `tiles * kcov`, because the
/// merge is a bitonic sort and the kernel takes the padded width as an argument (2,048 at the served
/// shape, 31 tiles x 64).
pub fn slots(vocab: usize) usize {
    var n: usize = 1;
    while (n < tilesOf(vocab) * kcov) n <<= 1;
    return n;
}

/// `TF_FLASHNEXT_LMHEAD_FP8`'s text: the artifact's directory, or null for today's bf16 head. Only
/// the path — whether the arm is taken is this function's, so `"0"` and an empty value are off and a
/// path that is not an artifact is refused at load rather than ignored.
pub fn dirFromEnv() ?[]const u8 {
    const p = std.c.getenv("TF_FLASHNEXT_LMHEAD_FP8") orelse return null;
    return choice(std.mem.span(p));
}

/// `dirFromEnv`'s parse, on the text alone: unset, empty and `"0"` are off, anything else is a path.
pub fn choice(text: []const u8) ?[]const u8 {
    if (text.len == 0 or std.mem.eql(u8, text, "0")) return null;
    return text;
}


/// The per-round staging the six launches leave behind, byte for byte: the rank's fp32 logits, its
/// per-tile spill, the surviving ids, their exact scores, and the gathered bf16 rows.
pub const Stage = struct {
    /// `[rows, vocab]` f32: the FP8 table's logits, patched with the exact survivors before the cast.
    lg: u64,
    /// `[tiles, rows, kcov]` packed `{fp8 logit, id}` rank keys, tile-major.
    part: u64,
    /// `[rows, kcov]` u32: the row's surviving ids, descending by FP8 logit.
    ids: u64,
    /// `[rows, kcov]` f32: the same ids' exact bf16 logits.
    resc: u64,
    /// `[rows * kcov, k]` bf16: the gathered exact rows the rescore reduces.
    gath: u64,
};

/// The staging's offsets, each 256-byte aligned; `placeStage` fills `out` with their addresses.
pub fn placeStage(vocab: usize, k: usize, base: u64, out: ?*Stage) !usize {
    const sizes = .{
        .{ "lg", try std.math.mul(usize, try std.math.mul(usize, rows_max, vocab), 4) },
        .{ "part", try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, tilesOf(vocab), rows_max), kcov), 8) },
        .{ "ids", rows_max * kcov * 4 },
        .{ "resc", rows_max * kcov * 4 },
        .{ "gath", try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, rows_max, kcov), k), 2) },
    };
    var total: usize = 0;
    inline for (sizes) |s| {
        const at = (total + 255) & ~@as(usize, 255);
        total = try std.math.add(usize, at, s[1]);
        if (out) |b| @field(b, s[0]) = base + at;
    }
    return total;
}

/// `placeStage(0, ...)`: 44.4 MiB at the served shape (31.8 MB of fp32 logits, 10.5 MB of gathered
/// rows, 0.5 MB of rank spill), against the 1.27 GB bf16 table the flag stops reading.
pub fn stageBytes(vocab: usize, k: usize) !usize {
    return placeStage(vocab, k, 0, null);
}

/// The FP8 table's own device bytes: `padded(vocab) * k` codes and `padded(vocab) * k / 64` fp32
/// scales (248,320 x 2,560 -> 635.7 MB + 39.7 MB).
pub fn tableBytes(vocab: usize, k: usize) !usize {
    return try std.math.add(usize, try std.math.mul(usize, fp8.padded(vocab), k), try std.math.mul(usize, try std.math.mul(usize, fp8.padded(vocab), k / qmmf.group), 4));
}

/// The whole arm's device bill, for the load log: the table and the staging.
pub fn bytes(vocab: usize, k: usize) !usize {
    return try std.math.add(usize, try tableBytes(vocab, k), try stageBytes(vocab, k));
}

/// e4m3 codes `[n, k]` row-major read with one f32 scale an output row, in the kernel's
/// `[npad/64][k/64][64]` fp32 scale tile: each 64-input group of a row takes that row's own scale
/// (`fp8.packScalesAt` for a block-FP8 checkpoint's 128 x 128 scales, which is the same tile).
///
/// The FP8G kernel applies a group's scale when its 64 inputs have been accumulated, so repeating
/// one scale over the `k / 64` groups is exactly `scale[row] * dot(code[row], x)` — no rounding
/// between the groups' partial sums and the scale, because the scale is a power-of-two-free f32 and
/// the multiply is the same one a single-scale path would do per group.
pub fn packRowScales(out: []f32, row_scale: []const f32, n: usize, k: usize) void {
    const npad = fp8.padded(n);
    const kg = k / qmmf.group;
    std.debug.assert(out.len == npad * kg and row_scale.len == n);
    for (0..npad / 64) |t| for (0..kg) |g| for (0..64) |c| {
        const row = t * 64 + c;
        out[(t * kg + g) * 64 + c] = if (row < n) row_scale[row] else 1.0;
    };
}

/// Element `i` of a row of the engine's tiled bf16 head (`qmmf.packBf16`) for vocabulary id `v` and
/// column `col`: the inverse of `packBf16`'s `(tg, jj, kt, lane, e)` decomposition, and the formula
/// `tiled_offset` in `kernels/cuda/flashnext_head_fp8.cu` gathers with. Pinned against `packBf16`
/// by the test at the end of this file — the CUDA copy of it is what a served rescore reads, and a
/// mismatch there would silently score the wrong rows.
pub fn tiledOffset(v: usize, col: usize, k: usize) usize {
    const kg = k / 64;
    const kb = v / 64;
    const rq = v % 64;
    const g = col / 64;
    const within = col % 64;
    const kt = within / 16;
    const rest = within % 16;
    const c = rest >> 3;
    const m = (rest >> 1) & 3;
    const e = 2 * c + (rest & 1);
    const lane = 4 * (rq % 8) + m;
    const tg = kb * kg + g;
    return tg * 4096 + (rq / 8) * 512 + kt * 128 + lane * 4 + e;
}

/// The quantised head, loaded: the FP8 table (packed for `qmmf`'s FP8G lane), the six entries that
/// rank, gather, rescore and write a row, and the staging they leave behind.
pub const Head = struct {
    gpa: std.mem.Allocator,
    mod: cuda.Module,
    /// A CTA's `kcov` best of its `tile`-wide slice, dynamic shared `tile * 8` (64 KiB, so the
    /// attribute is required).
    rank_tile: cuda.Function,
    /// A row's `tiles * kcov` candidates down to its `kcov` ids, dynamic shared `slots * 8`.
    rank_merge: cuda.Function,
    /// The exact bf16 rows of the survivors, out of the engine's tiled head.
    gather: cuda.Function,
    /// The exact bf16 dot of each survivor.
    rescore: cuda.Function,
    /// The exact logits over the FP8 ones, in the row's fp32 logits.
    patch: cuda.Function,
    /// The row rounded to bf16, which is what every arm writes.
    cast: cuda.Function,
    /// `fp8.zig`'s packed table: `npad * k` codes, then the `npad * k / 64` fp32 row scales.
    arena: cuda.DeviceBuffer,
    /// The view `qmmf.Lane.matmulF32` reads: `fp8g`, `n = vocab`, `k = hidden`, one K slice.
    w: qmmf.Weight,
    /// `lg`, `part`, `ids`, `resc`, `gath` (`Stage`), `stageBytes`.
    stage: cuda.DeviceBuffer,
    est: Stage,
    vocab: usize,
    k: usize,

    /// `dir`'s artifact, packed and uploaded, then the rank/rescore image. Every refusal is one a
    /// load should not survive: the manifest's geometry against the served one, the two files'
    /// lengths, and the FP8G plan the lane takes.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, vocab: usize, k: usize) !*Head {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        if (vocab == 0 or k == 0 or k % qmmf.group != 0 or vocab % 64 != 0) return error.HeadFp8Shape;
        try checkManifest(gpa, io, dir, vocab, k);
        const codes = try readExact(gpa, io, dir, name_codes, vocab * k);
        defer gpa.free(codes);
        const scale_bytes = try readExact(gpa, io, dir, name_scales, vocab * 4);
        defer gpa.free(scale_bytes);

        // the lane's fragment order, from the row-major codes the artifact stores
        const fragments = try gpa.alloc(u8, fp8.padded(vocab) * k);
        defer gpa.free(fragments);
        fp8.packCodes(fragments, codes, vocab, k);
        // and the per-row scales as the kernel's tile
        const scales = try gpa.alloc(f32, vocab);
        defer gpa.free(scales);
        for (scales, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, scale_bytes[i * 4 ..][0..4], .little));
        const groups = k / qmmf.group;
        const tile_scales = try gpa.alloc(f32, fp8.padded(vocab) * groups);
        defer gpa.free(tile_scales);
        packRowScales(tile_scales, scales, vocab, k);

        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        h.* = .{
            .gpa = gpa,
            .mod = undefined,    .rank_tile = undefined, .rank_merge = undefined,
            .gather = undefined, .rescore = undefined,   .patch = undefined,
            .cast = undefined,   .arena = undefined,     .w = undefined,
            .stage = undefined,  .est = undefined,       .vocab = vocab,
            .k = k,
        };
        const codes_bytes = fragments.len;
        const scale_float_bytes = std.mem.sliceAsBytes(tile_scales).len;
        h.arena = try cuda.DeviceBuffer.alloc(d, codes_bytes + scale_float_bytes);
        errdefer h.arena.free();
        try h.arena.upload(0, fragments);
        try h.arena.upload(codes_bytes, std.mem.sliceAsBytes(tile_scales));
        // one K slice: the FP8G plan then writes the fp32 logits itself, with no partials to reduce
        h.w = qmmf.pinned(fp8.weight(h.arena.ptr, h.arena.ptr + codes_bytes, vocab, k), 1) catch return error.HeadFp8Plan;
        h.stage = try cuda.DeviceBuffer.alloc(d, try stageBytes(vocab, k));
        errdefer h.stage.free();
        _ = try placeStage(vocab, k, h.stage.ptr, &h.est);
        h.mod = try cuda.Module.load(d, cuda.kernels.flashnext_head_fp8);
        errdefer h.mod.unload();
        h.rank_tile = try h.mod.function("tf_fxlh_rank_tile");
        try h.rank_tile.allowDynamicShared(tile * 8);
        h.rank_merge = try h.mod.function("tf_fxlh_rank_merge");
        try h.rank_merge.allowDynamicShared(@intCast(slots(vocab) * 8));
        h.gather = try h.mod.function("tf_fxlh_gather");
        h.rescore = try h.mod.function("tf_fxlh_rescore");
        h.patch = try h.mod.function("tf_fxlh_patch");
        h.cast = try h.mod.function("tf_fxlh_cast");
        std.log.info("flashnext: lm_head fp8 armed from {s}: {d} MB of codes, {d} MB of row scales, {d} MiB of staging; top-{d} rescored in bf16", .{
            dir,
            codes_bytes / 1_000_000,
            scale_float_bytes / 1_000_000,
            try stageBytes(vocab, k) / (1 << 20),
            kcov,
        });
        return h;
    }

    pub fn deinit(h: *Head) void {
        h.stage.free();
        h.arena.free();
        h.mod.unload();
        h.gpa.destroy(h);
    }

    /// One head over `n` rows into `out` (bf16 logits, `vocab * 2` bytes a row), in six launches:
    /// the FP8G GEMM, the two rank selects, the gather, the exact dot, the patch and the cast.
    ///
    /// `pass` is the batched pass (its `s.mixed` is the head's input rows, `m.head` the engine's
    /// tiled bf16 table the rescore gathers from, and `k.native_lane` the packed-weight lane).
    pub fn project(h: *Head, pass: anytype, out: u64, n: usize) !void {
        if (n == 0 or n > rows_max) return error.HeadFp8RowsUnsupported;
        const stream = pass.stream;
        const tiles = tilesOf(h.vocab);
        // 1. the rank's GEMM, over the quantised table
        try pass.k.native_lane.matmulF32(stream, pass.s.mixed, h.k, n, h.w, h.est.lg, 0);
        // 2. one candidate list a tile, 3. the row's list of the tiles' lists
        try launch(stream, h.rank_tile, .{ tiles, n, 1 }, 256, tile * 8, .{
            h.est.lg, h.est.part, ci(h.vocab), ci(kcov), ci(n), ci(tile),
        });
        try launch(stream, h.rank_merge, .{ n, 1, 1 }, 256, @intCast(slots(h.vocab) * 8), .{
            h.est.part, h.est.ids, ci(tiles), ci(n), ci(kcov), ci(slots(h.vocab)),
        });
        // 4. the survivors' exact bf16 rows, 5. their exact logits
        const cands = n * kcov;
        try launch(stream, h.gather, .{ cands, 1, 1 }, 128, 0, .{
            pass.m.head.w.ptr, h.est.ids, h.est.gath, ci(n), ci(kcov), ci(h.k),
        });
        try launch(stream, h.rescore, .{ (cands + 127) / 128, 1, 1 }, 128, 0, .{
            h.est.gath, pass.s.mixed, h.est.resc, ci(cands), ci(kcov), ci(h.k),
        });
        // 6. the exact values over the FP8 ones, then the row as bf16 — the shape every arm writes
        try launch(stream, h.patch, .{ n, 1, 1 }, 128, 0, .{
            h.est.lg, h.est.ids, h.est.resc, ci(h.vocab), ci(kcov), ci(n),
        });
        const total = n * h.vocab;
        try launch(stream, h.cast, .{ @min(@as(usize, 4096), (total + 255) / 256), 1, 1 }, 256, 0, .{
            h.est.lg, out, @as(i64, @intCast(total)), ci(h.vocab),
        });
    }
};

const name_codes = "head.fp8.e4m3.bin";
const name_scales = "head.fp8.scales.f32";
const name_manifest = "manifest.json";

fn path(gpa: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    return std.fs.path.join(gpa, &.{ dir, name });
}

/// `dir`'s `name`, exactly `len` bytes: a short or long file is a different table, not a table to
/// read at a guessed stride.
fn readExact(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, len: usize) ![]u8 {
    const p = try path(gpa, dir, name);
    defer gpa.free(p);
    const data = std.Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(len + 1)) catch |e| {
        std.log.err("flashnext: the quantised lm_head's {s} is missing or unreadable ({s}) at {s}", .{ name, @errorName(e), p });
        return e;
    };
    if (data.len != len) {
        gpa.free(data);
        return error.HeadFp8ArtifactShape;
    }
    return data;
}

/// The artifact's manifest, checked for the geometry this engine serves: a table fitted for another
/// vocabulary or width would be read at the wrong stride and produce plausible, wrong logits.
fn checkManifest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, vocab: usize, k: usize) !void {
    const p = try path(gpa, dir, name_manifest);
    defer gpa.free(p);
    const text = std.Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(1 << 20)) catch |e| {
        std.log.err("flashnext: the quantised lm_head needs {s} beside its tables ({s})", .{ name_manifest, @errorName(e) });
        return e;
    };
    defer gpa.free(text);
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return error.HeadFp8Manifest;
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return error.HeadFp8Manifest,
    };
    const format = switch (obj.get("format") orelse return error.HeadFp8Manifest) {
        .string => |s| s,
        else => return error.HeadFp8Manifest,
    };
    if (!std.mem.eql(u8, format, manifest_format)) return error.HeadFp8Manifest;
    const shape = switch (obj.get("shape") orelse return error.HeadFp8Manifest) {
        .array => |a| a,
        else => return error.HeadFp8Manifest,
    };
    if (shape.items.len != 2) return error.HeadFp8Manifest;
    const rows: usize = switch (shape.items[0]) {
        .integer => |v| if (v >= 0) @intCast(v) else return error.HeadFp8Manifest,
        else => return error.HeadFp8Manifest,
    };
    const cols: usize = switch (shape.items[1]) {
        .integer => |v| if (v >= 0) @intCast(v) else return error.HeadFp8Manifest,
        else => return error.HeadFp8Manifest,
    };
    if (rows != vocab or cols != k) {
        std.log.err("flashnext: the quantised lm_head is {d} x {d}, this engine serves {d} x {d}", .{ rows, cols, vocab, k });
        return error.HeadFp8ArtifactShape;
    }
}

/// The manifest's `format`, which `tools/dflash2/quantize/quantize_lm_head.py` writes.
pub const manifest_format = "flashnext-lmhead-fp8-rtn";

fn launch(stream: cuda.Stream, f: cuda.Function, grid: [3]usize, threads: u32, shared: u32, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |value| a.add(value);
    try cuda.launch.launch(f, .{
        .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) },
        .block = .{ .x = threads },
        .shared = shared,
    }, stream, &a);
}

fn ci(x: usize) c_int {
    return @intCast(x);
}

test "Flash Next head fp8: per-row scales fill the kernel's 64-input groups, padded rows one" {
    const n = 130;
    const k = 256;
    const gpa = std.testing.allocator;
    const row = try gpa.alloc(f32, n);
    defer gpa.free(row);
    for (row, 0..) |*v, i| v.* = @floatFromInt(1 + i);
    const out = try gpa.alloc(f32, fp8.padded(n) * (k / qmmf.group));
    defer gpa.free(out);
    packRowScales(out, row, n, k);
    // row 26, its fourth 64-input group: its own scale, not its 64-row block's or its neighbour's
    try std.testing.expectEqual(@as(f32, 27), out[((0 * 4 + 3) * 64) + 26]);
    // the next 64-row block's first row: its own scale
    try std.testing.expectEqual(@as(f32, 65), out[((1 * 4 + 0) * 64) + 0]);
    // every padded row reads 1.0, so a padded code of zero contributes nothing: row 190 is past 130
    try std.testing.expectEqual(@as(f32, 1), out[((2 * 4 + 0) * 64) + 62]);
}

test "Flash Next head fp8: the gather's tiled offset is packBf16's inverse, every element" {
    const n = 130;
    const k = 256;
    const npad = (n + 127) / 128 * 128;
    const gpa = std.testing.allocator;
    const w = try gpa.alloc(u16, n * k);
    defer gpa.free(w);
    for (w, 0..) |*x, i| x.* = @truncate(i * 31 + 7);
    const tiled = try gpa.alloc(u16, npad * k);
    defer gpa.free(tiled);
    qmmf.packBf16(tiled, w, n, k);
    for (0..n) |v| for (0..k) |col| {
        if (tiled[tiledOffset(v, col, k)] != w[v * k + col]) {
            std.debug.print("gather miss: id {d} col {d}: tiled {d} vs row-major {d}\n", .{ v, col, tiled[tiledOffset(v, col, k)], w[v * k + col] });
            return error.TestUnexpectedResult;
        }
    };
}

test "Flash Next head fp8: the flag is off unless a path is named, and 0 is off" {
    try std.testing.expect(choice("") == null);
    try std.testing.expect(choice("0") == null);
    try std.testing.expectEqualStrings("/tmp/head", choice("/tmp/head").?);
}
