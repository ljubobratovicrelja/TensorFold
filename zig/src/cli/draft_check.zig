//! `tensorfold draft-check` — one DFlash2 round over a pre-collected fixture, with only the
//! tensors that round reads resident.
//!
//! The serving path reaches the same check by booting the whole target (48 layers, ~79 GiB, ~60 s)
//! and hooking the prompt path. A round reads no decoder layer: it needs the fixture ring, the
//! drafter checkpoint, the target's embedding table (to embed the anchor) and the target's head
//! (for the top-k proposals). This command loads exactly those — a few GiB, seconds — so the
//! regression scaffold can iterate at a sane price.
//!
//!     tensorfold draft-check --drafter DIR --target DIR --fixture DIR --out DIR [--repeat N]
//!
//! Each repetition writes `--out/rep{i}` with `stages.json` and its tensors, the layout the
//! fixture harness writes, so `tools/dflash2/analysis/regress.py` reads either source unchanged.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const draftcheck = @import("flashnext").draft2_check;

const embed_name = "model.language_model.embed_tokens.weight";
const head_name = "lm_head.weight";
/// `packBf16` writes 64-row blocks; the head is tiled and uploaded one block at a time.
const tile_rows: usize = 64;

const usage =
    \\usage: tensorfold draft-check --drafter DIR --target DIR --fixture DIR --out DIR [--repeat N]
    \\                              [--drafter-precision bf16|fp8|affine4]
    \\
    \\  --drafter DIR   the DFlash2 drafter checkpoint
    \\  --drafter-precision P
    \\                  the drafter's weight precision: bf16 (as published), fp8 (weight-only, W8A16),
    \\                  or affine4 (4-bit MLX codes, the arm `affine4/manifest.json` names).
    \\                  Omitted, the checkpoint's own contents decide, and a checkpoint whose declared
    \\                  precision and contents disagree is refused.
    \\  --head-fp8 DIR  a block-FP8 copy of the target's lm_head (`head.fp8.safetensors` +
    \\                  `manifest.json`, as tools/dflash2/quantize/fit_head.py writes it) to rank
    \\                  through instead of the bf16 head. Off unless `TF_FLASHNEXT_HEAD_FP8=1` is
    \\                  ALSO set: either one alone is refused, so a run never swaps its head
    \\                  silently. The 64 surviving ids are rescored against the bf16 head, so the
    \\                  proposals stay exact (measured: bit-identical acceptance on 880 positions).
    \\  --target DIR    the target checkpoint; only its embed table and head are read
    \\  --fixture DIR   a pre-collected ring (manifest.json, ids.bin, ring.bin)
    \\  --out DIR       where rep{i}/ is written
    \\  --repeat N      rounds to capture on ONE state (default: every `positions` entry in the
    \\                  manifest, else 1). N == 1 writes into --out itself; N > 1 writes
    \\                  --out/rep{i}. A manifest `positions` list advancing by the accepted run is
    \\                  the served regime; one position repeated is the cold round every time.
    \\
;

pub fn wants(args: []const []const u8) bool {
    return args.len > 0 and std.mem.eql(u8, args[0], "draft-check");
}

const Args = struct {
    drafter: []const u8 = "",
    target: []const u8 = "",
    fixture: []const u8 = "",
    out: []const u8 = "",
    repeat: usize = 0,
    precision: ?draftcheck.Precision = null,
    /// The quantised head's artifact (`--head-fp8`), or null: read only under
    /// `TF_FLASHNEXT_HEAD_FP8=1`, and a path without the flag is refused at `open`.
    head_fp8: ?[]const u8 = null,
};

fn parse(args: []const []const u8) !Args {
    var a: Args = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (i + 1 >= args.len) return error.MissingValue;
        const v = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, flag, "--drafter")) {
            a.drafter = v;
        } else if (std.mem.eql(u8, flag, "--target")) {
            a.target = v;
        } else if (std.mem.eql(u8, flag, "--fixture")) {
            a.fixture = v;
        } else if (std.mem.eql(u8, flag, "--out")) {
            a.out = v;
        } else if (std.mem.eql(u8, flag, "--repeat")) {
            a.repeat = std.fmt.parseInt(usize, v, 10) catch return error.BadRepeat;
        } else if (std.mem.eql(u8, flag, "--drafter-precision")) {
            a.precision = draftcheck.Precision.parse(v) orelse return error.BadDrafterPrecision;
        } else if (std.mem.eql(u8, flag, "--head-fp8")) {
            a.head_fp8 = v;
        } else return error.UnknownFlag;
    }
    if (a.drafter.len == 0 or a.target.len == 0 or a.fixture.len == 0 or a.out.len == 0)
        return error.MissingFlag;
    return a;
}

/// The target's dense `[n, k]` bf16 head, packed into the matrix-core tiles the head path reads.
/// Blockwise, so the host never holds more than one 64-row block at a time.
fn tiledHead(
    gpa: std.mem.Allocator,
    driver: *cuda.Driver,
    ck: *core.Checkpoint,
    n: usize,
    k: usize,
) !cuda.DeviceBuffer {
    const npad = (n + 127) / 128 * 128;
    // `packBf16` takes the whole matrix: its `out` must be `npad * k` and its `w` exactly `n * k`,
    // so it cannot be driven a 64-row block at a time. The head is 1.27 GiB, which is fine here.
    const w = try gpa.alloc(u16, n * k);
    defer gpa.free(w);
    const tiled = try gpa.alloc(u16, npad * k);
    defer gpa.free(tiled);
    @memset(tiled, 0);

    const t = try ck.expect(head_name, .bf16, &.{ n, k });
    @memcpy(std.mem.sliceAsBytes(w), t.bytes[0 .. n * k * 2]);
    cuda.qmmf.packBf16(tiled, w, n, k);

    var out = try cuda.DeviceBuffer.alloc(driver, npad * k * 2);
    errdefer out.free();
    try out.upload(0, std.mem.sliceAsBytes(tiled));
    return out;
}

pub fn main(init: std.process.Init, argv: []const [:0]const u8) !u8 {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try gpa.alloc([]const u8, argv.len);
    for (args, argv) |*x, y| x.* = y;

    const o = parse(args) catch {
        try std.Io.File.stderr().writeStreamingAll(io, usage);
        return 2;
    };

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const stream = try cuda.Stream.init(&driver, true);

    // the target's two tensors, and nothing else of its 48 layers
    var ck = try core.Checkpoint.openModel(gpa, io, o.target);
    defer {}
    const emb = try ck.get(embed_name);
    if (emb.dtype != .bf16) return error.EmbedNotBf16;
    const vocab = emb.shape[0];
    const hidden = emb.shape[1];
    var embed = try cuda.DeviceBuffer.alloc(&driver, emb.bytes.len);
    defer embed.free();
    try embed.upload(0, emb.bytes);
    var head = try tiledHead(gpa, &driver, &ck, vocab, hidden);
    defer head.free();

    const m: draftcheck.Model = .{
        .embed = embed.ptr,
        .embed_rows = vocab,
        .head = head.ptr,
        .head_rows = vocab,
        .head_cols = hidden,
        .head_bf16 = true,
    };

    try draftcheck.runFixtureAt(
        gpa,
        io,
        &driver,
        stream,
        o.drafter,
        m,
        o.fixture,
        o.out,
        o.repeat,
        o.precision,
        o.head_fp8,
    );
    return 0;
}
