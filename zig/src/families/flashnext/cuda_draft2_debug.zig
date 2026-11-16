//! TF_FLASHNEXT_DRAFT_DEBUG=<dir>: leave one draft-check round's internals on disk.
//!
//! Research only, off unless the variable names a directory, and inert for the engine otherwise: it
//! is the analyser half of `cuda_draft2_check.zig`. The check answers "does the CUDA round land the
//! same tokens as `tools/dflash2/reference.py`"; when the answer is no, a single mean advance cannot
//! say *which stage* disagreed. This module is the file format that makes the stages separable:
//!
//! ```
//! <dir>/round<k>/ctx_fused.bin    [ctx_len, hidden]   bf16  tf_fxd2_fuse's whole context span
//! <dir>/round<k>/embed.bin        [block, hidden]     bf16  the block's input rows (anchor + masks)
//! <dir>/round<k>/layer<i>.bin     [block, hidden]     bf16  the block hidden after layer i
//! <dir>/round<k>/hidden_normed.bin[block-1, hidden]   bf16  the final norm the head reads
//! <dir>/round<k>/head_ids.bin     [block-1, topk]     u32   tf_fxd2_head_topk's candidate ids
//! <dir>/round<k>/head_unary.bin   [block-1, topk]     f32   and their unary logits
//! <dir>/round<k>/proposals.bin    [block-1]           u32   tf_fxd2_select's walk
//! <dir>/round<k>/tensors.json                            shape, dtype and round metadata
//! ```
//!
//! Every file is raw little-endian, no header, in the shape `tensors.json` names: the consumer is
//! `tools/dflash2/draft_compare.py`, which loads them beside `reference.py` and prints a stage by
//! stage difference. `tensors.json` is written last, so a directory without one is a dump that did
//! not finish rather than a dump with tensors missing.
//!
//! Which rounds: TF_FLASHNEXT_DRAFT_DEBUG_ROUNDS (a comma list of the walk's round indexes, default
//! `0,1`). Round indexes count the check's own loop, so round `k` is the `k`-th accepted-or-not
//! block, not position `k`; `tensors.json` records the `p` the round actually sat at, which is what
//! a replay needs. A round at `p` earlier than a few dozen tokens is a weak test of the fusion and
//! the context attention (round 0's context is one row), so a run that is diagnosing those should
//! name a later round by hand: `TF_FLASHNEXT_DRAFT_DEBUG_ROUNDS=0,64`.
const std = @import("std");

/// Rounds dumped when TF_FLASHNEXT_DRAFT_DEBUG_ROUNDS says nothing.
pub const default_rounds: [2]usize = .{ 0, 1 };

/// A round list read from the environment holds at most this many entries.
pub const max_rounds = 8;

/// One file of a round's dump: where it goes, what shape it is, and its bytes.
pub const Item = struct {
    /// The tensor's name in `reference.py`'s own words (`ctx_fused`, `layer3`, ...).
    name: []const u8,
    /// The file name inside the round's directory.
    file: []const u8,
    rows: usize,
    cols: usize,
    /// `bf16`, `u32` or `f32`: the file is the raw span, so a consumer needs the dtype to read it.
    dtype: []const u8,
    bytes: []const u8,
};

/// The round a dump belongs to: what a replay needs to sit where the engine sat.
pub const Meta = struct {
    /// The check's own round index (the `k` in `round<k>`).
    round: usize,
    /// The anchor position the round drafted from.
    p: usize,
    /// Rows in the round's context span, and the position of its first row.
    ctx_len: usize,
    ctx_lo: usize,
    /// The block's rows sit at `position .. position + block`.
    position: u64,
    /// The token at row 0 of the block.
    anchor: u32,
    block: usize,
    hidden: usize,
    layers: usize,
    topk: usize,
    vocab: usize,
    /// The tap capture the round read, and how many tokens it held.
    capture: []const u8,
    ids_len: usize,
};

/// The directory TF_FLASHNEXT_DRAFT_DEBUG names, or null. An empty value is off, not an error.
pub fn dirFromEnv() ?[]const u8 {
    const v = std.c.getenv("TF_FLASHNEXT_DRAFT_DEBUG") orelse return null;
    const text = std.mem.span(v);
    return if (text.len == 0) null else text;
}

/// What the environment asks for: the directory and the rounds to dump. Null when the variable is
/// unset or empty, which is every production run.
pub fn specFromEnv() ?Spec {
    const dir = dirFromEnv() orelse return null;
    var s: Spec = .{ .dir = dir };
    s.rounds[0] = default_rounds[0];
    s.rounds[1] = default_rounds[1];
    s.n = default_rounds.len;
    const v = std.c.getenv("TF_FLASHNEXT_DRAFT_DEBUG_ROUNDS") orelse return s;
    var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
    var built: usize = 0;
    while (it.next()) |part| {
        const n = std.fmt.parseInt(usize, std.mem.trim(u8, part, " \t"), 10) catch continue;
        if (built == max_rounds) break;
        s.rounds[built] = n;
        built += 1;
    }
    // A list with no readable entry is the default rather than an empty set, which would look like a
    // run that asked for dumps and got none.
    if (built == 0) return s;
    s.n = built;
    return s;
}

pub const Spec = struct {
    dir: []const u8,
    rounds: [max_rounds]usize = @splat(0),
    n: usize = 0,

    /// Whether round `round` is one the environment asked for.
    pub fn wants(s: Spec, round: usize) bool {
        for (s.rounds[0..s.n]) |r| if (r == round) return true;
        return false;
    }
};

/// Write one round's dump: every item as its own `.bin`, then the `tensors.json` that describes them.
pub fn writeRound(gpa: std.mem.Allocator, io: std.Io, spec: Spec, meta: Meta, items: []const Item) !void {
    // two buffers, and never one written while it is read: the round's directory is the prefix of
    // every file path, so it has to survive the formatting of the paths themselves
    var dir_buf: [1024]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, "{s}/round{d}", .{ spec.dir, meta.round });
    var path_buf: [1024]u8 = undefined;
    const d = std.Io.Dir.cwd();
    try d.createDirPath(io, dir);
    for (items) |item| {
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, item.file });
        try d.writeFile(io, .{ .sub_path = path, .data = item.bytes });
    }
    var text: std.Io.Writer.Allocating = .init(gpa);
    defer text.deinit();
    try writeJson(&text.writer, meta, items);
    const path = try std.fmt.bufPrint(&path_buf, "{s}/tensors.json", .{dir});
    try d.writeFile(io, .{ .sub_path = path, .data = text.written() });
}

/// The round's metadata and one `{name, file, shape, dtype, bytes}` a tensor, on one line.
pub fn writeJson(w: *std.Io.Writer, meta: Meta, items: []const Item) !void {
    try w.print(
        "{{\"round\":{d},\"p\":{d},\"ctx_len\":{d},\"ctx_lo\":{d},\"position\":{d},\"anchor\":{d}," ++
            "\"block\":{d},\"hidden\":{d},\"layers\":{d},\"topk\":{d},\"vocab\":{d}," ++
            "\"capture\":\"{s}\",\"ids_len\":{d},\"tensors\":[",
        .{
            meta.round,        meta.p,        meta.ctx_len, meta.ctx_lo, meta.position,
            meta.anchor,       meta.block,    meta.hidden,  meta.layers, meta.topk,
            meta.vocab,        meta.capture,  meta.ids_len,
        },
    );
    for (items, 0..) |item, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{{\"name\":\"{s}\",\"file\":\"{s}\",\"shape\":[{d},{d}],\"dtype\":\"{s}\",\"bytes\":{d}}}", .{
            item.name, item.file, item.rows, item.cols, item.dtype, item.bytes.len,
        });
    }
    try w.writeAll("]}\n");
}

// -----------------------------------------------------------------------------------------------
// Tests: the environment's spelling and the file's shape. Nothing here touches a device or a file.
// -----------------------------------------------------------------------------------------------

test "a round list is the rounds it names, and the default is the walk's first two" {
    var s: Spec = .{ .dir = "/tmp/x" };
    s.rounds[0] = default_rounds[0];
    s.rounds[1] = default_rounds[1];
    s.n = default_rounds.len;
    try std.testing.expect(s.wants(0) and s.wants(1));
    try std.testing.expect(!s.wants(2));
    // a list replaces the default rather than adding to it
    var t: Spec = .{ .dir = "/tmp/x", .rounds = @splat(0), .n = 2 };
    t.rounds[0] = 0;
    t.rounds[1] = 64;
    try std.testing.expect(t.wants(0) and t.wants(64));
    try std.testing.expect(!t.wants(1));
}

test "a round's json names every tensor and the geometry a replay needs" {
    const a = std.testing.allocator;
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    const items: [2]Item = .{
        .{ .name = "embed", .file = "embed.bin", .rows = 8, .cols = 2560, .dtype = "bf16", .bytes = &.{ 0, 0 } },
        .{ .name = "proposals", .file = "proposals.bin", .rows = 7, .cols = 1, .dtype = "u32", .bytes = &.{ 1, 0, 0, 0 } },
    };
    try writeJson(&text.writer, .{
        .round = 3,
        .p = 17,
        .ctx_len = 18,
        .ctx_lo = 0,
        .position = 17,
        .anchor = 1234,
        .block = 8,
        .hidden = 2560,
        .layers = 5,
        .topk = 16,
        .vocab = 248_320,
        .capture = "/tmp/tapdump",
        .ids_len = 1451,
    }, &items);
    const json = text.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"round\":3") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"ctx_len\":18") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"proposals\",\"file\":\"proposals.bin\",\"shape\":[7,1],\"dtype\":\"u32\"") != null);
    // one closing bracket a tensor and one for the list: a consumer that counts them reads the shape
    try std.testing.expect(std.mem.count(u8, json, "\"dtype\"") == items.len);
}


