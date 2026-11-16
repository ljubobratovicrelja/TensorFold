//! TF_FLASHNEXT_TAP_DUMP=<dir>: capture one prompt pass's tap ring to disk.
//!
//! Research only, and off unless the variable names a directory: it exists to answer
//! `docs/design/flashnext-dflash2-drafter.md` step 0 — which of a tapped layer's rows the
//! DFlash2 drafter was trained against — without a drafter in the engine yet. The pass sets the
//! five tap layers and a real ring (`Hooks.tap_ring`), so the rows it writes are the same bytes a
//! drafted run will read; nothing here is a second implementation of the tap.
//!
//! Layout in <dir>: `ids.bin` (u32 LE, the tokens the pass actually read, so a consumer never has
//! to re-tokenize), `ring.bin` (the ring's slots, `rows x n_taps x hidden` bf16, slot = pos % rows)
//! and `manifest.json` naming the geometry and the last position written.
const std = @import("std");
const cuda = @import("cuda");
const parts = @import("cuda_batch_state.zig");
const tapring = @import("cuda_tap_ring.zig");

/// The drafter's `target_layer_ids`, in its own tap order.
pub const taps: [5]u32 = .{ 5, 15, 25, 35, 45 };

/// TF_FLASHNEXT_TAP_LAYERS=a,b,c,d,e: which layers a capture taps, so the off-by-one between the
/// engine's layer index and the checkpoint's `hidden_states[i]` can be measured without a rebuild.
/// Anything malformed falls back to the drafter's own list rather than guessing a layer.
pub fn layersFromEnv() [5]u32 {
    const v = std.c.getenv("TF_FLASHNEXT_TAP_LAYERS") orelse return taps;
    var out: [5]u32 = taps;
    var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
    var i: usize = 0;
    while (it.next()) |part| : (i += 1) {
        if (i == out.len) return taps;
        const n = std.fmt.parseInt(u32, std.mem.trim(u8, part, " "), 10) catch return taps;
        if (n >= 48) return taps;
        out[i] = n;
    }
    return if (i == out.len) out else taps;
}
/// The drafter's window, from its config's `sliding_window`.
pub const window: usize = 2048;
/// The drafter's `block_size`: a ring holds a verify block past the window.
pub const block: usize = 8;
pub const hidden: usize = 2560;

/// A capture in flight: the ring, its host mirror, and the tokens the pass read.
pub const Active = struct {
    dir: []const u8,
    /// The layers this capture taps (TF_FLASHNEXT_TAP_LAYERS or the drafter's own list).
    layers: [5]u32 = taps,
    rows: usize,
    ring: cuda.DeviceBuffer,
    host: []u8,
    ids: []u32,
    pos_end: usize = 0,
    source: tapring.TapSource = tapring.TapSource.default,

    /// The pass's hooks with the tap ring attached.
    pub fn hooks(a: *const Active) parts.Hooks {
        var t: [8]u32 = @splat(0);
        for (a.layers, 0..) |v, i| t[i] = v;
        return .{
            .taps = t,
            .n_taps = a.layers.len,
            .tap_ring = a.ring.ptr,
            .tap_rows = a.rows,
            .tap_source = a.source,
        };
    }

    pub fn deinit(a: *Active, gpa: std.mem.Allocator) void {
        a.ring.free();
        gpa.free(a.host);
        gpa.free(a.ids);
    }
};

/// The directory TF_FLASHNEXT_TAP_DUMP names, or null. An empty value is off, not an error.
pub fn dirFromEnv() ?[]const u8 {
    const v = std.c.getenv("TF_FLASHNEXT_TAP_DUMP") orelse return null;
    const text = std.mem.span(v);
    return if (text.len == 0) null else text;
}

/// Bytes a ring of `rows` slots holds.
pub fn ringBytes(rows: usize) usize {
    return rows * taps.len * hidden * 2;
}

/// A capture over `ids`: allocates the ring and copies the ids. Refuses a pass that spans more than
/// one chunk, because the ring's slot for a position is written from that chunk's rows alone.
pub fn begin(gpa: std.mem.Allocator, driver: *cuda.Driver, dir: []const u8, ids: []const u32, chunk: usize) !Active {
    if (ids.len > chunk) return error.TapDumpChunkedPass;
    const rows = tapring.rowsFor(window, block);
    const bytes = ringBytes(rows);
    var ring = try cuda.DeviceBuffer.alloc(driver, bytes);
    errdefer ring.free();
    const host = try gpa.alloc(u8, bytes);
    errdefer gpa.free(host);
    const copy = try gpa.dupe(u32, ids);
    return .{ .dir = dir, .layers = layersFromEnv(), .rows = rows, .ring = ring, .host = host, .ids = copy, .pos_end = ids.len, .source = tapring.sourceFromEnv() };
}

/// Download the ring and write the three files. Called once, after the pass the capture opened on.
pub fn flush(a: *Active, gpa: std.mem.Allocator, io: std.Io, d: *cuda.Driver, dir: []const u8) !void {
    try cuda.memory.download(d, a.ring.ptr, a.host);
    try writeAll(io, dir, "ring.bin", a.host);

    const ids_bytes = std.mem.sliceAsBytes(a.ids);
    try writeAll(io, dir, "ids.bin", ids_bytes);

    var text: std.Io.Writer.Allocating = .init(gpa);
    defer text.deinit();
    const w = &text.writer;
    try w.print("{{\"rows\":{d},\"taps\":[{d},{d},{d},{d},{d}],\"hidden\":{d},\"window\":{d},\"block\":{d},\"source\":\"{s}\",\"pos_end\":{d},\"ids_file\":\"ids.bin\",\"ring_file\":\"ring.bin\",\"row_bytes\":{d}}}\n", .{
        a.rows,
        a.layers[0], a.layers[1], a.layers[2], a.layers[3], a.layers[4],
        hidden,
        window,
        block,
        @tagName(a.source),
        a.pos_end,
        taps.len * hidden * 2,
    });
    try writeAll(io, dir, "manifest.json", text.written());
}

fn writeAll(io: std.Io, dir: []const u8, name: []const u8, bytes: []const u8) !void {
    var path_buf: [1024]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, name });
    const d = std.Io.Dir.cwd();
    try d.createDirPath(io, dir);
    try d.writeFile(io, .{ .sub_path = path, .data = bytes });
}

test "a ring's bytes are the drafter's fixed cost, not the context's" {
    const rows = tapring.rowsFor(window, block);
    try std.testing.expectEqual(@as(usize, 2064), rows);
    // 2064 x 5 x 2560 x 2 = 52.8 MB a stream, whatever the served window is
    try std.testing.expectEqual(@as(usize, 52_838_400), ringBytes(rows));
}
