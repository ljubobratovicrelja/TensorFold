//! TF_FLASHNEXT_EXPERT_DUMP=<dir> / TF_FLASHNEXT_HIDDEN_DUMP=<dir>: what a pass routed, and how its
//! hidden rows move through the 48 layers. Research only, and off unless a variable names a directory.
//!
//! Two independent captures ride the same object, both read at points the pass already has the data:
//!
//!   * `Expert` — each layer's top-k picks and gate weights (`Scratch.picks` / `.wts`, the arrays
//!     `topkMany` leaves behind), copied device-to-device per layer and written out once per pass.
//!     With TF_FLASHNEXT_ROUTER_LOGITS=<rows> the first multi-row pass's full router logits
//!     (`Scratch.router_l`, 512 experts + the shared gate) are kept too, once, for a real softmax
//!     entropy rather than the top-k approximation.
//!   * `Hidden` — the post-norm hidden row the MoE reads (`Scratch.mixed`, `dims` bf16 a row) for
//!     every layer, on the first pass of two or more rows: the same rows at every depth, which is
//!     what a per-layer norm / cosine-drift / CKA curve needs.
//!
//! Cost model. Unset, nothing is allocated, no kernel is launched and every site in the pass is a
//! null check on `Pass.probe`. Set, the copies are D2D inside the pass (tens of kilobytes a layer,
//! no synchronise), and the ONE read-back is a `cuStreamSynchronize` plus a download at the pass's
//! end — the pass has already synchronised for its logits at that point (a decode round in
//! `Engine.round` copies the scores back right after), so the added wait is a few microseconds.
//! A prompt pass of 4096 rows costs 4096 x 48 x 88 B = 17 MB on the device and 17 MB in the dump.
//!
//! On disk, one file a pass, so a killed server loses at most the pass in flight:
//!
//!   p<N>.bin      u32 magic "TFEX", u32 rows, u32 pos0, u32 layers, u32 slots,
//!                 then ids [layers][rows][slots] u32 LE, then wts [layers][rows][slots] f32 LE
//!   hidden.bin    u32 magic "TFHD", u32 layers, u32 keep, u32 dims, u32 rows_of_pass, u32 k, u32 pos0,
//!                 then [layers][keep][dims] bf16 LE (rows [0, k) of each layer's block are real)
//!   router_logits.bin  u32 magic "TFRT", u32 layers, u32 rows, u32 width,
//!                 then [layers][rows][width] f32 LE (width = experts + 1, the shared gate last)
//!   manifest.json geometry, the two layouts, and how many passes were written.
//!
//! A capture cannot hold this: `cuStreamSynchronize` inside a capture is illegal. `Pass.chunk`
//! refuses a pass whose stream is already capturing (`error.ProbeInCapture`) before any copy, the
//! same rule `cuda_layer_time.zig` states for its events — serve with TF_FLASHNEXT_WINDOW_GRAPH=0.
const std = @import("std");
const cuda = @import("cuda");
const sys = @import("sys");

pub const expert_env: [:0]const u8 = "TF_FLASHNEXT_EXPERT_DUMP";
pub const hidden_env: [:0]const u8 = "TF_FLASHNEXT_HIDDEN_DUMP";
pub const logits_env: [:0]const u8 = "TF_FLASHNEXT_ROUTER_LOGITS";
pub const hidden_rows_env: [:0]const u8 = "TF_FLASHNEXT_HIDDEN_ROWS";

/// Rows of router logits the logits capture keeps: 128 x 48 x 513 x 4 B = 12.6 MB a pass.
pub const default_logit_rows: usize = 128;
/// Hidden rows a layer keeps: 32 x 48 x 2560 x 2 B = 7.9 MB.
pub const default_hidden_rows: usize = 32;
/// The fewest rows a pass must have to be worth a hidden capture (a decode round of one is not).
pub const hidden_min_rows: usize = 2;

const ex_magic: u32 = 0x5845_4654; // bytes "TFEX" little-endian
const hd_magic: u32 = 0x4448_4654; // "TFHD"
const rt_magic: u32 = 0x5452_4654; // "TFRT"

/// A directory from the environment, or null: unset or blank is off, never an error (as TF_FLASHNEXT_TAP_DUMP).
fn dirOf(name: [:0]const u8) ?[]const u8 {
    const v = sys.getenv(name) orelse return null;
    const text = std.mem.trim(u8, v, " \t");
    return if (text.len == 0) null else text;
}

fn countOf(name: [:0]const u8, default: usize) usize {
    const v = sys.getenv(name) orelse return default;
    const n = std.fmt.parseInt(usize, std.mem.trim(u8, v, " \t"), 10) catch return default;
    return if (n == 0) default else n;
}

/// Either flag: the whole object is worth building.
pub fn wanted() bool {
    return dirOf(expert_env) != null or dirOf(hidden_env) != null;
}

/// One engine's captures, made once at init (`Engine.probe`) and installed on each pass (`Pass.probe`).
pub const Probes = struct {
    expert: ?Expert = null,
    hidden: ?Hidden = null,
    /// A write or a read-back failed: every later site is a no-op, so a full disk cannot fail a reply.
    failed: bool = false,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, cap_rows: usize, layers: usize, dims: usize, slot_n: usize, experts: usize) !Probes {
        var p: Probes = .{};
        errdefer p.deinit();
        if (dirOf(expert_env)) |dir| {
            const logit_rows = if (dirOf(logits_env) != null) countOf(logits_env, default_logit_rows) else 0;
            p.expert = try Expert.init(gpa, io, d, dir, layers, dims, slot_n, experts, cap_rows, logit_rows);
        }
        if (dirOf(hidden_env)) |dir| {
            p.hidden = try Hidden.init(gpa, io, d, stream, dir, layers, dims, countOf(hidden_rows_env, default_hidden_rows));
        }
        return p;
    }

    pub fn deinit(p: *Probes) void {
        if (p.expert) |*x| x.deinit();
        if (p.hidden) |*h| h.deinit();
        p.* = undefined;
    }

    /// Inside a layer, where the pass already holds this layer's MoE input, picks, weights and logits.
    /// Never fails the pass: a capture error prints once and switches the whole probe off.
    pub fn moeLayer(p: *Probes, stream: cuda.Stream, mixed: u64, picks: u64, wts: u64, router_l: u64, layer: usize, rows: usize) void {
        if (p.failed) return;
        p.capture(stream, mixed, picks, wts, router_l, layer, rows) catch |err| {
            std.debug.print("probe: the layer {d} capture failed ({s}); probing is off for this engine\n", .{ layer, @errorName(err) });
            p.failed = true;
        };
    }

    fn capture(p: *Probes, stream: cuda.Stream, mixed: u64, picks: u64, wts: u64, router_l: u64, layer: usize, rows: usize) !void {
        if (p.expert) |*x| try x.records(stream, picks, wts, router_l, layer, rows);
        if (p.hidden) |*h| try h.records(stream, mixed, layer, rows);
    }

    /// The pass is over (its head has run): one read-back and one file a capture.
    pub fn afterPass(p: *Probes, stream: cuda.Stream, rows: usize, pos0: usize) void {
        if (p.failed) return;
        p.readBack(stream, rows, pos0) catch |err| {
            std.debug.print("probe: the pass dump failed ({s}); probing is off for this engine\n", .{@errorName(err)});
            p.failed = true;
        };
    }

    fn readBack(p: *Probes, stream: cuda.Stream, rows: usize, pos0: usize) !void {
        var any = false;
        if (p.expert) |*x| any = any or x.pending(rows);
        if (p.hidden) |*h| any = any or h.pending();
        if (!any) return;
        try stream.synchronize();
        if (p.expert) |*x| try x.flush(rows, pos0);
        if (p.hidden) |*h| try h.flush(rows, pos0);
    }
};

/// Per-layer routing: `picks` [rows, slots] u32 and `wts` [rows, slots] fp32, as `topkMany` wrote them.
pub const Expert = struct {
    dir: []const u8,
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    layers: usize,
    dims: usize,
    slots: usize,
    experts: usize,
    cap: usize,
    /// [layers][cap][slots], a layer's rows contiguous: one async copy a layer.
    ids: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    logits: ?cuda.DeviceBuffer = null,
    logit_rows: usize = 0,
    logit_w: usize = 0,
    /// The rows of the largest pass whose logits are on the device (0: none yet). A server measures
    /// its window costs on passes of 1..32 rows before it serves, so the widest pass is the prompt.
    logit_best: usize = 0,
    /// Rows of the pass that copied logits this time (0: nothing to write).
    logit_pending: usize = 0,
    /// The record buffer for the pass in flight: header, then ids, then wts.
    rec: []u8,
    passes: usize = 0,

    fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, layers: usize, dims: usize, slot_n: usize, experts: usize, cap: usize, logit_rows: usize) !Expert {
        if (layers == 0 or slot_n == 0 or cap == 0) return error.ProbeBadShape;
        const per_layer = cap * slot_n * 4;
        var ids = try cuda.DeviceBuffer.alloc(d, layers * per_layer);
        errdefer ids.free();
        var wts = try cuda.DeviceBuffer.alloc(d, layers * per_layer);
        errdefer wts.free();
        const rec = try gpa.alloc(u8, 20 + 2 * layers * per_layer);
        errdefer gpa.free(rec);
        var self: Expert = .{
            .dir = dir,
            .gpa = gpa,
            .io = io,
            .d = d,
            .layers = layers,
            .dims = dims,
            .slots = slot_n,
            .experts = experts,
            .cap = cap,
            .ids = ids,
            .wts = wts,
            .rec = rec,
        };
        if (logit_rows > 0 and experts > 0) {
            const w = experts + 1;
            self.logits = try cuda.DeviceBuffer.alloc(d, layers * logit_rows * w * 4);
            self.logit_rows = logit_rows;
            self.logit_w = w;
        }
        try std.Io.Dir.cwd().createDirPath(io, dir);
        try self.manifest(); // a directory that exists and is writable, proved at boot rather than at the first token
        std.log.info("probe: routing -> {s} ({d} layers x {d} rows x {d} slots)", .{ dir, layers, cap, slot_n });
        return self;
    }

    fn deinit(x: *Expert) void {
        x.ids.free();
        x.wts.free();
        if (x.logits) |*l| l.free();
        x.gpa.free(x.rec);
        x.* = undefined;
    }

    /// Nothing copied on this pass: the flush has nothing to read back.
    fn pending(x: *const Expert, rows: usize) bool {
        _ = x;
        return rows > 0;
    }

    fn records(x: *Expert, stream: cuda.Stream, picks: u64, wts: u64, router_l: u64, layer: usize, rows: usize) !void {
        if (layer >= x.layers or rows == 0) return;
        if (rows > x.cap) return error.ProbeRowsOverCap;
        const nb = rows * x.slots * 4;
        const off = layer * x.cap * x.slots * 4;
        const api = x.d.api;
        try x.d.check(api.cuMemcpyDtoDAsync_v2(x.ids.ptr + off, picks, nb, stream.handle), "cuMemcpyDtoDAsync");
        try x.d.check(api.cuMemcpyDtoDAsync_v2(x.wts.ptr + off, wts, nb, stream.handle), "cuMemcpyDtoDAsync");
        if (layer == 0) x.logit_pending = 0;
        const lg = x.logits orelse return;
        // The widest pass of the run: a prompt fill, whose logits are the only place a full
        // 512-expert softmax exists (a decode round's 1..32 rows sit far from the decision surface).
        if (rows < 8 or rows < x.logit_best) return;
        if (layer == 0) x.logit_pending = rows;
        const r = @min(rows, x.logit_rows);
        try x.d.check(api.cuMemcpyDtoDAsync_v2(lg.ptr + layer * x.logit_rows * x.logit_w * 4, router_l, r * x.logit_w * 4, stream.handle), "cuMemcpyDtoDAsync");
    }

    fn flush(x: *Expert, rows: usize, pos0: usize) !void {
        if (rows == 0) return;
        const nb = rows * x.slots * 4;
        const per_layer = x.cap * x.slots * 4;
        var at: usize = 20;
        std.mem.writeInt(u32, x.rec[0..4], ex_magic, .little);
        std.mem.writeInt(u32, x.rec[4..8], @intCast(rows), .little);
        std.mem.writeInt(u32, x.rec[8..12], @intCast(pos0), .little);
        std.mem.writeInt(u32, x.rec[12..16], @intCast(x.layers), .little);
        std.mem.writeInt(u32, x.rec[16..20], @intCast(x.slots), .little);
        for (0..x.layers) |l| {
            try x.ids.download(l * per_layer, x.rec[at..][0..nb]);
            at += nb;
        }
        for (0..x.layers) |l| {
            try x.wts.download(l * per_layer, x.rec[at..][0..nb]);
            at += nb;
        }
        var name_buf: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "p{d:0>4}.bin", .{x.passes});
        try x.write(name, x.rec[0..at]);
        if (x.logit_pending > 0) {
            if (x.logits) |lg| {
                const r = @min(x.logit_pending, x.logit_rows);
                const want = 12 + x.layers * r * x.logit_w * 4;
                const dst = try x.gpa.alloc(u8, want);
                defer x.gpa.free(dst);
                std.mem.writeInt(u32, dst[0..4], rt_magic, .little);
                std.mem.writeInt(u32, dst[4..8], @intCast(x.layers), .little);
                std.mem.writeInt(u32, dst[8..12], @intCast(r), .little);
                std.mem.writeInt(u32, dst[12..16], @intCast(x.logit_w), .little);
                var o: usize = 16;
                for (0..x.layers) |l| {
                    try lg.download(l * x.logit_rows * x.logit_w * 4, dst[o..][0 .. r * x.logit_w * 4]);
                    o += r * x.logit_w * 4;
                }
                try x.write("router_logits.bin", dst[0..o]);
            }
            x.logit_best = x.logit_pending;
            x.logit_pending = 0;
        }
        x.passes += 1;
        try x.manifest();
    }

    fn manifest(x: *Expert) !void {
        var text: std.Io.Writer.Allocating = .init(x.gpa);
        defer text.deinit();
        try text.writer.print(
            "{{\"kind\":\"flashnext-expert-routing\",\"layers\":{d},\"dims\":{d},\"slots\":{d},\"experts_n\":{d},\"passes_written\":{d}," ++
                "\"record\":\"p<NNNN>.bin\",\"record_layout\":\"u32 magic TFEX, u32 rows, u32 pos0, u32 layers, u32 slots; ids [layers][rows][slots] u32 LE; wts [layers][rows][slots] f32 LE\"," ++
                "\"router_logits\":\"{s}\",\"router_logits_rows\":{d},\"router_logits_width\":{d}," ++
                "\"dtype\":\"u32 ids, f32 weights, bf16 hidden, f32 router logits\"}}\n",
            .{ x.layers, x.dims, x.slots, x.experts, x.passes, if (x.logit_best > 0) @as([]const u8, "router_logits.bin") else "", x.logit_rows, x.logit_w },
        );
        try x.write("manifest.json", text.written());
    }

    fn write(x: *Expert, name: []const u8, bytes: []const u8) !void {
        var path_buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ x.dir, name });
        try std.Io.Dir.cwd().writeFile(x.io, .{ .sub_path = path, .data = bytes });
    }
};

/// Per-layer post-norm hidden rows (`Scratch.mixed`, the row the MoE reads), every layer, one pass.
pub const Hidden = struct {
    dir: []const u8,
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    layers: usize,
    dims: usize,
    keep: usize,
    buf: cuda.DeviceBuffer,
    /// The rows the file holds: the largest multi-row pass seen, not the first (a server measures its
    /// window costs on passes of 1..32 rows before it serves anything, and those are not a prompt).
    best: usize = 0,
    /// Rows of the pass that captured this time (0: this pass has nothing to write).
    pending_rows: usize = 0,
    k: usize = 0,
    passes: usize = 0,

    fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, dir: []const u8, layers: usize, dims: usize, keep: usize) !Hidden {
        if (layers == 0 or dims == 0 or keep == 0) return error.ProbeBadShape;
        var buf = try cuda.DeviceBuffer.alloc(d, layers * keep * dims * 2);
        errdefer buf.free();
        // the rows a pass shorter than `keep` does not fill read as zeros, not as the last pass's rows
        try buf.fill8(0, stream.handle);
        try std.Io.Dir.cwd().createDirPath(io, dir);
        std.log.info("probe: hidden -> {s} ({d} layers x {d} rows x {d} dims)", .{ dir, layers, keep, dims });
        return .{ .dir = dir, .gpa = gpa, .io = io, .d = d, .layers = layers, .dims = dims, .keep = keep, .buf = buf };
    }

    fn deinit(h: *Hidden) void {
        h.buf.free();
        h.* = undefined;
    }

    fn pending(h: *const Hidden) bool {
        return h.pending_rows > 0;
    }

    fn records(h: *Hidden, stream: cuda.Stream, mixed: u64, layer: usize, rows: usize) !void {
        if (layer == 0) h.pending_rows = 0;
        // A pass only as deep as one already written is not worth a read-back; a deeper one replaces it,
        // so the file ends up holding the fullest pass of the run (the prompt, not the warm-up).
        if (rows < hidden_min_rows or rows < h.best) return;
        if (layer == 0) {
            h.k = @min(rows, h.keep);
            h.pending_rows = rows;
        }
        if (layer >= h.layers) return;
        const nb = h.k * h.dims * 2;
        // the last `k` rows of the pass: the tokens nearest the generation this pass starts
        const src = mixed + (rows - h.k) * h.dims * 2;
        const dst = h.buf.ptr + layer * h.keep * h.dims * 2;
        try h.d.check(h.d.api.cuMemcpyDtoDAsync_v2(dst, src, nb, stream.handle), "cuMemcpyDtoDAsync");
    }

    fn flush(h: *Hidden, rows: usize, pos0: usize) !void {
        if (h.pending_rows == 0) return;
        const bytes = h.layers * h.keep * h.dims * 2;
        const out = try h.gpa.alloc(u8, 28 + bytes);
        defer h.gpa.free(out);
        std.mem.writeInt(u32, out[0..4], hd_magic, .little);
        std.mem.writeInt(u32, out[4..8], @intCast(h.layers), .little);
        std.mem.writeInt(u32, out[8..12], @intCast(h.keep), .little);
        std.mem.writeInt(u32, out[12..16], @intCast(h.dims), .little);
        std.mem.writeInt(u32, out[16..20], @intCast(rows), .little);
        std.mem.writeInt(u32, out[20..24], @intCast(h.k), .little);
        std.mem.writeInt(u32, out[24..28], @intCast(pos0), .little);
        try h.buf.download(0, out[28..]);
        var path_buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/hidden.bin", .{h.dir});
        try std.Io.Dir.cwd().writeFile(h.io, .{ .sub_path = path, .data = out });
        var text: std.Io.Writer.Allocating = .init(h.gpa);
        defer text.deinit();
        try text.writer.print(
            "{{\"kind\":\"flashnext-hidden\",\"layers\":{d},\"keep\":{d},\"dims\":{d},\"rows_of_pass\":{d},\"k\":{d},\"pos0\":{d},\"file\":\"hidden.bin\"," ++
                "\"layout\":\"u32 magic TFHD, u32 layers, u32 keep, u32 dims, u32 rows_of_pass, u32 k, u32 pos0; [layers][keep][dims] bf16 LE\",\"dtype\":\"bf16\"}}\n",
            .{ h.layers, h.keep, h.dims, rows, h.k, pos0 },
        );
        var mbuf: [1100]u8 = undefined;
        const mpath = try std.fmt.bufPrint(&mbuf, "{s}/hidden_manifest.json", .{h.dir});
        try std.Io.Dir.cwd().writeFile(h.io, .{ .sub_path = mpath, .data = text.written() });
        h.best = h.pending_rows;
        h.pending_rows = 0;
        h.passes += 1;
        std.log.info("probe: hidden written ({d} layers, {d} rows) to {s}", .{ h.layers, h.k, path });
    }
};

test "a pass's record is its header, its ids and its weights, layers-major" {
    const layers = 48;
    const cap = 131;
    const slots = 6;
    const per_layer = cap * slots * 4;
    try std.testing.expectEqual(@as(usize, 20 + 2 * layers * per_layer), 20 + 2 * 48 * 131 * 24);
    // a full 4096-row prompt: 9.4 MB on the device and on disk a pass
    try std.testing.expectEqual(@as(usize, 2 * 48 * 4096 * 24), 9_437_184);
}
