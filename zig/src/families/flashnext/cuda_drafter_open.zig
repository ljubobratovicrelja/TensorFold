//! `--drafter <dir>` over flashnext: the DFlash2 block drafter in front of the batched pass.
//!
//! This is the family's `openDrafter` (`native/cuda_drafted.zig`'s lookup), so it is reached only when
//! the server was given `--drafter` and drafts are on. Without the flag nothing here runs and MTP stays
//! what it was: the MTP head loads with the checkpoint and chains its own drafts. With it, the wrapper
//! in `core/lanes/drafted.zig` puts `cuda_drafter2.zig`'s rounds in front of the same head's window
//! verify, and the head keeps only the job the drafter cannot do — rewinding a verify window.
//!
//! Two facts about the target are not in the drafter's checkpoint and are reached here:
//!
//! * the embedding table, as the engine loaded it (`e.table`: dense rows, whole vocabulary);
//! * the LM head. The engine keeps it repacked into bf16 matrix-core tiles (`cuda/qmmf.zig`), and the
//!   drafter's `tf_fxd2_head_topk` reads row-major rows, so the whole table is detiled once, on the
//!   host, by `cuda_draft2_check.zig`'s `detile` — one implementation, not a second one here. It is
//!   1.27 GiB at this checkpoint, held for the server's life and shared by every stream.
//!   `TF_FLASHNEXT_HEAD_FP8=1` with `TF_FLASHNEXT_HEAD_FP8_DIR` adds a 636 MB block-FP8 copy of the
//!   same table, which the drafter ranks through before rescoring 64 survivors against these exact
//!   rows; it does not replace this one, because the rescore is what makes the arm lossless.
//!
//! The refusals are `cuda_drafter2.refusal`'s list, evaluated against the loaded engine before a byte
//! of the head is detiled: `--tp 2`, TF_FLASHNEXT_KV_PAGES, a stream already open, no MTP head, a
//! quantized head, a quantized embedding table, or bits other than 4.

const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const weights = @import("cuda_prompt_wt.zig");
const check = @import("cuda_draft2_check.zig");
const draft2 = @import("cuda_draft2.zig");
const d2 = @import("cuda_drafter2.zig");

/// A drafter opened over the engine, in the shape `native/cuda_drafted.zig`'s `wrap` expects.
pub const Opened = struct {
    drafter: lanes.drafter.Drafter,
    ctx: *anyopaque,
    close: *const fn (*anyopaque) void,
    /// The streams the server admits: their rings and rounds, allocated and warmed once.
    reserve: *const fn (*anyopaque, u32) anyerror!void,
    /// DFlash2 needs no window timing: flashnext's engine times its own windows, and the depth rule
    /// prices the target's forward from the target's costs. The field is here because `wrap` reads it
    /// when a family has one, and null is what "nothing to time" means.
    prepare: ?struct { ctx: *anyopaque, run: *const fn (*anyopaque) anyerror!lanes.drafter.Facts, bytes: usize } = null,
};

/// The family's opener, over the engine type that family declared (its `Cuda`).
pub fn Opener(comptime F: type) type {
    return struct {
        /// `dir`'s DFlash2 drafter over engine `p`, four bits, mirroring `families/qwen3_5/cuda_drafter_open.zig`.
        ///
        /// `precision` is the weights' — `--drafter-precision` — and null leaves it to the
        /// checkpoint's own tensors; either way a checkpoint whose declared precision and its
        /// tensors disagree is refused at load (`Draft.open` -> `DrafterPrecisionMismatch`).
        pub fn openDrafter(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8, precision: ?api.DrafterPrecision) !Opened {
            const self: *F.Cuda = @ptrCast(@alignCast(p));
            const e = self.engine;
            const head_q = e.head_q;
            // refused before the head is detiled (a second of host work and 1.27 GiB): a directory that
            // is not a DFlash2 drafter is not one, whatever the rest of the flags say
            if (!draft2.declared(gpa, io, dir)) return error.Draft2NotADrafter;
            const paged = self.kv.layout != null or (try @import("cuda_kv_lanes.zig").paging()) > 0;
            if (d2.refusal(.{
                .bits = bits,
                .tp = if (self.ranks != null) 2 else 1,
                .paged = paged,
                .streams = self.lanes.count(),
                .head = self.head != null,
                .head_bf16 = head_q.native != null,
                .embed_dense = e.table.dense,
            })) |why| return why;
            // the target's geometry the checkpoint's config is validated against: its hidden width, its
            // vocabulary (the drafter's head IS this one) and its layer count (a tap past it is refused)
            var cfg = check.targetConfig();
            cfg.layers = e.n_layers;
            cfg.hidden = weights.dims;
            cfg.vocab = prompt.vocabulary;
            if (e.table.rows != cfg.vocab) return error.Draft2EmbedNotWholeVocab;
            if (head_q.n != cfg.vocab or head_q.k != cfg.hidden) return error.Draft2HeadNotWholeVocab;
            // the checkpoint's own config, read before the detile so a tap past the last layer or a
            // drafter trained on another target costs a config read rather than a gigabyte
            const declared = try draft2.read(gpa, io, dir, cfg);
            if (d2.tapsRefusal(declared.taps[0..declared.n_taps], e.n_layers)) |why| return why;

            const head = try check.detile(gpa, io, e.driver, head_q.w.ptr, cfg.vocab, cfg.hidden);
            // the detiled table's ownership passes to the drafter: every failure from here frees it
            const want: ?d2.Precision = if (precision) |q| switch (q) {
                .bf16 => .bf16,
                .fp8 => .fp8,
            } else null;
            const dr = try d2.Draft2.init(gpa, io, e.driver, e.stream.*, dir, cfg, .{
                .embed = e.table.w.ptr,
                .head = head.ptr,
                // TF_FLASHNEXT_HEAD_FP8=1 (and only "1") ranks through the quantised table named here;
                // unset is the shipped bf16 head, byte for byte, and the drafter refuses a directory
                // that arrives without the flag rather than choosing for the caller
                .head_fp8 = d2.headFp8DirFromEnv(),
            }, head, want);
            errdefer dr.deinit();
            try self.setTaps(dr.taps());
            dr.host = .{ .ctx = self, .detach = detach };
            self.drafter = dr;
            std.log.info("flashnext: --drafter {s}: DFlash2, taps {any}, {d} drafts a round, {d:.2} ms a round, {d:.1} MiB a stream", .{
                dir, dr.taps(), d2.depth, dr.step_ms * @as(f64, @floatFromInt(d2.depth)), @as(f64, @floatFromInt(dr.ringBytes())) / (1024.0 * 1024.0),
            });
            return .{ .drafter = dr.drafter(), .ctx = dr, .close = close, .reserve = reserve };
        }

        fn cast(q: *anyopaque) *d2.Draft2 {
            return @ptrCast(@alignCast(q));
        }

        /// The engine drops its pointer before the drafter takes its memory down: a stream released
        /// after this point must not find a ring here.
        fn detach(q: *anyopaque) void {
            @as(*F.Cuda, @ptrCast(@alignCast(q))).drafter = null;
        }

        fn close(q: *anyopaque) void {
            cast(q).deinit();
        }

        fn reserve(q: *anyopaque, streams: u32) anyerror!void {
            try cast(q).reserve(streams);
        }
    };
}
