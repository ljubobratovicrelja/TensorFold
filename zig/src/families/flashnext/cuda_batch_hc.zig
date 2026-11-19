//! A layer's writeback with its read-out's normalize in one launch, and the native read-out (TF_FLASHNEXT_HC_FUSE).
const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");
const parts = @import("cuda_batch_state.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const low = wt.low;
const slots = wt.slots;
const eps = wt.eps;
const pss_n = (dims / 256) * streams;

/// On by default (the same bits either way); TF_FLASHNEXT_HC_FUSE=0: the writeback and the normalize as two launches.
pub fn on() bool {
    // the startup check refused every malformed value already, so a plain default stands here
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_HC_FUSE", true) catch true;
}

/// Whether hc_w's read-out takes its normed rows from the writeback: a table-backed down projection
/// (the bf16 lane's, or TF_FLASHNEXT_HC_FP8's), nothing kept, 64 rows or more, our glue.
pub fn fuses(pass: anytype, hc_w: wt.HcW, n: usize, keep: parts.HcKeep) bool {
    return n >= 64 and (hc_w.down.native != null or hc_w.fp8 != null) and std.meta.eql(keep, parts.HcKeep{}) and dims == 2560 and
        streams == 4 and pass.tri.writebackOwned() and pass.k.forms.hc_fuse;
}

/// The pass's pending writeback with hc_w's normalize in the same launch: the normed rows readoutNative reads.
pub fn writebackNorm(pass: anytype, hc_w: wt.HcW, r0: usize, n: usize) !void {
    const s = pass.s;
    const dummy = s.dummy;
    const h = s.h + r0 * wide * 2;
    const pss = s.pss + r0 * pss_n * 4;
    const inj = s.inj + r0 * streams * 2;
    const norm = s.norm + r0 * wide * 2;
    const rs = pass.rows * dims;
    const scale = hc_w.scale.ptr;
    if (pass.pending == null) return pass.tri.hcWritebackNorm(h, h, pss, dummy, dummy, dummy, dummy, scale, norm, eps, rs, n, dims, streams, 0, 1, 1, 1);
    pass.pending = null;
    if (pass.pending_f32) {
        pass.pending_f32 = false;
        return pass.tri.hcWritebackNorm(h, h, pss, s.moe_f32 + r0 * dims * 4, inj, dummy, dummy, scale, norm, eps, rs, n, dims, streams, 4, 1, 1, 1);
    }
    const y = s.y + r0 * slots * dims * 4;
    const wts = s.wts + r0 * slots * 4;
    try pass.tri.hcWritebackNorm(h, h, pss, dummy, inj, y, wts, scale, norm, eps, rs, n, dims, streams, 2, 10, 11, 1);
}

/// A native read-out over rows [r0, r0 + n) whose normed rows are in: down, act and gates, up, the mix.
pub fn readoutNative(pass: anytype, hc_w: wt.HcW, r0: usize, n: usize) !void {
    const s = pass.s;
    const inj = s.inj + r0 * streams * 2;
    const norm = s.norm + r0 * wide * 2;
    const act = s.act + r0 * low * 2;
    const xs_act = s.xs_act + r0 * (low / 32) * 4;
    // TF_FLASHNEXT_HC_FP8: the same two projections off the FP8 tables, and no K-slice partials --
    // each is pinned to one slice (`cuda_hc_fp8.slices`), so the kernel writes its own output
    if (n * hc_w.n_down * 4 > s.hc_part_bytes) return error.UnexpectedTensor; // one slice within hc_part (hcPartRows)
    const pick = if (hc_w.fp8) |p| .{ p.down, p.up, @as(u64, 0) } else .{
        hc_w.down.native orelse return error.UnexpectedTensor,
        hc_w.up.native orelse return error.UnexpectedTensor,
        s.native_part,
    };
    const down = pick[0];
    const up = pick[1];
    const part = pick[2];
    try pass.k.nativeMatmul(pass.stream, norm, wide, n, down, s.hc_part, part, .f32);
    const inj_ptr = if (hc_w.has_inj == 1) inj else act;
    try pass.tri.hcReduceAct(s.hc_part, act, xs_act, inj_ptr, 1, n, streams, low, hc_w.n_down, hc_w.has_inj);
    try pass.k.nativeMatmul(pass.stream, act, low, n, up, s.hc_up, part, .bf16);
    try pass.tri.hcDenseMix(s.hc_up, norm, s.mixed + r0 * dims * 2, s.xsm + r0 * (dims / 32) * 4, n, dims, streams);
}
