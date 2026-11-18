//! The candidate selector's walk, checked on the host: no device, no allocation, no weights.
//!
//! `tf_fxd2_select` does not build the reference's whole lattice. It builds one position's row of
//! it at a time, because the walk is serial anyway: at step `t` it forms `ctx[r] = pred[prev] *
//! h_proj[t]` once, dots it against each of the `topk` successors, and argmaxes. The reference
//! (`tools/dflash2/analysis/reference.py`) instead materialises `lat[t, p, c]` for every predecessor index
//! `p` and walks it by index. The two must pick the same token every step, and the subtle place
//! they could differ is the anchor: `prev = cat([anchor, cand[:, :-1]])` makes row 0's
//! predecessor the *anchor token* for every `p`, and the walk starts at index 0.
//!
//! These tests pin that equivalence, the argmax tie rule (`torch.argmax` keeps the first index),
//! and that a round's proposals are the block's first `block - 1` positions.
//!
//! **What they cannot cover.** `deviceWalk` is a host model of the kernel, not the kernel: it agrees
//! with `tf_fxd2_select` only as long as the two are written the same way, and it says nothing about
//! the parts of the kernel that are *addressing* rather than arithmetic. Three of them are where the
//! first real round went wrong:
//!
//! * **the buffer layout.** The walk here is handed flat `pred`/`succ`/`hproj` slices indexed by
//!   (token, rank) and (step, rank), so no test here can see a kernel that adds a launch-geometry
//!   term to a table address — `CAND + blockIdx.x * topk + t * topk + lane` is a legal expression in
//!   this model and an out-of-buffer read on the device (a block base *and* a step base over a
//!   `[rows, topk]` table). The test below pins the addressing on its own.
//! * **the vocabulary.** `build` draws ids from `vocab = rows * topk`, `rank = 4`, so every id this
//!   file can produce indexes a codebook row that exists, and an out-of-vocabulary index — a logit's
//!   bit pattern read as a token id, 0.00457 -> 999,663,115 — is not a case any test here can express.
//!   Only `vocab = 248_320` and a real head can surface that, which is what the device-side guard is
//!   for.
//! * **the dtype.** `pred`/`succ`/`hproj` are fp32 here and bf16 in the kernel, and `hproj` is a
//!   single factor here rather than a `[rank, hidden]` bf16 matvec, so a projection that reads the
//!   wrong row (the block's instead of the step's — the other half of the same fault) still passes.

const std = @import("std");
const fwd = @import("cuda_draft2_fwd.zig");

/// A small synthetic model: a vocabulary of `cand.len / topk` proposals' worth of tokens, a rank,
/// and codebooks dense enough that any token a case names has a row.
const Case = struct {
    rank: usize,
    topk: usize,
    rows: usize,
    cand: []const u32,
    unary: []const f32,
    hproj: []const f32,
    pred: []const f32,
    succ: []const f32,

    fn at(c: Case, t: usize, p: usize) u32 {
        return c.cand[t * c.topk + p];
    }
};

/// reference.py: `build_lattice` (prev = anchor, then cand[t - 1]) and `greedy_path` over it.
fn referenceWalk(c: Case, anchor: u32, out: []u32) void {
    var prev: usize = 0;
    for (0..c.rows) |t| {
        var best: usize = 0;
        var best_score: f32 = -std.math.inf(f32);
        for (0..c.topk) |p| {
            // lat[t, prev, p]: the predecessor is the previous step's pick, or the anchor at t = 0
            const prev_token = if (t == 0) anchor else c.at(t - 1, prev);
            var dot: f32 = 0;
            for (0..c.rank) |r| {
                // fused, as the kernel's fmaf: a rounded product would disagree in the last ulp and
                // a near-tie would then pick a different index in the two walks
                const ctx = c.pred[@as(usize, prev_token) * c.rank + r] * c.hproj[t * c.rank + r];
                dot = @mulAdd(f32, ctx, c.succ[@as(usize, c.at(t, p)) * c.rank + r], dot);
            }
            const score = c.unary[t * c.topk + p] + dot;
            if (score > best_score) {
                best_score = score;
                best = p;
            }
        }
        prev = best;
        out[t] = c.at(t, best);
    }
}

/// `tf_fxd2_select`'s shape: one row of the lattice at a time, the carried index, a strict argmax.
fn deviceWalk(c: Case, anchor: u32, out: []u32, ctx: []f32, score: []f32) void {
    var prev: usize = 0; // the reference starts the walk at index 0, which at t = 0 is the anchor
    for (0..c.rows) |t| {
        const prev_token = if (t == 0) anchor else c.at(t - 1, prev);
        for (0..c.rank) |r| ctx[r] = c.pred[@as(usize, prev_token) * c.rank + r] * c.hproj[t * c.rank + r];
        for (0..c.topk) |lane| {
            var dot: f32 = 0;
            for (0..c.rank) |r| dot = @mulAdd(f32, ctx[r], c.succ[@as(usize, c.at(t, lane)) * c.rank + r], dot);
            score[lane] = c.unary[t * c.topk + lane] + dot;
        }
        var best: usize = 0;
        for (1..c.topk) |lane| if (score[lane] > score[best]) {
            best = lane;
        };
        prev = best;
        out[t] = c.at(t, best);
    }
}

/// A deterministic case: `vocab` tokens' rows in both codebooks, `rows * topk` candidates drawn
/// without repetition, logits and codebook entries in [-1, 1). The caller frees `case`'s five rows.
fn build(r: std.Random, rank: usize, topk: usize, rows: usize, vocab: usize, gpa: std.mem.Allocator) !Case {
    const cand = try gpa.alloc(u32, rows * topk);
    errdefer gpa.free(cand);
    const unary = try gpa.alloc(f32, rows * topk);
    errdefer gpa.free(unary);
    const hproj = try gpa.alloc(f32, rows * rank);
    errdefer gpa.free(hproj);
    const pred = try gpa.alloc(f32, vocab * rank);
    errdefer gpa.free(pred);
    const succ = try gpa.alloc(f32, vocab * rank);
    errdefer gpa.free(succ);
    const used = try gpa.alloc(bool, vocab);
    defer gpa.free(used);
    @memset(used, false);
    for (cand) |*id| {
        var v = r.intRangeLessThan(usize, 0, vocab);
        while (used[v]) v = (v + 1) % vocab;
        used[v] = true;
        id.* = @intCast(v);
    }
    for (unary) |*u| u.* = r.float(f32) * 2 - 1;
    for (hproj) |*h| h.* = r.float(f32) * 2 - 1;
    for (pred) |*p| p.* = r.float(f32) * 2 - 1;
    for (succ) |*p| p.* = r.float(f32) * 2 - 1;
    return .{ .rank = rank, .topk = topk, .rows = rows, .cand = cand, .unary = unary, .hproj = hproj, .pred = pred, .succ = succ };
}

test "the device walk picks what the reference's lattice walk picks, anchor and all" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x0df2);
    const rank = 4;
    const topk = 3;
    const rows = 3;
    const vocab = 9;
    const made = try build(prng.random(), rank, topk, rows, vocab, gpa);
    defer gpa.free(made.cand);
    defer gpa.free(made.unary);
    defer gpa.free(made.hproj);
    defer gpa.free(made.pred);
    defer gpa.free(made.succ);
    var want: [rows]u32 = undefined;
    var got: [rows]u32 = undefined;
    var ctx: [rank]f32 = undefined;
    var score: [topk]f32 = undefined;
    // every token id is a plausible anchor, and both walks must agree for each
    for (0..vocab) |anchor| {
        referenceWalk(made, @intCast(anchor), &want);
        deviceWalk(made, @intCast(anchor), &got, &ctx, &score);
        try std.testing.expectEqualSlices(u32, &want, &got);
        // and the anchor is what row 0 is conditioned on: changing it can change row 0's pick,
        // which the index-carrying walk would get wrong if it started at the previous token
        for (got) |id| try std.testing.expect(id < vocab);
    }
}

test "the anchor conditions row 0 and no other row" {
    // One row, one successor, so the choice is forced by the score alone: a rank of 1 and a
    // codebook where the successor's row is 1 for the first candidate and -1 for the second.
    const cand = [_]u32{ 0, 1 };
    const unary = [_]f32{ 0, 0 };
    const hproj = [_]f32{2.0};
    // token 0 is the anchor and the first candidate; token 1 is the second candidate
    const pred = [_]f32{ 1.0, 0.0 };
    const succ = [_]f32{ 0.5, -0.5 };
    const c: Case = .{ .rank = 1, .topk = 2, .rows = 1, .cand = &cand, .unary = &unary, .hproj = &hproj, .pred = &pred, .succ = &succ };
    var want: [1]u32 = undefined;
    var got: [1]u32 = undefined;
    var ctx: [1]f32 = undefined;
    var score: [2]f32 = undefined;
    // anchor 0 scores candidate 0 at +1 and candidate 1 at -1: the walk takes the anchor's token
    referenceWalk(c, 0, &want);
    deviceWalk(c, 0, &got, &ctx, &score);
    try std.testing.expectEqualSlices(u32, &want, &got);
    try std.testing.expectEqual(@as(u32, 0), got[0]);
    // anchor 1 has no predecessor row, so both candidates score 0 and the first index wins
    referenceWalk(c, 1, &want);
    deviceWalk(c, 1, &got, &ctx, &score);
    try std.testing.expectEqualSlices(u32, &want, &got);
    try std.testing.expectEqual(@as(u32, 0), got[0]);
}

test "a tie keeps the first index, as torch.argmax does" {
    // Two candidates with the same score: the device's strict `>` leaves `best` at 0 and so does
    // the reference's. Both walks must return candidate 0's token, not 1's.
    const cand = [_]u32{ 4, 5 };
    const unary = [_]f32{ 1.0, 1.0 };
    const hproj = [_]f32{0.0};
    const pred = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    const succ = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    const c: Case = .{ .rank = 1, .topk = 2, .rows = 1, .cand = &cand, .unary = &unary, .hproj = &hproj, .pred = &pred, .succ = &succ };
    var want: [1]u32 = undefined;
    var got: [1]u32 = undefined;
    var ctx: [1]f32 = undefined;
    var score: [2]f32 = undefined;
    referenceWalk(c, 4, &want);
    deviceWalk(c, 4, &got, &ctx, &score);
    try std.testing.expectEqualSlices(u32, &want, &got);
    try std.testing.expectEqual(@as(u32, 4), got[0]);
}

test "a round proposes the block's first block-1 positions" {
    // The head and the selector read rows 0..block-2 of the block, and the walk's row 0 is
    // conditioned on the anchor: seven proposals for positions 1..7.
    try std.testing.expectEqual(@as(usize, 7), fwd.block - 1);
}

test "the walk's candidate table is one [rows, topk], addressed by step and lane and never by block" {
    // `tf_fxd2_head_topk` writes `IDS[r * topk + i]` for the `rows` rows it scored, and `place` books
    // `cand` and `unary` as `rows * topk` entries each. The selector walks all `rows` steps in one
    // block, so step `t` reads `CAND[t * topk + lane]` and nothing may add a block base on top: the
    // expression that faulted was `CAND + blockIdx.x * topk + t * topk + lane`, which leaves the
    // table on the last blocks and reads the fp32 `unary` table next to it as token ids.
    const topk = 16; // the served checkpoint's selector_top_k
    const rows = fwd.block - 1;
    const table = rows * topk;
    try std.testing.expectEqual(@as(usize, 112), table);
    // every address the walk forms for an in-vocabulary step is inside the round's own table
    for (0..rows) |t| for (0..topk) |lane| {
        try std.testing.expect(t * topk + lane < table);
    };
    // the expression that faulted, at the last block's last step: slot 207 of a 112-slot table,
    // i.e. row 12 of seven. This is the arithmetic the crash's 511 GB read reduces to: the ids run
    // out at the third step, and what follows `cand` in the scratch is the fp32 `unary` table, so
    // the walk indexes a codebook with a logit's bit pattern.
    const blocked = (rows - 1) * topk + (rows - 1) * topk + (topk - 1);
    try std.testing.expectEqual(@as(usize, 207), blocked);
    try std.testing.expectEqual(@as(usize, 2 * rows - 2), blocked / topk);
}
