//! flashnext's DFlash2 drafter forward on CUDA: one block round from five tap rows.
//!
//! The checkpoint (`cuda_draft2.zig` parses it, `cuda_draft2_weights.zig` names its 81 tensors) is a
//! five-layer Qwen3 backbone over the target's residuals, a two-tap dynamic grouped convolution
//! around every attention and MLP, and a low-rank candidate selector over the target head's top-k.
//! A round is:
//!
//! 1. **fuse** the five taps of each of the *prefix's* positions — every position the target had
//!    already computed when it proposed the block — into one 2560-wide context row a position
//!    (`tf_fxd2_fuse`: `rms_norm(hidden_norm, taps @ fc^T)`). The block's own rows never come from
//!    taps: `reference.py` hands the prefix's taps in as the cross-attention context and starts
//!    the block from the target's embedding table (`anchor`, then `block - 1` mask tokens).
//!    Fusing the block's own taps instead is the "leaky" variant of `tools/dflash2/tap_accept.py`,
//!    which shows the block the next seven positions' hidden states.
//! 2. **five layers**: input norm, attention conv, q/k/v of the block's rows, the prefix's own k/v,
//!    RoPE, block attention (bidirectional over the block, plus the whole prefix), output conv,
//!    then the MLP with its own conv. Both convs are `tf_q35d_dyn_conv` unchanged; the attention is
//!    `tf_fxd2_attn`, which is the Qwen3.5 drafter's loop with DFlash2's visibility.
//! 3. **head**: the target's `lm_head` over the whole 248,320-token vocabulary, top-16 a row
//!    (`tf_fxd2_head_topk`); no softcap and no output multiplier, as the reference's
//!    `transform_unary_logits` is an identity here.
//! 4. **select**: `hidden_projection`, the predecessor/successor lattice, and the serial greedy walk
//!    of `reference.py`'s `greedy_path` (`tf_fxd2_select`). The head's two tables are the round's,
//!    `[block - 1, topk]` each, and the selector reads row `t` for step `t`: one block walks all of
//!    the steps, because step `t` needs what step `t - 1` picked. Every codebook index is checked
//!    against the vocabulary on the device, and an index past it is `SelectorIndexOutOfVocab` on
//!    this side rather than a read past a 127 MB codebook.
//!
//! What this module deliberately is not: it keeps one stream and one bf16 precision (no 4-bit
//! backbone, no snapshots, no lanes, no trees), and its geometry is exactly the served
//! checkpoint's — five taps, hidden 2560, vocab 248,320, block 8. Any other shape is refused at
//! `open`, before a round can read the wrong rows.
//!
//! The caller owns the two things the drafter does not ship: the target's embedding table and its
//! LM head (`Target`). It also owns collapsing flashnext's four hyper-connection streams into one
//! 2560-wide row a tap, and gathering those rows into the prefix span a round reads: a round takes a
//! contiguous `[ctx_len, n_taps * hidden]` bf16 device buffer and never looks at how it was made
//! (see `tap_source`, `Round.ctx`).

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const draft2 = @import("cuda_draft2.zig");
const contract = @import("cuda_draft2_weights.zig");
const tap_ring = @import("cuda_tap_ring.zig");

/// The drafter's config, as `cuda_draft2.zig` parsed and validated it.
pub const Config = draft2.Draft;

/// The one drafter this module drives: five taps, hidden 2560, the target's 248,320-token
/// vocabulary, and a block of 8 (row 0 the anchor, 7 proposals). Compared against the parsed
/// config at `open` so a checkpoint trained on another target is refused, not misread.
pub const n_taps = 5;
pub const hidden = 2560;
pub const vocab = 248_320;
pub const block = 8;

/// The largest block `cuda_draft2.zig` admits, which is the size of the per-round id staging.
const max_block = draft2.max_block;

/// Which flashnext row stands in for "the completed layer output" a tap reads: one flip, in
/// `cuda_tap_ring.zig`. This module never branches on it — the caller collapses the four
/// hyper-connection streams before it hands the rows over — so `tapsFrom` is the only place the
/// choice is read, and it is read to name the pointer, not to change any arithmetic.
pub const tap_source = tap_ring.TapSource.default;

/// The pointer a stream's tap rows must come from, for the three sources of `TapSource`. Null when
/// a batch keeps none of that source: the caller then has no taps to offer and must not draft.
pub fn tapsFrom(src: tap_ring.Source) ?u64 {
    return src.ptr(tap_source);
}

/// Rows of the prefix the k/v staging is sized for: a chunk of the context is fused, projected,
/// rotated and put into the ring `ctx_chunk` rows at a time, so a state costs one chunk's scratch
/// rather than one window's. Every step of it is per row, so a chunk boundary changes no byte.
pub const ctx_chunk = 64;

/// One round's inputs: a block of `block` rows to draft, and the prefix it cross-attends to.
pub const Round = struct {
    /// `[ctx_len, n_taps * hidden]` bf16 on the device: the fused tap rows of the positions
    /// `position - ctx_len + 1 .. position`, taps in the config's `target_layer_ids` order (a tap's
    /// row is the `hidden`-wide slice at `k * hidden`). That is the prefix — everything the target
    /// had already computed when it proposed the block — and it is what the layers' cross-attention
    /// reads; the block's own rows start from `Target.embed`. Collapsing flashnext's four streams
    /// into one row, and gathering the span out of a ring, is not this module's job.
    ctx: u64,
    /// Rows in `ctx`. Zero is a round with an empty prefix (a block at position 0), and `ctx` is then
    /// unread. At most the drafter's window: the keys land in a `window + 16` row ring.
    ctx_len: usize,
    /// The token at row 0: the target's last verified or sampled one.
    anchor: u32,
    /// The anchor's position. The block's rows sit at `position .. position + block`.
    position: u64,
};

/// A stream's drafter state: five layers' key and value rings, and the block's scratch.
///
/// The ring holds *one round's prefix*: every round writes its own context keys at the positions
/// they were captured at, and reads them back through the same round's attention, so the ring
/// carries nothing between rounds but the span `end` records — which is what makes a gap (a round
/// whose prefix starts past the previous round's block) a refusal rather than a stale read.
///
/// One round at a time per state: the scratch is reused by the next round, and a round ends by
/// synchronising to read its proposals back.
pub const State = struct {
    /// `layers * 2 * ringRows * kvWidth` bf16: layer i's keys at `ringK(i)`, then its values.
    ring: cuda.DeviceBuffer,
    /// The block's rows and the head's candidates (`Bufs`).
    scratch: cuda.DeviceBuffer,
    /// The position after the last row written; 0 until the first round.
    end: u64 = 0,

    const Self = @This();

    pub fn deinit(s: *Self) void {
        s.ring.free();
        s.scratch.free();
    }
};

/// Where a round's rows are in `State.scratch`; every offset is 256-byte aligned.
pub const Bufs = struct {
    /// The block's token ids, u32: the anchor then `block - 1` mask tokens.
    ids: u64,
    /// The context chunk being fused, `[ctx_stage, hidden]` bf16 (`ctx_stage` is `ctx_chunk`).
    ctx: u64,
    /// The running hidden state and the post-attention row.
    h: u64,
    h2: u64,
    /// A sublayer's normed input, and the convolution's input-side output.
    x: u64,
    xin: u64,
    /// The dynamic-kernel projection's rows, `[block, 2 * conv_kernel * groups]` bf16.
    dyn: u64,
    /// The queries before and after their norm and rotation.
    qraw: u64,
    q: u64,
    /// The block's keys before rotation, and after it.
    kraw: u64,
    kb: u64,
    /// The block's values.
    vb: u64,
    /// A context chunk's keys before and after their norm and rotation, and its values: `ctx_stage`
    /// rows each, the prefix the layers cross-attend to before it goes into the ring.
    kcr: u64,
    kc: u64,
    vc: u64,
    /// The attention's output rows, and the sublayer output the output-side conv wraps.
    att: u64,
    o: u64,
    /// The MLP's gate, up and SwiGLU rows.
    gate: u64,
    up: u64,
    act: u64,
    /// The final norm's rows: the ones the head and the selector read (`block - 1` of them).
    hn: u64,
    /// The head's top-k: token ids (u32) and their fp32 logits, `[block - 1, topk]` each — one
    /// table for the round, which the selector reads by (step, lane) and by nothing else.
    cand: u64,
    unary: u64,
    /// The proposals the selector walked, `[block - 1]` u32.
    prop: u64,
    /// The selector's one fault record (`Fault`), cleared before a round and written by the kernel
    /// only if a codebook index is at or past the vocabulary.
    err: u64,
};

/// The proposal `tf_fxd2_select` writes instead of a token when it refuses a codebook index: no
/// vocabulary holds this id, so a caller that only reads the proposals still cannot accept one.
pub const bad_proposal: u32 = 0xFFFF_FFFF;

/// `tf_fxd2_select`'s fault record, as the kernel writes `ERR`: `{ flag, kind, position, index }`.
/// `kind` names the codebook the rejected index was about to index; `position` is the block step
/// that hit it, which is the proposal slot (`position + 1` of the round's absolute positions) the
/// walk was filling.
pub const Fault = extern struct {
    flag: u32 = 0,
    kind: u32 = 0,
    position: u32 = 0,
    index: u32 = 0,

    /// The kernel's `FXD2_KIND_*`, spelled for a log line: which codebook the index was for.
    pub fn codebook(f: Fault) []const u8 {
        return switch (f.kind) {
            1 => "successor_codebook",
            2 => "predecessor_codebook",
            else => "codebook",
        };
    }
};

/// Where a round's internals go when a caller asks for them (`TF_FLASHNEXT_DRAFT_DEBUG`, written by
/// `cuda_draft2_check.zig`): two device spans the **caller owns and sizes**, which a round fills with
/// device-to-device copies on its own stream. Null — the default — and a round branches twice and
/// copies nothing, so a serving round pays nothing for the format existing.
///
/// The copies are the round's own bytes, not a second computation: `ctx` is `tf_fxd2_fuse`'s output
/// as the context attention reads it, and `layers` is the block hidden state the next sublayer reads.
/// That is what lets `tools/dflash2/draft_compare.py` put a finger on one stage instead of on the
/// mean advance.
pub const Debug = struct {
    /// `[ctx_rows, hidden]` bf16: the fused prefix rows of the round, `ctx_rows >= ctx_len`. Written
    /// from layer 0's pass over the prefix; the fusion is re-run per layer and is row-wise, so the
    /// later layers would write the same bytes.
    ctx: u64 = 0,
    /// Rows `ctx` holds: a round whose context outruns it is refused rather than allowed to write
    /// past the caller's buffer.
    ctx_rows: usize = 0,
    /// `[layers + 1, block, hidden]` bf16: the block's input rows at row 0, then its hidden after each
    /// of the `layers` layers.
    layers: u64 = 0,
};

/// The drafter's weights, one arena, bound by `cuda_draft2_weights.zig`'s names and shapes.
pub const Weights = struct {
    arena: cuda.DeviceBuffer,
    at: usize = 0,
    /// The tap fusion: `[hidden, n_taps * hidden]`.
    fc: u64,
    hidden_norm: u64,
    /// The final norm, before the head.
    norm: u64,
    /// The selector's rank projection, `[rank, hidden]`, and its two codebooks, `[vocab, rank]`.
    hidden_proj: u64,
    pred: u64,
    succ: u64,
    layers: []Layer,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        w.arena.free();
        if (w.layers.len > 0) gpa.free(w.layers);
        w.* = undefined;
    }

    fn take(w: *Weights, ck: *core.Checkpoint, name: []const u8, shape: []const usize) !u64 {
        const t = try ck.expect(name, .bf16, shape);
        const at = (w.at + 255) & ~@as(usize, 255);
        if (at + t.bytes.len > w.arena.len) return error.ArenaExhausted;
        w.at = at + t.bytes.len;
        try w.arena.upload(at, t.bytes);
        return w.arena.ptr + at;
    }
};

/// One layer's tensors, in `cuda_draft2_weights.layerTensor`'s order.
pub const Layer = struct {
    input_norm: u64,
    post_norm: u64,
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    q_norm: u64,
    k_norm: u64,
    gate: u64,
    up: u64,
    down: u64,
    /// The two-tap dynamic convolution's base kernel `[2, 2, hidden]` and its projection
    /// `[2 * 2 * groups, hidden]`, input side then output side.
    attn_base: u64,
    attn_conv: u64,
    mlp_base: u64,
    mlp_conv: u64,
};

/// The target's two tensors the drafter reads but does not ship.
pub const Target = struct {
    /// The embedding table, `[vocab, hidden]` bf16, row-major: a round's `block` rows start from it.
    embed: u64,
    /// The LM head, `[vocab, hidden]` bf16, row-major: the drafter's head IS this one, and
    /// `tf_fxd2_head_topk` reads it a row at a time. Note that the engine's resident head is the same
    /// tensor repacked into matrix-core tiles (`cuda/qmmf.zig`), which none of this module's kernels
    /// can address: a serving caller has to hand over rows, not tiles (see `cuda_draft2_check.zig`).
    head: u64,
};

/// The kernels a round launches, by symbol.
///
/// `tf_fxd2_knorm` — the norm without a rotation, for a context whose keys were all rotated at
/// position 0 — is deliberately not among them: a round's prefix keys are rotated at the positions
/// they were captured at, so `tf_q35d_norm_rope` is the entry it resolves. The kernel stays in the
/// image because `kernels/cuda/qwen35_draft.cu`'s bytes are pinned by `dense_serving_test.zig`,
/// which is a reviewed-bytes gate rather than this module's to move.
const Fns = struct {
    fuse: cuda.Function,
    rmsnorm: cuda.Function,
    attn: cuda.Function,
    head_topk: cuda.Function,
    select: cuda.Function,
    silu_mul: cuda.Function,
    dyn_conv: cuda.Function,
    norm_rope: cuda.Function,
    ring_put: cuda.Function,
};

/// The drafter, loaded: its config, its weights, and its kernels.
pub const Draft = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    c: Config,
    w: Weights,
    target: Target,
    /// The drafter's own image (`qwen35_draft`) and the bf16 row kernels it borrows, whose
    /// `cuda`-side wrapper carries the projection and embedding-lookup launches.
    mod: cuda.Module,
    rows: cuda.qwen35_bf16_rows.Kernels,
    fns: Fns,
    /// Where the next round's internals are copied, or null (`Debug`). Set by `trace`, off by default,
    /// and read only by the two copy sites a round has.
    dbg: ?Debug = null,

    const Self = @This();

    /// Send the next rounds' internals to `g`, or (null) stop. The spans are the caller's: this module
    /// never allocates, sizes or frees one, and a span too small for a round is `DebugCtxTooSmall`.
    pub fn trace(s: *Self, g: ?Debug) void {
        s.dbg = g;
    }

    /// Open `dir`'s drafter against `cfg` (the served target's own config) and `target`'s two
    /// tensors. Every refusal here is one a load should never survive: a shape this module's
    /// kernels do not fix, or a checkpoint that is not the served drafter's.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, dir: []const u8, cfg: config.Config, target: Target) !*Self {
        if (target.embed == 0 or target.head == 0) return error.TargetTensorsMissing;
        const c = try draft2.read(gpa, io, dir, cfg);
        if (c.n_taps != n_taps) return error.DraftTapCountUnsupported;
        if (c.hidden != hidden) return error.DraftHiddenUnsupported;
        if (c.vocab != vocab) return error.DraftVocabUnsupported;
        if (c.block != block) return error.DraftBlockUnsupported;
        // The kernels' fixed widths: a per-head rotation at 128, a top-k a warp holds, a rank the
        // selector's shared row is sized for, a ring the window cannot outrun.
        if (c.head_dim != 128 or c.topk > draft2.lp_top or c.rank > 256) return error.UnsupportedDraftConfig;
        if (c.hidden % 256 != 0 or c.hidden % 8 != 0 or c.intermediate % 8 != 0 or c.window > c.ringRows()) return error.UnsupportedDraftConfig;

        const s = try gpa.create(Self);
        errdefer gpa.destroy(s);
        s.* = .{ .gpa = gpa, .io = io, .d = d, .stream = stream, .c = c, .w = undefined, .target = target, .mod = undefined, .rows = undefined, .fns = undefined };
        s.mod = try cuda.Module.load(d, cuda.kernels.qwen35_draft);
        errdefer s.mod.unload();
        s.rows = try cuda.qwen35_bf16_rows.Kernels.load(d);
        errdefer s.rows.unload();
        // flashnext's own entries, then the two this file's Qwen3.5 drafter already had (the
        // convolution and the rows' norm-plus-rotation); the plain projections and the embedding
        // lookup are `rows`, whose module carries its own launches.
        s.fns = .{
            .fuse = try s.mod.function("tf_fxd2_fuse"),
            .rmsnorm = try s.mod.function("tf_fxd2_rmsnorm"),
            .attn = try s.mod.function("tf_fxd2_attn"),
            .head_topk = try s.mod.function("tf_fxd2_head_topk"),
            .select = try s.mod.function("tf_fxd2_select"),
            .silu_mul = try s.mod.function("tf_fxd2_silu_mul"),
            .dyn_conv = try s.mod.function("tf_q35d_dyn_conv"),
            .norm_rope = try s.mod.function("tf_q35d_norm_rope"),
            .ring_put = try s.mod.function("tf_q35d_ring_put"),
        };
        s.w = try loadWeights(gpa, io, s, dir);
        errdefer s.w.deinit(gpa);
        return s;
    }

    pub fn deinit(s: *Self) void {
        s.w.deinit(s.gpa);
        s.rows.unload();
        s.mod.unload();
        s.gpa.destroy(s);
    }

    /// A new stream's state: its ring and scratch, both zeroed so a stray read sees a key of 0.
    pub fn create(s: *Self) !State {
        var st: State = .{ .ring = try cuda.DeviceBuffer.alloc(s.d, s.c.ringBytes()), .scratch = undefined };
        errdefer st.ring.free();
        st.scratch = try cuda.DeviceBuffer.alloc(s.d, try scratchBytes(s.c));
        errdefer st.scratch.free();
        try st.ring.fill8(0, s.stream.handle);
        try st.scratch.fill8(0, s.stream.handle);
        try s.stream.synchronize();
        return st;
    }

    /// The bytes a stream's state takes, before it is created: its ring and one block's scratch.
    pub fn stateBytes(c: Config) !usize {
        return std.math.add(usize, c.ringBytes(), try scratchBytes(c));
    }

    /// One layer's keys and values: the keys' `ringRows * kvWidth` rows, then the values'.
    fn layerStride(c: Config) usize {
        return 2 * c.ringRows() * c.kvWidth() * 2;
    }

    /// Layer `i`'s keys in the ring, and its values (the keys' row block, then the values').
    fn ringK(s: *Self, st: *State, i: usize) u64 {
        return st.ring.ptr + i * layerStride(s.c);
    }

    fn ringV(s: *Self, st: *State, i: usize) u64 {
        return s.ringK(st, i) + s.c.ringRows() * s.c.kvWidth() * 2;
    }

    /// One draft round: the prefix's tap rows, the anchor, and the position it sits at.
    ///
    /// `out` takes `block - 1` proposals: the selector's greedy path over the block's first
    /// `block - 1` rows, i.e. the tokens for positions `position + 1 .. position + block - 1`
    /// (row `t`'s hidden proposes the token at `position + t + 1`, and row 0's predecessor is the
    /// anchor — `reference.py`'s `prev = cat([anchor, cand[:, :-1]])`).
    ///
    /// Ends with the stream synchronised: the proposals are read back to the host, and the next
    /// round's host-side writes to the id staging or the context span are then ordered after this
    /// round's kernels.
    ///
    /// Refuses the round with `error.SelectorIndexOutOfVocab` — and logs the index, the step and the
    /// codebook — if `tf_fxd2_select` found a codebook index at or past the vocabulary. That is a
    /// layout fault, not a token the drafter got wrong, and the alternative to refusing is reading
    /// somewhere past a 127 MB codebook (it read 511 GB past one before the guard existed).
    pub fn draft(s: *Self, st: *State, r: Round, out: []u32) !void {
        const c = s.c;
        const B = c.block;
        const rows = B - 1;
        if (out.len != rows) return error.BadProposalCount;
        if (r.ctx_len > c.window) return error.ContextPastWindow;
        if (r.ctx_len > 0 and r.ctx == 0) return error.ContextRowsMissing;
        // the prefix ends at the anchor, so a block at position 0 has none
        if (@as(u64, r.ctx_len) > r.position + 1) return error.ContextPastPosition;
        if (r.anchor >= c.vocab) return error.AnchorPastVocab;
        if (r.position > @as(u64, std.math.maxInt(i64)) -| B) return error.PositionOutOfRange;
        // Rounds may overlap (a verify accepts fewer rows than it was offered, and the accepted
        // prefix's rows are the target's own), but the ring the keys land in must stay a
        // contiguous span: a gap would leave older rows under the positions the block reads.
        if (st.end != 0 and r.position > st.end) return error.DraftContextGap;

        var b: Bufs = undefined;
        _ = try place(c, st.scratch.ptr, &b);
        var ids: [max_block]u32 = @splat(0);
        ids[0] = r.anchor;
        for (ids[1..B]) |*id| id.* = c.mask;
        try st.scratch.upload(@intCast(b.ids - st.scratch.ptr), std.mem.sliceAsBytes(ids[0..B]));
        // The selector's fault record is cleared here, before any kernel of the round is queued: a
        // record left by an earlier round would refuse a round that never faulted, and a copy
        // issued after the round's kernels would not be ordered against them on this stream.
        const clear: Fault = .{};
        try st.scratch.upload(@intCast(b.err - st.scratch.ptr), std.mem.asBytes(&clear));

        try embed(s, b.ids, b.h);
        // the block's input rows (the anchor's embedding and the masks), then one row block a layer:
        // a reader of the dump then sees the state each sublayer was handed, not just the last one
        const row_bytes = B * c.hidden * 2;
        if (s.dbg) |g| if (g.layers != 0) try copyOut(s, g.layers, row_bytes, b.h);
        for (s.w.layers, 0..) |l, i| {
            try s.layer(st, &b, l, i, r);
            if (s.dbg) |g| if (g.layers != 0) try copyOut(s, g.layers + (i + 1) * row_bytes, row_bytes, b.h);
        }
        // The head and the selector read the block's rows 1..7, not 0..6: row r sits at position
        // p+r and proposes the token at p+r, so row 0 only re-predicts the anchor. `reference.py`'s
        // `propose` heads `hidden[:, -block+1:]` for the same reason; starting at row 0 lags every
        // proposal by one slot, which reads as chance agreement rather than as an error.
        try rmsnorm(s, b.h + c.hidden * 2, s.w.norm, b.hn, rows);
        try headTopk(s, b.hn, rows, b.cand, b.unary);
        try select(s, b.hn, rows, b.cand, b.unary, b.err, r.anchor, b.prop);
        // the proposals are read back on the host, so the round's kernels must have landed first:
        // a blocking DtoH copy does not order itself after work queued on a non-blocking stream
        var held: [max_block - 1]u32 = @splat(0);
        var fault: Fault = .{};
        try s.stream.synchronize();
        try st.scratch.download(@intCast(b.prop - st.scratch.ptr), std.mem.sliceAsBytes(held[0..rows]));
        try st.scratch.download(@intCast(b.err - st.scratch.ptr), std.mem.asBytes(&fault));
        // A codebook index the selector refused outranks the proposals: a round is all of its
        // positions or none, and an index at or past the vocabulary is a layout fault, not a token
        // the drafter got wrong. Both codebooks are `[vocab, rank]`, so nothing legitimate produces
        // one — see `tf_fxd2_select`'s guard.
        if (fault.flag != 0) {
            std.log.err("draft2 select: {s}[{d}] at block step {d} (position {d}) is past the {d}-token vocabulary; refusing the round", .{
                fault.codebook(), fault.index, fault.position, r.position + 1 + @as(u64, fault.position), c.vocab,
            });
            return error.SelectorIndexOutOfVocab;
        }
        if (std.mem.indexOfScalar(u32, held[0..rows], bad_proposal) != null) return error.SelectorIndexOutOfVocab;
        @memcpy(out, held[0..rows]);
        st.end = r.position + B;
    }

    /// One layer of the block: input norm and conv, q/k/v of the block's rows plus the prefix's own,
    /// the block attention, the output conv, then the MLP with its own conv (`reference.py`'s
    /// `DraftLayer.forward`, step for step).
    fn layer(s: *Self, st: *State, b: *const Bufs, l: Layer, i: usize, r: Round) !void {
        const c = s.c;
        const B = c.block;
        const H = c.hidden;
        const qw = c.qWidth();
        const kvw = c.kvWidth();
        const dyn = c.dynamic();
        const pos: c_longlong = @intCast(r.position);
        const lo: c_longlong = @intCast(r.position + 1 - r.ctx_len);
        // attention: norm, prepare, project, rotate, attend, finish
        try rmsnorm(s, b.h, l.input_norm, b.x, B);
        try dense(s, b.x, l.attn_conv, b.dyn, dyn, H, B);
        try dynConv(s, b.x, b.dyn, l.attn_base, 0, b.xin, 0);
        try dense(s, b.xin, l.q, b.qraw, qw, H, B);
        try dense(s, b.xin, l.k, b.kraw, kvw, H, B);
        try dense(s, b.xin, l.v, b.vb, kvw, H, B);
        try normRope(s, b.qraw, l.q_norm, b.q, qw, c.heads, pos, B);
        try normRope(s, b.kraw, l.k_norm, b.kb, kvw, c.kv_heads, pos, B);
        // the prefix's keys and values, into this layer's ring at the positions they were captured at
        try s.prefix(st, b, l, i, r);
        try attn(s, st, i, b, lo, r.ctx_len, B);
        try dense(s, b.att, l.o, b.o, H, qw, B);
        try dynConv(s, b.o, b.dyn, l.attn_base, b.h, b.h2, 1);
        // mlp: norm, prepare, gate/up/silu, down, finish
        try rmsnorm(s, b.h2, l.post_norm, b.x, B);
        try dense(s, b.x, l.mlp_conv, b.dyn, dyn, H, B);
        try dynConv(s, b.x, b.dyn, l.mlp_base, 0, b.xin, 0);
        try dense(s, b.xin, l.gate, b.gate, c.intermediate, H, B);
        try dense(s, b.xin, l.up, b.up, c.intermediate, H, B);
        try siluMul(s, b.gate, b.up, b.act, B * c.intermediate);
        try dense(s, b.act, l.down, b.o, H, c.intermediate, B);
        try dynConv(s, b.o, b.dyn, l.mlp_base, b.h2, b.h, 1);
    }

    /// The layer's cross-attention context: the prefix's fused tap rows, `ctx_chunk` rows at a time,
    /// from the tap fusion through the k/v projections and the per-head norm and rotation at each
    /// row's **own** position, into this layer's ring at those positions.
    ///
    /// Chunking is exact rather than approximate: fusion norms one row's 2560 outputs against that
    /// row alone, the projections are row-wise, and the norm-plus-rotation is per head per row, so a
    /// chunk boundary changes no byte of the result — it only bounds the scratch a state needs to
    /// `ctx_chunk` rows instead of one window's. `reference.py` rotates the context keys at the
    /// positions they were captured at (`ctx_positions`), not at zero: giving every prefix key the
    /// same rotation destroys the q/k geometry the taps were trained against.
    fn prefix(s: *Self, st: *State, b: *const Bufs, l: Layer, i: usize, r: Round) !void {
        const c = s.c;
        const H = c.hidden;
        const k = c.n_taps * H;
        const kvw = c.kvWidth();
        const lo: c_longlong = @intCast(r.position + 1 - r.ctx_len);
        var done: usize = 0;
        while (done < r.ctx_len) {
            const n = @min(ctx_chunk, r.ctx_len - done);
            const at: c_longlong = lo + @as(c_longlong, @intCast(done));
            try fuse(s, r.ctx + done * k * 2, b.ctx, n);
            // the whole prefix's fusion, taken from layer 0's pass: `fuse` is row-wise, so the four
            // later layers would write these same bytes and one copy of them is the fusion's output
            if (i == 0) {
                if (s.dbg) |g| {
                    if (g.ctx != 0) {
                        if (g.ctx_rows < done + n) return error.DebugCtxTooSmall;
                        try copyOut(s, g.ctx + done * H * 2, n * H * 2, b.ctx);
                    }
                }
            }
            try dense(s, b.ctx, l.k, b.kcr, kvw, H, n);
            try dense(s, b.ctx, l.v, b.vc, kvw, H, n);
            try normRope(s, b.kcr, l.k_norm, b.kc, kvw, c.kv_heads, at, n);
            try ringPut(s, b.kc, b.vc, s.ringK(st, i), s.ringV(st, i), kvw, at, n);
            done += n;
        }
    }
};

// -----------------------------------------------------------------------------------------------
// The round's launches. Every argument list is the .cu source's, in order; a pointer is a device
// address, a count an `int` and a scale an `f32`.
// -----------------------------------------------------------------------------------------------

fn launch(s: *Draft, f: cuda.Function, grid: [3]u32, threads: u32, shared: u32, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |value| a.add(value);
    try cuda.launch.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = threads }, .shared = shared }, s.stream, &a);
}

fn ci(x: usize) c_int {
    return @intCast(x);
}

/// One device-to-device copy into a caller's debug span, on the round's own stream: queued, not
/// waited on, so the check's single synchronisation at the end of the round covers it too.
fn copyOut(s: *Draft, dst: u64, len: usize, src: u64) !void {
    if (len == 0) return;
    // a borrowed view of the caller's span: `copyFrom` writes rows, it does not own them
    const view: cuda.memory.DeviceBuffer = .{ .d = s.d, .ptr = dst, .len = len, .borrowed = true };
    try view.copyFrom(0, src, len, s.stream.handle);
}

/// tf_fxd2_fuse(X, W, NORM, OUT, K, H, eps): grid (rows), 256 threads, `(K + 2H) * 2` shared.
///
/// `taps` is the first of `rows` contiguous `n_taps * hidden` rows: a prefix chunk, or the whole
/// prefix at once when a caller has the room. `rows` is a chunk of the context, so it may not
/// outrun the scratch's `ctx_chunk`.
fn fuse(s: *Draft, taps: u64, out: u64, rows: usize) !void {
    if (rows == 0) return;
    if (rows > ctx_chunk) return error.ChunkPastScratch;
    const c = s.c;
    const k = c.n_taps * c.hidden;
    try launch(s, s.fns.fuse, .{ @intCast(rows), 1, 1 }, 256, @intCast((k + 2 * c.hidden) * 2), .{ taps, s.w.fc, s.w.hidden_norm, out, ci(k), ci(c.hidden), c.eps });
}

/// tf_fxd2_rmsnorm(X, W, OUT, H, eps): grid (rows), 256 threads.
fn rmsnorm(s: *Draft, x: u64, w: u64, out: u64, rows: usize) !void {
    try launch(s, s.fns.rmsnorm, .{ @intCast(rows), 1, 1 }, 256, 0, .{ x, w, out, ci(s.c.hidden), s.c.eps });
}

/// tf_q35d_norm_rope(X, W, OUT, heads, stride, pos0, eps, theta): grid (rows * heads), one warp a head.
fn normRope(s: *Draft, x: u64, w: u64, out: u64, stride: usize, heads: usize, pos0: c_longlong, rows: usize) !void {
    try launch(s, s.fns.norm_rope, .{ @intCast(rows * heads), 1, 1 }, 32, 0, .{ x, w, out, ci(heads), ci(stride), pos0, s.c.eps, s.c.theta });
}

/// tf_q35d_ring_put(K, V, RK, RV, width, pos0, W): grid (rows), 256 threads.
fn ringPut(s: *Draft, k: u64, v: u64, rk: u64, rv: u64, width: usize, pos0: c_longlong, rows: usize) !void {
    try launch(s, s.fns.ring_put, .{ @intCast(rows), 1, 1 }, 256, 0, .{ k, v, rk, rv, ci(width), pos0, ci(s.c.ringRows()) });
}

/// tf_q35d_dyn_conv(X, DYN, BASE, RES, OUT, D, group, branch): grid (rows), 256 threads.
fn dynConv(s: *Draft, x: u64, dyn_w: u64, base: u64, res: u64, out: u64, branch: usize) !void {
    try launch(s, s.fns.dyn_conv, .{ @intCast(s.c.block), 1, 1 }, 256, 0, .{ x, dyn_w, base, res, out, ci(s.c.hidden), ci(s.c.group), ci(branch) });
}

/// tf_fxd2_attn(Q, KB, VB, RK, RV, rows, heads, kv_heads, W, begin, pos0, window, ctx, OUT):
/// grid (rows, heads), one warp a (row, head).
///
/// The ring half of the kernel scans `[max(begin, pos0 + m - window + 1), pos0 + ctx)` for block row
/// `m`, so a round hands it `pos0 = begin = ` the context's first position and `ctx = ctx_len`: the
/// kernel then reads the prefix's ring rows and nothing else. (Its per-row window still lets a context
/// of a full window through: `pos0 + m - window + 1 <= pos0` for every block row `m < window`.)
fn attn(s: *Draft, st: *State, i: usize, b: *const Bufs, pos0: c_longlong, ctx_len: usize, rows: usize) !void {
    const c = s.c;
    try launch(s, s.fns.attn, .{ @intCast(rows), @intCast(c.heads), 1 }, 32, 0, .{
        b.q,      b.kb,         b.vb,           s.ringK(st, i),   s.ringV(st, i),
        ci(rows), ci(c.heads),  ci(c.kv_heads), ci(c.ringRows()), pos0,
        pos0,     ci(c.window), ci(ctx_len),    b.att,
    });
}

/// `Y[r, j] = X[r, :] @ W[j, :]`, bf16 in and out with fp32 sums in a fixed order
/// (`qwen35_bf16_rows.cu`'s `tf_q35_dense_bf16`: grid (n, rows), 256 threads).
fn dense(s: *Draft, x: u64, w: u64, out: u64, n: usize, k: usize, rows: usize) !void {
    return s.rows.denseRows(s.stream, x, w, out, rows, n, k);
}

/// tf_fxd2_silu_mul(GATE, UP, OUT, n): one thread an element.
fn siluMul(s: *Draft, gate: u64, up: u64, out: u64, n: usize) !void {
    try launch(s, s.fns.silu_mul, .{ @intCast((n + 255) / 256), 1, 1 }, 256, 0, .{ gate, up, out, ci(n) });
}

/// `out[r] = table[ids[r]]`, a bit copy a row (`tf_q35_embed_bf16`: grid (rows), 256 threads).
fn embed(s: *Draft, ids: u64, out: u64) !void {
    return s.rows.embedRows(s.stream, ids, s.target.embed, out, s.c.block, s.c.hidden, 0);
}

/// tf_fxd2_head_topk(X, HEAD, IDS, VAL, H, V, topk): grid (rows), 256 threads,
/// `H * 2 + 256 * topk * 8` shared.
fn headTopk(s: *Draft, x: u64, rows: usize, ids: u64, val: u64) !void {
    const c = s.c;
    // `cand` and `unary` are booked for the block's first `block - 1` rows and no more: a row past
    // that writes past the scratch, and a caller that scored eight rows would be as wrong as one
    // that scored none.
    if (rows == 0 or rows > c.block - 1) return error.BadProposalCount;
    const shared = c.hidden * 2 + 256 * c.topk * 8;
    if (shared > 48 * 1024) return error.HeadTopkSharedTooLarge;
    try launch(s, s.fns.head_topk, .{ @intCast(rows), 1, 1 }, 256, @intCast(shared), .{ x, s.target.head, ids, val, ci(c.hidden), ci(c.vocab), ci(c.topk) });
}

/// tf_fxd2_select(X, WPROJ, PRED, SUCC, CAND, UNARY, OUT, ERR, H, rank, topk, rows, vocab, anchor):
/// grid (1), 256 threads.
///
/// One block for the whole round, not one a row: the walk is serial in the step, so a block a row
/// would have to be told the previous blocks' choices, and the only thing a second block adds is a
/// second copy of the work. `rows` is the number of steps (the head's rows), which the kernel takes
/// as an argument because the launch geometry no longer says it.
fn select(s: *Draft, x: u64, rows: usize, ids: u64, val: u64, err: u64, anchor: u32, out: u64) !void {
    const c = s.c;
    // exactly the rows `cand`, `unary`, `prop` and `hn` are booked for (`place`'s `block - 1`): the
    // walk reads every one of them, and one more row is a read past the head's table
    if (rows != c.block - 1) return error.BadProposalCount;
    try launch(s, s.fns.select, .{ 1, 1, 1 }, 256, 0, .{
        x,      s.w.hidden_proj, s.w.pred, s.w.succ, ids, val, out, err,
        ci(c.hidden), ci(c.rank), ci(c.topk), ci(rows), ci(c.vocab), @as(c_int, @intCast(anchor)),
    });
}

// -----------------------------------------------------------------------------------------------
// Weights and the block's layout
// -----------------------------------------------------------------------------------------------

fn loadWeights(gpa: std.mem.Allocator, io: std.Io, s: *Draft, dir: []const u8) !Weights {
    const c = s.c;
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    var w: Weights = .{ .arena = try cuda.DeviceBuffer.alloc(s.d, contract.bytes(c)), .fc = 0, .hidden_norm = 0, .norm = 0, .hidden_proj = 0, .pred = 0, .succ = 0, .layers = &.{} };
    errdefer w.arena.free();
    w.layers = try gpa.alloc(Layer, c.layers);
    errdefer gpa.free(w.layers);

    // the six tensors outside the layers, in the contract's own order
    var want: [contract.max_tensors]contract.Tensor = undefined;
    var shape: [contract.max_tensors][4]usize = undefined;
    const top = contract.expected(c, &want, &shape);
    if (top != contract.top_count) return error.UnexpectedTensor;
    var at: [contract.top_count]u64 = undefined;
    for (0..top) |i| at[i] = try w.take(&ck, want[i].name, want[i].shape);
    w.fc, w.hidden_norm, w.norm = .{ at[0], at[1], at[2] };
    w.hidden_proj, w.pred, w.succ = .{ at[3], at[4], at[5] };

    var name_buf: [64]u8 = undefined;
    var shape_buf: [4]usize = undefined;
    for (w.layers, 0..) |*l, i| {
        var t: [contract.per_layer]u64 = undefined;
        for (&t, 0..) |*ptr, k| {
            const e = try contract.layerTensor(c, i, k, &name_buf, &shape_buf);
            ptr.* = try w.take(&ck, e.name, e.shape);
        }
        l.* = .{
            .input_norm = t[0],
            .post_norm = t[1],
            .q = t[2],
            .k = t[3],
            .v = t[4],
            .o = t[5],
            .q_norm = t[6],
            .k_norm = t[7],
            .gate = t[8],
            .up = t[9],
            .down = t[10],
            .attn_base = t[11],
            .attn_conv = t[12],
            .mlp_base = t[13],
            .mlp_conv = t[14],
        };
    }
    // The contract is a closed set: a tensor the drafter does not name is a checkpoint this loader
    // does not understand, which is worse to half-read than to refuse.
    if (ck.unused() != 0) return error.UnexpectedTensor;
    return w;
}

fn bytes(factors: anytype) !usize {
    var value: usize = 1;
    inline for (factors) |factor| value = try std.math.mul(usize, value, factor);
    return value;
}

/// The block's scratch, byte for byte: every row the round writes, 256-byte aligned.
fn scratchBytes(c: Config) !usize {
    return place(c, 0, null);
}

/// The scratch's offsets; with `out`, the rows' addresses from `base`.
pub fn place(c: Config, base: u64, out: ?*Bufs) !usize {
    const B = c.block;
    const R = B - 1;
    const dyn = c.dynamic();
    // the context staging: one chunk of the prefix, which is never narrower than the block
    const S = @max(B, ctx_chunk);
    const sizes = .{
        .{ "ids", try bytes(.{ B, 4 }) },
        .{ "ctx", try bytes(.{ S, c.hidden, 2 }) },
        .{ "h", try bytes(.{ B, c.hidden, 2 }) },
        .{ "h2", try bytes(.{ B, c.hidden, 2 }) },
        .{ "x", try bytes(.{ B, c.hidden, 2 }) },
        .{ "xin", try bytes(.{ B, c.hidden, 2 }) },
        .{ "dyn", try bytes(.{ B, dyn, 2 }) },
        .{ "qraw", try bytes(.{ B, c.qWidth(), 2 }) },
        .{ "q", try bytes(.{ B, c.qWidth(), 2 }) },
        .{ "kraw", try bytes(.{ B, c.kvWidth(), 2 }) },
        .{ "kb", try bytes(.{ B, c.kvWidth(), 2 }) },
        .{ "vb", try bytes(.{ B, c.kvWidth(), 2 }) },
        .{ "kcr", try bytes(.{ S, c.kvWidth(), 2 }) },
        .{ "kc", try bytes(.{ S, c.kvWidth(), 2 }) },
        .{ "vc", try bytes(.{ S, c.kvWidth(), 2 }) },
        .{ "att", try bytes(.{ B, c.qWidth(), 2 }) },
        .{ "o", try bytes(.{ B, c.hidden, 2 }) },
        .{ "gate", try bytes(.{ B, c.intermediate, 2 }) },
        .{ "up", try bytes(.{ B, c.intermediate, 2 }) },
        .{ "act", try bytes(.{ B, c.intermediate, 2 }) },
        .{ "hn", try bytes(.{ R, c.hidden, 2 }) },
        // one `[rows, topk]` table a tensor, the shape `tf_fxd2_select` walks step by step
        .{ "cand", try bytes(.{ R, c.topk, 4 }) },
        .{ "unary", try bytes(.{ R, c.topk, 4 }) },
        .{ "prop", try bytes(.{ R, 4 }) },
        // `Fault`: flag, kind, position, index
        .{ "err", try bytes(.{ 4, 4 }) },
    };
    var total: usize = 0;
    inline for (sizes) |s| {
        const at = (total + 255) & ~@as(usize, 255);
        total = try std.math.add(usize, at, s[1]);
        if (out) |b| @field(b, s[0]) = base + at;
    }
    return total;
}

// -----------------------------------------------------------------------------------------------
// Tests: the shapes this module fixes, and the scratch a round takes. Nothing here touches a device.
// -----------------------------------------------------------------------------------------------

/// A flashnext-shaped target, as `cuda_draft2.zig`'s own tests spell one.
fn testTarget() config.Config {
    var c: config.Config = undefined;
    c.hidden = hidden;
    c.vocab = vocab;
    c.layers = 48;
    c.heads = 24;
    c.kv_heads = 2;
    c.value_heads = 2;
    c.eps = 1e-6;
    return c;
}

const served_json =
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

test "the served drafter is exactly the shape this module's kernels fix" {
    const c = try draft2.parse(std.testing.allocator, served_json, testTarget());
    // The constants above are not a guess: the checkpoint parses to them, and `open` refuses any
    // other value before a round reads a row.
    try std.testing.expectEqual(n_taps, c.n_taps);
    try std.testing.expectEqual(hidden, c.hidden);
    try std.testing.expectEqual(vocab, c.vocab);
    try std.testing.expectEqual(block, c.block);
    // a round offers the block's first seven rows: seven proposals for positions 1..7
    try std.testing.expectEqual(@as(usize, 7), c.block - 1);
    // and the kernel-side widths `open` also checks: a 128-wide head, a warp's top-k, the shared row.
    try std.testing.expectEqual(@as(usize, 128), c.head_dim);
    try std.testing.expectEqual(@as(usize, 16), c.topk);
    try std.testing.expectEqual(@as(usize, 256), c.rank);
    try std.testing.expect(c.topk <= 32 and c.rank <= 256);
}

test "a round's scratch is one block plus one context chunk, 256-byte aligned, and measured the same twice" {
    const c = try draft2.parse(std.testing.allocator, served_json, testTarget());
    const total = try scratchBytes(c);
    try std.testing.expectEqual(total, try place(c, 0, null));
    try std.testing.expect(total < 4 << 20);
    var b: Bufs = undefined;
    _ = try place(c, 4096, &b);
    // every row starts on a 256-byte boundary inside the buffer
    inline for (@typeInfo(Bufs).@"struct".field_names) |name| {
        const at = @field(b, name) - 4096;
        try std.testing.expect(at % 256 == 0);
        try std.testing.expect(at < total);
    }
    // The prefix's staging is `ctx_chunk` rows, not one window's: that is what keeps a state's
    // scratch independent of how long the prefix a caller hands over is.
    try std.testing.expectEqual(@as(u64, ctx_chunk * c.hidden * 2), b.h - b.ctx);
    try std.testing.expectEqual(@as(u64, ctx_chunk * c.kvWidth() * 2), b.kc - b.kcr);
    // The head's candidates and their logits are one `[rows, topk]` table a tensor — 7 * 16 * 4
    // bytes, padded to the next 256-byte row — and the selector walks those two tables and nothing
    // else. A per-row base on top of the step (`CAND + blockIdx * topk + t * topk + lane`) walks off
    // the ids and into `unary`, which is the 511 GB codebook read the selector now refuses.
    try std.testing.expectEqual(@as(u64, 512), b.unary - b.cand);
    try std.testing.expectEqual(@as(u64, 512), b.prop - b.unary);
    try std.testing.expectEqual(@as(u64, 256), b.err - b.prop);
    // the head's shared plane is inside the default 48 KiB a dynamic launch may ask for
    try std.testing.expect(c.hidden * 2 + 256 * c.topk * 8 <= 48 * 1024);
    // the ring is what a stream costs besides: the window, a block, a tile of slack
    try std.testing.expectEqual(@as(usize, 5 * 2 * 2064 * 1024 * 2), c.ringBytes());
    try std.testing.expectEqual(c.ringBytes() + total, try Draft.stateBytes(c));
}

test "the tap source is one constant, read once" {
    try std.testing.expectEqual(tap_ring.TapSource.default, tap_source);
    const src: tap_ring.Source = .{ .readout = 0x1000, .normed = 0x2000, .streams = 0x3000, .stride = 12800 };
    try std.testing.expectEqual(@as(?u64, 0x1000), tapsFrom(src)); // the default is the read-out row
    try std.testing.expect(tapsFrom(.{}) == null);
}

test {
    _ = @import("cuda_draft2_fwd_test.zig");
}
