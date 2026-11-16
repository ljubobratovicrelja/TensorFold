//! flashnext's DFlash2 drafter forward on CUDA: one block round from five tap rows.
//!
//! The checkpoint (`cuda_draft2.zig` parses it, `cuda_draft2_weights.zig` names its 81 tensors) is a
//! five-layer Qwen3 backbone over the target's residuals, a two-tap dynamic grouped convolution
//! around every attention and MLP, and a low-rank candidate selector over the target head's top-k.
//! A round is:
//!
//! 1. **fuse** the five taps of each of the *prefix's* positions — every position the target had
//!    already computed when it proposed the block — into one 2560-wide context row a position
//!    (`tf_fxd2_fuse`: `rms_norm(hidden_norm, taps @ fc^T)`). The kernel streams the whole of `fc`
//!    a row, so it is a round's work and not a layer's: the fused rows are kept at their positions
//!    (`State.fused`) and every layer cross-attends to the same ones, which also lets a round read
//!    the rows the round before it fused — consecutive rounds share all but the positions the
//!    target accepted. The block's own rows never come from taps: `reference.py` hands the prefix's
//!    taps in as the cross-attention context and starts the block from the target's embedding table
//!    (`anchor`, then `block - 1` mask tokens).
//!    Fusing the block's own taps instead is the "leaky" variant of `tools/dflash2/tap_accept.py`,
//!    which shows the block the next seven positions' hidden states.
//! 2. **five layers**: input norm, attention conv, q/k/v of the block's rows, the prefix's own k/v,
//!    RoPE, block attention (bidirectional over the block, plus the whole prefix), output conv,
//!    then the MLP with its own conv. Both convs are `tf_q35d_dyn_conv` unchanged; the attention is
//!    `tf_fxd2_attn`, which is the Qwen3.5 drafter's loop with DFlash2's visibility.
//! 3. **head**: the target's `lm_head` over the whole 248,320-token vocabulary, top-16 a row, in
//!    two passes (`tf_fxd2_headtile` scores 1,024 entries of one row a CTA, `tf_fxd2_headmerge`
//!    merges the tiles' lists a row); no softcap and no output multiplier, as the reference's
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

/// Rows of the prefix the k/v staging is sized for: a chunk of the context is copied out of the
/// fused ring, projected, rotated and put into the ring `ctx_chunk` rows at a time, so a state costs
/// one chunk's scratch rather than one window's. Every step of it is per row, so a chunk boundary
/// changes no byte.
pub const ctx_chunk = 64;

/// The vocabulary head's tile: the entries of the target's vocabulary one CTA scores in the head's
/// first pass.
///
/// The head is the round's one bandwidth-bound stage — 248,320 rows of 2,560 bf16 is 1.27 GB read a
/// round — and one CTA a row is seven CTAs for the whole card, so the stage ran at whatever seven of
/// twenty SMs can keep in flight (93 ms, against 5-15 ms of the bytes at a full grid). Tiling the
/// vocabulary turns the scan into `tiles * rows` CTAs (1,701 at the served shape, 5 MiB of `lm_head`
/// each) and costs one partial top-k a tile that the second pass merges.
pub const head_tile = 1024;

/// The head's first pass's CTAs a row: the tiles that cover the vocabulary, the last one fewer
/// entries.
pub fn headTiles(c: Config) usize {
    return (c.vocab + head_tile - 1) / head_tile;
}

/// glm_mm16_16's dynamic shared: three stages of 16 rows of x and a 64-column weight tile.
fn m16Shared() u32 {
    return 3 * (16 * 128 + 8 * 32 * 8 * 4 + 128);
}

/// One round's inputs: a block of `block` rows to draft, and the prefix it cross-attends to.
pub const Round = struct {
    /// `[ctx_len, n_taps * hidden]` bf16 on the device: the fused tap rows of the positions
    /// `position - ctx_len .. position - 1`, taps in the config's `target_layer_ids` order (a tap's
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

/// A stream's drafter state: five layers' key and value rings, the prefix's fused rows, and the
/// block's scratch.
///
/// The k/v ring holds the prefix's projected keys and values, position-addressed and trusted only
/// across the span `[kv_lo, kv_end)`: a row is a function of its own position's fused row alone,
/// so a round projects only the positions the span does not cover and reuses the rest — which is
/// what makes a gap (a round whose prefix starts past the previous round's block) a refusal rather
/// than a stale read, and a rewind (`fusedReset`) a full re-projection.
///
/// The fused ring is the exception and carries by design: a fused row is a function of its own
/// position's taps and the same for every layer, so it outlives the round that paid for it
/// (`fusePrefix`). Its span is `[fused_lo, fused_end)`, and it is trusted no further than that: a
/// caller that re-uses positions must say so (`Draft.invalidate`).
///
/// One round at a time per state: the scratch is reused by the next round, and a round ends by
/// synchronising to read its proposals back.
pub const State = struct {
    /// `layers * 2 * ringRows * kvWidth` bf16: layer i's keys at `ringK(i)`, then its values.
    ring: cuda.DeviceBuffer,
    /// `window * hidden` bf16: the prefix's fused context rows, a position's at slot
    /// `position % window`. Sized by the window because that is the most positions a round reads.
    fused: cuda.DeviceBuffer,
    /// The block's rows and the head's candidates (`Bufs`).
    scratch: cuda.DeviceBuffer,
    /// The position after the last row written; 0 until the first round.
    end: u64 = 0,
    /// The first position `fused` does not hold, and the first one it does: the fused rows are
    /// `[fused_lo, fused_end)`, each at its own slot. A round fuses `fused_end .. hi` and reads
    /// what it needs of `lo .. fused_end`.
    fused_end: u64 = 0,
    fused_lo: u64 = 0,
    /// The positions whose projected keys and values the ring holds, for every layer:
    /// `[kv_lo, kv_end)`, each at its own slot. A round projects `reuseFrom(lo, hi, kv_lo, kv_end)
    /// .. hi` and then vouches for the whole prefix; `fusedReset` clears the span with the fusion's.
    kv_end: u64 = 0,
    kv_lo: u64 = 0,

    const Self = @This();

    pub fn deinit(s: *Self) void {
        s.ring.free();
        s.fused.free();
        s.scratch.free();
    }

    /// Forget the fused rows and the projected keys and values: both spans are empty, so the next
    /// round fuses and projects every position it reads, and no slot is trusted on the strength of
    /// an earlier round's write.
    pub fn fusedReset(s: *Self) void {
        s.fused_lo = 0;
        s.fused_end = 0;
        s.kv_lo = 0;
        s.kv_end = 0;
    }
};

/// Where a round's rows are in `State.scratch`; every offset is 256-byte aligned.
pub const Bufs = struct {
    /// The block's token ids, u32: the anchor then `block - 1` mask tokens.
    ids: u64,
    /// The context chunk being read out of the fused ring, `[ctx_stage, hidden]` bf16 (`ctx_stage`
    /// is `ctx_chunk`).
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
    /// The tensor-core head's fp32 logits, `[block - 1, vocab]`, the slice top-k reads: the matmul's
    /// whole-vocabulary output lands here before `tf_fxd2_topf32` trims it a row's tile.
    lg: u64,
    /// The proposals the selector walked, `[block - 1]` u32.
    prop: u64,
    /// The selector's one fault record (`Fault`), cleared before a round and written by the kernel
    /// only if a codebook index is at or past the vocabulary.
    err: u64,
    /// The head's first pass's candidates, `[tiles, block - 1, topk]` of `Fxd2Cand` (`{f32 logit,
    /// u32 id}`): tile `t` of row `r` at `(t * rows + r) * topk`, so a tile CTA writes one
    /// contiguous list and the merge pass reads its row strided. Written whole before the second
    /// pass reads a slot, so it needs no clearing between rounds.
    part: u64,
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
    /// from layer 0's pass over the prefix, out of the round's fused ring: the fusion is row-wise
    /// and a row's bytes are its position's, so every layer reads these same bytes.
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
    /// `tf_fxd2_headtile` reads it a row at a time. Note that the engine's resident head is the same
    /// tensor repacked into matrix-core tiles (`cuda/qmmf.zig`), which none of this module's kernels
    /// can address: a serving caller has to hand over rows, not tiles (see `cuda_draft2_check.zig`).
    head: u64,
};

/// A monotonic reading in nanoseconds: zig 0.17 dropped `std.time.Timer`, and a stage bill measured
/// in anything but wall-clock deltas around the launches is a guess.
fn nowNs(io: std.Io) u64 {
    const t = std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
    return @intCast(@max(0, t));
}

/// The kernels a round launches, by symbol.
///
/// `tf_fxd2_knorm` — the norm without a rotation, for a context whose keys were all rotated at
/// position 0 — is deliberately not among them: a round's prefix keys are rotated at the positions
/// they were captured at, so `tf_q35d_norm_rope` is the entry it resolves. The kernel stays in the
/// image because `kernels/cuda/qwen35_draft.cu`'s bytes are pinned by `dense_serving_test.zig`,
/// which is a reviewed-bytes gate rather than this module's to move. `tf_fxd2_head_topk` is in the
/// image and not resolved for the same reason: the round runs the head as the two passes below, and
/// the single-pass entry stays as the statement of the answer those passes reproduce.
const Fns = struct {
    fuse: cuda.Function,
    rmsnorm: cuda.Function,
    attn: cuda.Function,
    head_tile: cuda.Function,
    head_merge: cuda.Function,
    topf32: cuda.Function,
    select: cuda.Function,
    silu_mul: cuda.Function,
    dyn_conv: cuda.Function,
    norm_rope: cuda.Function,
    ring_put: cuda.Function,
};

/// The drafter, loaded: its config, its weights, and its kernels.
pub const Draft = struct {
    /// Per-stage nanoseconds, kept only when `TF_DRAFT2_TIME` is set: a round that is five times the
    /// target's cost is a fusion bill or a projection bill, and guessing which one wastes a build.
    t_fuse: u64 = 0,
    t_kv: u64 = 0,
    t_layers: u64 = 0,
    t_tail: u64 = 0,
    t_rounds: u64 = 0,
    timing: bool = false,
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
    /// The tensor-core head: glm_mm's bf16 kernels over the target's lm_head, tiled once at open
    /// (`headw`), so a round's head is one matmul into fp32 logits plus the slice top-k.
    mm: cuda.Module,
    m16: cuda.Function,
    tile16: cuda.Function,
    headw: cuda.DeviceBuffer,
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
        s.* = .{ .gpa = gpa, .io = io, .d = d, .stream = stream, .c = c, .w = undefined, .target = target, .mod = undefined, .rows = undefined, .fns = undefined, .mm = undefined, .m16 = undefined, .tile16 = undefined, .headw = undefined };
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
            .head_tile = try s.mod.function("tf_fxd2_headtile"),
            .head_merge = try s.mod.function("tf_fxd2_headmerge"),
            .topf32 = try s.mod.function("tf_fxd2_topf32"),
            .select = try s.mod.function("tf_fxd2_select"),
            .silu_mul = try s.mod.function("tf_fxd2_silu_mul"),
            .dyn_conv = try s.mod.function("tf_q35d_dyn_conv"),
            .norm_rope = try s.mod.function("tf_q35d_norm_rope"),
            .ring_put = try s.mod.function("tf_q35d_ring_put"),
        };
        // the tensor-core head: glm_mm's bf16 row kernels and the one-time tiling of lm_head into
        // their column-tile layout (glm_mm_tile16). The tiled bytes equal the source's — the tiling
        // is a permutation — so the target's own row-major head stays for the check path.
        s.mm = try cuda.Module.load(d, cuda.kernels.glm_mm);
        errdefer s.mm.unload();
        s.m16 = try s.mm.function("glm_mm16_16");
        try s.m16.allowDynamicShared(m16Shared());
        s.tile16 = try s.mm.function("glm_mm_tile16");
        s.headw = try cuda.DeviceBuffer.alloc(d, s.c.vocab * s.c.hidden * 2);
        errdefer s.headw.free();
        var targs: cuda.Args = .{};
        targs.add(s.target.head);
        targs.add(s.headw.ptr);
        targs.add(@as(c_int, @intCast(s.c.vocab)));
        targs.add(@as(c_int, @intCast(s.c.hidden)));
        try cuda.launch.launch(s.tile16, .{ .grid = .{ .x = @intCast((s.c.vocab / 8) * (s.c.hidden / 64) * 32 / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s.stream, &targs);
        s.w = try loadWeights(gpa, io, s, dir);
        errdefer s.w.deinit(gpa);
        if (std.c.getenv("TF_DRAFT2_TIME")) |p| {
            const v = std.mem.span(p);
            s.timing = v.len != 0 and !std.mem.eql(u8, v, "0");
        }
        return s;
    }

    pub fn deinit(s: *Self) void {
        s.w.deinit(s.gpa);
        s.headw.free();
        s.rows.unload();
        s.mod.unload();
        s.mm.unload();
        s.gpa.destroy(s);
    }

    /// A new stream's state: its rings and scratch, the k/v ring and the scratch zeroed so a stray
    /// read sees a key of 0.
    ///
    /// The fused ring is *not* zeroed: nothing reads a slot outside `[fused_lo, fused_end)`, so a
    /// slot no round has written is unreachable rather than a key of the last stream's, and a state's
    /// first round fills the span it reads before it reads it.
    pub fn create(s: *Self) !State {
        var st: State = .{ .ring = try cuda.DeviceBuffer.alloc(s.d, s.c.ringBytes()), .fused = undefined, .scratch = undefined };
        errdefer st.ring.free();
        st.fused = try cuda.DeviceBuffer.alloc(s.d, fusedBytes(s.c));
        errdefer st.fused.free();
        st.scratch = try cuda.DeviceBuffer.alloc(s.d, try scratchBytes(s.c));
        errdefer st.scratch.free();
        try st.ring.fill8(0, s.stream.handle);
        try st.scratch.fill8(0, s.stream.handle);
        try s.stream.synchronize();
        return st;
    }

    /// The bytes a stream's state takes, before it is created: its k/v ring, the fused prefix and
    /// one block's scratch.
    pub fn stateBytes(c: Config) !usize {
        return std.math.add(usize, try std.math.add(usize, c.ringBytes(), fusedBytes(c)), try scratchBytes(c));
    }

    /// The fused ring's bytes: a window of positions, `hidden` bf16 each — one row a position, and a
    /// round never reads more positions than the window holds.
    pub fn fusedBytes(c: Config) usize {
        return c.window * c.hidden * 2;
    }

    /// Forget the fused prefix: the next round fuses every position it needs.
    ///
    /// The caller owns this because it owns the positions. A round already refuses the cache to a
    /// prefix that starts before the fused span or ends inside it — a first round, one whose prefix
    /// grew, a rewind — so what is left is the case the span cannot see: a stream whose *same*
    /// positions now mean different tokens (a pooled state handed to another stream, a restored
    /// prompt, an anchor resampled under a block the last round fused). The rule is: wherever a
    /// caller drops a stream's positions — `cuda_drafter2.zig`'s `State.reset`, and its open path
    /// before a first round — it drops the fusion here too. Clearing `st.end` a round is *not* that
    /// signal, or the ring would never carry anything.
    pub fn invalidate(s: *Self, st: *State) void {
        _ = s;
        st.fusedReset();
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
        if (@as(u64, r.ctx_len) > r.position) return error.ContextPastPosition;
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

        // the five layers' shared cross-attention context, fused once for the round and kept: every
        // layer reads the same rows, and the rows the last round fused are still good
        try s.fusePrefix(st, r);
        try embed(s, b.ids, b.h);
        var rt = nowNs(s.io);
        // the block's input rows (the anchor's embedding and the masks), then one row block a layer:
        // a reader of the dump then sees the state each sublayer was handed, not just the last one
        const row_bytes = B * c.hidden * 2;
        if (s.dbg) |g| if (g.layers != 0) try copyOut(s, g.layers, row_bytes, b.h);
        const rt0 = rt;
        for (s.w.layers, 0..) |l, i| {
            try s.layer(st, &b, l, i, r);
            if (s.dbg) |g| if (g.layers != 0) try copyOut(s, g.layers + (i + 1) * row_bytes, row_bytes, b.h);
        }
        // The head and the selector read the block's rows 1..7, not 0..6: row r sits at position
        // p+r and proposes the token at p+r, so row 0 only re-predicts the anchor. `reference.py`'s
        // `propose` heads `hidden[:, -block+1:]` for the same reason; starting at row 0 lags every
        // proposal by one slot, which reads as chance agreement rather than as an error.
        var tL: u64 = 0;
        var tH: u64 = 0;
        var tS: u64 = 0;
        if (s.timing) {
            try s.stream.synchronize();
            tL = nowNs(s.io);
            s.t_layers += tL - rt;
            rt = tL;
        }
        try rmsnorm(s, b.h + c.hidden * 2, s.w.norm, b.hn, rows);
        try headTopk(s, b.hn, rows, b.part, b.cand, b.unary, b.lg);
        if (s.timing) {
            try s.stream.synchronize();
            tH = nowNs(s.io);
        }
        try select(s, b.hn, rows, b.cand, b.unary, b.err, r.anchor, b.prop);
        if (s.timing) {
            try s.stream.synchronize();
            tS = nowNs(s.io);
        }
        // the proposals are read back on the host, so the round's kernels must have landed first:
        // a blocking DtoH copy does not order itself after work queued on a non-blocking stream
        if (s.timing) {
            const t1 = nowNs(s.io);
            s.t_tail += t1 - rt;
            s.t_rounds += 1;
        }
        var held: [max_block - 1]u32 = @splat(0);
        var fault: Fault = .{};
        try s.stream.synchronize();
        try st.scratch.download(@intCast(b.prop - st.scratch.ptr), std.mem.sliceAsBytes(held[0..rows]));
        try st.scratch.download(@intCast(b.err - st.scratch.ptr), std.mem.asBytes(&fault));
        if (s.timing) {
            std.log.info("draft2 stages: layers {d:.1} ms, head {d:.1} ms, select {d:.1} ms ({d} ctx rows, {d} rows)", .{
                @as(f64, @floatFromInt(tL - rt0)) / 1e6,
                @as(f64, @floatFromInt(tH - tL)) / 1e6,
                @as(f64, @floatFromInt(tS - tH)) / 1e6,
                r.ctx_len,
                rows,
            });
        }
        if (s.timing and s.t_rounds >= 16) {
            const ms = struct {
                fn f(v: u64) f64 {
                    return @as(f64, @floatFromInt(v)) / 1e6;
                }
            }.f;
            const n = @as(f64, @floatFromInt(s.t_rounds));
            std.log.info("draft2 time: {d:.1} ms/round = layers {d:.1} (fuse {d:.1}, kv {d:.1}) + tail {d:.1} | ctx {d} rows, block {d}", .{
                ms(s.t_layers + s.t_tail) / n, ms(s.t_layers) / n, ms(s.t_fuse) / n, ms(s.t_kv) / n, ms(s.t_tail) / n, r.ctx_len, rows,
            });
            s.t_fuse = 0;
            s.t_kv = 0;
            s.t_layers = 0;
            s.t_tail = 0;
            s.t_rounds = 0;
        }
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
        // every layer's ring now holds the prefix's keys and values at their own positions; the
        // next round reuses them across [kv_lo, kv_end) and projects only what the span does not
        // cover. Set here, after the round's kernels landed, so a failed round vouches for nothing.
        st.kv_lo = r.position - r.ctx_len;
        st.kv_end = r.position;
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
        // the context ends one position before the anchor's own row: row 0 of the block is the anchor,
        // embedded and rotated at `position`, and duplicating that position in the keys phases every
        // context key one step late.
        const lo: c_longlong = @intCast(r.position - r.ctx_len);
        // attention: norm, prepare, project, rotate, attend, finish
        try rmsnorm(s, b.h, l.input_norm, b.x, B);
        try dense(s, b.x, l.attn_conv, b.dyn, dyn, H, B);
        try dynConv(s, b.x, b.dyn, l.attn_base, 0, b.xin, 0);
        // the block's attention projections: one launch each for all B rows, so a weight row streams
        // once a layer instead of once a row (`denseBatch`'s bytes are `dense`'s, bit for bit)
        try denseBatch(s, b.xin, l.q, b.qraw, qw, H, B);
        try denseBatch(s, b.xin, l.k, b.kraw, kvw, H, B);
        try denseBatch(s, b.xin, l.v, b.vb, kvw, H, B);
        try normRope(s, b.qraw, l.q_norm, b.q, qw, c.heads, pos, B);
        try normRope(s, b.kraw, l.k_norm, b.kb, kvw, c.kv_heads, pos, B);
        // the prefix's keys and values, into this layer's ring at the positions they were captured at
        try s.prefix(st, b, l, i, r);
        try attn(s, st, i, b, lo, r.ctx_len, B);
        try denseBatch(s, b.att, l.o, b.o, H, qw, B);
        try dynConv(s, b.o, b.dyn, l.attn_base, b.h, b.h2, 1);
        // mlp: norm, prepare, gate/up/silu, down, finish
        try rmsnorm(s, b.h2, l.post_norm, b.x, B);
        try dense(s, b.x, l.mlp_conv, b.dyn, dyn, H, B);
        try dynConv(s, b.x, b.dyn, l.mlp_base, 0, b.xin, 0);
        try denseBatch(s, b.xin, l.gate, b.gate, c.intermediate, H, B);
        try denseBatch(s, b.xin, l.up, b.up, c.intermediate, H, B);
        try siluMul(s, b.gate, b.up, b.act, B * c.intermediate);
        try denseBatch(s, b.act, l.down, b.o, H, c.intermediate, B);
        try dynConv(s, b.o, b.dyn, l.mlp_base, b.h2, b.h, 1);
    }

    /// The layer's cross-attention context: the prefix's fused tap rows, `ctx_chunk` rows at a time,
    /// out of the round's fused ring, through the k/v projections and the per-head norm and rotation
    /// at each row's **own** position, into this layer's ring at those positions.
    ///
    /// Chunking is exact rather than approximate: the projections are row-wise, and the
    /// norm-plus-rotation is per head per row, so a chunk boundary changes no byte of the result — it
    /// only bounds the scratch a state needs to `ctx_chunk` rows instead of one window's.
    /// `reference.py` rotates the context keys at the positions they were captured at
    /// (`ctx_positions`), not at zero: giving every prefix key the same rotation destroys the q/k
    /// geometry the taps were trained against.
    ///
    /// The fusion itself is not here: it is `fusePrefix`'s, once a round, because its output does not
    /// depend on the layer and fusing per layer would pay for it five times over.
    fn prefix(s: *Self, st: *State, b: *const Bufs, l: Layer, i: usize, r: Round) !void {
        const c = s.c;
        const H = c.hidden;
        const kvw = c.kvWidth();
        const lo: u64 = r.position - r.ctx_len;
        const hi: u64 = r.position;
        // the first position the ring does not hold for every layer: a row's keys and values are a
        // function of its own position's fused row alone, so the span [kv_lo, kv_end) a previous
        // round projected is reused under the fused ring's own rule (`reuseFrom`)
        const from: u64 = reuseFrom(lo, hi, st.kv_lo, st.kv_end);
        var done: usize = 0;
        var t = nowNs(s.io);
        while (done < r.ctx_len) {
            const n = @min(ctx_chunk, r.ctx_len - done);
            const at: u64 = lo + @as(u64, @intCast(done));
            // a chunk entirely before `from` is already in the ring at its positions: nothing to
            // project, and only layer 0's debug dump still wants its fused bytes
            const stale: bool = at + n <= from;
            if (!stale) try copyFused(s, st, at, n, b.ctx);
            if (i == 0) {
                if (s.dbg) |g| {
                    if (g.ctx != 0) {
                        if (g.ctx_rows < done + n) return error.DebugCtxTooSmall;
                        // the dump holds the whole prefix the round read, covered rows included
                        if (stale) try copyFused(s, st, at, n, b.ctx);
                        try copyOut(s, g.ctx + done * H * 2, n * H * 2, b.ctx);
                    }
                }
            }
            if (!stale) {
                try dense(s, b.ctx, l.k, b.kcr, kvw, H, n);
                try dense(s, b.ctx, l.v, b.vc, kvw, H, n);
                try normRope(s, b.kcr, l.k_norm, b.kc, kvw, c.kv_heads, @intCast(at), n);
                try ringPut(s, b.kc, b.vc, s.ringK(st, i), s.ringV(st, i), kvw, @intCast(at), n);
            }
            if (s.timing) {
                const t1 = nowNs(s.io);
                s.t_kv += t1 - t;
                t = t1;
            }
            done += n;
        }
    }

    /// The prefix's fused rows for this round: `tf_fxd2_fuse` over the positions the ring does not
    /// already hold, `ctx_chunk` rows a launch, each into its own position's slot.
    ///
    /// The kernel reads all 65 MB of `fc` a row, so fusing inside the layer loop streamed `fc` five
    /// times over one prefix for one answer; fusing once a round is five times less, and fusing only
    /// the positions the last round did not is the rest — the target accepts a few positions a round,
    /// so consecutive prefixes share all but those. Nothing here depends on a layer, and a position's
    /// tap rows are the same bytes every time they are fused, so reading a row an earlier round wrote
    /// is the same result as fusing it again.
    ///
    /// The ring is `window` slots and a prefix is at most a window long (`draft` refuses
    /// `ContextPastWindow`), so a span cannot alias its own slots, and a row a round still needs is
    /// overwritten only by a position a window ahead of it — which a prefix that short cannot fuse.
    fn fusePrefix(s: *Self, st: *State, r: Round) !void {
        const c = s.c;
        const H = c.hidden;
        const k = c.n_taps * H;
        const lo: u64 = r.position - r.ctx_len;
        const hi: u64 = r.position;
        var p: u64 = fuseFrom(lo, hi, st.fused_lo, st.fused_end);
        var t = nowNs(s.io);
        while (p < hi) {
            const n: usize = @intCast(@min(@as(u64, ctx_chunk), hi - p));
            const src = r.ctx + (p - lo) * k * 2;
            // the destination wraps at most once a chunk, because the ring is a window wide and a
            // chunk is `ctx_chunk`: two launches over one chunk are the same bytes as one, the
            // kernel being strictly row-wise
            const slot: usize = @intCast(p % c.window);
            const head = @min(n, c.window - slot);
            try fuse(s, src, st.fused.ptr + slot * H * 2, head);
            if (n > head) try fuse(s, src + head * k * 2, st.fused.ptr, n - head);
            if (s.timing) {
                const t1 = nowNs(s.io);
                s.t_fuse += t1 - t;
                t = t1;
            }
            p += n;
        }
        // the span this round can vouch for: everything it read, fused now or by the round that did
        // it at the same position
        st.fused_lo = lo;
        st.fused_end = hi;
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

/// One device-to-device copy on the round's own stream: queued, not waited on, so the round's single
/// synchronisation covers it too. `dst` is a borrowed span the module does not own — a caller's debug
/// buffer, or a staging row block in the round's own scratch.
fn copyOut(s: *Draft, dst: u64, len: usize, src: u64) !void {
    if (len == 0) return;
    // a borrowed view of the destination: `copyFrom` writes rows, it does not own them
    const view: cuda.memory.DeviceBuffer = .{ .d = s.d, .ptr = dst, .len = len, .borrowed = true };
    try view.copyFrom(0, src, len, s.stream.handle);
}

/// The fused rows of positions `[at, at + n)` out of the round's fused ring, into `dst` in position
/// order: at most two copies, because the ring wraps once a window and a chunk is `ctx_chunk` rows.
///
/// This is the whole of what a layer reads of the fusion, and the bytes are the fusion's own — a
/// row's slot holds the row `fusePrefix` wrote for that position, never a partial or a neighbouring
/// one, so a chunk read from here is what fusing the chunk would have produced.
fn copyFused(s: *Draft, st: *State, at: u64, n: usize, dst: u64) !void {
    const row = s.c.hidden * 2;
    const first: usize = @intCast(at % s.c.window);
    const head = @min(n, s.c.window - first);
    try copyOut(s, dst, head * row, st.fused.ptr + first * row);
    if (n > head) try copyOut(s, dst + head * row, (n - head) * row, st.fused.ptr);
}

/// The position a round's fusion starts at, `[p, hi)`, given the prefix it needs and the span the
/// ring holds: the first position the ring does not hold, when the prefix still covers what the
/// ring does hold, and the prefix's own first position when it does not.
///
/// The two cases the ring's span cannot cover, and why they are a miss rather than a reuse:
/// a prefix that starts *before* `fused_lo` needs rows no round fused (a first round, a prefix that
/// grew past its window), and a prefix that ends at or before `fused_end` is a round the target
/// rewound — its rows may have been re-derived from different tokens, and the anchor is one of
/// them. Both fuse the whole prefix, which is always correct, at the cost of the round's saving.
fn fuseFrom(lo: u64, hi: u64, fused_lo: u64, fused_end: u64) u64 {
    return if (lo >= fused_lo and hi > fused_end) @max(lo, fused_end) else lo;
}

/// Where a round's k/v projection starts: the fused ring's rule over the span of positions the
/// ring already holds for every layer. A row's keys and values are a function of its own
/// position's fused row alone, so covered rows are the same bytes a re-projection would produce,
/// and the two cases `fuseFrom` refuses — a prefix that starts before the covered span, and one
/// that ends inside it (a rewind) — refuse here too.
fn reuseFrom(lo: u64, hi: u64, kv_lo: u64, kv_end: u64) u64 {
    return fuseFrom(lo, hi, kv_lo, kv_end);
}

/// tf_fxd2_fuse(X, W, NORM, OUT, K, H, eps): grid (rows), 256 threads, `(K + 2H) * 2` shared.
///
/// `taps` is the first of `rows` contiguous `n_taps * hidden` rows: the prefix's rows from an
/// arbitrary position, since `fusePrefix` fuses only the ones the ring is missing. `rows` is a chunk
/// of the context, so it may not outrun the scratch's `ctx_chunk`, and its `out` is a slot of the
/// fused ring — which may be the ring's start, when a chunk wraps.
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
///
/// One CTA an output element, so the weight row a column streams once a row. For the block's own
/// projections that is the whole of a layer's DRAM traffic times the block: `denseBatch` is the same
/// dot with the block's rows in one CTA, and the same bytes out.
fn dense(s: *Draft, x: u64, w: u64, out: u64, n: usize, k: usize, rows: usize) !void {
    return s.rows.denseRows(s.stream, x, w, out, rows, n, k);
}

/// `Y[r, j] = X[r, :] @ W[j, :]` for the block's `rows` at once (`qwen35_bf16_rows.cu`'s
/// `tf_q35_dense_bf16_batch`: grid (n, 1), 256 threads, no shared memory).
///
/// Bit-identical to `dense` for every (row, j), so the kernels after it — the per-head norm and
/// rotation, the silu-mul, the dynamic convolution — read exactly the bytes they read before.
fn denseBatch(s: *Draft, x: u64, w: u64, out: u64, n: usize, k: usize, rows: usize) !void {
    return s.rows.denseBatch(s.stream, x, w, out, rows, n, k);
}

/// tf_fxd2_silu_mul(GATE, UP, OUT, n): one thread an element.
fn siluMul(s: *Draft, gate: u64, up: u64, out: u64, n: usize) !void {
    try launch(s, s.fns.silu_mul, .{ @intCast((n + 255) / 256), 1, 1 }, 256, 0, .{ gate, up, out, ci(n) });
}

/// `out[r] = table[ids[r]]`, a bit copy a row (`tf_q35_embed_bf16`: grid (rows), 256 threads).
fn embed(s: *Draft, ids: u64, out: u64) !void {
    return s.rows.embedRows(s.stream, ids, s.target.embed, out, s.c.block, s.c.hidden, 0);
}

/// The head, two passes: `tf_fxd2_headtile(X, HEAD, PARTIAL, H, V, topk, rows, tile)` on grid
/// (tiles) — one CTA a tile, looping the block's rows so the tile's weights stream once — then
/// `tf_fxd2_headmerge(PARTIAL, IDS, VAL, tiles, rows, topk)` on grid (rows);
/// 256 threads each, `part` between them a `[tiles, rows, topk]` table of candidates.
///
/// The split is occupancy, not arithmetic: the scan reads 1.27 GB of `lm_head` a round, and one CTA
/// a row is seven CTAs of the card's twenty SMs, which measured 93 ms against a 5-15 ms band. The
/// tiles are `head_tile` entries each, so `tiles` CTAs stream the head once at a grid the card can
/// fill; a CTA's list covers a 1,024-entry slice, which is enough headroom over `topk` that
/// the merge pass sees every entry that could be in the row's top-k.
///
/// What the passes must reproduce is the single-pass form's answer (`tf_fxd2_head_topk`): the head's
/// fp32 logits, the top `topk` a row descending, ids and values as `[rows, topk]` tables the
/// selector walks by step. A logit that ties across a tile boundary orders by which list the merge
/// read first, which is the same caveat `fxd2_insert` already carries for threads.
fn headTopk(s: *Draft, x: u64, rows: usize, part: u64, ids: u64, val: u64, lg: u64) !void {
    const c = s.c;
    // `cand` and `unary` are booked for the block's first `block - 1` rows and no more: a row past
    // that writes past the scratch, and a caller that scored eight rows would be as wrong as one
    // that scored none. `part` is booked for `rows` the same way.
    if (rows == 0 or rows > c.block - 1) return error.BadProposalCount;
    const tiles = headTiles(c);
    const threads: u32 = 256;
    // the dot on glm_mm's tensor cores: [rows, H] x [V, H]^T into fp32 logits — one BM=16 block
    // (rows <= 7 fit it) streaming the tiled head once — then the slice top-k over those logits
    // and the merge, which is exactly the two thirds the old fused tile pass split into
    try launch(s, s.m16, .{ @intCast((c.vocab + 63) / 64), 1, 1 }, 128, m16Shared(), .{
        x,                      @as(c_longlong, @intCast(c.hidden)), s.headw.ptr, @as(u64, 0), lg, @as(u64, 0),
        @as(c_int, 1),          @as(c_int, @intCast(rows)),          @as(c_int, @intCast(c.vocab)),
        @as(c_int, @intCast(c.hidden)), @as(c_int, 1),
    });
    const slice_shared: usize = (threads / 32) * c.topk * 8;
    try launch(s, s.fns.topf32, .{ @intCast(tiles), 1, 1 }, threads, @intCast(slice_shared), .{
        lg, part, ci(c.vocab), ci(c.topk), ci(rows), ci(head_tile),
    });
    const merge_shared = @as(usize, threads) * c.topk * 8;
    try launch(s, s.fns.head_merge, .{ @intCast(rows), 1, 1 }, threads, @intCast(merge_shared), .{
        part, ids, val, ci(tiles), ci(rows), ci(c.topk),
    });
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
        // the head's first pass: one `topk` list a tile a row, `{f32 logit, u32 id}` together, in
        // `tf_fxd2_headtile`'s tile-major order
        .{ "part", try bytes(.{ headTiles(c), R, c.topk, 8 }) },
        // the tensor-core head's logits, `[R, vocab]` fp32, read by the slice top-k; the tiled path
        // splits the dot and the slice into two kernels, so the whole-vocabulary output needs a home
        .{ "lg", try bytes(.{ R, c.vocab, 4 }) },
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

test "the block's seven projections are one launch each, at the widths and input widths they were" {
    // `layer` hands `denseBatch` the same buffers, widths and `B` rows the seven `dense` calls took, so
    // nothing about `Bufs` or `place` moves with this: what is pinned here is the shape of the seven
    // swaps — the weight's row count and the input width each projection is asked for.
    const c = try draft2.parse(std.testing.allocator, served_json, testTarget());
    const B = c.block;
    const kern = cuda.qwen35_bf16_rows;
    // q, k, v, o, gate, up, down: n (the weight's rows) and k (the input width), as `layer` passes them
    const proj = [_][2]usize{
        .{ c.qWidth(), c.hidden },
        .{ c.kvWidth(), c.hidden },
        .{ c.kvWidth(), c.hidden },
        .{ c.hidden, c.qWidth() },
        .{ c.intermediate, c.hidden },
        .{ c.intermediate, c.hidden },
        .{ c.hidden, c.intermediate },
    };
    try std.testing.expectEqual(@as(usize, 7), proj.len);
    // the served block is one launch a projection: the kernel takes up to 16 rows
    try std.testing.expect(B <= kern.batch_rows);
    for (proj) |p| {
        const n = p[0];
        const k = p[1];
        // `denseBatch`'s two refusals: a width its int argument cannot carry, and an input width that
        // is not whole uint4 words (the loop reads k / 8 of them, as the per-element kernel does)
        try std.testing.expect(n > 0 and n <= std.math.maxInt(i32));
        try std.testing.expectEqual(@as(usize, 0), k % 8);
    }
    // the seven widths at the served shape: 4096 q/o, 1024 k/v, 7680 gate/up, 2560 down's rows
    try std.testing.expectEqual(@as(usize, 4096), c.qWidth());
    try std.testing.expectEqual(@as(usize, 1024), c.kvWidth());
    try std.testing.expectEqual(@as(usize, 7680), c.intermediate);
}

test "a round's scratch is one block plus one context chunk, 256-byte aligned, and measured the same twice" {
    const c = try draft2.parse(std.testing.allocator, served_json, testTarget());
    const total = try scratchBytes(c);
    try std.testing.expectEqual(total, try place(c, 0, null));
    try std.testing.expect(total < 16 << 20);
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
    // The head's first pass leaves `[tiles, rows, topk]` candidates behind — 243 tiles of 16 a row
    // at the served shape, 213 KiB — and the second pass reads exactly that and nothing else. It is
    // the round's only table sized by the tile count, so its bytes move with `head_tile` alone.
    try std.testing.expectEqual(@as(u64, 256), b.part - b.err);
    try std.testing.expectEqual(headTiles(c) * (c.block - 1) * c.topk * 8 + (c.block - 1) * c.vocab * 4, total - @as(usize, @intCast(b.part - 4096)));
    try std.testing.expectEqual(@as(usize, 243), headTiles(c));
    // the head's shared plane is inside the default 48 KiB a dynamic launch may ask for
    try std.testing.expect(c.hidden * 2 + 256 * c.topk * 8 <= 48 * 1024);
    // the ring is what a stream costs besides: the window, a block, a tile of slack
    try std.testing.expectEqual(@as(usize, 5 * 2 * 2064 * 1024 * 2), c.ringBytes());
    // and the fused prefix is a window of `hidden`-wide rows, 10 MiB a stream: the most positions a
    // round reads, one row each, whichever of them this round has to fuse
    try std.testing.expectEqual(@as(usize, 2048 * 2560 * 2), Draft.fusedBytes(c));
    try std.testing.expectEqual(c.ringBytes() + Draft.fusedBytes(c) + total, try Draft.stateBytes(c));
}

test "a round fuses only the positions the round before it did not" {
    // A stream whose last round fused [0, 101): a round at position 107 over a 110-row prefix needs
    // [0, 108) and fuses the seven positions the target advanced — the 101 rows it already holds are
    // the same bytes, because a fused row is a function of its own position's taps alone.
    try std.testing.expectEqual(@as(u64, 101), fuseFrom(0, 108, 0, 101));
    // the same round twice: positions [0, 101) end where the fused span does, so nothing is trusted
    // (the anchor may have been resampled) and the whole prefix is fused again
    try std.testing.expectEqual(@as(u64, 0), fuseFrom(0, 101, 0, 101));
    // a first round, and a state whose fused span was dropped (`Draft.invalidate`): nothing to read
    try std.testing.expectEqual(@as(u64, 0), fuseFrom(0, 108, 0, 0));
    // a prefix that grew past the fused span's start needs rows no round fused: a window sliding
    // forward is the normal case, so a prefix reaching back behind it is a full fusion
    try std.testing.expectEqual(@as(u64, 900), fuseFrom(900, 1100, 950, 1000));
    // and the window sliding forward reuses everything it can: [1000, 1100) needs fusing, [901, 1000)
    // is what the last round left
    try std.testing.expectEqual(@as(u64, 1000), fuseFrom(901, 1100, 700, 1000));
}

test "a round projects only the k/v positions the round before it did not" {
    // The fused ring's rule, applied to the k/v span: a row's keys and values are a function of
    // its own position's fused row alone, so covered positions are the same bytes a
    // re-projection would produce and only the positions the target advanced are projected.
    try std.testing.expectEqual(@as(u64, 101), reuseFrom(0, 108, 0, 101));
    // the same round twice (an overlap or a rewind): project the whole prefix again
    try std.testing.expectEqual(@as(u64, 0), reuseFrom(0, 101, 0, 101));
    // a first round, and a state whose spans were dropped (`Draft.invalidate`)
    try std.testing.expectEqual(@as(u64, 0), reuseFrom(0, 108, 0, 0));
    // a prefix that grew back past the covered span's start: a full projection
    try std.testing.expectEqual(@as(u64, 900), reuseFrom(900, 1100, 950, 1000));
    // the window sliding forward reuses everything it can
    try std.testing.expectEqual(@as(u64, 1000), reuseFrom(901, 1100, 700, 1000));
}

test "fusedReset drops the k/v span with the fusion" {
    var st: State = .{ .ring = undefined, .fused = undefined, .scratch = undefined,
        .kv_lo = 7, .kv_end = 99, .fused_lo = 3, .fused_end = 77 };
    st.fusedReset();
    try std.testing.expectEqual(@as(u64, 0), st.kv_lo);
    try std.testing.expectEqual(@as(u64, 0), st.kv_end);
    try std.testing.expectEqual(@as(u64, 0), st.fused_lo);
    try std.testing.expectEqual(@as(u64, 0), st.fused_end);
}

test "the tap source is one constant, read once" {
    try std.testing.expectEqual(tap_ring.TapSource.default, tap_source);
    const src: tap_ring.Source = .{ .readout = 0x1000, .normed = 0x2000, .streams = 0x3000, .stride = 12800 };
    // The default is `mean4`, so the row a stream must offer is its four raw streams (which this
    // module's caller collapses); the read-out and normed rows stay reachable through the override.
    try std.testing.expectEqual(@as(?u64, 0x3000), tapsFrom(src));
    try std.testing.expectEqual(@as(?u64, 0x1000), src.ptr(.readout).?);
    try std.testing.expectEqual(@as(?u64, 0x2000), src.ptr(.normed).?);
    try std.testing.expect(tapsFrom(.{}) == null);
}

test {
    _ = @import("cuda_draft2_fwd_test.zig");
}
