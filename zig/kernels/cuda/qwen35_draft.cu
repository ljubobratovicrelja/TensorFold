// DFlash2 drafter glue on CUDA (cuda_draft.zig): a block of rows over the target's taps, draft_ops.metal's rounding.

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

namespace tf_q35d {

constexpr int HD = 128;  // the drafter's head width (q, k, v)

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

using tf_common::warp_sum;

}  // namespace tf_q35d

using namespace tf_q35d;

// Tap rows (start + i) of a stream's ring of R rows, `width` bf16 each, into OUT rows 0..n. Grid (n), 256 threads.
extern "C" __global__ void tf_q35d_gather_taps(const __nv_bfloat16* __restrict__ RING, int R, long long start, int width,
                                                 __nv_bfloat16* __restrict__ OUT) {
    const long long row = start + blockIdx.x;
    const __nv_bfloat16* src = RING + (size_t)(row % R) * width;
    __nv_bfloat16* dst = OUT + (size_t)blockIdx.x * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[i] = src[i];
}

// Rows of a window to gather, by value (a tree's kept path).
struct TapRows {
    int n;
    int rows[64];
};

// Tap rows (start + rows.rows[i]) of a stream's ring into OUT rows 0..n: a kept path's rows in order. Grid (n).
extern "C" __global__ void tf_q35d_gather_rows(const __nv_bfloat16* __restrict__ RING, int R, long long start, TapRows rows, int width,
                                                 __nv_bfloat16* __restrict__ OUT) {
    const long long row = start + rows.rows[blockIdx.x];
    const __nv_bfloat16* src = RING + (size_t)(row % R) * width;
    __nv_bfloat16* dst = OUT + (size_t)blockIdx.x * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[i] = src[i];
}

// y[t, c] = sum over lags l of (BASE[l, c] + DYN[t, l, c / group]) * x[t - l, c] in bf16 steps; x[-1] = 0; RES last.
extern "C" __global__ void tf_q35d_dyn_conv(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ DYN,
                                              const __nv_bfloat16* __restrict__ BASE, const __nv_bfloat16* __restrict__ RES,
                                              __nv_bfloat16* __restrict__ OUT, int D, int group, int branch) {
    const int t = blockIdx.x;
    const int groups = D / group;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        const int g = c / group;
        float total = 0.0f;
#pragma unroll
        for (int l = 0; l < 2; l++) {
            if (t - l < 0) continue;
            const float x = bf(X[(size_t)(t - l) * D + c]);
            total = rbf(total + rbf(bf(BASE[(branch * 2 + l) * D + c]) * x));
            total = rbf(total + rbf(bf(DYN[(size_t)t * (4 * groups) + (branch * 2 + l) * groups + g]) * x));
        }
        if (RES != nullptr) total = rbf(total + bf(RES[(size_t)t * D + c]));
        OUT[(size_t)t * D + c] = __float2bfloat16_rn(total);
    }
}

// Per-head RMSNorm then rotate-half RoPE at position pos0 + row; OUT packed (rows, heads, HD), grid (rows * heads).
extern "C" __global__ void __launch_bounds__(32) tf_q35d_norm_rope(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                     __nv_bfloat16* __restrict__ OUT, int heads, int stride, long long pos0,
                                                                     float eps, float theta) {
    const int head = blockIdx.x % heads;
    const size_t m = blockIdx.x / heads;
    const int t = threadIdx.x;
    const __nv_bfloat16* x = X + m * stride + head * HD;
    float v[4], acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = bf(x[4 * t + i]);
        acc += v[i] * v[i];
    }
    acc = warp_sum(acc);
    const float inv = rsqrtf(acc / (float)HD + eps);
    float n[4];
#pragma unroll
    for (int i = 0; i < 4; i++) n[i] = rbf(rbf(v[i] * inv) * bf(W[4 * t + i]));
    // the pair of lane element d is d + HD / 2: this lane's partner values come from the lane HD / 8 = 16 away
    const float pos = (float)(pos0 + (long long)m);
    __nv_bfloat16* out = OUT + (m * heads + head) * HD;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = 4 * t + i;
        const float partner = __shfl_xor_sync(0xffffffffu, n[i], 16);
        const int j = d < HD / 2 ? d : d - HD / 2;
        const float angle = pos * powf(theta, -(float)j / (float)(HD / 2));
        float sn, cs;
        sincosf(angle, &sn, &cs);
        out[d] = __float2bfloat16_rn(d < HD / 2 ? n[i] * cs - partner * sn : n[i] * cs + partner * sn);
    }
}

// Rows of packed (kv_heads, HD) keys and values at positions pos0 + i into a layer's rings (slot pos % W).
extern "C" __global__ void tf_q35d_ring_put(const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
                                              __nv_bfloat16* __restrict__ RK, __nv_bfloat16* __restrict__ RV, int width, long long pos0, int W) {
    const int i = blockIdx.x;
    const size_t slot = (size_t)((pos0 + i) % W);
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        RK[slot * width + c] = K[(size_t)i * width + c];
        RV[slot * width + c] = V[(size_t)i * width + c];
    }
}

// Block attention: a row sees ring positions [max(begin, p - window + 1), pos0) and every block row; a warp a row.
extern "C" __global__ void __launch_bounds__(32) tf_q35d_attn(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KB,
                                                                const __nv_bfloat16* __restrict__ VB, const __nv_bfloat16* __restrict__ RK,
                                                                const __nv_bfloat16* __restrict__ RV, int rows, int heads, int kv_heads, int W,
                                                                long long begin, long long pos0, int window, __nv_bfloat16* __restrict__ OUT) {
    constexpr int DPL = HD / 32;
    const int lane = threadIdx.x;
    const int m = blockIdx.x, h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int width = kv_heads * HD;
    const float scale = rsqrtf((float)HD);
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[((size_t)m * heads + h) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    const long long p = pos0 + m;
    long long first = p - window + 1;
    if (first < begin) first = begin;
    for (long long pos = first; pos < pos0; pos++) {
        const __nv_bfloat16* k = RK + (size_t)(pos % W) * width + kvh * HD + lane * DPL;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(k[i]), dot);
        const float s = warp_sum(dot) * scale;
        const float mn = fmaxf(mx, s);
        const float a = __expf(mx - mn), b = __expf(s - mn);
        l = l * a + b;
        const __nv_bfloat16* v = RV + (size_t)(pos % W) * width + kvh * HD + lane * DPL;
#pragma unroll
        for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(v[i]), o[i] * a);
        mx = mn;
    }
    for (int r = 0; r < rows; r++) {
        const __nv_bfloat16* k = KB + ((size_t)r * kv_heads + kvh) * HD + lane * DPL;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(k[i]), dot);
        const float s = warp_sum(dot) * scale;
        const float mn = fmaxf(mx, s);
        const float a = __expf(mx - mn), b = __expf(s - mn);
        l = l * a + b;
        const __nv_bfloat16* v = VB + ((size_t)r * kv_heads + kvh) * HD + lane * DPL;
#pragma unroll
        for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(v[i]), o[i] * a);
        mx = mn;
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) OUT[((size_t)m * heads + h) * HD + lane * DPL + i] = __float2bfloat16_rn(o[i] / l);
}

#include "qwen35_draft_frame.cuh"
#include "qwen35_dspine.cuh"

// --------------------------------------------------------------------------------------------
// flashnext's DFlash2 drafter (cuda_draft2_fwd.zig): the entries a Qwen3.5 drafter has no use for.
//
// A flashnext DFlash2 drafter reads five of the target's residuals as its cross-attention context
// and runs a block of 8 rows: row 0 the anchor the target already produced, rows 1..7 the mask
// token. Three things differ from the drafter above: its block attention is bidirectional within
// the block (`is_causal: false`), it ships no head so it scores the target's whole 248,320-token
// vocabulary every round, and a low-rank candidate selector re-ranks that top-k into one chain.
// The rest of a round reuses this file's dyn_conv, norm_rope, ring_put and attn loops unchanged,
// and qwen35_bf16_rows.cu's dense row for the plain bf16 projections.
// --------------------------------------------------------------------------------------------

// The selector's held candidates a row (cuda_draft2.zig's lp_top) and the rank its codebooks project to.
constexpr int FXD2_TOP = 20;
constexpr int FXD2_RANK = 256;

// The proposal a round whose selector faulted writes instead of a token: no vocabulary holds this
// id, and the host refuses the round on seeing it (cuda_draft2_fwd.zig's `bad_proposal`).
constexpr unsigned FXD2_BAD = 0xFFFFFFFFu;
// "no candidate of this step was out of vocabulary" in the shared slot that collects one.
constexpr unsigned FXD2_NONE = 0xFFFFFFFFu;
// Which codebook an out-of-vocabulary index was about to index (cuda_draft2_fwd.zig's `Fault`).
constexpr unsigned FXD2_KIND_SUCCESSOR = 1;
constexpr unsigned FXD2_KIND_PREDECESSOR = 2;

// The one fault record a round reports: `{ flag, kind, position, index }`, the layout
// cuda_draft2_fwd.zig's `Fault` reads back. Written once, by thread 0, together with FXD2_BAD into
// every proposal of the round, so a caller that only reads the proposals still cannot accept one.
__device__ __forceinline__ void fxd2_fault(unsigned* OUT, unsigned* ERR, unsigned kind, int t, unsigned index, int rows) {
    if (threadIdx.x != 0) return;
    if (ERR != nullptr && ERR[0] == 0u) {
        ERR[0] = 1u;
        ERR[1] = kind;
        ERR[2] = (unsigned)t;
        ERR[3] = index;
    }
    for (int i = 0; i < rows; i++) OUT[i] = FXD2_BAD;
}

// One candidate: the head's fp32 logit and the token it belongs to.
struct Fxd2Cand {
    float v;
    int id;
};

// `v` offered to the best `n` of `top`, strictly: a tie keeps the candidate offered first. For one
// thread's own scan that is the earlier token id, which is argmax's rule; across a block's threads
// it is the lower thread index, and an exact tie in the head's fp32 logits is not a case worth a
// second pass to order.
__device__ __forceinline__ void fxd2_insert(Fxd2Cand* top, int n, float v, int id) {
    if (!(v > top[n - 1].v)) return;
    int at = n - 1;
    while (at > 0 && v > top[at - 1].v) {
        top[at] = top[at - 1];
        at--;
    }
    top[at].v = v;
    top[at].id = id;
}

// ctx[r, :] = rms_norm(hidden_norm, taps[r, :] @ FC^T). Grid (rows), 256 threads, dynamic shared of
// (K + 2H) * 2 bytes: the row once, then its fp32 outputs. K % 8 == 0 (the weights are read eight
// bf16 at a time) and K even (the fp32 plane behind the row must stay 4-byte aligned).
extern "C" __global__ void __launch_bounds__(256) tf_fxd2_fuse(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                               const __nv_bfloat16* __restrict__ NORM, __nv_bfloat16* __restrict__ OUT,
                                                               int K, int H, float eps) {
    extern __shared__ __nv_bfloat16 fxd2_sh[];
    __nv_bfloat16* xs = fxd2_sh;                            // the row: every output reads all of it
    float* acc = reinterpret_cast<float*>(fxd2_sh + K);      // and the fp32 outputs the norm needs whole
    __shared__ float red[8];
    const __nv_bfloat16* x = X + (size_t)blockIdx.x * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const uint4* xv = reinterpret_cast<const uint4*>(xs);
    const int words = K / 8;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        const uint4* w = reinterpret_cast<const uint4*>(W + (size_t)j * K);
        float s = 0.0f;
        for (int c = 0; c < words; c++) {
            const uint4 a = xv[c], b = w[c];
            const __nv_bfloat16* ab = reinterpret_cast<const __nv_bfloat16*>(&a);
            const __nv_bfloat16* bb = reinterpret_cast<const __nv_bfloat16*>(&b);
#pragma unroll
            for (int e = 0; e < 8; e++) s = fmaf(bf(ab[e]), bf(bb[e]), s);
        }
        acc[j] = s;
    }
    __syncthreads();
    float ss = 0.0f;
    for (int j = threadIdx.x; j < H; j += blockDim.x) ss = fmaf(acc[j], acc[j], ss);
    const float inv = rsqrtf(tf_common::block_sum_serial(ss, red) / (float)H + eps);
    for (int j = threadIdx.x; j < H; j += blockDim.x)
        OUT[(size_t)blockIdx.x * H + j] = __float2bfloat16_rn(acc[j] * inv * bf(NORM[j]));
}

// out[r, :] = rms_norm(w, x[r, :]) over the whole hidden width. Grid (rows), 256 threads: a layer's
// input norm, its post-attention norm and the drafter's final one, so the head and the selector
// read one row rather than two roundings of it.
extern "C" __global__ void __launch_bounds__(256) tf_fxd2_rmsnorm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                  __nv_bfloat16* __restrict__ OUT, int H, float eps) {
    __shared__ float red[8];
    const size_t base = (size_t)blockIdx.x * H;
    float ss = 0.0f;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        const float v = bf(X[base + j]);
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(tf_common::block_sum_serial(ss, red) / (float)H + eps);
    for (int j = threadIdx.x; j < H; j += blockDim.x) OUT[base + j] = __float2bfloat16_rn(bf(X[base + j]) * inv * bf(W[j]));
}

// Per-head RMSNorm of packed (rows, heads, HD) rows with no rotation: the context keys' RoPE is at
// position 0, which is the identity, so the fused taps' keys are normalised and stored unrotated
// (reference.py's `rope(k, cat([zeros(1, ctx_len), positions]))`). Grid (rows * heads), warp a head.
extern "C" __global__ void __launch_bounds__(32) tf_fxd2_knorm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                               __nv_bfloat16* __restrict__ OUT, int heads, int stride, float eps) {
    const int head = blockIdx.x % heads;
    const size_t m = blockIdx.x / heads;
    const int t = threadIdx.x;
    const __nv_bfloat16* x = X + m * stride + head * HD;
    float v[4], acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = bf(x[4 * t + i]);
        acc += v[i] * v[i];
    }
    acc = warp_sum(acc);
    const float inv = rsqrtf(acc / (float)HD + eps);
    __nv_bfloat16* out = OUT + (m * heads + head) * HD;
#pragma unroll
    for (int i = 0; i < 4; i++) out[4 * t + i] = __float2bfloat16_rn(rbf(rbf(v[i] * inv) * bf(W[4 * t + i])));
}

// DFlash2's block attention: a row sees ring positions [max(begin, p - window + 1), pos0 + ctx) and
// every block row, with no causal guard (is_causal: false). The `ctx` ring rows at pos0 .. pos0 +
// ctx are this round's fused-tap keys and values, written by ring_put just before. A warp a
// (row, head); grid (rows, heads), as the Qwen3.5 drafter's.
extern "C" __global__ void __launch_bounds__(32) tf_fxd2_attn(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KB,
                                                              const __nv_bfloat16* __restrict__ VB, const __nv_bfloat16* __restrict__ RK,
                                                              const __nv_bfloat16* __restrict__ RV, int rows, int heads, int kv_heads, int W,
                                                              long long begin, long long pos0, int window, int ctx,
                                                              __nv_bfloat16* __restrict__ OUT) {
    constexpr int DPL = HD / 32;
    const int lane = threadIdx.x;
    const int m = blockIdx.x, h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int width = kv_heads * HD;
    const float scale = rsqrtf((float)HD);
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[((size_t)m * heads + h) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    const long long p = pos0 + m;
    long long first = p - window + 1;
    if (first < begin) first = begin;
    const long long last = pos0 + ctx;  // this round's context rows are visible to every block row
    for (long long pos = first; pos < last; pos++) {
        const __nv_bfloat16* k = RK + (size_t)(pos % W) * width + kvh * HD + lane * DPL;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(k[i]), dot);
        const float s = warp_sum(dot) * scale;
        const float mn = fmaxf(mx, s);
        const float a = __expf(mx - mn), b = __expf(s - mn);
        l = l * a + b;
        const __nv_bfloat16* v = RV + (size_t)(pos % W) * width + kvh * HD + lane * DPL;
#pragma unroll
        for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(v[i]), o[i] * a);
        mx = mn;
    }
    for (int r = 0; r < rows; r++) {
        const __nv_bfloat16* k = KB + ((size_t)r * kv_heads + kvh) * HD + lane * DPL;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(k[i]), dot);
        const float s = warp_sum(dot) * scale;
        const float mn = fmaxf(mx, s);
        const float a = __expf(mx - mn), b = __expf(s - mn);
        l = l * a + b;
        const __nv_bfloat16* v = VB + ((size_t)r * kv_heads + kvh) * HD + lane * DPL;
#pragma unroll
        for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(v[i]), o[i] * a);
        mx = mn;
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) OUT[((size_t)m * heads + h) * HD + lane * DPL + i] = __float2bfloat16_rn(o[i] / l);
}

// The head over the target's whole vocabulary: logits = x @ HEAD^T in fp32, then the top `topk` of
// each row, descending, as token ids and values. No softcap and no output multiplier: the
// reference's transform_unary_logits is an identity here. Grid (rows), 256 threads, dynamic shared
// of H * 2 + 256 * topk * 8 bytes. H % 8 == 0. Each thread keeps the best `topk` of its own stride
// of the vocabulary in registers, the block merges 256 lists in shared.
extern "C" __global__ void __launch_bounds__(256) tf_fxd2_head_topk(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ HEAD,
                                                                    unsigned* __restrict__ IDS, float* __restrict__ VAL,
                                                                    int H, int V, int topk) {
    extern __shared__ __nv_bfloat16 fxd2_sh[];
    __nv_bfloat16* xs = fxd2_sh;
    Fxd2Cand* all = reinterpret_cast<Fxd2Cand*>(fxd2_sh + H);
    const __nv_bfloat16* x = X + (size_t)blockIdx.x * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    Fxd2Cand mine[FXD2_TOP];
    for (int i = 0; i < topk; i++) {
        mine[i].v = -INFINITY;
        mine[i].id = 0;
    }
    const uint4* xv = reinterpret_cast<const uint4*>(xs);
    const int words = H / 8;
    // A warp owns one vocabulary entry at a time and its lanes split that row's words, so the row is
    // streamed as coalesced 512-byte reads. One thread per entry read the same bytes with a 5 KiB
    // stride between neighbours, which cost the round 105 ms of the 265 it took.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    for (int v = warp; v < V; v += warps) {
        const uint4* w = reinterpret_cast<const uint4*>(HEAD + (size_t)v * H);
        float s = 0.0f;
        for (int c = lane; c < words; c += 32) {
            const uint4 a = xv[c], b = w[c];
            const __nv_bfloat16* ab = reinterpret_cast<const __nv_bfloat16*>(&a);
            const __nv_bfloat16* bb = reinterpret_cast<const __nv_bfloat16*>(&b);
#pragma unroll
            for (int e = 0; e < 8; e++) s = fmaf(bf(ab[e]), bf(bb[e]), s);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
        if (lane == 0) fxd2_insert(mine, topk, s, v);
    }
    // the row's candidates are split by warp, so the merge reads every warp's list, not one thread's
    if (lane == 0)
        for (int i = 0; i < topk; i++) all[warp * topk + i] = mine[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        Fxd2Cand out[FXD2_TOP];
        for (int i = 0; i < topk; i++) {
            out[i].v = -INFINITY;
            out[i].id = 0;
        }
        for (int i = 0; i < (blockDim.x >> 5) * topk; i++) fxd2_insert(out, topk, all[i].v, all[i].id);
        for (int i = 0; i < topk; i++) {
            VAL[(size_t)blockIdx.x * topk + i] = out[i].v;
            IDS[(size_t)blockIdx.x * topk + i] = (unsigned)out[i].id;
        }
    }
}

// The candidate selector (reference.py's CandidateSelector): h_proj = x @ WPROJ^T a step, then the
// lattice score of every (predecessor, successor) pair and the greedy path over it.
//
// One round is one block, 256 threads. The walk is serial in `t` — step `t` needs the *index* step
// `t - 1` picked, and its own `h_proj` row — so a block a position would only re-serialise the same
// walk while giving every step the wrong projection. What is not serial, the projection and the
// `topk` successors of one step, is spread over the block.
//
// A step reads its own row of each of the three tables the head wrote, and only its own row:
// `X + t * H` for the projection, `CAND + t * topk` for the candidates, `UNARY + t * topk` for their
// logits. Nothing here adds a block offset on top of the step: those tables are `[rows, *]` for the
// single walk, not `[blocks, rows, *]` (cuda_draft2_fwd.zig's `place` books `cand`, `unary` as
// `rows * topk` and `hn` as `rows * hidden`). Doing both is what made a 2-byte read land 511 GB
// past a codebook: `CAND + blockIdx * topk + t * topk + lane` walks off the ids and into the fp32
// `unary` table, and a logit's bit pattern (0x3B95A60B, 0.00457) as a codebook row is ~1e9 * 512 B.
//
// **Every codebook index is checked against the vocabulary before it is read.** Both codebooks are
// `[vocab, rank]` bf16, so an index at or past `vocab` is a layout fault on this side of the ABI,
// never a rounding difference: the kernel writes FXD2_BAD into the round's proposals and one record
// into ERR, and the host refuses the round with `error.SelectorIndexOutOfVocab`, naming the index,
// the step and the codebook. The check is one compare a candidate and one a step, and neither can
// change a byte of a round that is in vocabulary.
extern "C" __global__ void __launch_bounds__(256) tf_fxd2_select(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ WPROJ,
                                                                const __nv_bfloat16* __restrict__ PRED, const __nv_bfloat16* __restrict__ SUCC,
                                                                const unsigned* __restrict__ CAND, const float* __restrict__ UNARY,
                                                                unsigned* __restrict__ OUT, unsigned* __restrict__ ERR,
                                                                int H, int rank, int topk, int rows, int vocab, int anchor) {
    __shared__ float hp[FXD2_RANK];
    __shared__ float ctx[FXD2_RANK];
    __shared__ float score[FXD2_TOP];
    __shared__ unsigned bad;  // the lowest candidate of this step that is past the vocabulary
    const int tid = threadIdx.x;
    const unsigned vocab_u = (unsigned)vocab;
    const int words = H / 8;
    int prev = 0;  // the reference starts at index 0, which at t = 0 is the anchor's own row
    for (int t = 0; t < rows; t++) {
        // this step's predecessor token: the anchor at t = 0, else the candidate the previous step
        // picked. `prev` is an index into the step's own candidate list, so this read is in bounds
        // whenever `prev < topk` and `t <= rows`.
        const unsigned prev_token = t == 0 ? (unsigned)anchor : CAND[(t - 1) * topk + prev];
        if (prev_token >= vocab_u) {
            fxd2_fault(OUT, ERR, FXD2_KIND_PREDECESSOR, t, prev_token, rows);
            return;
        }
        if (tid == 0) bad = FXD2_NONE;
        __syncthreads();  // the slot is clear, and the previous step's reads of hp/ctx/score are done
        unsigned cid = 0u;  // lane i's candidate id: slot i of this step's top-k
        if (tid < topk) {
            cid = CAND[t * topk + tid];
            if (cid >= vocab_u) atomicMin(&bad, (unsigned)tid);
        }
        // the projection, while the candidate guard resolves: h_proj[t] = x[t] @ WPROJ^T
        const uint4* xv = reinterpret_cast<const uint4*>(X + (size_t)t * H);
        for (int j = tid; j < rank; j += blockDim.x) {
            const uint4* w = reinterpret_cast<const uint4*>(WPROJ + (size_t)j * H);
            float s = 0.0f;
            for (int c = 0; c < words; c++) {
                const uint4 a = xv[c], b = w[c];
                const __nv_bfloat16* ab = reinterpret_cast<const __nv_bfloat16*>(&a);
                const __nv_bfloat16* bb = reinterpret_cast<const __nv_bfloat16*>(&b);
#pragma unroll
                for (int e = 0; e < 8; e++) s = fmaf(bf(ab[e]), bf(bb[e]), s);
            }
            hp[j] = s;
        }
        __syncthreads();  // hp is whole, and every candidate has been checked: both are uniform here
        if (bad != FXD2_NONE) {
            fxd2_fault(OUT, ERR, FXD2_KIND_SUCCESSOR, t, CAND[t * topk + bad], rows);
            return;
        }
        const __nv_bfloat16* p = PRED + (size_t)prev_token * rank;
        for (int r = tid; r < rank; r += blockDim.x) ctx[r] = bf(p[r]) * hp[r];
        __syncthreads();
        if (tid < topk) {
            const __nv_bfloat16* sc = SUCC + (size_t)cid * rank;
            float s = 0.0f;
            for (int r = 0; r < rank; r++) s = fmaf(ctx[r], bf(sc[r]), s);
            score[tid] = UNARY[t * topk + tid] + s;
        }
        __syncthreads();
        int best = 0;
        for (int c = 1; c < topk; c++)
            if (score[c] > score[best]) best = c;
        prev = best;
        if (tid == 0) OUT[t] = CAND[t * topk + best];
    }
}

// act = silu(gate) * up over `n` elements: the MLP's SwiGLU, bf16 in and out like the rest of a round.
extern "C" __global__ void tf_fxd2_silu_mul(const __nv_bfloat16* __restrict__ GATE, const __nv_bfloat16* __restrict__ UP,
                                            __nv_bfloat16* __restrict__ OUT, int n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n) return;
    const float g = bf(GATE[i]);
    OUT[i] = __float2bfloat16_rn(g * sigmoid(g) * bf(UP[i]));
}
