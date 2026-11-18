// DFlash2's draft attention on tensor cores, split over the context: block_attention.cu's `Fold`, chunked.
//
// `tf_fxd2_attn_tile` (qwen35_draft.cu) is one CTA a KV head — 8 CTAs of 1024 threads on a 48-SM part,
// one position a step, a scalar `LDG.E.U16` dot and a `warp_sum` in the dependency chain at every
// position. Measured, 14 reps, medians on this box: the five attention blocks cost 3.737 ms of a
// 15.3 ms round, ~0.675 ms each, against a 0.221 ms bandwidth floor (5 x 8.00 MiB = 41.9 MB at
// 190 GB/s) — 11 GB/s, 16.9x above the floor, while every other stage of the round sits at the DRAM
// wall (mlp 197 GB/s, head GEMM 234 GB/s, qkv 151 GB/s). Each CTA already reads its KV head once, so
// the loss is parallelism (8 of 48 SMs, one CTA each, 2048 threads never fit) and the serial per-position
// softmax chain.
//
// This pair is the same attention cut the other way, exactly as `tf_q35_attn_partial8` + `attn_rows8` +
// `tf_q35_attn_merge` cut Qwen3.5's decode attention: the logical key space `[0, ctx + rows)` — the
// context's ring rows, then the block's own `rows` keys, which every block row sees — into `nch` chunks
// of `ck` keys, one CTA a (chunk, KV head). A CTA keeps its chunk's running (max, sum) and unnormalised
// fp32 output in `PM`/`PL`/`PO`; `tf_fxd2_attn_split_merge` rescales each chunk by `exp(m - max)`, sums
// them and divides by the summed weights once, the same merge `tf_q35_attn_merge` performs.
//
// At the served shape (ctx 2048, block 8, 8 KV heads) the launcher's plan cuts 2056 logical keys into
// 5 chunks of 416, so the attention launches 8 x 5 = 40 CTAs — one wave of the 48 SMs, against the
// tiled kernel's 8 — and FA3's `num_splits_heuristic` gives 5 splits for this shape too. Every key a
// CTA reads is 16-byte `cp.async` into a swizzled key tile, and every product is `mma.sync`
// m16n8k16 (bf16 in, fp32 accumulate): one warp a query head, its 16 mma rows the block's rows, 32
// keys a step, and both K/V tiles double-buffered so the next tile streams while this one is folded.
//
// NOT bit-identical to `tf_fxd2_attn_tile`, by construction: the chunked merge reassociates a row's
// softmax sum and its output sum (one running max a chunk instead of one for the whole context), and
// the products are tensor-core fp32 accumulates rather than the per-lane scalar `fmaf` chain. The bf16
// output rows therefore differ in their low bits, and a proposal can move. That is not a licence to
// guess: this repo's own record (docs/design/flashnext-dflash2-gb10-optimisation-plan.md §5) is that a
// 1.7e-7 relative-L2 shift in one fused row moved proposals on 7 of 16 reps while acceptance held, and
// that "acceptance length per slot over many prompts is the only honest gate for a precision change" —
// proposal identity is the gate for a refactor, not for this. The tiled and legacy variants stay
// bit-identical to each other and to themselves, and remain one unset environment variable away.
//
// The mask is DFlash2's, not the reference's causal one: block row `r` sees the context's logical keys
// `[max(r, ctx + r - window + 1), ctx)` and then every block key `[ctx, ctx + rows)`. See
// `tf_fxd2_attn_tile`'s body for the measurement that pinned that asymmetric window trim.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "common/cp_async.cuh"
#include "common/qmm_frag.cuh"

namespace tf_fxd2_split {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::ldmatrix4t;
using qmm_frag::mma;
using qmm_frag::mma0;

constexpr int D = 128, C = D / 8, K16 = D / 16, NB = D / 8, KT = 32;
constexpr float LOG2E = 1.4426950408889634f;

// The chunks a row's context is cut into — the stride of PM/PL/PO and the launcher's
// `attn_split_max` in cuda_draft2_fwd.zig, which a host test pins this line's text against. A round
// whose plan wants more than this is a plan bug, not a shape: 8 chunks x 8 KV heads is already 64
// CTAs, past the 48 SMs one wave covers.
constexpr int NCH = 8;

using tf_common::cp_wait;

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// A 16-byte chunk `c` of a tile row `r`, swizzled so ldmatrix's eight rows fall on distinct banks:
// the layout block_attention.cu's `tf_block_attention` writes and reads, and `swz` in
// qwen35_attn_tc.cuh (same permutation, this file's spelling).
__device__ __forceinline__ int sw(int r, int c) { return r * C + (c ^ (r & 7)); }

// One warp's fold over a 32-key tile of one query head's 16 rows: scores masked by `vis(row, key)`,
// online softmax, then P x V. `tf_block_attention`'s Fold without its NaN/infinite re-run (this path
// has no `attn_odd` fallback: `tf_q35_attn_partial8` has none either, and a round's keys are the
// drafter's own projections).
struct Fold {
    float o[NB][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    __device__ __forceinline__ Fold() {
#pragma unroll
        for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    }

    template <typename Vis>
    __device__ __forceinline__ void tile(const uint32_t (&qa)[K16][4], const uint4* ks, const uint4* vs, float scale,
                                         int lane, int k0, Vis vis) {
        const int g8 = lane / 4, qd = lane % 4, mt = lane / 8;
        float sc[4][4];
#pragma unroll
        for (int k = 0; k < K16; ++k) {
#pragma unroll
            for (int n = 0; n < 4; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, ks + sw(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(sc[n], qa[k], b[0], b[1]);
                    mma0(sc[n + 1], qa[k], b[2], b[3]);
                } else {
                    mma(sc[n], qa[k], b[0], b[1]);
                    mma(sc[n + 1], qa[k], b[2], b[3]);
                }
            }
        }
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int c = k0 + 8 * n + 2 * qd;
            sc[n][0] = vis(g8, c) ? __fmul_rn(sc[n][0], scale) : -INFINITY;
            sc[n][1] = vis(g8, c + 1) ? __fmul_rn(sc[n][1], scale) : -INFINITY;
            sc[n][2] = vis(g8 + 8, c) ? __fmul_rn(sc[n][2], scale) : -INFINITY;
            sc[n][3] = vis(g8 + 8, c + 1) ? __fmul_rn(sc[n][3], scale) : -INFINITY;
            tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
            tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
        }
        for (int x = 1; x <= 2; x *= 2) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
        }
        const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
        const float a0 = tm0 == -INFINITY ? 1.f : (m0 == -INFINITY ? 0.f : texp(__fsub_rn(m0, n0)));
        const float a1 = tm1 == -INFINITY ? 1.f : (m1 == -INFINITY ? 0.f : texp(__fsub_rn(m1, n1)));
        float c0 = 0.f, c1 = 0.f;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            sc[n][0] = sc[n][0] != -INFINITY ? texp(__fsub_rn(sc[n][0], n0)) : 0.f;
            sc[n][1] = sc[n][1] != -INFINITY ? texp(__fsub_rn(sc[n][1], n0)) : 0.f;
            sc[n][2] = sc[n][2] != -INFINITY ? texp(__fsub_rn(sc[n][2], n1)) : 0.f;
            sc[n][3] = sc[n][3] != -INFINITY ? texp(__fsub_rn(sc[n][3], n1)) : 0.f;
            c0 = __fadd_rn(c0, __fadd_rn(sc[n][0], sc[n][1]));
            c1 = __fadd_rn(c1, __fadd_rn(sc[n][2], sc[n][3]));
        }
        for (int x = 1; x <= 2; x *= 2) {
            c0 = __fadd_rn(c0, __shfl_xor_sync(0xffffffffu, c0, x));
            c1 = __fadd_rn(c1, __shfl_xor_sync(0xffffffffu, c1, x));
        }
        l0 = __fmaf_rn(l0, a0, c0);
        l1 = __fmaf_rn(l1, a1, c1);
        m0 = n0;
        m1 = n1;
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            o[i][0] = __fmul_rn(o[i][0], a0);
            o[i][1] = __fmul_rn(o[i][1], a0);
            o[i][2] = __fmul_rn(o[i][2], a1);
            o[i][3] = __fmul_rn(o[i][3], a1);
        }
        uint32_t pa[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
            pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
            pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
            pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int i = 0; i < NB; i += 2) {
                uint32_t b[4];
                ldmatrix4t(b, vs + sw(16 * j + 8 * (mt % 2) + lane % 8, i + mt / 2));
                mma(o[i], pa[j], b[0], b[1]);
                mma(o[i + 1], pa[j], b[2], b[3]);
            }
    }
};

// One chunk (blockIdx.x) of one KV head (blockIdx.y): a warp a query head, 32 keys a step over the
// chunk's `[k0, k1)` of the logical key space, its (max, sum) and unnormalised output left as a
// partial. PM, PL: NCH fp32 a line; PO: NCH x HD fp32 a line, all indexed `line * NCH + chunk` with
// `line = row * heads + head` — the served output's own row order, so the merge's write needs no
// transpose and `attn_rows8`'s (row, head) slot rule is the same one.
//
// Grid (chunks, kv_heads), 32 * heads / kv_heads threads (the KV group's query heads a warp each,
// so 128 at the served shape). Dynamic shared: two double-buffered K and V tiles of KT x C uint4.
extern "C" __global__ void __launch_bounds__(128) tf_fxd2_attn_split(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KB, const __nv_bfloat16* __restrict__ VB,
    const __nv_bfloat16* __restrict__ RK, const __nv_bfloat16* __restrict__ RV, int rows, int heads, int kv_heads, int W,
    long long begin, long long pos0, int window, int ctx, int ck, float* __restrict__ PM, float* __restrict__ PL,
    float* __restrict__ PO) {
    extern __shared__ uint4 smem[];
    uint4* kbuf = smem;                // stage 0's key tile, then stage 1's
    uint4* vbuf = smem + 2 * KT * C;   // stage 0's value tile, then stage 1's
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int G = heads / kv_heads;
    const int kvh = blockIdx.y, c = blockIdx.x;
    const int head = kvh * G + warp;
    const int kvw = kv_heads * D;
    const float scale = rsqrtf((float)D);
    // the context's ring rows, then the block's own keys: every block row sees the last `rows` of them
    const int ktot = ctx + rows;
    const int k0 = c * ck, k1 = min(k0 + ck, ktot);

    uint32_t qa[K16][4];
    {
        const int r0 = lane / 4, r1 = r0 + 8;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(Q + ((size_t)r0 * heads + head) * D);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(Q + ((size_t)r1 * heads + head) * D);
        const int qd = lane % 4;
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = r0 < rows ? q0[8 * k + qd] : 0u;
            qa[k][1] = r1 < rows ? q1[8 * k + qd] : 0u;
            qa[k][2] = r0 < rows ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = r1 < rows ? q1[8 * k + qd + 4] : 0u;
        }
    }

    // Logical key t's key and value rows: the context's ring slot for t < ctx, else block key t - ctx.
    const auto ksrc = [&](int t) {
        return t < ctx ? RK + (size_t)((begin + t) % W) * kvw + kvh * D
                       : KB + ((size_t)(t - ctx) * kv_heads + kvh) * D;
    };
    const auto vsrc = [&](int t) {
        return t < ctx ? RV + (size_t)((begin + t) % W) * kvw + kvh * D
                       : VB + ((size_t)(t - ctx) * kv_heads + kvh) * D;
    };
    // A tile of the logical keys from t0, its rows past k1 zero-filled (the mask drops them).
    const auto load = [&](int t0, int s) {
        uint4* ks = kbuf + s * KT * C;
        uint4* vs = vbuf + s * KT * C;
        for (int i = tid; i < KT * C; i += blockDim.x) {
            const int key = t0 + i / C, col = i % C;
            const bool in = key < k1;
            const int t = in ? key : k1 - 1;
            cp16z(ks + sw(i / C, col), ksrc(t) + col * 8, in);
            cp16z(vs + sw(i / C, col), vsrc(t) + col * 8, in);
        }
        commit();
    };

    // Row r's first visible context key: `tf_fxd2_attn_tile`'s `first = max(begin + r, pos0 + r - window + 1)`
    // in logical coordinates. The block's own keys are visible to every row (is_causal is false), and a
    // key past the chunk is dropped whether or not its tile row was zero-filled.
    const long long shift = pos0 - begin + 1 - window;
    const auto vis = [&](int r, int key) {
        const long long first = shift + r > (long long)r ? shift + r : (long long)r;
        return r < rows && key < k1 && (key >= ctx || (long long)key >= first);
    };

    Fold f;
    if (k0 < k1) {
        load(k0, 0);
        int nt = 0;
        for (int t0 = k0; t0 < k1; t0 += KT, ++nt) {
            const int st = nt & 1;
            // the next tile streams while this one is folded; the stage it lands in was released by the
            // barrier two iterations back
            if (t0 + KT < k1) {
                load(t0 + KT, st ^ 1);
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncthreads();
            f.tile(qa, kbuf + st * KT * C, vbuf + st * KT * C, scale, lane, t0, vis);
            __syncthreads();
        }
    }

    const int g8 = lane / 4, qd = lane % 4;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int line = g8 + 8 * half;
        if (line >= rows) continue;
        const size_t slot = (size_t)(line * heads + head) * NCH + c;
        // the max and the sum are warp-uniform, so one lane a row writes them; the output row is the
        // lane's own column pair, so all four write
        if (qd == 0) {
            PM[slot] = half ? f.m1 : f.m0;
            PL[slot] = half ? f.l1 : f.l0;
        }
#pragma unroll
        for (int i = 0; i < NB; ++i)
            *reinterpret_cast<float2*>(PO + slot * D + 8 * i + 2 * qd) =
                half ? make_float2(f.o[i][2], f.o[i][3]) : make_float2(f.o[i][0], f.o[i][1]);
    }
}

// A line's chunk partials into its bf16 output row: `tf_q35_attn_merge`'s arithmetic — the largest
// chunk max, then each chunk's output and weight-sum rescaled by `exp(m - max)` and accumulated, then
// one division, where the unsplit kernel would have divided once too. A warp a line, four a CTA.
// Grid (rows * heads / 4), 128 threads.
extern "C" __global__ void __launch_bounds__(128) tf_fxd2_attn_split_merge(
    const float* __restrict__ PM, const float* __restrict__ PL, const float* __restrict__ PO,
    __nv_bfloat16* __restrict__ OUT, int rows, int heads, int nch) {
    constexpr int DPL = D / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int line = blockIdx.x * 4 + warp;
    if (line >= rows * heads) return;
    const size_t base = (size_t)line * NCH;
    float mx = -INFINITY;
    for (int c = 0; c < nch; ++c)
        if (PL[base + c] > 0.0f) mx = fmaxf(mx, PM[base + c]);
    float ls = 0.0f, acc[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = 0.0f;
    for (int c = 0; c < nch; ++c) {
        if (!(PL[base + c] > 0.0f)) continue;
        const float e = __expf(PM[base + c] - mx);
        ls = fmaf(PL[base + c], e, ls);
        const float* o = PO + (base + c) * D + lane * DPL;
#pragma unroll
        for (int i = 0; i < DPL; ++i) acc[i] = fmaf(o[i], e, acc[i]);
    }
    __nv_bfloat16* out = OUT + (size_t)line * D + lane * DPL;
#pragma unroll
    for (int i = 0; i < DPL; ++i) out[i] = __float2bfloat16_rn(acc[i] / ls);
}

}  // namespace tf_fxd2_split
