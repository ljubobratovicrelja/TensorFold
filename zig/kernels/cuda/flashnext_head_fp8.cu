// The target model's quantised head (`TF_FLASHNEXT_LMHEAD_FP8=<dir>`): rank a row's 248,320
// candidates through a row-scaled FP8 `lm_head`, then rescore the survivors with the exact bf16
// rows the engine already keeps.
//
// `lm_head` is 248,320 rows of 2,560 bf16 halves — 1.27 GB read once a decode round, and the round's
// one stage that sits at the box's memory wall (5.29 ms at 240 GB/s). A weight-only FP8 copy is
// 0.64 GB of e4m3 codes with one f32 scale an output row, so the stream halves and the tensor-core
// MACs do not change. `TF_FLASHNEXT_LMHEAD_FP8`'s rank is not exact, so it is only ever a *candidate*
// source: the six entries below nominate 64 ids a row from the FP8 table, then score those 64
// against the exact bf16 rows and write the exact values back into the row's fp32 logits before the
// row is rounded to bf16 for the sampler.
//
// In launch order:
//
//   1. `Lane.matmulF32` (qmmf's `fp8g`, not in this file): [rows, K] bf16 activations x the FP8 table
//      into [rows, V] fp32 logits. The scale tile it reads is `fp8.zig`'s
//      `[npad/64][K/64][64]` fp32 layout, filled with the row's own scale repeated over the 40
//      input groups — so the per-row scale is applied exactly once per 64-input group and the
//      sum is `scale[row] * dot(code[row], x)`.
//   2. tf_fxlh_rank_tile(LG, PART, V, kcov, rows, tile)       grid (tiles, rows), 256 threads
//        a CTA's `kcov` best {logit, id} out of its `tile`-wide slice of one row's fp32 logits
//   3. tf_fxlh_rank_merge(PART, IDS, tiles, rows, kcov, slots) grid (rows), 256 threads
//        the row's `tiles * kcov` candidates down to its `kcov` best ids, descending
//   4. tf_fxlh_gather(TILED, IDS, G, rows, kcov, K)            grid (rows * kcov), 128 threads
//        the exact bf16 rows of those ids, out of the engine's *tiled* head (qmmf's
//        `packBf16` order), into a row-major [rows * kcov, K] block: 320 KB an id
//   5. tf_fxlh_rescore(G, X, RESC, n, kcov, K)                 grid (n / 128), 128 threads
//        one thread an (id, row): the exact bf16 dot, ascending K, one fp32 accumulator
//   6. tf_fxlh_patch(LG, IDS, RESC, V, kcov, rows)             grid (rows), 128 threads
//        the exact logits written over the FP8 ones, then
//      tf_fxlh_cast(LG, OUT, n, V)                              grid (n * V / 256), 256 threads
//        the row rounded to bf16, which is what every arm of the head writes.
//
// The two selects are a bitonic sort in dynamic shared (`draft2_head_fp8.cu`'s argument for the same
// shape): at kcov = 64 a serial merge is four times the list it produces, one lane walking 1,984
// candidates measured 1.39 ms at 5 GB/s, while a sort is one thread an element and 91 barriers.
//
// The key is a packed `{value, id}` u64 that sorts *ascending*, so an ascending bitonic sort is a
// descending sort of logits with ties broken by the smaller id. NaN is ranked last.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

/// The logits' f32 bits as an order-preserving u32: larger logit, larger u32. `-inf` maps to 0 and
/// `+inf` to the top, so a sort of these is a sort of the values with no special case.
__device__ __forceinline__ unsigned ordered(float x) {
    const unsigned u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// A rank key: smaller is better, so an ascending sort of keys is a descending sort of logits.
/// `id` rides in the low 32 bits to break ties, and a NaN — which `ordered` would sort as the best —
/// is sent last instead.
__device__ __forceinline__ unsigned long long rank_key(float v, unsigned id) {
    if (v != v) return 0xFFFFFFFFFFFFFFFFull;
    return ((unsigned long long)(0xFFFFFFFFu - ordered(v)) << 32) | (unsigned long long)id;
}

/// Bitonic sort of `v[0, n)`, ascending, over a block of `threads`. `n` must be a power of two and
/// uniform across the block (`tile` and `slots` both are); `threads` need not divide it.
__device__ __forceinline__ void bitonic(unsigned long long* v, int n, int tid, int threads) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n / 2; i += threads) {
                const int a = (i / j) * 2 * j + (i % j);
                const int b = a + j;
                const bool up = (a & k) == 0;
                const unsigned long long x = v[a];
                const unsigned long long y = v[b];
                if ((x > y) == up) {
                    v[a] = y;
                    v[b] = x;
                }
            }
            __syncthreads();
        }
    }
}

} // namespace

extern "C" __global__ void __launch_bounds__(256) tf_fxlh_rank_tile(
        const float* __restrict__ LG,
        unsigned long long* __restrict__ PART,
        int V,
        int kcov,
        int rows,
        int tile) {
    extern __shared__ unsigned long long sh[];
    const int row = blockIdx.y;
    const int first = blockIdx.x * tile;
    const int last = min(V, first + tile);
    const float* x = LG + (size_t)row * (size_t)V;
    // the CTA's slice as rank keys; past the end of the vocabulary (the last tile) a key sorts last
    for (int i = threadIdx.x; i < tile; i += blockDim.x) {
        const int id = first + i;
        sh[i] = (id < last) ? rank_key(x[id], (unsigned)id) : 0xFFFFFFFFFFFFFFFFull;
    }
    __syncthreads();
    bitonic(sh, tile, threadIdx.x, blockDim.x);
    unsigned long long* dst = PART + ((size_t)blockIdx.x * (size_t)rows + (size_t)row) * (size_t)kcov;
    for (int i = threadIdx.x; i < kcov; i += blockDim.x) dst[i] = sh[i];
}

extern "C" __global__ void __launch_bounds__(256) tf_fxlh_rank_merge(
        const unsigned long long* __restrict__ PART,
        unsigned* __restrict__ IDS,
        int tiles,
        int rows,
        int kcov,
        int slots) {
    extern __shared__ unsigned long long sh[];
    const int row = blockIdx.x;
    const int n = tiles * kcov;
    // the row's per-tile lists as one array of keys, padded to `slots` with keys that sort last
    for (int i = threadIdx.x; i < slots; i += blockDim.x) {
        sh[i] = (i < n) ? PART[((size_t)(i / kcov) * (size_t)rows + (size_t)row) * (size_t)kcov + (size_t)(i % kcov)]
                        : 0xFFFFFFFFFFFFFFFFull;
    }
    __syncthreads();
    bitonic(sh, slots, threadIdx.x, blockDim.x);
    unsigned* out = IDS + (size_t)row * (size_t)kcov;
    for (int i = threadIdx.x; i < kcov; i += blockDim.x) out[i] = (unsigned)(sh[i] & 0xFFFFFFFFu);
}

// The engine's bf16 head is qmmf's tiled layout, not rows (`qmmf.zig:packBf16`), so a survivor's
// row is gathered out of it: 320 KB an id, from the 1.27 GB the engine already holds resident —
// no second copy of the table, and no bf16 stream a round (the whole point is that only the 0.64 GB
// FP8 table is read at width).
//
// `packBf16` writes element `i` of the packed buffer as
//
//   tg = i / 4096, jj = (i / 512) % 8, kt = (i / 128) % 4, lane = (i / 4) % 32, e = i % 4
//   row = (tg / kg) * 64 + jj * 8 + lane / 4          kg = K / 64
//   col = (tg % kg) * 64 + kt * 16 + 2 * (lane % 4) + (e & 1) + 8 * (e >> 1)
//
// so its inverse, for a vocabulary id `v` and one of its columns, is the closed form below:
// jj and `lane / 4` come from `v % 64`, the column's 16 {kt, lane % 4, e} slots from `col % 64`,
// and `tg` from the id's 64-row block and the column's 64-column group. `tiledOffset` in
// `cuda_head_fp8.zig` is the same formula and is pinned against `packBf16` by a unit test there.
__device__ __forceinline__ size_t tiled_offset(int v, int col, int K) {
    const int kg = K / 64;
    const int kb = v / 64;
    const int rq = v % 64;
    const int jj = rq / 8;
    const int lq = rq % 8;
    const int g = col / 64;
    const int within = col % 64;
    const int kt = within / 16;
    const int rest = within % 16;
    const int c = rest >> 3;
    const int m = (rest >> 1) & 3;
    const int b = rest & 1;
    const int e = 2 * c + b;
    const int lane = 4 * lq + m;
    const int tg = kb * kg + g;
    return (size_t)tg * 4096 + (size_t)jj * 512 + (size_t)kt * 128 + (size_t)lane * 4 + (size_t)e;
}

extern "C" __global__ void __launch_bounds__(128) tf_fxlh_gather(
        const unsigned short* __restrict__ TILED,
        const unsigned* __restrict__ IDS,
        __nv_bfloat16* __restrict__ G,
        int rows,
        int kcov,
        int K) {
    const int i = blockIdx.x; // row * kcov + candidate
    if (i >= rows * kcov) return;
    const unsigned id = IDS[i];
    __nv_bfloat16* dst = G + (size_t)i * (size_t)K;
    for (int col = threadIdx.x; col < K; col += blockDim.x) {
        const unsigned short bits = TILED[tiled_offset((int)id, col, K)];
        dst[col] = __ushort_as_bfloat16(bits);
    }
}

// The rescore: one thread an (id, row), the exact bf16 dot `bf16(g) * bf16(x)` accumulated in fp32
// with a single ascending-K fma chain — the same operands in the same order a row-at-a-time fp32
// reference takes, so two ids' scores are comparable to the last bit under the same rounding.
extern "C" __global__ void __launch_bounds__(128) tf_fxlh_rescore(
        const __nv_bfloat16* __restrict__ G,
        const __nv_bfloat16* __restrict__ X,
        float* __restrict__ RESC,
        int n,
        int kcov,
        int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int row = i / kcov;
    const __nv_bfloat16* g = G + (size_t)i * (size_t)K;
    const __nv_bfloat16* x = X + (size_t)row * (size_t)K;
    float s = 0.0f;
    for (int k = 0; k < K; ++k) s = fmaf(__bfloat162float(x[k]), __bfloat162float(g[k]), s);
    RESC[i] = s;
}

extern "C" __global__ void __launch_bounds__(128) tf_fxlh_patch(
        float* __restrict__ LG,
        const unsigned* __restrict__ IDS,
        const float* __restrict__ RESC,
        int V,
        int kcov,
        int rows) {
    const int row = blockIdx.x;
    if (row >= rows or threadIdx.x >= kcov) return;
    const int j = threadIdx.x;
    LG[(size_t)row * (size_t)V + (size_t)IDS[(size_t)row * (size_t)kcov + (size_t)j]] =
        RESC[(size_t)row * (size_t)kcov + (size_t)j];
}

extern "C" __global__ void __launch_bounds__(256) tf_fxlh_cast(
        const float* __restrict__ LG,
        __nv_bfloat16* __restrict__ OUT,
        long long n,
        int V) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) OUT[i] = __float2bfloat16_rn(LG[i]);
}
