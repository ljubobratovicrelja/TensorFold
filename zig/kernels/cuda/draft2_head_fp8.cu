// DFlash2's quantised head (`TF_FLASHNEXT_HEAD_FP8=1`): rank a row's 248,320 candidates through a
// block-FP8 `lm_head`, then rescore the survivors with the exact bf16 rows the drafter already has.
//
// The head is the round's bandwidth stage, and it is the target's own bf16 `lm_head` — 248,320 rows
// of 2,560 halves, 1.27 GB read a round. A weight-only FP8 copy of the same table is 636 MB (plus a
// 39.7 MB scale tile, `zig/src/cuda/fp8.zig`'s 128x128 block layout), so ranking through it reads
// half the bytes for the same number of tensor-core MACs. That ranking is not exact — measured, its
// top-64 contains the bf16 head's top-16 for 1.0000 of 6,160 held-out rows, but the order inside the
// 64 is not the bf16 order — so it is only ever a *candidate* source: `tf_fxd2_rescore` then scores
// those 64 ids against the exact bf16 rows in fp32 and `tf_fxd2_rescore_top` takes the top 16 of the
// 448 exact logits. See `tools/dflash2/quantize/fit_head.py` and the ledger's section 11.
//
// Four entries, in the order the round launches them:
//
//   1. tf_fxd2_rank_tile(X, PART, V, kcov, rows, tile)   grid (tiles, rows), 256 threads
//        a CTA's `kcov` best {logit, id} out of its `tile`-wide slice of one row's fp32 logits
//   2. tf_fxd2_rank_merge(PART, IDS, tiles, rows, kcov, slots)   grid (rows), 256 threads
//        the row's `tiles * kcov` candidates down to its `kcov` best ids, descending
//   3. tf_fxd2_rescore(IDS, X, HEAD, OUT, n, kcov, K)   grid (n / 128), 128 threads
//        one thread an (id, row): the exact bf16 dot, ascending K, one fp32 accumulator
//   4. tf_fxd2_rescore_top(RESC, IDS, ID, VAL, kcov, topk)   grid (rows), 32 threads
//        the top `topk` of the row's `kcov` rescored logits
//
// Both selects are a bitonic sort in dynamic shared rather than the shipped path's serial merge
// (`tf_fxd2_topf32` + `tf_fxd2_headmerge`, one lane a CTA walking 512 candidates): at kcov = 64 the
// merge's list is four times the top-16 it produces, and one lane merging 1,984 candidates is what
// measured 1.39 ms at 5 GB/s — slower than the GEMM it sits behind. A sort is `n log^2 n` work but it
// is one thread an element and 91 barriers, and the whole pass is 0.1 ms of a 3.4 ms stage.
//
// The key is a packed `{value, id}` u64 that sorts *ascending*, so an ascending bitonic sort is a
// descending sort of logits with ties broken by the smaller id — the order `fxd2_insert` already
// gives a list (an equal logit never displaces the one already in it, and a list is built from
// ascending ids, so the earlier — smaller — id stays ahead). NaN is ranked last, as `fxd2_insert`'s
// `v > mine[i].v` also never selects one.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

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

/// `rank_key`'s value back out: the inverse of `ordered`, not `ordered` itself. Returning
/// `ordered(v)` would hand back the same bits with the sign bit flipped — i.e. exactly `-v` for
/// every positive logit, which made the selector walk the worst candidates in service.
__device__ __forceinline__ float key_value(unsigned long long key) {
    const unsigned o = 0xFFFFFFFFu - (unsigned)(key >> 32);
    return __uint_as_float((o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o);
}

/// Bitonic sort of `v[0, n)`, ascending, over a block of `threads`. `n` must be a power of two and
/// uniform across the block (it is the launch's dynamic shared size); `threads` need not divide it.
///
/// One thread a pair, `n/2` pairs a stage, `log2(n) * (log2(n)+1) / 2` stages, one barrier a stage:
/// at n = 8,192 that is 91 barriers and 4,096 pairs a stage, so a 256-thread block does 16 pairs a
/// stage each. The barrier sits outside the pair loop because that loop's trip count is not uniform.
__device__ __forceinline__ void bitonic(unsigned long long* v, int n, int tid, int threads) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n / 2; i += threads) {
                // the i-th pair of this stage: block `i / j` of 2j entries, offset `i % j` inside it
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

extern "C" __global__ void __launch_bounds__(256) tf_fxd2_rank_tile(
        const float* __restrict__ X,
        unsigned long long* __restrict__ PART,
        int V,
        int kcov,
        int rows,
        int tile) {
    extern __shared__ unsigned long long sh[];
    const int row = blockIdx.y;
    const int first = blockIdx.x * tile;
    const int last = min(V, first + tile);
    const float* x = X + (size_t)row * (size_t)V;
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

extern "C" __global__ void __launch_bounds__(256) tf_fxd2_rank_merge(
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

// The rescore: one thread an (id, row), the exact bf16 dot `bf16(x) * bf16(w)` accumulated in fp32
// with a single ascending-K fma chain — the same operands in the same order as a row-at-a-time fp32
// reference, so two ids' scores are comparable to the last bit under the same rounding, and the
// result does not depend on the launch geometry.
//
// It is NOT bit-identical to the tensor-core GEMM's logit for that id: glm_mm16_16 accumulates a
// 16-wide k step at a time inside one mma.sync's own reduction tree and then chains 160 of those in
// fp32, so the association of the 2,560 products differs (measured: <= 1 ulp, relative ~1e-7).
// Nothing downstream needs it to be — the rescore only reorders 64 candidates whose bf16 order the
// fp8 rank could not be trusted to keep, and the fidelity run's acceptance was bit-identical over
// 880 positions (ledger 11).
extern "C" __global__ void __launch_bounds__(128) tf_fxd2_rescore(
        const unsigned* __restrict__ IDS,
        const __nv_bfloat16* __restrict__ X,
        const __nv_bfloat16* __restrict__ HEAD,
        float* __restrict__ OUT,
        int n,
        int kcov,
        int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int row = i / kcov;
    const unsigned id = IDS[i];
    const __nv_bfloat16* x = X + (size_t)row * (size_t)K;
    const __nv_bfloat16* h = HEAD + (size_t)id * (size_t)K;
    float s = 0.0f;
    for (int k = 0; k < K; ++k) s = fmaf(__bfloat162float(x[k]), __bfloat162float(h[k]), s);
    OUT[i] = s;
}

extern "C" __global__ void __launch_bounds__(32) tf_fxd2_rescore_top(
        const float* __restrict__ RESC,
        const unsigned* __restrict__ IDS,
        unsigned* __restrict__ ID,
        float* __restrict__ VAL,
        int kcov,
        int topk) {
    extern __shared__ unsigned long long sh[];
    const int row = blockIdx.x;
    const float* r = RESC + (size_t)row * (size_t)kcov;
    const unsigned* ids = IDS + (size_t)row * (size_t)kcov;
    // `kcov` is a power of two and one warp sorts it alone: no barrier is even needed at 64/a warp,
    // but the sort is written to the same contract as the two selects above
    for (int i = threadIdx.x; i < kcov; i += blockDim.x) sh[i] = rank_key(r[i], ids[i]);
    __syncthreads();
    bitonic(sh, kcov, threadIdx.x, blockDim.x);
    for (int j = threadIdx.x; j < topk; j += blockDim.x) {
        const unsigned long long key = sh[j];
        VAL[(size_t)row * (size_t)topk + (size_t)j] = key_value(key);
        ID[(size_t)row * (size_t)topk + (size_t)j] = (unsigned)(key & 0xFFFFFFFFu);
    }
}
