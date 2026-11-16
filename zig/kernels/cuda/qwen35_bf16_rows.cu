// Qwen3.5 bf16-weight pieces: token rows of a bf16 table and small dense projections (GDN in_proj_a / in_proj_b).

#include <cuda_bf16.h>
#include <stdint.h>

// P: device counts as qwen35.cu's; rows at or past P[2] are padding and left untouched; may be null.
__device__ __forceinline__ bool q35b_past_rows(const int *P, int m) { return P != nullptr && m >= P[2]; }

// out[r] = table[ids[r]], D wide, a bit copy; grid: rows; 16-byte vectors when D % 8 == 0 and aligned.
extern "C" __global__ void __launch_bounds__(256)
    tf_q35_embed_bf16(const int *__restrict__ ids, const __nv_bfloat16 *__restrict__ table,
                      __nv_bfloat16 *__restrict__ out, int D, const int *__restrict__ P) {
    if (q35b_past_rows(P, blockIdx.x))
        return;
    const size_t id = (size_t)ids[blockIdx.x];
    const __nv_bfloat16 *src = table + id * (size_t)D;
    __nv_bfloat16 *dst = out + (size_t)blockIdx.x * D;
    if ((D & 7) == 0 && ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0) {
        for (int i = threadIdx.x; i < D / 8; i += blockDim.x)
            reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(src)[i];
    } else {
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            dst[i] = src[i];
    }
}

// y[r][j] = bf16(sum_k x[r][k] * W[j][k]) in a fixed order whatever the row count; grid (n, rows), k % 8 == 0, aligned.
extern "C" __global__ void __launch_bounds__(256)
    tf_q35_dense_bf16(const __nv_bfloat16 *__restrict__ X, const __nv_bfloat16 *__restrict__ W,
                      __nv_bfloat16 *__restrict__ Y, int n, int k) {
    __shared__ float red[8];
    const int j = blockIdx.x;
    const size_t r = blockIdx.y;
    const uint4 *x = reinterpret_cast<const uint4 *>(X + r * (size_t)k);
    const uint4 *w = reinterpret_cast<const uint4 *>(W + (size_t)j * k);
    float acc = 0.0f;
    for (int c = threadIdx.x; c < k / 8; c += 256) {
        const uint4 xv = x[c], wv = w[c];
        const __nv_bfloat16 *xb = reinterpret_cast<const __nv_bfloat16 *>(&xv);
        const __nv_bfloat16 *wb = reinterpret_cast<const __nv_bfloat16 *>(&wv);
#pragma unroll
        for (int e = 0; e < 8; e++)
            acc = __fmaf_rn(__bfloat162float(xb[e]), __bfloat162float(wb[e]), acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = red[0];
#pragma unroll
        for (int i = 1; i < 8; i++)
            s += red[i];
        Y[r * (size_t)n + j] = __float2bfloat16_rn(s);
    }
}

// ------------------------------------------------------------------------------------------------
// The block's projections in one launch: Y[r][j] for every row at once, grid (n, 1).
//
// tf_q35_dense_bf16 above pays a weight row per output element: its grid is (n, rows), so the 8-row
// draft block streams each k x 2-byte weight row 8 times, and a layer's 183 MB of projections cost a
// round 8 x that in DRAM traffic. Here one CTA owns output column j and keeps an accumulator per row,
// so the weight row streams once and the block's rows are read once per CTA.
//
// The sums are the per-element kernel's, digit for digit: a thread walks the same stride-256 sequence
// of k / 8 uint4 words over the same 256-thread block, each word's eight products go through
// __fmaf_rn in the same order, and the reduce is the same warp shuffle followed by thread 0 summing
// the eight warps' partials in warp order. Every (r, j) is therefore the bit tf_q35_dense_bf16 would
// have stored, which is what lets a round's downstream kernels read the same bytes as before.
// ------------------------------------------------------------------------------------------------

// There is no staging of the block in shared memory here, and that is a measurement rather than an
// omission: each thread owns its own word index, so a CTA reads each of the block's words exactly once
// whether they come from global or from shared, and staging only adds the copy — while the 40 KiB a
// served 8-row, 2560-wide block would take caps a CTA at two an SM. Timed at the served widths
// (sm_121), the staged form was 1.1-2x slower than this one at every one of them, and a 16-row launch
// with no staging beat an 8-row launch with it by 1.7x on the same weight row.
constexpr int tf_q35_batch_rows = 16;

// One row's word against the weight's word: eight products into `a`, in the per-element kernel's order.
__device__ __forceinline__ void tf_q35_batch_word(float &a, const uint4 xv, const uint4 wv) {
    const __nv_bfloat16 *xb = reinterpret_cast<const __nv_bfloat16 *>(&xv);
    const __nv_bfloat16 *wb = reinterpret_cast<const __nv_bfloat16 *>(&wv);
#pragma unroll
    for (int e = 0; e < 8; e++)
        a = __fmaf_rn(__bfloat162float(xb[e]), __bfloat162float(wb[e]), a);
}

// R rows at once, the weight row read once for all of them.
template <int R>
__device__ __forceinline__ void tf_q35_batch_body(const __nv_bfloat16 *__restrict__ X, const __nv_bfloat16 *__restrict__ W,
                                                  __nv_bfloat16 *__restrict__ Y, int n, int k, float *red) {
    const int j = blockIdx.x;
    const int cw = k >> 3;
    const uint4 *w = reinterpret_cast<const uint4 *>(W + (size_t)j * k);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++)
        acc[r] = 0.0f;
    for (int c = threadIdx.x; c < cw; c += 256) {
        const uint4 wv = w[c];
#pragma unroll
        for (int r = 0; r < R; r++)
            tf_q35_batch_word(acc[r], reinterpret_cast<const uint4 *>(X + (size_t)r * k)[c], wv);
    }
    // The per-element kernel's reduce, once a row: the warp shuffle, then thread 0's ascending sum of
    // the eight warps. Same tree, same order, so same bits.
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
        if ((threadIdx.x & 31) == 0)
            red[(threadIdx.x >> 5) * tf_q35_batch_rows + r] = acc[r];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            float s = red[r]; // warp 0's partial first, as tf_q35_dense_bf16 starts its sum
#pragma unroll
            for (int i = 1; i < 8; i++)
                s += red[i * tf_q35_batch_rows + r];
            Y[(size_t)r * n + j] = __float2bfloat16_rn(s);
        }
    }
}

// The block's projections: grid (n, 1), 256 threads, rows <= tf_q35_batch_rows, k % 8 == 0, aligned.
extern "C" __global__ void __launch_bounds__(256)
    tf_q35_dense_bf16_batch(const __nv_bfloat16 *__restrict__ X, const __nv_bfloat16 *__restrict__ W,
                            __nv_bfloat16 *__restrict__ Y, int n, int k, int rows) {
    __shared__ float red[8 * tf_q35_batch_rows];
    if (rows < 1 || rows > tf_q35_batch_rows || k <= 0)
        return;
    switch (rows) {
#define TF_Q35_BATCH_CASE(R) \
    case R:                  \
        tf_q35_batch_body<R>(X, W, Y, n, k, red); \
        break
        TF_Q35_BATCH_CASE(1);
        TF_Q35_BATCH_CASE(2);
        TF_Q35_BATCH_CASE(3);
        TF_Q35_BATCH_CASE(4);
        TF_Q35_BATCH_CASE(5);
        TF_Q35_BATCH_CASE(6);
        TF_Q35_BATCH_CASE(7);
        TF_Q35_BATCH_CASE(8);
        TF_Q35_BATCH_CASE(9);
        TF_Q35_BATCH_CASE(10);
        TF_Q35_BATCH_CASE(11);
        TF_Q35_BATCH_CASE(12);
        TF_Q35_BATCH_CASE(13);
        TF_Q35_BATCH_CASE(14);
        TF_Q35_BATCH_CASE(15);
        TF_Q35_BATCH_CASE(16);
#undef TF_Q35_BATCH_CASE
        default:
            break;
    }
}
