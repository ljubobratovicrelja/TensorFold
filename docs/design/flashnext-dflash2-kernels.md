# DFlash2 drafter on CUDA: kernel spec for flashnext

What to write, what to reuse, and where the numerics sit. Grounded in
`zig/kernels/cuda/qwen35_draft.cu` (read, not assumed) and the working torch port at
`tools/dflash2/reference.py` (which runs the real checkpoint end to end).

The conclusion up front: **most of this is already written.** qwen3_5's drafter kernels are
argument-driven, not baked to its own dimensions, and the drafter's `head_dim` is 128 in both
families — so `dyn_conv`, `norm_rope`, `ring_put` and `attn` are reusable as-is. The genuinely new
work is the tap fusion, the bidirectional block attention variant, and the candidate selector.

## 1. Reusable as-is

Measured against `qwen35_draft.cu`:

| kernel | signature | reusable? |
| --- | --- | --- |
| `tf_q35d_dyn_conv` | `(X, DYN, BASE, RES, OUT, D, group, branch)` | **yes, unchanged.** `D`, `group` and `branch` are the only shape inputs and all are ours (2560, 16, 0/1). The inner loop is `#pragma unroll` over 2 taps, which is exactly our `conv_kernel: 2`. |
| `tf_q35d_norm_rope` | `(X, W, OUT, heads, stride, pos0, eps, theta)` | **yes, unchanged** — `HD` is a compile-time 128 and ours is 128. `heads`/`stride` are arguments. |
| `tf_q35d_ring_put` | `(K, V, RK, RV, width, pos0, W)` | **yes** — `width` is an argument (`kv_heads * 128` = 1024 for us). |
| `tf_q35d_gather_taps` | `(RING, R, start, width, OUT)` | **yes** — `width` is our `n_taps * hidden` = 12,800. |
| `tf_q35d_attn` | `(Q, KB, VB, RK, RV, rows, heads, kv_heads, W, begin, pos0, window, OUT)` | **partly.** It already attends over *both* the ring and the `rows` block, and `rows` is a parameter — but it excludes the query's own block row and is built for a causal draft. DFlash2 needs bidirectional attention across all 8 block rows (`is_causal: false`), so this needs a variant (see §2.3), not a copy. |

## 2. What must be written

Filenames follow the family's pattern; these are new `.cu` entries, not a new translation unit.

### 2.1 Tap fusion: `tf_fxd2_fuse`
- **Job:** `ctx[pos, :] = rms_norm(hidden_norm, concat(tap_0..tap_4)[pos, :] @ fc^T)`.
- **Shape in:** `[n_rows, 12800]` bf16 → out `[n_rows, 2560]` bf16.
- **Geometry:** grid `(n_rows)`, block 256 threads. One block a row; the 2560 outputs are 10 per thread.
- **Arithmetic:** a plain 12800×2560 matmul is 32.7 M MACs a row — for a 6-row *context* prefix this
  is done once; for the 8 block rows it is per round. Accumulate fp32, then RMS over the 2560
  outputs in a second pass within the same block (needs a block-wide reduction: `shared float` +
  `__syncthreads`), then multiply by `hidden_norm`.
- **Why one kernel:** the norm needs the whole 2560-row, so a separate launch would either re-read
  the matmul output from global memory or force a two-kernel split for no benefit.
- **Difficulty:** low-medium (a matmul + a block reduction).

### 2.2 Layer step: `tf_fxd2_layer_qkv` + reuse
Per layer, in order (mirrors `reference.py`'s `DraftLayer.forward`, which is verified to run):

1. `xn = rms_norm(x, input_layernorm)` — small standalone kernel or fused into the next step.
2. **input-side conv**: `delta = kernel_projection @ xn` (640×2560 matmul) then `tf_q35d_dyn_conv(xn, delta, base, null, xn_conv, 2560, 16, 0)`.
3. `q = xn_conv @ q_proj^T` → `[rows, 32, 128]`, `k_n = xn_conv @ k_proj^T` → `[rows, 8, 128]`, `v_n = xn_conv @ v_proj^T`.
4. `k_ctx = ctx @ k_proj^T`, `v_ctx = ctx @ v_proj^T` (the fused tap rows — **this is the cross-attention context**).
5. `q_norm`/`k_norm` then RoPE via `tf_q35d_norm_rope` (positions = context length + block row).
6. `tf_q35d_attn` variant (§2.3) over `[k_ctx, k_n]` and `[v_ctx, v_n]`.
7. `o_proj`, then **output-side conv**: `finish` with the stored delta, `branch = 1`.
8. residual add; then the MLP with the same shape: `post_attention_layernorm` → conv prepare → `gate`/`up`/`silu` → `down` → conv finish → residual.

- **Difficulty:** medium — it is 5 repetitions of a familiar step, and 1,2,7,8 all reuse existing kernels.

### 2.3 Block attention variant: `tf_fxd2_attn`
- **Difference from `tf_q35d_attn`:** every block row attends to *every* other block row (all 8),
  not just the rows before it — `is_causal: false` in the checkpoint config. The context prefix
  stays fully visible.
- **Geometry:** grid `(rows, heads)`, one warp a (row, head) — same as the existing kernel, since a
  warp already handles `HD/32 = 4` elements a lane.
- **Arithmetic:** identical online-softmax loop, but the `for pos in [first, pos0)` ring loop and the
  `for r in 0..rows` block loop are both included for every query row, with no `r < m` restriction.
- **Difficulty:** low (a copy of the existing kernel with the block loop unguarded).

### 2.4 Final norm + head + top-k: `tf_fxd2_head_topk`
- `hidden = rms_norm(x, norm)`, then `logits = hidden @ lm_head^T` over the **full 248,320 vocab**
  (the drafter ships no head slice — it uses the target's), then top-16 per row.
- **Geometry:** a row's 248k logits need a block-level top-k: grid `(rows)`, block 256, each thread
  keeps a running top-16 of its slice in registers, then a shared-memory merge.
- **Numerics:** fp32 logits, matching `transform_unary_logits` (which is an identity up to the fp32
  upcast here, `output_multiplier` 1.0 and no softcap).
- **Difficulty:** medium (248k a row is the dominant cost of the whole drafter — see §5).

### 2.5 Candidate selector: `tf_fxd2_select`
Three steps, and they are cheap — worth knowing before optimising anything else:
1. `h_proj = hidden @ hidden_projection^T` → `[rows, 256]` (1.3 MB of weights).
2. **lattice:** for every position `t` and every (predecessor, successor) pair,
   `score[t, p, c] = unary[t, c] + sum_r ( pred[prev(t,p), r] * h_proj[t, r] * succ[cand[t,c], r] )`.
   With top-k 16 that is 8 × 16 × 16 pairs, each a 256-long dot — a `[16, 256]` gather times a
   `[16, 256]` gather, i.e. 65 k MACs a position.
3. **greedy path:** a serial walk, `t`-sequential, argmax over the 16 successors given the previous
   choice. The reference's `greedy_path` is a Python `for` over positions; on device it is 8 sequential
   argmaxes of 16 values a row — best done by one thread a row (grid `(rows)`, 1 thread), not a warp,
   because the dependency is strictly serial.

- **Difficulty:** low-medium. The codebooks are 255.6 MB but each step touches only 16+16 rows of 256.

## 3. Numerics

- Storage bf16 throughout, matching the checkpoint; accumulation fp32.
- `dyn_conv` rounds through bf16 at each tap (`rbf`), which is the reference's behaviour and is kept.
- The selector's lattice sums are fp32 (reference eigsum in the model's dtype, upcast in our port).
- **No softcap, no output multiplier**: `transform_unary_logits` is an identity here, so the head
  kernel must not apply one.
- Verification is lossless by construction: the target's own argmax decides. So a numerically
  slightly-off drafter costs acceptance length, never correctness — which is exactly why an
  acceptance-length check (§ tap-ring memo) is the real gate, not a token comparison.

## 4. Per-stream buffers at 4 streams, window 2048

| item | bytes a stream | at four |
| --- | ---: | ---: |
| drafter KV rings (5 layers × 2 × 2064 rows × 1024 × 2 B) | 42.3 MB | 0.16 GiB |
| tap ring (2064 × 12,800 × 2 B) | 52.8 MB | 0.20 GiB |
| head top-k / lattice scratch (8 × 16 × 16 × 4 B, plus 8 × 16 × 4) | ~8.5 KB | ~34 KB |

Weights: 1.21 GB bf16, of which 255.6 MB is the selector's codebooks (not quantisable).
`--drafter-bits 4` takes the backbone to ~238 MB, so ~0.49 GB total.

## 5. Judgement: where this will be slow on GB10

- **The head is the cost, not the backbone.** 248,320 logits a row × 8 rows = 2.0 M logits a round,
  against 5 layers × 2560 × (4096+2048+2×7680) ≈ 0.3 G MACs. The MTP head only ever scores its own
  draft vocab. If the drafter uses the full head every round it can plausibly cost *more* than the
  MTP chain it replaces — measure the head's share first, and consider a candidate-vocab slice
  (the fork already has `core.proposal_vocab` for exactly this in `cuda_mtp_load.zig`).
- **Two convs a layer with a 640×2560 projection each** is 6.5 MB of weights and 6.5 M MACs a layer —
  small, but it serialises against the sublayer it wraps.
- **`greedy_path` is serial by construction**; 8 dependent argmaxes a row caps it at ~8 dependent
  small steps a round. Fine at 4 lanes, but it does not parallelise across positions.
- Nothing here needs more than one launch a step, so the launch count a round is ~5 layers × 6 + 3 ≈ 33,
  against MTP's chain of 6-8 steps. **That is the number to watch**: if a round costs more launches
  than the MTP chain it replaces, the win has to come from acceptance length alone.
