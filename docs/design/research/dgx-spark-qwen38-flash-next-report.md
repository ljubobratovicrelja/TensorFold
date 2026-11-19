# Qwen3.8-Flash-Next on DGX Spark: what people actually run, and what they turn

Compiled 2026-11-19. Every number below is quoted from a public source and tagged with its harness.
**Numbers across sources are not comparable**: single-stream decode on this model moves ±15 % between
runs on the same box, and 2–3× between prompt types (prose vs code vs "count to 100"), so a figure
without its prompt set, sampling and thinking setting is marketing. Where a source is self-critical
about its own numbers, that is noted.

---

## 0. The one structural fact everything follows from

The checkpoint is ~124–126 GiB: 78 GiB of routed experts + dense weights, plus a **47.7 GiB FP8 n-gram
(PLE) table** (16 heads × 20 M rows × 160 dims; 2,560 useful bytes/token). One GB10 has 128 GB unified,
~10 GiB of which the OS/driver/Docker hold. So **the model does not fit unless the PLE table leaves
memory** ([SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)):

- `VLLM_PLE_CPU_OFFLOAD=1` / `--ple-offload-embedding` is **useless on GB10** — pinned host RAM comes out
  of the same physical pool as "VRAM". This is the single most-repeated trap.
- The table is a lookup, not compute: 16 rows × 160 B per token. Serving it from NVMe by `mmap`/`pread`
  costs ~76–80 GiB resident and leaves 14–29 GiB for KV.

Per-token cost of the PLE, measured: a 160-byte fetch touches a 4 KiB page → **~26× page amplification**
and 16 major faults/token at c=1, falling to **3.6 at c=48** because batched tokens share n-gram rows
(jschmied, [what-generalises](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/what-generalises.md)).
Hence "quantise the table" buys *resident capacity*, not speed.

Roofline on this box: **~220 GB/s effective** (of ~273 GB/s spec). jschmied measured 66.1 ms per verify
cycle on code against a 53–54 ms byte floor → **1.2× off the floor**; dolf3131's nsys round shows large
kernels at **77–91 % of sustained bandwidth with "no single waste"**. Decode is bandwidth-bound and
mostly done; the remaining wins are structural (speculation, state dtype, I/O placement), not kernel tweaks.

---

## 1. Sources, configurations, and claims

### 1.1 MiaAI-Lab — single Spark, TensorFold (Current recommended recipe)
<https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold>

- **Stack:** TensorFold's Zig engine `tensorfold-native` (v1.1.4 `eab85828` + 17 MIT patches) on
  `nvcr.io/nvidia/pytorch:26.07-py3`. Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local` @ `b9ce18a6`
  (QAD FP4: NVFP4 group-16 experts, MXFP8 attention/shared, FP8 n-gram table, 512 experts / 10 active).
- **Config:** 262,144 ctx, `PARALLEL=5`, MTP drafts (≤6/round, confidence −0.4 chain), `KV_DTYPE=fp8`
  (5 full windows fit; stock v1.1.4 fits 3), vision on, 9 GiB engine reserve + 2 GiB vision helper,
  n-gram table (26.8 GiB) read from disk via `pread` thread pool with next-chunk read-ahead.
- **Measured** (sparkDash on the same box, no network; prose T=0.7 thinking on, code T=0 thinking off):
  - Decode prose: **55.6 tok/s at c1**, 95.4 / 145.1 / 156.1 aggregate at 2/4/5.
  - Decode code: **110.5 tok/s at c1**, up to 217.6 aggregate at 5.
  - Cold prefill: **3,050 tok/s @4k → 2,269 @131k**; TTFT 1.52 s @4.1k, 57.9 s @131k.
  - KV: 4.78 GiB per full 262k fp8 window; pool 27–29 GiB ≈ 1.4–1.6 M fp8 tokens.
- **Own caveats:** single-stream prose varied 46–55 tok/s across four runs with identical settings
  (+10 % at c1 is inside that spread); the 64k prefill point is "one noisy reading"; `tools/bench_streams.py`
  rows are not comparable to the headline table.
- **Knobs it turns:** fp8 KV (+1.7× pool); `TF_FLASHNEXT_GATHER_READ=pread` instead of mmap; early
  gather of all lanes' n-gram rows; draft read-ahead; `KEEP_LATE=1` (no DeltaNet replay on partial
  rejection); fused hyper-connection kernels; `PROMPT_GEMM2` two-stage MX128 tile; shared expert on a
  second stream; `PROMPT_EXPERT_TILE_ROWS=256`; solo-chunk evener at 512 rows.
- **Stated bottleneck:** prefill of long prompts (2.3k tok/s at 128k) and TTFT under load; decode
  acceptance is text-dependent. No single kernel soul-searching — the engine is already near roofline.
- ⚠️ **Internal inconsistency:** the intro cites TensorFold **v1.2.7** (`93a2e93e`) while the body,
  patch list and image tag all say **v1.1.4** (`eab85828`). Treat the version claim as unverified.
  Also: this README's own "performance" block is the *fp4* path; the INT4-AutoRound checkpoint is
  explicitly unsupported on this branch.

### 1.2 MiaAI-Lab — the older single-Spark **vLLM** recipe (still the reference for vLLM numbers)
<https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark>

- **Config:** vLLM, `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=2048` (2048-token prefill chunks),
  `KV_TARGET_GIB=20` (clipped to 16.67 by `HOST_RESERVE_GIB=26`), fp8 KV, `MAMBA_SSM_CACHE_DTYPE=bfloat16`,
  MTP 3, V2 model runner pinned, PLE table mmapped from NVMe.
- **Measured:** 48.7 tok/s c1 prose / 162.9 aggregate at 8 streams (600-token replies). Prefill
  2,200 @8k → 2,146 @128k. KV 992,584 tokens = 3.79 full 262k requests.
- **Knob measurements worth stealing:**
  - **MTP depth sweep (K=0/1/2/3):** K=3 wins at every concurrency, K=2 ties, **K=1 loses 8–14 %,
    K=0 loses 32–46 %**. No crossover, so adaptive-k has nothing to schedule.
  - **`MAMBA_SSM_CACHE_DTYPE=bfloat16`: +8.5 % at 8 streams** (151.6→164.5 tok/s), acceptance unchanged.
  - **Chunk width 2048→1024:** decode-under-prefill stall p95 1,111 ms → 666 ms and in-window delivery
    5.8 → 9.3 tok/s, **for 5.5 % of 64k prefill throughput**. Not shipped, because long prompts arriving
    against live streams wasn't the target workload.
  - Structured ("count-to-100") decode is **~35 % above prose** — they publish it separately precisely so
    other people's structured-prompt numbers can be compared like-for-like.
- **Stated bottleneck:** decode-under-prefill stalls (one engine step per 2,048-token chunk, by
  construction); prefill flat at ~2.1–2.3k tok/s; host memory headroom, with three server losses at
  `KV_TARGET_GIB=22` before the host-side cap existed.

### 1.3 NVIDIA developer forum — tonyd615 / tonyd2wild (the "37 tok/s" neighbourhood)
<https://forums.developer.nvidia.com/t/qwen3-8-flash-next-on-1-2-and-4-dgx-sparks-with-nvidias-official-nvfp4-quant-64-tok-s-peak-single-stream/382476>
and <https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark>

- **Stack:** upstream vLLM nightly `8a728663`, NVIDIA's official `nvidia/Qwen3.8-Flash-Next-NVFP4`
  byte for byte, plus a PLE-on-disk patch.
- **One Spark, final config:** staged gather in `prepare_inputs` **before** the forward into a fixed GPU
  buffer (so decode still runs as CUDA graphs, torch.compile off), **reduced-vocabulary MTP draft**
  (drafter projects onto the 65,536 lowest ids, target verifies full vocab → outputs exact), MTP3,
  6 seqs, **4096-token prefill chunk**, FP8 KV, `gmu 0.80`, 262k ctx, thinking off, prefix cache off.
- **Measured (40 real prompts, 8 categories × 5, cold prefill):** median **43.9 tok/s** c1;
  prose **29.0**, code 44.3, JSON 49.3, narrative 28.5. Morning stack (MTP4, piecewise graphs,
  no staged gather) was **32.5 median**. Aggregate 43.9 / 47.1 / 68.8 at ×2/×4/×6. KV 1,027,392 tokens.
  Cold prefill 1,269 @7k, 1,760 @113k. Count-to-100 peak 49.4 (footnote only).
- **Knobs / lessons:**
  - The **4096-token prefill chunk is ~2/3 of the TP2 speed jump, but only with torch.compile off**;
    with compile on, decode collapses to **8–15 tok/s**.
  - With the table resident, Inductor autotune makes a **second ~24 GB copy of the table** during
    compile and rebooted three Sparks (→ vLLM PR #55272).
  - TP4 on GB10 **requires `--enable-expert-parallel`** (FlashInfer CUTLASS can't pad the MoE
    intermediate 4 ways). EP on *top of* TP2 was a wash (−4 %).
  - MTP4 + index sharing: +3–4 % coding/JSON, **−6 % prose**.
  - Eager floor without MTP or graphs: **15.4 tok/s** on every category.
  - The MTP head costs ~290k tokens of KV pool at the same `gmu`.
- **Stated bottleneck:** PLE gather latency at batch 1 (one host↔device sync per step), and the draft
  head's cost over a 250k vocab. Both are what the two patches above attack.

**Cross-checks in the same thread:** clint25 (TP2, SGLang vs vLLM, 6-request aggregate 40.3 vs 38.55 tok/s,
tool-eval 90 vs 86, llama-benchy@259K 56.0 vs 42.2) and m0l0 on one Spark (vLLM 0.28.1rc1.dev388):
34.6 tg/s c1, 41.7 c2, 55.1 c4, 1,666 pp/s, tool-eval **89/100** with thinking off; with
`reasoning_effort: medium` the score is unchanged but wall time goes 1,160 → 1,720 s for 5 % more tokens.
m0l0's one-line fixes: add `"FP8_PB_WO"` to the quant-algo list in `modelopt.py`, and prefix caching
does work when enabled (~200k-input turns 3–4 s TTFT).
Thread consensus after that: **use thinking off (or medium) for agent loops, not xhigh** — this model
over-thinks relative to every comparable Qwen.

### 1.4 ursuciprian — 1× Spark, b12x kernels, quality-gated releases
<https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark>

- **Stack:** vLLM + **b12x GB10 kernels** (NVFP4 MoE, MXFP8 linears, 36 GDN, 12 QSA), plan and compile
  caches baked into the image, checkpoint `ursuciprian/...-NVFP4-GDN-MSE` @ `03f4a057` (GDN projections
  NVFP4 for decode, MXFP8 for prefill), **drafter D3 refit on the served model's outputs**, MTP ×4.
- **Config:** TP=1, 26.8 GiB PLE paged from NVMe (`VLLM_PLE_MMAP=1`, WILLNEED before each gather, 50 ms
  NVMe keepalive), ~72 GiB other weights resident, **14 GiB KV pool**, `max_num_seqs 8`, 262k ctx.
- **Measured (mean of 6 boots, llama-benchy task mode ISL/OSL 2048/512, T=1.0, thinking on):**
  **60.4 ± 3.2 tok/s c1** (per request 56.9), **136.7 ± 3.3 aggregate at c8**; coding T=0 thinking off
  **74.9 c1 / 192.0 c8**. Copy-heavy (acceptance ≈ 1) 80.8 / 183.5 / 274.1 at c1/c4/c8. Prefill
  1,916 @2K → 1,872 @128K. TTFT 7.1 s at 16K cold. Power: 24.2 W c1, **2.49 tok/s/W**, 28.9 W c8.
  Hardmode 92/100, TC-45 100/100, fidelity 20/20 to ~245K.
- **Knobs:** one GDN prefill staging buffer shared by all 36 GDN layers (**~8.8 GiB freed**), compact MTP
  draft records, 4 drafts over a **131k-id draft vocabulary** with rejection sampling, baked plan/compile
  caches (256 s boot incl. everything), `max_num_seqs 8` not 16.
- **Stated bottleneck:** per-request TG collapses past c4 (32.4 at c4, 22.0 at c8) because the 14 GiB pool
  and one-GPU bandwidth are the ceiling; the answers are TP=2 (86.3 c1) or DP=2 (two replicas, 203.6 at c8).

### 1.5 jschmied — the deepest engineering log on one GB10
<https://github.com/jschmied/qwen38-flash-next-gb10>

- **Stack:** vLLM `main` nightly `1ea7c63f4`; PLE read in place (vllm#58439); **RecoverSSM** for GDN
  (`--use-replayssm`, vllm#58863); **MTP K=5 with probabilistic drafting**, needs `--block-size 1728`,
  over a 32k-token NVFP4 draft head; bf16 SSM state; `--prefix-match-unit 64`; fused HC kernels;
  full CUDA graphs on the verify path; fast-loading set (vllm#58868 + expert-name index).
- **Measured (validated on the prod launcher):** code c1 **68.1–68.4 tok/s** greedy (14.6 ms/tok,
  4.37 accepted per cycle), prose **42.2–42.4**; 4 streams code 144.3–144.7 aggregate, 8 streams prose
  ~122, 16 streams ~162–165. TTFT 2.71 s @7.5k, 9.69 @29k. Cold start **2 min 39 s** (was ~12 min).
  KV 105,325 tokens per 4 GiB (791,113 at default size).
- **Magnitudes:** `--prefix-match-unit 64` → warm agent turns **0.567 s vs 0.87 s (−74 % recompute)**;
  fused HC → **TTFT −5.5…−7.9 % @8k, −6.2…−7.8 % @30k**; Triton NVFP4 prefill MoE + fused GDN norm/FP8
  quant → **−6 % TTFT**, prefill ~3,210 tok/s; bf16 SSM state → attention block 1,600→**832**, KV capacity
  **+21–36 %**, agent TTFT **−9.6 %** (but 127 of 2,504 modal predictions changed); K=5 better for code,
  **prose 5 % slower than K=3**; distance to byte floor 1.21–1.25×.
- **Stated bottleneck / doctrine:** *"Agent work is TTFT- and recompute-bound. Decode tok/s is a poor
  optimisation target past ~35 tok/s."* And: QSA makes decode essentially context-length invariant
  (26.8 → 27.1 tok/s from 4k to 60k no-spec), so **long context is a memory/admission problem, not a
  decode-speed problem**.
- **Doctrines worth quoting in any design review:** speculation and quantisation are *substitutes* on
  dense projections but *complements* on `lm_head`; the shared expert removing 8 % of bytes bought +1.9 %
  at c1 while costing ~2 % NLL; hyper-connections are 25 % of GPU time as tens of thousands of ~30 µs
  calls, so quantising them made c1 *slower*. "Field mechanisms transfer; field numbers do not."

### 1.6 blazux — 1× Spark / GX10, profile-driven vLLM wrapper + the clearest explanation of the PLE
<https://github.com/blazux/qwen3.8-Flash-DGX> · [HOW-IT-WORKS](https://github.com/blazux/qwen3.8-Flash-DGX/blob/main/docs/HOW-IT-WORKS.md)

- **Stack:** vLLM 0.30 → **0.33**, `VLLM_PLE_MMAP=1` patch on `Qwen4ExpNGramEmbedding`
  (placeholder instead of the resident table; drop the 128 shard tensors; gather through a custom op so
  it is a graph split point), so decode keeps **piecewise** CUDA graphs (never `FULL*` capture;
  `--enforce-eager` is slower).
- **Measured:** 17 tok/s no speculation; **25–28 tok/s** prose with MTP=2 (~63 % acceptance), up to
  **36 tok/s** on predictable text (~94 % acceptance). Prefill 2,400–2,660 tok/s @32k (llama.cpp ~540,
  and no MTP there). vLLM 0.33 `mtp-fast` (NVFP4 proposal head, 3 drafts, 16k prefill batch, 13 GiB
  FP8 KV, 500k ctx): cold prefill **3,516 @8k**, decode **42.6 solo / 128.3 at 8** — +21 % / +15 % over
  vLLM 0.32 MTP3.
- **Knob magnitudes (from `docs/SERVING.md`):** `DRAFT_VOCAB=1` (drafter scores 65,536 ids)
  **+20 % decode**, outputs unchanged; `FAST_ROWS=0` (every gather to the thread pool)
  **+8 % decode at 1 stream, +17 % aggregate at 4**, same rows; `EXACT_TOPK=1` **−8 % prefill @8k,
  −20–40 % @32k+**; `COMPILE_CACHE` **−37 s** per boot; `GPU_MEM 0.80` default because 0.85 drifted into
  swap after a day and 0.875 got OOM-killed on a 300k prefill with MTP; `MADVISE=random` (no readahead)
  cold prefill −4–8 % but a cleaner page cache; `BATCH_TOKENS` 8192 → 16384 in the fast profile.
- **Stated bottleneck:** the gather's host↔device sync is pure latency at batch 1 (MTP amortises it);
  at `GPU_MEM=0.80` the page cache holds only part of the 48 GiB table, and **~6.5 of the ~9.5 ms per
  lookup is the disk gather itself**. Also: first touch of a cold region makes single-shot prefill
  measurements 2–3× cache-state dependent — quote PREWARM'd or second-pass numbers.
- **Known correctness scars:** prefix caching crashed (`CUBLAS_STATUS_INTERNAL_ERROR`, later illegal
  access) on the GDN cached-block path; the root cause was a vLLM **block-size** bug — Mamba state
  captured at `mamba_block_size` boundaries while attention pages didn't agree, giving cache-hit answers
  that differed from cold ones. Fixing block size was necessary before any cache-hit number is trustworthy.

### 1.7 dolf3131 — EXL3 4.05 bpw on TensorFold/SparkFold (the non-NVFP4 single-Spark lane)
<https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark/tree/main/lanes/exl3-sparkfold>

- **Stack:** TensorFold 0.6.5 (Python) + SparkFold `806b345` + 18 patches; checkpoint turboderp EXL3
  `4.05bpw_h6_ng6` 99.94 GiB (trunk 62.9 GiB on GPU, 36.36 GiB n-gram mmapped, MTP head included).
- **Config:** `--parallel 5 --context 262144 --kv-dtype int8 --mtp-drafts 6 --mtp-confidence 0.60
  --thinking --vision`, memory floor 6 GiB (stock 12.2), `TF_SOLO_ROWS 65536`, `TENSORFOLD_PREFILL_ROWS
  4096`, `TF_EXL3_MOE_WINDOW 4096`, NVMe spill for evicted conversation states, Korean draft vocab.
- **Measured:** **5.28 GB read per decoded token** vs **7.09 GB** for an NVFP4 QAD build (header
  arithmetic) — that is the whole speed case for EXL3. Decode 62.9 tok/s English / 54.1 Korean c1
  (600-token replies); sgbench end-to-end 66.9 / 92.1 / 131.1 at 1/2/4 streams. Cold prefill
  1.8–2.1k tok/s to 49k. Resuming a 100k conversation: 40–50 s from scratch vs 0.8–2.3 s from NVMe spill.
- **Stated bottleneck:** nsys round = median 39.7 ms, **85 % in the main verify graph**, 2.6 ms drafts,
  2.4 ms idle gaps, large kernels at 77–91 % of sustained bandwidth. Its own honest note: two decode
  measurements a day apart (57.4 vs 42.6) established **no gain** from the patch set.
- **Quality:** pack card reports KLD 0.0067 vs BF16 at 4.05 bpw — *not measured by the lane author*.

### 1.8 Saren-Arterius — INT4/W4A16 AutoRound fork (the "1.8×" claim)
<https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound> (fork of blazux; active work on `magi-v5-public`)

- **Config:** Intel `W4A16 AutoRound` int4 experts + int8 GPTQ `lm_head` + blockwise-FP8 side layers,
  FP8 PLE table from NVMe, `MTP=3`, `PREFIX_CACHE=1` + never-evict prompt pin, `SEQS=16`,
  **8192-token prefill chunks**, and a deliberate `GPU_MEM=0.01` + `KV_BYTES=20g` instead of a fraction,
  "so the driver never oversubscribes the unified pool (`NV_ERR_NO_MEMORY` / Xid 31 freezes)".
- **Measured:** ~68 GiB resident; **49.1–62.1 tok/s** single-stream by task (Code 56.5, JSON 62.1,
  LongCode 48.5, Q&A 49.1), ~2,100–2,200 tok/s prefill. Claims **~1.8× decode vs the NVFP4 recipe on the
  same box**. Caveat it states itself: those numbers use the PLE table served from **another machine's
  RAM over RDMA**; local NVMe is **2–3 % slower**. Thinking on → ~70 % acceptance (2.1 of 3);
  thinking off → ~87 % and 5–10 % faster.
- **⚠️ Unverified / quality risk:** the 1.8× is a *checkpoint swap*, not a tuning gain. It is a different
  artifact from `azampatti`'s INT4 pack (which keeps 5 of 10 experts and, per MiaAI-Lab, scores ~10 % lower
  general capability: 46.6–47.6 vs 51.8). Whether Saren's W4A16 keeps all 10 experts and how much quality
  it loses is **not stated anywhere I could find** — no KLD, no eval. Do not adopt on the speed claim alone.
- **Cross-confirmed elsewhere:** RDMA-served PLE is only **+2–3 %** (thread post #27), so the RDMA part of
  the story is real but small.

### 1.9 adapt-ai-systems — TP2 recipe with the best "what each piece bought" table
<https://github.com/adapt-ai-systems/qwen3.8-flash-next-tp2-dgx-spark>

- **Config:** 2× GB10 over ConnectX-7 RoCE (MTU 9000, both RDMA devices), vLLM day-0 image + bind-mounted
  patches, TP2 + EP (`--all2all-backend allgather_reducescatter`), **MTP k=4**, reduced 79,591-id draft
  head, W8A16 dense, selective FP8 W8A8, FP8 PLE, RoCE one-shot all-reduce via b12x, unlocked clocks,
  `THP defrag=defer`, `sync + drop caches + compact` before boot.
- **Measured (T=0, thinking off, one request):** **26.18–26.22 speculative steps/s, 75.0–76.6 tok/s**
  at 2.87–2.93 tokens/step; code 100.8–103.1, prose 63.2–65.5; cold prefill 3,223 / 3,270 tok/s @8K/32K.
  Concurrency smoke: 192 / 261 / 286 aggregate at 4 / 16 / 32.
- **Per-change magnitude (control boot each time):** unlocked clocks **+4.6 %**; THP defer removes the
  slow outliers (worst request 19.6 → 25.7 steps/s, mean +6.7 %); boot from flushed/compacted memory
  avoids 21.7 vs 25.8–26.0 steps/s after copying 126 GB; megamoe ≤8-token batches **+0.8…+1.7 %**;
  RoCE one-shot all-reduce **+4.8 %**; balanced draft head **+3.6 %**; W8A16 + k=4 + 79K draft list
  **58.5 → 68.8 tok/s**.
- **Explicitly did NOT help:** Marlin FP8 kernel (−6–9 %), `--moe-backend marlin` (prefill −8 %),
  adaptive draft length (+0.5 %), `NCCL_MAX_NCHANNELS=1` (prefill −5 %), 4-bit dense weights (+7.3 % steps/s
  but **failed the quality gate**), FP8 KV (ruled out earlier), `--language-model-only` (+36 s boot stall),
  K=3/5/6 and full 248K draft vocab (all slower than k=4 + reduced list).
- **Stated bottleneck:** step cost at batch 1 (fabric + kernels); they tune for latency, not throughput,
  and say so.

### 1.10 PixelML — SGLang TP2, and the SM121 corruption story
<https://github.com/PixelML/qwen3-8-flash-next-sglang-2x-dgx-spark>

- **Config:** `lmsysorg/sglang` pin, TP=2 across two Sparks, NEXTN/MTP 3 steps top-k 1 (4 drafts), 262k ctx.
- **Measured (192-token coding requests):** 47.54 / 87.55 / 158.17 / 275.37 tok/s at 1/4/8/16 concurrent.
  **MTP: 26.09 → 47.54 tok/s (+82 %)** at acceptance 0.6–0.8. Client-observed uncached prefill
  2,328 / 2,759 / 2,960 tok/s at 1K/4K/16K.
- **The bug everyone should know:** on SM121, FlashInfer's TRT-LLM sparse-decode kernel **silently emits
  token id 0 (`!`)** at long context (sglang#36806, #36537), filling the output budget with `!`. Fix =
  force the Triton packed-varlen fallback for QSA on SM121, plus a guard that aborts after 16 consecutive
  token-0 samples and resets the radix cache. **Gated DeltaNet: Triton for prefill, FlashInfer for decode,
  BF16 Mamba state** — the split avoids incompatible state-dtype requirements on SM121.

### 1.11 SGLang cookbook (authoritative knobs for the Spark, even though not an official Spark recipe)
<https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next>

- **Single Spark:** only with `--ple-offload-embedding --ple-offload-backend file` (sparse 47.7 GiB file
  under `$SGLANG_CACHE_DIR/ple/`, filled on boot, hot pages in page cache capped by
  `SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=8`). 78.3 GiB resident, `--mem-fraction-static 0.85` → 12–18 GB
  for pools → **8 requests with MTP or 24 without**. Measured: **27.5 tok/s c1 with MTP (TPOT 33.6 ms),
  71.7 at 8; 15.9 c1 without MTP, 83 at 24.**
  - ⚠️ **Boot-time trap:** every boot rewrites the whole table; on an already-populated file that is
    ~17 MB/s and **~55 minutes**. Delete `ple_table_*.bin` before each boot.
- **Two Sparks TP2:** `--no-ple-offload-embedding` (table GPU-resident and sharded, ~65 GB/node).
  `--mamba-track-interval` left at default 256 (not 64) → **~40 % larger KV pool** (1.48M tokens at 24
  concurrent with MTP), at the cost of coarser prefix-cache reuse of the recurrent state.
  100k prefill 2,400–2,840 tok/s; **MTP accept length 3.5–3.7 of 4** (dropping to ~2.5 on thinking output).
- **TP2 NVDA export specifics:** `--moe-runner-backend flashinfer_cutlass` explicitly (the auto-default
  picks `flashinfer_trtllm` on GB10, which the NVFP4 MoE method rejects at autotune); no `--quantization`.
- **Studio-level advice:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fragmentation at 200k+),
  and a host watchdog that kills the server below a few GiB free — **a unified-memory exhaustion can take
  the whole box down and needs a power cycle**.

### 1.12 sostrowsk — engine comparison, and two measurement traps
<https://github.com/sostrowsk/dgx-spark-local-llm>

- **Config:** vLLM 0.28, `RadixArk` NVFP4 (126 GiB, FP8 PLE, 419 files) over the recommended
  `Inferact` (170 GiB, one 95.4 GiB file); PLE offloaded to a CPU process whose pages go to **swap**
  (50 GiB swap, 87 GiB on GPU); `--distributed-executor-backend mp` (otherwise the offload worker is never
  spawned and the engine **waits forever with no error**); `--max-num-seqs 32` (with MTP only 73 Mamba
  blocks remain; the default 256 aborts the start).
- **Measured:** **39.1 tok/s** prose c1, 47.4 counting (both MTP k=2, **95.8 % acceptance**);
  **17.9–18.0 tok/s without speculation, flat from 36 to 26,473 prompt tokens**. Prefill 913 @6.6K,
  **3,874 @26K**; 938 s cold start.
- **Traps it documents:** don't count streaming chunks under MTP (gave 12.3 tok/s, *slower* than no
  speculation) — use `usage.completion_tokens`; a counting task overstates the drafter by ~20 %;
  hybrid models log `enable_prefix_caching=False` unasked; `gpu-memory-utilization` > 0.89 aborts;
  missing `k_scale`/`v_scale` silently gives you a bf16 KV cache at half capacity.

### 1.13 Official vLLM recipe — **no Spark entry**
<https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next> ·
[`models/Qwen/Qwen3.8-Flash-Next.yaml`](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-Flash-Next.yaml)

Verified hardware is H100, H200, B200, GB200, B300, GB300, DGX Station (GB300), RTX Pro 6000 4×, MI300X/
MI325X/MI355X/MI350X. **GB10 / DGX Spark is not in the list**, and there is no Spark variant — so every
Spark recipe in this report is community work, and none of the official numbers apply. The official knobs
that *do* transfer as defaults: **TEP over plain TP** ("consistently faster for this model"),
`--max-num-batched-tokens 8192`, MTP `num_speculative_tokens 3`, `--kv-cache-dtype fp8` **plus**
`--attention-config.indexer_kv_dtype fp8`, PLE CPU offload on by default
(`--engram-config '{"cpu_offload":true}'`), and for GB300-class `-cc.mode none -cc.cudagraph_mode
full_decode_only`. It also warns: benchmark MTP with SPEED-Bench, not `random`.

### 1.14 TensorFold upstream (engine vendor) — the reference numbers and the chunk-size study
<https://github.com/ashhart/TensorFold> · [`docs/recipes/flash-next-cuda.md`](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/flash-next-cuda.md)

- **Supported path:** `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` on one GB10, 73.3 GiB uploaded, n-gram
  tables kept mapped, vision optional. NVFP4/two-rank paths are labelled **experimental**.
- **Measured on GB10 (greedy, 256 tokens, median of 3):** MLX 4-bit — **84.2 tok/s code / 62.6 chat drafted**;
  **35.1** both, `--no-drafts`; drafted 1/2/4 streams greedy code 83.8 / 82.7 / 82.2; cold first token
  2.67 s @3,001, 10.40 s @12,001; 45,030-token prompt 36.5 s no cache. `nvidia/...-FP4-Local` (ModelOpt,
  measured on a pre-1.1.0 branch build, **not remeasured**): 64.1–64.6 code / 40.7–41.1 chat; 1/4/8 streams
  64.1 / 44.9 / 66.2; TTFT 1.83 / 7.11 / 29.2 / 60.1 / 131.7 s at 2k→131k.
- **The cleanest chunk-size experiment anywhere:** `TF_FLASHNEXT_CHUNK` (default **1,536**), plus
  `TF_FLASHNEXT_CHUNK_SOLO` (default 8,192 at `--parallel 1`). On one GB10, 65,536-token prompt, three
  fresh servers per setting: **35.5 s (1,536) → 30.7 (4,096) → 29.5 (8,192) → 28.8 (16,384)**, decode
  unchanged, replies byte-identical, at 75.8 → 82.9 GiB. At `--parallel 3`: 4,096 rows takes most of
  8,192's gain for 2.25 GiB instead of 5.86 GiB; a short request arriving 2 s into a 31,736-token prompt
  waits **0.36 s (default) vs 1.91 s (solo chunk)**.
- **Other flags with numbers:** `TF_FLASHNEXT_READOUT_ROWS=128` keeps 16-stream decode on row kernels,
  **up to 7 % quicker**; `TF_PROMPT_INTERMEDIATE=1` resumes 65,536 instead of 2 tokens on a 104K request
  (TTFT **19.9 s vs 50.7 s**); `--tp 2` with two RDMA devices listed per port made cold 2,048–32,768-token
  prompts **26.5–28.1 % faster** (1,147 → 1,469 tok/s @8,192).
- **Its own known issue:** on one GB10 the 29.8 GiB n-gram tables page against the prompt cache —
  65k-prompt TTFT **34 → 38 s** as saved states fill at 1,536-row passes, vs **27 → 32 s** at 16,384-row
  passes (`docs/known-issues.md`).

---

## 2. Ranked knobs, by measured magnitude on this model and hardware

Magnitudes are as reported by the source named; they are **not additive** and several were measured on
different checkpoints.

| # | Knob | Reported magnitude | Source | Confidence |
|---|---|---|---|---|
| 1 | **Get the PLE table out of unified memory** (pread/mmap from NVMe, or RDMA from another box) | Makes the model servable at all (frees 26.8–48 GiB). Choose the reader: `FAST_ROWS=0` pool gather **+8 % c1 / +17 % aggregate @4**; RDMA serving **+2–3 %** over local NVMe | blazux, Saren, thread #27 | high (multiple independent) |
| 2 | **MTP/speculative decoding on, with a reduced-vocabulary draft head** | No-spec → spec: **26.09 → 47.54 (+82 %)** SGLang; **17.9 → 39.1** vLLM; **35.1 → 84.2 (code)** TensorFold. Reduced 65k-id draft head alone **+20 %**; 131k-id + rejection sampling in the 60 tok/s recipe | PixelML, sostrowsk, TensorFold, blazux, ursuciprian | very high |
| 3 | **MTP depth k** (model-specific optimum, not "more is better") | K=3 best at every concurrency in vLLM (K=1 **−8…−14 %**, K=0 **−32…−46 %**); k=4 optimum on TP2 (k=3/5/6 slower); K=5 better for code but **−5 % prose**; MTP4+index-share **+3–4 % code/JSON, −6 % prose** | MiaAI, adapt-ai, jschmied, tonyd615 | high |
| 4 | **bf16 (not fp32) recurrent/SSM state** | **+8.5 % at 8 streams**; attention block 1,600→832, **KV +21–36 %**, agent TTFT **−9.6 %** | MiaAI, jschmied, ursuciprian (`mtp-fast`) | high, but changes numerics (127/2,504 predictions) |
| 5 | **Prefix-cache geometry**: `--prefix-match-unit 64`, correct `mamba_block_size`, cache-hit determinism | Warm agent turns **0.87 → 0.567 s (−74 % recompute)**; unfixed block size = cache-hit answers that differ from cold ones | jschmied, blazux | high |
| 6 | **Prefill chunk size** (the most context-dependent knob in the whole model) | 4,096 chunk = **~2/3 of the TP2 speed jump, only with compile off**; 1,536→8,192 rows: 65k TTFT **35.5 → 29.5 s** (+7.1 GiB); 2048→1024 buys decode-under-prefill p95 **1,111 → 666 ms** for **−5.5 % prefill**; arriving-request wait **0.36 → 1.91 s** with a solo chunk | tonyd615, TensorFold, MiaAI | high (measured three ways) |
| 7 | **CUDA graphs / compile mode** | Piecewise-with-split-point is the only viable setting when the PLE gather is inline: compile ON = **decode 8–15 tok/s**; eager floor = **15.4 tok/s**; Inductor autotune duplicates the resident table (~24 GB) and rebooted boxes | tonyd615, blazux | high |
| 8 | **Kernel-fusion work on the prefill path (HC / GDN)** | Fused HC kernels **TTFT −5.5…−7.9 %**; Triton NVFP4 prefill MoE + fused GDN quant **−6 % TTFT**; `PROMPT_GEMM2` "same bits, +15–25 % per call"; FLA **99 KiB vs 100 KiB shared-memory gate** silently forced all 36 GDN layers onto small tiles | jschmied, MiaAI, blazux (via Saren) | medium-high |
| 9 | **KV dtype: fp8 is capacity, not speed** | Pool **×1.7–1.8** (4.78 GiB vs 7.52 GiB per 262k window; 4.11 → 7.07 concurrent 262k requests), decode neutral; ~98.8 % top-1 agreement with bf16 | MiaAI, jschmied, blazux | high |
| 10 | **Checkpoint choice (expert residency + weight bytes/token)** | EXL3 4.05 bpw: **5.28 vs 7.09 GB read/token** → 62.9 tok/s c1; INT4 W4A16 + int8 head + fp8 side layers: ~68 GiB resident, **~1.8× decode**, quality unmeasured; NVFP4 QAD is the safe default (10 experts, near-bf16 scores) | dolf3131, Saren, MiaAI | medium (Saren unverified) |
| 11 | **`gpu_memory_utilization` / KV sizing** | 0.80 default; 0.85 drifts into swap after a day, 0.875 OOM-killed on a 300k prefill with MTP; on a Spark, every GiB of budget you take from the page cache is prefill speed | blazux, tonyd615 | high |
| 12 | **Load/startup hygiene** | Cold start **575–651 s → 56–57 s** (fast-load + expert-name index), prod-ready start **2 min 39 s**; `COMPILE_CACHE` −37 s/boot; flush+compact before boot avoids a 21.7 vs 26 steps/s first boot | jschmied, blazux, adapt-ai | high |
| 13 | **TP2 / TP4 (not a knob on one box)** | TP2 **+43 % c1** (60.4 → 86.3, or 32.5 → 53.7 on the other stack) and **13.5× KV**; TP4+EP 40.5 c1 with 9.09M-token pool; **EP on TP2 is a wash (−4 %)**; TP4 requires `--enable-expert-parallel` | ursuciprian, tonyd615 | high |
| 14 | **Deterministic QSA top-k** | `DET_TOPK=1` (vllm#55122) same speed; `EXACT_TOPK` fallback **−8 % @8k, −20–40 % @32k+**. Stock top-k may drop attention candidates; 40 identical greedy requests gave **40 different completions** on stock | blazux, jschmied | high (correctness) |

### Cross-cutting negative results (don't spend a weekend on these)
Marlin FP8 for W8A16 (−6–9 %), `--moe-backend marlin` (prefill −8 %), `flashinfer_trtllm`/`flashinfer_cutedsl`
MoE on sm_121 (rejected by device check), b12x/CUTLASS FP4 MoE with EP off (boot failure), megamoe above
8-token batches, adaptive draft length (+0.5 %), `NCCL_MAX_NCHANNELS=1` (−5 % prefill), 4-bit dense weights
(+7.3 % steps/s, failed quality gate), `--language-model-only` (+36 s boot stall), quantising the shared
expert or hyper-connections (the latter is *slower*), and `--enforce-eager` except as a fallback.

---

## 3. Bottleneck consensus

1. **PLE gather latency at batch 1** — one host↔device sync per decode step; ~6.5 of ~9.5 ms per lookup
   is disk when the page cache doesn't hold the table. Mitigations: MTP to amortise, staged gather before
   the forward, thread-pool gathers, read-ahead, or a resident/RDMA mirror.
2. **Prefill, not decode, for long prompts** — 1.2–2.8k tok/s depending on stack; every TTFT complaint in
   the forum thread is a prefill or prefix-cache-reuse problem, and jschmied's data says TTFT is 53–69 %
   of an agent turn.
3. **Bandwidth floor** — decode is 1.2× the byte-count floor with kernels at 77–91 % of sustained
   bandwidth; further single-stream gains must come from *fewer bytes* (checkpoint, draft head) or *more
   tokens per pass* (speculation), not from a better GEMM.
4. **Unified-memory headroom** — not a speed bottleneck but the dominant *stability* one: three boxes
   lost to `KV_TARGET_GIB=22`, one OOM-killed at `gmu 0.875`, three rebooted by Inductor autotune,
   kernel `NV_ERR_NO_MEMORY` from 64 KiB contiguous allocations, and SGLang's explicit warning that
   exhaustion needs a power cycle.

---

## 4. Could not verify / looks like an unmeasured claim

- **"~37 tok/s at concurrency 1."** Consistent with the *median across mixed prompts* on vLLM NVFP4
  (32.5 morning / 43.9 evening stacks; 39.1 sostrowsk; 42.6 blazux sampled solo; 60.4 ursuciprian on a
  different checkpoint). It is **not** a ceiling: prose and counting prompts span 21 → 55 tok/s on the same
  box, and the TensorFold/ursuciprian stacks do 55–62 prose. Anyone quoting 37 without a prompt set is
  quoting a median, not a measurement of the model.
- **Saren-Arterius's "1.8× decode":** a checkpoint swap with **no quality evaluation published**, measured
  with the PLE table on another machine over RDMA (2–3 % of the delta). Not comparable to MiaAI/Intel
  numbers, and it sits next to an independent finding that the 4-bit dense variant *failed a quality gate*
  and that a 5-expert INT4 cut costs ~10 % capability.
- **MiaAI-Lab TensorFold README:** cites engine v1.2.7 (`93a2e93e`) in the intro and v1.1.4 (`eab85828`)
  everywhere else; the performance table cannot be attributed to one build as written. Also its
  TensorFold-vs-Python-engine deltas (+17…+43 %) are labelled by the author as "indicative" only.
- **TensorFold's own FP4-Local numbers** are from a pre-1.1.0 branch build by a contributor and "have not
  been remeasured on 1.1.2"; its NVFP4 two-rank path is explicitly experimental with unqualified
  numerical fidelity.
- **dolf3131's EXL3 gain** — the author withdraws any claim: two runs a day apart differ by more than the
  patch set could explain.
- **"1M context on one Spark":** only with YaRN factor 4 and a 17.46 GiB fp8 KV for a single sequence;
  blazux booted 800k and a 300k prefill was SIGTERM'd by earlyoom at 1.96 % free. 500k is the validated
  single-Spark ceiling; the 4.78 GiB/stream figure in MiaAI's recipe is at 262k, not 1M.
- **Official support:** vLLM's recipe does not list GB10 at all; SGLang's page is a cookbook *note* for
  Sparks rather than a verified cell; TensorFold calls its NVFP4 Spark paths experimental. Every "it works
  on a Spark" claim here is community, and several were only made true by out-of-tree patches.
