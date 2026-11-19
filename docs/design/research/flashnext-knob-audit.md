# Flash Next CUDA — performance knob audit (read-only)

Repo `~/Projects/tf-wt/dflash2`, branch `dflash2` @ `652d477cc`. Served model: `nvidia/Qwen3.8-Flash-Next-FP4-Local`
(ModelOpt NVFP4 → `native_experts` + qmmf native lane). Nothing was built or run.

Key structural fact that gates everything below: `Tri` is built as `.{ .set = self.set, .own = &self.glue }`
(`cuda_lanes.zig:204-206`). With no captured Triton set loaded (`TENSORFOLD_CUDA_KERNELS`), `set == null` and
`Tri.glue(name)` returns `t.own` for every name (`cuda_launch.zig:101-106`) — so `fusedAttentionOwned()`,
`readoutTilesOwned()`, `lanesGlue()`, `writebackOwned()` are all **true** on the GB10. Conversely any captured set
flips every "owned" fast path off unless `TF_FLASHNEXT_OWNED` names the entry.

---

## A. Runtime switches (all read once at load unless noted)

| knob | file:line | what it changes | default | evidence it changes behaviour |
|---|---|---|---|---|
| `TF_FLASHNEXT_CHUNK` | `cuda_chunk.zig:6-16` | prompt rows per pass (busy path) | 1536 (`cuda_batch_state.zig:57`) | parsed to `1..max_chunk(16384)`; `cuda_prompt.zig:159`, `cuda_lanes_load.zig:23` |
| `TF_FLASHNEXT_CHUNK_SOLO` | `cuda_chunk.zig:21-28`, `lone_rows` :22 | prompt rows when no other stream decodes | unset ⇒ `lone_rows = 8192` if lone, else = busy chunk | `parseSolo`; wired at `cuda_lanes_load.zig:23` (`one_stream and tp==1`) and `cuda_prompt.zig:159-160` (`soloRows(busy,false)` ⇒ off at engine load); `passRows` :31-34 picks per round |
| `TF_FLASHNEXT_READOUT_ROWS` | `cuda_chunk.zig:36-41` → `cuda_batch_kernels.zig:42,50` | rows from which the HC read-out takes the `flashnext_hcf` tiles; 0 = never | 64 | `readoutKeep` gate `cuda_batch.zig:297` `n >= self.k.forms.readout_rows` |
| `TF_FLASHNEXT_FACT_ROWS` | `cuda_nvfp4_tile.zig:6-10` | rows from which NVFP4 experts take the prompt (fact) tiles instead of the 16-pair decode form; 0 = never | 64 | `nvfp4_tile.size(k,n)` :14-16 |
| `TF_FLASHNEXT_PROMPT_EXPERT_TILE` | `cuda_batch_kernels.zig:104,136`; `cuda/experts_tile.zig:13-19` | loads `nvfp4_experts_tile` module; `0`=off, `1`=up only, `2`=up+down | off | `nvfp4_tile.run` :19 `if (pt != 16) if (k.expert_tile) … tile.takes(epi)` |
| `TF_NVFP4_DECODE_WARPS` | `cuda/experts.zig:375-385` | warps per CTA in the NVFP4 decode kernels (1 or 4) | 4 | symbol pick `…ELi4E`→`ELi1E` :396-402; `resident` = occupancy×SMs |
| `TF_QMMF_HALF_BELOW` | `cuda/qmmf.zig:120-125` | MXFP8/BF16 decode uses the 32-column half tile below N 64-blocks; 0 = never | 96 | `half_below_default` :121 |
| `TF_FLASHNEXT_PROMPT_BF16` | `cuda_batch_kernels.zig:126-127` | loads qmmf's 512-row **prompt** lane (`qmmf.prompt_rows` `qmmf.zig:158`) | on | `k.native_lane.loadPrompt(d)` |
| `TF_FLASHNEXT_GDN_FUSED` | `cuda_gdn_fused.zig:9-12` | front+chain+gated-norm in one kernel per value head instead of 3 launches | on | `cuda_batch.zig:353` `fused = lanes.len==0 and (n>=64 or inPass) and forms.gdn_fused` |
| `TF_FLASHNEXT_GDN_LANES` | `cuda_batch_gdn_lanes.zig:22-25` | a shared round's DeltaNet steps one launch per step, all lanes | on | `cuda_batch.zig:369` `gdn_lanes.once(...)` |
| `TF_FLASHNEXT_HC_FUSE` | `cuda_batch_hc.zig:14-17` | writeback+normalize fused, native read-out | on | `hc.fuses` :22-26 requires `n>=64`, `down.native != null`, our glue |
| `TF_FLASHNEXT_ATTN_LANES` | `cuda_batch_lanes.zig:41-45`; `useAttn` :69-79 | a shared round's attention in one launch for all lanes | **off** | `Pass.stage` :580 `self.lanes_attn = batch_lanes.useAttn(...)`; 2–32 lanes, `n<=256`, heads full, no pages/media |
| `TF_FLASHNEXT_LANE_PARTS` | `cuda_lanes_load.zig:50` | reuse a contiguous lane's fixed buffers across streams | on | `parts.on` |
| `TF_FLASHNEXT_KEEP_INPASS` | `cuda_lanes_snap.zig:10` → `Forms.in_pass` | prompt-cache marks filled inside the pass instead of a cut pass | on | `cuda_batch_kernels.zig:41,49`; feeds `fused` at `cuda_batch.zig:353` |
| `TF_FLASHNEXT_WINDOW_GRAPH` | `cuda_window_graph.zig:20-23` | CUDA graphs for drafted windows / draft-head pieces / rollbacks | on | `Cache.run` :199+; disabled ⇒ one `pass.run()` per call |
| `TF_FLASHNEXT_DECODE_GRAPH` | `cuda_round_graph.zig:6-9` | one bounded decode-round graph, recaptured on key change | **off** | predicate `cuda_prompt.zig:546` — **requires `e.collective != null`**, i.e. TP=2 only; also `!has_media`, no `dense_cut` |
| `TF_FLASHNEXT_ATTN_SPLIT` | `cuda_draft2_fwd.zig:762,1037-1044` | drafter attention on split-KV tensor cores | off (exact `"1"` only) | dispatch :1639-1655, not bit-identical |
| `TF_FLASHNEXT_ATTN_LEGACY` | `cuda_draft2_fwd.zig:50,876,1639` | drafter attention on the warp-per-(row,head) kernel (19× slower per the header) | off | same dispatch; bit-identical to the tiled kernel |
| `TF_FLASHNEXT_HEAD_FP8` + `_HEAD_FP8_DIR` | `cuda_draft2_fwd.zig:697-717` | drafter `lm_head` ranked in block-FP8, rescored bf16 | off; needs the artifact | refuses one without the other (:966-975) |
| `TF_FLASHNEXT_DRAFTS` | `cuda_lanes_load.zig:98-100` | MTP chain depth | 6 (`cuda_mtp.zig:33`), clamped 1..`max_chain` | `own.depth` |
| `TF_FLASHNEXT_DRAFTS_GREEDY` | `cuda_lanes_load.zig:101-102` | greedy stream chain depth | 8 (`cuda_mtp.zig:35`) | `own.depth_greedy` |
| `TF_FLASHNEXT_GREEDY_WIDE` | `cuda_lanes_draft.zig:39-42` | round width up to which greedy keeps the deeper chain | 4 | `rule()` :34-37 |
| `TF_FLASHNEXT_CONFIDENCE` | `cuda_lanes_load.zig:103` | chain stop threshold | −0.4 (`cuda_mtp.zig:48`) | `own.confidence` |
| `TF_FLASHNEXT_SHARED_DRAFTS` | `cuda_lanes_draft.zig:44-47` | greedy keeps its own stop rule beside other streams (0.6 `shared_stop` :21) | on | `rule()` |
| `TF_FLASHNEXT_BATCH_HEADS` | `cuda_lanes_draft.zig:7-11` | shared round's streams drafted in one batched head | on | `batchHeads()` |
| `TF_FLASHNEXT_SAMPLED_HEADS` | `cuda_lanes_draft.zig:14-17` | sampled streams take part in the batched head | on | `sampledHeads()` |
| `TF_FLASHNEXT_WINDOW_ROWS` | `cuda_lanes.zig:618` | rows of a shared verify window | `mtp.round_rows=64` (`cuda_mtp.zig:86`), capped by chunk | `cap` at :624-625 (drafter forces 1) |
| `TF_FLASHNEXT_FIRST_EARLY` | `cuda_mtp.zig:40` | MTP first-token path launches early | on | `cuda_mtp.zig` |
| `TF_FLASHNEXT_GPU_TOPK` | `cuda_mtp_draw.zig:21` | top-k of the draw on the device | on | `cuda_mtp_draw.zig` |
| `TF_FLASHNEXT_KEEP_LATE` | `cuda_mtp_draft.zig:131-141` | drafting streams keep a partly-kept window's DeltaNet rows pending | unset (loader decides) | `late.front` / `late.chain` `cuda_batch.zig:362-365` |
| `TF_FLASHNEXT_SHARED_SIDE` | `cuda_batch_side.zig:13` | shared-expert side pass runs beside the routed experts | on | `shared.fork` `cuda_moe_native.zig:23` |
| `TF_FLASHNEXT_KV_PAGES` | `cuda_kv_lanes.zig:15-22` | streams 2+ take n-position KV pages from the core pool | off (0) | `paging()`; `kv8`+pages refused (`cuda_glue.zig:258`) |
| `TF_FLASHNEXT_KV_PAGE_TEST` | `cuda_pages.zig:59-65` | diagnostic: caches read through reversed pages | off | `error.BadFlashNextPageTest` if not a multiple of 4 |
| `TF_FLASHNEXT_SPLIT_DENSE` | `cuda_dense_split.zig:25-34` | split bf16 dense projections at `--tp 2` | on (1) | `appliesDir`; conflicts with `DECODE_GRAPH` (`cuda_lanes_explain.zig:24`) |
| `TF_FLASHNEXT_WARM_AHEAD` | `cuda_ple.zig:381`; `cuda_ple_ahead.zig:4-20` | n-gram reader issues bounded file hints for future tokens | off | `warm()` no-ops when false |
| `TF_FLASHNEXT_TABLE_WARM` | `cuda_ple.zig:442-448` | read the whole n-gram table into page cache after load | off | spawns warmer thread |
| `TF_FLASHNEXT_PLE_IO` | `cuda_ple.zig:380` → `ple_reader.mode` | n-gram table read mode (buffered/direct/pool) | family default | `read_mode` |
| `TF_FLASHNEXT_LAYER_TIME` / `_LAYER_DUMP` | `cuda_layer_time.zig:42-75` | per-layer GPU stage times for the 48-layer pass | off | refused inside a capture (`cuda_batch.zig:597-600`, `cuda_layer_time.zig:143`) |
| `TF_FLASHNEXT_PROFILE` | `cuda_batch_kernels.zig:163-167` | per-chunk wall time + host gather time on stderr | off | `defer` in `Pass.chunk` :590-593 |
| `TF_FLASHNEXT_TAP_SOURCE` | `cuda_tap_ring.zig:69-74` | tap ring row: `readout` / `normed` / `mean4` | mean4 | `tapRingWrite` switch `cuda_batch.zig:247-265` |
| `TF_FLASHNEXT_LOAD_THREADS` | `cuda_modelopt_threads.zig:15-24` | loader threads, 1..32 | 8 | `configured()` |
| `--kv-cache fp8` | `server/cli.zig:115,241,545` → `Engine.kv8` | fp8 (e4m3) attention caches | bf16 | `cuda_glue.zig:264,282,300` picks `*_kv8` entries; refused with pages (:258) and lanes (:341) |
| `--decode-share` | `prefill.zig:7-17` | prompt chunk rows beside live replies | `decode_share=0.8`, `decode_rows=32`, `waited_rows=512` after 8 steps | `besideRows` |

`TENSORFOLD_CUDA_KERNELS` selects the captured AOT set (docs/environment.md:16). `TF_FLASHNEXT_OWNED`
(`cuda_glue.zig:100-110`) is the A/B list of entries that run our glue *beside* a captured set.

Docs table: `docs/environment.md:89-146` (30 flags), `docs/optimization-ledger.md:50-81` (fnext column).

---

## B. Tuning constants (compile-time, no switch)

| constant | file:line | value | controls | note |
|---|---|---|---|---|
| `att_rows` | `cuda_batch_state.zig:47` | 256 | rows per `_chunks` launch; scratch `rows×nch×heads×head_dim` fp32 | also the lanes cap `l.rows <= att_rows` |
| `fused_rows` | `cuda_batch_state.zig:51` | 1024 | rows per fused attention launch | `cuda_batch.zig:495-496`: `fused = n > att_rows` |
| `default_chunk` / `max_chunk` | `:57` / `:59` | 1536 / 16384 | prompt pass rows | max scratch ≈ +13 GiB per comment |
| `attn_lanes` | `:53` | 32 | lanes in one lane-attention launch (`fn_lanes MAX`) | 2..32 accepted |
| `chain_tpr_smem` | `:66` | `2*16*(3*128*4+16)` | GDN chain two-threads-a-row shared | used when `rows >= 64` (`cuda_batch.zig:421`) |
| `gdn_fused_threads` | `:70` | 384 | fused GDN CTA | |
| `nch` / `dense_nch` | `cuda_prompt_wt.zig:33-34` | 5 / 3 | 512-key chunks a row reads, sparse / dense | |
| `index_top` / `index_ratio` / `idw` | `:37-39` | 512 / 4 / 2052 | QSA selection budget | `dense_keys=(512+1)*4-1=2051` :41 |
| `hcf_down`/`hcf_up` tiles | `cuda_tile_kernels.zig:13-22` | 32×192 / 64×128, smem `hcf_*_smem` | HC read-out prompt tiles | |
| `fexp_tile` | `cuda_tile_kernels.zig:14` | 64 | pairs per prompt expert tile | `cuda_batch.zig:552-555` (`n>=64`) |
| `RowsExperts.tile` | `cuda/experts_rows.zig:20` | 32 | pairs per decode expert item | `cuda_batch.zig:558-561` |
| RowsExperts grid cap | `cuda/experts_rows.zig:37` | `min(items*nb/4, 4096)`, block 128 | hard 4096-CTA ceiling | no occupancy query — unlike NVFP4 |
| `grouped.small` | `cuda/grouped.zig:11` | 1024 | pairs the single-block plan holds; above it ranks in 1024-pair blocks | `route` :64-65 |
| `ahead_rows` | `cuda_glue.zig:20` | 16 | `rows <= 16` takes `_chunks_ahead` (2-buffer smem) instead of `_chunks_one` | `cuda_glue.zig:280-285` |
| `qmmf.prompt_rows` | `cuda/qmmf.zig:158` | 512 | rows at which the bf16 prompt lane replaces the decode lane | `promptEligible` :176 |
| `qmmf.half_below_default` | `cuda/qmmf.zig:121` | 96 | see `TF_QMMF_HALF_BELOW` | |
| `Nvfp4Experts.fact_wn` / `prompt_wn` | `cuda/experts.zig:363,366` | 4/8 and 8/4/8 | warps per 32-column block in the fact / prompt entries | "GB10's fastest tiles" |
| `experts_waves.pick` | `cuda/experts_waves.zig:18-21` | — | half kernel when `2*items*nb <= half`; classic when `items*nb <= classic`; else waves | decode form chooser |
| `max_graphs` | `cuda_window_graph.zig:19` | 128 | graphs a cache holds before starting over | |
| `attn_split_plan` | `cuda_draft2_fwd.zig:421-430` | — | split-KV chunk count | |
| `head_tile` | `cuda_draft2_fwd.zig` (top-k) | 8192 (was 1024) | drafter head top-k tile | measured 2.97→1.39 ms |
| `block` | `cuda_draft2_fwd.zig` | 8, refused otherwise | drafter block width | `DraftBlockUnsupported` |
| `late_rows` / `late_store` | `cuda_batch_state.zig:180-181` | 48 / 12 | pending-row log | |
| `max_marks` | `cuda_batch_state.zig:196` | 2 | prompt-cache marks per pass | |

---

## C. MoE — how expert weights are fetched

**Fully resident, no cache, no paging, one stream.**

- On a ModelOpt checkpoint: `native_experts = mixed.load(...)` at load (`cuda_prompt_load.zig:427-430`), giving
  `Owned{ up, down, up_scale, down_scale }` device buffers (`cuda_modelopt_experts.zig:44-49`) — whole experts, `count/world` per rank.
  On the MLX affine-4 checkpoint: `experts_up` / `experts_down` are packed at load by `packAll`/`stackUp`
  (`cuda_prompt_load.zig:466-467`) — all 512 routed + the shared expert at index 512, ~2 MB per expert, ~50 GB total, resident.
- Per-pass path: `moeBlockKeep` (`cuda_batch.zig:537-562`): `routerMany` → `topkMany` → `route` (dedup plan) → grouped expert kernels.
  Native: `moe_native.run` :549. Legacy: `n>=64` → `tiles.experts` on 64-pair `fexp` tiles; else `k.moe.run` (32-pair `RowsExperts`).
- **Dedup exists**: `k.router.route(...)` builds `Plan{members,items,counts,rank,hist}` (`cuda/grouped.zig:20`), 1024-pair blocks above `small`.
- **One stream**: every launch takes `self.stream` (`cuda_batch.zig:552-561`); no side stream, no prefetch. The shared expert
  can overlap only via `TF_FLASHNEXT_SHARED_SIDE` (`shared.fork`, `cuda_batch_side.zig:13`), and only on the native path.
- **No per-slot (non-dedup) GPU path exists.** The kernels take `(items, counts, members)`; a per-slot path would be an identity
  plan (`items = pairs`, `counts[i] = 1`, `members[i] = i`) — a small change, since `Affine4Experts.argsOf` and both
  `RowsExperts.run` / `Nvfp4Experts.run` already read only those arrays. The only per-slot code today is the host reference
  `cuda_moe.zig` (510 lines, `project()`/`swiglu()` on the CPU) and the row path `cuda_rows.zig` — both tools, not the batch.
- Also unused on this checkpoint: the whole affine-4 packed expert path (`experts_up/down`, `fexp` tiles, `RowsExperts`) is
  compiled and tuned but skipped, because the FP4 checkpoint takes `native_experts`.

---

## D. Attention — backends present and the thresholds that pick them

Target (48-layer pass), all under `Tri`/`Glue` (`cuda_glue.zig`), kernels in `kernels/cuda/flashnext_attn.cu`:

| entry | selected when | threshold |
|---|---|---|
| `fn_attn_prep` / `_paged` / `_kv8` / `_media` | pages? fp8? images? | `cuda_glue.zig:264` |
| `fn_attn_chunks_ahead` | `rows <= 16` and bf16 and unpaged | `ahead_rows=16`, smem `chunkSmem(2)` |
| `fn_attn_chunks_one` | `rows > 16` | smem `oneSmem()` |
| `fn_attn_chunks` (generic) | captured set | `cuda_launch.zig:212-231` |
| `fn_attn_fused` | `n > att_rows (256)` and `fusedAttentionOwned()` | `cuda_batch.zig:495`; group = `fused_rows` 1024 |
| `fn_attn_chunks_lanes` | `TF_FLASHNEXT_ATTN_LANES=1`, 2..32 lanes, `n<=256` | `cuda_batch_lanes.zig:69-79` |
| QSA sparse vs dense | `ends/ratio > index_top (512)` | `cuda_batch.zig:509` → `qsaScores`; NCH 5 vs 3 |
| chunk count | `cdiv(min(ends, dense_keys=2051), 512)` | `cuda_batch.zig:522` |

Drafter (`cuda_draft2_fwd.zig:1639-1670`): `warps = rows*(heads/kv_heads)`; `== 32` → `attnSplit` if `ATTN_SPLIT=1`,
else `attn_tile` (one CTA per KV head, 1024 threads, 16 KiB smem); otherwise the legacy warp-per-(row,head) `attn`.

Pre-existing dead/alternate attention: `attn2.zig` is **Metal only** (recorded Metal kernels, imported by
`engine_load.zig:146-147`), not a CUDA backend. `cuda_attn.zig` is a single-row verifier used by tools. `block_attention.cu`,
`prefill_attention.cu`, `window_attention.zig` belong to other families / the shared host.

---

## E. CUDA graphs

- **Decode**: `cuda_round_graph.zig` exists (`Cache.run` :60-90), keyed on owners/chunks/score-tiles/scratch. But the only call
  site requires **`e.collective != null`** — TP=2 (`cuda_prompt.zig:546`). On one GB10 `--tp 1`, decode rounds are always
  eager: `else try p.chunk(ids, 0, null)` :549. So `TF_FLASHNEXT_DECODE_GRAPH=1` is a **no-op on this box**.
- **Verify/draft windows**: `cuda_window_graph.zig`, `TF_FLASHNEXT_WINDOW_GRAPH`, **on by default**; a slot runs eagerly once,
  is captured on the second call (`:242-246`), re-recorded on key change, disabled for the life of the cache on capture failure.
  Capture is refused for probes (`cuda_batch.zig:597-600`) and layer timing (`cuda_layer_time.zig:143`).
- `TF_GRAPH_CACHE=0` is the global kill switch (docs/environment.md:130).

---

## F. Offload machinery

- **No expert/NVMe streaming on CUDA.** `--ssd-experts`, `--ple-on-ssd`, `--fsr-route-trace` are Metal-only
  (`server/cli.zig:132` `.native = builtin.os.tag == .macos`; `core/engine_api.zig:281-283` refuses them off Flash Next Metal).
  `fsr_*` / `replay_expert_source.zig` / `experts_pack.cu` describe the Metal FSR pool.
- **PLE n-gram table**: host-mapped shards, never on the device (`hostOnly` `cuda_prompt_wt.zig:395-401`), read per pass through
  `host_pool.zig` workers (`cuda_ple_pool.zig`). Knobs: `TF_FLASHNEXT_WARM_AHEAD`, `TF_FLASHNEXT_TABLE_WARM`,
  `TF_FLASHNEXT_PLE_IO`.
- **KV paging**: `cuda_kv_lanes.zig` + `cuda_pages.zig`, opt-in via `TF_FLASHNEXT_KV_PAGES=n`; ledger says Flash Next's is
  "experimental" (`docs/optimization-ledger.md:292`).
- **Disk tier**: `cuda_disk.zig` — kept prompt states only (`--learn-conversations`), `one_rank = true` (:66). Page-cache drop
  after load: `dropPageCache` `cuda_prompt_wt.zig:98-100`, called at `cuda_lanes_load.zig:108`.
- **Host KV tier**: not implemented for fnext (ledger row 7 = `open`).

---

## G. Things present but unreachable on this configuration

1. `TF_FLASHNEXT_DECODE_GRAPH` — gated on TP=2 (E above).
2. The affine-4 packed expert path (`experts_up/down`, `fexp` tiles, `RowsExperts`) — the FP4 checkpoint takes `native_experts`.
3. `TF_NVFP4_DECODE_WARPS=1` symbols and `experts_waves` half/waves forms — live but only exercised at decode shapes.
4. `TF_FLASHNEXT_PROMPT_EXPERT_TILE` — off by default; the tile kernel is fully built (`nvfp4_experts_tile`).
5. `TF_FLASHNEXT_ATTN_LANES` — off by default and additionally requires 2–32 lanes at `n<=256`.
6. `TF_FLASHNEXT_KV_PAGES` — off and refuses `kv8`, media and lanes.
7. fp8 KV + paged + lanes are mutually exclusive by refusal, not by priority.

## Uncertainties (could not determine statically)
- Whether a captured Triton kernel set is shipped for GB10 (would flip every `*Owned()` fast path off). The lane code suggests
  not (`TENSORFOLD_CUDA_KERNELS` unset ⇒ own glue), but the bundled-set resolution lives outside this family.
- Whether `cuda/promptTile(.nvfp4)` returns 64 or another tile — the call site only compares `!= 16`.
- Real cost of the per-round `vocab*2` logits download at `cuda_prompt.zig:552-553` (~496 KB × streams/round): no device-side
  sampling is used on that path even though `TF_FLASHNEXT_SAMPLED_HEADS` and the sampled-head kernels exist for drafts.
