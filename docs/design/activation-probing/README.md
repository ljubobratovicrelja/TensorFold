# What Qwen3.8-Flash-Next-4×512 does inside, on five real tasks

Five requests — prose, engineering reasoning, planning, an agentic shell instruction, and code — served one
at a time by `tensorfold-native serve` on the GB10, with two flag-gated probes in the engine
(`TF_FLASHNEXT_EXPERT_DUMP`, `TF_FLASHNEXT_HIDDEN_DUMP`, engine module `cuda_probe.zig`). The dumps are
the raw bytes; `plot_activations.py` turns them into the seven figures in `plots/`.

Everything here was measured on this box, on the served model, on 2026-11-19. Nothing was re-run in
PyTorch: the experts alone are ~225 GiB in bf16, so the engine *is* the only way in.

## The five tasks

| dir | task | prompt tokens | generated | wall | finish |
| --- | --- | ---: | ---: | ---: | --- |
| `data/a_prose` | ~600-word descriptive passage (rocky shore at dawn) | 156 | 1060 | 31.9 s | stop, 842 words |
| `data/b_engineering` | why a memory-bound kernel stalls at 186 GB/s of ~231 on GB10 | 363 | 1100 | 29.3 s | length |
| `data/c_planning` | dependency-ordered plan for an 11-day migration | 227 | 1100 | 25.8 s | length |
| `data/d_agentic` | shell command sequence to diagnose a broken import | 249 | 1000 | 22.9 s | length |
| `data/e_coding` | `merge_intervals` + self-test + complexity | 207 | 835 | 14.2 s | stop |

Thinking was off (`chat_template_kwargs.enable_thinking=false`), temperature 0.6, `--context 8192`,
`--kv-cache fp8`, MTP drafting on, no DFlash2 drafter. Decode ran at 34–60 tok/s.

## What was tapped and why

| tap | where | why |
| --- | --- | --- |
| top-11 expert ids + gate weights, every layer, every pass | `Pass.chunk` right after `moeBlockKeep`, `Scratch.picks`/`.wts` | the routing decision itself: who ran, with what weight |
| full 512+1 router logits, every layer, prompt pass, first 128 rows | same site, `Scratch.router_l` | a real softmax over all experts, so entropy is not a top-k approximation |
| post-norm MoE-input hidden row (2560 bf16), every layer, prompt pass | same site, `Scratch.mixed` | the same 32 tokens at 48 depths: norms, cosine drift, CKA, channel outliers |

Not tapped, and why: attention scores/maps (the pass never materialises a score matrix it could copy
without a second kernel), per-head outputs (the attention kernels write pooled rows only), and logit-lens
(the head is quantized Q4 and a vocab projection per layer is a 152k×2560 matmul a layer — 6× the pass's
own work; the `Scratch.mixed` rows are kept, so a logit lens is possible offline but not from this dump).
Expert *inputs* are not tapped either: 11 experts × 2560 × 2 B a token a layer is ~2.7 MB a token, and
`cuda_probe.zig` deliberately stays at tens of kilobytes a layer.

## Artifacts

```
data/<category>/prompt_source.md      the prompt as sent      data/<cat>/output.md     the reply
data/<category>/experts/              TF_FLASHNEXT_EXPERT_DUMP: p0000.bin ... + router_logits.bin + manifest.json
data/<category>/hidden/               TF_FLASHNEXT_HIDDEN_DUMP: hidden.bin + hidden_manifest.json
plots/01_layer_expert_heatmap.png     layer × expert, z against a uniform load, one row a category
plots/01b_expert_totals.png           the global hot set (top 40 experts) and each category's spread
plots/02_expert_load_balance.png      load entropy, hottest/mean, dead-expert %, shared gate, per layer
plots/03_router_entropy.png           full-512 softmax entropy and top-11 mass, per layer
plots/04_hidden_norm_drift.png        hidden norm, cosine(L, L+1), cosine(L, 47)
plots/05_layer_cka.png                linear CKA between all 96 layer pairs, per category
plots/06_channel_magnitudes.png       per-channel |x| against the layer's own mean: outlier channels
plots/07_category_separation.png      JSD and cosine between the five categories' load matrices
summary.json                          every number quoted below, machine-readable
collect_task.py  run_category.sh      the client and the one-category boot/serve/stop loop
logs/<category>.log                   the engine's own log for that request
plot_activations.py                   the only plotting script; matplotlib + numpy, no other dependency
```

Byte layouts are in `docs/design/flashnext-activation-dumps.md` in the engine repo. Short version:
`p<NNNN>.bin` = 20-byte header (magic `TFEX`, rows, pos0, layers, slots) then `[layers][rows][slots]` u32
ids and the same shape f32 weights; `hidden.bin` = 28-byte header (magic `TFHD`, layers, keep, dims,
rows_of_pass, k, pos0) then `[layers][keep][dims]` bf16.

Slot 10 of every row is the always-on shared expert (id 512) — the router's 513th logit row — with a gate
weight of 0.47 at layer 0 and ~0.10–0.15 in the middle of the stack. It is counted out of every
512-expert statistic here and reported separately in panel 4 of `02_expert_load_balance.png`.

## Reproduce

```bash
cd /home/relja/Projects/tf-wt/dflash2
zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc          # with the cuda_probe.zig change in place
cd ~/Projects/tf-wt/dflash2/docs/design/activation-probing
./run_category.sh a_prose 1200 170                              # boots the engine, serves one request, stops it
./run_category.sh b_engineering 1100 170                        # ... one at a time: one model fits the box
/home/relja/Projects/tf-wt/dflash2/.venv-plot/bin/python plot_activations.py \
    --data data --out plots --summary summary.json
```

`run_category.sh` sets `TF_FLASHNEXT_WINDOW_GRAPH=0` (a probe read-back synchronises the stream, which a
capture cannot hold), `TF_FLASHNEXT_EXPERT_DUMP`, `TF_FLASHNEXT_HIDDEN_DUMP` and
`TF_FLASHNEXT_ROUTER_LOGITS=128`, on port 8099. Each run boots its own engine because the probe directory
is fixed at engine init; a run costs ~65 s of load plus the generation.

Unset, the probes cost nothing: the same binary served a request with no `probe:` line in its log and no
directory created.

## What the numbers say

**Routing is close to uniform, and the layers that are not are few.** Over 1.2–1.6k generation tokens a
category, a layer routes 10 experts a token, so a uniform load would be ~30 hits an expert. Mean normalised
load entropy (1.0 = every expert chosen equally) is 0.70 for prose and 0.78–0.82 for the other four, and
24% (coding) to 46% (prose) of the 512 experts are never selected at all during a run. The hottest expert
in a layer takes 33–49× the mean, yet the top 10 experts hold only 4.4–6.5% of all routed mass (uniform
would be 1.95%) and no single expert exceeds 0.39% of it: this model routes broadly, not to a few
specialists.

**Different layers are hot in different categories.** By hottest/mean concentration: prose → layers 35,
10, 11, 19, 27; engineering → 44, 39, 9, 25, 41; planning → 15, 31, 43, 41, 17; agentic → 39, 43, 31,
32, 15; coding → 31, 15, 30, 36, 47. Layers 15 and 31 recur across the four procedural tasks and never in
prose; prose's concentration is the highest anywhere (49×, layer 35) yet its load is the least even.

**The router's full distribution flattens with depth.** Over the prompt pass's first 128 rows, the softmax
entropy over 512 experts rises from 0.83–0.86 of log(512) at layer 0 to 0.95–0.96 at layer 46, and the
mass on the eleven chosen experts falls from ~0.27 to ~0.10, with dips at layers 22 and 37 and a sharp
peak at layer 21 (0.34). Early layers commit; late layers barely choose.

**The stack builds magnitude with depth and has two visible resets.** The post-norm hidden norm falls to
19.6 at layer 1, then grows from ~20.7 to 57.1 at layer 45 (30.0 at layer 30, 53.9 at 46, 47.1 at 47): a
2.8× buildup of activation magnitude. Cosine between consecutive layers' row means is 0.82–0.83 on average
but 0.39 at 0→1, 0.60 at 1→2, 0.65 at 22→23 and 0.61 at 38→39 — the last two are the layers where the
router's mass also dips. Adjacent-layer linear CKA is 0.67 mean, 0.10 at 0→1, with no block structure:
every layer is a smooth rotation of its neighbour, except at 0, 1–2 and 38–39.

**The five categories separate, but along one axis.** Jensen–Shannon divergence between the layer×expert
load matrices is 0.21 (planning vs engineering) to 0.53 (prose vs coding); cosine similarity 0.05–0.47. The
procedural four (engineering, planning, agentic, coding) sit at 0.21–0.32 JSD from each other; prose is
0.39–0.53 from all of them and shares only 0.05–0.14 cosine. Prose is the outlier on every measure — the
highest concentration, the most dead experts, the flattest reverse pattern of shared-expert weight — while
the four task-shaped prompts are hard to tell apart by routing alone.

**Outlier channels exist but do not dominate.** The largest channel in a layer sits at 4.1–4.3× that
layer's mean magnitude (median over layers; 9.7× at the extreme), the same handful of channels recur at
every depth and in both categories (728, 2238, 192, 295), and the top 1% of the 2560 channels carry only
2.8% of the row's absolute mass. This is nothing like the "massive activation" picture of a few channels
carrying most of the norm: the post-norm hidden rows here are broad, not spiky.

**The shared expert's gate is largest at the edges.** 0.47 at layer 0, falling to 0.10–0.15 through the
middle, back to 0.38 at layer 47 — the always-on path is weighted where the routed path is least confident.

## What is inference, not measurement

* **Only one request per category.** n = 5 tasks, one sample each; a category difference of a few
  points of JSD could be that prompt, not that task type. The prose/rest split is large enough to survive
  that (0.39+ JSD against 0.21–0.32 within the rest) but the ordering *inside* the procedural four is not
  established.
* **Generation rows are verify windows, not one row a token.** With MTP drafting the pass covers the
  drafted window (2–7 rows here), so "token-slots" are draft candidates that were partly rejected. Routing
  statistics are unaffected in kind but are not exactly per-emitted-token.
* **The hidden capture is 32 prompt rows**, the last 32 of the fill pass, so norms/drift/CKA are over 32
  samples — enough for a per-layer mean, thin for CKA (a 32×2560 Gram matrix). CKA values should be read
  as relative, not absolute.
* **The warm-up passes are in the dumps.** The server measures its window costs on passes of 1..32 rows
  before serving; those are recorded too (about 160 of the ~350–700 files a category). The scripts take
  the widest pass as the prompt and everything after it as generation; a consumer that globs `p*.bin` and
  averages naively would include them.
* **Attention sinks and logit lens are not measured here** — the plots above say nothing about them.
* **A sync per pass.** The probe reads back once per pass; a decode round already synchronises for its
  logits, so the added wait is microseconds, but the timings in `timing.json` are under the flag and
  should not be quoted as the engine's throughput.
