# DFlash2 drafter: weight-only FP8 export

Turns the bf16 drafter (`incoai/Qwen3.8-Flash-Next-DFlash2`, one `model.safetensors`, 81
tensors, 1,206,140,008 bytes) into a stand-alone block-FP8 checkpoint at
`checkpoints/dflash2-fp8/`, and checks the result.

```sh
SNAP=~/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6

python3 tools/dflash2/quantize/quantize_drafter.py --self-test
python3 tools/dflash2/quantize/quantize_drafter.py --src "$SNAP" --out checkpoints/dflash2-fp8
python3 tools/dflash2/quantize/verify.py --fp8 checkpoints/dflash2-fp8 --src "$SNAP"
```

CPU only, numpy and the standard library: no torch, no safetensors package, no GPU, no build.
The export is deterministic — the same source gives the same `model.safetensors` bytes and
sha256, and `manifest.json` carries no wall-clock timestamp.

## The format (a contract with the loader)

Exactly the repo's block-FP8 layout, `zig/src/cuda/fp8.zig` + `fp8.Host`:

| tensor | dtype | shape |
| --- | --- | --- |
| `X.weight` | `F8_E4M3` | `[n, k]`, row-major e4m3 codes |
| `X.weight_scale_inv` | `F32` | `[ceil(n/128), ceil(k/128)]` |

For each 128 x 128 block of the bf16 source, with `amax` the block's largest magnitude:

```
weight_scale_inv = amax / 448            # 1.0 for a block of zeros (never 0, never NaN)
code             = round_to_nearest_even(w / weight_scale_inv), saturating at 448
dequant          = code * weight_scale_inv
```

So the stored f32 is the **multiplier** a block's codes are read with — small values
(`amax/448`, ~1e-4 for these weights), which is the direction `fp8.reference`
(`e4m3(code) * s`), the qmmf `FP8G` kernel (`acc * s`) and
`tools/step5_fp8_to_mlx.py:fp8_dequant` (`w.float() * s`) all take it. The multiplier that
turns a weight into codes is its reciprocal, `448 / amax`.

Rounding is `core.quant.fp8.toE4m3`'s: the step comes from the binade, floored at the
subnormals' `2^-9` below `2^-6`, ties go to the even mantissa (`1.0625 -> 1.0`,
`1.1875 -> 1.25`), and it saturates at ±448. `--self-test` pins those, checks every one of
the 127 finite codes survives decode/round/encode, and checks 200k random values each land on
a nearest grid point. No `0x7f`/`0xff` (NaN) code is ever written; a non-finite block `amax`
aborts the export rather than writing a NaN scale.

Quantized (35 tensors, 5 layers x 7): `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`,
`self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`. Copied through as BF16, byte for byte
(46 tensors): `fc.weight`, both convs' `base_kernel` and `kernel_projection.weight`, every
norm (`hidden_norm`, `norm`, `input_layernorm`, `post_attention_layernorm`, `q_norm`,
`k_norm`), and the candidate selector (`hidden_projection.weight`,
`predecessor_codebook`, `successor_codebook`). `--groups` changes the list.

The output directory is complete: `model.safetensors`, `config.json` (the source's object with
`dflash_config.precision = "fp8_w8a16"`, `quantized_weight_dtype`, `quantized_block` merged in,
every existing key kept), `manifest.json`, `SHA256SUMS`.

## How good the quantized weights are

Every matrix lands at **relL2 2.64–2.65 %** against bf16 and every one of the 26,000 blocks is
above 2 % (worst block 2.73 %). That is e4m3 itself, not a defect of this export: three
mantissa bits mean a half-step of `2^-4` of each binade — ~3.6 % RMS for a value uniform inside
its binade, ~2.6 % over weights that spread across binades. There is no threshold below that
except fewer bits of weight to begin with. Details per matrix are in `manifest.json`; the split
by name, including any matrix that strays, is what `verify.py` prints.

## What `verify.py` checks

* `SHA256SUMS` and `manifest.json`'s hashes are the files on disk; the container's offsets are
  contiguous, 8-byte aligned, cover the buffer and nothing past it.
* All 35 matrices are `F8_E4M3` `[n, k]` with a `F32` `[ceil(n/128), ceil(k/128)]` scale, every
  scale finite and positive, no NaN code.
* Per matrix: relL2 against the bf16 source, the worst block's relL2, blocks above 2 % — and
  that `manifest.json` records those same numbers.
* Round-trip idempotence: re-quantizing the dequantized weight reproduces the stored codes and
  the stored scales bit for bit.
* Every BF16 passthrough tensor is byte-identical to the source's, and `config.json` kept every
  key of the source's.

Exit status is non-zero on any failure. `--fail-over-rel-l2` (default 5 %) is the per-matrix
ceiling; the 2 % block count is reported, not enforced, because e4m3's own floor is above it.

## Not verified here

* No engine run: the loader in `zig/src/cuda` is the other agent's half of this contract, and
  nothing in this directory builds or runs CUDA. `verify.py` proves the arithmetic and the
  bytes, not that the kernel reads them the way `fp8.zig` says it will.
* No accept-rate or perplexity measurement of the drafter. The 2.65 % weight error bounds
  nothing about draft quality; that needs the target model on a GPU.
* The safetensors container was checked by this directory's own parser and by structural
  invariants, not by the `safetensors` package (not installed here).
