#!/usr/bin/env python
"""The REFERENCE side of the bf16 drafter alignment test: the vendor torch model, on the GPU,
in bfloat16, over the generated fixture, dumping every stage.

Why bf16 on the GPU and not fp32 on the CPU (which is what `reference/run_round.py` in
~/Projects/dflash2-align does): the question here is not "does the drafter predict" but "do the two
implementations produce the same bytes from the same inputs". A fp32 reference answers a different
question — every bf16 kernel would differ from it by the rounding it was designed to do, and the
comparison would be a tolerance band, not a number.

Weights are loaded as they are stored in the checkpoint (bf16) and never upcast; the target's
embedding table and `lm_head` are read from the target checkpoint, also bf16. Two torch defaults
are pinned for the duration, both because they would otherwise fold a second, undocumented
precision into the reference and be invisible in the dump:

  * `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` — a bf16 GEMM's
    reduction stays fp32 (bf16 accumulate is a split-k shortcut);
  * `torch.backends.cuda.matmul.fp32_precision`/`cudnn` left alone, but TF32 is irrelevant to a
    bf16 matmul and is not touched.

Stages are named exactly as `cuda_draft2_check.zig` names them, and are dumped as f32 widenings of
the tensor the stage actually holds (bf16 -> f32 is exact, so bit-identity of the bf16 is bit
identity of the dump; ids and logits are f32 on the device side already). `stages.json` records each
stage's source dtype, so a comparer never has to guess.

  python tools/dflash2/align_bf16/reference_bf16.py <fixture-case-dir> <out-dir>
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

VENDOR_MODEL = os.path.expanduser("~/Projects/refimpls/dflash/dflash/model.py")
CKPT_SNAP = (
    "/home/relja/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/"
    "snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6"
)
TARGET_SNAP = (
    "/home/relja/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/"
    "snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300"
)
TARGET_EMBED_KEY = "model.language_model.embed_tokens.weight"
TARGET_LMHEAD_KEY = "lm_head.weight"

TAPS = [5, 15, 25, 35, 45]
HIDDEN = 2560
BLOCK = 8
MASK_TOKEN = 248077
VOCAB = 248320
TOPK = 16


# ---------------------------------------------------------------------------------------------
# model, inputs
# ---------------------------------------------------------------------------------------------


def load_vendor():
    spec = importlib.util.spec_from_file_location("vendor_dflash_model", VENDOR_MODEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(vendor, device):
    from transformers.models.qwen3 import Qwen3Config

    with open(os.path.join(CKPT_SNAP, "config.json")) as f:
        cfg = Qwen3Config(**json.load(f))
    model = vendor.DFlash2DraftModel(cfg)

    from safetensors.torch import load_file

    sd = load_file(os.path.join(CKPT_SNAP, "model.safetensors"))
    remap = {}
    for k, v in sd.items():
        nk = k
        if k.startswith("candidate_selector.") and not k.endswith(".weight"):
            nk = k + ".weight"
        remap[nk] = v  # bf16, as stored: no upcast, so no second rounding of the weights
    missing, unexpected = model.load_state_dict(remap, strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    model.eval()
    return model.to(device=device, dtype=torch.bfloat16)


def load_target_aliases(device, dtype):
    """embed_tokens + lm_head are not in the drafter checkpoint: the vendor reads them off the target."""
    from safetensors import safe_open

    with open(os.path.join(TARGET_SNAP, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    out = {}
    for name, key in (("embed", TARGET_EMBED_KEY), ("lmhead", TARGET_LMHEAD_KEY)):
        shard = os.path.join(TARGET_SNAP, wm[key])
        with safe_open(shard, framework="pt") as f:
            out[name] = f.get_tensor(key).to(device=device, dtype=dtype)
    return out["embed"], out["lmhead"]


class LMHead(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


def load_fixture(d: str):
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    assert man["taps"] == TAPS and man["hidden"] == HIDDEN and man["block"] == BLOCK, man
    rows, row_bytes = man["rows"], man["row_bytes"]
    ring = np.fromfile(os.path.join(d, "ring.bin"), dtype="<u2").reshape(rows, row_bytes // 2)
    ids = np.fromfile(os.path.join(d, "ids.bin"), dtype="<u4")
    return man, ring, ids


def stage_bytes(t: torch.Tensor):
    """The f32 widening of a stage, as the `.npy` holds it, and the dtype it came from."""
    if t.dtype == torch.bfloat16:
        return t.detach().float().cpu().numpy(), "bf16"
    if t.dtype in (torch.float32,):
        return t.detach().float().cpu().numpy(), "f32"
    if t.dtype in (torch.int64, torch.int32):
        return t.detach().float().cpu().numpy(), "u32"
    raise AssertionError(f"stage dtype {t.dtype}")


# ---------------------------------------------------------------------------------------------
# the round, stage by stage, named as the engine names them
# ---------------------------------------------------------------------------------------------


def run_round(vendor, model, ctx_x, block_ids, pos_ids, embed_w, lm_head, anchor, stages):
    hd = model.layers[0].self_attn.head_dim
    ctx_len = ctx_x.shape[1]
    q_len = block_ids.shape[1]

    noise_emb = F.embedding(block_ids, embed_w)
    stages["embed"] = noise_emb[0]

    cos, sin = model.rotary_emb(noise_emb, pos_ids)

    # fc_ctx : the five taps of a position -> one hidden row
    ctx_hidden = model.hidden_norm(model.fc(ctx_x))
    stages["fc_ctx_rows"] = ctx_hidden[0]

    hidden = noise_emb
    for i, layer in enumerate(model.layers):
        att = layer.self_attn
        residual = hidden

        hs, attn_kernel = layer.attention_conv.prepare(layer.input_layernorm(hidden))
        stages[f"conv_attn_in_{i}"] = hs[0]

        # the block's own q/k/v, raw: `qkv_i` on the engine side is the projection before the
        # per-head norm and before the rotation
        q_raw = att.q_proj(hs)
        k_raw = att.k_proj(hs)
        v_raw = att.v_proj(hs)
        stages[f"qkv_{i}"] = torch.cat([q_raw, k_raw, v_raw], dim=-1)[0]

        q = att.q_norm(q_raw.view(1, q_len, -1, hd)).transpose(1, 2)
        k_noise = k_raw
        v_noise = v_raw

        # the prefix's keys and values, through the same projections, normed and rotated at their
        # own positions
        k_ctx = att.k_proj(ctx_hidden)
        v_ctx = att.v_proj(ctx_hidden)
        k = torch.cat([k_ctx, k_noise], dim=1).view(1, ctx_len + q_len, -1, hd)
        v = torch.cat([v_ctx, v_noise], dim=1).view(1, ctx_len + q_len, -1, hd)
        k = att.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q_r, k_r = vendor.apply_rotary_pos_emb(q, k, cos, sin)
        mask = None
        if att.is_causal or att.sliding_window is not None:
            mask = vendor._attention_mask(q_r, k_r, is_causal=att.is_causal,
                                          sliding_window=att.sliding_window)
        attn_output, _ = vendor.ALL_ATTENTION_FUNCTIONS["sdpa"](
            att, q_r, k_r, v, mask, dropout=0.0, scaling=att.scaling,
            sliding_window=att.sliding_window,
        )
        attn_output = attn_output.reshape(1, q_len, -1)
        stages[f"attn_{i}"] = attn_output[0]           # before o_proj

        attn_proj = att.o_proj(attn_output)
        stages[f"attn_proj_{i}"] = attn_proj[0]        # after o_proj

        conv_attn_out = layer.attention_conv.finish(attn_proj, attn_kernel)
        hidden = residual + conv_attn_out
        stages[f"conv_attn_out_{i}"] = hidden[0]       # the residual sum, as the engine dumps it
        residual = hidden

        hs, mlp_kernel = layer.mlp_conv.prepare(layer.post_attention_layernorm(hidden))
        stages[f"conv_mlp_in_{i}"] = hs[0]

        mlp_out = layer.mlp(hs)
        stages[f"mlp_{i}"] = mlp_out[0]

        conv_mlp_out = layer.mlp_conv.finish(mlp_out, mlp_kernel)
        hidden = residual + conv_mlp_out
        stages[f"conv_mlp_out_{i}"] = hidden[0]
        stages[f"block_hidden_{i}"] = hidden[0]

    final = model.norm(hidden)
    stages["final_norm_block"] = final[0, 1:, :]       # the seven proposing rows

    logits = model.compute_logits(final, lm_head)
    stages["logits_full"] = logits[0, 1:, :]           # all 248,320 a row, fp32

    blk_logits = logits[0, 1:, :]
    topv, topi = torch.topk(blk_logits, TOPK, dim=-1, sorted=True)
    stages["logits_top16_ids"] = topi
    stages["logits_top16_vals"] = blk_logits.gather(-1, topi)

    # the selector: unary[c] + <A[pred] * project(h_slot), B[c]>, greedy
    sel = model.candidate_selector
    proj = sel.hidden_projection(final[0, 1:, :])
    pred = torch.tensor([anchor], dtype=torch.long, device=final.device)
    path, scores_all = [], []
    for pos in range(proj.shape[0]):
        scores = topv[pos] + torch.einsum(
            "r,kr->k",
            sel.predecessor_codebook(pred[0]) * proj[pos],
            sel.successor_codebook(topi[pos]),
        )
        scores_all.append(scores)
        idx = int(torch.argmax(scores))
        pred = topi[pos, idx:idx + 1]
        path.append(int(pred[0]))
    stages["selector_scores"] = torch.stack(scores_all)
    stages["proposals"] = torch.tensor(path, dtype=torch.int64, device=final.device)

    return {"mask_all_visible": bool(mask.all()) if mask is not None else True}


def main(argv):
    fx_dir, out_dir = argv[1], argv[2]
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    device, dtype = "cuda", torch.bfloat16

    man, ring, ids = load_fixture(fx_dir)
    P = man["pos_end"] - 1
    ctx_len = min(man["window"], P)
    lo = P - ctx_len
    ctx_pos = np.arange(lo, lo + ctx_len)
    ctx_rows = ring[ctx_pos % man["rows"]]
    ctx_x = torch.from_numpy(ctx_rows.copy()).view(torch.bfloat16).to(device)  # (ctx_len, 12800)
    ctx_x = ctx_x[None]

    vendor = load_vendor()
    model = build_model(vendor, device)
    embed_w, lmhead_w = load_target_aliases(device, dtype)
    lm_head = LMHead(lmhead_w).to(device)

    anchor = int(ids[P])
    block_ids = torch.tensor([[anchor] + [MASK_TOKEN] * (BLOCK - 1)], dtype=torch.long, device=device)
    pos_ids = torch.arange(lo, lo + ctx_len + BLOCK, dtype=torch.long, device=device)[None]

    print(f"[ref] fixture {fx_dir}  P {P}  ctx_len {ctx_len}  anchor {anchor} "
          f"block {block_ids[0].tolist()[:1]}+mask x{BLOCK - 1}  positions {lo}..{lo + ctx_len + BLOCK - 1}",
          flush=True)

    args = (vendor, model, ctx_x, block_ids, pos_ids, embed_w, lm_head, anchor)
    # one warmup round: cuBLAS's first call on a shape is not the arithmetic
    run_round(*args, {})
    stages = {}
    t0 = time.perf_counter()
    info = run_round(*args, stages)
    torch.cuda.synchronize()
    round_ms = (time.perf_counter() - t0) * 1e3

    # the vendor's own forward, unmodified, as a self-check of the instrumented walk above
    with torch.no_grad():
        ref_final = model(position_ids=pos_ids, noise_embedding=F.embedding(block_ids, embed_w),
                          target_hidden=ctx_x, past_key_values=None, use_cache=False)
    drift = float((ref_final[0, 1:, :].float() - stages["final_norm_block"].float()).abs().max())

    written = []
    os.makedirs(os.path.join(out_dir, "tensors"), exist_ok=True)
    for name, t in stages.items():
        arr, src = stage_bytes(t)
        np.save(os.path.join(out_dir, "tensors", f"{name}.npy"), np.ascontiguousarray(arr))
        written.append({"stage": name, "shape": list(arr.shape), "src_dtype": src})

    doc = {
        "impl": "vendor-torch",
        "device": "cuda",
        "dtype": "bfloat16",
        "torch": torch.__version__,
        "reduced_precision_reduction": False,
        "checkpoint": CKPT_SNAP,
        "target": TARGET_SNAP,
        "round": {
            "position": P,
            "anchor": anchor,
            "mask": MASK_TOKEN,
            "ctx_len": ctx_len,
            "ctx_lo": lo,
            "block": BLOCK,
            "window": man["window"],
            "proposals": [int(x) for x in stages["proposals"].tolist()],
            "mask_all_visible": info["mask_all_visible"],
            "round_ms": round_ms,
            "vendor_forward_max_abs_diff": drift,
        },
        "n_stages": len(written),
        "stages": written,
    }
    with open(os.path.join(out_dir, "stages.json"), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"[ref] {len(written)} stages, round {round_ms:.1f} ms, vendor-forward drift {drift:.3e}, "
          f"proposals {doc['round']['proposals']}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
