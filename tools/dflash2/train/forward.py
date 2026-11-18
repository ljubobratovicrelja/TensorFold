"""DFlash2 drafter forward in torch, derived from the Zig/CUDA round, not from the old
`tools/dflash2/analysis/reference.py` (which scores 0/7 and is structurally wrong).

This file's only job is to reproduce the *shipped checkpoint's* proposals — i.e. what
`zig/src/families/flashnext/cuda_draft2_fwd.zig` + `zig/kernels/cuda/qwen35_draft.cu` compute —
closely enough that acceptance on held-out fixtures matches. Every choice below is read off the
CUDA kernel it imitates; where the kernel rounds to bf16 mid-computation, so does this.

THE COMPUTATION, exactly as the engine runs it
==============================================
Checkpoint `~/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a…`
(81 tensors, all bf16) — tensor contract in `cuda_draft2_weights.zig`:

  fc.weight                              [2560, 12800]   (= 5 taps x 2560, in tap order)
  hidden_norm.weight                     [2560]          (the fuse's RMSNorm)
  norm.weight                            [2560]          (final norm, before head/selector)
  candidate_selector.hidden_projection.weight  [256, 2560]
  candidate_selector.predecessor_codebook      [248320, 256]
  candidate_selector.successor_codebook        [248320, 256]
  layers.{0..4}.*  (15 each):
    input_layernorm.weight               [2560]
    post_attention_layernorm.weight      [2560]
    self_attn.q_proj.weight              [4096, 2560]    (32 heads x 128)
    self_attn.k_proj.weight              [1024, 2560]    (8 kv heads x 128)
    self_attn.v_proj.weight              [1024, 2560]
    self_attn.o_proj.weight              [2560, 4096]
    self_attn.q_norm.weight              [128]
    self_attn.k_norm.weight              [128]
    mlp.gate_proj.weight                 [7680, 2560]
    mlp.up_proj.weight                   [7680, 2560]
    mlp.down_proj.weight                 [2560, 7680]
    attention_conv.base_kernel           [2, 2, 2560]    (branch, lag, channel)
    attention_conv.kernel_projection.weight [640, 2560]  (2 branches x 2 lags x 160 groups)
    mlp_conv.base_kernel                 [2, 2, 2560]
    mlp_conv.kernel_projection.weight    [640, 2560]

Config (drafter config.json `dflash_config`): block 8, group 16, conv_kernel 2, mask 248077,
selector_rank 256, selector_top_k 16, taps [5,15,25,35,45]; hidden 2560, ffn 7680, heads 32,
kv_heads 8, head_dim 128, vocab 248320, eps 1e-6, rope theta 10000, sliding window 2048,
is_causal false.

A round is at eos position `P`, with a prefix of `ctx_len = min(window, P)` tap rows covering
absolute positions `[lo, P)` where `lo = P - ctx_len`:

1. FUSE (`tf_fxd2_fuse` + `tf_fxd2_rmsnorm`), once a round, shared by all 5 layers:
     ctx[i] = rmsnorm( bf16( taps[i] @ fc.weight^T ), hidden_norm, eps )
   `taps[i]` is the ring row of position `lo+i`: 5 x 2560 bf16, tap k at columns [k*2560, (k+1)*2560),
   and `ring.bin` already holds the *mean of the target's four hyper-connection streams* ("mean4":
   `tf_fxd2_mean4`, fp32 sum over the 4 streams then one rounding to bf16). The fc GEMM is rounded
   to bf16 once (the kernel's `acc[j] = rbf(s)`) and the RMSNorm then reads those bf16 values.
   The drafter runs on the PREFIX's taps only — the block's own 8 rows never come from taps; they
   are the target's embedding of the anchor and 7 mask tokens.

2. BLOCK (`embed`, then 5 x `layer`): ids = [anchor] + [mask]*7, h = embed_table[ids];
   the block's rows sit at absolute positions P..P+7.
   Each layer, in this order (residuals are bf16 buffers, the conv kernel adds RES itself):

     x    = rmsnorm(h, input_layernorm)                      # [8,2560]
     dyn  = x @ attn_conv.kernel_projection^T                # [8, 640] = [branch][lag][group]
     xin  = dynconv(x, dyn, attn_conv.base_kernel, branch=0, res=None)
     q    = norm_rope(xin @ q_proj^T, q_norm, 32 heads, pos=P+r)
     k    = norm_rope(xin @ k_proj^T, k_norm,  8 heads, pos=P+r)
     v    = xin @ v_proj^T
     # prefix k/v, per prefix row i (chunked only for scratch size, arithmetically identical):
     kc[i] = norm_rope(ctx[i] @ k_proj^T, k_norm, 8 heads, pos=lo+i)   # its OWN absolute position
     vc[i] = ctx[i] @ v_proj^T
     att   = attention(q, kc+vc, k+v)                        # see the visibility rule below
     o     = att @ o_proj^T                                  # o_proj wraps the sublayer OUTPUT
     h2    = dynconv(o, dyn, attn_conv.base_kernel, branch=1, res=h)   # residual inside the conv
     x     = rmsnorm(h2, post_attention_layernorm)
     dyn   = x @ mlp_conv.kernel_projection^T
     xin   = dynconv(x, dyn, mlp_conv.base_kernel, branch=0, res=None)
     act   = silu(xin @ gate^T) * (xin @ up^T)               # bf16, one rounding
     o     = act @ down^T
     h     = dynconv(o, dyn, mlp_conv.base_kernel, branch=1, res=h2)

   ATTENTION VISIBILITY (the non-obvious part, `tf_fxd2_attn`/`tf_fxd2_attn_tile`): block attention
   is bidirectional over the 8 block rows (is_causal false), and — because the reference builds its
   mask with `sliding_window` anchored on each row's own position — block row `m` does NOT see the
   first `m` prefix columns. It sees prefix positions `[lo+m, P)` and then all 8 block rows. So
   row 0 sees all `ctx_len` prefix rows, row 7 sees `ctx_len-7`. This is not a bug in the kernel:
   it is measured against the reference's own mask, and getting it wrong is the single biggest
   per-block divergence.
   softmax/scale: scale = 1/sqrt(128), softmax in fp32, output rounded to bf16.

   DYNAMIC CONV (`tf_q35d_dyn_conv`): y[t,c] = sum over lag l in {0,1} of
   (base[branch,l,c] + dyn[t,branch,l,c//16]) * x[t-l,c], x[-1]=0, and for branch=1 the residual
   `res[t,c]` is added last. The kernel rounds every product to bf16 and the running total after
   every add (`rbf`), which this file reproduces element-wise. Never crosses a block boundary.
   NORM_PLACEMENT: norms are pre-norm on the sublayer input; both convs are (input-side conv before
   the projection, output-side conv after the *output* projection, residual added inside it).

3. HEAD (`glm_mm16_16` + `tf_fxd2_topf32` + `tf_fxd2_headmerge`): the final `norm` is applied to
   block rows 1..7 ONLY (row 0 just re-predicts the anchor), then
     logits = hn @ lm_head^T          # fp32, the target's 248320 x 2560 head
     cand, unary = topk(logits, 16)   # descending, one table a row
   No softcap, no output multiplier.

4. SELECT (`tf_fxd2_select`, serial over the block's 7 steps):
     prev_token = anchor if t == 0 else cand[t-1][prev_index]
     hp    = hn[t] @ hidden_projection^T                  # fp32 (bf16 products, fp32 accum)
     ctxr  = predecessor_codebook[prev_token] * hp        # bf16 codebook value times fp32 hp
     score[c] = unary[t,c] + dot(ctxr, successor_codebook[cand[t,c]])   # fp32
     prev_index = argmax(score)  # strictly-greater comparisons -> the FIRST index wins a tie
     proposal[t] = cand[t][prev_index]
   proposal[t] is the token at absolute position P+1+t. Acceptance = longest common prefix of
   `proposal` with the target's own tokens at P+1..P+7 (a round advances 1 + accepted).

Weights stay bf16 and the arithmetic runs on cuda; nothing here writes the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

HIDDEN = 2560
FFN = 7680
VOCAB = 248_320
N_TAPS = 5
BLOCK = 8
MASK = 248_077
HEADS = 32
KV_HEADS = 8
HD = 128
GROUP = 16
KERNEL = 2
RANK = 256
TOP_K = 16
EPS = 1e-6
THETA = 10000.0
WINDOW = 2048

EMBED_NAME = "model.language_model.embed_tokens.weight"
HEAD_NAME = "lm_head.weight"


# --------------------------------------------------------------------------------------------
# Elementary ops, each one a kernel
# --------------------------------------------------------------------------------------------

def rbf(t: torch.Tensor) -> torch.Tensor:
    """Round an fp32 tensor through bf16 and back: the kernels' `rbf` (`__float2bfloat16_rn`)."""
    return t.to(torch.bfloat16).float()


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """`tf_fxd2_rmsnorm`: fp32 sum of squares of the bf16 inputs, then bf16(x*inv) * w, bf16 out."""
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (rbf(xf * inv) * w.float()).to(torch.bfloat16)


def norm_rope(x: torch.Tensor, w: torch.Tensor, heads: int, pos0: int, eps: float = EPS,
              theta: float = THETA) -> torch.Tensor:
    """`tf_q35d_norm_rope`: per-head RMSNorm over HD (weight `w`, length HD), then rotate-half RoPE
    at absolute positions pos0 .. pos0+T-1. Input/output [T, heads*HD] bf16."""
    t = x.shape[0]
    xh = x.view(t, heads, HD).float()
    inv = torch.rsqrt(xh.pow(2).mean(-1, keepdim=True) + eps)
    n = rbf(xh * inv) * w.float()                    # bf16(x*inv) * bf16(w), in fp32
    half = HD // 2
    j = torch.arange(half, device=x.device, dtype=torch.float32)
    freqs = theta ** (-j / half)                     # theta^(-j/(HD/2))
    ang = (pos0 + torch.arange(t, device=x.device, dtype=torch.float32))[:, None] * freqs[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]          # [T,1,half]
    x1, x2 = n[..., :half], n[..., half:]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rbf(out).view(t, heads * HD).to(torch.bfloat16)


def dyn_conv(x: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, branch: int,
             res: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`tf_q35d_dyn_conv`.

    x    [T, H] bf16 — the block's rows only (a conv never crosses a block boundary)
    dyn  [T, 2*2*groups] bf16 — the whole projection output, layout [branch][lag][group]
    base [2, 2, H] bf16 — [branch][lag][channel]
    The kernel rounds each product and each partial sum to bf16, so this does too.
    """
    t_dim, d = x.shape
    groups = d // GROUP
    xf = x.float()
    dy = dyn.float().view(t_dim, 2, KERNEL, groups)
    total = torch.zeros(t_dim, d, device=x.device, dtype=torch.float32)
    for lag in range(KERNEL):
        src = torch.zeros_like(xf)
        if lag == 0:
            src = xf
        else:
            src[lag:] = xf[:-lag]
        b = base[branch, lag].float()[None, :]
        total = rbf(total + rbf(b * src))
        g = dy[:, branch, lag, :].repeat_interleave(GROUP, dim=-1)
        total = rbf(total + rbf(g * src))
    if res is not None:
        total = rbf(total + res.float())
    return total.to(torch.bfloat16)


# --------------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------------

class Drafter:
    """The 81-tensor drafter checkpoint plus the two tensors it does not ship (the target's
    embedding table and LM head). Read-only; every tensor is loaded once, bf16, on `device`."""

    def __init__(self, snap: str | Path, target: str | Path, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16) -> None:
        self.device = torch.device(device)
        self.snap = Path(snap)
        self.cfg_raw = json.loads((self.snap / "config.json").read_text())
        self._f = safe_open(str(self.snap / "model.safetensors"), "pt")
        self._cache: dict[str, torch.Tensor] = {}
        self.top = {
            "fc": self.get("fc.weight"),
            "hidden_norm": self.get("hidden_norm.weight"),
            "norm": self.get("norm.weight"),
            "hidden_proj": self.get("candidate_selector.hidden_projection.weight"),
            "pred": self.get("candidate_selector.predecessor_codebook"),
            "succ": self.get("candidate_selector.successor_codebook"),
        }
        self.layers = [self._layer(i) for i in range(int(self.cfg_raw["num_hidden_layers"]))]
        # the target's two tensors, out of a sharded checkpoint by way of its index
        index = json.loads((Path(target) / "model.safetensors.index.json").read_text())["weight_map"]
        self.embed = self._target_tensor(target, index, EMBED_NAME, dtype)
        self.head = self._target_tensor(target, index, HEAD_NAME, dtype)
        assert self.embed.shape == (VOCAB, HIDDEN), self.embed.shape
        assert self.head.shape == (VOCAB, HIDDEN), self.head.shape

    def _target_tensor(self, target, index: dict, name: str, dtype) -> torch.Tensor:
        shard = str(Path(target) / index[name])
        with safe_open(shard, "pt") as f:
            return f.get_tensor(name).to(self.device, dtype)

    def get(self, name: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        if name not in self._cache:
            self._cache[name] = self._f.get_tensor(name).to(self.device, dtype)
        return self._cache[name]

    def _layer(self, i: int) -> dict:
        p = f"layers.{i}"
        return {
            "input_norm": self.get(f"{p}.input_layernorm.weight"),
            "post_norm": self.get(f"{p}.post_attention_layernorm.weight"),
            "q": self.get(f"{p}.self_attn.q_proj.weight"),
            "k": self.get(f"{p}.self_attn.k_proj.weight"),
            "v": self.get(f"{p}.self_attn.v_proj.weight"),
            "o": self.get(f"{p}.self_attn.o_proj.weight"),
            "q_norm": self.get(f"{p}.self_attn.q_norm.weight"),
            "k_norm": self.get(f"{p}.self_attn.k_norm.weight"),
            "gate": self.get(f"{p}.mlp.gate_proj.weight"),
            "up": self.get(f"{p}.mlp.up_proj.weight"),
            "down": self.get(f"{p}.mlp.down_proj.weight"),
            "attn_base": self.get(f"{p}.attention_conv.base_kernel").view(KERNEL, KERNEL, HIDDEN),
            "attn_kp": self.get(f"{p}.attention_conv.kernel_projection.weight"),
            "mlp_base": self.get(f"{p}.mlp_conv.base_kernel").view(KERNEL, KERNEL, HIDDEN),
            "mlp_kp": self.get(f"{p}.mlp_conv.kernel_projection.weight"),
        }

    # -- stages ------------------------------------------------------------------------------

    def fuse(self, taps: torch.Tensor) -> torch.Tensor:
        """taps [N, 5*H] bf16 -> ctx [N, H] bf16 (`tf_fxd2_fuse` then `tf_fxd2_rmsnorm`)."""
        y = torch.matmul(taps, self.top["fc"].t())      # bf16 out == rbf(fp32 acc)
        return rms_norm(y, self.top["hidden_norm"])

    def head_logits(self, hn: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
        """[R, H] bf16 -> [R, vocab] fp32 (bf16 products, fp32 accumulation, like the m16 GEMM)."""
        hf = hn.float()
        out = torch.empty(hf.shape[0], self.head.shape[0], device=self.device, dtype=torch.float32)
        for c0 in range(0, self.head.shape[0], chunk):
            c1 = min(c0 + chunk, self.head.shape[0])
            out[:, c0:c1] = torch.matmul(hf, self.head[c0:c1].float().t())
        return out

    def select(self, hn: torch.Tensor, cand: torch.Tensor, unary: torch.Tensor,
               anchor: int) -> list[int]:
        """`tf_fxd2_select`: the serial greedy walk over the head's top-k."""
        wproj = self.top["hidden_proj"]
        pred, succ = self.top["pred"], self.top["succ"]
        rows, _ = cand.shape
        props: list[int] = []
        prev = 0
        for t in range(rows):
            prev_token = int(anchor) if t == 0 else int(cand[t - 1, prev])
            hp = torch.matmul(hn[t].float()[None, :], wproj.float().t())[0]      # [rank] fp32
            ctxr = pred[prev_token].float() * hp
            sc = unary[t].float() + torch.matmul(succ[cand[t]].float(), ctxr)    # [top_k] fp32
            prev = int(torch.argmax(sc))       # first maximum: the kernel uses strict `>`
            props.append(int(cand[t, prev]))
        return props

    # -- one round ---------------------------------------------------------------------------

    def round(self, ctx_rows: torch.Tensor, anchor: int, position: int,
              trace: Optional[dict] = None) -> dict:
        """One draft round.

        ctx_rows [ctx_len, 5*H] bf16 — the prefix's taps, positions [position-ctx_len, position).
        Returns {'proposals': [7 ints], 'cand', 'unary', 'hn', ...}.
        """
        ctx = self.fuse(ctx_rows)
        if trace is not None:
            trace["fc_ctx_rows"] = ctx

        ids = torch.tensor([anchor] + [MASK] * (BLOCK - 1), device=self.device, dtype=torch.long)
        h = F.embedding(ids, self.embed)                     # [8, H] bf16
        if trace is not None:
            trace["embed"] = h

        ctx_len = ctx.shape[0]
        lo = position - ctx_len
        pos0 = position

        for i, L in enumerate(self.layers):
            h = self._layer_forward(L, h, ctx, lo, pos0, i, trace)

        rows = BLOCK - 1
        hn = rms_norm(h[1:], self.top["norm"])               # rows 1..7 only
        logits = self.head_logits(hn)
        unary, cand = torch.topk(logits, TOP_K, dim=-1)      # descending
        props = self.select(hn, cand, unary, anchor)
        if trace is not None:
            trace["final_norm_block"] = hn
            trace["logits_top16_ids"] = cand
            trace["logits_top16_vals"] = unary
            trace["proposals"] = props
        return {"proposals": props, "hn": hn, "cand": cand, "unary": unary, "ctx": ctx}

    def _layer_forward(self, L: dict, h: torch.Tensor, ctx: torch.Tensor, lo: int, pos0: int,
                       i: int, trace: Optional[dict]) -> torch.Tensor:
        B = h.shape[0]

        x = rms_norm(h, L["input_norm"])
        dyn = torch.matmul(x, L["attn_kp"].t())              # [8, 640]
        xin = dyn_conv(x, dyn, L["attn_base"], 0)
        if trace is not None:
            trace[f"conv_attn_in_{i}"] = xin

        qraw = torch.matmul(xin, L["q"].t())                 # [8, 4096]
        kraw = torch.matmul(xin, L["k"].t())                 # [8, 1024]
        vb = torch.matmul(xin, L["v"].t())
        if trace is not None:
            trace[f"qkv_{i}"] = torch.cat([qraw, kraw, vb], dim=-1)

        q = norm_rope(qraw, L["q_norm"], HEADS, pos0)                          # [8, 4096]
        kb = norm_rope(kraw, L["k_norm"], KV_HEADS, pos0)                       # [8, 1024]
        kc = norm_rope(torch.matmul(ctx, L["k"].t()), L["k_norm"], KV_HEADS, lo)  # [ctx, 1024]
        vc = torch.matmul(ctx, L["v"].t())                                     # [ctx, 1024]

        att = self._attention(q, kb, vb, kc, vc, lo, pos0, i)
        if trace is not None:
            trace[f"attn_{i}"] = att
        o = torch.matmul(att, L["o"].t())                    # [8, H]
        if trace is not None:
            trace[f"attn_proj_{i}"] = o
        h2 = dyn_conv(o, dyn, L["attn_base"], 1, res=h)
        if trace is not None:
            trace[f"conv_attn_out_{i}"] = h2

        x = rms_norm(h2, L["post_norm"])
        dyn = torch.matmul(x, L["mlp_kp"].t())
        xin = dyn_conv(x, dyn, L["mlp_base"], 0)
        if trace is not None:
            trace[f"conv_mlp_in_{i}"] = xin
        gate = torch.matmul(xin, L["gate"].t())
        up = torch.matmul(xin, L["up"].t())
        act = (gate.float() * torch.sigmoid(gate.float()) * up.float()).to(torch.bfloat16)
        o = torch.matmul(act, L["down"].t())
        if trace is not None:
            trace[f"mlp_{i}"] = o
        h = dyn_conv(o, dyn, L["mlp_base"], 1, res=h2)
        if trace is not None:
            trace[f"conv_mlp_out_{i}"] = h
            trace[f"block_hidden_{i}"] = h
        return h

    @staticmethod
    def _attention(q, kb, vb, kc, vc, lo: int, pos0: int, i: int) -> torch.Tensor:
        """`tf_fxd2_attn`: row m sees prefix positions [lo+m, pos0) and all BLOCK block rows."""
        B = kb.shape[0]
        ctx_len = kc.shape[0]
        rep = HEADS // KV_HEADS
        qh = q.view(B, HEADS, HD)
        kh = torch.cat([kc, kb], dim=0).view(ctx_len + B, KV_HEADS, HD).repeat_interleave(rep, dim=1)
        vh = torch.cat([vc, vb], dim=0).view(ctx_len + B, KV_HEADS, HD).repeat_interleave(rep, dim=1)

        scores = torch.einsum("qhd,khd->hqk", qh.float(), kh.float()) * (HD ** -0.5)
        idx = torch.arange(ctx_len + B, device=q.device)
        m = torch.arange(B, device=q.device)[:, None]
        visible = torch.ones(B, ctx_len + B, dtype=torch.bool, device=q.device)
        visible[:, :ctx_len] = idx[None, :ctx_len] >= m      # row m skips the first m prefix rows
        scores = scores.masked_fill(~visible[None, :, :], float("-inf"))
        att = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", att, vh.float())
        return rbf(out).to(torch.bfloat16).reshape(B, HEADS * HD)


# --------------------------------------------------------------------------------------------
# Fixture I/O and one round over a fixture (the format `draft-check` reads)
# --------------------------------------------------------------------------------------------

def load_fixture(path: str | Path, device: str = "cuda") -> dict:
    """manifest.json + ids.bin + ring.bin as the engine reads them. ring.bin is rows x 5 x 2560."""
    p = Path(path)
    man = json.loads((p / "manifest.json").read_text())
    rows, taps, hidden = man["rows"], len(man["taps"]), man["hidden"]
    ids = torch.from_numpy(np.fromfile(p / man.get("ids_file", "ids.bin"), dtype="<u4").copy())
    ring = torch.from_numpy(
        np.fromfile(p / man.get("ring_file", "ring.bin"), dtype="<u2").copy()
    ).view(torch.bfloat16).view(rows, taps * hidden)
    return {
        "manifest": man, "ids": ids, "ring": ring.to(device),
        "rows": rows, "hidden": hidden, "n_taps": taps,
        "window": man["window"], "block": man["block"], "pos_end": man["pos_end"],
    }


def ctx_rows_for(fx: dict, position: int) -> torch.Tensor:
    """The prefix's tap rows, positions [position-ctx_len, position), out of the ring by slot."""
    ctx_len = min(fx["window"], position)
    lo = position - ctx_len
    slots = torch.arange(lo, position, device=fx["ring"].device) % fx["rows"]
    return fx["ring"].index_select(0, slots)


def run_position(d: Drafter, fx: dict, position: int, trace: Optional[dict] = None) -> dict:
    ctx = ctx_rows_for(fx, position)
    return d.round(ctx, int(fx["ids"][position]), position, trace=trace)


def acceptance(proposed, posterior) -> int:
    n = 0
    for a, b in zip(proposed, posterior):
        if int(a) != int(b):
            break
        n += 1
    return n
