"""Reference DFlash2 drafter, torch-only, for the flashnext tap-row experiment.

The point of this file is NOT to serve anything. It exists to answer one question
before any Zig/CUDA work is committed to: which single 2560-wide row of a flashnext
layer is the tap that incoai's DFlash2 drafter was trained against.

flashnext keeps 4 hyper-connection streams, so a layer's output is 4x2560 wide
(`input_mix_weight_down` is [320, 10240], `hc_norm` is [10240]). The drafter's `fc`
is [2560, 12800] = 5 taps x 2560, so exactly one 2560 row per tapped layer is wanted.
The SGLang capture patch averages the 4 streams; on flashnext the mix is a learned
projection, so that average is not the same tensor. Candidates (see TapSource).

Weights are bf16 on disk; everything runs in fp32 for comparison, which is fine
because the experiment measures ACCEPTANCE LENGTH, a discrete quantity, not logits.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open


class TapSource(Enum):
    """Which flashnext row stands in for 'the completed layer output'."""

    READOUT = "readout"  # the mixed/down-projected single 2560 row
    NORMED = "normed"  # the normed variant of that row
    MEAN4 = "mean4"  # the SGLang patch's mean over the 4 raw streams (control)


DEFAULT_TAP = TapSource.READOUT


class DFlash2Weights:
    """Lazy access to the 81-tensor checkpoint."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._f = safe_open(self.path, "pt")
        self.keys = set(self._f.keys())

    def get(self, name: str) -> torch.Tensor:
        if name not in self.keys:
            raise KeyError(f"missing tensor {name}")
        return self._f.get_tensor(name).to(torch.float32)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._f.get_slice(name).get_shape())


class GroupedConv:
    """DFlash2's two-tap dynamic depthwise conv, grouped by `group` channels.

    One projection of the sublayer input predicts per-position, per-group kernel
    deltas for both the input side (before the sublayer) and the output side (after).
    base_kernel starts as an identity, so the module is a stable extension of DFlash.

    y[t, c] = sum_l (base[l, c] + dyn[t, l, c // group]) * x[t - l, c],  x[-1] = 0

    The conv never crosses a block boundary: blocks are the unit of a verify round.
    """

    def __init__(self, w: DFlash2Weights, prefix: str, hidden: int, block: int, taps: int, group: int) -> None:
        self.block, self.taps, self.group = block, taps, group
        self.groups = hidden // group
        self.base = w.get(f"{prefix}.base_kernel").reshape(2, taps, hidden)
        self.proj = w.get(f"{prefix}.kernel_projection.weight")  # [2*taps*groups, hidden]

    def _convolve(self, x: torch.Tensor, delta: torch.Tensor, side: int) -> torch.Tensor:
        # x: [B, S, D] with S % block == 0; delta: [B, S, taps, groups]
        b, s, d = x.shape
        nb = s // self.block
        blocks = x.reshape(b, nb, self.block, self.groups, self.group)
        dyn = delta.reshape(b, nb, self.block, self.taps, self.groups)
        base = self.base[side].reshape(1, 1, 1, self.taps, self.groups, self.group)
        coef = base + dyn.unsqueeze(-1)
        out = coef[:, :, :, 0] * blocks
        for tap in range(1, self.taps):
            shifted = F.pad(blocks[:, :, : self.block - tap], (0, 0, 0, 0, tap, 0))
            out = out + coef[:, :, :, tap] * shifted
        return out.reshape(b, s, d)

    def prepare(self, x: torch.Tensor):
        b, s, d = x.shape
        coef = (self.proj @ x.reshape(b * s, d).T).T  # [B*S, 2*taps*groups]
        coef = coef.reshape(b, s, 2, self.taps, self.groups)
        return self._convolve(x, coef[..., 0, :, :], side=0), coef[..., 1, :, :]

    def finish(self, x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        return self._convolve(x, coef, side=1)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary embedding over the last dim, applied to [B, H, S, HD]; positions [S] or [B, S]."""
    hd = x.shape[-1]
    half = hd // 2
    if positions.dim() == 1:
        positions = positions[None, :]
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=x.device) / half))
    ang = positions.float()[:, None, :, None] * freqs[None, None, None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class DraftLayer:
    def __init__(self, w: DFlash2Weights, i: int, cfg: dict) -> None:
        h, ffn = cfg["hidden"], cfg["ffn"]
        p = f"layers.{i}"
        self.attn_conv = GroupedConv(w, f"{p}.attention_conv", h, cfg["block"], cfg["conv_kernel"], cfg["conv_group"])
        self.mlp_conv = GroupedConv(w, f"{p}.mlp_conv", h, cfg["block"], cfg["conv_kernel"], cfg["conv_group"])
        self.n1 = w.get(f"{p}.input_layernorm.weight")
        self.n2 = w.get(f"{p}.post_attention_layernorm.weight")
        self.q = w.get(f"{p}.self_attn.q_proj.weight")
        self.k = w.get(f"{p}.self_attn.k_proj.weight")
        self.v = w.get(f"{p}.self_attn.v_proj.weight")
        self.o = w.get(f"{p}.self_attn.o_proj.weight")
        self.qn = w.get(f"{p}.self_attn.q_norm.weight")
        self.kn = w.get(f"{p}.self_attn.k_norm.weight")
        self.gate = w.get(f"{p}.mlp.gate_proj.weight")
        self.up = w.get(f"{p}.mlp.up_proj.weight")
        self.down = w.get(f"{p}.mlp.down_proj.weight")

    def forward(self, x, ctx_k, ctx_v, ctx_len, positions, cfg):
        h = x.shape[-1]
        heads, kvh, hd = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
        eps = cfg["eps"]
        rep = heads // kvh

        r = x
        xn = rms_norm(x, self.n1, eps)
        xn, coef = self.attn_conv.prepare(xn)
        b, s, _ = xn.shape

        q = (xn @ self.q.T).view(b, s, heads, hd)
        q = rms_norm(q, self.qn, eps).transpose(1, 2)
        k_ctx = (ctx_k @ self.k.T).view(b, ctx_len, kvh, hd)
        k_noise = (xn @ self.k.T).view(b, s, kvh, hd)
        v_ctx = (ctx_v @ self.v.T).view(b, ctx_len, kvh, hd)
        v_noise = (xn @ self.v.T).view(b, s, kvh, hd)
        k = torch.cat([k_ctx, k_noise], dim=1)
        v = torch.cat([v_ctx, v_noise], dim=1)
        k = rms_norm(k, self.kn, eps).transpose(1, 2)
        v = v.transpose(1, 2)
        q = rope(q, positions, cfg["theta"])
        p1 = positions if positions.dim() == 2 else positions[None, :]
        k = rope(k, torch.cat([p1.new_zeros(1, ctx_len), p1], dim=1), cfg["theta"])

        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        # block attention is bidirectional within the block, and every row sees the whole context
        att = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        mask = torch.zeros(s, ctx_len + s, dtype=torch.bool, device=q.device)
        mask[:, :ctx_len] = True
        mask[:, ctx_len:] = True  # is_causal: false
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(b, s, heads * hd)
        out = out @ self.o.T  # [B, S, hidden]: the conv wraps the sublayer output, not the head-width rows
        out = self.attn_conv.finish(out, coef)
        x = r + out

        r = x
        xn = rms_norm(x, self.n2, eps)
        xn, coef = self.mlp_conv.prepare(xn)
        xn = F.silu(xn @ self.gate.T) * (xn @ self.up.T)
        xn = xn @ self.down.T  # back to hidden width before the conv's output side
        xn = self.mlp_conv.finish(xn, coef)
        return r + xn


class CandidateSelector:
    """Low-rank predecessor -> successor re-ranker over the head's top-k."""

    def __init__(self, w: DFlash2Weights, cfg: dict) -> None:
        self.top_k = cfg["selector_top_k"]
        self.pred = w.get("candidate_selector.predecessor_codebook")
        self.succ = w.get("candidate_selector.successor_codebook")
        self.hidden_proj = w.get("candidate_selector.hidden_projection.weight")

    def build_lattice(self, cand_ids, unary, hidden, anchor):
        b, s, k = cand_ids.shape
        prev = torch.cat([anchor[:, None, None].expand(-1, 1, k), cand_ids[:, :-1]], dim=1)
        p = self.pred[prev]
        sc = self.succ[cand_ids]
        ctx = p * (hidden @ self.hidden_proj.T)[:, :, None, :]
        return unary[:, :, None] + torch.einsum("bspr,bscr->bspc", ctx, sc)

    def greedy_path(self, cand_ids, unary, hidden, anchor):
        lat = self.build_lattice(cand_ids, unary, hidden, anchor)
        b, s, _, k = lat.shape
        prev_idx = torch.zeros(b, dtype=torch.long, device=lat.device)
        path = []
        for t in range(s):
            scores = lat[:, t, prev_idx, :]
            prev_idx = scores.argmax(-1)
            path.append(cand_ids[:, t].gather(1, prev_idx.reshape(-1, 1)))
        return torch.cat(path, dim=1)


class DFlash2Draft:
    """The drafter's own forward: taps -> fc -> 5 layers -> norm -> head + selector."""

    def __init__(self, w: DFlash2Weights, cfg: dict, lm_head: torch.Tensor) -> None:
        self.cfg = cfg
        self.fc = w.get("fc.weight")
        self.hidden_norm = w.get("hidden_norm.weight")
        self.norm = w.get("norm.weight")
        self.layers = [DraftLayer(w, i, cfg) for i in range(cfg["layers"])]
        self.selector = CandidateSelector(w, cfg)
        self.lm_head = lm_head  # the target's head, fp32, [vocab, hidden]
        self.embed: Optional[torch.Tensor] = None  # the target's embedding, set by the caller

    def forward(self, taps: torch.Tensor, block_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # taps: [B, ctx, 5*hidden] -> [B, ctx, hidden]
        ctx = rms_norm(taps @ self.fc.T, self.hidden_norm, self.cfg["eps"])
        x = F.embedding(block_ids, self.embed)  # set by caller
        for layer in self.layers:
            x = layer.forward(x, ctx, ctx, ctx.shape[1], positions, self.cfg)
        return rms_norm(x, self.norm, self.cfg["eps"])

    def propose(self, hidden: torch.Tensor, anchor_ids: torch.Tensor) -> torch.Tensor:
        unary = hidden @ self.lm_head.T
        top, cand = unary.topk(self.selector.top_k, dim=-1)
        return self.selector.greedy_path(cand, top, hidden, anchor_ids)


def load(path: str | Path, lm_head: Optional[torch.Tensor] = None, embed: Optional[torch.Tensor] = None):
    p = Path(path)
    cfg_raw = json.loads((p / "config.json").read_text())
    d = cfg_raw["dflash_config"]
    cfg = dict(
        hidden=cfg_raw["hidden_size"],
        layers=cfg_raw["num_hidden_layers"],
        heads=cfg_raw["num_attention_heads"],
        kv_heads=cfg_raw["num_key_value_heads"],
        head_dim=cfg_raw["head_dim"],
        ffn=cfg_raw["intermediate_size"],
        vocab=cfg_raw["vocab_size"],
        eps=cfg_raw["rms_norm_eps"],
        theta=float(cfg_raw["rope_parameters"]["rope_theta"]),
        block=d["block_size"],
        mask_token=d["mask_token_id"],
        taps=list(d["target_layer_ids"]),
        conv_kernel=d["conv_kernel_size"],
        conv_group=d["conv_group_size"],
        selector_rank=d["selector_rank"],
        selector_top_k=d["selector_top_k"],
        is_causal=cfg_raw.get("is_causal", False),
    )
    w = DFlash2Weights(p / "model.safetensors")
    model = DFlash2Draft(w, cfg, lm_head if lm_head is not None else torch.zeros(cfg["vocab"], cfg["hidden"]))
    model.embed = embed
    return model, cfg


def acceptance(proposed: torch.Tensor, posterior: torch.Tensor) -> int:
    """Longest matching prefix between the drafts and the target's own tokens."""
    n = 0
    for a, b in zip(proposed.tolist(), posterior.tolist()):
        if a != b:
            break
        n += 1
    return n
