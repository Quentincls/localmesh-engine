"""A PyTorch-SDPA stand-in for the sliver of xformers that TRELLIS.2 uses.

Why this exists
---------------
TRELLIS.2's *sparse* attention path only implements xformers, flash_attn and
flash_attn_3 - unlike its dense path, it has no sdpa fallback. On Blackwell
(sm_120) neither of the wheel-installable options works:

* xformers 0.0.32 dispatches to its bundled Hopper flash-attention kernel and
  aborts with `CUDA error ... flash_fwd_launch_template.h:188: invalid argument`;
* flash-attn has no prebuilt Windows wheel for python 3.12 + torch 2.8 + cu128,
  and we are pinned to python 3.12 by the only Blackwell NATTEN build.

Rather than fork TRELLIS.2, we register a module under `xformers.ops` that
implements the exact two entry points its code calls. TRELLIS.2 keeps thinking
it is talking to xformers, and every kernel underneath is stock PyTorch, which
has supported sm_120 since 2.7.

What it must reproduce
----------------------
`memory_efficient_attention(q, k, v, attn_bias)` where q/k/v are *packed*
variable-length sequences of shape [1, T, H, C] and `attn_bias` is a
BlockDiagonalMask carrying the sequence lengths. Attention must not cross
sequence boundaries.

Rather than materialise a [T, T] block-diagonal mask (T reaches ~49k tokens, so
that alone would be 2.4 G elements), sequences are grouped by length and each
group is run as one batched, mask-free SDPA call. Windowed attention produces
thousands of equal-length windows, so it collapses to one or two calls.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = ["install", "memory_efficient_attention", "BlockDiagonalMask"]


class BlockDiagonalMask:
    """Carries sequence lengths; no tensor is ever materialised."""

    def __init__(self, q_seqlen: Sequence[int], kv_seqlen: Optional[Sequence[int]] = None):
        self.q_seqlen = [int(x) for x in q_seqlen]
        self.kv_seqlen = [int(x) for x in (kv_seqlen if kv_seqlen is not None else q_seqlen)]
        if len(self.q_seqlen) != len(self.kv_seqlen):
            raise ValueError(
                f"q/kv sequence count mismatch: {len(self.q_seqlen)} vs {len(self.kv_seqlen)}")

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None) -> "BlockDiagonalMask":
        if isinstance(q_seqlen, torch.Tensor):
            q_seqlen = q_seqlen.tolist()
        if isinstance(kv_seqlen, torch.Tensor):
            kv_seqlen = kv_seqlen.tolist()
        return cls(q_seqlen, kv_seqlen)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"BlockDiagonalMask(n={len(self.q_seqlen)}, "
                f"q_tokens={sum(self.q_seqlen)}, kv_tokens={sum(self.kv_seqlen)})")


def _offsets(seqlen: Sequence[int], device) -> torch.Tensor:
    out = torch.zeros(len(seqlen) + 1, dtype=torch.long, device=device)
    out[1:] = torch.tensor(seqlen, dtype=torch.long, device=device).cumsum(0)
    return out


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[BlockDiagonalMask] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    **_ignored,
) -> torch.Tensor:
    """xformers-compatible signature. q/k/v are [B, T, H, C]."""
    if query.ndim != 4:
        raise ValueError(f"expected [B, T, H, C], got {tuple(query.shape)}")

    # No block structure: a plain batched attention.
    if attn_bias is None:
        q = query.transpose(1, 2)  # [B, H, T, C]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=p, scale=scale)
        return out.transpose(1, 2)

    if query.shape[0] != 1:
        raise ValueError("packed varlen input must have batch dimension 1")

    q_all, k_all, v_all = query[0], key[0], value[0]  # [T, H, C]
    q_seq, kv_seq = attn_bias.q_seqlen, attn_bias.kv_seqlen
    device = q_all.device

    if sum(q_seq) != q_all.shape[0] or sum(kv_seq) != k_all.shape[0]:
        raise ValueError(
            f"sequence lengths do not cover the packed tensors: "
            f"q {sum(q_seq)} vs {q_all.shape[0]}, kv {sum(kv_seq)} vs {k_all.shape[0]}")

    # Fast path: one sequence, nothing to mask.
    if len(q_seq) == 1:
        out = F.scaled_dot_product_attention(
            q_all.transpose(0, 1).unsqueeze(0),
            k_all.transpose(0, 1).unsqueeze(0),
            v_all.transpose(0, 1).unsqueeze(0),
            dropout_p=p, scale=scale,
        )
        return out.squeeze(0).transpose(0, 1).unsqueeze(0)

    q_off = _offsets(q_seq, device)
    kv_off = _offsets(kv_seq, device)
    out = torch.empty_like(q_all)

    # Group sequences that share a (q_len, kv_len) shape so each group is a
    # single batched SDPA call with no padding and no mask.
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (lq, lkv) in enumerate(zip(q_seq, kv_seq)):
        groups.setdefault((lq, lkv), []).append(i)

    ar_cache: dict[int, torch.Tensor] = {}

    def arange(n: int) -> torch.Tensor:
        if n not in ar_cache:
            ar_cache[n] = torch.arange(n, device=device)
        return ar_cache[n]

    for (lq, lkv), members in groups.items():
        idx = torch.tensor(members, dtype=torch.long, device=device)
        q_idx = q_off[idx].unsqueeze(1) + arange(lq).unsqueeze(0)      # [G, lq]
        kv_idx = kv_off[idx].unsqueeze(1) + arange(lkv).unsqueeze(0)   # [G, lkv]

        qg = q_all[q_idx].permute(0, 2, 1, 3)    # [G, H, lq, C]
        kg = k_all[kv_idx].permute(0, 2, 1, 3)
        vg = v_all[kv_idx].permute(0, 2, 1, 3)

        og = F.scaled_dot_product_attention(qg, kg, vg, dropout_p=p, scale=scale)
        out[q_idx] = og.permute(0, 2, 1, 3).to(out.dtype)

    return out.unsqueeze(0)


def install() -> None:
    """Register this module as `xformers.ops` for the rest of the process.

    Must run before trellis2 imports it. Idempotent. The real xformers package
    stays installed and untouched; we simply shadow the import.
    """
    if getattr(sys.modules.get("xformers.ops"), "_lumengen_shim", False):
        return

    fmha = _module("xformers.ops.fmha")
    fmha.BlockDiagonalMask = BlockDiagonalMask

    ops = _module("xformers.ops")
    ops.memory_efficient_attention = memory_efficient_attention
    ops.fmha = fmha
    ops._lumengen_shim = True

    root = sys.modules.get("xformers")
    if root is None or not isinstance(root, types.ModuleType):
        root = _module("xformers", paquet=True)
    root.ops = ops

    sys.modules["xformers"] = root
    sys.modules["xformers.ops"] = ops
    sys.modules["xformers.ops.fmha"] = fmha


def _module(nom: str, paquet: bool = False) -> types.ModuleType:
    """Un module synthétique COMPLET, `__spec__` compris.

    `types.ModuleType` laisse `__spec__` à None, et c'est un piège qui a coûté
    une fonctionnalité entière : `importlib.util.find_spec("xformers")` ne rend
    pas None pour un module déjà présent dans `sys.modules`, il rend son
    `__spec__` — et LÈVE `ValueError: xformers.__spec__ is None` quand il n'y en
    a pas. Or c'est exactement ce que diffusers appelle pour savoir si xformers
    est là. Résultat : dès que ce shim était posé (donc dès qu'un splat avait
    été généré), IDArb refusait de se charger et le maillage sortait SANS carte
    metallic, en silence — la fonctionnalité était morte dans le produit tout
    en marchant dans chaque banc, où le shim n'existe pas.
    """
    m = types.ModuleType(nom)
    m.__spec__ = importlib.machinery.ModuleSpec(nom, None, is_package=paquet)
    if paquet:
        m.__path__ = []
        m.__spec__.submodule_search_locations = []
    return m
