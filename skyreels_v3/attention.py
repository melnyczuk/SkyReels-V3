# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# NOTE: The original flash-attn (FA2/FA3) code path has been removed.
# flash-attn requires CUDA to build and import, so on MPS
# `FLASH_ATTN_2_AVAILABLE`/`FLASH_ATTN_3_AVAILABLE` were always False and
# `attention()` always fell through to `scaled_dot_product_attention` anyway.
# That fallback is what's kept here. If you ever run this on a CUDA machine
# and want flash-attn back for speed, restore the original flash_attention()
# implementation from upstream Wan/SkyReels.
import warnings

import torch
import torch.nn.functional as F

__all__ = [
    "attention",
]


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B]. Unsupported here; only used to emit a warning.
    k_lens:         [B]. Unsupported here; only used to emit a warning.
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    unused by scaled_dot_product_attention; kept for
                    signature compatibility with callers.
    deterministic:  unused by scaled_dot_product_attention; kept for
                    signature compatibility with callers.
    dtype:          torch.dtype. q/k/v are cast to this dtype before the op.
    fa_version:     unused (no flash-attn on this backend); kept for
                    signature compatibility with callers.
    """
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            "Padding mask is disabled when using scaled_dot_product_attention. "
            "It can have a significant impact on performance."
        )
    attn_mask = None

    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p
    )

    out = out.transpose(1, 2).contiguous()
    return out
