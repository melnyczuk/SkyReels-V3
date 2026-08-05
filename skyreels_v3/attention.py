# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# NOTE ON THIS FILE'S HISTORY:
# - The original flash-attn (FA2/FA3) code path has been removed. flash-attn
#   requires CUDA to build and import, so on MPS it was always unavailable
#   anyway.
# - We also don't call torch.nn.functional.scaled_dot_product_attention here.
#   On this MPS setup, the fused SDPA kernel hits a native (uncatchable)
#   assertion failure ("Destination NDArray and Accumulator NDArray cannot
#   have different datatype in MPSNDArrayMatrixMultiplication") even when
#   q/k/v all have matching, correct dtypes — it's a bug in the fused
#   kernel's internal accumulator handling, not something fixable from the
#   Python side.
# - A naive manual attention (full matmul(q, k^T) -> softmax -> matmul(., v))
#   avoids that crash, but materializes the full [Lq, Lk] score matrix, which
#   is exactly what runs out of memory for video-length sequences (tens of
#   thousands of tokens -> a Lq x Lk matrix with tens of billions of entries).
#
# What's implemented below instead is a tiled ("flash-attention-style")
# attention: it processes small blocks of queries against small blocks of
# keys/values at a time, using the online-softmax algorithm to combine
# partial results incrementally. This is mathematically IDENTICAL to full
# attention (not an approximation) — it just never materializes the whole
# score matrix at once. Peak memory becomes roughly
# (q_chunk_size x kv_chunk_size) instead of (Lq x Lk), independent of how
# long the video is. The cost is speed: this is a pure-Python double loop
# over blocks issuing many small matmuls, with none of the fused-kernel /
# hardware-specific speed tricks that make real FlashAttention fast on CUDA.
# It trades wall-clock time for the ability to run at all on this hardware.
import os
import warnings

import torch
import torch.nn.functional as F

__all__ = [
    "attention",
]

# Tunable without touching code, e.g.:
#   SKYREELS_ATTN_Q_CHUNK=512 SKYREELS_ATTN_KV_CHUNK=512 python generate_video.py ...
# Smaller values -> less peak memory per block, but more (slower) loop
# iterations. Larger values -> faster, but higher peak memory. Tune down
# further if you still hit MPS OOM; tune up if generation feels too slow and
# you have memory headroom.
_DEFAULT_Q_CHUNK = int(os.environ.get("SKYREELS_ATTN_Q_CHUNK", 1024))
_DEFAULT_KV_CHUNK = int(os.environ.get("SKYREELS_ATTN_KV_CHUNK", 1024))


def _tiled_attention(
    q,
    k,
    v,
    scale,
    causal=False,
    dropout_p=0.0,
    q_chunk_size=_DEFAULT_Q_CHUNK,
    kv_chunk_size=_DEFAULT_KV_CHUNK,
):
    """
    q, k, v: [B, H, L, D] (batch, heads, sequence length, head dim).
    Returns: [B, H, Lq, D], same dtype as v.

    Implements the online-softmax / FlashAttention algorithm: for each query
    block, iterate over key/value blocks while maintaining a running max
    score (m), running sum of exponentials (l), and a running weighted
    output accumulator (acc). Each time a new key/value block is visited,
    the running state is rescaled to account for the new information — this
    is what makes the result exactly equal to a full-matrix softmax despite
    never seeing more than one block of keys at a time.
    """
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    out_dtype = v.dtype
    out = torch.empty(B, H, Lq, D, device=q.device, dtype=out_dtype)

    for q_start in range(0, Lq, q_chunk_size):
        q_end = min(q_start + q_chunk_size, Lq)
        q_blk = q[:, :, q_start:q_end, :].float()

        # Running softmax statistics for this block of queries, accumulated
        # in float32 for numerical stability regardless of q/k/v's dtype.
        m_i = torch.full(
            (B, H, q_end - q_start, 1),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        l_i = torch.zeros(
            (B, H, q_end - q_start, 1), device=q.device, dtype=torch.float32
        )
        acc = torch.zeros(
            (B, H, q_end - q_start, D), device=q.device, dtype=torch.float32
        )

        # For causal attention, a query can never attend to a key that comes
        # after it, so any key-block entirely past this query-block's end
        # can be skipped outright.
        kv_limit = q_end if causal else Lk

        for k_start in range(0, kv_limit, kv_chunk_size):
            k_end = min(k_start + kv_chunk_size, kv_limit)
            k_blk = k[:, :, k_start:k_end, :].float()
            v_blk = v[:, :, k_start:k_end, :].float()

            # [B, H, qb, kb] — only ONE block this size exists at a time,
            # never the full [Lq, Lk] matrix.
            scores = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * scale

            if causal:
                q_idx = torch.arange(q_start, q_end, device=q.device).view(-1, 1)
                k_idx = torch.arange(k_start, k_end, device=q.device).view(1, -1)
                mask = k_idx > q_idx
                if mask.any():
                    scores = scores.masked_fill(mask, float("-inf"))

            block_max = scores.amax(dim=-1, keepdim=True)  # [B, H, qb, 1]
            m_new = torch.maximum(m_i, block_max)
            # Guard against -inf - (-inf) = nan, which only happens if a
            # query row's running max AND this block's max are both -inf
            # (i.e. every key seen so far, including this block, is masked
            # out for that row — only possible with causal masking).
            m_new_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)

            p = torch.exp(scores - m_new_safe)  # unnormalized probs, this block
            alpha = torch.exp(m_i - m_new_safe)  # rescale factor for prior state

            l_i = alpha * l_i + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(p, v_blk)
            m_i = m_new

        out_blk = acc / l_i.clamp_min(1e-20)
        out[:, :, q_start:q_end, :] = out_blk.to(out_dtype)

    if dropout_p > 0.0:
        # Applying dropout after online-softmax accumulation isn't exactly
        # equivalent to dropping individual attention weights pre-accumulation
        # (as a textbook implementation would), but every caller in this
        # codebase runs inference with dropout_p=0.0, so this fallback is
        # never actually exercised.
        out = F.dropout(out, p=dropout_p)

    return out


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
    dtype=None,
    fa_version=None,
    q_chunk_size=None,
    kv_chunk_size=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B]. Unsupported here; only used to emit a warning.
    k_lens:         [B]. Unsupported here; only used to emit a warning.
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    q_scale:        optional extra scaling applied to q before the dot
                    product (kept for signature compatibility with callers;
                    applied if given).
    causal:         bool. Whether to apply causal attention mask.
    window_size:    unused (this codebase always calls with global attention,
                    window_size=(-1, -1)); kept for signature compatibility.
    deterministic:  unused; this implementation has no randomness to control
                    besides dropout. Kept for signature compatibility.
    dtype:          torch.dtype or None. If given, q/k/v are cast to this
                    dtype before the op (the softmax accumulation itself
                    always happens in float32 regardless, for stability).
                    If None (default), q/k/v are left in whatever dtype the
                    caller already put them in.
    fa_version:     unused (no flash-attn on this backend); kept for
                    signature compatibility with callers.
    q_chunk_size, kv_chunk_size:
                    Block sizes for the tiled attention below. Defaults come
                    from the SKYREELS_ATTN_Q_CHUNK / SKYREELS_ATTN_KV_CHUNK
                    environment variables (1024 if unset). Smaller = less
                    peak memory, more (slower) loop iterations.
    """
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            "Padding mask is disabled in this attention implementation. "
            "It can have a significant impact on performance."
        )

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if dtype is not None:
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
    if q_scale is not None:
        q = q * q_scale

    scale = softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)

    out = _tiled_attention(
        q,
        k,
        v,
        scale,
        causal=causal,
        dropout_p=dropout_p,
        q_chunk_size=q_chunk_size or _DEFAULT_Q_CHUNK,
        kv_chunk_size=kv_chunk_size or _DEFAULT_KV_CHUNK,
    )

    out = out.transpose(1, 2).contiguous()
    return out
