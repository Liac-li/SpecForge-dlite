# Copyright (c) 2025 sgl-project
# SPDX-License-Identifier: MIT
# Adapted from SJTU-DAI-Sys/LiteSpec (4e87f22), core/attention_masks.py.
"""DLite block isolation and teacher-only history visibility."""

import torch


def create_dlite_mask(
    anchors,
    keep,
    *,
    seq_len,
    block_size,
    chs_slots,
    history_window=None,
    backend="flex_attention"
):
    batch, blocks = anchors.shape
    prefix = seq_len if history_window is not None else 0
    query_start = prefix + blocks * chs_slots
    q_len, kv_len = blocks * block_size, query_start + blocks * block_size

    def mask_mod(b, h, q, kv):
        block = q // block_size
        safe_block = block.clamp(max=blocks - 1)
        anchor = anchors[b, safe_block]
        valid = keep[b, safe_block] & (block < blocks)
        history = (
            (kv < prefix) & (kv >= anchor - (history_window or 1)) & (kv <= anchor - 2)
        )
        chs = (
            (kv >= prefix) & (kv < query_start) & ((kv - prefix) // chs_slots == block)
        )
        query = (kv >= query_start) & ((kv - query_start) // block_size == block)
        # Padded blocks see their own CHS, avoiding empty softmax rows.
        return (valid & (history | chs | query)) | (chs & ~valid & (block < blocks))

    if backend == "flex_attention":
        from .flex_attention import compile_friendly_create_block_mask

        return compile_friendly_create_block_mask(
            mask_mod,
            B=batch,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=anchors.device,
        )
    b = torch.arange(batch, device=anchors.device)[:, None, None, None]
    q = torch.arange(q_len, device=anchors.device)[None, None, :, None]
    kv = torch.arange(kv_len, device=anchors.device)[None, None, None, :]
    mask = mask_mod(b, 0, q, kv)
    if backend == "eager":
        return torch.zeros_like(mask, dtype=torch.float32).masked_fill(
            ~mask, float("-inf")
        )
    return mask
