from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn


def normalize_legacy_pst_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(kwargs)
    legacy_map = {
        "tpt_enabled": "pst_enabled",
        "tpt_seq_len": "pst_seq_len",
        "tpt_prompt_len": "pst_prompt_len",
        "tpt_start_layer": "pst_start_layer",
        "tpt_groups": "pst_prompt_groups",
    }
    for legacy_key, pst_key in legacy_map.items():
        if pst_key not in normalized and legacy_key in normalized:
            normalized[pst_key] = normalized.pop(legacy_key)
    return normalized


def pop_pst_kwargs(kwargs: Dict[str, Any], default_seq_len: int = 8) -> Tuple[bool, int, int, int, int, Dict[str, Any]]:
    normalized = normalize_legacy_pst_kwargs(kwargs)
    pst_enabled = bool(normalized.pop("pst_enabled", False))
    pst_seq_len = int(normalized.pop("pst_seq_len", default_seq_len))
    pst_prompt_len = int(normalized.pop("pst_prompt_len", pst_seq_len))
    pst_start_layer = int(normalized.pop("pst_start_layer", 0))
    pst_prompt_groups = int(normalized.pop("pst_prompt_groups", 2))
    return pst_enabled, pst_seq_len, pst_prompt_len, pst_start_layer, pst_prompt_groups, normalized


def append_pst_prompt_tokens(x: Tensor, prompt_embedding: Tensor, pst_seq_len: int, pst_prompt_len: int) -> Tensor:
    bt, _, d = x.shape
    if pst_seq_len <= 0 or bt % pst_seq_len != 0 or pst_prompt_len <= 0:
        return x
    b = bt // pst_seq_len
    prompts = prompt_embedding.to(dtype=x.dtype, device=x.device)
    prompts = prompts.unsqueeze(0).expand(b, -1, -1, -1).reshape(bt, pst_prompt_len, d)
    return torch.cat([x, prompts], dim=1)


def pst_shuffle_shift(x: Tensor, pst_seq_len: int, pst_prompt_len: int, pst_prompt_groups: int) -> Tensor:
    bt, l, d = x.shape
    if pst_seq_len <= 0 or bt % pst_seq_len != 0 or pst_prompt_len <= 0:
        return x
    b = bt // pst_seq_len
    patch_len = l - pst_prompt_len
    if patch_len < 1:
        return x

    hidden = x.reshape(b, pst_seq_len, l, d)
    prompts = hidden[:, :, patch_len:, :]

    groups = max(1, min(pst_prompt_groups, pst_prompt_len))
    if pst_prompt_len % groups == 0:
        prompts = prompts.reshape(b, pst_seq_len, groups, pst_prompt_len // groups, d)
        prompts = prompts.transpose(2, 3).contiguous().reshape(b, pst_seq_len, pst_prompt_len, d)

    prompt_half = pst_prompt_len // 2
    prompt_forward = prompts[:, :, :prompt_half, :]
    prompt_backward = prompts[:, :, prompt_half:, :]

    # Keep boundary prompts instead of zeroing them.
    # Forward group: frame t receives prompts from t-1, while the first frame keeps itself.
    # Backward group: frame t receives prompts from t+1, while the last frame keeps itself.
    prompt_forward_orig = prompt_forward
    prompt_backward_orig = prompt_backward

    prompt_forward = torch.roll(prompt_forward, shifts=1, dims=1)
    prompt_forward[:, 0, :, :] = prompt_forward_orig[:, 0, :, :]

    prompt_backward = torch.roll(prompt_backward, shifts=-1, dims=1)
    prompt_backward[:, -1, :, :] = prompt_backward_orig[:, -1, :, :]

    hidden[:, :, patch_len:, :] = torch.cat([prompt_forward, prompt_backward], dim=2)
    return hidden.reshape(bt, l, d)


def aggregate_pst_tokens(
    x_norm: Tensor,
    pst_seq_len: int,
    pst_prompt_len: int,
    pst_attn: nn.MultiheadAttention,
    pst_ln: nn.Module,
) -> Tensor | None:
    bt, l, _ = x_norm.shape
    if pst_seq_len <= 0 or bt % pst_seq_len != 0 or pst_prompt_len <= 0:
        return None
    b = bt // pst_seq_len
    patch_len = l - pst_prompt_len
    if patch_len < 1:
        return None

    cls_tokens = x_norm[:, 0, :].reshape(b, pst_seq_len, -1)
    prompt_tokens = x_norm[:, patch_len:, :].reshape(b, pst_seq_len * pst_prompt_len, -1)
    aggregated, _ = pst_attn(cls_tokens, prompt_tokens, prompt_tokens)
    return pst_ln(aggregated + cls_tokens).reshape(bt, -1)
