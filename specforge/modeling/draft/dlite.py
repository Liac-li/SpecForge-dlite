# Copyright (c) 2025 sgl-project
# SPDX-License-Identifier: MIT
# Adapted from SJTU-DAI-Sys/LiteSpec (4e87f22), litespec/modeling/draft/dlite.py.
# See LICENSE and the DLite recipe for attribution.
"""DLite teacher/student backbones with official inference checkpoint names."""

import logging
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Unpack

from .flex_attention import compile_friendly_flex_attention
from .registry import register_draft

SEQUENTIAL_HEAD_TYPES = ("rnn",)
DLITE_ARCHITECTURE_VERSIONS = ("dlite_v1", "dlite_v2")
DLITE_ARCHITECTURE_VERSION = "dlite_v2"
DLITE_MODEL_ROLES = ("swa_teacher", "student")


class DLiteSequentialHead(nn.Module):
    """Low-rank serial vocabulary head for DLite block prediction."""

    def __init__(
        self,
        *,
        head_type: str,
        vocab_size: int,
        sequential_rank: int,
        hidden_size: int,
        max_prediction_length: int,
    ) -> None:
        super().__init__()
        self.head_type = str(head_type).lower()
        self.vocab_size = int(vocab_size)
        self.sequential_rank = int(sequential_rank)
        self.hidden_size = int(hidden_size)
        self.max_prediction_length = int(max_prediction_length)

        if self.head_type not in SEQUENTIAL_HEAD_TYPES:
            raise ValueError(
                f"Unknown sequential head type {self.head_type!r}; "
                f"expected one of {SEQUENTIAL_HEAD_TYPES}."
            )
        if self.sequential_rank <= 0:
            raise ValueError(
                f"sequential_rank must be positive, got {self.sequential_rank}."
            )
        if self.max_prediction_length <= 0:
            raise ValueError(
                "max_prediction_length must be positive, got "
                f"{self.max_prediction_length}."
            )
        self.prev_token_embedding = nn.Embedding(self.vocab_size, self.sequential_rank)
        self.output_proj = nn.Linear(self.sequential_rank, self.vocab_size, bias=False)
        self.state_proj = nn.Linear(2 * self.sequential_rank, 2 * self.sequential_rank)
        self.state_hidden_mlp = nn.Linear(
            self.sequential_rank + self.hidden_size,
            self.sequential_rank,
        )

    def project_logits(self, latent_states: torch.Tensor) -> torch.Tensor:
        """Project low-rank head states to full-vocabulary logits."""
        return self.output_proj(latent_states)

    def _compute_step_latent(
        self,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        prev_embeddings = self.prev_token_embedding(prev_token_ids.long())
        if state is None:
            state = torch.zeros_like(prev_embeddings)
        mem_inputs = torch.cat([state, prev_embeddings], dim=-1)
        gate_raw, candidate_raw = self.state_proj(mem_inputs).chunk(2, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        fused_inputs = torch.cat([new_state, hidden_states], dim=-1)
        return self.state_hidden_mlp(fused_inputs), new_state

    def forward_teacher_forcing(
        self,
        *,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return low-rank states for teacher-forced block predictions."""
        if hidden_states.shape[:-1] != prev_token_ids.shape:
            raise ValueError(
                "hidden_states and prev_token_ids leading shapes must match, "
                f"got {tuple(hidden_states.shape)} and "
                f"{tuple(prev_token_ids.shape)}."
            )
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"got {hidden_states.size(-1)}."
            )

        prediction_length = hidden_states.size(-2)
        if prediction_length > self.max_prediction_length:
            raise ValueError(
                f"prediction_length={prediction_length} exceeds configured "
                f"maximum {self.max_prediction_length}."
            )
        batch_shape = hidden_states.shape[:-2]
        state = torch.zeros(
            *batch_shape,
            self.sequential_rank,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        outputs: list[torch.Tensor] = []
        for position in range(prediction_length):
            latent, state = self._compute_step_latent(
                prev_token_ids=prev_token_ids[..., position],
                hidden_states=hidden_states[..., position, :],
                state=state,
            )
            outputs.append(latent.unsqueeze(-2))
        if not outputs:
            return hidden_states.new_empty(
                *hidden_states.shape[:-2], 0, self.sequential_rank
            )
        return torch.cat(outputs, dim=-2)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    k_len = k.size(-2)
    q_cos = cos[..., -q_len:, :]
    q_sin = sin[..., -q_len:, :]
    k_cos = cos[..., -k_len:, :]
    k_sin = sin[..., -k_len:, :]
    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
    return q_embed, k_embed


def build_target_layer_ids(
    num_transformer_layers: int, chs_num_layers: int
) -> list[int]:
    """Select exactly ``chs_num_layers`` layers, including the first and last."""
    L = num_transformer_layers
    S = int(chs_num_layers)
    if L < 2:
        raise ValueError(f"DLite requires at least 2 target layers, got {L}.")
    if not 2 <= S <= L:
        raise ValueError(f"chs_num_layers must be in [2, {L}], got {chs_num_layers}.")
    selected_layer_ids: set[int] = {0, L - 1}
    n_middle = S - 2
    lo, hi = 1, L - 2
    if n_middle > 0 and lo <= hi:
        interior_span = hi - lo
        num_interior_layers = hi - lo + 1
        n_take = min(n_middle, num_interior_layers)
        if n_take == 1:
            selected_layer_ids.add((lo + hi) // 2)
        else:
            for i in range(n_take):
                if interior_span == 0:
                    idx = lo
                else:
                    idx = lo + int(round(i * interior_span / (n_take - 1)))
                selected_layer_ids.add(int(max(lo, min(hi, idx))))
    target_layer_ids = sorted(selected_layer_ids)
    if len(target_layer_ids) != S:
        raise RuntimeError(
            f"Failed to select exactly {S} CHS layers from target depth {L}: {target_layer_ids}"
        )
    return target_layer_ids


class Qwen3DLiteAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        cos, sin = position_embeddings

        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.config._attn_implementation == "flex_attention":
            # Call FlexAttention directly.  Transformers requests LSE during
            # training, but the FLASH backend cannot backpropagate through the
            # auxiliary LSE output.  DLite never consumes it.
            attn_output = compile_friendly_flex_attention(
                q,
                k,
                v,
                block_mask=attention_mask,
                enable_gqa=True,
                scale=self.scaling,
                kernel_options=kwargs.get("kernel_options"),
            ).transpose(1, 2)
            attn_weights = None
        else:
            attn_fn: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attn_fn(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DLiteDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DLiteAttention(
            config=config,
            layer_idx=layer_idx,
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@register_draft
class DLiteDraftModel(Qwen3PreTrainedModel):
    method_config_key = "dlite_config"

    config_class = Qwen3Config

    _no_split_modules = ["Qwen3DLiteDecoderLayer"]

    def __init__(self, config, attention_backend=None) -> None:
        if attention_backend is not None:
            config._attn_implementation = attention_backend
        super().__init__(config)
        self.config = config
        if config.architectures is None:
            config.architectures = [type(self).__name__]
        if config.architectures != [type(self).__name__]:
            raise ValueError(
                f"{type(self).__name__} requires its own architecture/config"
            )
        dlite_config = dict(getattr(config, "dlite_config", None) or {})
        architecture_version = dlite_config.get(
            "architecture_version", DLITE_ARCHITECTURE_VERSION
        )
        if architecture_version not in DLITE_ARCHITECTURE_VERSIONS:
            raise ValueError(
                "Incompatible DLite checkpoint/config architecture_version: "
                f"expected one of {DLITE_ARCHITECTURE_VERSIONS}, got "
                f"{architecture_version!r}."
            )
        self.architecture_version = str(architecture_version)
        self.model_role = str(dlite_config.get("model_role", "")).lower()
        if self.model_role not in DLITE_MODEL_ROLES:
            raise ValueError(
                f"model_role must be one of {DLITE_MODEL_ROLES}, got "
                f"{self.model_role!r}."
            )
        required_field_names = {
            "model_role",
            "chs_num_layers",
            "target_layer_ids",
            "sequential_head",
            "sequential_rank",
        }
        missing_field_names = sorted(required_field_names - dlite_config.keys())
        if missing_field_names:
            raise ValueError(
                f"dlite_config is missing required fields: {missing_field_names}"
            )
        self.swa_window_size = int(
            dlite_config["swa_window_size"] if self.is_teacher else 1
        )
        if self.swa_window_size < 1:
            raise ValueError(
                f"swa_window_size must be >= 1, got {self.swa_window_size}."
            )
        self.chs_num_layers = int(dlite_config["chs_num_layers"])
        self.target_layer_ids = [
            int(value) for value in dlite_config["target_layer_ids"]
        ]
        if len(self.target_layer_ids) != self.chs_num_layers:
            raise ValueError(
                "target_layer_ids length must equal chs_num_layers: "
                f"{len(self.target_layer_ids)} != {self.chs_num_layers}."
            )
        if self.target_layer_ids != sorted(set(self.target_layer_ids)):
            raise ValueError("target_layer_ids must be unique and strictly increasing.")
        invalid_layer_ids = [
            layer_id
            for layer_id in self.target_layer_ids
            if not 0 <= layer_id < int(config.num_target_layers)
        ]
        if invalid_layer_ids:
            raise ValueError(
                f"target_layer_ids {invalid_layer_ids} are outside [0, "
                f"{int(config.num_target_layers) - 1}]."
            )
        default_history_ids = [
            0,
            config.num_target_layers // 2,
            config.num_target_layers - 1,
        ]
        self.history_layer_ids = list(
            dlite_config["history_layer_ids"]
            if self.is_teacher
            else default_history_ids
        )
        if self.history_layer_ids != default_history_ids:
            raise ValueError(
                "history_layer_ids are fixed to target first/middle/last layers "
                f"{default_history_ids}, got {self.history_layer_ids}."
            )

        dlite_config["architecture_version"] = self.architecture_version
        dlite_config["model_role"] = self.model_role
        self.sequential_head_type = str(dlite_config["sequential_head"]).lower()
        self.sequential_rank = int(dlite_config["sequential_rank"])
        if self.sequential_head_type not in SEQUENTIAL_HEAD_TYPES:
            raise ValueError(
                f"Unknown sequential_head={self.sequential_head_type!r}; expected "
                f"one of {SEQUENTIAL_HEAD_TYPES}."
            )
        if self.sequential_rank <= 0:
            raise ValueError(
                f"sequential_rank must be positive, got {self.sequential_rank}."
            )
        self.sequential_head = DLiteSequentialHead(
            head_type=self.sequential_head_type,
            vocab_size=config.vocab_size,
            sequential_rank=self.sequential_rank,
            hidden_size=config.hidden_size,
            max_prediction_length=config.block_size - 1,
        )
        config.dlite_config = dlite_config

        self.layers = nn.ModuleList(
            [
                Qwen3DLiteDecoderLayer(
                    config,
                    layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.mask_token_id = dlite_config.get("mask_token_id", None)

        hidden_size = config.hidden_size
        if self.is_teacher:
            self.history_fuse = nn.Linear(3 * hidden_size, hidden_size, bias=False)
            self.history_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.layer_depth_embedding = nn.Embedding(config.num_target_layers, hidden_size)
        self.context_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        logging.getLogger(__name__).debug(
            f"DLite: architecture_version={self.architecture_version}, "
            f"model_role={self.model_role}, "
            f"swa_window_size={self.swa_window_size}, "
            f"fuse_slots={self.fuse_slot_count}, "
            f"chs_num_layers={self.chs_num_layers}, "
            f"condition_slots={self.condition_slot_count}, "
            f"target_layer_ids={self.target_layer_ids}, "
            f"history_layer_ids={self.history_layer_ids}, "
            f"sequential_head={self.sequential_head_type}, "
            f"sequential_rank={self.sequential_rank}"
        )

        self.post_init()

    @property
    def chs_len_per_block(self) -> int:
        return self.condition_slot_count + (
            self.fuse_slot_count if self.is_teacher else 0
        )

    @property
    def condition_slot_count(self) -> int:
        return self.chs_num_layers

    @property
    def is_teacher(self) -> bool:
        return self.model_role == "swa_teacher"

    @property
    def is_student(self) -> bool:
        return self.model_role == "student"

    @property
    def fuse_slot_count(self) -> int:
        return self.swa_window_size - 1

    @property
    def token_prefix_count(self) -> int:
        return 1

    def build_block_position_ids(
        self,
        anchor_positions: torch.Tensor,
        token_position_ids: torch.Tensor,
        token_keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return role-specific RoPE ids; student queries start at 1 after CHS 0."""
        if token_position_ids.shape != token_keep_mask.shape:
            raise ValueError("token positions and keep mask must have equal shapes")
        pivot = (anchor_positions - 1).clamp(min=0)
        if self.is_teacher:
            context_positions = torch.cat(
                [
                    torch.arange(
                        self.fuse_slot_count,
                        device=anchor_positions.device,
                    ).view(1, 1, -1)
                    + anchor_positions.unsqueeze(-1)
                    - self.swa_window_size,
                    pivot.unsqueeze(-1).expand(-1, -1, self.chs_num_layers),
                ],
                dim=-1,
            ).clamp(min=0)
            draft_positions = token_position_ids
            mask_offsets = torch.arange(
                1, self.block_size, device=anchor_positions.device
            ).view(1, 1, -1)
            draft_positions = torch.cat(
                [draft_positions, anchor_positions.unsqueeze(-1) + mask_offsets],
                dim=-1,
            )
        else:
            has_tokens = token_keep_mask.any(dim=-1)
            sentinel = torch.iinfo(token_position_ids.dtype).max
            first_valid = torch.where(
                token_keep_mask,
                token_position_ids,
                torch.full_like(token_position_ids, sentinel),
            ).amin(dim=-1)
            origin = torch.where(has_tokens, first_valid, pivot)
            local_offset = 0 if self.architecture_version == "dlite_v1" else 1
            local_tokens = torch.where(
                token_keep_mask,
                token_position_ids - origin.unsqueeze(-1) + local_offset,
                torch.zeros_like(token_position_ids),
            )
            context_positions = (
                (pivot - origin)
                .clamp(min=0)
                .unsqueeze(-1)
                .expand(-1, -1, self.chs_num_layers)
            )
            mask_offsets = torch.arange(
                1, self.block_size, device=anchor_positions.device
            ).view(1, 1, -1)
            local_masks = (
                anchor_positions.unsqueeze(-1)
                + mask_offsets
                - origin.unsqueeze(-1)
                + local_offset
            )
            draft_positions = torch.cat([local_tokens, local_masks], dim=-1)
        return (
            context_positions.reshape(anchor_positions.shape[0], -1),
            draft_positions.reshape(anchor_positions.shape[0], -1),
        )

    @property
    def draft_block_len(self) -> int:
        """Parallel draft slots per anchor (1 anchor + remaining MASK tokens)."""
        return self.block_size

    @property
    def draft_query_length(self) -> int:
        return self.token_prefix_count + self.proposal_length

    @property
    def proposal_length(self) -> int:
        """Draft tokens proposed after the anchor; total span is block_size."""
        return self.block_size - 1

    def fuse_history_hidden(self, history_hidden: torch.Tensor) -> torch.Tensor:
        """Fuse ``(..., 3, H)`` target states into one vector per token."""
        if not self.is_teacher:
            raise RuntimeError("History fusion is only available on teacher models.")
        if history_hidden.ndim < 3:
            raise ValueError(
                "history_hidden must have shape (..., 3, H); got "
                f"{tuple(history_hidden.shape)}."
            )
        if history_hidden.shape[-2] != 3:
            raise ValueError(
                "history_hidden must have exactly three selected layers; got shape "
                f"{tuple(history_hidden.shape)}."
            )
        expected_in = self.history_fuse.in_features
        if history_hidden.shape[-1] * 3 != expected_in:
            raise ValueError(
                "history_hidden feature dimension does not match history_fuse; "
                f"got H={history_hidden.shape[-1]} (flattened {history_hidden.shape[-1] * 3}), "
                f"expected {expected_in}."
            )
        flat = history_hidden.flatten(start_dim=-2)
        return self.history_norm(self.history_fuse(flat))

    def _apply_chs_depth_embedding(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Add target-layer identity to current CHS hidden states."""
        _, _, num_chs_slots, hidden_size = target_hidden.shape
        if num_chs_slots != self.chs_num_layers:
            raise ValueError(
                f"Expected {self.chs_num_layers} current CHS slots, got {num_chs_slots}."
            )
        depth_ids = torch.tensor(
            self.target_layer_ids, device=target_hidden.device, dtype=torch.long
        )
        depth_emb = self.layer_depth_embedding(depth_ids).view(
            1, 1, self.chs_num_layers, hidden_size
        )
        return self.context_norm(target_hidden + depth_emb)

    def _build_shared_context_kv(
        self,
        shared_history: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate per-anchor CHS slots and one shared history sequence."""
        if shared_history.ndim != 3:
            raise ValueError(
                "shared_history must have shape (B,T,H), got "
                f"{tuple(shared_history.shape)}."
            )
        current_chs_hidden = self._apply_chs_depth_embedding(target_hidden)
        chs_flat = current_chs_hidden.reshape(
            current_chs_hidden.shape[0], -1, current_chs_hidden.shape[-1]
        )
        if self.is_teacher:
            return torch.cat([shared_history, chs_flat], dim=1)
        return chs_flat

    def _fuse_target_hidden(
        self,
        target_hidden: torch.Tensor,
        history_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Build the configured per-block CHS/history context."""
        bsz, num_blocks, _, _ = target_hidden.shape
        if history_hidden.shape[:2] != (bsz, num_blocks):
            raise ValueError(
                "history_hidden batch/block dimensions must match target_hidden: "
                f"{tuple(history_hidden.shape)} vs {tuple(target_hidden.shape)}."
            )
        current_chs_hidden = self._apply_chs_depth_embedding(target_hidden)
        context_hidden = (
            torch.cat([history_hidden, current_chs_hidden], dim=2)
            if self.is_teacher
            else current_chs_hidden
        )
        return context_hidden.reshape(
            bsz, num_blocks * context_hidden.shape[2], current_chs_hidden.shape[-1]
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        history_hidden: Optional[torch.Tensor] = None,
        shared_history: Optional[torch.Tensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        if shared_history is not None:
            if target_hidden is None or target_hidden.ndim != 4:
                raise ValueError(
                    "shared_history training requires target_hidden shaped (B,N,S,H)."
                )
            target_hidden = self._build_shared_context_kv(shared_history, target_hidden)
        else:
            assert target_hidden is not None and target_hidden.ndim == 4
            assert history_hidden is not None and history_hidden.ndim == 4
            target_hidden = self._fuse_target_hidden(target_hidden, history_hidden)
        noise_len = hidden_states.shape[1]
        if position_ids.shape[1] != noise_len:
            draft_pos_ids = position_ids[:, -noise_len:]
        else:
            draft_pos_ids = position_ids

        rotary_position_ids = (
            rotary_position_ids if rotary_position_ids is not None else draft_pos_ids
        )
        # Qwen3RotaryEmbedding only reads x.device and x.dtype. Passing the
        # existing activation avoids allocating a dense (B, KV_LEN, H) dummy;
        position_embeddings = self.rotary_emb(hidden_states, rotary_position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=draft_pos_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)
