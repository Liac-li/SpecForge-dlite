# Copyright (c) 2025 sgl-project
# SPDX-License-Identifier: MIT
# DLite semantics adapted from SJTU-DAI-Sys/LiteSpec (4e87f22), core/dlite.py.
"""DLite block preparation and memory-bounded training objectives."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.modeling.draft.dlite_mask import create_dlite_mask


def position_weights(weights, gamma):
    if gamma is None or gamma <= 0:
        return weights.float()
    offsets = torch.arange(weights.shape[-1], device=weights.device)
    return weights.float() * torch.exp(-offsets.float() / gamma)


class OnlineDLiteModel(nn.Module):
    def __init__(
        self,
        draft_model,
        embed_tokens,
        lm_head,
        *,
        teacher_draft=None,
        capture_layers,
        training
    ):
        super().__init__()
        self.draft_model = draft_model
        self.embed_tokens = embed_tokens.requires_grad_(False)
        self.lm_head = lm_head.requires_grad_(False)
        self.teacher_draft = teacher_draft
        if teacher_draft is not None:
            teacher_draft.requires_grad_(False).eval()
        self.capture_layers = list(capture_layers)
        self.training_config = training
        self.phase_boundaries = None

    def configure_online_schedule(self, schedule):
        if self.training_config.dlite_mode == "two_stage":
            self.phase_boundaries = tuple(schedule["dlite_phase_boundaries"])

    def phase(self, step):
        if self.training_config.dlite_mode == "teacher":
            return "teacher", 0.0, 1.0
        if self.phase_boundaries is None:
            raise RuntimeError(
                "DLite two-stage training requires the producer schedule"
            )
        first, transition, _ = self.phase_boundaries
        if step < first:
            return "stage1", 1.0, 0.0
        if step < transition:
            # Match the official disaggregated one-window transition: KL only.
            progress = (step - first) / max(transition - first - 1, 1)
            kl_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            return "transition", kl_scale, 1.0 - kl_scale
        return "stage2", 0.0, 1.0

    def _layers(self, hidden_states, last_hidden):
        width = self.draft_model.config.hidden_size
        if hidden_states.shape[-1] != len(self.capture_layers) * width:
            raise ValueError(
                "DLite auxiliary feature width does not match capture layers"
            )
        layers = dict(zip(self.capture_layers, hidden_states.split(width, dim=-1)))
        layers[self.draft_model.config.num_target_layers - 1] = last_hidden
        return layers

    def _sample_anchors(self, loss_mask, block_size):
        # Like the reference, require a complete block and two supervised tokens.
        candidates = loss_mask.shape[1] - block_size + 1
        if candidates <= 0:
            raise ValueError("DLite sequence is shorter than block_size")
        valid = (loss_mask[:, :candidates] > 0.5) & (
            loss_mask[:, 1 : candidates + 1] > 0.5
        )
        counts = valid.sum(-1)
        count = min(self.training_config.num_anchors, int(counts.max().item()))
        if count == 0:
            raise ValueError("DLite batch has no complete supervised anchor block")
        scores = torch.rand(valid.shape, device=valid.device).masked_fill(~valid, 2.0)
        anchors = scores.argsort(-1)[:, :count].sort(-1).values
        keep = valid.gather(1, anchors)
        return anchors, keep

    def _backbone(self, draft, layers, anchors, keep, queries, seq_len):
        batch, blocks = anchors.shape
        pivot = (anchors - 1).clamp(min=0)
        rows = torch.arange(batch, device=anchors.device)[:, None]
        chs = torch.stack(
            [layers[i][rows, pivot] for i in draft.target_layer_ids], dim=2
        )
        context_pos, query_pos = draft.build_block_position_ids(
            anchors, anchors.unsqueeze(-1), keep.unsqueeze(-1)
        )
        if draft.is_teacher:
            history = draft.fuse_history_hidden(
                torch.stack([layers[i] for i in draft.history_layer_ids], dim=2)
            )
            history_pos = torch.arange(seq_len, device=anchors.device).expand(batch, -1)
            chs_pos = context_pos.reshape(batch, blocks, -1)[
                :, :, -draft.chs_num_layers :
            ]
            rotary = torch.cat(
                [history_pos, chs_pos.reshape(batch, -1), query_pos], dim=-1
            )
        else:
            history = queries.new_empty(batch, 0, queries.shape[-1])
            rotary = torch.cat([context_pos, query_pos], dim=-1)
        output = draft(
            position_ids=query_pos,
            rotary_position_ids=rotary,
            noise_embedding=queries.reshape(batch, -1, queries.shape[-1]),
            target_hidden=chs,
            shared_history=history,
            attention_mask=create_dlite_mask(
                anchors,
                keep,
                seq_len=seq_len,
                block_size=draft.block_size,
                chs_slots=draft.chs_num_layers,
                history_window=draft.swa_window_size if draft.is_teacher else None,
                backend=draft.config._attn_implementation,
            ),
            kernel_options={"BACKEND": "TRITON"},
        )
        return output.reshape(batch, blocks, draft.block_size, -1)[:, :, 1:]

    def forward(
        self,
        input_ids,
        loss_mask,
        hidden_states,
        target_last_hidden_states,
        *,
        global_step=0,
        collect_detailed_metrics=True,
        anchors=None,
        keep=None
    ):
        draft, t = self.draft_model, self.training_config
        phase, kl_scale, supervised_scale = self.phase(global_step)
        if anchors is None:
            anchors, keep = self._sample_anchors(loss_mask, draft.block_size)
        layers = self._layers(hidden_states, target_last_hidden_states)
        batch, blocks = anchors.shape
        rows = torch.arange(batch, device=input_ids.device)[:, None, None]
        offsets = torch.arange(draft.block_size, device=input_ids.device)
        positions = anchors.unsqueeze(-1) + offsets
        tokens = input_ids[rows, positions]
        weights = loss_mask[rows, positions][:, :, 1:].float() * keep.unsqueeze(-1)
        query_ids = torch.full_like(tokens, draft.mask_token_id)
        query_ids[:, :, 0] = tokens[:, :, 0]
        queries = self.embed_tokens(query_ids)
        queries[:, :, :1] *= keep[:, :, None, None]
        prediction = self._backbone(
            draft, layers, anchors, keep, queries, input_ids.shape[1]
        )
        teacher_prediction = None
        if phase in ("stage1", "transition"):
            self.teacher_draft.eval()
            with torch.no_grad():
                teacher_prediction = self._backbone(
                    self.teacher_draft,
                    layers,
                    anchors,
                    keep,
                    queries,
                    input_ids.shape[1],
                )
        target_prediction = target_last_hidden_states[rows, positions[:, :, :-1]]

        def objective(pred, teacher_pred, target_pred, prev_ids, raw_weights):
            active = raw_weights > 0
            # Sanitize padding before the recurrent head, not after softmax.
            pred = torch.where(active.unsqueeze(-1), pred, 0)
            prev_ids = torch.where(active, prev_ids, 0)
            latent = draft.sequential_head.forward_teacher_forcing(
                hidden_states=pred, prev_token_ids=prev_ids
            )
            logits = draft.sequential_head.project_logits(latent[active]).float()
            zero = logits.sum() * 0.0
            kl_num = ce_num = l1_num = base_num = zero
            kl_den = sup_den = final_den = base_den = zero.detach()
            correct = token_count = eal_num = block_count = zero.detach()
            if teacher_pred is not None:
                with torch.no_grad():
                    teacher_latent = (
                        self.teacher_draft.sequential_head.forward_teacher_forcing(
                            hidden_states=torch.where(
                                active.unsqueeze(-1), teacher_pred, 0
                            ),
                            prev_token_ids=prev_ids,
                        )
                    )
                    teacher_logits = self.teacher_draft.sequential_head.project_logits(
                        teacher_latent[active]
                    ).float()
                    teacher_logp = F.log_softmax(teacher_logits, -1)
                    teacher_p = teacher_logp.exp()
                kw = position_weights(raw_weights, t.dlite_stage1_loss_decay_gamma)[
                    active
                ]
                kl_num = (
                    (teacher_p * (teacher_logp - F.log_softmax(logits, -1))).sum(-1)
                    * kw
                ).sum()
                kl_den = kw.sum()
            if phase != "stage1":
                with torch.no_grad():
                    target_logits = self.lm_head(target_pred[active]).float()
                    target_p = F.softmax(target_logits, -1)
                    labels = target_logits.argmax(-1)
                final_weights = position_weights(raw_weights, t.loss_decay_gamma)[
                    active
                ]
                base_weights = position_weights(
                    raw_weights, t.dlite_base_ce_decay_gamma
                )[active]
                ce_num = (
                    F.cross_entropy(logits, labels, reduction="none") * final_weights
                ).sum()
                l1 = (F.softmax(logits, -1) - target_p).abs().sum(-1)
                l1_num = (l1 * final_weights).sum()
                if t.dlite_base_ce_weight:
                    base_logits = self.lm_head(pred[active]).float()
                    base_num = (
                        F.cross_entropy(base_logits, labels, reduction="none")
                        * base_weights
                    ).sum()
                fw = t.dlite_final_ce_weight + t.dlite_l1_weight
                final_den = final_weights.sum()
                base_den = base_weights.sum()
                sup_den = (fw * final_den + t.dlite_base_ce_weight * base_den) / (
                    fw + t.dlite_base_ce_weight
                )
                if collect_detailed_metrics:
                    correct = (logits.detach().argmax(-1) == labels).float().sum()
                    token_count = active.sum()
                    acceptance = torch.zeros_like(raw_weights)
                    acceptance[active] = (1 - 0.5 * l1.detach()).clamp(0, 1)
                    valid_blocks = active.any(-1)
                    eal_num = (
                        (1 + acceptance.cumprod(-1).sum(-1)) * valid_blocks
                    ).sum()
                    block_count = valid_blocks.sum()
            supervised = (
                t.dlite_final_ce_weight * ce_num
                + t.dlite_l1_weight * l1_num
                + t.dlite_base_ce_weight * base_num
            )
            numerator = (
                kl_scale * t.dlite_stage1_kl_weight * kl_num
                + supervised_scale * supervised
            )
            denominator = kl_scale * kl_den + supervised_scale * sup_den
            return (
                numerator,
                denominator.detach(),
                kl_num.detach(),
                kl_den.detach(),
                ce_num.detach(),
                l1_num.detach(),
                base_num.detach(),
                final_den.detach(),
                base_den.detach(),
                correct,
                token_count,
                eal_num,
                block_count,
            )

        # Flatten batch/block only. Recurrent positions stay together in every chunk.
        values = checkpointed_chunk_reduce(
            objective,
            prediction.flatten(0, 1),
            (
                teacher_prediction.flatten(0, 1)
                if teacher_prediction is not None
                else None
            ),
            target_prediction.flatten(0, 1),
            tokens[:, :, :-1].flatten(0, 1),
            weights.flatten(0, 1),
            chunk_size=t.objective_chunk_blocks,
        )
        (
            numerator,
            denominator,
            kl,
            kd,
            ce,
            l1,
            base,
            fd,
            bd,
            correct,
            count,
            eal,
            bc,
        ) = values
        ratios = {}
        if collect_detailed_metrics:
            if phase in ("stage1", "transition"):
                ratios["dlite/kl"] = (kl, kd)
            if phase != "stage1":
                ratios.update(
                    {
                        "dlite/ce": (ce, fd),
                        "dlite/l1": (l1, fd),
                        "dlite/base_ce": (base, bd),
                        "accuracy": (correct, count),
                        "dlite/teacher_forced_expected_accept_len": (eal, bc),
                    }
                )
        return numerator, denominator, ratios
