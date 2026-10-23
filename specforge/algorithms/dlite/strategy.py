"""DLite phase selection through SpecForge's existing training strategy port."""

from specforge.training.strategies.base import DraftTrainStrategy, StepOutput


class DLiteTrainStrategy(DraftTrainStrategy):
    name = "dlite"
    required_features = {
        "input_ids",
        "loss_mask",
        "hidden_states",
        "target_last_hidden_states",
    }

    def __init__(self, model):
        self.model = model
        self.current_phase = None

    @property
    def inner(self):
        return getattr(self.model, "module", self.model)

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx=None):
        self.validate_batch(batch)
        step = ctx.global_step if ctx else 0
        self.current_phase, kl_scale, _ = self.inner.phase(step)
        device = next(self.model.parameters()).device
        numerator, denominator, ratios = self.model(
            **{
                key: batch.tensors[key].to(device, non_blocking=True)
                for key in self.required_features
            },
            global_step=step,
            collect_detailed_metrics=ctx.collect_detailed_metrics if ctx else True,
        )
        return StepOutput(
            loss=numerator / denominator.clamp_min(1e-8),
            loss_terms=(numerator, denominator),
            ratio_metrics=ratios,
            metrics={
                "dlite/phase": ("teacher", "stage1", "transition", "stage2").index(
                    self.current_phase
                ),
                "dlite/kl_scale": kl_scale,
            },
        )

    def before_optimizer_step(self):
        if self.current_phase == "stage1":
            # Keep the backbone gradient through the head, but do not create
            # Adam moments or apply weight decay to the copied head in Stage 1.
            for parameter in self.inner.draft_model.sequential_head.parameters():
                parameter.grad = None

    def checkpoint_state_filter(self, state_dict):
        prefix = "draft_model."
        return {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
