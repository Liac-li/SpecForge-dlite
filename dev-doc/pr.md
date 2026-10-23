# Add DLite online teacher and two-stage student training

## Motivation

Add DLite training for Qwen3.5-9B text through SpecForge's existing
`specforge train` lifecycle. The exported student uses the official DLite
architecture and checkpoint keys. This PR covers training and HF export;
SGLang DLite serving, offline/multimodal training and full online resume are
not included.

## Modifications

- Register the dense DLite teacher/student model and streaming feature provider.
  Preserve CHS depth embeddings, teacher-only history, local student positions
  and the recurrent sequential head from the official implementation.
- Reuse SGLang/Mooncake capture, assembling the final target layer from its
  post-norm feature. Keep frozen target tables and teacher outside FSDP.
- Add teacher supervision and continuous student KL, cosine transition and
  supervised phases. Use one optimizer/scheduler and exclude head updates/Adam
  state during Stage 1. Phase boundaries follow the existing online step plan.
- Reuse additive loss normalization and checkpointed block reductions. Persist
  resolved draft config so HF export needs no separately reconstructed student
  JSON or MASK ID.
- Add two managed-local 4×80 GB recipes, configuration documentation and runbook.

## Related Issues

Reference implementations: SJTU-DAI-Sys/LiteSpec at 4e87f22 and
SJTU-DAI-Sys/DLite at e0aa3d5. Ported files retain MIT attribution.

## Accuracy Test

CPU comparison with the official training implementation covered teacher,
Stage 1, transition and Stage 2, including padding, loss masks, position decay,
base-head CE and chunked projection. Maximum FP32 absolute differences were
2.3842e-7 for loss and 1.4902e-8 for gradients; BF16 differences were 1.1921e-7
and 4.8828e-4. This used CPU SDPA/dense masks, not CUDA/FlexAttention.

Six CPU TrainerCore/BF16Optimizer steps verified the Stage 1 head remained
unchanged with no Adam state, followed by head updates at transition.
The exported student loaded in official DLite with exact keys/weights and
bitwise-equal BF16 single-block outputs.

Actual Qwen3.5-9B target feature parity and multi-GPU FSDP validation are pending.
Keep this PR in Draft until those checks are recorded. Local dependencies were
torch 2.14.0 / transformers 5.17.0; repository pins were not changed.

Existing relevant checks: 230 passed, 11 skipped, 534 subtests passed, covering
algorithm/config contracts, Trainer, schedules, CLI, export, chunking and the
optimizer (including CPU Gloo checks).

## Benchmark & Profiling

No throughput, convergence or serving-speedup claim. The four-GPU recipe is an
initial configuration; server memory and performance measurements are pending.

## Checklist

- [x] Format changed code with repository Black/isort versions.
- [x] Update documentation and example recipes.
- [x] Run existing relevant checks and direct official-reference comparisons.
- [ ] New unit tests (not included; existing algorithm-catalog expectations
  were updated).
- [ ] Validate pinned Linux/CUDA dependencies and the 4×80 GB managed-local run.
- [ ] Record actual target feature alignment, phase transitions, checkpoint and
  shutdown evidence from the GPU server.
