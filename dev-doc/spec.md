# DLite training contract

Reference revisions: SpecForge f586b1e, DLite e0aa3d5, LiteSpec 4e87f22.

## Model and features

- Register `DLiteDraftModel` and `training.strategy: dlite` independently of DFlash.
- Dense Qwen3-style draft uses Qwen3.5 text dimensions and full one-dimensional
  RoPE. Student CHS positions are zero, query positions are 1..block_size.
- CHS uses target features at anchor-1. Teacher additionally uses fused target
  layers [0, depth//2, depth-1] at positions [anchor-window, anchor-2].
- Queries attend only their own block and CHS, plus teacher history. The RNN
  starts at zero for every block and consumes ground-truth previous tokens.
- Capture intermediate CHS/history layers through the existing dflash transport;
  reconstruct the final layer from post-norm target_last_hidden_states. Never
  include the final layer in the auxiliary-layer request.
- Freeze target embedding/head. Persist the resolved in-vocabulary MASK row.

## Training

- Teacher: 6 epochs, CE weight .1, probability L1 weight 1, gamma 4.
- Student: random backbone; copy only teacher sequential head. Stage 1 (1 epoch)
  optimizes KL(teacher||student), with head updates disabled. Transition (1 epoch)
  unfreezes head updates and cosine-mixes KL and supervision. Stage 2 (6 epochs)
  optimizes CE + probability L1. Optional base-head CE defaults to zero.
- Supervised CE labels are target prefill argmax tokens; RNN inputs remain data
  tokens. Probability L1 has no factor of one half.
- One optimizer and scheduler cover all student phases. At cumulative epoch E,
  step boundary is floor(N*E/(DP*batch_size*accumulation_steps)). Each phase must
  contain a complete optimizer window. max_steps truncates without rescheduling.
- Loss numerators and denominators are additive across ranks/accumulation.
  Transition mixes both numerators and denominators with the same weights.
- Full-vocabulary projections use existing checkpointed block reductions.
- Stage 1 clears head gradients before optimizer clipping/update, so head Adam
  state and weights remain unchanged. All student parameters are registered at
  optimizer construction. Frozen teacher is excluded from FSDP and optimizer.
- Keep capture layout fixed across student phases; Stage 2 stops calling teacher.

## Interfaces and acceptance

Use standard `specforge train --config` and `specforge export --to hf`.
Teacher exports initialize a fresh two-stage run; student exports load in the
official DLite class with matching config, state keys, and block logits.
Keep managed-local resume unsupported. No new test files. Validate configuration,
imports, formatting, fixed-input official parity, and existing relevant checks.
GPU acceptance requires teacher and all student phases, continuous LR/steps,
correct head updates, bounded memory, checkpoint/export, and clean shutdown.
Do not claim GPU parity, convergence, throughput, or serving support from static
or CPU checks.
