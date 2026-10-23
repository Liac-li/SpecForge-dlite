# DLite on Qwen3.5-9B

This recipe trains a DLite SWA teacher and then a history-free student using
SpecForge's online managed-local runtime. It targets four 80 GB GPUs: GPU 0
captures target features with TP1; GPUs 1-3 train the draft with FSDP. Follow the
[disaggregated setup guide](../sections/basic_usage/disaggregated_training.md)
to install the matching SGLang capture patch/plugin and Mooncake first.

The implementation follows [LiteSpec](https://github.com/SJTU-DAI-Sys/LiteSpec/tree/4e87f22)
for training and exports weights for the
[DLite inference package](https://github.com/SJTU-DAI-Sys/DLite/tree/e0aa3d5).
Ported model/mask code retains its MIT license and upstream attribution.
This is training and checkpoint compatibility support, not SGLang DLite serving.
Offline, multimodal, and managed-local full resume are outside this recipe.

## Teacher

Run from the SpecForge repository root. Set the model path and a text training
JSONL path in the YAML, or override them on the command line. Data uses the
existing SpecForge conversation format, Qwen3.5 chat template, and assistant
loss mask; online means target **prefill feature capture**, not on-policy
completion generation. Prepare/regenerate the response data separately if needed.

```bash
specforge train --config examples/configs/online/disaggregated/managed-local/qwen3.5-9b-dlite-teacher.yaml \
  model.target_model_path=/models/Qwen3.5-9B \
  data.train_data_path=/data/train.jsonl

specforge export --to hf \
  --checkpoint outputs/qwen3.5-9b-dlite-teacher \
  --output-dir outputs/qwen3.5-9b-dlite-teacher-hf
```

Use `--plan` on the train command to inspect the processes before launching.
The first recipe uses five dense draft layers, block size eight (one anchor
and seven predictions), twelve CHS layers, RNN rank 512, and history window 512.
The target keeps its native Qwen3.5 architecture. Draft hidden/FFN/head/vocabulary
dimensions match the target text config, with full one-dimensional Qwen3 RoPE.

CHS uses selected layers at anchor-1; the teacher also sees fused target
first/middle/last-layer features at positions anchor-512 through anchor-2.
Auxiliary capture excludes the final layer: final CHS/history and target logits
use the separately captured post-norm state. The MASK row is resolved from the
target tokenizer/embedding without resizing and saved in the checkpoint config.

Teacher defaults: six epochs, LR 4e-4, warmup ratio .04, CE weight .1,
probability L1 weight 1, position decay gamma 4. CE labels are target prefill
argmax tokens. RNN inputs are the recorded preceding tokens, with a fresh zero
state for each block. `dlite_base_ce_weight` optionally enables supervision
through the frozen target LM head; its default is zero.

## Student

```bash
specforge train --config examples/configs/online/disaggregated/managed-local/qwen3.5-9b-dlite-two-stage.yaml \
  model.target_model_path=/models/Qwen3.5-9B \
  data.train_data_path=/data/train.jsonl \
  model.dlite_teacher_checkpoint_path=./outputs/qwen3.5-9b-dlite-teacher-hf

specforge export --to hf \
  --checkpoint outputs/qwen3.5-9b-dlite-two-stage \
  --output-dir outputs/qwen3.5-9b-dlite-student-hf
```

The student inherits the teacher architecture and MASK row, initializes a fresh
backbone, and copies only its sequential head. Do not specify
`model.draft_model_config` or `model.draft_checkpoint_path` for `two_stage`.
DLite exports use the resolved configuration embedded in the training checkpoint;
omit `--draft-config`. Only the final student is accepted by official DLite inference.

| Phase | Epoch budget | Objective | Student head updates |
| --- | --- | --- | --- |
| Stage 1 | 1 | KL(teacher \|\| student) | Frozen, including Adam state |
| Transition | 1 | Cosine blend of KL and supervision | Enabled |
| Stage 2 | 6 | CE + probability L1 | Enabled |

One optimizer and scheduler cover all eight epochs. Set `training.num_epochs`
to `dlite_stage1_epochs + 1 + dlite_stage2_epochs`. For N eligible prompts and
global window G = DP × batch size × accumulation steps, cumulative phase
boundaries are floor(N × cumulative epochs / G). Each phase needs at least one
window. The transition starts with pure KL and ends with pure supervision;
when it contains only one window, it uses pure KL, matching the official
disaggregated implementation. The stream retains SpecForge's existing dispatch
and tail policy; these are optimizer-step budgets, not strict data-epoch barriers.

The producer records boundaries in its schedule file even for short runs.
`training.max_steps` truncates execution without shortening the full LR/phase
schedule. It therefore does not force a short run to enter every phase. Use a
small representative dataset to inspect both transitions. Keep output/run IDs
distinct for smoke and full runs. Checkpoints are saved, but restarting with
`training.resume_from` is unsupported in this first online integration.

## Memory and inspection

Both recipes expose GPU IDs, capture TP, trainer process count, port, sequence
length, batch size, and anchor count. Full-vocabulary objectives use
`training.objective_chunk_blocks: 16` with activation recomputation; each chunk
keeps a complete recurrent block. The frozen teacher and target tables are
replicated outside FSDP; only student parameters enter its optimizer. Stage 2
stops calling the teacher but keeps its allocation, avoiding FSDP reconstruction.

Inspect `dlite/phase` (0 teacher, 1 Stage 1, 2 transition, 3 Stage 2),
`dlite/kl_scale`, component losses, and
`dlite/teacher_forced_expected_accept_len`. The last metric uses target/draft
distributions under recorded prefixes; it is not measured serving acceptance
length. Stage 1 logs KL rather than target-vs-draft acceptance.

Before a production run, verify actual SGLang/HF feature alignment on the server,
all phase transitions, unchanged head weights/Adam state in Stage 1, head updates
afterwards, continuous LR, checkpoint/export round-trip and clean stack shutdown.
The 4×80 GB recipe is an initial memory configuration, not a measured throughput
or convergence claim. CPU comparison results do not establish CUDA/FSDP parity.
