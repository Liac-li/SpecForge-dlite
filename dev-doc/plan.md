# Implementation plan

1. Add DLite draft, sequential head and mask semantics from official sources;
   preserve checkpoint names and license attribution.
2. Add algorithm provider, dense target-derived configuration, target tables,
   CHS/history reconstruction and streaming-only contract.
3. Add chunked supervised/distillation objectives and DLite strategy using
   existing loss_terms normalization. Add a no-op optimizer-boundary hook for
   Stage 1 head freezing and exclude frozen teacher from FSDP.
4. Extend producer schedule metadata with DLite phase boundaries, forwarding
   them to the strategy without another training loop or transport protocol.
5. Add model JSON, teacher/two-stage managed-local YAML, export compatibility
   and training recipe documentation.
6. Run static checks, CPU fixed-input comparisons against official sources and
   relevant existing checks; record missing GPU evidence and PR status.

## Progress

- Implementation complete: registered draft and streaming-only algorithm;
  teacher/student masks and objectives; producer-derived phase schedule; shared
  optimizer with Stage 1 head update suppression; checkpoint-contained config
  and official-compatible HF export; two four-GPU recipes and documentation.
- Existing catalog assertions were refreshed for the seventh algorithm. No new
  test files or test cases were added.
- The public export CLI now permits omitting `--draft-config` for DLite: the
  saved resolved config is authoritative, including student role and MASK row.
- Full-vocabulary chunking preserves recurrent blocks. Component metrics use
  their own final/base position-weight denominators, while the optimization
  objective keeps the reference coefficient-weighted denominator.

## Validation evidence (2026-10-23)

- Local environment: macOS CPU, uv-base, torch 2.14.0, transformers 5.17.0.
  Repository pins (torch 2.13.0 / transformers 5.12.1 / SGLang 0.5.18) and CUDA
  have not been exercised here; no dependency pins were changed.
- Final existing algorithm/config/trainer/schedule/export/CLI/chunking/optimizer
  check run: 230 passed, 11 skipped, 534 subtests passed. Local Gloo socket checks
  ran outside the sandbox. Skips are retained platform/dependency requirements.
- Fixed-input reference comparison against LiteSpec 4e87f22: teacher, Stage 1,
  transition and Stage 2; padded blocks, an interior masked position, nonzero
  base-head CE, distinct positional decays, and one-block objective chunks.
  Both implementations used CPU SDPA with the official mask predicate expanded
  densely, so this checks semantic math rather than CUDA/FlexAttention kernels.
  Maximum FP32 loss absolute difference 2.3842e-7, gradient difference 1.4902e-8;
  maximum BF16 loss difference 1.1921e-7, gradient difference 4.8828e-4.
- Six real CPU TrainerCore/BF16Optimizer updates across all student phases:
  copied head remained bitwise unchanged with zero Adam states during Stage 1;
  all six head parameter states appeared at transition, with subsequent updates.
- HF student export loaded through official DLite e0aa3d5 with no missing or
  unexpected keys. Stored weights and fixed single-block outputs were bitwise
  identical in BF16 CPU SDPA.
- Both recipes resolve as managed_supervisor plans with two worker commands and
  two services, using synthetic target config metadata matching Qwen3.5-9B;
  no target weights, capture services or GPUs were loaded for this check.
- AST, Ruff F/E9, Black/isort, and diff-whitespace inspection performed.

## Outstanding external validation and delivery

- No GPU/server access supplied. Still required: actual Qwen3.5-9B capture/HF
  feature parity, pinned dependencies, three-rank FSDP head-freeze behavior,
  four-GPU memory/phase/checkpoint lifecycle, and normal managed-stack cleanup.
- No formal convergence, acceptance-speedup, or serving claim is supported.
- GitHub CLI/auth configuration and a browser are unavailable in this session.
  Remote PR submission is not completed; prepare a Draft PR using the adjacent
  PR description after connecting GitHub. Working branch: dev/dlite.
