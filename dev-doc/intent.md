# DLite training in SpecForge

Implement the official DLite teacher and two-stage student training chain for
Qwen3.5-9B text inputs through SpecForge's unified training entry point. The
first recipe targets online managed-local training on four 80 GB GPUs, with
one TP1 capture GPU and three FSDP trainer GPUs.

Reuse SpecForge's configuration, capture, transport, trainer, optimizer and
checkpoint lifecycle. Export student checkpoints compatible with the official
DLite inference package. Do not add a separate trainer, new test files, offline
training, multimodal input, SGLang serving, or managed-local full resume.

Approved defaults: five draft layers, block size eight, twelve CHS layers,
RNN rank 512, teacher history window 512, and dlite_v2 positions. Student phases
switch by optimizer step using the existing finite online prompt schedule,
not strict data-epoch barriers. GPU validation requires the user's server.
