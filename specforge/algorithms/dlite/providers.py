"""DLite registration, configuration and model assembly."""

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
    online_needs_input_tools,
)
from specforge.algorithms.common.hidden_states_data import build_dspark_collator
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepProvider,
    TargetDerivedDraftDefaults,
    make_registration,
)
from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
)
from specforge.data.loss_mask import has_consecutive_supervised_tokens


def populate_config(payload, target_config, cfg):
    from specforge.modeling.draft.dlite import build_target_layer_ids
    from specforge.training.model_loading import load_draft_config_source

    if cfg.training.dlite_mode == "two_stage":
        teacher = load_draft_config_source(
            cfg.model.dlite_teacher_checkpoint_path, cache_dir=cfg.model.cache_dir
        )
        if (
            teacher.architectures != ["DLiteDraftModel"]
            or teacher.dlite_config["model_role"] != "swa_teacher"
        ):
            raise ValueError(
                "DLite two_stage requires an exported swa_teacher checkpoint"
            )
        payload.clear()
        payload.update(teacher.to_dict())
        method = dict(payload["dlite_config"])
        method["model_role"] = "student"
        method.pop("swa_window_size", None)
        method.pop("history_layer_ids", None)
        payload["dlite_config"] = method
    else:
        depth = int(target_config.num_hidden_layers)
        payload.update(
            block_size=8,
            num_target_layers=depth,
            layer_types=["full_attention"] * payload["num_hidden_layers"],
            rope_theta=(getattr(target_config, "rope_parameters", None) or {}).get(
                "rope_theta", 10000000.0
            ),
            dlite_config={
                "architecture_version": "dlite_v2",
                "model_role": "swa_teacher",
                "chs_num_layers": 12,
                "target_layer_ids": build_target_layer_ids(depth, 12),
                "sequential_head": "rnn",
                "sequential_rank": 512,
                "swa_window_size": 512,
                "history_layer_ids": [0, depth // 2, depth - 1],
            },
        )
        # The dense draft has full 1-D RoPE, not Qwen3.5's partial/multimodal RoPE.
        payload["rope_scaling"] = None


def validate_config(cfg, draft_config):
    role = "swa_teacher" if cfg.training.dlite_mode == "teacher" else "student"
    if draft_config.dlite_config.get("model_role") != role:
        raise ValueError(f"DLite {cfg.training.dlite_mode} requires model_role={role}")
    if draft_config.block_size < 2:
        raise ValueError("DLite block_size must be at least two")
    if (
        list(draft_config.layer_types)
        != ["full_attention"] * draft_config.num_hidden_layers
    ):
        raise ValueError("DLite uses a dense full-attention draft backbone")


def resolve_capture_layers(cfg, draft_config, target_config):
    from specforge.modeling.target.target_utils import target_text_config

    target = target_text_config(target_config)
    draft = draft_config if isinstance(draft_config, dict) else draft_config.to_dict()
    for field in ("hidden_size", "vocab_size"):
        if draft[field] != getattr(target, field):
            raise ValueError(f"DLite draft {field} must match the target text model")
    depth = int(target.num_hidden_layers)
    if draft["num_target_layers"] != depth:
        raise ValueError("DLite num_target_layers must match the target")
    selected = set(draft["dlite_config"]["target_layer_ids"])
    selected.update((0, depth // 2, depth - 1))
    # HF's final-layer feature includes final norm. SGLang transports it
    # separately, so do not request it as an auxiliary decoder-layer output.
    return sorted(selected - {depth - 1})


def minimum_loss_tokens(cfg, draft_config):
    draft = draft_config if isinstance(draft_config, dict) else draft_config.to_dict()
    # Together with adjacency filtering, ensures a complete anchor block.
    return int(draft["block_size"])


def build_draft(cfg, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    return build_registered_draft(cfg, draft_config)


def build_training_model(cfg, draft_model, draft_config, target_config, tokenizer):
    import torch

    from specforge.algorithms.dlite.model import OnlineDLiteModel
    from specforge.algorithms.model_providers import AlgorithmModelParts
    from specforge.modeling.draft.dlite import DLiteDraftModel
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
    from specforge.training.model_utils import resolve_mask_token_id
    from specforge.utils import get_local_device

    device, dtype = get_local_device(), getattr(torch, cfg.model.torch_dtype)
    teacher = None
    if cfg.training.dlite_mode == "two_stage":
        teacher = (
            DLiteDraftModel.from_pretrained(
                cfg.model.dlite_teacher_checkpoint_path,
                torch_dtype=dtype,
                attn_implementation=cfg.training.attention_backend,
            )
            .to(device)
            .requires_grad_(False)
            .eval()
        )
        draft_model.sequential_head.load_state_dict(
            teacher.sequential_head.state_dict(), strict=True
        )
    saved_mask = draft_config.dlite_config.get("mask_token_id")
    if (
        teacher is not None
        and cfg.model.mask_token_id is not None
        and cfg.model.mask_token_id != saved_mask
    ):
        raise ValueError("two_stage must preserve the teacher MASK token")
    mask_id = resolve_mask_token_id(
        explicit=(
            cfg.model.mask_token_id
            if cfg.model.mask_token_id is not None
            else saved_mask
        ),
        tokenizer=tokenizer,
        embedding_vocab_size=draft_config.vocab_size,
    )
    draft_model.mask_token_id = mask_id
    draft_config.dlite_config["mask_token_id"] = mask_id
    tables = TargetEmbeddingsAndHead.from_pretrained(
        cfg.model.target_model_path,
        embed_key=cfg.model.embedding_key,
        lm_head_key=cfg.model.lm_head_key,
        cache_dir=cfg.model.cache_dir,
        device=device.type,
        dtype=dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    layers = resolve_capture_layers(cfg, draft_config, target_config)
    model = OnlineDLiteModel(
        draft_model,
        tables.embed_tokens,
        tables.lm_head,
        teacher_draft=teacher,
        capture_layers=layers,
        training=cfg.training,
    ).to(device=device, dtype=dtype)
    return AlgorithmModelParts(model=model, capture_layers=layers)


def build_step(wrapped_model, *, target_head=None):
    from specforge.algorithms.dlite.strategy import DLiteTrainStrategy

    return DLiteTrainStrategy(wrapped_model)


def resume_contract(cfg, draft_model, training_model):
    return {
        "draft_config": draft_model.config.to_dict(),
        "dlite_config": dict(draft_model.config.dlite_config),
        "dlite_training": {
            key: value
            for key, value in cfg.training.model_dump().items()
            if key.startswith("dlite_")
            or key in ("loss_decay_gamma", "num_anchors", "num_epochs")
        },
        "dlite_teacher_checkpoint_path": cfg.model.dlite_teacher_checkpoint_path,
    }


def create_registration():
    ready = {"input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"}
    spec = AlgorithmSpec(
        name="dlite",
        draft=DraftRequirement(
            compatible_architectures={"DLiteDraftModel"},
            default_architecture="DLiteDraftModel",
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.STREAMING, modality="text", required_tensors=ready
            ),
        ),
        capabilities=AlgorithmCapabilities(attention_backends={"flex_attention"}),
    )
    providers = AlgorithmProviders(
        algorithm_name="dlite",
        step=StepProvider(
            build=build_step,
            options=empty_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture="DLiteDraftModel",
                apply_overrides=validate_config,
                target_defaults=TargetDerivedDraftDefaults(
                    model_type="qwen3", num_hidden_layers=5, populate=populate_config
                ),
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=online_needs_input_tools,
            default_dataloader_num_workers=8,
            loss_mask_filter=has_consecutive_supervised_tokens,
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dflash",
                target_representation=None,
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=build_dspark_collator,
            ),
        ),
    )
    return make_registration(spec, providers)
