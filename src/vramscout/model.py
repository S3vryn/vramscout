from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from .types import ModelSpec


class ModelInspectionError(RuntimeError):
    pass


NONSTANDARD_CACHE_TYPES = {
    "deepseek_v2",
    "deepseek_v3",
    "mamba",
    "mamba2",
    "jamba",
    "recurrentgemma",
}


def _first_int(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _normalize_dtype(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower().replace("torch.", "")
    aliases = {
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
        "float8_e4m3fn": "fp8",
        "float8_e4m3fnuz": "fp8",
        "float8_e5m2": "fp8",
    }
    return aliases.get(text, text)


def _infer_quantization(config: dict[str, Any]) -> str | None:
    q = config.get("quantization_config")
    if not isinstance(q, dict):
        return None

    algo = str(q.get("quant_algo") or "").lower()
    if "nvfp4" in algo:
        return "nvfp4"
    if "fp8" in algo or "mxfp8" in algo:
        return "fp8"

    method = str(q.get("quant_method") or q.get("quantization_method") or "").lower()
    if "fp8" in method:
        return "fp8"
    bits = q.get("bits")
    load4 = bool(q.get("load_in_4bit"))
    load8 = bool(q.get("load_in_8bit"))
    if load4 or bits == 4:
        return f"{method or 'quantized'}-int4"
    if load8 or bits == 8:
        return f"{method or 'quantized'}-int8"

    groups = q.get("config_groups")
    if isinstance(groups, dict):
        seen_bits: set[int] = set()
        seen_types: set[str] = set()
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            weights = group.get("weights")
            if isinstance(weights, dict):
                if isinstance(weights.get("num_bits"), int):
                    seen_bits.add(weights["num_bits"])
                if weights.get("type"):
                    seen_types.add(str(weights["type"]).lower())
        if 4 in seen_bits and "float" in seen_types:
            return "nvfp4" if "modelopt" in method else "int4"
        if 8 in seen_bits and "float" in seen_types:
            return "fp8"
        if 4 in seen_bits:
            return "int4"
        if 8 in seen_bits:
            return "int8"
    return method or None


def _extract_hf_param_count(info: Any) -> int | None:
    st = getattr(info, "safetensors", None)
    if st is not None:
        total = getattr(st, "total", None)
        if isinstance(total, int) and total > 0:
            return total
        if isinstance(st, dict):
            total = st.get("total")
            if isinstance(total, int) and total > 0:
                return total

    card = getattr(info, "card_data", None)
    if isinstance(card, dict):
        total = card.get("safetensors", {}).get("total") if isinstance(card.get("safetensors"), dict) else None
        if isinstance(total, int) and total > 0:
            return total
    return None


def _extract_checkpoint_size_bytes(info: Any) -> int | None:
    """Sum actual safetensors shard bytes when the Hub returns file metadata."""
    siblings = getattr(info, "siblings", None)
    if not siblings:
        return None
    total = 0
    found = False
    for sibling in siblings:
        name = getattr(sibling, "rfilename", None)
        size = getattr(sibling, "size", None)
        if not isinstance(name, str) or not name.endswith(".safetensors"):
            continue
        if not isinstance(size, int) or size <= 0:
            continue
        total += size
        found = True
    return total if found else None


def _estimate_dense_params(config: dict[str, Any]) -> int:
    hidden = _first_int(config, "hidden_size", "n_embd", "d_model")
    layers = _first_int(config, "num_hidden_layers", "n_layer", "num_layers")
    vocab = _first_int(config, "vocab_size")
    q_heads = _first_int(config, "num_attention_heads", "n_head")
    kv_heads = _first_int(config, "num_key_value_heads", "num_kv_heads") or q_heads
    intermediate = _first_int(config, "intermediate_size", "ffn_dim", "n_inner")
    head_dim = _first_int(config, "head_dim", "qk_head_dim")

    if not all([hidden, layers, vocab, q_heads, kv_heads]):
        raise ModelInspectionError(
            "The checkpoint does not expose enough architecture metadata to estimate parameter count."
        )
    if intermediate is None:
        intermediate = 4 * hidden
    if head_dim is None:
        head_dim = hidden // q_heads

    model_type = str(config.get("model_type") or "").lower()
    if (
        model_type in NONSTANDARD_CACHE_TYPES
        or model_type in {"deepseek_v4", "glm_moe_dsa"}
        or _first_int(config, "num_local_experts", "num_experts", "n_routed_experts")
    ):
        raise ModelInspectionError(
            "This architecture needs Hugging Face safetensors parameter metadata; "
            "the dense fallback would be misleading."
        )

    q_out = q_heads * head_dim
    kv_out = kv_heads * head_dim
    attention = hidden * q_out + hidden * kv_out * 2 + q_out * hidden
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    per_layer = attention + mlp + norms

    tied = bool(config.get("tie_word_embeddings", False))
    embeddings = vocab * hidden
    lm_head = 0 if tied else vocab * hidden
    final_norm = hidden
    return int(embeddings + lm_head + layers * per_layer + final_norm)


def _load_config(model_ref: str, revision: str | None = None) -> tuple[dict[str, Any], str]:
    path = Path(model_ref).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
        if not config_path.exists():
            raise ModelInspectionError(f"No config.json found in {path}")
        return json.loads(config_path.read_text(encoding="utf-8")), str(config_path)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), str(path)

    try:
        config_path = hf_hub_download(repo_id=model_ref, filename="config.json", revision=revision)
    except Exception as exc:
        raise ModelInspectionError(f"Could not fetch config.json for {model_ref}: {exc}") from exc
    return json.loads(Path(config_path).read_text(encoding="utf-8")), config_path


def _qwen35_cache_metadata(text: dict[str, Any]) -> tuple[int, int, int, str]:
    layer_types = text.get("layer_types")
    layers = _first_int(text, "num_hidden_layers") or 0
    if isinstance(layer_types, list) and layer_types:
        full_layers = sum(1 for x in layer_types if x == "full_attention")
        linear_layers = sum(1 for x in layer_types if x == "linear_attention")
    else:
        interval = _first_int(text, "full_attention_interval") or 4
        full_layers = layers // interval
        linear_layers = layers - full_layers

    num_v = _first_int(text, "linear_num_value_heads")
    kdim = _first_int(text, "linear_key_head_dim")
    vdim = _first_int(text, "linear_value_head_dim")
    num_k = _first_int(text, "linear_num_key_heads")
    conv_kernel = _first_int(text, "linear_conv_kernel_dim")
    if not all([num_v, kdim, vdim, num_k, conv_kernel]):
        raise ModelInspectionError("Qwen3.5 hybrid config is missing linear-attention state dimensions.")

    recurrent_bytes_per_layer = num_v * kdim * vdim * 4
    key_dim = num_k * kdim
    value_dim = num_v * vdim
    conv_dim = 2 * key_dim + value_dim
    conv_bytes_per_layer = conv_dim * conv_kernel * 2
    state_bytes_per_batch = linear_layers * (recurrent_bytes_per_layer + conv_bytes_per_layer)
    label = f"GatedDeltaNet state ({linear_layers} linear-attn layers)"
    return full_layers, linear_layers, state_bytes_per_batch, label


def _deepseek_v4_cache_metadata(config: dict[str, Any]) -> dict[str, Any]:
    layers = _first_int(config, "num_hidden_layers")
    head_dim = _first_int(config, "head_dim", "qk_head_dim")
    rope_dim = _first_int(config, "qk_rope_head_dim")
    index_head_dim = _first_int(config, "index_head_dim")
    sliding_window = _first_int(config, "sliding_window")
    ratios = config.get("compress_ratios")
    if not all([layers, head_dim, rope_dim, index_head_dim, sliding_window]) or not isinstance(ratios, list):
        raise ModelInspectionError("DeepSeek-V4 config is missing hybrid-cache dimensions/compression ratios.")

    active = [int(x) for x in ratios[:layers] if isinstance(x, int)]
    if len(active) != layers:
        raise ModelInspectionError("DeepSeek-V4 compress_ratios does not cover every decoder layer.")
    supported = {0, 4, 128}
    unknown = sorted(set(active) - supported)
    if unknown:
        raise ModelInspectionError(f"Unsupported DeepSeek-V4 compression ratios: {unknown}")

    return {
        "sliding_only_layers": active.count(0),
        "csa_layers": active.count(4),
        "hca_layers": active.count(128),
        "sliding_window": sliding_window,
        "csa_compress_ratio": 4,
        "hca_compress_ratio": 128,
        "main_head_dim": head_dim,
        "rope_head_dim": rope_dim,
        "index_head_dim": index_head_dim,
        "fp8_scale_bytes_per_entry": 8,
        "fp4_indexer_scale_bytes_per_entry": 4,
    }


def _glm_moe_dsa_cache_metadata(config: dict[str, Any]) -> dict[str, Any]:
    layers = _first_int(config, "num_hidden_layers")
    kv_lora_rank = _first_int(config, "kv_lora_rank")
    rope_dim = _first_int(config, "qk_rope_head_dim")
    index_head_dim = _first_int(config, "index_head_dim")
    indexer_types = config.get("indexer_types")
    if not all([layers, kv_lora_rank, rope_dim, index_head_dim]) or not isinstance(indexer_types, list):
        raise ModelInspectionError("GLM MoE DSA config is missing MLA/IndexShare cache metadata.")
    active = [str(x).lower() for x in indexer_types[:layers]]
    if len(active) != layers:
        raise ModelInspectionError("GLM indexer_types does not cover every decoder layer.")
    unknown = sorted(set(active) - {"full", "shared"})
    if unknown:
        raise ModelInspectionError(f"Unsupported GLM indexer types: {unknown}")
    return {
        "mla_latent_dim": kv_lora_rank + rope_dim,
        "kv_lora_rank": kv_lora_rank,
        "rope_head_dim": rope_dim,
        "index_head_dim": index_head_dim,
        "indexer_full_layers": active.count("full"),
        "indexer_shared_layers": active.count("shared"),
        "fp8_mla_scale_bytes_per_entry": 16,
        "fp8_indexer_scale_bytes_per_entry": 4,
        "fp4_indexer_scale_bytes_per_entry": 4,
    }


def inspect_model(model_ref: str, revision: str | None = None) -> ModelSpec:
    root_config, _ = _load_config(model_ref, revision=revision)
    warnings: list[str] = []

    root_model_type = str(root_config.get("model_type") or "unknown").lower()
    text_config = root_config.get("text_config")
    config = text_config if isinstance(text_config, dict) else root_config
    model_type = str(config.get("model_type") or root_model_type).lower()

    if root_model_type in NONSTANDARD_CACHE_TYPES or model_type in NONSTANDARD_CACHE_TYPES:
        raise ModelInspectionError(
            f"{root_model_type} uses non-standard inference state/cache semantics. "
            "VRAMScout currently fails closed instead of applying the ordinary Transformer KV formula."
        )

    layers = _first_int(config, "num_hidden_layers", "n_layer", "num_layers")
    hidden = _first_int(config, "hidden_size", "n_embd", "d_model")
    q_heads = _first_int(config, "num_attention_heads", "n_head")
    kv_heads = _first_int(config, "num_key_value_heads", "num_kv_heads") or q_heads
    head_dim = _first_int(config, "head_dim", "qk_head_dim")
    if head_dim is None and hidden and q_heads:
        head_dim = hidden // q_heads

    missing = [
        name
        for name, value in {
            "num_hidden_layers": layers,
            "hidden_size": hidden,
            "num_attention_heads": q_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
        }.items()
        if not value
    ]
    if missing:
        raise ModelInspectionError("Unsupported/incomplete decoder config; missing: " + ", ".join(missing))

    max_context = _first_int(
        config,
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
    )
    sliding_window = _first_int(config, "sliding_window")
    if sliding_window and root_model_type != "deepseek_v4":
        warnings.append(
            "This model declares sliding-window attention. VRAMScout currently reports a conservative full-cache upper bound; "
            "some serving engines can use less KV memory."
        )

    num_params: int | None = None
    checkpoint_size_bytes: int | None = None
    parameter_source = "architecture estimate"
    checkpoint_size_source: str | None = None
    path = Path(model_ref).expanduser()
    if not path.exists():
        try:
            info = HfApi().model_info(model_ref, revision=revision, files_metadata=True)
            num_params = _extract_hf_param_count(info)
            checkpoint_size_bytes = _extract_checkpoint_size_bytes(info)
            if num_params:
                parameter_source = "Hugging Face safetensors metadata"
            if checkpoint_size_bytes:
                checkpoint_size_source = "Hugging Face safetensors file sizes"
        except Exception:
            warnings.append("Hugging Face model metadata was unavailable; using architecture-derived estimates where possible.")

    if not num_params:
        if config is not root_config:
            raise ModelInspectionError(
                "This multimodal/hybrid checkpoint needs Hugging Face safetensors parameter metadata; "
                "the text-only dense fallback would under-count weights."
            )
        num_params = _estimate_dense_params(config)

    quantization = _infer_quantization(root_config) or _infer_quantization(config)

    cache_kind = "standard"
    kv_layers = int(layers)
    recurrent_layers = 0
    recurrent_state_bytes_per_batch = 0
    recurrent_state_label = None
    cache_metadata: dict[str, Any] = {}

    if root_model_type == "qwen3_5" or model_type == "qwen3_5_text":
        kv_layers, recurrent_layers, recurrent_state_bytes_per_batch, recurrent_state_label = _qwen35_cache_metadata(config)
        cache_kind = "qwen3_5_hybrid"
        warnings.append(
            f"Hybrid cache detected: {kv_layers} full-attention layers grow with context; "
            f"{recurrent_layers} linear-attention layers use constant GatedDeltaNet state."
        )
    elif root_model_type == "deepseek_v4" or model_type == "deepseek_v4":
        cache_kind = "deepseek_v4_hybrid"
        cache_metadata = _deepseek_v4_cache_metadata(config)
        kv_layers = 0
        warnings.append(
            "DeepSeek-V4 hybrid cache detected: shared K=V sliding window + C4 CSA + C128 HCA; "
            "VRAMScout uses architecture-specific compressed-cache accounting."
        )
    elif root_model_type == "glm_moe_dsa" or model_type == "glm_moe_dsa":
        cache_kind = "glm_moe_dsa"
        cache_metadata = _glm_moe_dsa_cache_metadata(config)
        kv_layers = 0
        warnings.append(
            f"GLM DSA/IndexShare cache detected: {cache_metadata['mla_latent_dim']}-dim MLA latent per layer + "
            f"{cache_metadata['indexer_full_layers']} materialized indexer caches shared across the remaining layers."
        )

    dtype_value = config.get("dtype", config.get("torch_dtype", root_config.get("dtype", root_config.get("torch_dtype"))))
    has_vision = isinstance(root_config.get("vision_config"), dict) and not bool(root_config.get("language_model_only", False))

    return ModelSpec(
        model_id=model_ref,
        model_type=model_type,
        num_params=num_params,
        num_layers=int(layers),
        hidden_size=int(hidden),
        num_attention_heads=int(q_heads),
        num_kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        max_context=max_context,
        vocab_size=_first_int(config, "vocab_size"),
        torch_dtype=_normalize_dtype(dtype_value),
        quantization=quantization,
        sliding_window=sliding_window,
        parameter_source=parameter_source,
        warnings=warnings,
        cache_kind=cache_kind,
        kv_layers=kv_layers,
        recurrent_layers=recurrent_layers,
        recurrent_state_bytes_per_batch=recurrent_state_bytes_per_batch,
        recurrent_state_label=recurrent_state_label,
        has_vision_encoder=has_vision,
        checkpoint_size_bytes=checkpoint_size_bytes,
        checkpoint_size_source=checkpoint_size_source,
        cache_metadata=cache_metadata,
    )
