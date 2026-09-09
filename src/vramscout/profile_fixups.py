from __future__ import annotations

from . import modern_core as base


def apply_profile_fixups(profile, model_ref: str, revision: str | None = None):
    """Normalize config details that affect reusable memory components.

    These are architecture/config semantics, not fitted memory constants.
    Keeping them here separates config interpretation from the component math.
    """
    root = base._load_config(model_ref, revision)
    text = root.get("text_config") if isinstance(root.get("text_config"), dict) else root
    md = profile.metadata

    if profile.family == "glm53_hybrid":
        if bool(text.get("index_kpool_compress")):
            md["indexer_ratio"] = max(1, int(text.get("index_kpool") or 1))
        else:
            md["indexer_ratio"] = 1

    if profile.family == "gemma4_hybrid":
        layer_types = text.get("layer_types") if isinstance(text.get("layer_types"), list) else []
        shared = int(text.get("num_kv_shared_layers") or 0)
        non_shared = layer_types[: max(0, len(layer_types) - shared)] if layer_types else []
        md["kv_shared_layers"] = shared
        md["full_alloc_layers"] = sum(str(x).lower() == "full_attention" for x in non_shared)
        md["sliding_alloc_layers"] = sum(str(x).lower() == "sliding_attention" for x in non_shared)

    if profile.family == "mimo_hybrid":
        md["v_head_dim"] = int(text.get("v_head_dim") or profile.head_dim)
        md["swa_v_head_dim"] = int(
            text.get("swa_v_head_dim") or text.get("swa_head_dim") or profile.head_dim
        )
        # MiMo names the bounded local window attention_chunk_size in the
        # released checkpoints.
        md["sliding_window"] = int(
            text.get("attention_chunk_size")
            or text.get("sliding_window")
            or md.get("sliding_window")
            or 128
        )

    # DeepSeek's current CUDA index-cache layout stores a 4-byte scale bundle
    # beside each FP8/FP4 packed 128-d vector; the component builder accounts
    # for that when a quantized indexer dtype is selected.
    if profile.family == "deepseek_v4_compressed":
        md["indexer_quant_scale_bytes"] = 4

    return profile
