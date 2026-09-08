import math

from vramscout.memory import cache_memory_parts_gib, resolve_indexer_dtype, resolve_kv_dtype
from vramscout.model import _glm_moe_dsa_cache_metadata
from vramscout.types import ModelSpec


GLM52_CFG = {
    "model_type": "glm_moe_dsa",
    "num_hidden_layers": 78,
    "hidden_size": 6144,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "qk_head_dim": 256,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "kv_lora_rank": 512,
    "index_head_dim": 128,
    "max_position_embeddings": 1048576,
    "indexer_types": ["full"] * 21 + ["shared"] * 57,
}


def _spec() -> ModelSpec:
    md = _glm_moe_dsa_cache_metadata(GLM52_CFG)
    return ModelSpec(
        model_id="zai-org/GLM-5.2-FP8",
        model_type="glm_moe_dsa",
        num_params=743_000_000_000,
        num_layers=78,
        hidden_size=6144,
        num_attention_heads=64,
        num_kv_heads=64,
        head_dim=256,
        max_context=1048576,
        cache_kind="glm_moe_dsa",
        kv_layers=0,
        cache_metadata=md,
        quantization="fp8",
    )


def test_glm52_indexshare_metadata():
    md = _glm_moe_dsa_cache_metadata(GLM52_CFG)
    assert md["mla_latent_dim"] == 576
    assert md["indexer_full_layers"] == 21
    assert md["indexer_shared_layers"] == 57


def test_glm52_auto_cache_dtypes_match_public_vllm_path():
    spec = _spec()
    kv = resolve_kv_dtype("auto", "fp8", spec)
    idx = resolve_indexer_dtype("auto", spec, kv)
    assert kv == "fp8"
    assert idx == "bf16"


def test_glm52_fp8_mla_bf16_indexshare_bytes_per_token():
    spec = _spec()
    parts = cache_memory_parts_gib(spec, 1, 1, "fp8", "bf16")
    bytes_per_token = sum(parts.values()) * 1024**3
    assert math.isclose(bytes_per_token, 78 * 656 + 21 * 256, rel_tol=1e-12)
    assert math.isclose(bytes_per_token, 56544, rel_tol=1e-12)


def test_glm52_public_vllm_effective_cache_is_within_3_percent():
    observed_bytes_per_token = 126.41 * 1024**3 / 2_342_528
    predicted = sum(cache_memory_parts_gib(_spec(), 1, 1, "fp8", "bf16").values()) * 1024**3
    error_pct = abs(predicted - observed_bytes_per_token) / observed_bytes_per_token * 100
    assert error_pct < 3.0
