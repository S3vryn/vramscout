import math

from vramscout.memory import cache_memory_parts_gib
from vramscout.model import _deepseek_v4_cache_metadata
from vramscout.types import ModelSpec


FLASH_CFG = {
    "model_type": "deepseek_v4",
    "num_hidden_layers": 43,
    "hidden_size": 4096,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "index_head_dim": 128,
    "sliding_window": 128,
    "max_position_embeddings": 1048576,
    "compress_ratios": [0, 0] + [x for _ in range(20) for x in (4, 128)] + [4, 0],
}


def _flash_spec() -> ModelSpec:
    md = _deepseek_v4_cache_metadata(FLASH_CFG)
    return ModelSpec(
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        model_type="deepseek_v4",
        num_params=284_000_000_000,
        num_layers=43,
        hidden_size=4096,
        num_attention_heads=64,
        num_kv_heads=1,
        head_dim=512,
        max_context=1048576,
        cache_kind="deepseek_v4_hybrid",
        kv_layers=0,
        sliding_window=128,
        cache_metadata=md,
    )


def test_flash_layer_schedule():
    md = _deepseek_v4_cache_metadata(FLASH_CFG)
    assert md["sliding_only_layers"] == 2
    assert md["csa_layers"] == 21
    assert md["hca_layers"] == 20


def test_deepseek_v4_flash_cache_at_1m():
    spec = _flash_spec()
    bf16 = cache_memory_parts_gib(spec, 1048576, 1, "bf16", "bf16")
    optimized = cache_memory_parts_gib(spec, 1048576, 1, "fp8", "fp4")
    assert math.isclose(sum(bf16.values()), 6.7239990234375, rel_tol=1e-9)
    assert math.isclose(sum(optimized.values()), 3.434878349304199, rel_tol=1e-9)


def test_vllm_public_pro_bf16_arithmetic_matches_9_62_gib():
    md = dict(_deepseek_v4_cache_metadata(FLASH_CFG))
    md.update({"sliding_only_layers": 0, "csa_layers": 30, "hca_layers": 31})
    spec = _flash_spec()
    spec.num_layers = 61
    spec.cache_metadata = md
    parts = cache_memory_parts_gib(spec, 1048576, 1, "bf16", "bf16")
    assert abs(sum(parts.values()) - 9.62) < 0.01
