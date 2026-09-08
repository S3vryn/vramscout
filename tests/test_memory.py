import math

from vramscout.memory import kv_cache_per_token_gib, weight_memory_gib
from vramscout.types import ModelSpec


def _spec() -> ModelSpec:
    return ModelSpec(
        model_id="toy",
        model_type="llama",
        num_params=8_000_000_000,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_context=131072,
        vocab_size=128256,
    )


def test_weight_memory_bf16():
    assert math.isclose(weight_memory_gib(_spec(), "bf16"), 8_000_000_000 * 2 / 1024**3)


def test_gqa_kv_formula():
    spec = _spec()
    expected_bytes = 2 * 32 * 1 * 8 * 128 * 2
    assert math.isclose(kv_cache_per_token_gib(spec, 1, "bf16"), expected_bytes / 1024**3)
