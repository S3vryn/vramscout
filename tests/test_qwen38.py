import json
import math
from pathlib import Path

import pytest

from vramscout.memory import kv_cache_per_token_gib, recurrent_state_gib
from vramscout.model import _qwen35_cache_metadata, inspect_model
from vramscout.types import ModelSpec


QWEN38_TEXT = {
    "model_type": "qwen3_5_text",
    "dtype": "bfloat16",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "full_attention_interval": 4,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
}


def test_qwen38_hybrid_cache_metadata():
    full, linear, state_bytes, label = _qwen35_cache_metadata(QWEN38_TEXT)
    assert full == 16
    assert linear == 48
    assert state_bytes > 140 * 1024**2
    assert state_bytes < 160 * 1024**2
    assert "48" in label


def test_qwen38_fp8_kv_is_16_full_attention_layers():
    full, linear, state_bytes, label = _qwen35_cache_metadata(QWEN38_TEXT)
    spec = ModelSpec(
        model_id="Qwen/Qwen3.8-27B",
        model_type="qwen3_5_text",
        num_params=27_700_000_000,
        num_layers=64,
        hidden_size=5120,
        num_attention_heads=24,
        num_kv_heads=4,
        head_dim=256,
        max_context=262144,
        cache_kind="qwen3_5_hybrid",
        kv_layers=full,
        recurrent_layers=linear,
        recurrent_state_bytes_per_batch=state_bytes,
        recurrent_state_label=label,
    )
    # 2(K/V) * 16 layers * 4 KV heads * 256 dim * 1 byte = 32768 B/token.
    assert math.isclose(kv_cache_per_token_gib(spec, 1, "fp8"), 32768 / 1024**3)
    assert 0.14 < recurrent_state_gib(spec, 1) < 0.16


def test_nested_qwen38_config_requires_hf_weight_metadata(tmp_path: Path):
    cfg = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "language_model_only": False,
        "text_config": QWEN38_TEXT,
        "vision_config": {"depth": 27, "hidden_size": 1152},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(Exception, match="safetensors parameter metadata"):
        inspect_model(str(path))
