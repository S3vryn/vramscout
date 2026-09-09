import json

import pytest

from vramscout.modern_v07 import plan_modern


@pytest.fixture()
def tiny_config(tmp_path):
    cfg = {
        "model_type": "tiny_standard",
        "architectures": ["TinyForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "intermediate_size": 256,
        "vocab_size": 1000,
        "max_position_embeddings": 4096,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_component_planner_handles_local_standard_config(tiny_config):
    r = plan_modern(
        str(tiny_config),
        context=1024,
        vram_gib=16,
        weight_dtype="bf16",
        calibration="none",
    )
    assert r.fits
    assert r.profile.family == "standard"
    assert "KV cache" in r.parts_gib
    assert any("reusable memory component" in note for note in r.notes)


def test_vllm_mode_does_not_double_count_default_safety_headroom(tiny_config):
    r = plan_modern(
        str(tiny_config),
        context=1024,
        vram_gib=16,
        engine="vllm",
        weight_dtype="bf16",
        calibration="none",
    )
    assert r.memory_budget_gib == pytest.approx(16 * 0.92)
    assert r.parts_gib["Safety reserve"] == 0.0


def test_dcp_fails_closed_for_unvalidated_family(tiny_config):
    with pytest.raises(ValueError, match="not publicly validated"):
        plan_modern(
            str(tiny_config),
            context=1024,
            vram_gib=16,
            tp=2,
            dcp=2,
            weight_dtype="bf16",
        )
