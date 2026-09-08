import math
import pytest

from vramscout.engine import VLLM_DEFAULT_GPU_MEMORY_UTILIZATION, resolve_engine_budget
from vramscout.planner import plan_inference
from vramscout.types import GPUInfo, ModelSpec


def _tiny_model() -> ModelSpec:
    return ModelSpec(
        model_id="tiny",
        model_type="llama",
        num_params=1_000_000_000,
        num_layers=16,
        hidden_size=2048,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        max_context=131072,
        vocab_size=32000,
        torch_dtype="bf16",
    )


def test_vllm_default_budget_is_total_times_point_92():
    gpu = GPUInfo(index=0, name="test", total_gib=80, used_gib=5, free_gib=75)
    budget = resolve_engine_budget(gpu, "vllm")
    assert budget.gpu_memory_utilization == VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
    assert math.isclose(budget.memory_budget_gib, 73.6, rel_tol=1e-12)
    assert budget.startup_ok


def test_vllm_startup_fails_when_free_is_below_requested_budget():
    gpu = GPUInfo(index=0, name="busy", total_gib=80, used_gib=12, free_gib=68)
    result = plan_inference(gpu, _tiny_model(), context=1024, engine="vllm")
    assert result.memory_budget_gib == pytest.approx(73.6)
    assert not result.engine_startup_ok
    assert not result.fits


def test_custom_vllm_utilization_changes_max_context_budget():
    gpu = GPUInfo(index=0, name="test", total_gib=80, used_gib=0, free_gib=80)
    low = plan_inference(gpu, _tiny_model(), context=1024, engine="vllm", gpu_memory_utilization=0.5)
    high = plan_inference(gpu, _tiny_model(), context=1024, engine="vllm", gpu_memory_utilization=0.95)
    assert high.memory_budget_gib > low.memory_budget_gib
    assert high.max_context_vram >= low.max_context_vram


def test_gpu_memory_utilization_rejected_in_generic_mode():
    gpu = GPUInfo(index=0, name="test", total_gib=80, used_gib=0, free_gib=80)
    with pytest.raises(ValueError, match="requires --engine vllm"):
        plan_inference(gpu, _tiny_model(), gpu_memory_utilization=0.9)
