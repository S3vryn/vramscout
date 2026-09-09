from vramscout.calibration import known_calibrations, public_cache_calibration


def test_exact_engine_family_dtype_match_only():
    qwen = public_cache_calibration("vllm", "qwen4_exp_hybrid", "bf16")
    assert qwen is not None
    assert qwen.factor == 1.029

    assert public_cache_calibration("sglang", "qwen4_exp_hybrid", "bf16") is None
    assert public_cache_calibration("vllm", "qwen4_exp_hybrid", "fp8") is None


def test_gemma_is_not_silently_fitted():
    assert public_cache_calibration("vllm", "gemma4_hybrid", "fp8") is None


def test_calibration_factors_are_small_layout_corrections():
    values = known_calibrations()
    assert values
    assert all(1.0 <= x.factor < 1.2 for x in values)
    assert all(x.source.startswith("https://") for x in values)
