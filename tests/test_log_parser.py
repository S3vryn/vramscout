from vramscout.log_parser import parse_engine_log


def test_parse_vllm_receipt():
    text = """
    Model loading took 63.31 GiB
    Available KV cache memory: 9.45 GiB
    GPU KV cache size: 755,316 tokens, Maximum concurrency for 262,144 tokens per request: 2.88x
    Graph capturing finished in 85 secs, took 3.26 GiB
    Actual usage is 64.46 GiB for consumed memory (weights + non-torch), 1.27 GiB for peak activation, and 3.26 GiB for CUDAGraph memory.
    Current kv cache memory in use is 9.45 GiB.
    """
    r = parse_engine_log(text)
    assert r.engine == "vllm"
    assert r.model_loading_gib == 63.31
    assert r.available_kv_gib == 9.45
    assert r.active_kv_gib == 9.45
    assert r.kv_tokens == 755_316
    assert r.context_tokens == 262_144
    assert r.max_concurrency == 2.88
    assert r.cuda_graph_gib == 3.26
    assert r.peak_activation_gib == 1.27
    assert r.consumed_memory_gib == 64.46


def test_parse_sglang_receipt():
    text = """
    [TP0] Load weight end. elapsed=2.13 s, type=PixtralForConditionalGeneration, quant=fp8, avail mem=64.46 GB, mem usage=29.69 GB.
    [TP0] Using KV cache dtype: torch.bfloat16
    [TP0] KV Cache is allocated. #tokens: 2433591, KV size: 52.22 GB
    """
    r = parse_engine_log(text)
    assert r.engine == "sglang"
    assert r.model_loading_gib == 29.69
    assert r.available_kv_gib == 52.22
    assert r.active_kv_gib == 52.22
    assert r.kv_tokens == 2_433_591
