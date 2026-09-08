from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((ROOT / "public" / name).read_text())


def pct_error(predicted: float, measured: float) -> float:
    return abs(predicted - measured) / measured * 100.0


def qwen() -> None:
    data = load("qwen3_8_27b.json")
    print("Qwen3.8-27B · checkpoint-weight validation")
    print("variant                 predicted   public   abs error")
    print("-------------------------------------------------------")
    for case in data["weight_validation"]:
        pred = case["predicted_gib_from_checkpoint_bytes"]
        ref = case["public_vllm_weight_gib"]
        print(f"{case['variant']:<22} {pred:>8.3f}   {ref:>6.2f}   {pct_error(pred, ref):>7.3f}%")
    print()


def deepseek() -> None:
    data = load("deepseek_v4.json")
    case = data["cache_validation"][0]
    pred = case["predicted_gib"]
    ref = case["public_vllm_gib"]
    print("DeepSeek-V4 · public cache arithmetic")
    print(f"61-layer BF16 @ 1M     {pred:>8.3f}   {ref:>6.2f}   {pct_error(pred, ref):>7.3f}%")
    flash = data["flash_static_estimates"]
    print(f"Flash BF16 @ 1M        {flash['bf16_cache_gib_at_1m']:>8.3f} GiB  (architecture estimate)")
    print(f"Flash FP8+FP4 @ 1M     {flash['fp8_main_fp4_indexer_cache_gib_at_1m']:>8.3f} GiB  (architecture estimate)")
    print()


def glm() -> None:
    data = load("glm_5_2.json")
    case = data["cache_validation"]
    pred = case["predicted_bytes_per_logical_token"]
    obs = case["public_effective_bytes_per_logical_token"]
    print("GLM-5.2 · vLLM effective cache validation")
    print(f"predicted             {pred / 1024:>8.3f} KiB/logical token")
    print(f"public vLLM           {obs / 1024:>8.3f} KiB/logical token")
    print(f"absolute error        {pct_error(pred, obs):>8.3f}%")
    print()


def kimi() -> None:
    data = load("kimi_k2_5.json")
    case = data["cache_validation"]
    pred = case["predicted_bytes_per_logical_token"]
    obs = case["public_effective_bytes_per_logical_token"]
    print("Kimi-K2.5 · vLLM MLA cache validation (TP8, DCP1)")
    print(f"predicted             {pred / 1024:>8.3f} KiB/logical token")
    print(f"public vLLM           {obs / 1024:>8.3f} KiB/logical token")
    print(f"absolute error        {pct_error(pred, obs):>8.4f}%")
    print()


def minimax() -> None:
    data = load("minimax_m2_7.json")
    case = data["cache_validation"]
    pred = case["predicted_gib"]
    obs = case["public_vllm_required_gib"]
    print("MiniMax-M2.7 · vLLM BF16 KV requirement")
    print(f"204,800-token cache   {pred:>8.4f}   {obs:>7.2f}   {pct_error(pred, obs):>7.4f}%")
    print()


def main() -> None:
    qwen()
    deepseek()
    glm()
    kimi()
    minimax()


if __name__ == "__main__":
    main()
