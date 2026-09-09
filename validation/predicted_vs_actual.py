from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"


def pct_error(predicted: float, actual: float) -> float:
    return abs(predicted - actual) / actual * 100.0


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def legacy_rows() -> list[dict]:
    rows: list[dict] = []

    qwen = json.loads((PUBLIC / "qwen3_8_27b.json").read_text())
    for case in qwen["weight_validation"]:
        rows.append(
            {
                "target": f"Qwen3.8-27B {case['variant']} weights",
                "engine": "vLLM/HF",
                "metric": "GiB",
                "raw": case["predicted_gib_from_checkpoint_bytes"],
                "cal": case["predicted_gib_from_checkpoint_bytes"],
                "actual": case["public_vllm_weight_gib"],
                "source": qwen["sources"]["vllm_recipe"],
                "kind": "legacy",
            }
        )

    ds = json.loads((PUBLIC / "deepseek_v4.json").read_text())
    case = ds["cache_validation"][0]
    rows.append(
        {
            "target": "DeepSeek-V4 61-layer BF16 cache @1M",
            "engine": "vLLM",
            "metric": "GiB",
            "raw": case["predicted_gib"],
            "cal": case["predicted_gib"],
            "actual": case["public_vllm_gib"],
            "source": ds["sources"]["vllm_cache_blog"],
            "kind": "legacy",
        }
    )

    glm = json.loads((PUBLIC / "glm_5_2.json").read_text())
    case = glm["cache_validation"]
    rows.append(
        {
            "target": "GLM-5.2 FP8 MLA + BF16 IndexShare",
            "engine": "vLLM",
            "metric": "KiB/token",
            "raw": case["predicted_bytes_per_logical_token"] / 1024,
            "cal": case["predicted_bytes_per_logical_token"] / 1024,
            "actual": case["public_effective_bytes_per_logical_token"] / 1024,
            "source": glm["sources"]["vllm_measured_issue"],
            "kind": "legacy",
        }
    )

    kimi = json.loads((PUBLIC / "kimi_k2_5.json").read_text())
    case = kimi["cache_validation"]
    rows.append(
        {
            "target": "Kimi-K2.5 MLA TP8/DCP1",
            "engine": "vLLM",
            "metric": "KiB/token",
            "raw": case["predicted_bytes_per_logical_token"] / 1024,
            "cal": case["predicted_bytes_per_logical_token"] / 1024,
            "actual": case["public_effective_bytes_per_logical_token"] / 1024,
            "source": case["source"],
            "kind": "legacy",
        }
    )

    minimax = json.loads((PUBLIC / "minimax_m2_7.json").read_text())
    case = minimax["cache_validation"]
    rows.append(
        {
            "target": "MiniMax-M2.7 BF16 KV @204,800",
            "engine": "vLLM",
            "metric": "GiB",
            "raw": case["predicted_gib"],
            "cal": case["predicted_gib"],
            "actual": case["public_vllm_required_gib"],
            "source": case["source"],
            "kind": "legacy",
        }
    )
    return rows


def current_rows(data: dict) -> list[dict]:
    out: list[dict] = []
    for case in data["exact_cache_receipts"]:
        out.append(
            {
                "target": case["model"],
                "engine": case["engine"],
                "metric": case["metric"],
                "raw": case["raw_predicted"],
                "cal": case["calibrated_predicted"],
                "actual": case["actual"],
                "source": case["source"],
                "kind": "v0.7",
            }
        )
    return out


def render() -> str:
    data = json.loads((ROOT / "public_runs_v07.json").read_text())
    rows = legacy_rows() + current_rows(data)
    new_rows = [x for x in rows if x["kind"] == "v0.7"]
    raw_mape = sum(pct_error(x["raw"], x["actual"]) for x in new_rows) / len(new_rows)
    cal_mape = sum(pct_error(x["cal"], x["actual"]) for x in new_rows) / len(new_rows)

    lines = [
        "# Predicted vs Actual",
        "",
        f"Snapshot: **{data['snapshot_date']}**",
        "",
        "This table separates **config-derived architecture math** (`Raw`) from optional, explicit **engine-layout calibration** (`Calibrated`). Calibration never changes checkpoint/config interpretation; it only represents measured page/layout overhead for an exact engine/family/dtype profile.",
        "",
        f"Across the **{len(new_rows)} new hard cache receipts** added in v0.7, architecture-only MAPE is **{raw_mape:.3f}%** and the audited public-profile fit is **{cal_mape:.3f}%**. This is an audit fit on public receipts, not a held-out guarantee for every vLLM/SGLang version or hardware target.",
        "",
        "| Target | Engine | Metric | Raw | Actual | Raw err | Calibrated | Cal err |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        raw_err = pct_error(row["raw"], row["actual"])
        cal_err = pct_error(row["cal"], row["actual"])
        target = f"[{row['target']}]({row['source']})"
        lines.append(
            f"| {target} | {row['engine']} | {row['metric']} | {fmt(row['raw'])} | {fmt(row['actual'])} | {raw_err:.3f}% | {fmt(row['cal'])} | **{cal_err:.3f}%** |"
        )

    lines.extend(
        [
            "",
            "## Known version-sensitive gap",
            "",
            "The following public receipt is intentionally **not auto-calibrated** because the engine's hybrid allocator/page grouping has changed across releases:",
            "",
            "| Target | Metric | Raw | Actual | Raw err | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for case in data["version_sensitive_receipts"]:
        target = f"[{case['model']}]({case['source']})"
        lines.append(
            f"| {target} | {case['metric']} | {case['raw_predicted']:.3f} | {case['actual']:.3f} | {case['raw_error_pct']:.3f}% | {case['status']} |"
        )

    lines.extend(
        [
            "",
            "## Deployment evidence without a clean comparable memory receipt",
            "",
            "These models were audited too. A recipe or successful long-context deployment is useful evidence, but VRAMScout does **not** turn it into a fake numeric error when the public source lacks a comparable per-rank GiB/token receipt.",
            "",
            "| Model | Engine(s) | Public evidence |",
            "|---|---|---|",
        ]
    )
    for case in data["deployment_evidence_without_exact_comparable_cache_receipt"]:
        model = f"[{case['model']}]({case['source']})"
        lines.append(f"| {model} | {', '.join(case['engines'])} | {case['evidence']} |")

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python validation/predicted_vs_actual.py",
            "python validation/run.py",
            "```",
            "",
            "To ingest your own serving receipt:",
            "",
            "```bash",
            "vramscout --parse-log vllm.log",
            "vramscout --parse-log sglang.log --json",
            "```",
            "",
            "The source dataset is [`public_runs_v07.json`](public_runs_v07.json).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    report = render()
    out = ROOT / "PREDICTED_VS_ACTUAL.md"
    out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
