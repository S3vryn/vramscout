# Public validation

VRAMScout keeps public reference data next to the estimator so accuracy claims are auditable. v0.7 separates four evidence layers:

1. **checkpoint bytes** — actual safetensors shard sizes for weight residency;
2. **architecture math** — persistent KV / MLA / recurrent / sparse-index state implied by released configs;
3. **engine receipts** — vLLM/SGLang startup telemetry such as cache GiB and logical cache tokens;
4. **engine calibration** — small, explicit page/layout factors attached only to an exact engine/family/dtype profile.

The main report is [`PREDICTED_VS_ACTUAL.md`](PREDICTED_VS_ACTUAL.md), generated from [`public_runs_v07.json`](public_runs_v07.json) plus the earlier validation files.

```bash
python validation/run.py
python validation/predicted_vs_actual.py
```

## v0.7 headline

Across six newly audited hard cache receipts:

```text
architecture-only MAPE       2.850%
audited public-profile fit    0.237%
```

The largest raw gaps are not hidden: Qwen3.8-Flash-Next, GLM-5.3-Flash and Nemotron-3.5 expose vLLM page/layout overhead that pure tensor algebra does not capture. Those corrections are stored in `vramscout.calibration`, appear as a separate output row, and can be disabled with `--calibration none`.

Gemma4 is intentionally **not** auto-calibrated. Its public vLLM receipt differs substantially from simple global/sliding tensor arithmetic, and vLLM's hybrid cache grouping/padding behavior has changed across releases. One version-specific number is not enough evidence for a universal factor.

## Receipt ingestion

VRAMScout can normalize future public or local logs:

```bash
vramscout --parse-log vllm.log
vramscout --parse-log sglang.log --json
```

The parser extracts whichever fields are present, including model-load residency, available/active KV memory, logical cache tokens, reference context/concurrency, CUDA graph memory, peak activation memory and consumed memory.

## Validation policy

A model being **supported** does not mean every deployment peak is numerically validated. We use these labels deliberately:

- **supported** — released config maps to known reusable memory components;
- **hard cache receipt** — public engine output exposes comparable cache GiB/token data;
- **calibrated** — an exact engine/family/dtype layout correction has auditable evidence;
- **recipe/boundary evidence** — a public deployment proves a hardware/context boundary but lacks enough telemetry for an exact error row;
- **unresolved** — public evidence exposes a gap that should not be fitted away yet.

This keeps architecture constants, serving-engine behavior and empirical calibration separate rather than turning all three into one opaque fudge factor.
