from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from .gpu import GPUDetectionError, get_gpu_group
from .model import ModelInspectionError, inspect_model
from .planner import plan_inference


def _positive_int(value: str) -> int:
    parsed = int(value.replace("_", ""))
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vramscout",
        description="Check whether a modern Hugging Face LLM fits your GPU(s), explain VRAM use, and estimate maximum context.",
    )
    p.add_argument("model", help="Hugging Face model id, local model directory, or local config.json")
    p.add_argument("--revision", default=None, help="Optional Hugging Face revision/commit")
    p.add_argument("--context", type=_positive_int, default=None, help="Context length to evaluate (default: min(8192, model limit))")
    p.add_argument("--batch-size", type=_positive_int, default=1, help="Active sequences / batch size (default: 1)")
    p.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16", "fp8", "int8", "int4", "nvfp4"], default="auto", help="Weight precision")
    p.add_argument("--kv-dtype", choices=["auto", "fp32", "fp16", "bf16", "fp8"], default="auto", help="KV/cache precision")
    p.add_argument("--indexer-dtype", choices=["auto", "bf16", "fp16", "fp8", "fp4"], default="auto", help="Sparse-indexer cache precision for architectures that use one")
    p.add_argument("--tp", type=_positive_int, default=1, help="Tensor-parallel GPU count (default: 1)")
    p.add_argument("--dcp", type=_positive_int, default=1, help="Decode Context Parallel size for MLA cache sharding (default: 1)")
    p.add_argument("--prefill-chunk", type=_positive_int, default=8192, help="Active prefill tokens used for scratch estimate (default: 8192)")
    p.add_argument("--gpu-index", type=int, default=0, help="First NVIDIA GPU index for local TP group (default: 0)")
    p.add_argument("--vram-gib", type=float, default=None, help="Manual VRAM budget per GPU; skips local GPU auto-detection")
    p.add_argument("--reserve-gib", type=float, default=None, help="Override per-GPU safety reserve (default: max(1 GiB, 5%% of VRAM))")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return p


def _print_human(result) -> None:
    console = Console()
    status = "[bold green]CAN RUN[/bold green]" if result.fits else "[bold red]DOES NOT FIT[/bold red]"
    console.print(f"\n[bold]VRAMScout[/bold]  {status}\n")

    gpu = Table(title="GPU budget", show_header=False)
    gpu.add_column("Field", style="bold")
    gpu.add_column("Value", justify="right")
    gpu.add_row("Device / group", f"GPU {result.gpu.index} · {result.gpu.name}")
    gpu.add_row("TP / DCP", f"{result.tp_size} / {result.dcp_size}")
    gpu.add_row("Per-GPU total VRAM", f"{result.gpu.total_gib:.2f} GiB")
    gpu.add_row("Limiting used", f"{result.gpu.used_gib:.2f} GiB")
    gpu.add_row("Limiting free", f"{result.gpu.free_gib:.2f} GiB")
    console.print(gpu)

    model = Table(title="Model", show_header=False)
    model.add_column("Field", style="bold")
    model.add_column("Value", justify="right")
    model.add_row("Checkpoint", result.model.model_id)
    model.add_row("Architecture", result.model.model_type)
    if result.model.cache_kind == "qwen3_5_hybrid":
        model.add_row(
            "Cache architecture",
            f"Qwen hybrid · {result.model.effective_kv_layers} full-attn KV + {result.model.recurrent_layers} linear-attn",
        )
    elif result.model.cache_kind == "deepseek_v4_hybrid":
        md = result.model.cache_metadata
        model.add_row(
            "Cache architecture",
            f"DeepSeek V4 · SWA + {md.get('csa_layers', '?')} CSA(C4) + {md.get('hca_layers', '?')} HCA(C128)",
        )
    elif result.model.cache_kind == "glm_moe_dsa":
        md = result.model.cache_metadata
        model.add_row(
            "Cache architecture",
            f"GLM DSA · {md.get('mla_latent_dim', '?')}-dim MLA + {md.get('indexer_full_layers', '?')} IndexShare groups",
        )
    elif result.model.cache_kind == "kimi_mla":
        md = result.model.cache_metadata
        model.add_row(
            "Cache architecture",
            f"Kimi MLA · {md.get('mla_latent_dim', '?')}-dim latent · replicated by TP, sharded by DCP",
        )
    elif result.model.cache_kind == "minimax_m2":
        model.add_row(
            "Cache architecture",
            f"MiniMax M2 · standard GQA ({result.model.num_kv_heads} KV heads)",
        )
    if result.model.has_vision_encoder:
        model.add_row("Vision encoder", "included in checkpoint weights")
    model.add_row("Parameters", f"{result.model.num_params / 1e9:.3f} B")
    model.add_row("Param source", result.model.parameter_source)
    model.add_row("Weights", result.weight_dtype)
    model.add_row("KV/cache", result.kv_dtype)
    if result.indexer_dtype:
        model.add_row("Indexer cache", result.indexer_dtype)
    model.add_row("Batch size", str(result.batch_size))
    model.add_row("Requested context", f"{result.context:,}")
    if result.model.max_context:
        model.add_row("Config context limit", f"{result.model.max_context:,}")
    guide_limit = result.model.cache_metadata.get("official_single_sequence_limit")
    if guide_limit:
        model.add_row("Official serving guide", f"{int(guide_limit):,}")
    console.print(model)

    b = result.breakdown
    title = f"Per-GPU VRAM breakdown @ {result.context:,} tokens"
    mem = Table(title=title)
    mem.add_column("Part")
    mem.add_column("GiB", justify="right")
    mem.add_row("Model weights / rank", f"{b.weights_gib:.2f}")
    if b.cache_parts_gib:
        for part, gib in b.cache_parts_gib.items():
            mem.add_row(part, f"{gib:.2f}")
        if len(b.cache_parts_gib) > 1:
            mem.add_row("Cache subtotal", f"[bold]{b.kv_cache_gib:.2f}[/bold]")
    else:
        mem.add_row("KV/cache", f"{b.kv_cache_gib:.2f}")
    if b.recurrent_state_gib > 0:
        mem.add_row(result.model.recurrent_state_label or "Recurrent state", f"{b.recurrent_state_gib:.2f}")
    mem.add_row("CUDA / runtime", f"{b.runtime_fixed_gib:.2f}")
    mem.add_row("Prefill scratch", f"{b.prefill_scratch_gib:.2f}")
    mem.add_row("Safety reserve", f"{b.safety_reserve_gib:.2f}")
    mem.add_section()
    mem.add_row("Estimated peak / rank", f"[bold]{b.total_gib:.2f}[/bold]")
    mem.add_row("Free after estimate", f"{result.spare_gib:.2f}")
    console.print(mem)

    ctx = Table(title="Context budget", show_header=False)
    ctx.add_column("Field", style="bold")
    ctx.add_column("Tokens", justify="right")
    ctx.add_row("VRAM-derived max", f"{result.max_context_vram:,}")
    ctx.add_row("Usable max", f"[bold]{result.max_context_usable:,}[/bold]")
    console.print(ctx)

    if result.warnings:
        console.print("\n[bold yellow]Notes[/bold yellow]")
        for warning in result.warnings:
            console.print(f"  • {warning}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        gpu = get_gpu_group(start_index=args.gpu_index, count=args.tp, vram_gib=args.vram_gib)
        model = inspect_model(args.model, revision=args.revision)
        result = plan_inference(
            gpu=gpu,
            model=model,
            context=args.context,
            batch_size=args.batch_size,
            weight_dtype=args.dtype,
            kv_dtype=args.kv_dtype,
            indexer_dtype=args.indexer_dtype,
            safety_reserve_gib=args.reserve_gib,
            prefill_chunk_tokens=args.prefill_chunk,
            tp_size=args.tp,
            dcp_size=args.dcp,
        )
    except (GPUDetectionError, ModelInspectionError, ValueError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            Console(stderr=True).print(f"[bold red]Error:[/bold red] {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_human(result)
    return 0 if result.fits else 2


if __name__ == "__main__":
    sys.exit(main())
