from __future__ import annotations

import subprocess

from .types import GPUInfo

MIB_PER_GIB = 1024.0


class GPUDetectionError(RuntimeError):
    pass


def _parse_nvidia_smi_line(line: str) -> GPUInfo:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        raise GPUDetectionError(f"Unexpected nvidia-smi output: {line!r}")
    index, name, total_mib, used_mib, free_mib = parts
    return GPUInfo(
        index=int(index),
        name=name,
        total_gib=float(total_mib) / MIB_PER_GIB,
        used_gib=float(used_mib) / MIB_PER_GIB,
        free_gib=float(free_mib) / MIB_PER_GIB,
    )


def detect_nvidia_gpus() -> list[GPUInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise GPUDetectionError(
            "nvidia-smi was not found. VRAMScout auto-detects NVIDIA GPUs; "
            "use --vram-gib to provide a manual per-GPU budget."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise GPUDetectionError(f"nvidia-smi failed: {detail or exc}") from exc

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise GPUDetectionError("nvidia-smi returned no GPUs.")
    return [_parse_nvidia_smi_line(line) for line in lines]


def get_gpu(index: int = 0, vram_gib: float | None = None) -> GPUInfo:
    if vram_gib is not None:
        if vram_gib <= 0:
            raise GPUDetectionError("--vram-gib must be positive.")
        return GPUInfo(
            index=index,
            name=f"Manual VRAM budget ({vram_gib:g} GiB)",
            total_gib=vram_gib,
            used_gib=0.0,
            free_gib=vram_gib,
        )

    gpus = detect_nvidia_gpus()
    for gpu in gpus:
        if gpu.index == index:
            return gpu
    available = ", ".join(str(g.index) for g in gpus)
    raise GPUDetectionError(f"GPU index {index} not found. Available indices: {available}")


def get_gpu_group(start_index: int = 0, count: int = 1, vram_gib: float | None = None) -> GPUInfo:
    """Return the limiting per-rank VRAM budget for a TP group.

    For manual planning, ``--vram-gib`` means VRAM *per GPU*, not aggregate VRAM.
    For local auto-detection, consecutive GPU indices are selected and the rank with the
    smallest free memory becomes the planning budget.
    """
    if count < 1:
        raise GPUDetectionError("GPU count must be >= 1.")
    if count == 1:
        return get_gpu(index=start_index, vram_gib=vram_gib)
    if vram_gib is not None:
        if vram_gib <= 0:
            raise GPUDetectionError("--vram-gib must be positive.")
        return GPUInfo(
            index=start_index,
            name=f"{count}× manual GPU budget ({vram_gib:g} GiB each)",
            total_gib=vram_gib,
            used_gib=0.0,
            free_gib=vram_gib,
        )

    by_index = {gpu.index: gpu for gpu in detect_nvidia_gpus()}
    indices = list(range(start_index, start_index + count))
    missing = [i for i in indices if i not in by_index]
    if missing:
        available = ", ".join(str(i) for i in sorted(by_index))
        raise GPUDetectionError(
            f"Need {count} consecutive GPUs starting at {start_index}; missing {missing}. "
            f"Available indices: {available}"
        )
    selected = [by_index[i] for i in indices]
    limiting = min(selected, key=lambda x: x.free_gib)
    names = {g.name for g in selected}
    name = next(iter(names)) if len(names) == 1 else "heterogeneous NVIDIA GPUs"
    return GPUInfo(
        index=start_index,
        name=f"{count}× {name} · limiting rank GPU {limiting.index}",
        total_gib=min(g.total_gib for g in selected),
        used_gib=max(g.used_gib for g in selected),
        free_gib=limiting.free_gib,
    )
