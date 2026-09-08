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
            "nvidia-smi was not found. VRAMScout v0.1 auto-detects NVIDIA GPUs; "
            "use --vram-gib to provide a manual budget."
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
