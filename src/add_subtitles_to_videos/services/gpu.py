from __future__ import annotations

from dataclasses import dataclass
import subprocess

import torch


@dataclass(slots=True)
class GpuSnapshot:
    name: str
    total_memory_mib: int | None
    free_memory_mib: int | None
    used_memory_mib: int | None
    allocated_memory_mib: int | None
    reserved_memory_mib: int | None
    utilization_gpu_percent: int | None
    temperature_c: int | None


def current_gpu_snapshot() -> GpuSnapshot | None:
    if not torch.cuda.is_available():
        return None

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    total_memory_mib = int(properties.total_memory / (1024 * 1024))
    free_memory_mib: int | None = None
    used_memory_mib: int | None = None
    if hasattr(torch.cuda, "mem_get_info"):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        free_memory_mib = int(free_bytes / (1024 * 1024))
        used_memory_mib = int((total_bytes - free_bytes) / (1024 * 1024))
    allocated_memory_mib = int(torch.cuda.memory_allocated(device_index) / (1024 * 1024))
    reserved_memory_mib = int(torch.cuda.memory_reserved(device_index) / (1024 * 1024))

    utilization_gpu_percent: int | None = None
    temperature_c: int | None = None
    try:
        response = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except OSError:
        response = None

    if response is not None:
        lines = [line.strip() for line in response.stdout.splitlines() if line.strip()]
        if lines:
            parts = [part.strip() for part in lines[0].split(",")]
            if len(parts) >= 2:
                try:
                    utilization_gpu_percent = int(parts[0])
                except ValueError:
                    utilization_gpu_percent = None
                try:
                    temperature_c = int(parts[1])
                except ValueError:
                    temperature_c = None

    return GpuSnapshot(
        name=str(properties.name),
        total_memory_mib=total_memory_mib,
        free_memory_mib=free_memory_mib,
        used_memory_mib=used_memory_mib,
        allocated_memory_mib=allocated_memory_mib,
        reserved_memory_mib=reserved_memory_mib,
        utilization_gpu_percent=utilization_gpu_percent,
        temperature_c=temperature_c,
    )


def format_gpu_snapshot(snapshot: GpuSnapshot) -> str:
    used = (
        f"{snapshot.used_memory_mib}/{snapshot.total_memory_mib} MiB"
        if snapshot.used_memory_mib is not None and snapshot.total_memory_mib is not None
        else "unknown"
    )
    allocated = (
        f"{snapshot.allocated_memory_mib} MiB"
        if snapshot.allocated_memory_mib is not None
        else "unknown"
    )
    reserved = (
        f"{snapshot.reserved_memory_mib} MiB"
        if snapshot.reserved_memory_mib is not None
        else "unknown"
    )
    utilization = (
        f"{snapshot.utilization_gpu_percent}%"
        if snapshot.utilization_gpu_percent is not None
        else "unknown"
    )
    temperature = (
        f"{snapshot.temperature_c}C"
        if snapshot.temperature_c is not None
        else "unknown"
    )
    return (
        f"{snapshot.name} | VRAM {used} | Torch alloc {allocated} | "
        f"Torch reserved {reserved} | Util {utilization} | Temp {temperature}"
    )
