from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .whisper_worker import WhisperWorkerClient
from ..models import ProcessingOptions


class WorkerPool:
    """Manages 1–2 persistent WhisperWorkerClient instances, one per device.

    On CUDA machines: slot 0 = CUDA, slot 1 = CPU (overflow).
    On CPU-only machines: slot 0 = CPU only.
    """

    def __init__(self) -> None:
        if torch.cuda.is_available():
            self._slots: list[tuple[WhisperWorkerClient, str, bool]] = [
                (WhisperWorkerClient(), "cuda", False),  # (client, device, is_busy)
                (WhisperWorkerClient(), "cpu", False),
            ]
        else:
            self._slots = [
                (WhisperWorkerClient(), "cpu", False),
            ]

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    def acquire(self) -> tuple[int, WhisperWorkerClient, str] | None:
        """Return (slot_index, client, device) for the first free slot. None if all busy."""
        for i, (client, device, busy) in enumerate(self._slots):
            if not busy:
                self._slots[i] = (client, device, True)
                return i, client, device
        return None

    def release(self, slot_index: int) -> None:
        """Mark slot as free."""
        client, device, _ = self._slots[slot_index]
        self._slots[slot_index] = (client, device, False)

    def worker_device_label(self, slot_index: int) -> str:
        """Human-readable label for status display: 'GPU' or 'CPU'."""
        _, device, _ = self._slots[slot_index]
        return "GPU" if device == "cuda" else "CPU"

    def close(self) -> None:
        for client, _, _ in self._slots:
            client.close()
