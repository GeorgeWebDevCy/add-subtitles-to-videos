from __future__ import annotations

import torch

from .whisper_worker import WhisperWorkerClient


class WorkerPool:
    """Manages 1–2 persistent WhisperWorkerClient instances, one per device.

    On CUDA machines: slot 0 = CUDA, slot 1 = CPU (overflow).
    On CPU-only machines: slot 0 = CPU only.
    """

    def __init__(self) -> None:
        if torch.cuda.is_available():
            self._slots: list[tuple[WhisperWorkerClient, str]] = [
                (WhisperWorkerClient(), "cuda"),
                (WhisperWorkerClient(), "cpu"),
            ]
        else:
            self._slots = [
                (WhisperWorkerClient(), "cpu"),
            ]
        self._busy: list[bool] = [False] * len(self._slots)

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    def acquire(self) -> tuple[int, WhisperWorkerClient, str] | None:
        """Return (slot_index, client, device) for the first free slot. None if all busy."""
        for i, (client, device) in enumerate(self._slots):
            if not self._busy[i]:
                self._busy[i] = True
                return i, client, device
        return None

    def release(self, slot_index: int) -> None:
        """Mark slot as free."""
        self._busy[slot_index] = False

    def worker_device_label(self, slot_index: int) -> str:
        """Human-readable label for status display: 'GPU' or 'CPU'."""
        _, device = self._slots[slot_index]
        return "GPU" if device == "cuda" else "CPU"

    def close(self) -> None:
        for client, _ in self._slots:
            client.close()
