from __future__ import annotations

import torch
from dataclasses import dataclass
from pathlib import Path

from .whisper_worker import WhisperWorkerClient
from ..models import ProcessingOptions


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


@dataclass
class DispatchJob:
    file_index: int
    video_path: Path
    options: ProcessingOptions


class QueueDispatcher:
    """Pure-Python queue state machine. No Qt dependencies.

    Usage:
        dispatcher.enqueue(job)
        result = dispatcher.dispatch_next()  # call after enqueue or after release
        if result:
            job, slot_index, client, device = result
            # start TranscriptionThread(client, device_override=device, ...)
        # When thread finishes:
        dispatcher.release(slot_index)
        dispatcher.dispatch_next()  # pick up next pending job
    """

    def __init__(self, pool: WorkerPool) -> None:
        self._pool = pool
        self._queue: list[DispatchJob] = []

    def enqueue(self, job: DispatchJob) -> None:
        self._queue.append(job)

    def dispatch_next(self) -> tuple[DispatchJob, int, WhisperWorkerClient, str] | None:
        """Assign next pending job to a free worker. Returns None if queue empty or all workers busy."""
        if not self._queue:
            return None
        slot = self._pool.acquire()
        if slot is None:
            return None
        slot_index, client, device = slot
        job = self._queue.pop(0)
        return job, slot_index, client, device

    def release(self, slot_index: int) -> None:
        self._pool.release(slot_index)

    def cancel_all(self) -> list[DispatchJob]:
        jobs = list(self._queue)
        self._queue.clear()
        return jobs

    def cancel_job(self, file_index: int) -> bool:
        for i, job in enumerate(self._queue):
            if job.file_index == file_index:
                del self._queue[i]
                return True
        return False

    @property
    def pending_count(self) -> int:
        return len(self._queue)
