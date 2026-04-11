from __future__ import annotations

from collections.abc import Callable
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from threading import Lock
from time import monotonic

from ..models import (
    OutputMode,
    ProcessingOptions,
    SubtitleSegment,
    TranscriptionMetadata,
    WorkflowProfile,
)
from . import OperationCancelledError
from .whisper import WhisperService

LogReporter = Callable[[str], None]
CancelChecker = Callable[[], bool]
_CANCEL_GRACE_SECONDS = 3.0


def _serialize_options(options: ProcessingOptions) -> dict[str, object]:
    return {
        "source_language": options.source_language,
        "target_language": options.target_language,
        "translation_provider": options.translation_provider,
        "whisper_model": options.whisper_model,
        "output_mode": OutputMode(str(options.output_mode)).value,
        "output_directory": str(options.output_directory),
        "max_line_length": options.max_line_length,
        "subtitle_font_size": options.subtitle_font_size,
        "workflow_profile": WorkflowProfile(str(options.workflow_profile)).value,
    }


def _deserialize_options(payload: dict[str, object]) -> ProcessingOptions:
    return ProcessingOptions(
        source_language=payload["source_language"],  # type: ignore[arg-type]
        target_language=str(payload["target_language"]),
        translation_provider=payload["translation_provider"],  # type: ignore[arg-type]
        whisper_model=str(payload["whisper_model"]),
        output_mode=OutputMode(str(payload["output_mode"])),
        output_directory=Path(str(payload["output_directory"])),
        max_line_length=int(payload["max_line_length"]),
        subtitle_font_size=int(payload["subtitle_font_size"]),
        workflow_profile=WorkflowProfile(str(payload["workflow_profile"])),
    )


def _serialize_segments(segments: list[SubtitleSegment]) -> list[dict[str, object]]:
    return [
        {
            "start_seconds": segment.start_seconds,
            "end_seconds": segment.end_seconds,
            "text": segment.text,
        }
        for segment in segments
    ]


def _deserialize_segments(payload: list[dict[str, object]]) -> list[SubtitleSegment]:
    return [
        SubtitleSegment(
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
            text=str(item["text"]),
        )
        for item in payload
    ]


def _serialize_metadata(metadata: TranscriptionMetadata) -> dict[str, object]:
    return {
        "detected_language": metadata.detected_language,
        "detected_language_probability": metadata.detected_language_probability,
        "duration_seconds": metadata.duration_seconds,
        "device_label": metadata.device_label,
        "task_label": metadata.task_label,
        "translation_provider": metadata.translation_provider,
        "target_language": metadata.target_language,
        "translation_applied": metadata.translation_applied,
        "stage_durations": dict(metadata.stage_durations),
    }


def _deserialize_metadata(payload: dict[str, object]) -> TranscriptionMetadata:
    return TranscriptionMetadata(
        detected_language=payload["detected_language"],  # type: ignore[arg-type]
        detected_language_probability=payload["detected_language_probability"],  # type: ignore[arg-type]
        duration_seconds=payload["duration_seconds"],  # type: ignore[arg-type]
        device_label=payload["device_label"],  # type: ignore[arg-type]
        task_label=payload["task_label"],  # type: ignore[arg-type]
        translation_provider=payload["translation_provider"],  # type: ignore[arg-type]
        target_language=payload["target_language"],  # type: ignore[arg-type]
        translation_applied=bool(payload.get("translation_applied", False)),
        stage_durations={
            str(key): float(value)
            for key, value in dict(payload.get("stage_durations", {})).items()
        },
    )


def _worker_main(command_queue, event_queue) -> None:
    service = WhisperService()

    while True:
        command = command_queue.get()
        command_type = command.get("type")
        if command_type == "shutdown":
            return
        if command_type not in {"transcribe", "preload"}:
            continue

        job_id = int(command["job_id"])

        def on_log(message: str) -> None:
            event_queue.put({"type": "log", "job_id": job_id, "message": message})

        try:
            if command_type == "preload":
                service.preload_model(str(command["model_name"]), log=on_log)
                event_queue.put({"type": "preload_complete", "job_id": job_id})
                continue

            options = _deserialize_options(command["options"])
            audio_path = Path(str(command["audio_path"]))
            segments, metadata = service.transcribe(audio_path, options, log=on_log)
            event_queue.put(
                {
                    "type": "result",
                    "job_id": job_id,
                    "segments": _serialize_segments(segments),
                    "metadata": _serialize_metadata(metadata),
                }
            )
        except Exception as exc:
            event_queue.put({"type": "error", "job_id": job_id, "message": str(exc)})


class WhisperWorkerClient:
    def __init__(self) -> None:
        self._context = get_context("spawn")
        self._lock = Lock()
        self._command_queue = None
        self._event_queue = None
        self._process = None
        self._job_counter = 0

    def transcribe(
        self,
        audio_path: Path,
        options: ProcessingOptions,
        *,
        log: LogReporter | None = None,
        cancel_requested: CancelChecker | None = None,
    ) -> tuple[list[SubtitleSegment], TranscriptionMetadata]:
        with self._lock:
            self._ensure_process()
            assert self._command_queue is not None
            assert self._event_queue is not None

            self._job_counter += 1
            job_id = self._job_counter
            cancel_deadline: float | None = None

            self._command_queue.put(
                {
                    "type": "transcribe",
                    "job_id": job_id,
                    "audio_path": str(audio_path),
                    "options": _serialize_options(options),
                }
            )

            while True:
                if cancel_requested is not None and cancel_requested() and cancel_deadline is None:
                    cancel_deadline = monotonic() + _CANCEL_GRACE_SECONDS
                    if log is not None:
                        log(
                            "Stop requested. Waiting briefly for Whisper to finish before cancelling the worker."
                        )

                if cancel_deadline is not None and monotonic() >= cancel_deadline:
                    self._terminate_process()
                    raise OperationCancelledError("Processing stopped by user.")

                if self._process is None or not self._process.is_alive():
                    self._cleanup_process()
                    raise RuntimeError("Whisper worker stopped unexpectedly.")

                try:
                    event = self._event_queue.get(timeout=0.1)
                except Empty:
                    continue

                if int(event.get("job_id", -1)) != job_id:
                    continue

                event_type = event.get("type")
                if event_type == "log":
                    if log is not None:
                        log(str(event["message"]))
                    continue

                if event_type == "result":
                    return (
                        _deserialize_segments(event["segments"]),
                        _deserialize_metadata(event["metadata"]),
                    )

                if event_type == "error":
                    raise RuntimeError(str(event["message"]))

    def preload_model(
        self,
        model_name: str,
        *,
        log: LogReporter | None = None,
    ) -> None:
        with self._lock:
            self._ensure_process()
            assert self._command_queue is not None
            assert self._event_queue is not None

            self._job_counter += 1
            job_id = self._job_counter

            self._command_queue.put(
                {
                    "type": "preload",
                    "job_id": job_id,
                    "model_name": model_name,
                }
            )

            while True:
                if self._process is None or not self._process.is_alive():
                    self._cleanup_process()
                    raise RuntimeError("Whisper worker stopped unexpectedly.")

                try:
                    event = self._event_queue.get(timeout=0.1)
                except Empty:
                    continue

                if int(event.get("job_id", -1)) != job_id:
                    continue

                event_type = event.get("type")
                if event_type == "log":
                    if log is not None:
                        log(str(event["message"]))
                    continue

                if event_type == "preload_complete":
                    return

                if event_type == "error":
                    raise RuntimeError(str(event["message"]))

    def close(self) -> None:
        with self._lock:
            self._shutdown_process()

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.is_alive():
            return

        self._command_queue = self._context.Queue()
        self._event_queue = self._context.Queue()
        self._process = self._context.Process(
            target=_worker_main,
            args=(self._command_queue, self._event_queue),
            daemon=True,
        )
        self._process.start()

    def _shutdown_process(self) -> None:
        if self._process is None:
            return

        if self._process.is_alive() and self._command_queue is not None:
            self._command_queue.put({"type": "shutdown"})
            self._process.join(timeout=2)

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)

        self._cleanup_process()

    def _terminate_process(self) -> None:
        if self._process is None:
            return

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)

        self._cleanup_process()

    def _cleanup_process(self) -> None:
        for queue in (self._command_queue, self._event_queue):
            if queue is None:
                continue
            queue.close()
            queue.join_thread()

        self._command_queue = None
        self._event_queue = None
        self._process = None
