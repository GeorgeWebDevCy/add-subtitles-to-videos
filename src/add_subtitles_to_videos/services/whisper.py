from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
import wave

import numpy as np
import torch
import whisper

from . import OperationCancelledError
from .gpu import current_gpu_snapshot, format_gpu_snapshot
from ..models import ProcessingOptions, SubtitleSegment, TranscriptionMetadata

LogReporter = Callable[[str], None]
CancelChecker = Callable[[], bool]


class WhisperService:
    _GLOBAL_MODEL_CACHE: dict[tuple[str, str], whisper.model.Whisper] = {}

    def __init__(self) -> None:
        self._model_cache = self._GLOBAL_MODEL_CACHE

    def transcribe(
        self,
        audio_path: Path,
        options: ProcessingOptions,
        *,
        log: LogReporter | None = None,
        cancel_requested: CancelChecker | None = None,
    ) -> tuple[list[SubtitleSegment], TranscriptionMetadata]:
        started_at = perf_counter()
        self._check_cancel(cancel_requested)
        source_language = None
        if options.source_language and options.source_language != "auto":
            source_language = options.source_language
        device = self._preferred_device()
        fp16_enabled = device == "cuda"

        cache_key = (options.whisper_model, device)
        load_message = "Reusing" if cache_key in self._model_cache else "Loading"
        self._emit_log(
            log,
            f"{load_message} Whisper model '{options.whisper_model}' on {self._device_label(device)}",
        )
        self._emit_gpu_snapshot(log, "CUDA before model load")
        self._emit_log(log, "Whisper task: transcribe")

        model = self._get_model(options.whisper_model, device)
        self._emit_gpu_snapshot(log, "CUDA after model load")
        self._check_cancel(cancel_requested)
        audio = self._read_pcm_wave(audio_path)
        self._check_cancel(cancel_requested)
        self._emit_gpu_snapshot(log, "CUDA before inference")
        with torch.inference_mode():
            result = model.transcribe(
                audio,
                language=source_language,
                task="transcribe",
                fp16=fp16_enabled,
                verbose=None,
                condition_on_previous_text=False,
            )
        self._check_cancel(cancel_requested)
        self._emit_gpu_snapshot(log, "CUDA after inference")

        items: list[SubtitleSegment] = []
        for segment in result.get("segments", []):
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            items.append(
                SubtitleSegment(
                    start_seconds=float(segment["start"]),
                    end_seconds=float(segment["end"]),
                    text=text,
                )
            )

        metadata = TranscriptionMetadata(
            detected_language=result.get("language"),
            detected_language_probability=None,
            duration_seconds=len(audio) / 16000 if len(audio) else None,
            device_label=self._device_label(device),
            task_label="transcribe",
        )

        if metadata.detected_language:
            self._emit_log(log, f"Detected language: {metadata.detected_language}")
        metadata.stage_durations["whisper_seconds"] = perf_counter() - started_at

        return items, metadata

    def preload_model(self, model_name: str, *, log: LogReporter | None = None) -> None:
        device = self._preferred_device()
        cache_key = (model_name, device)
        load_message = "Reusing" if cache_key in self._model_cache else "Loading"
        self._emit_log(
            log,
            f"{load_message} Whisper model '{model_name}' on {self._device_label(device)}",
        )
        self._get_model(model_name, device)
        self._emit_gpu_snapshot(log, "CUDA after warmup")

    def _get_model(self, model_name: str, device: str) -> whisper.model.Whisper:
        cache_key = (model_name, device)
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = whisper.load_model(model_name, device=device)
        return self._model_cache[cache_key]

    @staticmethod
    def _preferred_device() -> str:
        if torch.cuda.is_available():
            return "cuda"

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"

        return "cpu"

    @staticmethod
    def _device_label(device: str) -> str:
        if device == "cuda":
            return f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        if device == "mps":
            return "Apple Metal"
        return "CPU"

    @staticmethod
    def _read_pcm_wave(audio_path: Path) -> np.ndarray:
        with wave.open(str(audio_path), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise RuntimeError("Expected mono 16-bit PCM audio from FFmpeg.")
            sample_rate = handle.getframerate()
            if sample_rate != 16000:
                raise RuntimeError("Expected 16kHz audio from FFmpeg.")
            frames = handle.readframes(handle.getnframes())

        audio = np.frombuffer(frames, np.int16).astype(np.float32)
        return audio / 32768.0

    @staticmethod
    def _emit_log(log: LogReporter | None, message: str) -> None:
        if log is not None:
            log(message)

    @classmethod
    def _emit_gpu_snapshot(cls, log: LogReporter | None, prefix: str) -> None:
        snapshot = current_gpu_snapshot()
        if snapshot is None:
            return
        cls._emit_log(log, f"{prefix}: {format_gpu_snapshot(snapshot)}")

    @staticmethod
    def _check_cancel(cancel_requested: CancelChecker | None) -> None:
        if cancel_requested is not None and cancel_requested():
            raise OperationCancelledError("Processing stopped by user.")
