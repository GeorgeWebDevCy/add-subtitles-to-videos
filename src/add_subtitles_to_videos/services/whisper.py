from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any
import wave

import numpy as np
import torch
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except ImportError:  # pragma: no cover - dependency is installed in normal app environments
    FasterWhisperModel = None
import whisper

from . import OperationCancelledError
from .gpu import current_gpu_snapshot, format_gpu_snapshot
from ..models import ProcessingOptions, SubtitleSegment, TranscriptionMetadata
from ..languages import MIXED_LANGUAGE_CODE, OPPOSITE_LANGUAGE_CODE

LogReporter = Callable[[str], None]
CancelChecker = Callable[[], bool]


class WhisperService:
    _GLOBAL_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}

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
        multilingual = (
            options.source_language == MIXED_LANGUAGE_CODE
            or options.target_language == OPPOSITE_LANGUAGE_CODE
        )
        if options.source_language and options.source_language != "auto" and not multilingual:
            source_language = options.source_language
        device = self._preferred_device()
        backend = self._backend_name(device)
        if multilingual and backend != "faster-whisper":
            raise RuntimeError("Mixed-language transcription requires faster-whisper on CPU or CUDA.")

        cache_key = (options.whisper_model, device, backend)
        load_message = "Reusing" if cache_key in self._model_cache else "Loading"
        self._emit_log(
            log,
            f"{load_message} Whisper model '{options.whisper_model}' on {self._device_label(device)} via {backend}",
        )
        self._emit_gpu_snapshot(log, "CUDA before model load")
        self._emit_log(log, "Whisper task: transcribe")

        model = self._get_model(options.whisper_model, device)
        self._emit_gpu_snapshot(log, "CUDA after model load")
        self._check_cancel(cancel_requested)
        self._emit_gpu_snapshot(log, "CUDA before inference")
        if backend == "faster-whisper":
            if multilingual:
                items, metadata = self._transcribe_mixed_language(
                    model, audio_path, log=log, cancel_requested=cancel_requested,
                )
            else:
                items, metadata = self._transcribe_with_faster_whisper(
                    model, audio_path, source_language=source_language,
                )
        else:
            items, metadata = self._transcribe_with_openai_whisper(
                model,
                audio_path,
                source_language=source_language,
                cancel_requested=cancel_requested,
            )
        self._check_cancel(cancel_requested)
        self._emit_gpu_snapshot(log, "CUDA after inference")

        metadata.device_label = self._device_label(device)
        metadata.task_label = "transcribe"

        if metadata.detected_language:
            self._emit_log(log, f"Detected language: {metadata.detected_language}")
        metadata.stage_durations["whisper_seconds"] = perf_counter() - started_at

        return items, metadata

    def preload_model(self, model_name: str, *, log: LogReporter | None = None) -> None:
        device = self._preferred_device()
        backend = self._backend_name(device)
        cache_key = (model_name, device, backend)
        load_message = "Reusing" if cache_key in self._model_cache else "Loading"
        self._emit_log(
            log,
            f"{load_message} Whisper model '{model_name}' on {self._device_label(device)} via {backend}",
        )
        self._get_model(model_name, device)
        self._emit_gpu_snapshot(log, "CUDA after warmup")

    def _get_model(self, model_name: str, device: str) -> Any:
        backend = self._backend_name(device)
        cache_key = (model_name, device, backend)
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = self._load_backend_model(model_name, device, backend)
        return self._model_cache[cache_key]

    @staticmethod
    def _load_backend_model(model_name: str, device: str, backend: str) -> Any:
        if backend == "faster-whisper":
            if FasterWhisperModel is None:
                raise RuntimeError("faster-whisper is not installed.")
            return FasterWhisperModel(
                model_name,
                device=device,
                compute_type=WhisperService._compute_type(device),
            )
        return whisper.load_model(model_name, device=device)

    @staticmethod
    def _backend_name(device: str) -> str:
        if device != "mps" and FasterWhisperModel is not None:
            return "faster-whisper"
        return "openai-whisper"

    @staticmethod
    def _compute_type(device: str) -> str:
        if device == "cuda":
            return "float16"
        return "float32"

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

    def _transcribe_with_faster_whisper(
        self,
        model: Any,
        audio_path: Path,
        *,
        source_language: str | None,
        multilingual: bool = False,
    ) -> tuple[list[SubtitleSegment], TranscriptionMetadata]:
        segments, info = model.transcribe(
            str(audio_path),
            language=source_language,
            task="transcribe",
            condition_on_previous_text=False,
            **({"multilingual": True} if multilingual else {}),
        )

        items: list[SubtitleSegment] = []
        for segment in segments:
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue
            items.append(
                SubtitleSegment(
                    start_seconds=float(getattr(segment, "start")),
                    end_seconds=float(getattr(segment, "end")),
                    text=text,
                )
            )

        metadata = TranscriptionMetadata(
            detected_language=getattr(info, "language", None),
            detected_language_probability=getattr(info, "language_probability", None),
            duration_seconds=getattr(info, "duration", None),
        )
        return items, metadata

    def _transcribe_mixed_language(
        self, model: Any, audio_path: Path, *, log: LogReporter | None,
        cancel_requested: CancelChecker | None,
    ) -> tuple[list[SubtitleSegment], TranscriptionMetadata]:
        from faster_whisper.vad import get_speech_timestamps, VadOptions

        audio = self._read_pcm_wave(audio_path)
        spans = get_speech_timestamps(audio, vad_options=VadOptions(
            max_speech_duration_s=12, min_silence_duration_ms=350, speech_pad_ms=200,
        ))
        items: list[SubtitleSegment] = []
        for index, span in enumerate(spans, 1):
            self._check_cancel(cancel_requested)
            chunk = audio[span['start']:span['end']]
            _, _, probabilities = model.detect_language(chunk)
            language = max(
                ((code, probability) for code, probability in probabilities if code in {'el', 'en'}),
                key=lambda pair: pair[1],
            )[0]
            self._emit_log(log, f"Speech section {index}/{len(spans)}: {language}")
            segments, _ = model.transcribe(
                chunk, language=language, task='transcribe',
                condition_on_previous_text=False, word_timestamps=True,
            )
            offset = span['start'] / 16000
            for segment in segments:
                text = str(segment.text).strip()
                if text:
                    words = getattr(segment, 'words', None)
                    if words:
                        group = []
                        for word in words:
                            if group and (len(''.join(w.word for w in group)) + len(word.word) > 64
                                          or word.end - group[0].start > 5):
                                items.append(SubtitleSegment(offset + group[0].start, offset + group[-1].end,
                                                             ''.join(w.word for w in group).strip()))
                                group = []
                            group.append(word)
                        if group:
                            items.append(SubtitleSegment(offset + group[0].start, offset + group[-1].end,
                                                         ''.join(w.word for w in group).strip()))
                    else:
                        items.append(SubtitleSegment(
                            start_seconds=offset + float(segment.start),
                            end_seconds=min(offset + float(segment.end), len(audio) / 16000), text=text,
                        ))
        return items, TranscriptionMetadata(
            detected_language=MIXED_LANGUAGE_CODE, detected_language_probability=None,
            duration_seconds=len(audio) / 16000,
        )

    def _transcribe_with_openai_whisper(
        self,
        model: Any,
        audio_path: Path,
        *,
        source_language: str | None,
        cancel_requested: CancelChecker | None,
    ) -> tuple[list[SubtitleSegment], TranscriptionMetadata]:
        audio = self._read_pcm_wave(audio_path)
        self._check_cancel(cancel_requested)
        with torch.inference_mode():
            result = model.transcribe(
                audio,
                language=source_language,
                task="transcribe",
                fp16=self._preferred_device() == "cuda",
                verbose=None,
                condition_on_previous_text=False,
            )

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
        )
        return items, metadata

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
