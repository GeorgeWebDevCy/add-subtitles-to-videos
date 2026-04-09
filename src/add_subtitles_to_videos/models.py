from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class SubtitleMode(StrEnum):
    SOURCE = "source"
    ENGLISH = "english"


class OutputMode(StrEnum):
    SRT_ONLY = "srt_only"
    BURNED_VIDEO = "burned_video"


@dataclass(slots=True)
class ProcessingOptions:
    source_language: str | None
    subtitle_mode: SubtitleMode
    whisper_model: str
    output_mode: OutputMode
    output_directory: Path
    max_line_length: int
    subtitle_font_size: int


@dataclass(slots=True)
class SubtitleSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(slots=True)
class TranscriptionMetadata:
    detected_language: str | None
    detected_language_probability: float | None
    duration_seconds: float | None = None
    device_label: str | None = None
    task_label: str | None = None


@dataclass(slots=True)
class PipelineResult:
    input_video: Path
    subtitle_file: Path
    burned_video: Path | None
    detected_language: str | None
    device_label: str | None
    segment_count: int
    elapsed_seconds: float
    preview_text: str
    warning_messages: tuple[str, ...]
