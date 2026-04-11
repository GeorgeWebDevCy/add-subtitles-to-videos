from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class WorkflowProfile(StrEnum):
    EUROPE_MULTILINGUAL = "europe_multilingual"


class OutputMode(StrEnum):
    SRT_ONLY = "srt_only"
    BURNED_VIDEO = "burned_video"


@dataclass(slots=True)
class ProcessingOptions:
    source_language: str | None
    target_language: str
    translation_provider: str | None
    whisper_model: str
    output_mode: OutputMode
    output_directory: Path
    max_line_length: int
    subtitle_font_size: int
    workflow_profile: WorkflowProfile = WorkflowProfile.EUROPE_MULTILINGUAL


@dataclass(slots=True)
class SubtitleSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(slots=True)
class TranslationSegment:
    start_seconds: float
    end_seconds: float
    source_text: str
    translated_text: str


@dataclass(slots=True)
class TranscriptionMetadata:
    detected_language: str | None
    detected_language_probability: float | None
    duration_seconds: float | None = None
    device_label: str | None = None
    task_label: str | None = None
    translation_provider: str | None = None
    target_language: str | None = None
    translation_applied: bool = False
    stage_durations: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TranscriptionResult:
    input_video: Path
    source_segments: list[SubtitleSegment]
    review_segments: list[TranslationSegment]
    metadata: TranscriptionMetadata
    warning_messages: tuple[str, ...]
    source_srt_text: str
    translated_srt_text: str


@dataclass(slots=True)
class PipelineResult:
    input_video: Path
    subtitle_file: Path
    burned_video: Path | None
    detected_language: str | None
    target_language: str | None
    translation_provider: str | None
    device_label: str | None
    segment_count: int
    elapsed_seconds: float
    preview_text: str
    warning_messages: tuple[str, ...]
    stage_durations: dict[str, float] = field(default_factory=dict)
