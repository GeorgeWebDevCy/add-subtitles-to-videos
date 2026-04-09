from __future__ import annotations

from pathlib import Path

from .models import OutputMode, SubtitleMode

APP_NAME = "Subtitle Foundry"
APP_TAGLINE = "Translate speech into subtitles and burn them straight into video."
DEFAULT_OUTPUT_DIRECTORY = Path.cwd() / "exports"
DEFAULT_MAX_LINE_LENGTH = 42
DEFAULT_SUBTITLE_FONT_SIZE = 18
DEFAULT_WHISPER_MODEL = "medium"
VIDEO_FILE_FILTER = "Video Files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm)"

LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("auto", "Auto detect"),
    ("el", "Greek"),
    ("en", "English"),
    ("tr", "Turkish"),
    ("de", "German"),
    ("fr", "French"),
    ("it", "Italian"),
    ("es", "Spanish"),
]

SUBTITLE_MODE_OPTIONS: list[tuple[SubtitleMode, str]] = [
    (SubtitleMode.ENGLISH, "English translation"),
    (SubtitleMode.SOURCE, "Source language"),
]

OUTPUT_MODE_OPTIONS: list[tuple[OutputMode, str]] = [
    (OutputMode.BURNED_VIDEO, "Generate SRT and burned-in video"),
    (OutputMode.SRT_ONLY, "Generate SRT only"),
]

WHISPER_MODEL_OPTIONS: list[tuple[str, str]] = [
    ("base", "base - smallest practical local model"),
    ("small", "small - quickest download, lighter accuracy"),
    ("medium", "medium - balanced default for Greek speech"),
    ("large-v3", "large-v3 - best accuracy, larger download"),
]
