from __future__ import annotations

from pathlib import Path

from .languages import source_language_options, target_language_options
from .models import OutputMode, WorkflowProfile

APP_NAME = "Subtitle Foundry"
APP_TAGLINE = "Transcribe local video, translate subtitles, and review everything before export."
DEFAULT_OUTPUT_DIRECTORY = Path.cwd() / "exports"
DEFAULT_MAX_LINE_LENGTH = 42
DEFAULT_SUBTITLE_FONT_SIZE = 18
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_OUTPUT_MODE = OutputMode.SRT_ONLY
DEFAULT_SOURCE_LANGUAGE = "auto"
DEFAULT_TARGET_LANGUAGE = "en"
DEFAULT_TRANSLATION_PROVIDER = "openai_compatible"
DEFAULT_TRANSLATION_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TRANSLATION_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_PARALLEL_TRANSLATION_BATCHES = 2
DEFAULT_WORKFLOW_PROFILE = WorkflowProfile.EUROPE_MULTILINGUAL
VIDEO_FILE_FILTER = "Video Files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm)"
AUDIO_FILE_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"})
MEDIA_FILE_FILTER = (
    "Video or Audio Files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm "
    "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma);;"
    f"{VIDEO_FILE_FILTER};;"
    "Audio Files (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma)"
)
IMAGE_FILE_FILTER = "Image Files (*.jpg *.jpeg *.png *.webp *.bmp)"


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_FILE_EXTENSIONS

SOURCE_LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    (language.code, language.label) for language in source_language_options()
]

TARGET_LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    (language.code, language.label) for language in target_language_options()
]

TRANSLATION_PROVIDER_OPTIONS: list[tuple[str, str]] = [
    ("openai_compatible", "OpenAI-compatible text translation"),
]

OUTPUT_MODE_OPTIONS: list[tuple[OutputMode, str]] = [
    (OutputMode.SRT_ONLY, "Generate SRT only"),
    (OutputMode.BURNED_VIDEO, "Generate SRT and burned-in video"),
]

WHISPER_MODEL_OPTIONS: list[tuple[str, str]] = [
    ("base", "base - smallest practical local model"),
    ("small", "small - quickest download, lighter accuracy"),
    ("medium", "medium - faster balance for everyday use"),
    ("large-v3", "large-v3 - best accuracy, larger download"),
]
