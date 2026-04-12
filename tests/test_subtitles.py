import importlib
import runpy
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from add_subtitles_to_videos import app as app_module
from add_subtitles_to_videos.languages import source_language_options, target_language_options
from add_subtitles_to_videos.models import (
    OutputMode,
    PipelineResult,
    ProcessingOptions,
    SubtitleSegment,
    TranscriptionMetadata,
    TranscriptionResult,
    TranslationSegment,
)
from add_subtitles_to_videos.services import OperationCancelledError
from add_subtitles_to_videos.services import ffmpeg as ffmpeg_module
from add_subtitles_to_videos.services.gpu import GpuSnapshot
from add_subtitles_to_videos.services.pipeline import SubtitlePipeline
from add_subtitles_to_videos.services.subtitles import (
    format_srt_timestamp,
    parse_srt_text,
    segments_to_srt_text,
    translation_segments_to_srt_text,
    validate_review_srt_text,
    wrap_subtitle_text,
    write_srt,
)
from add_subtitles_to_videos.services.translation import (
    OpenAICompatibleTranslationService,
    TranslationProviderConfig,
    TranslationTransportError,
)
from add_subtitles_to_videos.services.translation import _translation_text_from_item
from add_subtitles_to_videos.services.whisper import WhisperService
from add_subtitles_to_videos.services.whisper_worker import _serialize_options
from add_subtitles_to_videos.ui import main_window as main_window_module


def _application() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _pump_events_until(predicate, timeout_seconds: float = 2.0) -> bool:
    app = _application()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)

    app.processEvents()
    return predicate()


class FakeSettings:
    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._values = dict(initial or {})

    def value(self, key: str, default=None, _type=None):
        return self._values.get(key, default)

    def setValue(self, key: str, value) -> None:
        self._values[key] = value

    def sync(self) -> None:
        return None


class FakeTranslationService:
    provider_name = "fake"

    def __init__(self, translated_prefix: str = "DE") -> None:
        self.calls: list[tuple[str, str, list[str]]] = []
        self._translated_prefix = translated_prefix

    def translate_segments(
        self,
        segments: list[SubtitleSegment],
        *,
        source_language: str,
        target_language: str,
        log=None,
        cancel_requested=None,
    ) -> list[TranslationSegment]:
        self.calls.append((source_language, target_language, [segment.text for segment in segments]))
        return [
            TranslationSegment(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                source_text=segment.text,
                translated_text=f"{self._translated_prefix}: {segment.text}",
            )
            for segment in segments
        ]


class FakeSleepInhibitor:
    def __init__(self) -> None:
        self.activations = 0
        self.releases = 0
        self.active = False

    def activate(self) -> bool:
        if self.active:
            return False
        self.active = True
        self.activations += 1
        return True

    def release(self) -> bool:
        if not self.active:
            return False
        self.active = False
        self.releases += 1
        return True


def _options(
    tmp_path: Path,
    *,
    source_language: str | None = "el",
    target_language: str = "en",
) -> ProcessingOptions:
    return ProcessingOptions(
        source_language=source_language,
        target_language=target_language,
        translation_provider="openai_compatible",
        whisper_model="large-v3",
        output_mode=OutputMode.SRT_ONLY,
        output_directory=tmp_path,
        max_line_length=42,
        subtitle_font_size=18,
    )


def _transcription_result(video_path: Path) -> TranscriptionResult:
    source_segments = [
        SubtitleSegment(0.0, 1.0, "γειά σου"),
        SubtitleSegment(1.1, 2.0, "τι κάνεις"),
    ]
    review_segments = [
        TranslationSegment(0.0, 1.0, "γειά σου", "hello"),
        TranslationSegment(1.1, 2.0, "τι κάνεις", "how are you"),
    ]
    return TranscriptionResult(
        input_video=video_path,
        source_segments=source_segments,
        review_segments=review_segments,
        metadata=TranscriptionMetadata(
            detected_language="el",
            detected_language_probability=None,
            device_label="CPU",
            task_label="transcribe",
            translation_provider="openai_compatible",
            target_language="en",
            translation_applied=True,
            stage_durations={
                "audio_extraction_seconds": 0.2,
                "whisper_seconds": 1.1,
                "translation_seconds": 0.5,
            },
        ),
        warning_messages=(),
        source_srt_text=translation_segments_to_srt_text(
            review_segments,
            text_field="source_text",
            max_line_length=42,
        ),
        translated_srt_text=translation_segments_to_srt_text(
            review_segments,
            text_field="translated_text",
            max_line_length=42,
        ),
    )


def _patch_settings(monkeypatch, initial: dict[str, str] | None = None) -> FakeSettings:
    settings = FakeSettings(initial)
    monkeypatch.setattr(main_window_module, "QSettings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(main_window_module, "current_gpu_snapshot", lambda: None)
    monkeypatch.setattr(main_window_module, "create_sleep_inhibitor", lambda: FakeSleepInhibitor())
    return settings


class DelayedTranscriptionThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, video_path, options, translation_service, file_index, total_files) -> None:
        super().__init__()
        self._video_path = video_path

    def run(self) -> None:
        self.completed.emit(_transcription_result(self._video_path))
        self.msleep(200)

    def request_stop(self) -> None:
        self.requestInterruption()


class CancellableTranscriptionThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, video_path, options, translation_service, file_index, total_files) -> None:
        super().__init__()

    def request_stop(self) -> None:
        self.requestInterruption()

    def run(self) -> None:
        while not self.isInterruptionRequested():
            self.msleep(20)
        self.cancelled.emit("Processing stopped by user.")


class ImmediateTranscriptionThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, video_path, options, translation_service, file_index, total_files) -> None:
        super().__init__()
        self._video_path = video_path

    def run(self) -> None:
        self.completed.emit(_transcription_result(self._video_path))

    def request_stop(self) -> None:
        self.requestInterruption()


class ImmediateFinalizeThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, transcription, srt_text, options, file_index, total_files) -> None:
        super().__init__()
        self._transcription = transcription
        self._options = options

    def run(self) -> None:
        self.completed.emit(
            PipelineResult(
                input_video=self._transcription.input_video,
                subtitle_file=self._options.output_directory / f"{self._transcription.input_video.stem}.{self._options.target_language}.srt",
                burned_video=None,
                detected_language="el",
                target_language=self._options.target_language,
                translation_provider="openai_compatible",
                device_label="CPU",
                segment_count=2,
                elapsed_seconds=0.5,
                preview_text="1. hello\n2. how are you",
                warning_messages=(),
                stage_durations={
                    "audio_extraction_seconds": 0.2,
                    "whisper_seconds": 1.1,
                    "translation_seconds": 0.5,
                    "finalize_seconds": 0.5,
                },
            )
        )

    def request_stop(self) -> None:
        self.requestInterruption()


class ImmediateExistingBurnThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, video_path, subtitle_path, output_path, *, font_size) -> None:
        super().__init__()
        self._video_path = video_path
        self._subtitle_path = subtitle_path
        self._output_path = output_path

    def run(self) -> None:
        self.completed.emit((self._video_path, self._subtitle_path, self._output_path))

    def request_stop(self) -> None:
        self.requestInterruption()


class CancellableExistingBurnThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, video_path, subtitle_path, output_path, *, font_size) -> None:
        super().__init__()

    def run(self) -> None:
        while not self.isInterruptionRequested():
            self.msleep(20)
        self.cancelled.emit("Processing stopped by user.")

    def request_stop(self) -> None:
        self.requestInterruption()


class RecordingImmediateTranscriptionThread(ImmediateTranscriptionThread):
    started_files: list[str] = []

    def run(self) -> None:
        type(self).started_files.append(self._video_path.name)
        super().run()


class ImmediateThenDelayedPrefetchThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, video_path, options, translation_service, file_index, total_files) -> None:
        super().__init__()
        self._video_path = video_path
        self._file_index = file_index

    def run(self) -> None:
        if self._file_index == 0:
            self.completed.emit(_transcription_result(self._video_path))
            return
        self.msleep(200)
        self.completed.emit(_transcription_result(self._video_path))

    def request_stop(self) -> None:
        self.requestInterruption()


class ImmediateThenCancellablePrefetchThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)
    started_files: list[str] = []

    def __init__(self, video_path, options, translation_service, file_index, total_files) -> None:
        super().__init__()
        self._video_path = video_path
        self._file_index = file_index

    def run(self) -> None:
        type(self).started_files.append(self._video_path.name)
        if self._file_index == 0:
            self.completed.emit(_transcription_result(self._video_path))
            return

        while not self.isInterruptionRequested():
            self.msleep(20)
        self.cancelled.emit("Processing stopped by user.")

    def request_stop(self) -> None:
        self.requestInterruption()


def test_language_catalog_matches_launch_plan() -> None:
    source_codes = [language.code for language in source_language_options()]
    target_codes = [language.code for language in target_language_options()]

    assert source_codes == [
        "auto",
        "en",
        "el",
        "tr",
        "de",
        "fr",
        "it",
        "es",
        "pt",
        "nl",
        "ro",
        "pl",
        "cs",
    ]
    assert target_codes == source_codes[1:]


def test_format_srt_timestamp_rounds_to_milliseconds() -> None:
    assert format_srt_timestamp(3723.456) == "01:02:03,456"


def test_wrap_subtitle_text_keeps_output_compact() -> None:
    wrapped = wrap_subtitle_text(
        "This subtitle line is intentionally long so the formatter has to wrap it neatly.",
        max_line_length=24,
    )
    assert "\n" in wrapped
    assert len(wrapped.splitlines()) <= 2


def test_write_srt_creates_numbered_blocks(tmp_path) -> None:
    output_path = tmp_path / "demo.srt"
    segment_count = write_srt(
        [
            SubtitleSegment(0.0, 1.5, "Gamma"),
            SubtitleSegment(2.0, 3.5, "Delta"),
        ],
        output_path,
        max_line_length=40,
    )

    assert segment_count == 2
    assert output_path.read_text(encoding="utf-8").startswith("1\n00:00:00,000 --> 00:00:01,500")


def test_parse_srt_text_round_trips_translated_segments() -> None:
    srt = segments_to_srt_text(
        [
            SubtitleSegment(0.0, 1.5, "Alpha"),
            SubtitleSegment(2.0, 3.5, "Beta"),
        ],
        max_line_length=40,
    )

    parsed = parse_srt_text(srt)

    assert [segment.text for segment in parsed] == ["Alpha", "Beta"]
    assert parsed[1].start_seconds == 2.0


def test_validate_review_srt_rejects_timing_changes() -> None:
    reference = [
        TranslationSegment(0.0, 1.0, "γειά", "hello"),
        TranslationSegment(1.0, 2.0, "σου", "there"),
    ]
    invalid_srt = (
        "1\n00:00:00,000 --> 00:00:01,500\nhello\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nthere\n"
    )

    assert "changed timing" in validate_review_srt_text(invalid_srt, reference)


def test_validate_review_srt_allows_inserted_missing_blocks() -> None:
    reference = [
        TranslationSegment(0.0, 1.0, "Γειά", "hello"),
        TranslationSegment(1.0, 2.0, "σου", "there"),
    ]
    valid_srt = (
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n"
        "2\n00:00:00,400 --> 00:00:00,800\n[missing line added]\n\n"
        "3\n00:00:01,000 --> 00:00:02,000\nthere\n"
    )

    assert validate_review_srt_text(valid_srt, reference) is None


def test_translation_segments_to_srt_uses_requested_text_field() -> None:
    review_segments = [
        TranslationSegment(0.0, 1.0, "γειά", "hello"),
        TranslationSegment(1.0, 2.0, "κόσμε", "world"),
    ]

    source_srt = translation_segments_to_srt_text(
        review_segments,
        text_field="source_text",
        max_line_length=40,
    )
    translated_srt = translation_segments_to_srt_text(
        review_segments,
        text_field="translated_text",
        max_line_length=40,
    )

    assert "γειά" in source_srt
    assert "hello" in translated_srt


def test_translation_parser_accepts_string_items() -> None:
    assert _translation_text_from_item("Dies ist ein Verbindungstest.") == "Dies ist ein Verbindungstest."


def test_translation_service_retries_smaller_batches_when_count_mismatches(monkeypatch) -> None:
    service = OpenAICompatibleTranslationService(
        TranslationProviderConfig.from_values(
            base_url="https://api.openai.com/v1",
            api_key="secret",
            model="gpt-5.4",
        )
    )
    seen_batch_sizes: list[int] = []

    def fake_translate_batch(segments, *, source_label, target_label, log=None):
        seen_batch_sizes.append(len(segments))
        if len(segments) > 2:
            return ["only one item"]
        return [f"translated-{index}" for index, _segment in enumerate(segments, start=1)]

    monkeypatch.setattr(service, "_translate_batch", fake_translate_batch)

    result = service.translate_segments(
        [
            SubtitleSegment(0.0, 1.0, "one"),
            SubtitleSegment(1.0, 2.0, "two"),
            SubtitleSegment(2.0, 3.0, "three"),
            SubtitleSegment(3.0, 4.0, "four"),
        ],
        source_language="en",
        target_language="de",
    )

    assert seen_batch_sizes == [4, 2, 2]
    assert [segment.translated_text for segment in result] == [
        "translated-1",
        "translated-2",
        "translated-1",
        "translated-2",
    ]


def test_translation_service_runs_batches_in_parallel_and_preserves_order(monkeypatch) -> None:
    service = OpenAICompatibleTranslationService(
        TranslationProviderConfig.from_values(
            base_url="https://api.openai.com/v1",
            api_key="secret",
            model="gpt-5.4",
        ),
        max_parallel_batches=2,
    )
    seen_threads: set[str] = set()
    release = threading.Event()
    started = 0
    lock = threading.Lock()

    def fake_translate_batch(segments, *, source_label, target_label, log=None):
        nonlocal started
        with lock:
            started += 1
            seen_threads.add(threading.current_thread().name)
            if started == 2:
                release.set()
        release.wait(0.5)
        return [f"{segment.text}-translated" for segment in segments]

    monkeypatch.setattr(service, "_translate_batch_adaptive", fake_translate_batch)

    result = service.translate_segments(
        [SubtitleSegment(float(index), float(index + 1), f"segment-{index}") for index in range(25)],
        source_language="en",
        target_language="de",
    )

    assert len(seen_threads) >= 2
    assert [segment.translated_text for segment in result[:3]] == [
        "segment-0-translated",
        "segment-1-translated",
        "segment-2-translated",
    ]
    assert result[-1].translated_text == "segment-24-translated"


def test_translation_service_falls_back_to_sequential_after_transport_error(monkeypatch) -> None:
    service = OpenAICompatibleTranslationService(
        TranslationProviderConfig.from_values(
            base_url="https://api.openai.com/v1",
            api_key="secret",
            model="gpt-5.4",
        ),
        max_parallel_batches=2,
    )
    logs: list[str] = []
    failed_once = False

    def fake_translate_batch(segments, *, source_label, target_label, log=None):
        nonlocal failed_once
        first_text = segments[0].text
        if (
            first_text == "segment-20"
            and threading.current_thread().name != "MainThread"
            and not failed_once
        ):
            failed_once = True
            raise TranslationTransportError("Translation provider request failed: timed out")
        return [f"{segment.text}-translated" for segment in segments]

    monkeypatch.setattr(service, "_translate_batch_adaptive", fake_translate_batch)

    result = service.translate_segments(
        [SubtitleSegment(float(index), float(index + 1), f"segment-{index}") for index in range(25)],
        source_language="en",
        target_language="de",
        log=logs.append,
    )

    assert len(result) == 25
    assert result[20].translated_text == "segment-20-translated"
    assert any("Falling back to sequential translation" in message for message in logs)


def test_pipeline_transcribe_runs_translation_and_preserves_timing(tmp_path) -> None:
    fake_segments = [
        SubtitleSegment(0.0, 1.0, "γειά"),
        SubtitleSegment(1.0, 2.0, "σου"),
    ]
    fake_metadata = TranscriptionMetadata(
        detected_language="el",
        detected_language_probability=None,
        device_label="CPU",
        task_label="transcribe",
    )
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = (fake_segments, fake_metadata)
    translation_service = FakeTranslationService()

    with patch.object(ffmpeg_module, "extract_audio"), patch.object(
        ffmpeg_module, "ffmpeg_binary", return_value="/fake/ffmpeg"
    ):
        pipeline = SubtitlePipeline(
            whisper_service=mock_whisper,
            translation_service=translation_service,
        )
        result = pipeline.transcribe(
            tmp_path / "video.mp4",
            _options(tmp_path, source_language="el", target_language="de"),
        )

    assert translation_service.calls == [("el", "de", ["γειά", "σου"])]
    assert result.review_segments[0].start_seconds == 0.0
    assert result.review_segments[1].end_seconds == 2.0
    assert "DE: γειά" in result.translated_srt_text


def test_worker_serialization_accepts_strenum_like_strings(tmp_path) -> None:
    options = ProcessingOptions(
        source_language="el",
        target_language="de",
        translation_provider="openai_compatible",
        whisper_model="large-v3",
        output_mode="srt_only",  # type: ignore[arg-type]
        output_directory=tmp_path,
        max_line_length=42,
        subtitle_font_size=18,
        workflow_profile="europe_multilingual",  # type: ignore[arg-type]
    )

    payload = _serialize_options(options)

    assert payload["output_mode"] == "srt_only"
    assert payload["workflow_profile"] == "europe_multilingual"


def test_pipeline_transcribe_skips_translation_when_source_matches_target(tmp_path) -> None:
    fake_segments = [SubtitleSegment(0.0, 1.0, "Hello")]
    fake_metadata = TranscriptionMetadata(
        detected_language="en",
        detected_language_probability=None,
        device_label="CPU",
        task_label="transcribe",
    )
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = (fake_segments, fake_metadata)

    with patch.object(ffmpeg_module, "extract_audio"), patch.object(
        ffmpeg_module, "ffmpeg_binary", return_value="/fake/ffmpeg"
    ):
        pipeline = SubtitlePipeline(whisper_service=mock_whisper)
        result = pipeline.transcribe(
            tmp_path / "video.mp4",
            _options(tmp_path, source_language="en", target_language="en"),
        )

    assert result.metadata.translation_applied is False
    assert result.metadata.translation_provider is None
    assert result.review_segments[0].translated_text == "Hello"


def test_pipeline_transcribe_records_auto_detected_language_in_warnings(tmp_path) -> None:
    fake_segments = [
        SubtitleSegment(0.0, 1.0, "γειά"),
        SubtitleSegment(1.0, 2.0, "σου"),
        SubtitleSegment(2.0, 3.0, "φίλε"),
    ]
    fake_metadata = TranscriptionMetadata(
        detected_language="el",
        detected_language_probability=None,
        device_label="CPU",
        task_label="transcribe",
    )
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = (fake_segments, fake_metadata)

    with patch.object(ffmpeg_module, "extract_audio"), patch.object(
        ffmpeg_module, "ffmpeg_binary", return_value="/fake/ffmpeg"
    ):
        pipeline = SubtitlePipeline(
            whisper_service=mock_whisper,
            translation_service=FakeTranslationService(),
        )
        result = pipeline.transcribe(
            tmp_path / "video.mp4",
            _options(tmp_path, source_language="auto", target_language="de"),
        )

    assert result.metadata.detected_language == "el"
    assert any("auto-detected as Greek" in warning for warning in result.warning_messages)


def test_pipeline_finalize_writes_target_language_srt(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    transcription = _transcription_result(video_path)
    options = _options(tmp_path, source_language="el", target_language="en")
    pipeline = SubtitlePipeline()

    result = pipeline.finalize(transcription, transcription.translated_srt_text, options)

    subtitle_file = tmp_path / "video.en.srt"
    assert subtitle_file.exists()
    assert result.segment_count == 2
    assert "hello" in result.preview_text


def test_pipeline_logs_stage_timings_for_transcribe_and_finalize(tmp_path) -> None:
    fake_segments = [
        SubtitleSegment(0.0, 1.0, "γειά"),
        SubtitleSegment(1.0, 2.0, "σου"),
    ]
    fake_metadata = TranscriptionMetadata(
        detected_language="el",
        detected_language_probability=None,
        device_label="CPU",
        task_label="transcribe",
    )
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = (fake_segments, fake_metadata)
    logs: list[str] = []

    with patch.object(ffmpeg_module, "extract_audio"), patch.object(
        ffmpeg_module, "ffmpeg_binary", return_value="/fake/ffmpeg"
    ):
        pipeline = SubtitlePipeline(
            whisper_service=mock_whisper,
            translation_service=FakeTranslationService(),
        )
        transcription = pipeline.transcribe(
            tmp_path / "video.mp4",
            _options(tmp_path, source_language="el", target_language="de"),
            log=logs.append,
        )
        pipeline.finalize(
            transcription,
            transcription.translated_srt_text,
            _options(tmp_path, source_language="el", target_language="de"),
            log=logs.append,
        )

    assert any("Audio extraction completed in" in message for message in logs)
    assert any("Whisper transcription completed in" in message for message in logs)
    assert any("Translation completed in" in message for message in logs)
    assert any("Finalize completed in" in message for message in logs)


def test_whisper_service_uses_faster_whisper_defaults(tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def transcribe(self, audio_path, **kwargs):
            captured["audio_path"] = audio_path
            captured.update(kwargs)
            segment = type("Segment", (), {"start": 0.0, "end": 1.0, "text": "Hello"})()
            info = type(
                "Info",
                (),
                {"language": "el", "language_probability": 0.99, "duration": 1.0},
            )()
            return iter([segment]), info

    service = WhisperService()
    service._get_model = lambda model_name, device: FakeModel()  # type: ignore[method-assign]
    service._backend_name = lambda device: "faster-whisper"  # type: ignore[method-assign]
    service._preferred_device = lambda: "cpu"  # type: ignore[method-assign]

    audio_path = tmp_path / "audio.wav"
    import wave

    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)

    service.transcribe(audio_path, _options(tmp_path, source_language="el", target_language="en"))

    assert Path(str(captured["audio_path"])) == audio_path
    assert captured["language"] == "el"
    assert captured["task"] == "transcribe"
    assert captured["condition_on_previous_text"] is False


def test_whisper_service_reuses_loaded_models_across_instances(monkeypatch) -> None:
    loaded_models: list[tuple[str, str, str]] = []
    monkeypatch.setattr(WhisperService, "_GLOBAL_MODEL_CACHE", {}, raising=False)

    def fake_load_backend_model(model_name: str, device: str, backend: str):
        loaded_models.append((model_name, device, backend))
        return object()

    with patch.object(WhisperService, "_load_backend_model", side_effect=fake_load_backend_model):
        first_service = WhisperService()
        second_service = WhisperService()

        first_model = first_service._get_model("medium", "cpu")
        second_model = second_service._get_model("medium", "cpu")

    assert first_model is second_model
    assert loaded_models == [("medium", "cpu", "faster-whisper")]


def test_whisper_service_preload_model_warms_cache(monkeypatch) -> None:
    loaded_models: list[tuple[str, str, str]] = []
    monkeypatch.setattr(WhisperService, "_GLOBAL_MODEL_CACHE", {}, raising=False)

    def fake_load_backend_model(model_name: str, device: str, backend: str):
        loaded_models.append((model_name, device, backend))
        return object()

    with patch.object(WhisperService, "_load_backend_model", side_effect=fake_load_backend_model):
        service = WhisperService()
        service._preferred_device = lambda: "cpu"  # type: ignore[method-assign]
        service.preload_model("large-v3")
        warmed_model = service._get_model("large-v3", "cpu")

    assert warmed_model is service._model_cache[("large-v3", "cpu", "faster-whisper")]
    assert loaded_models == [("large-v3", "cpu", "faster-whisper")]


def test_ffmpeg_run_can_be_cancelled(monkeypatch) -> None:
    class FakeStream:
        def readline(self):
            return ""

        def close(self) -> None:
            return None

    class FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            self._stopped = False
            self.stderr = FakeStream()

        def poll(self):
            return 0 if self._stopped else None

        def terminate(self) -> None:
            self._stopped = True

        def wait(self, timeout=None) -> int:
            self._stopped = True
            return 0

        def kill(self) -> None:
            self._stopped = True

    monkeypatch.setattr(ffmpeg_module.subprocess, "Popen", lambda *a, **kw: FakeProcess())

    with pytest.raises(OperationCancelledError):
        ffmpeg_module._run_ffmpeg(["ffmpeg"], cancel_requested=lambda: True)


def test_ffmpeg_run_drains_stderr_without_hanging(monkeypatch) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self._lines = ["frame=1\n", "frame=2\n", ""]
            self.closed = False

        def readline(self):
            return self._lines.pop(0)

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            self.stderr = FakeStream()
            self._calls = 0

        def poll(self):
            self._calls += 1
            return 0 if self._calls > 1 else None

    monkeypatch.setattr(ffmpeg_module.subprocess, "Popen", lambda *a, **kw: FakeProcess())
    monkeypatch.setattr(ffmpeg_module, "sleep", lambda *_args: None)

    ffmpeg_module._run_ffmpeg(["ffmpeg"])


def test_app_replaces_missing_console_streams(monkeypatch) -> None:
    created_streams = list(app_module._NULL_STREAMS)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    app_module._NULL_STREAMS.clear()

    try:
        app_module._ensure_standard_streams()

        assert sys.stdout is not None
        assert sys.stderr is not None
        sys.stdout.write("")
        sys.stderr.write("")
    finally:
        for stream in app_module._NULL_STREAMS:
            stream.close()
        app_module._NULL_STREAMS[:] = created_streams


def test_app_opens_window_maximized(monkeypatch) -> None:
    fake_app = MagicMock()
    fake_window = MagicMock()
    fake_icon = MagicMock()
    fake_icon.isNull.return_value = False

    monkeypatch.setattr(app_module, "_ensure_standard_streams", lambda: None)
    monkeypatch.setattr(app_module, "QApplication", lambda argv: fake_app)
    monkeypatch.setattr(app_module, "MainWindow", lambda: fake_window)
    monkeypatch.setattr(app_module, "application_stylesheet", lambda: "")
    monkeypatch.setattr(app_module, "_branding_icon_path", lambda: Path("assets/branding/subtitle-foundry-icon.ico"))
    monkeypatch.setattr(app_module, "QIcon", lambda path: fake_icon)

    fake_app.exec.return_value = 0
    fake_app.windowIcon.return_value = fake_icon

    assert app_module.run() == 0
    fake_app.setWindowIcon.assert_called_once_with(fake_icon)
    fake_window.setWindowIcon.assert_called_once_with(fake_icon)
    fake_window.showMaximized.assert_called_once()


def test_branding_icon_path_uses_runtime_root(monkeypatch, tmp_path) -> None:
    icon_path = tmp_path / "assets" / "branding" / "subtitle-foundry-icon.ico"
    icon_path.parent.mkdir(parents=True)
    icon_path.write_bytes(b"icon")

    monkeypatch.setattr(app_module, "_runtime_root", lambda: tmp_path)

    assert app_module._branding_icon_path() == icon_path


def test_branding_icon_path_returns_none_when_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app_module, "_runtime_root", lambda: tmp_path)

    assert app_module._branding_icon_path() is None


def test_package_main_supports_script_execution(monkeypatch) -> None:
    called = False
    main_module = importlib.import_module("add_subtitles_to_videos.main")

    def fake_main() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(main_module, "main", fake_main)

    runpy.run_path(
        str(Path("src/add_subtitles_to_videos/__main__.py")),
        run_name="__main__",
    )

    assert called is True


def test_main_window_defaults_to_multilingual_workflow(monkeypatch) -> None:
    _application()
    _patch_settings(monkeypatch)

    window = main_window_module.MainWindow()

    assert window.output_mode_combo.currentData() == OutputMode.SRT_ONLY
    assert window.whisper_model_combo.currentData() == "large-v3"
    assert window.source_language_combo.currentData() == "auto"
    assert window.target_language_combo.currentData() == "en"
    window.close()


def test_main_window_shows_launch_languages_and_missing_key_status(monkeypatch) -> None:
    _application()
    _patch_settings(monkeypatch, {"translation/base_url": "https://example.test/v1", "translation/model": "demo-model"})

    window = main_window_module.MainWindow()

    source_items = [window.source_language_combo.itemData(i) for i in range(window.source_language_combo.count())]
    target_items = [window.target_language_combo.itemData(i) for i in range(window.target_language_combo.count())]

    assert source_items == ["auto", "en", "el", "tr", "de", "fr", "it", "es", "pt", "nl", "ro", "pl", "cs"]
    assert target_items == ["en", "el", "tr", "de", "fr", "it", "es", "pt", "nl", "ro", "pl", "cs"]
    assert "missing API key" in window.translation_status_label.text()
    window.close()


def test_main_window_refresh_gpu_status_shows_cuda_snapshot(monkeypatch) -> None:
    _application()
    _patch_settings(monkeypatch)
    monkeypatch.setattr(
        main_window_module,
        "current_gpu_snapshot",
        lambda: GpuSnapshot(
            name="NVIDIA GeForce RTX 4060",
            total_memory_mib=8188,
            free_memory_mib=4096,
            used_memory_mib=4092,
            allocated_memory_mib=2048,
            reserved_memory_mib=2304,
            utilization_gpu_percent=37,
            temperature_c=54,
        ),
    )

    window = main_window_module.MainWindow()

    assert "RTX 4060" in window.gpu_value.text()
    assert "37% util" in window.gpu_value.text()
    assert "4092/8188 MiB used" == window.vram_value.text()

    window.close()


def test_main_window_save_translation_settings_persists_locally(monkeypatch) -> None:
    _application()
    settings = _patch_settings(monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, _title, message: messages.append(message),
    )

    window = main_window_module.MainWindow()
    window.translation_base_url_edit.setText("https://example.test/v1")
    window.translation_model_edit.setText("demo-model")
    window.translation_api_key_edit.setText("secret-key")

    window._save_translation_settings()

    assert settings._values["translation/base_url"] == "https://example.test/v1"
    assert settings._values["translation/model"] == "demo-model"
    assert settings._values["translation/api_key"] == "secret-key"
    assert messages == ["Translation settings saved locally on this machine."]
    window.close()


def test_main_window_test_translation_connection_logs_provider_output(monkeypatch) -> None:
    _application()
    _patch_settings(monkeypatch, {"translation/api_key": "secret"})
    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, _title, message: infos.append(message),
    )

    class DummyService:
        def is_configured(self) -> bool:
            return True

        def configuration_status(self) -> str:
            return "Configured"

        def test_connection(self, *, source_language="en", target_language="de", log=None) -> str:
            if log is not None:
                log("Translation response preview: hallo welt")
            return "hallo welt"

    window = main_window_module.MainWindow()
    window._build_translation_service = lambda: DummyService()  # type: ignore[method-assign]

    window._test_translation_connection()

    assert "Translation test succeeded: hallo welt" in window.log_output.toPlainText()
    assert infos == ["Translation test succeeded.\n\nProvider output:\nhallo welt"]
    window.close()


def test_main_window_blocks_review_when_srt_structure_changes(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)

    window = main_window_module.MainWindow()
    video_path = tmp_path / "demo.mp4"
    window._current_transcription = _transcription_result(video_path)
    window.translated_srt_editor.setPlainText(
        "1\n00:00:00,000 --> 00:00:09,000\nhello\n\n"
        "2\n00:00:01,100 --> 00:00:02,000\nhow are you\n"
    )

    started: list[str] = []
    window._start_finalize = lambda srt_text: started.append(srt_text)  # type: ignore[method-assign]
    window._on_approve_clicked()

    assert not started
    assert not window.review_warning_label.isHidden()
    assert "changed timing" in window.review_warning_label.text()
    window._current_transcription = None
    window.close()


def test_main_window_can_insert_missing_segment_template(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)

    window = main_window_module.MainWindow()
    video_path = tmp_path / "demo.mp4"
    window._show_review(_transcription_result(video_path), 0)

    window._on_insert_missing_segment_clicked()

    updated_text = window.translated_srt_editor.toPlainText()
    assert "[Add missing subtitle text here]" in updated_text
    assert "3\n00:00:02,000 --> 00:00:03,500" in updated_text

    window._on_cancelled("Processing stopped by user.")
    window.close()


def test_main_window_restores_autosaved_review_draft_from_disk(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)
    video_path = tmp_path / "demo.mp4"
    transcription = _transcription_result(video_path)
    drafts_dir = tmp_path / "review-drafts"

    window = main_window_module.MainWindow()
    monkeypatch.setattr(window, "_review_drafts_directory", lambda: drafts_dir)
    window._show_review(transcription, 0)
    edited_text = window.translated_srt_editor.toPlainText().replace("hello", "hello there", 1)
    window.translated_srt_editor.setPlainText(edited_text)
    window._flush_review_draft(force=True)

    draft_files = list(drafts_dir.glob("*.draft.srt"))
    assert len(draft_files) == 1
    assert draft_files[0].read_text(encoding="utf-8") == edited_text

    window._on_cancelled("Processing stopped by user.")
    window.close()

    restored_window = main_window_module.MainWindow()
    monkeypatch.setattr(restored_window, "_review_drafts_directory", lambda: drafts_dir)
    restored_window._show_review(transcription, 0)

    assert restored_window.translated_srt_editor.toPlainText() == edited_text
    assert "Restored a local draft" in restored_window.review_autosave_label.text()

    restored_window._on_cancelled("Processing stopped by user.")
    restored_window.close()


def test_main_window_keeps_transcription_thread_alive_until_finished(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch, {"translation/api_key": "secret"})
    monkeypatch.setattr(main_window_module, "TranscriptionThread", DelayedTranscriptionThread)

    window = main_window_module.MainWindow()
    window._schedule_model_preload = lambda: None  # type: ignore[method-assign]
    video_path = tmp_path / "demo.mp4"
    window._selected_files = [video_path]
    window.output_directory_edit.setText(str(tmp_path))

    window._start_processing()
    thread = window._transcription_thread

    assert thread is not None
    assert _pump_events_until(lambda: window._content_stack.currentIndex() == 1)
    assert thread.isRunning()
    assert window._transcription_thread is thread
    assert _pump_events_until(lambda: window._transcription_thread is None)

    window._on_cancelled("Processing stopped by user.")
    window.close()


def test_main_window_ignores_close_while_transcribing(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch, {"translation/api_key": "secret"})
    monkeypatch.setattr(main_window_module, "TranscriptionThread", DelayedTranscriptionThread)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *a, **kw: None)

    window = main_window_module.MainWindow()
    window._schedule_model_preload = lambda: None  # type: ignore[method-assign]
    video_path = tmp_path / "demo.mp4"
    window._selected_files = [video_path]
    window.output_directory_edit.setText(str(tmp_path))

    window._start_processing()
    assert window._transcription_thread is not None
    assert window._transcription_thread.isRunning()

    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()

    assert _pump_events_until(lambda: window._transcription_thread is None)
    window._on_cancelled("Processing stopped by user.")
    window.close()


def test_main_window_can_stop_active_processing(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch, {"translation/api_key": "secret"})
    monkeypatch.setattr(main_window_module, "TranscriptionThread", CancellableTranscriptionThread)

    window = main_window_module.MainWindow()
    window._schedule_model_preload = lambda: None  # type: ignore[method-assign]
    video_path = tmp_path / "demo.mp4"
    window._selected_files = [video_path]
    window.output_directory_edit.setText(str(tmp_path))

    window._start_processing()
    assert _pump_events_until(lambda: window.stop_button.isEnabled())

    window._request_stop()

    assert _pump_events_until(lambda: window.status_label.text() == "Processing stopped.")
    assert window.queue_value.text() == "Stopped after 0 of 1 finished"
    assert not window.stop_button.isEnabled()
    assert _pump_events_until(lambda: window._transcription_thread is None)

    window.close()


def test_main_window_can_start_standalone_subtitle_burn(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)
    monkeypatch.setattr(main_window_module, "ExistingSubtitleBurnThread", ImmediateExistingBurnThread)

    video_path = tmp_path / "burn-demo.mp4"
    subtitle_path = tmp_path / "burn-demo.en.srt"
    video_path.write_bytes(b"video")
    subtitle_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    window = main_window_module.MainWindow()
    window.output_directory_edit.setText(str(tmp_path))
    window.existing_burn_video_edit.setText(str(video_path))
    window.existing_burn_subtitle_edit.setText(str(subtitle_path))

    window._start_existing_burn()

    assert _pump_events_until(lambda: window.status_label.text() == "Standalone subtitle burn finished.")
    assert window.queue_value.text() == "Standalone burn finished"
    assert "burn-demo.subtitled.mp4" in window.summary_label.text()
    window.close()


def test_main_window_can_stop_standalone_subtitle_burn(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)
    monkeypatch.setattr(main_window_module, "ExistingSubtitleBurnThread", CancellableExistingBurnThread)

    video_path = tmp_path / "burn-demo.mp4"
    subtitle_path = tmp_path / "burn-demo.en.srt"
    video_path.write_bytes(b"video")
    subtitle_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    window = main_window_module.MainWindow()
    window.output_directory_edit.setText(str(tmp_path))
    window.existing_burn_video_edit.setText(str(video_path))
    window.existing_burn_subtitle_edit.setText(str(subtitle_path))

    window._start_existing_burn()
    assert _pump_events_until(lambda: window.stop_button.isEnabled())

    window._request_stop()

    assert _pump_events_until(lambda: window.status_label.text() == "Standalone subtitle burn stopped.")
    assert window.queue_value.text() == "Standalone burn stopped"
    assert _pump_events_until(lambda: window._existing_burn_thread is None)
    window.close()


def test_main_window_releases_sleep_inhibitor_after_queue_finishes(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch, {"translation/api_key": "secret"})
    inhibitor = FakeSleepInhibitor()
    monkeypatch.setattr(main_window_module, "create_sleep_inhibitor", lambda: inhibitor)
    monkeypatch.setattr(main_window_module, "TranscriptionThread", ImmediateTranscriptionThread)
    monkeypatch.setattr(main_window_module, "FinalizeThread", ImmediateFinalizeThread)

    window = main_window_module.MainWindow()
    window._schedule_model_preload = lambda: None  # type: ignore[method-assign]
    window._selected_files = [tmp_path / "demo.mp4"]
    window.output_directory_edit.setText(str(tmp_path))

    window._start_processing()

    assert inhibitor.activations == 1
    assert _pump_events_until(lambda: window._content_stack.currentIndex() == 1)

    window._on_approve_clicked()

    assert _pump_events_until(lambda: window.status_label.text() == "All subtitle jobs finished.")
    assert inhibitor.releases == 1

    window.close()


def test_main_window_queue_smoke_handles_three_files(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch, {"translation/api_key": "secret"})
    monkeypatch.setattr(main_window_module, "TranscriptionThread", ImmediateTranscriptionThread)
    monkeypatch.setattr(main_window_module, "FinalizeThread", ImmediateFinalizeThread)

    window = main_window_module.MainWindow()
    window._schedule_model_preload = lambda: None  # type: ignore[method-assign]
    window._selected_files = [tmp_path / "one.mp4", tmp_path / "two.mp4", tmp_path / "three.mp4"]
    window.output_directory_edit.setText(str(tmp_path))

    window._start_processing()
    assert _pump_events_until(lambda: window._content_stack.currentIndex() == 1)
    window._on_approve_clicked()
    assert _pump_events_until(lambda: window._content_stack.currentIndex() == 1)
    window._on_approve_clicked()
    assert _pump_events_until(lambda: window._content_stack.currentIndex() == 1)
    window._on_approve_clicked()

    assert _pump_events_until(lambda: window.status_label.text() == "All subtitle jobs finished.")
    assert window.queue_value.text() == "3 of 3 finished"
    window.close()


def test_main_window_prefetches_next_file_during_review(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch, {"translation/api_key": "secret"})
    RecordingImmediateTranscriptionThread.started_files = []
    monkeypatch.setattr(main_window_module, "TranscriptionThread", RecordingImmediateTranscriptionThread)

    window = main_window_module.MainWindow()
    window._schedule_model_preload = lambda: None  # type: ignore[method-assign]
    window._selected_files = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    window.output_directory_edit.setText(str(tmp_path))

    window._start_processing()

    assert _pump_events_until(lambda: window._content_stack.currentIndex() == 1)
    assert _pump_events_until(lambda: RecordingImmediateTranscriptionThread.started_files == ["one.mp4", "two.mp4"])
    assert "ready next" in window.review_queue_label.text() or "Preparing two.mp4 in background" in window.review_queue_label.text()

    window._on_cancelled("Processing stopped by user.")
    window.close()


def test_main_window_waits_for_prefetched_review_when_next_file_is_still_running(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch, {"translation/api_key": "secret"})
    monkeypatch.setattr(main_window_module, "TranscriptionThread", ImmediateThenDelayedPrefetchThread)
    monkeypatch.setattr(main_window_module, "FinalizeThread", ImmediateFinalizeThread)

    window = main_window_module.MainWindow()
    window._schedule_model_preload = lambda: None  # type: ignore[method-assign]
    window._selected_files = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    window.output_directory_edit.setText(str(tmp_path))

    window._start_processing()
    assert _pump_events_until(lambda: window.review_file_label.text() == "one.mp4")

    window._on_approve_clicked()

    assert _pump_events_until(lambda: "Waiting for background preparation of two.mp4" in window.status_label.text())
    assert _pump_events_until(lambda: window.review_file_label.text() == "two.mp4")
    assert window._content_stack.currentIndex() == 1

    window._on_cancelled("Processing stopped by user.")
    window.close()


def test_main_window_stop_cancels_background_prefetch_from_review(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch, {"translation/api_key": "secret"})
    ImmediateThenCancellablePrefetchThread.started_files = []
    monkeypatch.setattr(main_window_module, "TranscriptionThread", ImmediateThenCancellablePrefetchThread)

    window = main_window_module.MainWindow()
    window._schedule_model_preload = lambda: None  # type: ignore[method-assign]
    window._selected_files = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    window.output_directory_edit.setText(str(tmp_path))

    window._start_processing()
    assert _pump_events_until(lambda: window.review_file_label.text() == "one.mp4")
    assert _pump_events_until(lambda: ImmediateThenCancellablePrefetchThread.started_files == ["one.mp4", "two.mp4"])

    window._request_stop()

    assert _pump_events_until(lambda: window.status_label.text() == "Processing stopped.")
    assert window._current_transcription is None
    assert window._prefetched_transcription is None
    assert window.queue_value.text() == "Stopped after 0 of 2 finished"

    window.close()
