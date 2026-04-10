import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from add_subtitles_to_videos import app as app_module
from add_subtitles_to_videos.models import (
    OutputMode,
    ProcessingOptions,
    SubtitleMode,
    PipelineResult,
    SubtitleSegment,
    TranscriptionMetadata,
    TranscriptionResult,
)
from add_subtitles_to_videos.services.pipeline import SubtitlePipeline
from add_subtitles_to_videos.services.subtitles import (
    format_srt_timestamp,
    segments_to_srt_text,
    wrap_subtitle_text,
    write_srt,
)
from add_subtitles_to_videos.services.whisper import WhisperService
from add_subtitles_to_videos.services import ffmpeg as ffmpeg_module
from add_subtitles_to_videos.ui import main_window as main_window_module


def test_transcription_result_stores_srt_text() -> None:
    result = TranscriptionResult(
        input_video=Path("video.mp4"),
        segments=[SubtitleSegment(0.0, 1.0, "Hello")],
        metadata=TranscriptionMetadata(detected_language="en", detected_language_probability=None),
        warning_messages=(),
        srt_text="1\n00:00:00,000 --> 00:00:01,000\nHello\n",
    )
    assert result.srt_text.startswith("1\n")
    assert "Hello" in result.srt_text


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


class DelayedCompletionThread(QThread):
    progress_changed = Signal(int, str)
    job_started = Signal(int, int, str)
    log_message = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, files, options) -> None:
        super().__init__()
        self._files = files

    def run(self) -> None:
        file_name = self._files[0].name
        self.job_started.emit(1, 1, file_name)
        self.completed.emit(
            [
                PipelineResult(
                    input_video=self._files[0],
                    subtitle_file=self._files[0].with_suffix(".en.srt"),
                    burned_video=None,
                    detected_language="el",
                    device_label="CUDA GPU",
                    segment_count=4,
                    elapsed_seconds=0.2,
                    preview_text="1. preview",
                    warning_messages=(),
                )
            ]
        )
        self.msleep(200)


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


def test_segments_to_srt_text_produces_valid_srt() -> None:
    srt = segments_to_srt_text(
        [
            SubtitleSegment(0.0, 1.5, "Alpha"),
            SubtitleSegment(2.0, 3.5, "Beta"),
        ],
        max_line_length=40,
    )
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\nAlpha")
    assert "2\n00:00:02,000 --> 00:00:03,500\nBeta" in srt


def test_segments_to_srt_text_empty_segments_returns_empty_string() -> None:
    assert segments_to_srt_text([], max_line_length=40) == ""


def test_pipeline_flags_language_mismatch_and_non_english_translation() -> None:
    options = ProcessingOptions(
        source_language="el",
        subtitle_mode=SubtitleMode.ENGLISH,
        whisper_model="medium",
        output_mode=OutputMode.SRT_ONLY,
        output_directory=Path("exports"),
        max_line_length=42,
        subtitle_font_size=18,
    )
    metadata = TranscriptionMetadata(
        detected_language="tr",
        detected_language_probability=None,
        duration_seconds=12.0,
        device_label="CUDA GPU",
        task_label="translate",
    )
    segments = [
        SubtitleSegment(0.0, 1.0, "Καλημέρα σας"),
        SubtitleSegment(1.0, 2.0, "είμαστε εδώ"),
        SubtitleSegment(2.0, 3.0, "για δοκιμή"),
    ]

    warnings = SubtitlePipeline._build_review_flags(segments, options, metadata)

    assert any("detected 'tr'" in warning for warning in warnings)
    assert any("contains a lot of Greek text" in warning for warning in warnings)


def test_pipeline_preview_contains_first_lines() -> None:
    preview = SubtitlePipeline._build_preview_text(
        [
            SubtitleSegment(0.0, 1.0, "First line"),
            SubtitleSegment(1.0, 2.0, "Second line"),
        ]
    )

    assert "1. First line" in preview
    assert "2. Second line" in preview


def test_whisper_service_disables_console_progress_in_windowed_apps(tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            return {
                "language": "el",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
            }

    service = WhisperService()
    service._get_model = lambda model_name, device: FakeModel()  # type: ignore[method-assign]
    service._preferred_device = lambda: "cpu"  # type: ignore[method-assign]

    audio_path = tmp_path / "audio.wav"
    import wave

    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)

    options = ProcessingOptions(
        source_language="el",
        subtitle_mode=SubtitleMode.ENGLISH,
        whisper_model="medium",
        output_mode=OutputMode.SRT_ONLY,
        output_directory=tmp_path,
        max_line_length=42,
        subtitle_font_size=18,
    )

    service.transcribe(audio_path, options)

    assert captured["verbose"] is None


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


def test_pipeline_transcribe_returns_transcription_result(tmp_path) -> None:
    fake_segments = [SubtitleSegment(0.0, 1.0, "Hello")]
    fake_metadata = TranscriptionMetadata(
        detected_language="en",
        detected_language_probability=None,
        device_label="CPU",
        task_label="transcribe",
    )

    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = (fake_segments, fake_metadata)

    options = ProcessingOptions(
        source_language="en",
        subtitle_mode=SubtitleMode.SOURCE,
        whisper_model="base",
        output_mode=OutputMode.SRT_ONLY,
        output_directory=tmp_path,
        max_line_length=42,
        subtitle_font_size=18,
    )

    with patch.object(ffmpeg_module, "extract_audio"), \
         patch.object(ffmpeg_module, "ffmpeg_binary", return_value="/fake/ffmpeg"):
        pipeline = SubtitlePipeline(whisper_service=mock_whisper)
        result = pipeline.transcribe(tmp_path / "video.mp4", options)

    assert isinstance(result, TranscriptionResult)
    assert result.segments == fake_segments
    assert result.metadata == fake_metadata
    assert "Hello" in result.srt_text


def test_pipeline_finalize_writes_srt_and_returns_result(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    srt_text = "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
    fake_metadata = TranscriptionMetadata(
        detected_language="en",
        detected_language_probability=None,
        device_label="CPU",
        task_label="transcribe",
    )
    transcription = TranscriptionResult(
        input_video=video_path,
        segments=[SubtitleSegment(0.0, 1.0, "Hello")],
        metadata=fake_metadata,
        warning_messages=(),
        srt_text=srt_text,
    )
    options = ProcessingOptions(
        source_language="en",
        subtitle_mode=SubtitleMode.SOURCE,
        whisper_model="base",
        output_mode=OutputMode.SRT_ONLY,
        output_directory=tmp_path,
        max_line_length=42,
        subtitle_font_size=18,
    )

    pipeline = SubtitlePipeline()
    result = pipeline.finalize(transcription, srt_text, options)

    subtitle_file = tmp_path / "video.native.srt"
    assert subtitle_file.exists()
    assert subtitle_file.read_text(encoding="utf-8") == srt_text
    assert result.segment_count == 1
    assert "Hello" in result.preview_text


def test_main_window_keeps_worker_alive_until_finished(monkeypatch, tmp_path) -> None:
    _application()
    monkeypatch.setattr(main_window_module, "BatchProcessingThread", DelayedCompletionThread)

    window = main_window_module.MainWindow()
    video_path = tmp_path / "demo.mp4"
    window._selected_files = [video_path]
    window.output_directory_edit.setText(str(tmp_path))

    window._start_processing()
    worker = window._worker

    assert worker is not None
    assert _pump_events_until(lambda: window.status_label.text() == "All subtitle jobs finished.")
    assert worker.isRunning()
    assert window._worker is worker
    assert _pump_events_until(lambda: window._worker is None)

    window.close()


def test_main_window_ignores_close_while_worker_is_running(monkeypatch, tmp_path) -> None:
    _application()
    monkeypatch.setattr(main_window_module, "BatchProcessingThread", DelayedCompletionThread)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *args, **kwargs: None)

    window = main_window_module.MainWindow()
    video_path = tmp_path / "demo.mp4"
    window._selected_files = [video_path]
    window.output_directory_edit.setText(str(tmp_path))

    window._start_processing()
    assert window._worker is not None
    assert window._worker.isRunning()

    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()

    assert _pump_events_until(lambda: window._worker is None)
    window.close()
