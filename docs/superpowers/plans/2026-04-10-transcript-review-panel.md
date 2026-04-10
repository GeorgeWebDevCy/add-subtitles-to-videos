# Transcript Review Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an inline, per-file SRT editor panel to the main window so users can review and edit the Whisper transcript before the SRT file is written or burned into video.

**Architecture:** Split `SubtitlePipeline.process_video()` into `transcribe()` and `finalize()`. Replace `BatchProcessingThread` with `TranscriptionThread` + `FinalizeThread`. `MainWindow` orchestrates the per-file loop via a state machine; a `QStackedWidget` switches between the normal layout and a full-width SRT editor panel between the two thread phases.

**Tech Stack:** Python 3.12, PySide6, openai-whisper, imageio-ffmpeg, pytest

---

## File Map

| File | Change |
|---|---|
| `src/add_subtitles_to_videos/models.py` | Add `TranscriptionResult` dataclass |
| `src/add_subtitles_to_videos/services/subtitles.py` | Add `segments_to_srt_text()`; refactor `write_srt()` to call it |
| `src/add_subtitles_to_videos/services/pipeline.py` | Add `transcribe()`, `finalize()`, `_parse_srt_summary()`; keep existing static helpers |
| `src/add_subtitles_to_videos/ui/main_window.py` | Remove `BatchProcessingThread`; add `TranscriptionThread`, `FinalizeThread`; rewrite `MainWindow` state machine; add `QStackedWidget` + review panel |
| `tests/test_subtitles.py` | Add `segments_to_srt_text` test; update two MainWindow tests |

---

## Task 1: Add `TranscriptionResult` to `models.py`

**Files:**
- Modify: `src/add_subtitles_to_videos/models.py`
- Test: `tests/test_subtitles.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_subtitles.py`:

```python
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
```

Also add `TranscriptionResult` to the import at the top of the test file:

```python
from add_subtitles_to_videos.models import (
    OutputMode,
    ProcessingOptions,
    SubtitleMode,
    PipelineResult,
    SubtitleSegment,
    TranscriptionMetadata,
    TranscriptionResult,
)
```

- [ ] **Step 2: Run the test to verify it fails**

```
cd "d:/GitHub Projects/add-subtitles-to-videos"
.venv/Scripts/python -m pytest tests/test_subtitles.py::test_transcription_result_stores_srt_text -v
```

Expected: `ImportError` or `NameError` — `TranscriptionResult` does not exist yet.

- [ ] **Step 3: Add `TranscriptionResult` to `models.py`**

In `src/add_subtitles_to_videos/models.py`, after the `TranscriptionMetadata` dataclass, add:

```python
@dataclass(slots=True)
class TranscriptionResult:
    input_video: Path
    segments: list[SubtitleSegment]
    metadata: TranscriptionMetadata
    warning_messages: tuple[str, ...]
    srt_text: str
```

- [ ] **Step 4: Run the test to verify it passes**

```
.venv/Scripts/python -m pytest tests/test_subtitles.py::test_transcription_result_stores_srt_text -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```
rtk git add src/add_subtitles_to_videos/models.py tests/test_subtitles.py
rtk git commit -m "feat: add TranscriptionResult dataclass to models"
```

---

## Task 2: Add `segments_to_srt_text()` to `subtitles.py`

**Files:**
- Modify: `src/add_subtitles_to_videos/services/subtitles.py`
- Test: `tests/test_subtitles.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_subtitles.py`:

```python
from add_subtitles_to_videos.services.subtitles import (
    format_srt_timestamp,
    segments_to_srt_text,
    wrap_subtitle_text,
    write_srt,
)
```

Replace the existing import line (it currently imports `format_srt_timestamp`, `wrap_subtitle_text`, `write_srt`) with the one above. Then add:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```
.venv/Scripts/python -m pytest tests/test_subtitles.py::test_segments_to_srt_text_produces_valid_srt tests/test_subtitles.py::test_segments_to_srt_text_empty_segments_returns_empty_string -v
```

Expected: `ImportError` — `segments_to_srt_text` does not exist yet.

- [ ] **Step 3: Add `segments_to_srt_text()` and refactor `write_srt()`**

Replace the entire content of `src/add_subtitles_to_videos/services/subtitles.py` with:

```python
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from textwrap import wrap

from ..models import SubtitleSegment


def format_srt_timestamp(total_seconds: float) -> str:
    total_milliseconds = max(0, int(round(total_seconds * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def wrap_subtitle_text(text: str, max_line_length: int) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    wrapped_lines = wrap(
        cleaned,
        width=max_line_length,
        break_long_words=False,
        break_on_hyphens=False,
    )

    if len(wrapped_lines) <= 2:
        return "\n".join(wrapped_lines)

    midpoint = max(1, len(wrapped_lines) // 2)
    first_line = " ".join(wrapped_lines[:midpoint])
    second_line = " ".join(wrapped_lines[midpoint:])
    return "\n".join((first_line, second_line))


def segments_to_srt_text(
    segments: Iterable[SubtitleSegment],
    *,
    max_line_length: int,
) -> str:
    """Format subtitle segments as a valid SRT string without writing to disk.

    Returns an empty string when no segments produce non-empty text.
    """
    blocks: list[str] = []
    seq = 0
    for segment in segments:
        text = wrap_subtitle_text(segment.text, max_line_length).strip()
        if not text:
            continue
        seq += 1
        start = format_srt_timestamp(segment.start_seconds)
        end = format_srt_timestamp(max(segment.end_seconds, segment.start_seconds + 0.01))
        blocks.append(f"{seq}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def write_srt(
    segments: Iterable[SubtitleSegment],
    output_path: Path,
    *,
    max_line_length: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srt_text = segments_to_srt_text(segments, max_line_length=max_line_length)
    output_path.write_text(srt_text if srt_text else "\n", encoding="utf-8")
    return srt_text.strip().count("\n\n") + 1 if srt_text.strip() else 0
```

- [ ] **Step 4: Run all subtitles tests**

```
.venv/Scripts/python -m pytest tests/test_subtitles.py -v -k "not main_window and not pipeline and not whisper and not app"
```

Expected: all pass, including the pre-existing `test_write_srt_creates_numbered_blocks`, `test_format_srt_timestamp_rounds_to_milliseconds`, and `test_wrap_subtitle_text_keeps_output_compact`.

- [ ] **Step 5: Commit**

```
rtk git add src/add_subtitles_to_videos/services/subtitles.py tests/test_subtitles.py
rtk git commit -m "feat: add segments_to_srt_text helper; refactor write_srt to use it"
```

---

## Task 3: Add `transcribe()` and `finalize()` to `SubtitlePipeline`

**Files:**
- Modify: `src/add_subtitles_to_videos/services/pipeline.py`
- Test: `tests/test_subtitles.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_subtitles.py`:

```python
from unittest.mock import MagicMock, patch
from add_subtitles_to_videos.models import TranscriptionResult
from add_subtitles_to_videos.services import ffmpeg as ffmpeg_module


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

    subtitle_file = tmp_path / "video.en.srt"
    assert subtitle_file.exists()
    assert subtitle_file.read_text(encoding="utf-8") == srt_text
    assert result.segment_count == 1
    assert "Hello" in result.preview_text
```

- [ ] **Step 2: Run the tests to verify they fail**

```
.venv/Scripts/python -m pytest tests/test_subtitles.py::test_pipeline_transcribe_returns_transcription_result tests/test_subtitles.py::test_pipeline_finalize_writes_srt_and_returns_result -v
```

Expected: `AttributeError` — `transcribe` / `finalize` methods do not exist yet.

- [ ] **Step 3: Add `transcribe()`, `finalize()`, and `_parse_srt_summary()` to `pipeline.py`**

In `src/add_subtitles_to_videos/services/pipeline.py`, update the imports at the top to include `TranscriptionResult`:

```python
from ..models import OutputMode, PipelineResult, ProcessingOptions, SubtitleMode, TranscriptionResult
```

Then add these three methods to `SubtitlePipeline`, after `process_video()`:

```python
def transcribe(
    self,
    video_path: Path,
    options: ProcessingOptions,
    *,
    progress: ProgressReporter | None = None,
    log: LogReporter | None = None,
) -> TranscriptionResult:
    input_video = video_path.expanduser().resolve()

    self._emit(progress, 0.03, "Preparing audio")
    self._emit_log(log, f"Using FFmpeg binary at {ffmpeg.ffmpeg_binary()}")

    with TemporaryDirectory(prefix="subtitle-foundry-audio-") as temp_dir:
        audio_path = Path(temp_dir) / "audio.wav"
        self._emit_log(log, f"Extracting mono audio from {input_video.name}")
        ffmpeg.extract_audio(input_video, audio_path)

        self._emit(progress, 0.22, "Running Whisper")
        segments, metadata = self._whisper_service.transcribe(
            audio_path,
            options,
            log=log,
        )

    if not segments:
        raise RuntimeError(
            "Whisper returned no subtitle segments. Try a larger model or confirm the source language."
        )

    warning_messages = self._build_review_flags(segments, options, metadata)
    for warning in warning_messages:
        self._emit_log(log, f"Review flag: {warning}")

    srt_text = write_srt_to_string(segments, max_line_length=options.max_line_length)
    self._emit(progress, 0.76, "Transcription complete — ready for review")

    return TranscriptionResult(
        input_video=input_video,
        segments=segments,
        metadata=metadata,
        warning_messages=warning_messages,
        srt_text=srt_text,
    )

def finalize(
    self,
    transcription: TranscriptionResult,
    srt_text: str,
    options: ProcessingOptions,
    *,
    progress: ProgressReporter | None = None,
    log: LogReporter | None = None,
) -> PipelineResult:
    started_at = perf_counter()
    video_path = transcription.input_video
    output_directory = options.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    subtitle_suffix = "en" if options.subtitle_mode == SubtitleMode.ENGLISH else "native"
    subtitle_file = output_directory / f"{video_path.stem}.{subtitle_suffix}.srt"
    burned_video = (
        output_directory / f"{video_path.stem}.subtitled{video_path.suffix}"
        if options.output_mode == OutputMode.BURNED_VIDEO
        else None
    )

    self._emit(progress, 0.76, "Writing subtitle file")
    subtitle_file.parent.mkdir(parents=True, exist_ok=True)
    subtitle_file.write_text(srt_text, encoding="utf-8")
    segment_count, preview_text = self._parse_srt_summary(srt_text)
    self._emit_log(log, f"Wrote subtitle file to {subtitle_file.name}")

    if burned_video is not None:
        self._emit(progress, 0.86, "Burning subtitles into video")
        ffmpeg.burn_subtitles(
            video_path,
            subtitle_file,
            burned_video,
            font_size=options.subtitle_font_size,
        )
        self._emit_log(log, f"Wrote subtitled video to {burned_video.name}")

    elapsed_seconds = perf_counter() - started_at
    self._emit(progress, 1.0, "Done")

    return PipelineResult(
        input_video=video_path,
        subtitle_file=subtitle_file,
        burned_video=burned_video,
        detected_language=transcription.metadata.detected_language,
        device_label=transcription.metadata.device_label,
        segment_count=segment_count,
        elapsed_seconds=elapsed_seconds,
        preview_text=preview_text,
        warning_messages=transcription.warning_messages,
    )

@staticmethod
def _parse_srt_summary(srt_text: str) -> tuple[int, str]:
    """Return (segment_count, preview_text) parsed from a raw SRT string."""
    blocks = [b.strip() for b in srt_text.strip().split("\n\n") if b.strip()]
    segment_count = len(blocks)
    preview_lines: list[str] = []
    for i, block in enumerate(blocks[:5], start=1):
        lines = block.splitlines()
        text = " ".join(lines[2:]) if len(lines) > 2 else ""
        if text:
            preview_lines.append(f"{i}. {text}")
    return segment_count, "\n".join(preview_lines)
```

Also update the import of `write_srt` at the top of `pipeline.py` to also import `segments_to_srt_text` under an alias:

```python
from .subtitles import segments_to_srt_text as write_srt_to_string, write_srt
```

- [ ] **Step 4: Run the new tests**

```
.venv/Scripts/python -m pytest tests/test_subtitles.py::test_pipeline_transcribe_returns_transcription_result tests/test_subtitles.py::test_pipeline_finalize_writes_srt_and_returns_result -v
```

Expected: both `PASSED`.

- [ ] **Step 5: Run the full test suite to check for regressions**

```
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all existing tests still pass (the two MainWindow tests will fail if run — that is expected and will be fixed in Task 7).

- [ ] **Step 6: Commit**

```
rtk git add src/add_subtitles_to_videos/services/pipeline.py src/add_subtitles_to_videos/models.py tests/test_subtitles.py
rtk git commit -m "feat: split SubtitlePipeline into transcribe() and finalize() methods"
```

---

## Task 4: Add `TranscriptionThread` and `FinalizeThread`; remove `BatchProcessingThread`

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`

- [ ] **Step 1: Add imports for new types at the top of `main_window.py`**

Update the models import line (currently ends with `PipelineResult, ProcessingOptions`) to also include `TranscriptionResult`:

```python
from ..models import PipelineResult, ProcessingOptions, TranscriptionResult
```

- [ ] **Step 2: Replace `BatchProcessingThread` with `TranscriptionThread` and `FinalizeThread`**

Delete the entire `BatchProcessingThread` class (lines 48–95) and replace it with:

```python
class TranscriptionThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)  # TranscriptionResult
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        options: ProcessingOptions,
        file_index: int,
        total_files: int,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._options = options
        self._file_index = file_index
        self._total_files = total_files

    def run(self) -> None:
        try:
            pipeline = SubtitlePipeline()

            def on_progress(stage_progress: float, message: str) -> None:
                overall = int(((self._file_index + stage_progress) / self._total_files) * 100)
                self.progress_changed.emit(overall, f"{self._video_path.name}: {message}")

            def on_log(message: str) -> None:
                self.log_message.emit(f"{self._video_path.name}: {message}")

            result = pipeline.transcribe(
                self._video_path,
                self._options,
                progress=on_progress,
                log=on_log,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class FinalizeThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)  # PipelineResult
    failed = Signal(str)

    def __init__(
        self,
        transcription: TranscriptionResult,
        srt_text: str,
        options: ProcessingOptions,
        file_index: int,
        total_files: int,
    ) -> None:
        super().__init__()
        self._transcription = transcription
        self._srt_text = srt_text
        self._options = options
        self._file_index = file_index
        self._total_files = total_files

    def run(self) -> None:
        try:
            pipeline = SubtitlePipeline()
            video_name = self._transcription.input_video.name

            def on_progress(stage_progress: float, message: str) -> None:
                overall = int(((self._file_index + stage_progress) / self._total_files) * 100)
                self.progress_changed.emit(overall, f"{video_name}: {message}")

            def on_log(message: str) -> None:
                self.log_message.emit(f"{video_name}: {message}")

            result = pipeline.finalize(
                self._transcription,
                self._srt_text,
                self._options,
                progress=on_progress,
                log=on_log,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
```

- [ ] **Step 3: Commit**

```
rtk git add src/add_subtitles_to_videos/ui/main_window.py
rtk git commit -m "feat: add TranscriptionThread and FinalizeThread; remove BatchProcessingThread"
```

---

## Task 5: Rewrite `MainWindow` state machine

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`

- [ ] **Step 1: Update `MainWindow.__init__` — replace `_worker` with new state vars**

In `MainWindow.__init__`, replace:

```python
self._worker: BatchProcessingThread | None = None
```

with:

```python
self._current_options: ProcessingOptions | None = None
self._current_file_index: int = 0
self._all_results: list[PipelineResult] = []
self._current_transcription: TranscriptionResult | None = None
self._transcription_thread: TranscriptionThread | None = None
self._finalize_thread: FinalizeThread | None = None
```

- [ ] **Step 2: Rewrite `_start_processing()`**

Replace the entire `_start_processing` method with:

```python
def _start_processing(self) -> None:
    transcribing = self._transcription_thread is not None and self._transcription_thread.isRunning()
    finalizing = self._finalize_thread is not None and self._finalize_thread.isRunning()
    if transcribing or finalizing:
        return

    if not self._selected_files:
        QMessageBox.warning(self, APP_NAME, "Add at least one video before starting.")
        return

    output_directory_text = self.output_directory_edit.text().strip()
    if not output_directory_text:
        QMessageBox.warning(self, APP_NAME, "Choose an output directory first.")
        return

    self._current_options = ProcessingOptions(
        source_language=self.source_language_combo.currentData(),
        subtitle_mode=self.subtitle_mode_combo.currentData(),
        whisper_model=self.whisper_model_combo.currentData(),
        output_mode=self.output_mode_combo.currentData(),
        output_directory=Path(output_directory_text),
        max_line_length=self.max_line_length_spinbox.value(),
        subtitle_font_size=self.subtitle_font_size_spinbox.value(),
    )
    self._current_file_index = 0
    self._all_results = []

    self.progress_bar.setValue(0)
    self.status_label.setText("Starting subtitle pipeline...")
    self.active_file_value.setText("Preparing queue")
    self.queue_value.setText(f"0 of {len(self._selected_files)} finished")
    self.engine_value.setText("Detecting best device")
    self.review_flags_label.setText("Automatic review flags will appear after transcription.")
    self.preview_output.setPlainText("")
    self._job_started_at = datetime.now()
    self._elapsed_timer.start()
    self._update_elapsed_label()
    self._set_busy(True)

    self._start_next_file()
```

- [ ] **Step 3: Add `_start_next_file()` method**

Add after `_start_processing()`:

```python
def _start_next_file(self) -> None:
    assert self._current_options is not None
    video_path = self._selected_files[self._current_file_index]
    total = len(self._selected_files)

    self.active_file_value.setText(video_path.name)
    self.queue_value.setText(f"File {self._current_file_index + 1} of {total}")
    self._append_log(f"[{self._current_file_index + 1}/{total}] Starting {video_path.name}")

    thread = TranscriptionThread(
        video_path,
        self._current_options,
        self._current_file_index,
        total,
    )
    thread.progress_changed.connect(self._on_progress)
    thread.log_message.connect(self._append_log)
    thread.completed.connect(self._on_transcription_completed)
    thread.failed.connect(self._on_failed)
    thread.finished.connect(lambda t=thread: self._on_thread_finished(t))
    self._transcription_thread = thread
    thread.start()
```

- [ ] **Step 4: Add `_on_transcription_completed()`, `_on_approve_clicked()`, `_on_use_original_clicked()`, `_start_finalize()`, `_on_finalize_completed()`, `_on_all_done()`**

Add these methods:

```python
def _on_transcription_completed(self, result: TranscriptionResult) -> None:
    self._current_transcription = result
    total = len(self._selected_files)

    for warning in result.warning_messages:
        self._append_log(f"{result.input_video.name}: Review flag: {warning}")

    self.review_file_label.setText(result.input_video.name)
    self.review_queue_label.setText(f"File {self._current_file_index + 1} of {total}")
    self.srt_editor.setPlainText(result.srt_text)
    self.review_warning_label.setVisible(False)
    self.status_label.setText(f"Review transcript for {result.input_video.name}")
    self._content_stack.setCurrentIndex(1)

def _on_approve_clicked(self) -> None:
    srt_text = self.srt_editor.toPlainText().strip()
    if not srt_text:
        self.review_warning_label.setVisible(True)
        return
    self._start_finalize(srt_text)

def _on_use_original_clicked(self) -> None:
    assert self._current_transcription is not None
    self._start_finalize(self._current_transcription.srt_text)

def _start_finalize(self, srt_text: str) -> None:
    assert self._current_options is not None
    assert self._current_transcription is not None

    self._content_stack.setCurrentIndex(0)
    self.status_label.setText(
        f"Writing subtitles for {self._current_transcription.input_video.name}..."
    )

    total = len(self._selected_files)
    thread = FinalizeThread(
        self._current_transcription,
        srt_text,
        self._current_options,
        self._current_file_index,
        total,
    )
    thread.progress_changed.connect(self._on_progress)
    thread.log_message.connect(self._append_log)
    thread.completed.connect(self._on_finalize_completed)
    thread.failed.connect(self._on_failed)
    thread.finished.connect(lambda t=thread: self._on_thread_finished(t))
    self._finalize_thread = thread
    self._current_transcription = None
    thread.start()

def _on_finalize_completed(self, result: PipelineResult) -> None:
    self._all_results.append(result)
    self._refresh_summary(result)
    self._current_file_index += 1
    if self._current_file_index < len(self._selected_files):
        self._start_next_file()
    else:
        self._on_all_done()

def _on_all_done(self) -> None:
    self._set_busy(False)
    self._elapsed_timer.stop()
    self.status_label.setText("All subtitle jobs finished.")
    self.progress_bar.setValue(100)
    self._append_log(f"Finished all {len(self._all_results)} queued videos.")
    self.queue_value.setText(f"{len(self._all_results)} of {len(self._selected_files)} finished")
```

- [ ] **Step 5: Replace `_on_completed()` and `_on_failed()` and `_on_worker_finished()`**

Delete `_on_completed`, `_on_failed`, and `_on_worker_finished`. Add a new `_on_failed` and `_on_thread_finished`:

```python
def _on_failed(self, error_message: str) -> None:
    self._set_busy(False)
    self._elapsed_timer.stop()
    self._content_stack.setCurrentIndex(0)
    self.status_label.setText("The subtitle pipeline hit an error.")
    self._append_log(error_message)
    self.review_flags_label.setText(
        "Processing stopped with an error. Check the session log for the exact FFmpeg or Whisper failure."
    )
    QMessageBox.critical(self, APP_NAME, error_message)

def _on_thread_finished(self, thread: QThread) -> None:
    if self._transcription_thread is thread:
        self._transcription_thread = None
    elif self._finalize_thread is thread:
        self._finalize_thread = None
    thread.deleteLater()
```

- [ ] **Step 6: Update `closeEvent()` and `_set_busy()`**

Replace `closeEvent`:

```python
def closeEvent(self, event: QCloseEvent) -> None:
    active = self._transcription_thread or self._finalize_thread
    if active is not None and active.isRunning():
        QMessageBox.information(
            self,
            APP_NAME,
            "Subtitle generation is still running. Wait for the current job to finish before closing the app.",
        )
        event.ignore()
        return

    for thread in (self._transcription_thread, self._finalize_thread):
        if thread is not None:
            thread.wait(1000)
    super().closeEvent(event)
```

Also delete the `_on_job_started` method — it is no longer used (the state machine handles the active file label directly in `_start_next_file`).

- [ ] **Step 7: Commit**

```
rtk git add src/add_subtitles_to_videos/ui/main_window.py
rtk git commit -m "feat: rewrite MainWindow state machine for per-file transcribe/review/finalize loop"
```

---

## Task 6: Add `QStackedWidget` and review panel to `MainWindow` UI

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`

- [ ] **Step 1: Add `QStackedWidget` to the imports**

In the `PySide6.QtWidgets` import block, add `QStackedWidget`:

```python
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
```

- [ ] **Step 2: Update `_build_ui()` to wrap content in a `QStackedWidget`**

Replace the existing `_build_ui` method with:

```python
def _build_ui(self) -> None:
    root = QWidget()
    root.setObjectName("root")
    root_layout = QVBoxLayout(root)
    root_layout.setContentsMargins(22, 22, 22, 22)
    root_layout.setSpacing(18)

    root_layout.addWidget(self._create_hero_card())

    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.setChildrenCollapsible(False)
    splitter.addWidget(self._create_left_column())
    splitter.addWidget(self._create_right_column())
    splitter.setSizes([640, 520])

    self._content_stack = QStackedWidget()
    self._content_stack.addWidget(splitter)                  # index 0 — normal view
    self._content_stack.addWidget(self._create_review_panel())  # index 1 — review mode
    root_layout.addWidget(self._content_stack, stretch=1)

    self.setCentralWidget(root)
```

- [ ] **Step 3: Add `_create_review_panel()` method**

Add this method to `MainWindow` (alongside the other `_create_*` methods):

```python
def _create_review_panel(self) -> QWidget:
    card = self._card()
    layout = QVBoxLayout(card)
    layout.setContentsMargins(22, 22, 22, 22)
    layout.setSpacing(14)

    header_row = QHBoxLayout()
    header_row.setSpacing(16)
    header_row.addWidget(self._section_title("Review Transcript"))

    self.review_file_label = QLabel()
    self.review_file_label.setObjectName("statusValue")
    self.review_file_label.setWordWrap(False)
    header_row.addWidget(self.review_file_label, stretch=1)

    self.review_queue_label = QLabel()
    self.review_queue_label.setObjectName("supportingText")
    header_row.addWidget(self.review_queue_label)

    layout.addLayout(header_row)

    self.srt_editor = QPlainTextEdit()
    self.srt_editor.setPlaceholderText(
        "The Whisper SRT transcript will appear here. Edit any mistakes before continuing."
    )
    layout.addWidget(self.srt_editor, stretch=1)

    self.review_warning_label = QLabel("SRT content cannot be empty.")
    self.review_warning_label.setObjectName("warningText")
    self.review_warning_label.setVisible(False)
    layout.addWidget(self.review_warning_label)

    button_row = QHBoxLayout()
    button_row.setSpacing(10)
    button_row.addStretch(1)

    self.use_original_button = QPushButton("Use Original")
    self.use_original_button.setObjectName("secondaryButton")
    self.use_original_button.clicked.connect(self._on_use_original_clicked)

    self.approve_button = QPushButton("Approve & Continue")
    self.approve_button.setObjectName("runButton")
    self.approve_button.clicked.connect(self._on_approve_clicked)

    button_row.addWidget(self.use_original_button)
    button_row.addWidget(self.approve_button)
    layout.addLayout(button_row)

    return card
```

- [ ] **Step 4: Add `"warningText"` style to the theme**

Open `src/add_subtitles_to_videos/ui/theme.py`. Find where `"supportingText"` is styled and add a warning style nearby. Read the file first to find the right location, then add:

```python
QLabel#warningText {
    color: #e05c5c;
    font-size: 12px;
}
```

(Use the same CSS-in-QSS syntax as the rest of the file.)

- [ ] **Step 5: Manually verify the app launches**

```
.venv/Scripts/python -m add_subtitles_to_videos
```

The app window should open normally showing the existing layout. No errors in the terminal.

- [ ] **Step 6: Commit**

```
rtk git add src/add_subtitles_to_videos/ui/main_window.py src/add_subtitles_to_videos/ui/theme.py
rtk git commit -m "feat: add QStackedWidget and review panel to MainWindow UI"
```

---

## Task 7: Update tests for the new architecture

**Files:**
- Modify: `tests/test_subtitles.py`

- [ ] **Step 1: Add `DelayedTranscriptionThread` stub and replace the two broken tests**

In `tests/test_subtitles.py`, replace the `DelayedCompletionThread` class and the two `test_main_window_*` tests with:

```python
from add_subtitles_to_videos.models import TranscriptionResult


class DelayedTranscriptionThread(QThread):
    """Fake TranscriptionThread that emits a TranscriptionResult then sleeps 200 ms."""

    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, video_path, options, file_index, total_files) -> None:
        super().__init__()
        self._video_path = video_path

    def run(self) -> None:
        result = TranscriptionResult(
            input_video=self._video_path,
            segments=[],
            metadata=TranscriptionMetadata(
                detected_language="el",
                detected_language_probability=None,
                device_label="CPU",
                task_label="transcribe",
            ),
            warning_messages=(),
            srt_text="",
        )
        self.completed.emit(result)
        self.msleep(200)


def test_main_window_keeps_transcription_thread_alive_until_finished(
    monkeypatch, tmp_path
) -> None:
    _application()
    monkeypatch.setattr(main_window_module, "TranscriptionThread", DelayedTranscriptionThread)

    window = main_window_module.MainWindow()
    video_path = tmp_path / "demo.mp4"
    window._selected_files = [video_path]
    window.output_directory_edit.setText(str(tmp_path))

    window._start_processing()
    thread = window._transcription_thread

    assert thread is not None
    # After completed fires, review panel is shown; thread is still in msleep
    assert _pump_events_until(lambda: window._content_stack.currentIndex() == 1)
    assert thread.isRunning()
    assert window._transcription_thread is thread
    # After finished fires, the reference is cleared
    assert _pump_events_until(lambda: window._transcription_thread is None)

    window.close()


def test_main_window_ignores_close_while_transcribing(monkeypatch, tmp_path) -> None:
    _application()
    monkeypatch.setattr(main_window_module, "TranscriptionThread", DelayedTranscriptionThread)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *a, **kw: None)

    window = main_window_module.MainWindow()
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
    window.close()
```

Also update the imports at the top of the test file to remove `BatchProcessingThread` (no longer exists) and add the new thread classes:

```python
from add_subtitles_to_videos.ui.main_window import (
    TranscriptionThread,
    FinalizeThread,
)
```

(These imports are only needed if you reference the classes by name in tests. The monkeypatch approach above doesn't require them, so just remove any reference to `BatchProcessingThread`.)

- [ ] **Step 2: Run the full test suite**

```
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```
rtk git add tests/test_subtitles.py
rtk git commit -m "test: update MainWindow tests for TranscriptionThread/FinalizeThread architecture"
```

---

## Self-Review Notes

- **Spec coverage:** All four spec sections (pipeline split, threading, review panel UI, error handling) are fully covered across Tasks 1–7.
- **Type consistency:** `TranscriptionResult` is defined in Task 1 and used consistently by name in Tasks 3–5 and 7. `srt_text: str` field (not property) is used throughout.
- **`_on_job_started` removal:** The old `job_started` signal and `_on_job_started` slot are removed in Task 5. The active file label is set directly in `_start_next_file()`.
- **`warningText` QSS:** Task 6 Step 4 adds the warning style — read `theme.py` before editing to match its existing format.
- **`segments_to_srt_text` import alias:** Named `write_srt_to_string` in `pipeline.py` to avoid clash with the `write_srt` function import.
- **Empty `srt_text` in test stub:** `DelayedTranscriptionThread` emits `srt_text=""`. The review panel will show an empty editor. The test checks thread lifecycle only, not the editor content.
