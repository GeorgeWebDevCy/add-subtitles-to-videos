# Transcript Review Panel — Design Spec

**Date:** 2026-04-10
**Status:** Approved

## Overview

Add an inline transcript review step between Whisper transcription and SRT/burn output. After each file is transcribed, the pipeline pauses and shows a full-width SRT editor panel in the main window. The user edits the raw SRT text (or accepts Whisper's output as-is), then the pipeline continues with write + optional burn.

Works for both single-file and multi-file (batch) workflows. Each file in the queue gets its own review step.

---

## Pipeline Changes

### New method: `SubtitlePipeline.transcribe()`

```python
def transcribe(
    self,
    video_path: Path,
    options: ProcessingOptions,
    *,
    progress: ProgressReporter | None = None,
    log: LogReporter | None = None,
) -> TranscriptionResult:
```

Extracts audio and runs Whisper. Returns a `TranscriptionResult` (new dataclass). Does **not** write any files.

### New method: `SubtitlePipeline.finalize()`

```python
def finalize(
    self,
    video_path: Path,
    srt_text: str,
    options: ProcessingOptions,
    *,
    progress: ProgressReporter | None = None,
    log: LogReporter | None = None,
) -> PipelineResult:
```

Accepts the (possibly edited) SRT text as a plain string. Writes it to the `.srt` file and optionally burns it into the video. No Whisper involved.

### New dataclass: `TranscriptionResult`

```python
@dataclass(slots=True)
class TranscriptionResult:
    input_video: Path
    segments: list[SubtitleSegment]
    metadata: TranscriptionMetadata
    warning_messages: tuple[str, ...]
    max_line_length: int

    @property
    def srt_text(self) -> str:
        return segments_to_srt_text(self.segments, self.max_line_length)
```

### New helper: `subtitles.segments_to_srt_text()`

```python
def segments_to_srt_text(
    segments: list[SubtitleSegment],
    max_line_length: int = DEFAULT_MAX_LINE_LENGTH,
) -> str:
```

Formats segments into a valid SRT string without writing to disk. Used to pre-populate the editor.

The existing `write_srt()` is updated to call `segments_to_srt_text()` internally, eliminating duplicated formatting logic.

---

## Threading

`BatchProcessingThread` is removed and replaced with two focused thread classes.

### `TranscriptionThread`

Runs `pipeline.transcribe()` for a single file.

Signals:
- `progress_changed(int, str)`
- `log_message(str)`
- `completed(TranscriptionResult)`
- `failed(str)`

### `FinalizeThread`

Runs `pipeline.finalize()` for a single file.

Signals:
- `progress_changed(int, str)`
- `log_message(str)`
- `completed(PipelineResult)`
- `failed(str)`

---

## MainWindow State Machine

`MainWindow` orchestrates the per-file queue:

```
[idle] → _start_next_file()
    → launch TranscriptionThread
    → [transcribing]
        → on completed → switch to review panel
        → [reviewing] (no thread running)
            → user clicks "Approve & Continue" or "Use Original"
            → switch back to normal view
            → launch FinalizeThread
            → [finalizing]
                → on completed → if more files: _start_next_file()
                               → else: _on_all_done()
        → on failed → show error, stop queue
    → [finalizing]
        → on failed → show error, stop queue
```

`MainWindow` gains:
- `_pending_files: list[Path]` — remaining files to process
- `_current_transcription: TranscriptionResult | None` — held while reviewing
- `_all_results: list[PipelineResult]` — accumulated results
- `_transcription_thread: TranscriptionThread | None`
- `_finalize_thread: FinalizeThread | None`

---

## UI — Review Panel

The content area below the hero card becomes a `QStackedWidget` with two pages.

### Page 0 — Normal view
The existing horizontal splitter (files + log on the left; settings + output + run + summary on the right). Unchanged.

### Page 1 — Review mode

Shown when `TranscriptionThread` emits `completed`. Switches back to Page 0 when `FinalizeThread` starts.

Layout (single full-width card):

```
┌─────────────────────────────────────────────────────────────┐
│  Review Transcript          [filename]   File 2 of 5        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   QPlainTextEdit (editable, stretch=1)                      │
│   Pre-populated with SRT text from Whisper                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  [warning: "SRT content cannot be empty."]  (hidden by def) │
│                             [Use Original]  [Approve & Continue] │
└─────────────────────────────────────────────────────────────┘
```

- Header row: section title + filename label + queue position label
- Editor: `QPlainTextEdit`, `setReadOnly(False)`, `stretch=1` — fills all available vertical space
- Warning label: hidden by default, shown inline (above footer row) if user tries to approve empty content
- Footer buttons: "Use Original" (secondary style), "Approve & Continue" (primary/run style), right-aligned

The panel is a single resizable card with no fixed heights, so it adapts naturally to any window size.

---

## Error Handling & Edge Cases

| Scenario | Behaviour |
|---|---|
| App close during review (no thread running) | Close is allowed immediately; batch is abandoned |
| App close during transcription or finalize | `closeEvent` warns and blocks (same as today) |
| Transcription fails mid-batch | Error dialog shown, queue stops |
| User clears editor and clicks "Approve & Continue" | Inline warning shown; `FinalizeThread` not started until content is non-empty |
| "Use Original" clicked | Passes unmodified `TranscriptionResult.srt_text` to `FinalizeThread` |
| "Approve & Continue" clicked | Passes current editor text to `FinalizeThread` |

---

## Files Changed

| File | Change |
|---|---|
| `models.py` | Add `TranscriptionResult` dataclass |
| `services/subtitles.py` | Add `segments_to_srt_text()`; refactor `write_srt()` to use it |
| `services/pipeline.py` | Add `transcribe()` and `finalize()` methods; keep or deprecate `process_video()` |
| `ui/main_window.py` | Replace `BatchProcessingThread` with `TranscriptionThread` + `FinalizeThread`; add `QStackedWidget`; add review panel; add state machine |
