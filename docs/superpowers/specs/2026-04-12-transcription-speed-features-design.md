# Transcription Speed & Features Design

**Date:** 2026-04-12
**Approach:** B — Parallel Workers
**Status:** Approved

## Goals

- Reduce per-file transcription latency using architectural improvements only (no accuracy trade-offs)
- Enable simultaneous transcription of multiple files (queue throughput)
- Add workflow, quality, and power-user features

## Constraints

- No accuracy degradation: no int8 quantization, no beam_size reduction, no model downgrades
- faster-whisper 1.2.1 is already installed — `BatchedInferencePipeline` is available
- Windows platform; multiprocessing uses spawn context

---

## Section 1: Core Speed Architecture

### BatchedInferencePipeline

`WhisperService` currently calls `WhisperModel.transcribe()`, which processes audio sequentially. Replace with `BatchedInferencePipeline` from faster-whisper 1.2.1. It wraps the same underlying model but splits audio into chunks and runs them through the GPU encoder in parallel, then stitches segments back together. The API is nearly identical to `WhisperModel.transcribe()` so the changeset is small.

- CUDA device: `BatchedInferencePipeline` with `compute_type="float16"`
- CPU device: `BatchedInferencePipeline` with `compute_type="float32"`
- Zero accuracy change; pure GPU utilisation improvement

### Persistent Worker Process

`whisper_worker.py` currently spawns a fresh process per job (~2–5s overhead per file). New design: one worker process per device, started at app launch, kept alive as a daemon. It listens on a command queue and handles sequential jobs without restarting. The model loads once and stays hot in VRAM/RAM for the lifetime of the app.

### WorkerPool

New class `WorkerPool` in `src/add_subtitles_to_videos/services/worker_pool.py`.

- **Worker 0**: CUDA device (if available), `BatchedInferencePipeline`, float16
- **Worker 1**: CPU device, `BatchedInferencePipeline`, float32 — overflow when Worker 0 is busy
- On CPU-only machines (no CUDA), only Worker 1 starts; max concurrency is 1 but persistent-worker and batching gains still apply
- Each worker owns its model, device, and current job state
- Workers start during the existing model preload phase at app launch

**Model reload on profile switch:** If a per-language model profile selects a different model than the worker currently has loaded, the worker reloads the model before processing that job. The existing `(model_name, device)` cache means a previously used model is a fast cache hit; a new model evicts and reloads. This reload happens transparently and is logged to the status bar.

---

## Section 2: Parallel File Processing & Queue Dispatcher

### QueueDispatcher

New class `QueueDispatcher` in `services/worker_pool.py` alongside `WorkerPool`.

Assignment priority:

1. If Worker 0 (CUDA) is free → assign there
2. Else if Worker 1 (CPU) is free → assign there
3. Else → file waits in queue until a worker finishes

Up to 2 files may be in-flight simultaneously.

### MainWindow Changes

- `MainWindow` holds a reference to `WorkerPool` (created at startup)
- One `TranscriptionThread` is spawned per in-flight file, each connected to its worker's progress signals
- Completed transcriptions are appended to the review queue as they finish (possibly out of order — arrival order = whichever worker finished first)
- State machine gains a `QUEUE_RUNNING` state distinct from `TRANSCRIBING` (single file)

### Cancellation

- Each worker supports an independent cancel signal
- **Skip file**: cancels the in-flight job for the currently selected file in the queue; dispatcher immediately assigns the next queued file to that worker
- **Stop all**: cancels both workers and clears the pending queue

---

## Section 3: UI — Parallel Progress, ETA & Queue Controls

### Per-file progress in the file list

The existing `QListWidget` is upgraded with custom item delegates rendering:

- File name
- Status chip: `Queued` / `Transcribing` / `Awaiting Review` / `Done` / `Failed`
- Progress bar (0–100%, driven by faster-whisper's segment progress callback)
- ETA — calculated as `(audio_duration_seconds × elapsed_ratio) − elapsed` once at least 10% of the audio duration has been processed; hidden until then

### Queue control toolbar

Three buttons added, active only when the queue is running:

- **Pause** — suspends dispatching new files; in-flight jobs finish normally
- **Skip** — cancels the in-flight job for the selected file; moves to next in queue
- **Stop All** — cancels all in-flight jobs and clears the pending queue

### Worker status indicator

Two-slot indicator in the status bar (below the GPU monitor):

- `GPU: transcribing "filename.mp4"` or `GPU: idle`
- `CPU: transcribing "filename.mp4"` or `CPU: idle`

### Review panel

No redesign. Files arrive in the review queue faster and potentially out of order. The existing review queue handles this naturally.

---

## Section 4: New Features

### Word-level timestamps

- Enable `word_timestamps=True` in `BatchedInferencePipeline.transcribe()`
- Each segment carries per-word start/end times
- SRT writer gains an optional `word_level` output mode: each word becomes its own subtitle entry with its exact start/end timestamp (no phrase grouping)
- Exposed as a checkbox in the output settings panel
- When disabled, behaviour is identical to today

### Confidence highlighting in the review panel

- faster-whisper returns `avg_logprob` per segment (range ~−1.0 to 0.0; higher = more confident)
- Segments with `avg_logprob < −0.6` receive a subtle amber background in the review panel editor
- Threshold is a named constant `CONFIDENCE_HIGHLIGHT_THRESHOLD = -0.6` in `config.py`
- Reviewers can edit or approve any segment regardless of confidence colour

### VAD parameter tuning

Two settings added to the advanced settings panel:

| Setting | Range | Default | Effect |
| --- | --- | --- | --- |
| Speech threshold | 0.1–0.9 | 0.5 | Silero VAD sensitivity; lower catches more speech |
| Min silence duration | 500–5000ms | 2000ms | Minimum gap before a pause is treated as silence |

### Per-language model profiles

- A `language_model_profiles` mapping added to `config.py`
- Default: `{"el": "large-v3", "en": "medium", "auto": "large-v3"}`
- When source language is explicitly selected, the dispatcher resolves the model name from the profile before assigning the job
- If the resolved model differs from what the worker has loaded, the worker reloads before starting (see Section 1 — Model reload on profile switch)
- User-editable via a table in the settings panel

### Review panel hotkeys

| Key | Action |
| --- | --- |
| `Enter` | Approve current segment, advance to next |
| `Delete` | Remove current segment |
| `Tab` / `Shift+Tab` | Next / previous segment |
| `Ctrl+Z` | Undo last change |
| `Ctrl+Enter` | Approve all and proceed to SRT generation |

---

## File Changes Summary

| File | Change |
| --- | --- |
| `services/worker_pool.py` | New — `WorkerPool` + `QueueDispatcher` |
| `services/whisper.py` | Swap `WhisperModel` → `BatchedInferencePipeline`; add word timestamps |
| `services/whisper_worker.py` | Persistent daemon mode; remove per-job spawn |
| `services/subtitles.py` | Add word-level SRT output mode |
| `ui/main_window.py` | Worker pool wiring, parallel TranscriptionThreads, new state, queue controls |
| `ui/theme.py` | Amber confidence highlight colour token |
| `config.py` | `language_model_profiles`, VAD defaults, `CONFIDENCE_HIGHLIGHT_THRESHOLD` |
| `models.py` | `word_timestamps: bool`, `vad_threshold: float`, `vad_min_silence_ms: int` on `ProcessingOptions` |

---

## Testing Notes

- Unit test `WorkerPool` assignment logic with mocked workers
- Test `QueueDispatcher` ordering: CUDA preference, fallback to CPU, queue-wait behaviour
- Test cancellation: skip-file leaves queue intact; stop-all empties queue
- Test model reload on profile switch: worker reloads when profile resolves a different model
- Test word-level SRT output produces valid per-word timing with no phrase grouping
- Test confidence threshold: segments at `avg_logprob = −0.6` (boundary) not highlighted; `−0.61` highlighted
- Regression: single-file workflow (no queue) must behave identically to today
