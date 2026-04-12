# Transcription Speed Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sequential `WhisperModel` transcription with `BatchedInferencePipeline`, add a two-worker pool with a queue dispatcher, and update the UI to show per-file progress and parallel worker status.

**Architecture:** A `WorkerPool` (in `services/worker_pool.py`) manages two persistent `WhisperWorkerClient` subprocesses — one CUDA, one CPU. A `QueueDispatcher` (same file) holds pending jobs and assigns each to the next free worker. `MainWindow` uses the dispatcher instead of driving a single shared `TRANSCRIPTION_WORKER`.

**Tech Stack:** Python 3.11+, faster-whisper 1.2.1 (`BatchedInferencePipeline`), PyQt6 (`QThread`, `QListWidget`), existing `WhisperWorkerClient` / `SubtitlePipeline` infrastructure.

**Note:** Plan B (`2026-04-12-transcription-features.md`) covers Section 4 features (word timestamps, confidence highlighting, VAD tuning, profiles, hotkeys) and builds on this plan.

---

## File Map

| Action | Path | Responsibility |
| --- | --- | --- |
| Modify | `src/add_subtitles_to_videos/models.py` | Add `word_timestamps`, `vad_threshold`, `vad_min_silence_ms` to `ProcessingOptions` |
| Modify | `src/add_subtitles_to_videos/config.py` | Add `CONFIDENCE_HIGHLIGHT_THRESHOLD`, VAD defaults, `LANGUAGE_MODEL_PROFILES` |
| Modify | `src/add_subtitles_to_videos/services/whisper.py` | Swap `WhisperModel` → `BatchedInferencePipeline`; add `device_override` + `progress_callback` params |
| Modify | `src/add_subtitles_to_videos/services/whisper_worker.py` | Serialize new `ProcessingOptions` fields; route `device_override`; emit `progress` events |
| Create | `src/add_subtitles_to_videos/services/worker_pool.py` | `WorkerPool` + `QueueDispatcher` — pure Python, no Qt |
| Create | `tests/test_worker_pool.py` | Unit tests for `WorkerPool` and `QueueDispatcher` |
| Modify | `src/add_subtitles_to_videos/ui/main_window.py` | Wire `WorkerPool`; parallel `TranscriptionThread`s; per-file progress; queue controls; worker status |

---

## Task 1: Add new fields to ProcessingOptions

**Files:**
- Modify: `src/add_subtitles_to_videos/models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py  (create if missing)
from add_subtitles_to_videos.models import ProcessingOptions, OutputMode, WorkflowProfile
from pathlib import Path

def _base_options(**overrides):
    defaults = dict(
        source_language=None,
        target_language="en",
        translation_provider=None,
        whisper_model="large-v3",
        output_mode=OutputMode.SRT_ONLY,
        output_directory=Path("."),
        max_line_length=42,
        subtitle_font_size=18,
    )
    defaults.update(overrides)
    return ProcessingOptions(**defaults)

def test_processing_options_new_fields_have_defaults():
    opts = _base_options()
    assert opts.word_timestamps is False
    assert opts.vad_threshold == 0.5
    assert opts.vad_min_silence_ms == 2000

def test_processing_options_new_fields_are_settable():
    opts = _base_options(word_timestamps=True, vad_threshold=0.3, vad_min_silence_ms=1000)
    assert opts.word_timestamps is True
    assert opts.vad_threshold == 0.3
    assert opts.vad_min_silence_ms == 1000
```

- [ ] **Step 2: Run test to verify it fails**

```
cd "d:/GitHub Projects/add-subtitles-to-videos"
.venv/Scripts/python -m pytest tests/test_models.py -v
```

Expected: `TypeError: ProcessingOptions.__init__() got an unexpected keyword argument 'word_timestamps'`

- [ ] **Step 3: Add fields to ProcessingOptions**

In `src/add_subtitles_to_videos/models.py`, add three fields **after** `workflow_profile` (keep existing field order intact — `slots=True` dataclasses require defaults to follow non-defaults):

```python
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
    word_timestamps: bool = False
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 2000
```

- [ ] **Step 4: Run test to verify it passes**

```
.venv/Scripts/python -m pytest tests/test_models.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
rtk git add tests/test_models.py src/add_subtitles_to_videos/models.py
rtk git commit -m "feat: add word_timestamps, vad_threshold, vad_min_silence_ms to ProcessingOptions"
```

---

## Task 2: Add constants to config.py

**Files:**
- Modify: `src/add_subtitles_to_videos/config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py  (create if missing)
from add_subtitles_to_videos import config

def test_confidence_threshold_constant_exists():
    assert hasattr(config, "CONFIDENCE_HIGHLIGHT_THRESHOLD")
    assert isinstance(config.CONFIDENCE_HIGHLIGHT_THRESHOLD, float)
    assert config.CONFIDENCE_HIGHLIGHT_THRESHOLD == -0.6

def test_vad_defaults_exist():
    assert config.DEFAULT_VAD_THRESHOLD == 0.5
    assert config.DEFAULT_VAD_MIN_SILENCE_MS == 2000

def test_language_model_profiles_exist():
    profiles = config.LANGUAGE_MODEL_PROFILES
    assert isinstance(profiles, dict)
    assert "auto" in profiles
    assert "el" in profiles
    assert "en" in profiles
    assert all(isinstance(v, str) for v in profiles.values())
```

- [ ] **Step 2: Run to verify it fails**

```
.venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: `AttributeError: module ... has no attribute 'CONFIDENCE_HIGHLIGHT_THRESHOLD'`

- [ ] **Step 3: Add constants to config.py**

Append to the end of `src/add_subtitles_to_videos/config.py`:

```python
CONFIDENCE_HIGHLIGHT_THRESHOLD: float = -0.6

DEFAULT_VAD_THRESHOLD: float = 0.5
DEFAULT_VAD_MIN_SILENCE_MS: int = 2000

LANGUAGE_MODEL_PROFILES: dict[str, str] = {
    "auto": "large-v3",
    "el": "large-v3",
    "en": "medium",
    "de": "medium",
    "fr": "medium",
    "it": "medium",
    "es": "medium",
    "tr": "medium",
}
```

- [ ] **Step 4: Run to verify it passes**

```
.venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
rtk git add tests/test_config.py src/add_subtitles_to_videos/config.py
rtk git commit -m "feat: add confidence threshold, VAD defaults, and language model profiles to config"
```

---

## Task 3: Swap BatchedInferencePipeline in WhisperService

**Files:**
- Modify: `src/add_subtitles_to_videos/services/whisper.py`

The `_load_backend_model` method currently returns a `WhisperModel`. We wrap it in `BatchedInferencePipeline` instead. We also add `device_override: str | None` and `progress_callback: Callable[[float], None] | None` to `transcribe()` and `_transcribe_with_faster_whisper()`.

- [ ] **Step 1: Update the import block in whisper.py**

Replace:
```python
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except ImportError:  # pragma: no cover - dependency is installed in normal app environments
    FasterWhisperModel = None
```

With:
```python
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    from faster_whisper import BatchedInferencePipeline
except ImportError:  # pragma: no cover - dependency is installed in normal app environments
    FasterWhisperModel = None
    BatchedInferencePipeline = None
```

- [ ] **Step 2: Add `device_override` and `progress_callback` to `transcribe()`**

Replace the `transcribe()` signature and the `device =` line:

```python
def transcribe(
    self,
    audio_path: Path,
    options: ProcessingOptions,
    *,
    log: LogReporter | None = None,
    cancel_requested: CancelChecker | None = None,
    device_override: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[list[SubtitleSegment], TranscriptionMetadata]:
    started_at = perf_counter()
    self._check_cancel(cancel_requested)
    source_language = None
    if options.source_language and options.source_language != "auto":
        source_language = options.source_language
    device = device_override if device_override else self._preferred_device()
    backend = self._backend_name(device)
```

Also add `progress_callback=progress_callback` to the `_transcribe_with_faster_whisper` call:

```python
        if backend == "faster-whisper":
            items, metadata = self._transcribe_with_faster_whisper(
                model,
                audio_path,
                source_language=source_language,
                progress_callback=progress_callback,
            )
```

Also update `_get_model` and `_load_backend_model` to accept `device` explicitly (they already receive it via `cache_key`). The `preload_model` method also needs `device_override`:

```python
def preload_model(self, model_name: str, *, log: LogReporter | None = None, device_override: str | None = None) -> None:
    device = device_override if device_override else self._preferred_device()
    backend = self._backend_name(device)
    cache_key = (model_name, device, backend)
    load_message = "Reusing" if cache_key in self._model_cache else "Loading"
    self._emit_log(
        log,
        f"{load_message} Whisper model '{model_name}' on {self._device_label(device)} via {backend}",
    )
    self._get_model(model_name, device)
    self._emit_gpu_snapshot(log, "CUDA after warmup")
```

- [ ] **Step 3: Update `_load_backend_model` to use `BatchedInferencePipeline`**

Replace the `_load_backend_model` static method:

```python
@staticmethod
def _load_backend_model(model_name: str, device: str, backend: str) -> Any:
    if backend == "faster-whisper":
        if FasterWhisperModel is None:
            raise RuntimeError("faster-whisper is not installed.")
        base_model = FasterWhisperModel(
            model_name,
            device=device,
            compute_type=WhisperService._compute_type(device),
        )
        return BatchedInferencePipeline(model=base_model)
    return whisper.load_model(model_name, device=device)
```

- [ ] **Step 4: Update `_transcribe_with_faster_whisper` to accept and use `progress_callback`**

Replace the method:

```python
def _transcribe_with_faster_whisper(
    self,
    model: Any,
    audio_path: Path,
    *,
    source_language: str | None,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[list[SubtitleSegment], TranscriptionMetadata]:
    segments_gen, info = model.transcribe(
        str(audio_path),
        language=source_language,
        task="transcribe",
        condition_on_previous_text=False,
    )

    audio_duration: float = getattr(info, "duration", 0.0) or 0.0

    items: list[SubtitleSegment] = []
    for segment in segments_gen:
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
        if progress_callback is not None and audio_duration > 0:
            progress = min(1.0, float(getattr(segment, "end", 0.0)) / audio_duration)
            progress_callback(progress)

    metadata = TranscriptionMetadata(
        detected_language=getattr(info, "language", None),
        detected_language_probability=getattr(info, "language_probability", None),
        duration_seconds=audio_duration if audio_duration > 0 else None,
    )
    return items, metadata
```

- [ ] **Step 5: Smoke-test the change**

```
.venv/Scripts/python -c "
from add_subtitles_to_videos.services.whisper import WhisperService
s = WhisperService()
print('WhisperService import OK')
"
```

Expected: `WhisperService import OK` with no errors.

- [ ] **Step 6: Commit**

```bash
rtk git add src/add_subtitles_to_videos/services/whisper.py
rtk git commit -m "feat: swap WhisperModel for BatchedInferencePipeline with progress callback and device_override"
```

---

## Task 4: Update whisper_worker.py — device routing, new fields, progress events

**Files:**
- Modify: `src/add_subtitles_to_videos/services/whisper_worker.py`

Three changes: (1) serialize/deserialize the three new `ProcessingOptions` fields, (2) pass `device_override` from command into `service.transcribe()`, (3) emit `progress` events from `_worker_main`.

- [ ] **Step 1: Update `_serialize_options`**

Replace the `_serialize_options` function:

```python
def _serialize_options(options: ProcessingOptions) -> dict[str, object]:
    return {
        "source_language": options.source_language,
        "target_language": options.target_language,
        "translation_provider": options.translation_provider,
        "whisper_model": options.whisper_model,
        "output_mode": OutputMode(str(options.output_mode)).value,
        "output_directory": str(options.output_directory),
        "max_line_length": options.max_line_length,
        "subtitle_font_size": options.subtitle_font_size,
        "workflow_profile": WorkflowProfile(str(options.workflow_profile)).value,
        "word_timestamps": options.word_timestamps,
        "vad_threshold": options.vad_threshold,
        "vad_min_silence_ms": options.vad_min_silence_ms,
    }
```

- [ ] **Step 2: Update `_deserialize_options`**

Replace the `_deserialize_options` function:

```python
def _deserialize_options(payload: dict[str, object]) -> ProcessingOptions:
    return ProcessingOptions(
        source_language=payload["source_language"],  # type: ignore[arg-type]
        target_language=str(payload["target_language"]),
        translation_provider=payload["translation_provider"],  # type: ignore[arg-type]
        whisper_model=str(payload["whisper_model"]),
        output_mode=OutputMode(str(payload["output_mode"])),
        output_directory=Path(str(payload["output_directory"])),
        max_line_length=int(payload["max_line_length"]),
        subtitle_font_size=int(payload["subtitle_font_size"]),
        workflow_profile=WorkflowProfile(str(payload["workflow_profile"])),
        word_timestamps=bool(payload.get("word_timestamps", False)),
        vad_threshold=float(payload.get("vad_threshold", 0.5)),
        vad_min_silence_ms=int(payload.get("vad_min_silence_ms", 2000)),
    )
```

- [ ] **Step 3: Update `_worker_main` to route device and emit progress**

Replace the `transcribe` block inside `_worker_main` (the `if command_type not in {"transcribe", "preload"}` block and below):

```python
def _worker_main(command_queue, event_queue) -> None:
    service = WhisperService()

    while True:
        command = command_queue.get()
        command_type = command.get("type")
        if command_type == "shutdown":
            return
        if command_type not in {"transcribe", "preload"}:
            continue

        job_id = int(command["job_id"])
        device_override: str | None = command.get("device_override")  # type: ignore[assignment]

        def on_log(message: str) -> None:
            event_queue.put({"type": "log", "job_id": job_id, "message": message})

        def on_progress(progress: float) -> None:
            event_queue.put({"type": "progress", "job_id": job_id, "progress": progress})

        try:
            if command_type == "preload":
                service.preload_model(
                    str(command["model_name"]),
                    log=on_log,
                    device_override=device_override,
                )
                event_queue.put({"type": "preload_complete", "job_id": job_id})
                continue

            options = _deserialize_options(command["options"])
            audio_path = Path(str(command["audio_path"]))
            segments, metadata = service.transcribe(
                audio_path,
                options,
                log=on_log,
                device_override=device_override,
                progress_callback=on_progress,
            )
            event_queue.put(
                {
                    "type": "result",
                    "job_id": job_id,
                    "segments": _serialize_segments(segments),
                    "metadata": _serialize_metadata(metadata),
                }
            )
        except Exception as exc:
            event_queue.put({"type": "error", "job_id": job_id, "message": str(exc)})
```

- [ ] **Step 4: Add `device_override` and `progress_callback` params to `WhisperWorkerClient.transcribe()`**

Add the two new parameters and handle the `progress` event type. Replace the `transcribe` method signature and event loop:

```python
def transcribe(
    self,
    audio_path: Path,
    options: ProcessingOptions,
    *,
    log: LogReporter | None = None,
    cancel_requested: CancelChecker | None = None,
    device_override: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[list[SubtitleSegment], TranscriptionMetadata]:
    with self._lock:
        self._ensure_process()
        assert self._command_queue is not None
        assert self._event_queue is not None

        self._job_counter += 1
        job_id = self._job_counter
        cancel_deadline: float | None = None

        self._command_queue.put(
            {
                "type": "transcribe",
                "job_id": job_id,
                "audio_path": str(audio_path),
                "options": _serialize_options(options),
                "device_override": device_override,
            }
        )

        while True:
            if cancel_requested is not None and cancel_requested() and cancel_deadline is None:
                cancel_deadline = monotonic() + _CANCEL_GRACE_SECONDS
                if log is not None:
                    log(
                        "Stop requested. Waiting briefly for Whisper to finish before cancelling the worker."
                    )

            if cancel_deadline is not None and monotonic() >= cancel_deadline:
                self._terminate_process()
                raise OperationCancelledError("Processing stopped by user.")

            if self._process is None or not self._process.is_alive():
                self._cleanup_process()
                raise RuntimeError("Whisper worker stopped unexpectedly.")

            try:
                event = self._event_queue.get(timeout=0.1)
            except Empty:
                continue

            if int(event.get("job_id", -1)) != job_id:
                continue

            event_type = event.get("type")
            if event_type == "log":
                if log is not None:
                    log(str(event["message"]))
                continue

            if event_type == "progress":
                if progress_callback is not None:
                    progress_callback(float(event["progress"]))
                continue

            if event_type == "result":
                return (
                    _deserialize_segments(event["segments"]),
                    _deserialize_metadata(event["metadata"]),
                )

            if event_type == "error":
                raise RuntimeError(str(event["message"]))
```

Also add `device_override` to `preload_model()` command dict:

```python
self._command_queue.put(
    {
        "type": "preload",
        "job_id": job_id,
        "model_name": model_name,
        "device_override": device_override,  # add this line
    }
)
```

And add `device_override: str | None = None` to `preload_model()`'s signature.

- [ ] **Step 5: Smoke-test**

```
.venv/Scripts/python -c "
from add_subtitles_to_videos.services.whisper_worker import WhisperWorkerClient, _serialize_options, _deserialize_options
from add_subtitles_to_videos.models import ProcessingOptions, OutputMode, WorkflowProfile
from pathlib import Path
opts = ProcessingOptions(
    source_language=None, target_language='en', translation_provider=None,
    whisper_model='medium', output_mode=OutputMode.SRT_ONLY,
    output_directory=Path('.'), max_line_length=42, subtitle_font_size=18,
    word_timestamps=True, vad_threshold=0.3, vad_min_silence_ms=1000,
)
serialized = _serialize_options(opts)
restored = _deserialize_options(serialized)
assert restored.word_timestamps is True
assert restored.vad_threshold == 0.3
assert restored.vad_min_silence_ms == 1000
print('serialize round-trip OK')
"
```

Expected: `serialize round-trip OK`

- [ ] **Step 6: Commit**

```bash
rtk git add src/add_subtitles_to_videos/services/whisper_worker.py
rtk git commit -m "feat: route device_override and emit progress events from whisper worker"
```

---

## Task 5: Create WorkerPool

**Files:**
- Create: `src/add_subtitles_to_videos/services/worker_pool.py`
- Create: `tests/test_worker_pool.py`

`WorkerPool` manages 1 or 2 `WhisperWorkerClient` instances. On CUDA machines it creates a CUDA worker (slot 0) and a CPU worker (slot 1). On CPU-only machines it creates one CPU worker (slot 0).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_worker_pool.py
from unittest.mock import MagicMock, patch
from add_subtitles_to_videos.services.worker_pool import WorkerPool


def _mock_pool(cuda_available: bool) -> WorkerPool:
    with patch("add_subtitles_to_videos.services.worker_pool.torch") as mock_torch, \
         patch("add_subtitles_to_videos.services.worker_pool.WhisperWorkerClient") as MockClient:
        mock_torch.cuda.is_available.return_value = cuda_available
        MockClient.return_value = MagicMock()
        pool = WorkerPool()
    return pool


def test_worker_pool_cuda_creates_two_slots():
    with patch("add_subtitles_to_videos.services.worker_pool.torch") as mock_torch, \
         patch("add_subtitles_to_videos.services.worker_pool.WhisperWorkerClient") as MockClient:
        mock_torch.cuda.is_available.return_value = True
        MockClient.return_value = MagicMock()
        pool = WorkerPool()
    assert pool.slot_count == 2


def test_worker_pool_cpu_only_creates_one_slot():
    with patch("add_subtitles_to_videos.services.worker_pool.torch") as mock_torch, \
         patch("add_subtitles_to_videos.services.worker_pool.WhisperWorkerClient") as MockClient:
        mock_torch.cuda.is_available.return_value = False
        MockClient.return_value = MagicMock()
        pool = WorkerPool()
    assert pool.slot_count == 1


def test_acquire_returns_slot_and_client():
    with patch("add_subtitles_to_videos.services.worker_pool.torch") as mock_torch, \
         patch("add_subtitles_to_videos.services.worker_pool.WhisperWorkerClient") as MockClient:
        mock_torch.cuda.is_available.return_value = False
        MockClient.return_value = MagicMock()
        pool = WorkerPool()
    result = pool.acquire()
    assert result is not None
    slot_index, client = result
    assert slot_index == 0


def test_acquire_returns_none_when_all_busy():
    with patch("add_subtitles_to_videos.services.worker_pool.torch") as mock_torch, \
         patch("add_subtitles_to_videos.services.worker_pool.WhisperWorkerClient") as MockClient:
        mock_torch.cuda.is_available.return_value = False
        MockClient.return_value = MagicMock()
        pool = WorkerPool()
    pool.acquire()
    assert pool.acquire() is None


def test_release_makes_slot_available_again():
    with patch("add_subtitles_to_videos.services.worker_pool.torch") as mock_torch, \
         patch("add_subtitles_to_videos.services.worker_pool.WhisperWorkerClient") as MockClient:
        mock_torch.cuda.is_available.return_value = False
        MockClient.return_value = MagicMock()
        pool = WorkerPool()
    slot_index, _ = pool.acquire()
    pool.release(slot_index)
    assert pool.acquire() is not None


def test_worker_device_label_cuda_slot():
    with patch("add_subtitles_to_videos.services.worker_pool.torch") as mock_torch, \
         patch("add_subtitles_to_videos.services.worker_pool.WhisperWorkerClient") as MockClient:
        mock_torch.cuda.is_available.return_value = True
        MockClient.return_value = MagicMock()
        pool = WorkerPool()
    assert pool.worker_device_label(0) == "GPU"
    assert pool.worker_device_label(1) == "CPU"
```

- [ ] **Step 2: Run to verify they fail**

```
.venv/Scripts/python -m pytest tests/test_worker_pool.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `worker_pool` doesn't exist yet.

- [ ] **Step 3: Create worker_pool.py with WorkerPool**

Create `src/add_subtitles_to_videos/services/worker_pool.py`:

```python
from __future__ import annotations

import torch

from .whisper_worker import WhisperWorkerClient


class WorkerPool:
    """Manages 1–2 persistent WhisperWorkerClient instances, one per device.

    On CUDA machines: slot 0 = CUDA, slot 1 = CPU (overflow).
    On CPU-only machines: slot 0 = CPU only.
    """

    def __init__(self) -> None:
        if torch.cuda.is_available():
            self._slots: list[tuple[WhisperWorkerClient, str, bool]] = [
                (WhisperWorkerClient(), "cuda", False),  # (client, device, is_busy)
                (WhisperWorkerClient(), "cpu", False),
            ]
        else:
            self._slots = [
                (WhisperWorkerClient(), "cpu", False),
            ]

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    def acquire(self) -> tuple[int, WhisperWorkerClient, str] | None:
        """Return (slot_index, client, device) for the first free slot. None if all busy."""
        for i, (client, device, busy) in enumerate(self._slots):
            if not busy:
                self._slots[i] = (client, device, True)
                return i, client, device
        return None

    def release(self, slot_index: int) -> None:
        """Mark slot as free."""
        client, device, _ = self._slots[slot_index]
        self._slots[slot_index] = (client, device, False)

    def worker_device_label(self, slot_index: int) -> str:
        """Human-readable label for status display: 'GPU' or 'CPU'."""
        _, device, _ = self._slots[slot_index]
        return "GPU" if device == "cuda" else "CPU"

    def close(self) -> None:
        for client, _, _ in self._slots:
            client.close()
```

- [ ] **Step 4: Fix tests to match updated `acquire()` return type**

The tests use `slot_index, client = result` but `acquire()` now returns 3-tuple. Update tests:

```python
def test_acquire_returns_slot_and_client():
    ...
    result = pool.acquire()
    assert result is not None
    slot_index, client, device = result
    assert slot_index == 0

def test_acquire_returns_none_when_all_busy():
    ...
    pool.acquire()
    assert pool.acquire() is None

def test_release_makes_slot_available_again():
    ...
    slot_index, _, _ = pool.acquire()
    pool.release(slot_index)
    assert pool.acquire() is not None
```

- [ ] **Step 5: Run tests**

```
.venv/Scripts/python -m pytest tests/test_worker_pool.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 6: Commit**

```bash
rtk git add src/add_subtitles_to_videos/services/worker_pool.py tests/test_worker_pool.py
rtk git commit -m "feat: add WorkerPool managing CUDA and CPU worker clients"
```

---

## Task 6: Create QueueDispatcher

**Files:**
- Modify: `src/add_subtitles_to_videos/services/worker_pool.py`
- Modify: `tests/test_worker_pool.py`

`QueueDispatcher` is a pure-Python state machine. It holds pending jobs and, when asked, assigns the next job to the next free worker.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_worker_pool.py`:

```python
from pathlib import Path
from add_subtitles_to_videos.services.worker_pool import QueueDispatcher, DispatchJob
from add_subtitles_to_videos.models import ProcessingOptions, OutputMode, WorkflowProfile


def _options() -> ProcessingOptions:
    return ProcessingOptions(
        source_language=None, target_language="en", translation_provider=None,
        whisper_model="medium", output_mode=OutputMode.SRT_ONLY,
        output_directory=Path("."), max_line_length=42, subtitle_font_size=18,
    )


def _mock_worker_pool(slots: int = 2) -> WorkerPool:
    with patch("add_subtitles_to_videos.services.worker_pool.torch") as mock_torch, \
         patch("add_subtitles_to_videos.services.worker_pool.WhisperWorkerClient") as MockClient:
        mock_torch.cuda.is_available.return_value = slots > 1
        MockClient.return_value = MagicMock()
        return WorkerPool()


def test_dispatcher_dispatch_next_returns_none_when_queue_empty():
    pool = _mock_worker_pool(slots=1)
    dispatcher = QueueDispatcher(pool)
    assert dispatcher.dispatch_next() is None


def test_dispatcher_dispatch_next_assigns_job_to_free_worker():
    pool = _mock_worker_pool(slots=1)
    dispatcher = QueueDispatcher(pool)
    job = DispatchJob(file_index=0, video_path=Path("a.mp4"), options=_options())
    dispatcher.enqueue(job)
    result = dispatcher.dispatch_next()
    assert result is not None
    dispatched_job, slot_index, client, device = result
    assert dispatched_job.file_index == 0
    assert slot_index == 0


def test_dispatcher_dispatch_next_returns_none_when_all_workers_busy():
    pool = _mock_worker_pool(slots=1)
    dispatcher = QueueDispatcher(pool)
    dispatcher.enqueue(DispatchJob(file_index=0, video_path=Path("a.mp4"), options=_options()))
    dispatcher.enqueue(DispatchJob(file_index=1, video_path=Path("b.mp4"), options=_options()))
    dispatcher.dispatch_next()  # assigns job 0 to slot 0
    assert dispatcher.dispatch_next() is None  # no free workers


def test_dispatcher_release_allows_next_dispatch():
    pool = _mock_worker_pool(slots=1)
    dispatcher = QueueDispatcher(pool)
    dispatcher.enqueue(DispatchJob(file_index=0, video_path=Path("a.mp4"), options=_options()))
    dispatcher.enqueue(DispatchJob(file_index=1, video_path=Path("b.mp4"), options=_options()))
    _, slot_index, _, _ = dispatcher.dispatch_next()
    dispatcher.release(slot_index)
    result = dispatcher.dispatch_next()
    assert result is not None
    job, _, _, _ = result
    assert job.file_index == 1


def test_dispatcher_cancel_all_clears_queue():
    pool = _mock_worker_pool(slots=1)
    dispatcher = QueueDispatcher(pool)
    dispatcher.enqueue(DispatchJob(file_index=0, video_path=Path("a.mp4"), options=_options()))
    dispatcher.enqueue(DispatchJob(file_index=1, video_path=Path("b.mp4"), options=_options()))
    cancelled = dispatcher.cancel_all()
    assert len(cancelled) == 2
    assert dispatcher.pending_count == 0


def test_dispatcher_cancel_job_removes_specific_file():
    pool = _mock_worker_pool(slots=1)
    dispatcher = QueueDispatcher(pool)
    dispatcher.enqueue(DispatchJob(file_index=0, video_path=Path("a.mp4"), options=_options()))
    dispatcher.enqueue(DispatchJob(file_index=1, video_path=Path("b.mp4"), options=_options()))
    removed = dispatcher.cancel_job(file_index=0)
    assert removed is True
    assert dispatcher.pending_count == 1
    result = dispatcher.dispatch_next()
    assert result is not None
    job, _, _, _ = result
    assert job.file_index == 1
```

- [ ] **Step 2: Run to verify they fail**

```
.venv/Scripts/python -m pytest tests/test_worker_pool.py::test_dispatcher_dispatch_next_returns_none_when_queue_empty -v
```

Expected: `ImportError: cannot import name 'QueueDispatcher'`

- [ ] **Step 3: Add DispatchJob and QueueDispatcher to worker_pool.py**

Append to `src/add_subtitles_to_videos/services/worker_pool.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import ProcessingOptions


@dataclass
class DispatchJob:
    file_index: int
    video_path: Path
    options: ProcessingOptions


class QueueDispatcher:
    """Pure-Python queue state machine. No Qt dependencies.

    Usage:
        dispatcher.enqueue(job)
        result = dispatcher.dispatch_next()  # call after enqueue or after release
        if result:
            job, slot_index, client, device = result
            # start TranscriptionThread(client, device_override=device, ...)
        # When thread finishes:
        dispatcher.release(slot_index)
        dispatcher.dispatch_next()  # pick up next pending job
    """

    def __init__(self, pool: WorkerPool) -> None:
        self._pool = pool
        self._queue: list[DispatchJob] = []

    def enqueue(self, job: DispatchJob) -> None:
        self._queue.append(job)

    def dispatch_next(self) -> tuple[DispatchJob, int, WhisperWorkerClient, str] | None:
        """Assign next pending job to a free worker. Returns None if queue empty or all workers busy."""
        if not self._queue:
            return None
        slot = self._pool.acquire()
        if slot is None:
            return None
        slot_index, client, device = slot
        job = self._queue.pop(0)
        return job, slot_index, client, device

    def release(self, slot_index: int) -> None:
        self._pool.release(slot_index)

    def cancel_all(self) -> list[DispatchJob]:
        jobs = list(self._queue)
        self._queue.clear()
        return jobs

    def cancel_job(self, file_index: int) -> bool:
        for i, job in enumerate(self._queue):
            if job.file_index == file_index:
                del self._queue[i]
                return True
        return False

    @property
    def pending_count(self) -> int:
        return len(self._queue)
```

Note: the `from __future__ import annotations` and `dataclass` import are already needed — add them to the top of the file alongside the existing imports.

- [ ] **Step 4: Run all worker_pool tests**

```
.venv/Scripts/python -m pytest tests/test_worker_pool.py -v
```

Expected: all 13 tests PASS

- [ ] **Step 5: Commit**

```bash
rtk git add src/add_subtitles_to_videos/services/worker_pool.py tests/test_worker_pool.py
rtk git commit -m "feat: add QueueDispatcher and DispatchJob to worker_pool"
```

---

## Task 7: Wire WorkerPool into MainWindow

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`

Replace the module-level `TRANSCRIPTION_WORKER = WhisperWorkerClient()` singleton with a `WorkerPool`. Update `TranscriptionThread` to accept a specific client and `device_override`. Update `ModelPreloadThread` to use the pool's first slot. Wire dispatch into `_start_processing` and `_on_transcription_completed`.

- [ ] **Step 1: Update imports in main_window.py**

Find:
```python
from ..services.whisper_worker import WhisperWorkerClient

TRANSCRIPTION_WORKER = WhisperWorkerClient()
```

Replace with:
```python
from ..services.worker_pool import WorkerPool, QueueDispatcher, DispatchJob

_WORKER_POOL = WorkerPool()
_QUEUE_DISPATCHER = QueueDispatcher(_WORKER_POOL)
```

- [ ] **Step 2: Update TranscriptionThread to accept client + device_override**

The `run()` method currently passes `TRANSCRIPTION_WORKER` to `SubtitlePipeline`. Update `__init__` and `run()`:

```python
class TranscriptionThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        options: ProcessingOptions,
        translation_service: OpenAICompatibleTranslationService | None,
        file_index: int,
        total_files: int,
        whisper_worker,   # WhisperWorkerClient — typed as Any to avoid import cycle risk
        device_override: str | None = None,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._options = options
        self._translation_service = translation_service
        self._file_index = file_index
        self._total_files = total_files
        self._whisper_worker = whisper_worker
        self._device_override = device_override

    def run(self) -> None:
        try:
            pipeline = SubtitlePipeline(
                whisper_service=self._whisper_worker,
            )

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
                cancel_requested=self.isInterruptionRequested,
            )
            self.completed.emit(result)
        except OperationCancelledError as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))

    def request_stop(self) -> None:
        self.requestInterruption()
```

- [ ] **Step 3: Update ModelPreloadThread to use pool's first client**

Replace:
```python
    def run(self) -> None:
        try:
            TRANSCRIPTION_WORKER.preload_model(self._model_name, log=self.log_message.emit)
            self.completed.emit(self._model_name)
        except Exception as exc:
            self.failed.emit(str(exc))
```

With:
```python
    def run(self) -> None:
        try:
            slot = _WORKER_POOL.acquire()
            if slot is None:
                self.failed.emit("No free worker available for preload.")
                return
            slot_index, client, device = slot
            try:
                client.preload_model(
                    self._model_name,
                    log=self.log_message.emit,
                    device_override=device,
                )
            finally:
                _WORKER_POOL.release(slot_index)
            self.completed.emit(self._model_name)
        except Exception as exc:
            self.failed.emit(str(exc))
```

- [ ] **Step 4: Update MainWindow.__init__ to store active threads by slot**

In `MainWindow.__init__`, replace:
```python
self._transcription_thread: TranscriptionThread | None = None
```

With:
```python
self._transcription_threads: dict[int, TranscriptionThread] = {}  # slot_index → thread
```

Update all references to `self._transcription_thread` throughout the class:
- `self._transcription_thread is not None` → `bool(self._transcription_threads)`
- `self._transcription_thread` → look it up by the relevant slot index

This requires reading the full usage context. Run this grep first to find all occurrences:

```bash
grep -n "_transcription_thread" src/add_subtitles_to_videos/ui/main_window.py
```

Update each occurrence appropriately. For single-thread operations (like `_request_stop`), iterate all threads:

```python
def _request_stop(self) -> None:
    for thread in self._transcription_threads.values():
        thread.request_stop()
```

- [ ] **Step 5: Update `_start_transcription` to use dispatcher**

Find `_start_transcription` (line ~907). Update it to accept a pre-acquired slot:

```python
def _start_transcription(
    self,
    file_index: int,
    *,
    worker_slot: int,
    worker_client,
    worker_device: str,
    as_prefetch: bool = False,
) -> None:
    ...
    thread = TranscriptionThread(
        video_path,
        options,
        translation_service,
        file_index,
        len(self._selected_files),
        whisper_worker=worker_client,
        device_override=worker_device,
    )
    self._transcription_threads[worker_slot] = thread
    thread.completed.connect(lambda result, idx=file_index, slot=worker_slot: self._on_transcription_completed(result, idx, slot))
    thread.failed.connect(lambda msg, slot=worker_slot: self._on_transcription_failed(msg, slot))
    thread.cancelled.connect(lambda msg, slot=worker_slot: self._on_transcription_cancelled(msg, slot))
    thread.progress_changed.connect(self._on_progress)
    thread.log_message.connect(self._on_log_message)
    thread.start()
```

- [ ] **Step 6: Update `_start_processing` to use dispatcher**

Find `_start_processing` (line ~844). Replace the single `_start_transcription` call with dispatcher logic:

```python
def _start_processing(self) -> None:
    ...
    # After validating inputs and building options, enqueue all files:
    _QUEUE_DISPATCHER.cancel_all()  # clear any stale queue from previous run
    for i in range(len(self._selected_files)):
        _QUEUE_DISPATCHER.enqueue(DispatchJob(
            file_index=i,
            video_path=self._selected_files[i],
            options=options,
        ))
    self._dispatch_pending()

def _dispatch_pending(self) -> None:
    while True:
        result = _QUEUE_DISPATCHER.dispatch_next()
        if result is None:
            break
        job, slot_index, client, device = result
        self._start_transcription(
            job.file_index,
            worker_slot=slot_index,
            worker_client=client,
            worker_device=device,
        )
```

- [ ] **Step 7: Update `_on_transcription_completed` to release slot and dispatch next**

Add `slot_index: int` parameter and release after completion:

```python
def _on_transcription_completed(self, result: TranscriptionResult, file_index: int, slot_index: int) -> None:
    self._transcription_threads.pop(slot_index, None)
    _QUEUE_DISPATCHER.release(slot_index)
    self._dispatch_pending()
    # ... rest of existing completion logic
```

- [ ] **Step 8: Run existing test suite to verify no regressions**

```
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all previously passing tests still pass. Fix any failures before committing.

- [ ] **Step 9: Commit**

```bash
rtk git add src/add_subtitles_to_videos/ui/main_window.py
rtk git commit -m "feat: wire WorkerPool and QueueDispatcher into MainWindow for parallel transcription"
```

---

## Task 8: Per-file progress in the file list

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`

`TranscriptionThread.progress_changed` currently emits `(int, str)` — overall queue %. We also need per-file Whisper progress (0.0–1.0) for the file list progress bars. Add a second signal and wire it up.

- [ ] **Step 1: Add `file_progress_changed` signal to TranscriptionThread**

In `TranscriptionThread`:

```python
file_progress_changed = Signal(int, float)  # file_index, progress 0.0–1.0
```

In `run()`, the `progress_callback` passed to `pipeline.transcribe` receives Whisper's per-segment progress. But `pipeline.transcribe` calls `WhisperWorkerClient.transcribe` internally. We need to surface this.

Update `SubtitlePipeline.transcribe()` to accept and forward a `whisper_progress_callback`. Check `services/pipeline.py` for the current signature and add the forwarding.

In `TranscriptionThread.run()`:

```python
def on_whisper_progress(progress: float) -> None:
    self.file_progress_changed.emit(self._file_index, progress)

result = pipeline.transcribe(
    self._video_path,
    self._options,
    progress=on_progress,
    log=on_log,
    cancel_requested=self.isInterruptionRequested,
    whisper_progress_callback=on_whisper_progress,
)
```

- [ ] **Step 2: Check and update SubtitlePipeline.transcribe() signature**

Read `src/add_subtitles_to_videos/services/pipeline.py` to find where `whisper_service.transcribe()` is called. Add `whisper_progress_callback: Callable[[float], None] | None = None` to `pipeline.transcribe()` and forward it to `whisper_service.transcribe(progress_callback=whisper_progress_callback)`.

- [ ] **Step 3: Add per-file status tracking to MainWindow**

In `MainWindow.__init__`:

```python
self._file_status: dict[int, str] = {}   # file_index → status label
self._file_progress: dict[int, float] = {}  # file_index → 0.0–1.0
```

- [ ] **Step 4: Connect file_progress_changed and update file list display**

In `_start_transcription`, connect the new signal:

```python
thread.file_progress_changed.connect(self._on_file_progress)
```

Add handler:

```python
def _on_file_progress(self, file_index: int, progress: float) -> None:
    self._file_progress[file_index] = progress
    self._file_status[file_index] = "Transcribing"
    self._refresh_file_list()
```

- [ ] **Step 5: Update `_refresh_file_list` to show status and progress**

Find the method that populates the file list widget (search for `_create_files_card` or `_refresh_file_list` in main_window.py). Update each item to show:

```python
def _refresh_file_list(self) -> None:
    self._files_list.clear()
    for i, path in enumerate(self._selected_files):
        status = self._file_status.get(i, "Queued")
        progress = self._file_progress.get(i, 0.0)
        pct = int(progress * 100)
        label = f"{path.name}  [{status}]" + (f"  {pct}%" if status == "Transcribing" else "")
        item = QListWidgetItem(label)
        self._files_list.addItem(item)
```

(A full custom delegate with embedded `QProgressBar` is a follow-up UX improvement; this text-based approach ships the behaviour first.)

- [ ] **Step 6: Commit**

```bash
rtk git add src/add_subtitles_to_videos/services/pipeline.py src/add_subtitles_to_videos/ui/main_window.py
rtk git commit -m "feat: add per-file transcription progress display in file list"
```

---

## Task 9: Queue controls toolbar and worker status indicator

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`

- [ ] **Step 1: Add Pause / Skip / Stop All buttons**

In `_create_run_card()` (or the controls area near the Start button), add three buttons, initially disabled:

```python
self._pause_btn = QPushButton("Pause")
self._skip_btn = QPushButton("Skip File")
self._stop_all_btn = QPushButton("Stop All")
self._pause_btn.setEnabled(False)
self._skip_btn.setEnabled(False)
self._stop_all_btn.setEnabled(False)
self._pause_btn.clicked.connect(self._on_pause_clicked)
self._skip_btn.clicked.connect(self._on_skip_clicked)
self._stop_all_btn.clicked.connect(self._on_stop_all_clicked)
```

Add them to the run card layout alongside the existing Start/Stop button.

- [ ] **Step 2: Implement Pause handler**

```python
def _on_pause_clicked(self) -> None:
    if not hasattr(self, "_queue_paused"):
        self._queue_paused = False
    self._queue_paused = not self._queue_paused
    self._pause_btn.setText("Resume" if self._queue_paused else "Pause")
```

Update `_dispatch_pending` to respect the flag:

```python
def _dispatch_pending(self) -> None:
    if getattr(self, "_queue_paused", False):
        return
    while True:
        result = _QUEUE_DISPATCHER.dispatch_next()
        if result is None:
            break
        ...
```

- [ ] **Step 3: Implement Skip handler**

```python
def _on_skip_clicked(self) -> None:
    # Cancel the first active thread (CUDA worker priority)
    for slot_index, thread in list(self._transcription_threads.items()):
        if _WORKER_POOL.worker_device_label(slot_index) == "GPU":
            thread.request_stop()
            return
    # Fallback: cancel any active thread
    for thread in self._transcription_threads.values():
        thread.request_stop()
        return
```

- [ ] **Step 4: Implement Stop All handler**

```python
def _on_stop_all_clicked(self) -> None:
    _QUEUE_DISPATCHER.cancel_all()
    for thread in self._transcription_threads.values():
        thread.request_stop()
    self._queue_paused = False
    self._pause_btn.setText("Pause")
```

- [ ] **Step 5: Enable/disable queue control buttons with queue state**

Add a helper called in `_start_processing`, `_dispatch_pending`, and `_on_transcription_completed`:

```python
def _refresh_queue_controls(self) -> None:
    running = bool(self._transcription_threads) or _QUEUE_DISPATCHER.pending_count > 0
    self._pause_btn.setEnabled(running)
    self._skip_btn.setEnabled(bool(self._transcription_threads))
    self._stop_all_btn.setEnabled(running)
```

- [ ] **Step 6: Add worker status labels to the status bar**

In `_build_ui()` or wherever the status bar is configured, add two labels:

```python
self._worker_status_labels: dict[int, QLabel] = {}
for i in range(_WORKER_POOL.slot_count):
    label = QLabel(f"{_WORKER_POOL.worker_device_label(i)}: idle")
    label.setObjectName("workerStatus")
    self.statusBar().addPermanentWidget(label)
    self._worker_status_labels[i] = label
```

Update worker status in `_start_transcription` and `_on_transcription_completed`:

```python
# On start:
self._worker_status_labels[worker_slot].setText(
    f"{_WORKER_POOL.worker_device_label(worker_slot)}: transcribing \"{video_path.name}\""
)

# On completion:
self._worker_status_labels[slot_index].setText(
    f"{_WORKER_POOL.worker_device_label(slot_index)}: idle"
)
```

- [ ] **Step 7: Launch app and verify manually**

```
cd "d:/GitHub Projects/add-subtitles-to-videos" && start pythonw -m add_subtitles_to_videos
```

- Add 2+ video files
- Click Start
- Verify: status bar shows worker status, file list shows `[Transcribing] 42%`, Pause/Skip/Stop All buttons are enabled
- Verify Skip cancels one file and moves to next
- Verify Stop All clears the queue

- [ ] **Step 8: Commit**

```bash
rtk git add src/add_subtitles_to_videos/ui/main_window.py
rtk git commit -m "feat: add Pause/Skip/Stop All queue controls and worker status indicator"
```

---

## Task 10: Run full test suite and tag

- [ ] **Step 1: Run all tests**

```
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all tests PASS. Fix any failures.

- [ ] **Step 2: Final commit**

```bash
rtk git add -A
rtk git commit -m "feat: complete transcription speed core — BatchedInferencePipeline, WorkerPool, parallel queue"
```

---

## Regression Checklist

Before marking this plan complete, verify manually:

- [ ] Single file without a queue: works identically to pre-change behaviour
- [ ] CUDA GPU is used (check log: `Loading Whisper model 'large-v3' on CUDA GPU ... via faster-whisper`)
- [ ] No CMD window flash on Start (from the earlier fix)
- [ ] Model preload on model selector change still works
- [ ] Cancel mid-transcription works; worker recovers for next job
- [ ] Review panel receives completed transcription and allows editing
- [ ] SRT and burned video export still work after the pipeline changes
