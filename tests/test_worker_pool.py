from unittest.mock import MagicMock, patch

from add_subtitles_to_videos.services.worker_pool import WorkerPool


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


def test_acquire_returns_slot_client_and_device():
    with patch("add_subtitles_to_videos.services.worker_pool.torch") as mock_torch, \
         patch("add_subtitles_to_videos.services.worker_pool.WhisperWorkerClient") as MockClient:
        mock_torch.cuda.is_available.return_value = False
        MockClient.return_value = MagicMock()
        pool = WorkerPool()
    result = pool.acquire()
    assert result is not None
    slot_index, client, device = result
    assert slot_index == 0
    assert device == "cpu"


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
    slot_index, _, _ = pool.acquire()
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


def test_worker_pool_cuda_prefers_gpu_slot_first():
    with patch("add_subtitles_to_videos.services.worker_pool.torch") as mock_torch, \
         patch("add_subtitles_to_videos.services.worker_pool.WhisperWorkerClient") as MockClient:
        mock_torch.cuda.is_available.return_value = True
        MockClient.return_value = MagicMock()
        pool = WorkerPool()
    slot_index, _, device = pool.acquire()
    assert slot_index == 0
    assert device == "cuda"


from pathlib import Path
from add_subtitles_to_videos.services.worker_pool import QueueDispatcher, DispatchJob
from add_subtitles_to_videos.models import ProcessingOptions, OutputMode, WorkflowProfile


def _options() -> ProcessingOptions:
    return ProcessingOptions(
        source_language=None, target_language="en", translation_provider=None,
        whisper_model="medium", output_mode=OutputMode.SRT_ONLY,
        output_directory=Path("."), max_line_length=42, subtitle_font_size=18,
    )


def _mock_pool(slots: int = 1) -> WorkerPool:
    with patch("add_subtitles_to_videos.services.worker_pool.torch") as mock_torch, \
         patch("add_subtitles_to_videos.services.worker_pool.WhisperWorkerClient") as MockClient:
        mock_torch.cuda.is_available.return_value = slots > 1
        MockClient.return_value = MagicMock()
        return WorkerPool()


def test_dispatcher_dispatch_next_returns_none_when_queue_empty():
    pool = _mock_pool(slots=1)
    dispatcher = QueueDispatcher(pool)
    assert dispatcher.dispatch_next() is None


def test_dispatcher_dispatch_next_assigns_job_to_free_worker():
    pool = _mock_pool(slots=1)
    dispatcher = QueueDispatcher(pool)
    job = DispatchJob(file_index=0, video_path=Path("a.mp4"), options=_options())
    dispatcher.enqueue(job)
    result = dispatcher.dispatch_next()
    assert result is not None
    dispatched_job, slot_index, client, device = result
    assert dispatched_job.file_index == 0
    assert slot_index == 0


def test_dispatcher_dispatch_next_returns_none_when_all_workers_busy():
    pool = _mock_pool(slots=1)
    dispatcher = QueueDispatcher(pool)
    dispatcher.enqueue(DispatchJob(file_index=0, video_path=Path("a.mp4"), options=_options()))
    dispatcher.enqueue(DispatchJob(file_index=1, video_path=Path("b.mp4"), options=_options()))
    dispatcher.dispatch_next()  # assigns job 0 to slot 0
    assert dispatcher.dispatch_next() is None  # no free workers


def test_dispatcher_release_allows_next_dispatch():
    pool = _mock_pool(slots=1)
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
    pool = _mock_pool(slots=1)
    dispatcher = QueueDispatcher(pool)
    dispatcher.enqueue(DispatchJob(file_index=0, video_path=Path("a.mp4"), options=_options()))
    dispatcher.enqueue(DispatchJob(file_index=1, video_path=Path("b.mp4"), options=_options()))
    cancelled = dispatcher.cancel_all()
    assert len(cancelled) == 2
    assert dispatcher.pending_count == 0


def test_dispatcher_cancel_job_removes_specific_file():
    pool = _mock_pool(slots=1)
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


def test_dispatcher_cancel_job_returns_false_for_nonexistent():
    pool = _mock_pool(slots=1)
    dispatcher = QueueDispatcher(pool)
    assert dispatcher.cancel_job(file_index=99) is False
