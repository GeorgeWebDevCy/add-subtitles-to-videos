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
