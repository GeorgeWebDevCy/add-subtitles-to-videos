from __future__ import annotations

from collections import deque
from collections.abc import Callable
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import sleep

import imageio_ffmpeg

from . import OperationCancelledError

CancelChecker = Callable[[], bool]


class MediaToolError(RuntimeError):
    pass


def ffmpeg_binary() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(
    video_path: Path,
    audio_path: Path,
    *,
    cancel_requested: CancelChecker | None = None,
) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            ffmpeg_binary(),
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        cancel_requested=cancel_requested,
    )


def burn_subtitles(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
    *,
    font_size: int,
    cancel_requested: CancelChecker | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    style = (
        f"FontSize={font_size},OutlineColour=&H4A000000,"
        "BorderStyle=3,Outline=1,Shadow=0,MarginV=28"
    )

    with TemporaryDirectory(prefix="subtitle-foundry-burn-") as temp_dir:
        temp_path = Path(temp_dir)
        local_subtitle_path = temp_path / "captions.srt"
        shutil.copy2(subtitle_path, local_subtitle_path)
        filter_expression = f"subtitles={local_subtitle_path.name}:force_style='{style}'"
        _run_ffmpeg(
            [
                ffmpeg_binary(),
                "-y",
                "-i",
                str(video_path),
                "-vf",
                filter_expression,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            cwd=temp_path,
            cancel_requested=cancel_requested,
        )


def _run_ffmpeg(
    command: list[str],
    cwd: Path | None = None,
    *,
    cancel_requested: CancelChecker | None = None,
) -> None:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_tail: deque[str] = deque(maxlen=24)
    stderr_thread = _start_stderr_reader(process, stderr_tail)

    try:
        while True:
            if cancel_requested is not None and cancel_requested():
                _terminate_process(process)
                raise OperationCancelledError("Processing stopped by user.")

            result_code = process.poll()
            if result_code is None:
                sleep(0.1)
                continue
            break
    finally:
        stderr_thread.join(timeout=2)

    if result_code == 0:
        return

    detail = "\n".join(stderr_tail).strip()
    raise MediaToolError(
        "FFmpeg failed while processing media.\n"
        f"Command: {' '.join(command)}\n"
        f"Details:\n{detail}"
    )


def _start_stderr_reader(
    process: subprocess.Popen[str],
    stderr_tail: deque[str],
) -> Thread:
    def drain() -> None:
        stream = process.stderr
        if stream is None:
            return

        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                stripped = line.rstrip()
                if stripped:
                    stderr_tail.append(stripped)
        finally:
            stream.close()

    thread = Thread(target=drain, daemon=True)
    thread.start()
    return thread


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
