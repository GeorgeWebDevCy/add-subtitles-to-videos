from __future__ import annotations

from collections import deque
from collections.abc import Callable
import re
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


_DURATION_PATTERN = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")

# Still-image videos are made at this size; photos are fitted inside it with black bars.
COVER_VIDEO_WIDTH = 1920
COVER_VIDEO_HEIGHT = 1080
COVER_VIDEO_FPS = 25


def probe_duration(media_path: Path) -> float:
    """Read the media duration in seconds from FFmpeg's input summary."""
    completed = subprocess.run(
        [ffmpeg_binary(), "-hide_banner", "-i", str(media_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    match = _DURATION_PATTERN.search(completed.stderr or "")
    if match is None:
        raise MediaToolError(f"Couldn't read the duration of {media_path.name}. Is it a valid audio or video file?")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def build_prepare_command(
    input_path: Path,
    output_path: Path,
    *,
    keep: list[tuple[float, float]],
    audio_only: bool,
    cover_image: Path | None,
) -> list[str]:
    """Build an FFmpeg command that joins the kept parts of the input into one MP4.

    Each kept part is opened as its own input with -ss/-t so FFmpeg seeks straight
    to it, then the parts are joined with the concat filter. For audio input the
    picture is the cover image (or black) for the length of the edited audio.
    """
    command = [ffmpeg_binary(), "-y", "-hide_banner"]
    for start, end in keep:
        command += ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(input_path)]

    count = len(keep)
    filters: list[str] = []
    if audio_only:
        if cover_image is not None:
            command += ["-loop", "1", "-framerate", str(COVER_VIDEO_FPS), "-i", str(cover_image)]
        else:
            command += [
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={COVER_VIDEO_WIDTH}x{COVER_VIDEO_HEIGHT}:r={COVER_VIDEO_FPS}",
            ]
        audio_inputs = "".join(f"[{index}:a:0]" for index in range(count))
        filters.append(f"{audio_inputs}concat=n={count}:v=0:a=1[a]")
        filters.append(
            # out_range=tv: JPEG photos are full-range; keep output in the standard range players expect.
            f"[{count}:v]scale={COVER_VIDEO_WIDTH}:{COVER_VIDEO_HEIGHT}:force_original_aspect_ratio=decrease:out_range=tv,"
            f"pad={COVER_VIDEO_WIDTH}:{COVER_VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
            "setsar=1,format=yuv420p[v]"
        )
        tune = ["-tune", "stillimage"]
        # The picture input never ends, so stop at the edited audio length.
        length = ["-t", f"{sum(end - start for start, end in keep):.3f}"]
    else:
        pairs = "".join(f"[{index}:v:0][{index}:a:0]" for index in range(count))
        filters.append(f"{pairs}concat=n={count}:v=1:a=1[vc][a]")
        filters.append("[vc]format=yuv420p[v]")
        tune = []
        length = []

    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        *tune,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        *length,
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return command


def prepare_media(
    input_path: Path,
    output_path: Path,
    *,
    keep: list[tuple[float, float]],
    audio_only: bool,
    cover_image: Path | None = None,
    cancel_requested: CancelChecker | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        build_prepare_command(
            input_path,
            output_path,
            keep=keep,
            audio_only=audio_only,
            cover_image=cover_image,
        ),
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
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
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
