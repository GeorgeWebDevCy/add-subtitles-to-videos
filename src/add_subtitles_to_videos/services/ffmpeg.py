from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio_ffmpeg


class MediaToolError(RuntimeError):
    pass


def ffmpeg_binary() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: Path, audio_path: Path) -> None:
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
        ]
    )


def burn_subtitles(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
    *,
    font_size: int,
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
        filter_expression = (
            f"subtitles={local_subtitle_path.name}:force_style='{style}'"
        )
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
        )


def _run_ffmpeg(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return

    stderr_tail = result.stderr.strip().splitlines()[-12:]
    detail = "\n".join(stderr_tail).strip()
    raise MediaToolError(
        "FFmpeg failed while processing media.\n"
        f"Command: {' '.join(command)}\n"
        f"Details:\n{detail}"
    )
