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


def write_srt(
    segments: Iterable[SubtitleSegment],
    output_path: Path,
    *,
    max_line_length: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blocks: list[str] = []
    written_segments = 0
    for segment in segments:
        text = wrap_subtitle_text(segment.text, max_line_length).strip()
        if not text:
            continue

        written_segments += 1
        start = format_srt_timestamp(segment.start_seconds)
        end = format_srt_timestamp(max(segment.end_seconds, segment.start_seconds + 0.01))
        blocks.append(f"{written_segments}\n{start} --> {end}\n{text}")

    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return written_segments
