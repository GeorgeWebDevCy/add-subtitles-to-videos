from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re
from textwrap import wrap

from ..models import SubtitleSegment, TranslationSegment

_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2}),(?P<milliseconds>\d{3})$"
)


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
    return _blocks_to_srt_text(
        [
            SubtitleSegment(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                text=segment.text,
            )
            for segment in segments
        ],
        max_line_length=max_line_length,
    )


def translation_segments_to_srt_text(
    segments: Iterable[TranslationSegment],
    *,
    text_field: str,
    max_line_length: int,
) -> str:
    return _blocks_to_srt_text(
        [
            SubtitleSegment(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                text=getattr(segment, text_field),
            )
            for segment in segments
        ],
        max_line_length=max_line_length,
    )


def parse_srt_text(srt_text: str) -> list[SubtitleSegment]:
    if not srt_text.strip():
        return []

    blocks = [block.strip() for block in re.split(r"\r?\n\r?\n", srt_text.strip()) if block.strip()]
    segments: list[SubtitleSegment] = []

    for index, block in enumerate(blocks, start=1):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(f"Subtitle block {index} is incomplete.")

        timestamp_line = lines[1] if lines[0].isdigit() else lines[0]
        text_lines = lines[2:] if lines[0].isdigit() else lines[1:]

        if "-->" not in timestamp_line:
            raise ValueError(f"Subtitle block {index} is missing a timestamp line.")

        raw_start, raw_end = [part.strip() for part in timestamp_line.split("-->", 1)]
        start_seconds = parse_srt_timestamp(raw_start)
        end_seconds = parse_srt_timestamp(raw_end)
        if end_seconds < start_seconds:
            raise ValueError(f"Subtitle block {index} ends before it starts.")

        text = "\n".join(text_lines).strip()
        if not text:
            raise ValueError(f"Subtitle block {index} does not contain subtitle text.")

        segments.append(
            SubtitleSegment(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                text=text,
            )
        )

    return segments


def validate_review_srt_text(
    srt_text: str,
    reference_segments: Iterable[TranslationSegment],
) -> str | None:
    try:
        parsed_segments = parse_srt_text(srt_text)
    except ValueError as exc:
        return str(exc)

    references = list(reference_segments)
    if not references:
        return None

    parsed_index = 0
    for index, reference in enumerate(references, start=1):
        found_reference = False
        while parsed_index < len(parsed_segments):
            parsed = parsed_segments[parsed_index]
            parsed_index += 1
            if (
                abs(parsed.start_seconds - reference.start_seconds) <= 0.001
                and abs(parsed.end_seconds - reference.end_seconds) <= 0.001
            ):
                found_reference = True
                break

        if not found_reference:
            return (
                f"Subtitle block {index} is missing or changed timing. "
                "Keep every original subtitle block and timestamp, but you can insert extra blocks for missed lines."
            )

    return None


def write_srt(
    segments: Iterable[SubtitleSegment],
    output_path: Path,
    *,
    max_line_length: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srt_text = segments_to_srt_text(segments, max_line_length=max_line_length)
    output_path.write_text(srt_text if srt_text else "\n", encoding="utf-8")
    return len(parse_srt_text(srt_text)) if srt_text.strip() else 0


def parse_srt_timestamp(value: str) -> float:
    match = _TIMESTAMP_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(f"Invalid SRT timestamp: {value}")

    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    milliseconds = int(match.group("milliseconds"))
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def _blocks_to_srt_text(
    segments: list[SubtitleSegment],
    *,
    max_line_length: int,
) -> str:
    blocks: list[str] = []
    sequence = 0
    for segment in segments:
        text = wrap_subtitle_text(segment.text, max_line_length).strip()
        if not text:
            continue
        sequence += 1
        start = format_srt_timestamp(segment.start_seconds)
        end = format_srt_timestamp(max(segment.end_seconds, segment.start_seconds + 0.01))
        blocks.append(f"{sequence}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n" if blocks else ""
