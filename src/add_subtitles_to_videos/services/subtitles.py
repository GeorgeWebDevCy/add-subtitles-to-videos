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

    # Keep the review editor compact; split_long_segments() breaks over-long
    # blocks into separate timed blocks on export.
    midpoint = max(1, len(wrapped_lines) // 2)
    first_line = " ".join(wrapped_lines[:midpoint])
    second_line = " ".join(wrapped_lines[midpoint:])
    return "\n".join((first_line, second_line))


_BREAK_AFTER = (".", "?", "!", ",", ";", ":")


def _wrapped_lines(words: list[str], max_line_length: int) -> list[str]:
    return wrap(" ".join(words), width=max_line_length, break_long_words=False, break_on_hyphens=False)


def _fits(words: list[str], max_line_length: int, max_lines: int) -> bool:
    return len(_wrapped_lines(words, max_line_length)) <= max_lines


def _balanced_lines(words: list[str], max_line_length: int, max_lines: int) -> list[str]:
    """Wrap into lines; for two lines pick the break that evens out the line lengths
    (slightly preferring a break after punctuation) so no word is left dangling."""
    greedy = _wrapped_lines(words, max_line_length)
    if len(greedy) != 2 or max_lines < 2:
        return greedy
    best: tuple[float, list[str]] | None = None
    for index in range(1, len(words)):
        first, second = " ".join(words[:index]), " ".join(words[index:])
        if len(first) > max_line_length or len(second) > max_line_length:
            continue
        score = max(len(first), len(second)) - (4 if words[index - 1].endswith(_BREAK_AFTER) else 0)
        if best is None or score < best[0]:
            best = (score, [first, second])
    return best[1] if best else greedy


def _balanced_chunks(words: list[str], max_line_length: int, max_lines: int) -> list[list[str]]:
    """Split words into the fewest similar-length chunks that each fit on max_lines lines,
    preferring to cut after punctuation."""
    greedy: list[list[str]] = []
    current: list[str] = []
    for word in words:
        if current and not _fits(current + [word], max_line_length, max_lines):
            greedy.append(current)
            current = []
        current.append(word)
    if current:
        greedy.append(current)

    cumulative: list[int] = []
    total = 0
    for word in words:
        total += len(word) + 1
        cumulative.append(total)

    for count in range(len(greedy), len(words) + 1):
        chunk_size = total / count
        cuts: list[int] = []
        previous = -1
        for part in range(1, count):
            target = total * part / count
            candidates = range(previous + 1, len(words) - (count - part))
            if not candidates:
                break
            window = [
                index
                for index in candidates
                if words[index].endswith(_BREAK_AFTER) and abs(cumulative[index] - target) <= chunk_size * 0.25
            ]
            pool = window or list(candidates)
            cut = min(pool, key=lambda index: abs(cumulative[index] - target))
            cuts.append(cut)
            previous = cut
        if len(cuts) != count - 1:
            continue
        bounds = [0, *(cut + 1 for cut in cuts), len(words)]
        chunks = [words[bounds[i] : bounds[i + 1]] for i in range(count)]
        if all(chunk and _fits(chunk, max_line_length, max_lines) for chunk in chunks):
            return chunks
    return greedy


def split_long_segments(
    segments: Iterable[SubtitleSegment],
    *,
    max_line_length: int,
    max_lines: int = 2,
) -> list[SubtitleSegment]:
    """Make every block fit on max_lines lines of max_line_length characters.

    Blocks that already fit keep their text and line breaks exactly. Longer blocks
    are split into consecutive blocks, sharing the original time by text length.
    """
    result: list[SubtitleSegment] = []
    for segment in segments:
        lines = segment.text.splitlines()
        if not segment.text.strip() or (
            len(lines) <= max_lines and all(len(line) <= max_line_length for line in lines)
        ):
            result.append(segment)
            continue

        words = segment.text.split()
        if _fits(words, max_line_length, max_lines):
            chunks = [words]
        else:
            chunks = _balanced_chunks(words, max_line_length, max_lines)

        weights = [len(" ".join(chunk)) + 1 for chunk in chunks]
        duration = max(segment.end_seconds - segment.start_seconds, 0.0)
        start = segment.start_seconds
        for index, (chunk, weight) in enumerate(zip(chunks, weights)):
            end = (
                segment.end_seconds
                if index == len(chunks) - 1
                else round(start + duration * weight / sum(weights), 3)
            )
            result.append(
                SubtitleSegment(
                    start_seconds=start,
                    end_seconds=end,
                    text="\n".join(_balanced_lines(chunk, max_line_length, max_lines)),
                )
            )
            start = end
    return result


def clamp_segments_to_window(
    segments: Iterable[SubtitleSegment], start: float, end: float
) -> list[SubtitleSegment]:
    """Keep subtitles inside [start, end] (e.g. off a silent logo intro/outro).

    Whisper can stamp the first line from 0:00 even when the file opens with
    silence. Segments are trimmed to the window; ones left with no time are dropped.
    """
    clamped: list[SubtitleSegment] = []
    for segment in segments:
        new_start = min(max(segment.start_seconds, start), end)
        new_end = min(max(segment.end_seconds, start), end)
        if new_end - new_start < 0.05:
            continue
        clamped.append(SubtitleSegment(start_seconds=new_start, end_seconds=new_end, text=segment.text))
    return clamped


def fit_srt_text_to_lines(srt_text: str, *, max_line_length: int, max_lines: int = 2) -> tuple[str, int]:
    """Return SRT text where every block fits, and how many blocks were split or rewrapped."""
    segments = parse_srt_text(srt_text, allow_empty_text=True)
    fitted = split_long_segments(segments, max_line_length=max_line_length, max_lines=max_lines)
    changed = sum(1 for segment in segments if segment not in fitted)
    if not changed:
        return srt_text, 0
    return segments_to_plain_srt_text(fitted, include_empty_text=True), changed


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


def segments_to_plain_srt_text(
    segments: Iterable[SubtitleSegment],
    *,
    include_empty_text: bool = False,
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
        preserve_text=True,
        include_empty_text=include_empty_text,
    )


def parse_srt_text(
    srt_text: str,
    *,
    allow_empty_text: bool = False,
) -> list[SubtitleSegment]:
    if not srt_text.strip():
        return []

    blocks = [block.strip() for block in re.split(r"\r?\n\r?\n", srt_text.strip()) if block.strip()]
    segments: list[SubtitleSegment] = []

    for index, block in enumerate(blocks, start=1):
        lines = [line.rstrip() for line in block.splitlines()]
        if len(lines) < 2:
            raise ValueError(f"Subtitle block {index} is incomplete.")

        has_sequence = lines[0].strip().isdigit()
        timestamp_line = lines[1].strip() if has_sequence else lines[0].strip()
        text_lines = lines[2:] if has_sequence else lines[1:]

        if "-->" not in timestamp_line:
            raise ValueError(f"Subtitle block {index} is missing a timestamp line.")

        raw_start, raw_end = [part.strip() for part in timestamp_line.split("-->", 1)]
        start_seconds = parse_srt_timestamp(raw_start)
        end_seconds = parse_srt_timestamp(raw_end)
        if end_seconds < start_seconds:
            raise ValueError(f"Subtitle block {index} ends before it starts.")

        text = _preserve_subtitle_text("\n".join(text_lines))
        if not text and not allow_empty_text:
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
        parsed_segments = parse_srt_text(srt_text, allow_empty_text=True)
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
    max_line_length: int | None = None,
    preserve_text: bool = False,
    include_empty_text: bool = False,
) -> str:
    blocks: list[str] = []
    sequence = 0
    for segment in segments:
        text = _preserve_subtitle_text(segment.text) if preserve_text else wrap_subtitle_text(
            segment.text,
            max_line_length or 1,
        ).strip()
        if not text and not include_empty_text:
            continue
        sequence += 1
        start = format_srt_timestamp(segment.start_seconds)
        end = format_srt_timestamp(max(segment.end_seconds, segment.start_seconds + 0.01))
        if text:
            blocks.append(f"{sequence}\n{start} --> {end}\n{text}")
        else:
            blocks.append(f"{sequence}\n{start} --> {end}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _preserve_subtitle_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()
