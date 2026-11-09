"""Parse client cut lists ("00:10-00:13") and turn them into segments to keep."""

from __future__ import annotations

import re

TimeRange = tuple[float, float]

_RANGE_PATTERN = re.compile(
    r"^\s*(?P<start>[0-9:.]+)\s*(?:-|–|—|to)\s*(?P<end>[0-9:.]+)\s*$",
    re.IGNORECASE,
)


def parse_timestamp(text: str) -> float:
    """Parse SS, M:SS, MM:SS or H:MM:SS (seconds may have decimals)."""
    parts = text.strip().split(":")
    if not parts or len(parts) > 3 or any(part == "" for part in parts):
        raise ValueError(f"'{text}' is not a valid time. Use MM:SS or H:MM:SS.")
    try:
        seconds = float(parts[-1])
        whole_units = [int(part) for part in parts[:-1]]
    except ValueError:
        raise ValueError(f"'{text}' is not a valid time. Use MM:SS or H:MM:SS.") from None

    if len(parts) > 1 and seconds >= 60:
        raise ValueError(f"'{text}' has {parts[-1]} seconds; seconds must be below 60.")
    if len(parts) == 3 and whole_units[1] >= 60:
        raise ValueError(f"'{text}' has {parts[1]} minutes; minutes must be below 60.")

    total = seconds
    for multiplier, value in zip((60, 3600), reversed(whole_units)):
        total += value * multiplier
    return total


def parse_time_ranges(text: str) -> list[TimeRange]:
    """Parse one range per line (commas and semicolons also separate ranges).

    Returns sorted ranges with overlaps merged. Raises ValueError naming the bad entry.
    """
    ranges: list[TimeRange] = []
    for entry in re.split(r"[\n,;]", text):
        if not entry.strip():
            continue
        match = _RANGE_PATTERN.match(entry)
        if match is None:
            raise ValueError(f"'{entry.strip()}' is not a time range. Use START-END, for example 01:33-02:26.")
        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        if end <= start:
            raise ValueError(f"'{entry.strip()}' ends before it starts.")
        ranges.append((start, end))
    return merge_ranges(ranges)


def merge_ranges(ranges: list[TimeRange]) -> list[TimeRange]:
    merged: list[TimeRange] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def keep_segments(remove_ranges: list[TimeRange], duration: float) -> list[TimeRange]:
    """Return the parts of [0, duration] that are not removed.

    Raises ValueError if a range starts past the end of the media (usually a typo)
    or if nothing would be left.
    """
    for start, end in remove_ranges:
        if start >= duration:
            raise ValueError(
                f"Cut {format_clock(start)}-{format_clock(end)} starts after the end of the media "
                f"({format_clock(duration)})."
            )

    kept: list[TimeRange] = []
    cursor = 0.0
    for start, end in merge_ranges(remove_ranges):
        if start > cursor:
            kept.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        kept.append((cursor, duration))

    # Drop slivers too short to encode cleanly.
    kept = [(start, end) for start, end in kept if end - start >= 0.05]
    if not kept:
        raise ValueError("The cuts remove the whole recording.")
    return kept


def total_duration(ranges: list[TimeRange]) -> float:
    return sum(end - start for start, end in ranges)


def format_clock(seconds: float) -> str:
    whole = int(round(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
