from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from add_subtitles_to_videos.models import OutputMode, ProcessingOptions, SubtitleSegment, TranscriptionMetadata
from add_subtitles_to_videos.services import ffmpeg as ffmpeg_module
from add_subtitles_to_videos.services.media_edit import (
    keep_segments,
    parse_time_ranges,
    parse_timestamp,
    total_duration,
)
from add_subtitles_to_videos.services.pipeline import SubtitlePipeline

CLIENT_CUT_LIST = """
00:10-00:13

00:28-00:31

00:50-00:59

01:33-02:26

07:09-7:55
"""


def _options(tmp_path: Path, **overrides) -> ProcessingOptions:
    options = ProcessingOptions(
        source_language="el",
        target_language="el",
        translation_provider=None,
        whisper_model="base",
        output_mode=OutputMode.SRT_ONLY,
        output_directory=tmp_path / "out",
        max_line_length=42,
        subtitle_font_size=18,
    )
    return replace(options, **overrides)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("45", 45.0), ("00:10", 10.0), ("7:55", 475.0), ("1:02:03", 3723.0), ("00:10.5", 10.5)],
)
def test_parse_timestamp_accepts_common_formats(text, expected) -> None:
    assert parse_timestamp(text) == expected


@pytest.mark.parametrize("text", ["00:75", "1:60:00", "abc", "1::2", "1:2:3:4"])
def test_parse_timestamp_rejects_bad_times(text) -> None:
    with pytest.raises(ValueError):
        parse_timestamp(text)


def test_parse_time_ranges_reads_client_cut_list_with_blank_lines() -> None:
    ranges = parse_time_ranges(CLIENT_CUT_LIST)

    assert ranges == [(10, 13), (28, 31), (50, 59), (93, 146), (429, 475)]


def test_parse_time_ranges_sorts_merges_and_accepts_dashes_and_commas() -> None:
    assert parse_time_ranges("01:00 – 01:10, 00:05-00:08; 01:05 to 01:20") == [(5, 8), (60, 80)]


@pytest.mark.parametrize("text", ["00:13-00:10", "00:10", "00:10-", "hello"])
def test_parse_time_ranges_rejects_bad_entries(text) -> None:
    with pytest.raises(ValueError):
        parse_time_ranges(text)


def test_parse_time_ranges_empty_text_means_no_cuts() -> None:
    assert parse_time_ranges("  \n\n ") == []


def test_keep_segments_returns_gaps_between_cuts() -> None:
    keep = keep_segments([(0, 5), (10, 13), (55, 70)], duration=60)

    assert keep == [(5, 10), (13, 55)]
    assert total_duration(keep) == 47


def test_keep_segments_without_cuts_keeps_everything() -> None:
    assert keep_segments([], duration=30) == [(0, 30)]


def test_keep_segments_rejects_cut_after_end_of_media() -> None:
    with pytest.raises(ValueError, match="after the end"):
        keep_segments([(100, 110)], duration=60)


def test_keep_segments_rejects_removing_everything() -> None:
    with pytest.raises(ValueError, match="whole recording"):
        keep_segments([(0, 60)], duration=60)


def test_prepare_command_for_video_concats_each_kept_part() -> None:
    with patch.object(ffmpeg_module, "ffmpeg_binary", return_value="ffmpeg"):
        command = ffmpeg_module.build_prepare_command(
            Path("in.mov"),
            Path("out.mp4"),
            keep=[(0, 10), (13, 60)],
            audio_only=False,
            cover_image=None,
        )

    assert command.count("-i") == 2
    assert command[command.index("-ss", 6) + 1] == "13.000"
    graph = command[command.index("-filter_complex") + 1]
    assert "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1" in graph
    assert "-shortest" not in command


def test_prepare_command_for_audio_uses_cover_image_and_edited_length() -> None:
    with patch.object(ffmpeg_module, "ffmpeg_binary", return_value="ffmpeg"):
        command = ffmpeg_module.build_prepare_command(
            Path("in.mp3"),
            Path("out.mp4"),
            keep=[(0, 10), (13, 60)],
            audio_only=True,
            cover_image=Path("photo.jpg"),
        )

    assert command[command.index("-loop") : command.index("-loop") + 6] == [
        "-loop", "1", "-framerate", "25", "-i", "photo.jpg"
    ]
    graph = command[command.index("-filter_complex") + 1]
    assert "[0:a:0][1:a:0]concat=n=2:v=0:a=1[speech]" in graph
    assert "[2:v]scale=1920:1080" in graph
    assert command[command.index("-t", command.index("-filter_complex")) + 1] == "57.000"


def test_prepare_command_for_audio_without_image_uses_black_picture() -> None:
    with patch.object(ffmpeg_module, "ffmpeg_binary", return_value="ffmpeg"):
        command = ffmpeg_module.build_prepare_command(
            Path("in.wav"), Path("out.mp4"), keep=[(0, 5)], audio_only=True, cover_image=None
        )

    assert "color=c=black:s=1920x1080:r=25" in command


def _fake_whisper() -> MagicMock:
    whisper = MagicMock()
    whisper.transcribe.return_value = (
        [SubtitleSegment(0, 1, "α"), SubtitleSegment(1, 2, "β"), SubtitleSegment(2, 3, "γ")],
        TranscriptionMetadata(detected_language="el", detected_language_probability=None),
    )
    return whisper


def test_pipeline_cuts_media_first_and_subtitles_the_edited_file(tmp_path) -> None:
    source = tmp_path / "interview.mp3"
    options = _options(tmp_path, remove_ranges=((10, 13), (93, 146)), cover_image=tmp_path / "photo.jpg")
    logs: list[str] = []

    with patch.object(ffmpeg_module, "probe_duration", return_value=300.0), patch.object(
        ffmpeg_module, "probe_image_size", return_value=(1200, 1600)
    ), patch.object(
        ffmpeg_module, "prepare_media"
    ) as prepare, patch.object(ffmpeg_module, "extract_audio") as extract, patch.object(
        ffmpeg_module, "ffmpeg_binary", return_value="ffmpeg"
    ):
        result = SubtitlePipeline(whisper_service=_fake_whisper()).transcribe(source, options, log=logs.append)

    edited = (tmp_path / "out" / "interview.edited.mp4").resolve()
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["keep"] == [(0, 10), (13, 93), (146, 300.0)]
    assert prepare.call_args.kwargs["audio_only"] is True
    assert prepare.call_args.kwargs["cover_image"] == tmp_path / "photo.jpg"
    assert prepare.call_args.kwargs["canvas"] == (1080, 1440)
    assert extract.call_args.args[0] == edited
    assert result.input_video == edited
    assert "media_prepare_seconds" in result.metadata.stage_durations
    assert any("05:00 -> 04:04 (00:56 removed)" in line for line in logs)


def test_pipeline_skips_media_prep_for_video_without_cuts(tmp_path) -> None:
    with patch.object(ffmpeg_module, "prepare_media") as prepare, patch.object(
        ffmpeg_module, "extract_audio"
    ), patch.object(ffmpeg_module, "ffmpeg_binary", return_value="ffmpeg"):
        result = SubtitlePipeline(whisper_service=_fake_whisper()).transcribe(
            tmp_path / "clip.mov", _options(tmp_path)
        )

    prepare.assert_not_called()
    assert result.input_video == (tmp_path / "clip.mov").resolve()


def test_pipeline_reports_cut_past_end_with_file_name(tmp_path) -> None:
    options = _options(tmp_path, remove_ranges=((400, 410),))
    with patch.object(ffmpeg_module, "probe_duration", return_value=300.0), patch.object(
        ffmpeg_module, "ffmpeg_binary", return_value="ffmpeg"
    ):
        with pytest.raises(RuntimeError, match="clip.mov: Cut 06:40-06:50 starts after the end"):
            SubtitlePipeline(whisper_service=_fake_whisper()).transcribe(tmp_path / "clip.mov", options)


@pytest.mark.parametrize(
    ("size", "canvas"),
    [(None, (1920, 1080)), ((4000, 3000), (1920, 1080)), ((1200, 1600), (1080, 1440)), ((1080, 2400), (1080, 1920))],
)
def test_canvas_matches_portrait_pictures(size, canvas) -> None:
    assert ffmpeg_module.canvas_for_picture(size) == canvas


def _branded_command(**overrides) -> list[str]:
    kwargs = dict(
        keep=[(0, 10), (13, 60)],
        audio_only=True,
        cover_image=Path("photo.jpg"),
        canvas=(1080, 1440),
        intro_image=Path("logo.png"),
        show_waveform=True,
    )
    kwargs.update(overrides)
    with patch.object(ffmpeg_module, "ffmpeg_binary", return_value="ffmpeg"):
        return ffmpeg_module.build_prepare_command(Path("in.m4a"), Path("out.mp4"), **kwargs)


def test_branded_command_adds_logo_intro_outro_and_silent_padding() -> None:
    command = _branded_command()
    graph = command[command.index("-filter_complex") + 1]

    assert command[command.index("logo.png") - 5 : command.index("logo.png") + 1][0] == "-loop"
    assert graph.count("xfade=") == 4
    # Speech starts after the intro plus the short lead-in, so the intro is silent.
    delay_ms = round((ffmpeg_module.intro_main_offset() + ffmpeg_module.SPEECH_LEAD) * 1000)
    assert f"adelay=delays={delay_ms}:all=1" in graph
    total = 57 + ffmpeg_module.SPEECH_LEAD + ffmpeg_module.SPEECH_TAIL + ffmpeg_module.intro_outro_extra_seconds()
    assert command[command.index("-t", command.index("-filter_complex")) + 1] == f"{total:.3f}"
    assert "[2:v]scale=1080:1440" in graph


def test_branded_command_overlays_waveform_on_picture() -> None:
    graph = _branded_command()[_branded_command().index("-filter_complex") + 1]

    assert "asplit[speech][wave_audio]" in graph
    assert "showwaves=" in graph
    assert "[picture][wave]overlay" in graph


def test_plain_audio_command_has_no_intro_or_wave() -> None:
    command = _branded_command(intro_image=None, show_waveform=False)
    graph = command[command.index("-filter_complex") + 1]

    assert "xfade" not in graph and "showwaves" not in graph
    assert "-tune" in command


def test_pipeline_ignores_branding_for_video_input(tmp_path) -> None:
    options = _options(
        tmp_path, remove_ranges=((1, 2),), intro_image=tmp_path / "logo.png", show_waveform=True
    )
    with patch.object(ffmpeg_module, "probe_duration", return_value=30.0), patch.object(
        ffmpeg_module, "prepare_media"
    ) as prepare, patch.object(ffmpeg_module, "extract_audio"), patch.object(
        ffmpeg_module, "ffmpeg_binary", return_value="ffmpeg"
    ):
        SubtitlePipeline(whisper_service=_fake_whisper()).transcribe(tmp_path / "clip.mov", options)

    assert prepare.call_args.kwargs["intro_image"] is None
    assert prepare.call_args.kwargs["show_waveform"] is False


def test_burn_subtitles_outline_style() -> None:
    style = ffmpeg_module.OUTLINE_STYLE.format(size=11)
    assert "FontSize=11" in style and "BorderStyle=1" in style
    assert "BorderStyle=3" in ffmpeg_module.BOXED_STYLE


def test_split_long_segments_keeps_fitting_blocks_untouched() -> None:
    from add_subtitles_to_videos.services.subtitles import split_long_segments

    segment = SubtitleSegment(1.0, 3.0, "Short line\nkept as the reviewer wrote it")
    assert split_long_segments([segment], max_line_length=42) == [segment]


def test_split_long_segments_splits_long_block_into_timed_two_line_blocks() -> None:
    from add_subtitles_to_videos.services.subtitles import split_long_segments

    text = (
        "Yomori, right now, even in the Kuwait, even in the Lebanon, even in here, she couldn't\n"
        "find the perfect Yomori ingredients to make it."
    )
    result = split_long_segments([SubtitleSegment(1107.0, 1117.0, text)], max_line_length=42)

    assert len(result) == 2
    assert result[0].start_seconds == 1107.0 and result[-1].end_seconds == 1117.0
    assert result[0].end_seconds == result[1].start_seconds
    for block in result:
        lines = block.text.splitlines()
        assert len(lines) <= 2 and all(len(line) <= 42 for line in lines)
    assert " ".join(" ".join(b.text.split()) for b in result) == " ".join(text.split())
    # Cuts prefer punctuation.
    assert result[0].text.endswith("even in here,")


def test_fit_srt_text_reports_changes() -> None:
    from add_subtitles_to_videos.services.subtitles import fit_srt_text_to_lines

    srt = "1\n00:00:00,000 --> 00:00:02,000\nfine\n\n2\n00:00:02,000 --> 00:00:09,000\n" + "word " * 30 + "\n"
    fitted, changed = fit_srt_text_to_lines(srt, max_line_length=42)

    assert changed == 1
    assert fitted.count("-->") == 3

