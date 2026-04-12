from add_subtitles_to_videos.models import ProcessingOptions, OutputMode, WorkflowProfile
from pathlib import Path


def _base_options(**overrides):
    defaults = dict(
        source_language=None,
        target_language="en",
        translation_provider=None,
        whisper_model="large-v3",
        output_mode=OutputMode.SRT_ONLY,
        output_directory=Path("."),
        max_line_length=42,
        subtitle_font_size=18,
    )
    defaults.update(overrides)
    return ProcessingOptions(**defaults)


def test_processing_options_new_fields_have_defaults():
    opts = _base_options()
    assert opts.word_timestamps is False
    assert opts.vad_threshold == 0.5
    assert opts.vad_min_silence_ms == 2000


def test_processing_options_new_fields_are_settable():
    opts = _base_options(word_timestamps=True, vad_threshold=0.3, vad_min_silence_ms=1000)
    assert opts.word_timestamps is True
    assert opts.vad_threshold == 0.3
    assert opts.vad_min_silence_ms == 1000
