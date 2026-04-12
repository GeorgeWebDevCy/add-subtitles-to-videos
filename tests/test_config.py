from add_subtitles_to_videos import config


def test_confidence_threshold_constant_exists():
    assert hasattr(config, "CONFIDENCE_HIGHLIGHT_THRESHOLD")
    assert isinstance(config.CONFIDENCE_HIGHLIGHT_THRESHOLD, float)
    assert config.CONFIDENCE_HIGHLIGHT_THRESHOLD == -0.6


def test_vad_defaults_exist():
    assert config.DEFAULT_VAD_THRESHOLD == 0.5
    assert config.DEFAULT_VAD_MIN_SILENCE_MS == 2000


def test_language_model_profiles_exist():
    profiles = config.LANGUAGE_MODEL_PROFILES
    assert isinstance(profiles, dict)
    assert "auto" in profiles
    assert "el" in profiles
    assert "en" in profiles
    assert all(isinstance(v, str) for v in profiles.values())
