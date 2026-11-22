"""Deterministic tests for the shared Milestone 3D data models."""

from __future__ import annotations

from decimal import Decimal

import pytest

from audio_models import (
    AudioPipelineError,
    AudioTimelinePolicy,
    DuckingConfig,
    MusicSettings,
    NarrationConfig,
    SfxSettings,
    decimal_text,
)


@pytest.mark.parametrize("language", ["en", "hi", "hinglish", "HI"])
def test_narration_config_accepts_supported_languages(language: str) -> None:
    config = NarrationConfig.from_dict({"language": language})

    assert config.language == language.lower()
    assert config.speed == 1.0
    assert config.voice == "chatterbox-default-narrator"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"language": "fr"}, "Unsupported language"),
        ({"speed": 1.25}, "no speech-rate parameter"),
        ({"output_root": "../escape"}, "must be relative"),
        ({"output_root": "C:/narration"}, "must be relative"),
        ({"timeout_s": 0}, "must be positive"),
    ],
)
def test_narration_config_rejects_unsupported_settings(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        NarrationConfig.from_dict(payload)


def test_narration_config_round_trip_preserves_reference_and_expressiveness() -> None:
    payload: dict[str, object] = {
        "language": "hinglish",
        "expressiveness": 0.6,
        "reference_wav": "assets/voice/sample.wav",
        "output_root": "audio/narration",
        "timeout_s": 120,
    }

    restored = NarrationConfig.from_dict(NarrationConfig.from_dict(payload).to_dict())

    assert restored.to_dict() == NarrationConfig.from_dict(payload).to_dict()
    assert restored.reference_wav == "assets/voice/sample.wav"
    assert restored.expressiveness == 0.6


def test_timeline_policy_defaults_to_allow_and_serializes_decimals() -> None:
    policy = AudioTimelinePolicy()

    assert policy.missing_narration_policy == "allow"
    assert policy.to_dict()["duration_tolerance_s"] == "0.05"

    strict = AudioTimelinePolicy.from_dict(
        {"missing_narration_policy": "error", "duration_tolerance_s": "0.25"}
    )

    assert strict.missing_narration_policy == "error"
    assert strict.duration_tolerance_s == Decimal("0.25")


def test_timeline_policy_rejects_unknown_missing_policy() -> None:
    with pytest.raises(ValueError, match="missing_narration_policy"):
        AudioTimelinePolicy.from_dict({"missing_narration_policy": "skip"})


def test_ducking_ratio_and_bounds() -> None:
    config = DuckingConfig.from_dict(
        {"level": 0.25, "attack_ms": 80, "release_ms": 300, "threshold": 0.01}
    )

    assert config.ratio == pytest.approx(4.0)
    assert config.to_dict()["attack_ms"] == 80

    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        DuckingConfig(level=1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DuckingConfig(threshold=-0.1)


def test_music_settings_validate_placement_and_ducking() -> None:
    settings = MusicSettings.from_dict(
        {
            "path": "assets/audio/bed.wav",
            "volume": 0.3,
            "start_offset_s": "1.5",
            "trim_start_s": "0.25",
            "trim_end_s": "0.5",
            "fade_in_s": "0.2",
            "fade_out_s": "0.4",
            "loop": True,
            "ducking": {"level": 0.5},
        }
    )

    payload = settings.to_dict()
    assert payload["start_offset_s"] == "1.5"
    assert payload["loop"] is True
    assert isinstance(payload["ducking"], dict)
    assert payload["ducking"]["level"] == 0.5
    assert MusicSettings.from_dict(payload) == settings


@pytest.mark.parametrize(
    "payload",
    [
        {"path": "bed.wav", "volume": -0.1},
        {"path": "bed.wav", "start_offset_s": "-1"},
        {"path": "bed.wav", "loop": "yes"},
    ],
)
def test_music_settings_reject_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        MusicSettings.from_dict(payload)


def test_sfx_settings_require_scene_and_keep_relative_path() -> None:
    settings = SfxSettings.from_dict(
        {"path": "assets/sfx/whoosh.wav", "scene_id": "scene-1", "start_offset_s": "0.75"}
    )

    assert settings.scene_id == "scene-1"
    assert settings.to_dict()["start_offset_s"] == "0.75"

    with pytest.raises(ValueError, match="scene_id"):
        SfxSettings(path="whoosh.wav")


def test_decimal_text_and_pipeline_error_are_structured() -> None:
    assert decimal_text(Decimal("1.2300")) == "1.2300"

    error = AudioPipelineError("audio.test.failed", "Broken", "scene:one", stage="verify")
    payload = error.to_dict()

    assert payload["code"] == "audio.test.failed"
    assert payload["stage"] == "verify"
    issues = payload["issues"]
    assert isinstance(issues, list)
    assert issues[0]["location"] == "scene:one"
