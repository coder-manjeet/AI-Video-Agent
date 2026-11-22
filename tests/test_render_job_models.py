"""Tests for persistent render job models and their JSON representations."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from project_paths import is_iso8601_timestamp
from render_job_models import (
    ACTIVE_RENDER_STATUSES,
    RENDER_PRESETS,
    RENDER_STATUSES,
    TERMINAL_RENDER_STATUSES,
    RenderConfig,
    RenderJob,
    RenderJobError,
    RenderProgress,
    preset_values,
    utc_now,
)


def test_status_sets_partition_expected_states() -> None:
    assert len(RENDER_STATUSES) == 12
    assert RENDER_STATUSES == ACTIVE_RENDER_STATUSES | TERMINAL_RENDER_STATUSES
    assert ACTIVE_RENDER_STATUSES.isdisjoint(TERMINAL_RENDER_STATUSES)
    assert TERMINAL_RENDER_STATUSES == {"completed", "failed", "cancelled"}
    assert RENDER_PRESETS == {"720p", "1080p", "custom"}


def test_preset_values_are_normalized() -> None:
    assert preset_values("720p") == (1280, 720)
    assert preset_values(" 1080P ") == (1920, 1080)
    with pytest.raises(ValueError, match="Unsupported preset"):
        preset_values("4k")


def test_config_defaults_are_json_serializable() -> None:
    config = RenderConfig()

    assert config.preset == "1080p"
    assert (config.width, config.height, config.fps) == (1920, 1080, 24.0)
    assert config.duration_tolerance_s == Decimal("0.5")
    assert config.sync_tolerance_s == Decimal("0.5")
    assert config.container == "mp4"
    assert config.burn_subtitles is False
    assert json.loads(json.dumps(config.to_dict())) == config.to_dict()


@pytest.mark.parametrize(
    ("preset", "expected"),
    [("720p", (1280, 720)), ("1080P", (1920, 1080))],
)
def test_config_expands_named_presets(preset: str, expected: tuple[int, int]) -> None:
    config = RenderConfig.from_dict({"preset": preset})

    assert (config.width, config.height) == expected
    assert RenderConfig(preset="720p").width == 1280


def test_config_custom_requires_both_dimensions() -> None:
    for payload in ({"preset": "custom"}, {"preset": "custom", "width": 1280}):
        with pytest.raises(ValueError, match="custom preset requires both"):
            RenderConfig.from_dict(payload)


def test_config_custom_accepts_explicit_dimensions() -> None:
    config = RenderConfig.from_dict(
        {"preset": "custom", "width": 960, "height": 540, "fps": 30}
    )

    assert (config.width, config.height, config.fps) == (960, 540, 30.0)
    assert config.to_dict()["preset"] == "custom"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"preset": "custom", "width": 1281, "height": 720}, "even"),
        ({"preset": "custom", "width": 1280, "height": 0}, "positive"),
    ],
)
def test_config_rejects_invalid_dimensions(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RenderConfig.from_dict(payload)


def test_config_rejects_invalid_preset_and_fps() -> None:
    with pytest.raises(ValueError, match="Unsupported preset"):
        RenderConfig.from_dict({"preset": "4k"})
    for payload in ({"fps": 0}, {"fps": -24}, {"fps": "24"}):
        with pytest.raises((ValueError, TypeError)):
            RenderConfig.from_dict(payload)


def test_config_rejects_invalid_audio_settings() -> None:
    for key in ("audio_bitrate_kbps", "audio_sample_rate", "audio_channels"):
        with pytest.raises((ValueError, TypeError)):
            RenderConfig.from_dict({key: 0})


def test_config_rejects_negative_tolerances() -> None:
    for payload in (
        {"duration_tolerance_s": "-0.1"},
        {"sync_tolerance_s": -0.1},
        {"fps_tolerance": -0.1},
    ):
        with pytest.raises(ValueError):
            RenderConfig.from_dict(payload)


def test_config_rejects_unsupported_container_codecs_and_formats() -> None:
    for payload in (
        {"container": "mkv"},
        {"video_codec": "av1"},
        {"audio_codec": "pcm_s16le"},
        {"pixel_format": "rgb24"},
        {"preset_x264": "turbo"},
        {"subtitle_formats": ["txt"]},
        {"missing_narration_policy": "skip"},
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            RenderConfig.from_dict(payload)


def test_config_rejects_non_boolean_burn_subtitles() -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        RenderConfig.from_dict({"burn_subtitles": "yes"})


def test_config_rejects_absolute_or_parent_render_root() -> None:
    for root in ("C:/renders", "../escape"):
        with pytest.raises(ValueError, match="must be relative"):
            RenderConfig.from_dict({"render_root": root})


def test_config_serializes_decimals_as_strings_and_formats_as_lists() -> None:
    payload = RenderConfig().to_dict()

    assert payload["duration_tolerance_s"] == "0.5"
    assert payload["sync_tolerance_s"] == "0.5"
    assert payload["subtitle_formats"] == ["srt", "vtt"]


def test_config_roundtrip_preserves_unknown_extra_keys() -> None:
    config = RenderConfig.from_dict(
        {"preset": "720p", "custom_flag": True, "extra": {"nested": {"value": 1}}}
    )

    assert (config.width, config.height) == (1280, 720)
    assert config.extra == {"custom_flag": True, "nested": {"value": 1}}
    restored = RenderConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()


def test_config_is_frozen() -> None:
    config = RenderConfig()

    with pytest.raises(FrozenInstanceError):
        config.width = 1280  # type: ignore[misc]


def test_progress_validation_and_serialization() -> None:
    progress = RenderProgress(stage="merging", percent=None, label="Waiting")
    assert progress.to_dict() == {
        "stage": "merging",
        "percent": None,
        "label": "Waiting",
        "completed": 0,
        "total": 0,
    }
    for payload in ({"stage": "merging", "percent": 101}, {"stage": "merging", "completed": -1}):
        with pytest.raises(ValueError):
            RenderProgress.from_dict(payload)


def test_job_status_properties() -> None:
    for status in sorted(RENDER_STATUSES):
        job = RenderJob(job_id="job-1", project_id="demo", render_id="render-1", status=status)
        assert job.is_active is (status in ACTIVE_RENDER_STATUSES)
        assert job.is_terminal is (status in TERMINAL_RENDER_STATUSES)

    default = RenderJob(job_id="job-1", project_id="demo", render_id="render-1")
    assert default.status == "queued"
    assert default.stage == "queued"
    assert default.progress.stage == "queued"


def test_job_validates_identifiers_status_and_stage() -> None:
    base: dict[str, object] = {
        "job_id": "job-1",
        "project_id": "demo",
        "render_id": "render-1",
    }
    for updates in (
        {"job_id": "../bad"},
        {"project_id": "bad id"},
        {"render_id": "x/y"},
        {"status": "unknown"},
        {"stage": "unknown"},
    ):
        with pytest.raises(ValueError):
            RenderJob.from_dict({**base, **updates})


def test_job_validates_timestamps() -> None:
    for updates in (
        {"created_at": "not-a-time"},
        {"started_at": "2026-01-01T00:00:00+05:30"},
        {"updated_at": 123},
        {"completed_at": "yesterday"},
    ):
        with pytest.raises(ValueError):
            RenderJob.from_dict(
                {
                    "job_id": "job-1",
                    "project_id": "demo",
                    "render_id": "render-1",
                    **updates,
                }
            )


def test_job_dict_roundtrip_is_json_safe() -> None:
    job = RenderJob(
        job_id="job-1",
        project_id="demo",
        render_id="render-1",
        status="mixing_audio",
        stage="mixing_audio",
        progress=RenderProgress(stage="mixing_audio", percent=60.0, completed=3, total=5),
        config={"preset": "1080p"},
        created_at="2000-01-01T00:00:00Z",
        updated_at="2000-01-01T00:00:05Z",
        output_paths={"master": "video/renders/render-1/master.mp4"},
        validation_summary={"valid": True},
        stages=[{"stage": "merging", "status": "completed"}],
        cancel_requested=True,
        attempt=2,
        warnings=[{"code": "render.warn"}],
    )

    payload = job.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert RenderJob.from_dict(payload).to_dict() == payload


def test_job_evolve_and_with_status_are_immutable_updates() -> None:
    job = RenderJob(
        job_id="job-1",
        project_id="demo",
        render_id="render-1",
        created_at="2000-01-01T00:00:00Z",
        updated_at="2000-01-01T00:00:00Z",
    )

    updated = job.with_status(
        "merging",
        stage="merging",
        progress=RenderProgress(stage="merging", percent=50.0),
        updated_at="2000-01-01T00:01:00Z",
    )
    assert updated.status == "merging"
    assert updated.progress.percent == 50.0
    assert job.status == "queued"

    cancelled = updated.evolve(cancel_requested=True)
    assert cancelled.cancel_requested is True
    assert cancelled.updated_at > updated.updated_at
    with pytest.raises(ValueError, match="identity"):
        updated.evolve(job_id="job-2")
    with pytest.raises(ValueError, match="Unknown render job fields"):
        updated.evolve(unexpected=1)


def test_render_job_error_is_structured() -> None:
    error = RenderJobError("render.job.corrupt", "bad payload", "job-1.json")

    assert error.to_dict() == {
        "code": "render.job.corrupt",
        "message": "bad payload",
        "location": "job-1.json",
    }
    assert str(error) == "bad payload"


def test_utc_now_is_iso8601_utc() -> None:
    now: str = utc_now()

    assert is_iso8601_timestamp(now)
    assert now.endswith("Z")
