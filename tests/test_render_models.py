"""Tests for typed render models, Decimal serialization and immutability."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from manifest_validation import ValidationIssue
from render_models import (
    DEFAULT_DURATION_MODE,
    RenderExpectations,
    RenderMediaInfo,
    RenderPlan,
    RenderPolicy,
    RenderScene,
    RenderValidationSettings,
)


def _media() -> RenderMediaInfo:
    return RenderMediaInfo(
        path="video/scenes/s1.mp4",
        duration_s=Decimal("4.041667"),
        size_bytes=167200,
        width=768,
        height=512,
        fps=24.0,
        video_codec="h264",
        format_name="mov,mp4",
    )


def _scene() -> RenderScene:
    return RenderScene(
        scene_id="s1",
        scene_number=1,
        title="Scene One",
        status="video_generated",
        planned_duration_s=Decimal("4.0"),
        measured_duration_s=Decimal("4.041667"),
        effective_duration_s=Decimal("4.041667"),
        video_hold_duration_s=Decimal(0),
        silent_tail_duration_s=Decimal(0),
        timeline_start_s=Decimal(0),
        timeline_end_s=Decimal("4.041667"),
        planned_timeline_start_s=Decimal(0),
        planned_timeline_end_s=Decimal("4.0"),
        duration_delta_s=Decimal("0.041667"),
        narration_duration_s=None,
        narration_delta_s=None,
        narration_audio_path=None,
        narration_text="",
        generated_video_path="video/scenes/s1.mp4",
        video=_media(),
    )


def _plan(scene: RenderScene) -> RenderPlan:
    return RenderPlan(
        project_id="demo",
        policy=RenderPolicy(),
        scenes=(scene,),
        warnings=(ValidationIssue("render.narration.none", "No narration"),),
        total_planned_duration_s=Decimal("4.0"),
        total_render_duration_s=Decimal("4.041667"),
        output_width=768,
        output_height=512,
        output_fps=24.0,
    )


def test_default_policy_uses_actual_measured_extend_narration() -> None:
    policy = RenderPolicy()

    assert policy.duration_mode == DEFAULT_DURATION_MODE
    assert policy.duration_mode == "actual_measured_extend_narration"
    assert policy.require_audio is True
    assert isinstance(policy.validation, RenderValidationSettings)


def test_invalid_duration_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported duration_mode"):
        RenderPolicy(duration_mode="trim_audio")
    with pytest.raises(ValueError, match="Unsupported duration_mode"):
        RenderPolicy.from_dict({"duration_mode": "extend_video"})


def test_negative_durations_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RenderPolicy(duration_tolerance_s=Decimal("-0.1"))
    with pytest.raises(ValueError, match="positive"):
        RenderExpectations(expected_duration_s=Decimal(-1))
    with pytest.raises(ValueError, match="positive"):
        replace(_scene(), planned_duration_s=Decimal(-4))
    with pytest.raises(ValueError, match="non-negative"):
        replace(_scene(), video_hold_duration_s=Decimal("-0.5"))


def test_decimal_timeline_serialization_is_lossless_and_json_safe() -> None:
    payload = _plan(_scene()).to_dict()

    json.dumps(payload)
    assert payload["total_render_duration_s"] == "4.041667"
    assert payload["total_planned_duration_s"] == "4.0"
    assert payload["policy"]["duration_tolerance_s"] == "0.5"
    scene_payload = payload["scenes"][0]
    assert Decimal(scene_payload["timeline_end_s"]) == Decimal("4.041667")
    assert Decimal(scene_payload["duration_delta_s"]) == Decimal("0.041667")
    assert scene_payload["video"]["duration_s"] == "4.041667"
    assert payload["warnings"][0]["code"] == "render.narration.none"


def test_plan_and_scene_are_immutable() -> None:
    plan = _plan(_scene())

    assert isinstance(plan.scenes, tuple)
    assert isinstance(plan.warnings, tuple)
    with pytest.raises(FrozenInstanceError):
        plan.scenes = ()
    with pytest.raises(FrozenInstanceError):
        plan.scenes[0].measured_duration_s = Decimal(1)
    with pytest.raises(FrozenInstanceError):
        plan.policy.duration_mode = "trim_audio"


def test_expectations_derive_validation_settings_from_policy() -> None:
    policy = RenderPolicy(
        width=1920,
        height=1080,
        fps=30.0,
        require_audio=False,
        duration_tolerance_s=Decimal("0.25"),
        validation=RenderValidationSettings(fps_tolerance=0.1),
    )

    expectations = RenderExpectations.from_policy(policy, duration_s=Decimal(5))

    assert expectations.width == 1920
    assert expectations.height == 1080
    assert expectations.fps == 30.0
    assert expectations.expected_duration_s == Decimal(5)
    assert expectations.duration_tolerance_s == Decimal("0.25")
    assert expectations.fps_tolerance == 0.1
    assert expectations.require_video_stream is True
