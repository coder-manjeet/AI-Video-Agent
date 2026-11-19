"""Tests for project/scene model serialization and extensibility."""

from __future__ import annotations

import pytest

from project_models import Project, ProjectSettings, Scene


def test_project_defaults_and_roundtrip() -> None:
    project = Project(
        project_id="demo",
        title="Demo",
        created_at="2026-11-19T00:00:00+00:00",
        updated_at="2026-11-19T00:00:00+00:00",
    )

    payload = project.to_dict()
    restored = Project.from_dict(payload)

    assert restored == project
    assert payload["settings"]["width"] == 768
    assert payload["settings"]["height"] == 512
    assert payload["settings"]["fps"] == 24
    assert payload["settings"]["target_duration"] == 60.0
    assert payload["settings"]["video_format"] == "mp4"
    assert payload["settings"]["aspect_ratio"] == "16:9"


def test_unknown_fields_survive_roundtrip() -> None:
    payload = {
        "project_id": "demo",
        "title": "Demo",
        "custom_note": "keep me",
        "settings": {"width": 1920, "future_flag": True},
    }

    project = Project.from_dict(payload)
    output = project.to_dict()

    assert project.extra["custom_note"] == "keep me"
    assert project.settings.extra["future_flag"] is True
    assert output["custom_note"] == "keep me"
    assert output["settings"]["future_flag"] is True
    assert output["settings"]["width"] == 1920


def test_scene_roundtrip_preserves_all_fields() -> None:
    scene = Scene(
        scene_id="intro-1",
        scene_number=1,
        title="Intro",
        description="Opening shot",
        duration=8.5,
        narration="Hello world",
        image_prompt="a sunrise",
        animation_prompt="slow pan",
        negative_prompt="text, watermark",
        character_references=["characters/hero.png"],
        image_path="assets/images/intro.png",
        generated_video_path="video/scenes/intro.mp4",
        narration_audio_path="audio/narration/intro.wav",
        sfx_paths=["audio/sfx/wind.wav"],
        status="completed",
        timeline_start=0.0,
        timeline_end=8.5,
        thumbnail_path="assets/images/intro-thumb.png",
        generation_provider="custom-workflow",
        provider_options={"workflow": "other.json"},
    )

    restored = Scene.from_dict(scene.to_dict())

    assert restored == scene
    assert restored.provider_options == {"workflow": "other.json"}
    assert restored.sfx_paths == ["audio/sfx/wind.wav"]


def test_scene_from_dict_tolerates_partial_manifest() -> None:
    scene = Scene.from_dict(
        {"scene_id": "s1", "scene_number": 1, "title": "Draft scene"}
    )

    assert scene.duration == 5.0
    assert scene.status == "draft"
    assert scene.image_path is None
    assert scene.generated_video_path is None
    assert scene.narration_audio_path is None
    assert scene.character_references == []
    assert scene.sfx_paths == []


def test_settings_from_dict_ignores_unknown_keys() -> None:
    settings = ProjectSettings.from_dict({"height": 1080, "color_space": "rec709"})

    assert settings.height == 1080
    assert settings.width == 768
    assert settings.extra == {"color_space": "rec709"}


def test_missing_required_identifiers_raise_value_error() -> None:
    with pytest.raises(ValueError, match="project_id"):
        Project.from_dict({"title": "No ID"})

    with pytest.raises(ValueError, match="scene_id"):
        Scene.from_dict({"title": "No ID", "scene_number": 1})
