"""Tests for visual identity persistence, scene links and backward compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from character_models import CharacterSceneOverride, VisualIdentity
from manifest_validation import (
    validate_project_directory,
    validate_project_metadata,
    validate_scene_manifest,
)
from project_models import Project, Scene
from project_service import ProjectError, ProjectService
from scene_service import SceneService


def _project(tmp_path: Path) -> tuple[ProjectService, SceneService]:
    projects = ProjectService(tmp_path)
    projects.create_project("demo", "Demo")
    return projects, SceneService(projects)


def test_project_model_visual_identity_roundtrip() -> None:
    project = Project(
        project_id="demo",
        title="Demo",
        visual_identity=VisualIdentity(illustration_style="cinematic"),
    )

    restored = Project.from_dict(project.to_dict())

    assert restored == project
    assert restored.visual_identity is not None
    assert restored.visual_identity.illustration_style == "cinematic"
    assert Project.from_dict({"project_id": "demo", "title": "Demo"}).visual_identity is None


def test_visual_identity_persistence_merge_and_clear(tmp_path: Path) -> None:
    projects, _scenes = _project(tmp_path)

    first = projects.update_visual_identity(
        "demo",
        identity={
            "illustration_style": "cinematic",
            "color_palette": "warm amber",
            "custom_flag": True,
        },
    )
    assert first.visual_identity is not None
    assert first.visual_identity.illustration_style == "cinematic"

    second = projects.update_visual_identity(
        "demo", identity={"color_palette": "cold blue"}
    )
    assert second.visual_identity is not None
    assert second.visual_identity.illustration_style == "cinematic"
    assert second.visual_identity.color_palette == "cold blue"
    assert second.visual_identity.extra["custom_flag"] is True

    from_disk = projects.load_project("demo")
    assert from_disk.visual_identity == second.visual_identity

    cleared = projects.update_visual_identity("demo", identity=None)
    assert cleared.visual_identity is None
    assert projects.load_project("demo").visual_identity is None


def test_visual_identity_rejects_invalid_field_type(tmp_path: Path) -> None:
    projects, _scenes = _project(tmp_path)

    with pytest.raises(ProjectError, match="visual_identity"):
        projects.update_visual_identity("demo", identity={"color_palette": 123})


def test_invalid_visual_identity_is_reported_by_validation() -> None:
    result = validate_project_metadata(
        {
            "project_id": "demo",
            "title": "Demo",
            "status": "active",
            "language": "en",
            "created_at": "2026-11-20T00:00:00+00:00",
            "updated_at": "2026-11-20T00:00:00+00:00",
            "visual_identity": {"color_palette": 123},
        }
    )

    assert any(issue.code == "project.invalid_visual_identity" for issue in result.errors)


def test_scene_character_fields_roundtrip_via_scene_service(tmp_path: Path) -> None:
    _projects, scenes = _project(tmp_path)
    scenes.add_scene(
        "demo", title="Opening", scene_id="s1", character_references=["legacy.png"]
    )

    updated = scenes.update_scene(
        "demo",
        "s1",
        {
            "character_ids": ["aria", "boman"],
            "character_overrides": {
                "aria": CharacterSceneOverride(expression="determined").to_dict()
            },
            "visual_identity_overrides": {"color_palette": "cold blue"},
        },
    )

    reloaded = scenes.get_scene("demo", "s1")
    assert reloaded.character_ids == ["aria", "boman"]
    assert reloaded.character_overrides["aria"]["expression"] == "determined"
    assert reloaded.visual_identity_overrides == {"color_palette": "cold blue"}
    assert reloaded.character_references == ["legacy.png"]
    assert updated.to_dict() == reloaded.to_dict()


def test_legacy_scene_manifest_is_valid_and_upgrade_is_additive(tmp_path: Path) -> None:
    projects, scenes = _project(tmp_path)
    manifest_path = projects.project_dir("demo") / "script" / "scenes.json"
    legacy_scene = {
        "scene_id": "s1",
        "scene_number": 1,
        "title": "Opening",
        "description": "A quiet morning",
        "duration": 5.0,
        "narration": "Hello",
        "image_prompt": "A sunrise",
        "animation_prompt": "Slow pan",
        "negative_prompt": "blurry",
        "character_references": ["legacy.png"],
        "image_path": None,
        "generated_video_path": None,
        "narration_audio_path": None,
        "sfx_paths": [],
        "status": "draft",
        "timeline_start": 0.0,
        "timeline_end": 5.0,
        "thumbnail_path": None,
        "generation_provider": "default",
        "provider_options": {},
        "schema_version": 1,
    }
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "scenes": [legacy_scene]}), encoding="utf-8"
    )

    loaded = scenes.get_scene("demo", "s1")
    assert loaded.character_ids == []
    assert loaded.character_overrides == {}
    assert loaded.visual_identity_overrides == {}
    assert validate_scene_manifest(json.loads(manifest_path.read_text(encoding="utf-8"))).valid
    assert validate_project_directory(projects.project_dir("demo")).valid

    scenes.update_scene("demo", "s1", {"character_ids": ["aria"]})
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["scenes"][0]["character_ids"] == ["aria"]
    assert payload["scenes"][0]["character_references"] == ["legacy.png"]


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"character_ids": ["bad id"]}, "scene.invalid_character_ids"),
        (
            {"character_overrides": {"aria": {"expression": 123}}},
            "scene.invalid_character_overrides",
        ),
        (
            {"visual_identity_overrides": {"color_palette": 123}},
            "scene.invalid_visual_identity_overrides",
        ),
    ],
)
def test_invalid_scene_character_fields_are_reported(
    changes: dict[str, Any], code: str
) -> None:
    scene = Scene(scene_id="s1", scene_number=1, title="Opening")
    payload = scene.to_dict()
    payload.update(changes)

    result = validate_scene_manifest({"schema_version": 1, "scenes": [payload]})

    assert any(issue.code == code for issue in result.errors)
