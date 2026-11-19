"""Tests for structured manifest and asset validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from manifest_validation import (
    ValidationResult,
    validate_project_directory,
    validate_project_metadata,
    validate_scene_manifest,
)
from project_service import ProjectService
from scene_service import SceneService


def _project_payload() -> dict[str, Any]:
    return {
        "project_id": "demo",
        "title": "Demo",
        "description": "",
        "status": "active",
        "language": "en",
        "created_at": "2026-11-19T00:00:00+00:00",
        "updated_at": "2026-11-19T00:00:00+00:00",
        "schema_version": 1,
        "settings": {
            "width": 768,
            "height": 512,
            "fps": 24,
            "target_duration": 60.0,
            "video_format": "mp4",
            "narrator_voice": "chatterbox-default-narrator",
            "visual_style": "cinematic",
            "aspect_ratio": "16:9",
        },
    }


def _scene_payload(number: int, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scene_id": f"s{number}",
        "scene_number": number,
        "title": f"Scene {number}",
        "description": "",
        "duration": 5.0,
        "narration": "Narration",
        "image_prompt": "An image",
        "animation_prompt": "",
        "negative_prompt": "",
        "character_references": [],
        "image_path": None,
        "generated_video_path": None,
        "narration_audio_path": None,
        "sfx_paths": [],
        "status": "draft",
        "timeline_start": 5.0 * (number - 1),
        "timeline_end": 5.0 * number,
    }
    payload.update(overrides)
    return payload


def _manifest(*scenes: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "scenes": list(scenes)}


def _codes(result: ValidationResult) -> set[str]:
    return {issue.code for issue in result.errors}


def test_valid_project_and_scene_manifest() -> None:
    project_result = validate_project_metadata(_project_payload())
    scene_result = validate_scene_manifest(_manifest(_scene_payload(1), _scene_payload(2)))

    assert project_result.valid
    assert scene_result.valid
    assert project_result.to_dict()["valid"] is True
    assert scene_result.warnings == []


def test_result_warnings_do_not_invalidate_manifest(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    result = validate_scene_manifest(
        _manifest(_scene_payload(1, image_path="assets/images/missing.png")),
        project_root=root,
    )

    assert result.valid
    assert any(issue.code == "scene.missing_asset" for issue in result.warnings)


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("project_id", "../escape", "project.invalid_id"),
        ("language", "123", "project.invalid_language"),
        ("status", "rendering", "project.invalid_status"),
        ("created_at", "yesterday", "project.invalid_timestamp"),
        ("updated_at", "2026-11-19T00:00:00", "project.invalid_timestamp"),
    ],
)
def test_invalid_project_metadata(
    field: str, value: str, expected_code: str
) -> None:
    payload = _project_payload()
    payload[field] = value

    result = validate_project_metadata(payload)

    assert not result.valid
    assert expected_code in _codes(result)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 0),
        ("height", -1),
        ("fps", 0),
        ("target_duration", "sixty"),
        ("aspect_ratio", "wide"),
        ("video_format", ""),
    ],
)
def test_invalid_project_settings(field: str, value: object) -> None:
    payload = _project_payload()
    payload["settings"][field] = value

    result = validate_project_metadata(payload)

    assert not result.valid
    assert "project.invalid_setting" in _codes(result)


def test_missing_project_manifest_fields_are_errors() -> None:
    payload = _project_payload()
    del payload["title"]
    del payload["settings"]

    result = validate_project_metadata(payload)

    assert not result.valid
    assert {"project.missing_field", "project.invalid_settings"} <= _codes(result)


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"duration": 0}, "scene.invalid_duration"),
        ({"duration": -2.5}, "scene.invalid_duration"),
        ({"title": ""}, "scene.invalid_title"),
        ({"status": "rendering"}, "scene.invalid_status"),
        ({"scene_number": 2}, "scene.invalid_order"),
        ({"scene_id": "../escape"}, "scene.invalid_id"),
        ({"narration": ["not", "text"]}, "scene.invalid_text"),
        ({"sfx_paths": "one.wav"}, "scene.invalid_path_list"),
    ],
)
def test_invalid_scene_fields(overrides: dict[str, Any], expected_code: str) -> None:
    result = validate_scene_manifest(_manifest(_scene_payload(1, **overrides)))

    assert not result.valid
    assert expected_code in _codes(result)


def test_duplicate_scene_ids_are_rejected() -> None:
    first = _scene_payload(1)
    duplicate = _scene_payload(2, scene_id=first["scene_id"])

    result = validate_scene_manifest(_manifest(first, duplicate))

    assert not result.valid
    assert "scene.duplicate_id" in _codes(result)


def test_missing_generated_assets_are_warnings_not_errors(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    (root / "assets" / "images").mkdir(parents=True)
    (root / "assets" / "images" / "one.png").write_bytes(b"image")
    scene = _scene_payload(
        1,
        status="draft",
        image_path="assets/images/one.png",
        generated_video_path="video/scenes/one.mp4",
        narration_audio_path="audio/narration/one.wav",
        sfx_paths=["audio/sfx/wind.wav"],
    )

    result = validate_scene_manifest(_manifest(scene), project_root=root)

    assert result.valid
    missing = [
        issue.message for issue in result.warnings if issue.code == "scene.missing_asset"
    ]
    assert len(missing) == 3
    assert all("missing file" in message for message in missing)


def test_asset_path_escaping_project_is_an_error(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    scene = _scene_payload(1, image_path="../outside.png")

    result = validate_scene_manifest(_manifest(scene), project_root=root)

    assert not result.valid
    assert "scene.invalid_asset_path" in _codes(result)


def test_absolute_asset_path_is_an_error_without_root() -> None:
    scene = _scene_payload(1, generated_video_path="C:/outside/video.mp4")

    result = validate_scene_manifest(_manifest(scene))

    assert not result.valid
    assert "scene.invalid_asset_path" in _codes(result)


def test_stale_timeline_is_a_warning() -> None:
    scene = _scene_payload(1, timeline_start=99.0, timeline_end=100.0)

    result = validate_scene_manifest(_manifest(scene))

    assert result.valid
    assert any(issue.code == "scene.timeline_stale" for issue in result.warnings)


def test_validate_project_directory(tmp_path: Path) -> None:
    projects = ProjectService(tmp_path)
    projects.create_project("demo", "Demo")
    SceneService(projects).ensure_manifest("demo")

    assert validate_project_directory(tmp_path / "demo").valid


def test_validate_project_directory_reports_missing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    result = validate_project_directory(root)

    assert not result.valid
    assert "manifest.missing_file" in _codes(result)


def test_validate_project_directory_reports_corrupt_json(tmp_path: Path) -> None:
    projects = ProjectService(tmp_path)
    projects.create_project("demo", "Demo")
    manifest_path = tmp_path / "demo" / "script" / "scenes.json"
    manifest_path.write_text("{not json", encoding="utf-8")

    result = validate_project_directory(tmp_path / "demo")

    assert not result.valid
    assert "manifest.invalid_json" in _codes(result)
