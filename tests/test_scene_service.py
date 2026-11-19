"""Tests for scene CRUD, manifest handling, ordering and asset registration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from project_paths import PathValidationError
from project_service import ProjectService
from scene_service import (
    DuplicateSceneError,
    SceneManifestError,
    SceneNotFoundError,
    SceneService,
    SceneServiceError,
)


def _services(tmp_path: Path) -> tuple[ProjectService, SceneService]:
    projects = ProjectService(tmp_path)
    projects.create_project("demo", "Demo")
    return projects, SceneService(projects)


def _manifest(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "demo" / "script" / "scenes.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_add_scene_creates_manifest_and_defaults(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)

    scene = scenes.add_scene("demo", title="Opening", duration=8.0)

    assert scene.scene_id.startswith("opening-")
    assert scene.scene_number == 1
    assert scene.duration == 8.0
    assert scene.status == "draft"
    assert scene.image_path is None
    assert scene.generated_video_path is None
    payload = _manifest(tmp_path)
    assert payload["schema_version"] == 1
    assert len(payload["scenes"]) == 1
    assert payload["scenes"][0]["scene_id"] == scene.scene_id


def test_add_scene_rejects_duplicate_explicit_id(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    scenes.add_scene("demo", title="One", scene_id="fixed-id")

    with pytest.raises(DuplicateSceneError):
        scenes.add_scene("demo", title="Two", scene_id="fixed-id")

    with pytest.raises(PathValidationError):
        scenes.add_scene("demo", title="Bad", scene_id="../escape")


def test_generated_scene_ids_are_unique(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)

    first = scenes.add_scene("demo", title="Same title")
    second = scenes.add_scene("demo", title="Same title")

    assert first.scene_id != second.scene_id
    assert [scene.scene_number for scene in scenes.list_scenes("demo")] == [1, 2]


def test_update_scene_preserves_unrelated_fields(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    original = scenes.add_scene(
        "demo",
        title="Original",
        duration=8.0,
        narration="Keep this",
        image_prompt="Keep this too",
    )
    scenes.update_scene(
        "demo", original.scene_id, {"image_path": "assets/images/original.png"}
    )

    updated = scenes.update_scene(
        "demo", original.scene_id, {"title": "Renamed", "duration": 9.5}
    )

    assert updated.title == "Renamed"
    assert updated.duration == 9.5
    assert updated.narration == "Keep this"
    assert updated.image_prompt == "Keep this too"
    assert updated.image_path == "assets/images/original.png"
    assert scenes.get_scene("demo", original.scene_id) == updated


def test_update_scene_rejects_immutable_fields(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    scene = scenes.add_scene("demo", title="One")

    with pytest.raises(ValueError, match="cannot be updated directly"):
        scenes.update_scene("demo", scene.scene_id, {"scene_number": 7})

    with pytest.raises(ValueError, match="cannot be null"):
        scenes.update_scene("demo", scene.scene_id, {"title": None})


def test_delete_scene_renumbers_and_never_deletes_assets(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    first = scenes.add_scene("demo", title="One", scene_id="one")
    scenes.add_scene("demo", title="Two", scene_id="two")
    scenes.add_scene("demo", title="Three", scene_id="three")
    image = tmp_path / "demo" / "assets" / "images" / "one.png"
    image.write_bytes(b"fake image")
    scenes.register_asset("demo", "one", "image", image)

    removed = scenes.delete_scene("demo", first.scene_id)

    assert removed.scene_id == "one"
    assert image.is_file()
    assert [
        (scene.scene_id, scene.scene_number)
        for scene in scenes.list_scenes("demo")
    ] == [("two", 1), ("three", 2)]


def test_reorder_scenes_renumbers_and_preserves_data(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    for title in ("One", "Two", "Three"):
        scenes.add_scene("demo", title=title, scene_id=title.lower(), duration=5.0)
    scenes.update_scene("demo", "three", {"narration": "keep me"})

    ordered = scenes.reorder_scenes("demo", ["three", "one", "two"])

    assert [scene.scene_id for scene in ordered] == ["three", "one", "two"]
    assert [scene.scene_number for scene in ordered] == [1, 2, 3]
    assert scenes.get_scene("demo", "three").narration == "keep me"


def test_reorder_scenes_rejects_invalid_order(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    scenes.add_scene("demo", title="One", scene_id="one")
    scenes.add_scene("demo", title="Two", scene_id="two")

    with pytest.raises(ValueError, match="every scene ID exactly once"):
        scenes.reorder_scenes("demo", ["one", "one"])
    with pytest.raises(ValueError):
        scenes.reorder_scenes("demo", ["one"])


def test_move_scene_updates_order(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    for title in ("One", "Two", "Three"):
        scenes.add_scene("demo", title=title, scene_id=title.lower())

    ordered = scenes.move_scene("demo", "three", 0)

    assert [scene.scene_id for scene in ordered] == ["three", "one", "two"]
    assert scenes.list_scenes("demo") == ordered


def test_update_scene_status(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    scene = scenes.add_scene("demo", title="One")

    updated = scenes.update_scene_status("demo", scene.scene_id, "video_generated")

    assert updated.status == "video_generated"
    with pytest.raises(ValueError):
        scenes.update_scene_status("demo", scene.scene_id, "rendering")


def test_register_asset_paths_are_relative_and_stored_per_kind(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    scene = scenes.add_scene("demo", title="One", scene_id="one")
    root = tmp_path / "demo"
    image = root / "assets" / "images" / "one.png"
    video = root / "video" / "scenes" / "one.mp4"
    audio = root / "audio" / "narration" / "one.wav"
    sfx = root / "audio" / "sfx" / "one.wav"
    for path in (image, video, audio, sfx):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    for kind, path in (
        ("image", image),
        ("generated_video", video),
        ("narration_audio", audio),
        ("sfx", sfx),
    ):
        scenes.register_asset("demo", scene.scene_id, kind, path)

    updated = scenes.get_scene("demo", scene.scene_id)
    assert updated.image_path == "assets/images/one.png"
    assert updated.generated_video_path == "video/scenes/one.mp4"
    assert updated.narration_audio_path == "audio/narration/one.wav"
    assert updated.sfx_paths == ["audio/sfx/one.wav"]


def test_register_asset_rejects_paths_outside_project(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    scene = scenes.add_scene("demo", title="One")

    with pytest.raises(PathValidationError):
        scenes.register_asset("demo", scene.scene_id, "image", "../outside.png")
    with pytest.raises(PathValidationError):
        scenes.register_asset("demo", scene.scene_id, "image", tmp_path / "outside.png")
    with pytest.raises(ValueError):
        scenes.register_asset("demo", scene.scene_id, "unknown", "assets/images/a.png")


def test_manifest_serialization_and_deserialization(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    added = scenes.add_scene(
        "demo",
        title="Serialization",
        scene_id="serial",
        duration=8.25,
        narration="Hello",
        description="A scene",
    )

    loaded = scenes.list_scenes("demo")

    assert loaded == [added]
    assert loaded[0].to_dict() == added.to_dict()
    assert set(_manifest(tmp_path)) == {"schema_version", "scenes"}


def test_missing_manifest_returns_empty_list(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)

    assert scenes.list_scenes("demo") == []
    assert scenes.ensure_manifest("demo") == []
    assert scenes.manifest_path("demo").is_file()


def test_corrupt_manifest_raises_scene_manifest_error(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)
    manifest = scenes.manifest_path("demo")
    manifest.write_text("{broken", encoding="utf-8")

    with pytest.raises(SceneManifestError):
        scenes.load_scenes("demo")


def test_scene_not_found_raises(tmp_path: Path) -> None:
    _, scenes = _services(tmp_path)

    with pytest.raises(SceneNotFoundError):
        scenes.get_scene("demo", "missing")
    with pytest.raises(SceneNotFoundError):
        scenes.delete_scene("demo", "missing")


def test_scene_service_requires_existing_project(tmp_path: Path) -> None:
    projects = ProjectService(tmp_path)
    scenes = SceneService(projects)

    with pytest.raises(SceneServiceError, match="Project not found"):
        scenes.save_scenes("missing", [])
