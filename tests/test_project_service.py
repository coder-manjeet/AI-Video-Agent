"""Tests for project creation, loading, updating and path safety."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from project_models import ProjectSettings
from project_paths import PathValidationError, is_iso8601_timestamp
from project_service import (
    PROJECT_DIRECTORIES,
    PROJECT_FILES,
    ProjectError,
    ProjectExistsError,
    ProjectNotFoundError,
    ProjectService,
)


def _payload(root: Path) -> dict[str, object]:
    return json.loads((root / "project.json").read_text(encoding="utf-8"))


def test_create_project_builds_directory_structure(tmp_path: Path) -> None:
    service = ProjectService(tmp_path)

    project = service.create_project("demo", "Demo Video")

    assert project.project_id == "demo"
    assert service.exists("demo")
    root = tmp_path / "demo"
    for relative in PROJECT_DIRECTORIES:
        assert (root / relative).is_dir(), relative
    for relative in PROJECT_FILES:
        assert (root / relative).is_file(), relative


def test_create_project_manifest_has_iso_timestamps_and_defaults(tmp_path: Path) -> None:
    service = ProjectService(tmp_path)
    service.create_project("demo", "Demo Video", description="A test")

    payload = _payload(tmp_path / "demo")

    assert set(payload) >= {
        "project_id",
        "title",
        "description",
        "status",
        "language",
        "created_at",
        "updated_at",
        "settings",
        "schema_version",
    }
    assert payload["description"] == "A test"
    assert payload["status"] == "active"
    assert isinstance(payload["created_at"], str)
    assert is_iso8601_timestamp(payload["created_at"])
    assert is_iso8601_timestamp(payload["updated_at"])
    assert payload["settings"]["width"] == 768
    assert payload["settings"]["fps"] == 24


def test_create_project_refuses_existing_manifest(tmp_path: Path) -> None:
    service = ProjectService(tmp_path)
    first = service.create_project("demo", "Demo")

    with pytest.raises(ProjectExistsError):
        service.create_project("demo", "Replacement")

    assert service.load_project("demo") == first


def test_create_project_refuses_nonempty_directory(tmp_path: Path) -> None:
    service = ProjectService(tmp_path)
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    note = orphan / "keep.txt"
    note.write_text("do not delete", encoding="utf-8")

    with pytest.raises(ProjectExistsError):
        service.create_project("orphan", "Orphan")

    assert note.read_text(encoding="utf-8") == "do not delete"


def test_load_project_roundtrip(tmp_path: Path) -> None:
    service = ProjectService(tmp_path)
    created = service.create_project(
        "demo",
        "Demo",
        settings=ProjectSettings(width=1920, height=1080, fps=30),
    )

    loaded = service.load_project("demo")

    assert loaded == created
    assert loaded.settings.width == 1920
    assert loaded.settings.fps == 30


def test_load_missing_project_raises(tmp_path: Path) -> None:
    with pytest.raises(ProjectNotFoundError):
        ProjectService(tmp_path).load_project("missing")


def test_load_corrupt_manifest_raises_project_error(tmp_path: Path) -> None:
    service = ProjectService(tmp_path)
    service.create_project("demo", "Demo")
    (tmp_path / "demo" / "project.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ProjectError, match="Could not read project manifest"):
        service.load_project("demo")


def test_update_project_preserves_unrelated_fields(tmp_path: Path) -> None:
    service = ProjectService(tmp_path)
    service.create_project("demo", "Demo", description="Original")

    updated = service.update_project(
        "demo",
        title="Renamed",
        settings={"width": 1920},
    )

    assert updated.title == "Renamed"
    assert updated.description == "Original"
    assert updated.settings.width == 1920
    assert updated.settings.height == 512
    assert updated.settings.video_format == "mp4"
    assert updated.language == "en"
    assert is_iso8601_timestamp(updated.updated_at)
    assert service.load_project("demo") == updated


def test_list_projects_only_returns_manifest_dirs(tmp_path: Path) -> None:
    service = ProjectService(tmp_path)
    service.create_project("beta", "Beta")
    service.create_project("alpha", "Alpha")
    (tmp_path / "stray").mkdir()

    assert service.list_projects() == ["alpha", "beta"]


@pytest.mark.parametrize(
    "project_id",
    ["../escape", "nested/name", "absolute/path", ".", "..", "", "bad id"],
)
def test_project_id_traversal_and_invalid_ids_are_rejected(
    tmp_path: Path, project_id: str
) -> None:
    service = ProjectService(tmp_path)

    with pytest.raises(PathValidationError):
        service.project_dir(project_id)
    with pytest.raises(PathValidationError):
        service.exists(project_id)
