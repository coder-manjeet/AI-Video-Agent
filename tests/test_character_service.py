"""Tests for the project-level character registry service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from character_service import (
    CharacterNotFoundError,
    CharacterService,
    DuplicateCharacterError,
)
from project_paths import PathValidationError
from project_service import ProjectService


def _harness(tmp_path: Path) -> tuple[ProjectService, CharacterService]:
    projects = ProjectService(tmp_path)
    projects.create_project("demo", "Demo")
    return projects, CharacterService(projects)


def _manifest(projects: ProjectService) -> dict[str, object]:
    path = projects.project_dir("demo") / "assets" / "characters" / "characters.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_create_character_persists_and_lists(tmp_path: Path) -> None:
    projects, characters = _harness(tmp_path)

    created = characters.create_character(
        "demo",
        name="Aria",
        character_id="aria",
        description="A cartographer",
        visual_prompt="soft portrait",
        future_field="keep",
    )

    assert created.character_id == "aria"
    assert created.created_at and created.updated_at
    assert [item.character_id for item in characters.list_characters("demo")] == ["aria"]
    assert characters.get_character("demo", "aria") == created
    manifest = _manifest(projects)
    assert manifest["schema_version"] == 1
    assert manifest["characters"][0]["future_field"] == "keep"
    reloaded = CharacterService(projects).get_character("demo", "aria")
    assert reloaded.extra["future_field"] == "keep"


def test_create_character_generates_stable_id_from_name(tmp_path: Path) -> None:
    _projects, characters = _harness(tmp_path)

    created = characters.create_character("demo", name="Boman The Guard")

    assert created.character_id.startswith("boman-the-guard-")
    assert len(created.character_id.split("-")[-1]) == 8


def test_duplicate_character_id_is_rejected(tmp_path: Path) -> None:
    _projects, characters = _harness(tmp_path)
    characters.create_character("demo", name="Aria", character_id="aria")

    with pytest.raises(DuplicateCharacterError, match="already exists"):
        characters.create_character("demo", name="Other", character_id="aria")


@pytest.mark.parametrize("character_id", ["../escape", "bad id", "", "a/b"])
def test_invalid_character_id_is_rejected(tmp_path: Path, character_id: str) -> None:
    _projects, characters = _harness(tmp_path)

    with pytest.raises(PathValidationError):
        characters.create_character("demo", name="Aria", character_id=character_id)


def test_update_character_is_scoped_and_rejects_immutable_fields(tmp_path: Path) -> None:
    _projects, characters = _harness(tmp_path)
    characters.create_character("demo", name="Aria", character_id="aria")
    characters.create_character("demo", name="Boman", character_id="boman")

    updated = characters.update_character(
        "demo", "aria", {"clothing": "rain cloak", "new_note": "saved"}
    )

    assert updated.clothing == "rain cloak"
    assert updated.extra["new_note"] == "saved"
    assert characters.get_character("demo", "boman").clothing == ""
    with pytest.raises(ValueError, match="cannot be updated directly"):
        characters.update_character("demo", "aria", {"character_id": "other"})
    with pytest.raises(CharacterNotFoundError):
        characters.update_character("demo", "missing", {"name": "X"})


def test_reference_registration_copies_file_and_is_idempotent(tmp_path: Path) -> None:
    projects, characters = _harness(tmp_path)
    characters.create_character("demo", name="Aria", character_id="aria")
    source = tmp_path / "portrait.png"
    source.write_bytes(b"fake png")

    first = characters.register_reference_image("demo", "aria", source)
    second = characters.register_reference_image("demo", "aria", source)

    assert first.reference_images == second.reference_images
    assert len(second.reference_images) == 1
    relative = second.reference_images[0]
    assert relative.startswith("assets/characters/aria/")
    assert (projects.project_dir("demo") / relative).is_file()
    assert characters.get_references("demo", "aria") == [relative]


def test_delete_character_metadata_never_deletes_reference_images(
    tmp_path: Path,
) -> None:
    projects, characters = _harness(tmp_path)
    characters.create_character("demo", name="Aria", character_id="aria")
    source = tmp_path / "portrait.png"
    source.write_bytes(b"fake png")
    registered = characters.register_reference_image("demo", "aria", source)
    reference = projects.project_dir("demo") / registered.reference_images[0]
    assert reference.is_file()

    characters.delete_character("demo", "aria")

    assert reference.is_file()
    assert characters.list_characters("demo") == []
    with pytest.raises(CharacterNotFoundError):
        characters.delete_character("demo", "aria")


def test_missing_characters_and_resolve_reference_paths(tmp_path: Path) -> None:
    projects, characters = _harness(tmp_path)
    characters.create_character("demo", name="Aria", character_id="aria")
    source = tmp_path / "portrait.png"
    source.write_bytes(b"fake png")
    registered = characters.register_reference_image("demo", "aria", source)

    assert characters.missing_characters("demo", ["aria", "ghost"]) == ["ghost"]
    resolved = characters.resolve_reference_paths("demo", ["aria", "ghost"])
    assert list(resolved) == ["aria"]
    assert resolved["aria"] == [projects.project_dir("demo") / registered.reference_images[0]]

    reference = projects.project_dir("demo") / registered.reference_images[0]
    reference.unlink()
    assert characters.resolve_reference_paths("demo", ["aria"]) == {"aria": []}


def test_validate_characters_reports_missing_reference_file(tmp_path: Path) -> None:
    projects, characters = _harness(tmp_path)
    characters.create_character("demo", name="Aria", character_id="aria")
    manifest_path = projects.project_dir("demo") / "assets" / "characters" / "characters.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["characters"][0]["reference_images"] = ["assets/characters/aria/missing.png"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = characters.validate_characters("demo")

    assert any(issue.code == "scene.missing_asset" for issue in result.warnings)
    assert any("missing.png" in issue.message for issue in result.warnings)
