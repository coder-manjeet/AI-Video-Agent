"""Character registry backed by ``assets/characters/characters.json``.

Character metadata is stored in one atomic project-level manifest, mirroring the
scene service.  Reference images live under ``assets/characters/{character_id}/``;
deleting metadata never deletes those files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from character_models import Character
from manifest_validation import (
    CHARACTERS_RELATIVE_PATH,
    ValidationResult,
    validate_character_manifest,
)
from project_paths import (
    PathValidationError,
    resolve_within_project,
    to_project_relative,
    validate_identifier,
    write_json_atomic,
)
from project_service import ProjectService

logger = logging.getLogger(__name__)

IMMUTABLE_FIELDS = frozenset({"character_id", "created_at"})


class CharacterServiceError(RuntimeError):
    """Base error for character registry failures."""


class CharacterManifestError(CharacterServiceError):
    """Raised when a character manifest is missing or corrupted."""


class CharacterNotFoundError(CharacterServiceError):
    """Raised when a character ID does not exist in the manifest."""


class DuplicateCharacterError(CharacterServiceError):
    """Raised when a caller supplies an existing character ID."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class CharacterService:
    """Create, update, delete and query project character identities."""

    def __init__(self, project_service: ProjectService) -> None:
        self.projects = project_service

    def manifest_path(self, project_id: str) -> Path:
        """Return the validated path of the character manifest."""
        return self.projects.project_dir(project_id) / CHARACTERS_RELATIVE_PATH

    def load_characters(self, project_id: str) -> list[Character]:
        """Load characters; a missing manifest means an empty registry."""
        path = self.manifest_path(project_id)
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CharacterManifestError(
                f"Could not read character manifest {path}: {exc}"
            ) from exc
        if isinstance(payload, Mapping):
            raw_characters = payload.get("characters", [])
        elif isinstance(payload, list):
            raw_characters = payload
        else:
            raise CharacterManifestError(
                f"Invalid character manifest root in {path}: expected an object or array"
            )
        if not isinstance(raw_characters, list):
            raise CharacterManifestError(
                f"Invalid character manifest in {path}: 'characters' must be an array"
            )
        try:
            return [Character.from_dict(entry) for entry in raw_characters]
        except (TypeError, ValueError) as exc:
            raise CharacterManifestError(
                f"Invalid character entry in {path}: {exc}"
            ) from exc

    def save_characters(
        self, project_id: str, characters: Sequence[Character]
    ) -> None:
        """Atomically persist the character manifest."""
        self.projects.project_dir(project_id)
        result = validate_character_manifest(list(characters))
        if not result.valid:
            raise CharacterServiceError(
                "Invalid character manifest: "
                + "; ".join(issue.message for issue in result.errors)
            )
        path = self.manifest_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            path,
            {"schema_version": 1, "characters": [item.to_dict() for item in characters]},
        )

    def list_characters(self, project_id: str) -> list[Character]:
        """Return all characters in manifest order."""
        return self.load_characters(project_id)

    def get_character(self, project_id: str, character_id: str) -> Character:
        """Return one character by ID."""
        for character in self.load_characters(project_id):
            if character.character_id == character_id:
                return character
        raise CharacterNotFoundError(f"Character not found: {character_id}")

    def create_character(
        self,
        project_id: str,
        *,
        name: str,
        character_id: str | None = None,
        **fields: Any,
    ) -> Character:
        """Create and persist a new character identity."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        characters = self.load_characters(project_id)
        if character_id is None:
            slug = "".join(char.lower() if char.isalnum() else "-" for char in name)
            candidate = "-".join(part for part in slug.split("-") if part)[:32] or "character"
            character_id = f"{candidate}-{uuid.uuid4().hex[:8]}"
        validate_identifier(character_id, "character_id")
        if any(item.character_id == character_id for item in characters):
            raise DuplicateCharacterError(
                f"Character ID already exists: {character_id}"
            )
        now = _utc_now()
        payload = {"character_id": character_id, "name": name, **fields}
        payload.setdefault("created_at", now)
        payload.setdefault("updated_at", now)
        try:
            character = Character.from_dict(payload)
        except (TypeError, ValueError) as exc:
            raise CharacterServiceError(f"Invalid character data: {exc}") from exc
        characters.append(character)
        self.save_characters(project_id, characters)
        logger.info("Created character %s in project %s", character_id, project_id)
        return character

    def update_character(
        self,
        project_id: str,
        character_id: str,
        changes: Mapping[str, Any],
    ) -> Character:
        """Update mutable character fields without touching other characters."""
        conflicts = sorted(IMMUTABLE_FIELDS.intersection(changes))
        if conflicts:
            raise ValueError(
                "Fields cannot be updated directly: " + ", ".join(conflicts)
            )
        characters = self.load_characters(project_id)
        index = next(
            (i for i, item in enumerate(characters) if item.character_id == character_id),
            None,
        )
        if index is None:
            raise CharacterNotFoundError(f"Character not found: {character_id}")
        updates = dict(changes)
        updates["updated_at"] = _utc_now()
        merged = dict(characters[index].to_dict())
        merged.update(updates)
        try:
            characters[index] = Character.from_dict(merged)
        except (TypeError, ValueError) as exc:
            raise CharacterServiceError(f"Invalid character data: {exc}") from exc
        self.save_characters(project_id, characters)
        return characters[index]

    def delete_character(self, project_id: str, character_id: str) -> None:
        """Delete character metadata only; reference image files are untouched."""
        characters = self.load_characters(project_id)
        remaining = [item for item in characters if item.character_id != character_id]
        if len(remaining) == len(characters):
            raise CharacterNotFoundError(f"Character not found: {character_id}")
        self.save_characters(project_id, remaining)
        logger.info("Deleted metadata for character %s in %s", character_id, project_id)

    def register_reference_image(
        self, project_id: str, character_id: str, path: str | Path
    ) -> Character:
        """Register a reference image, copying external files into the project."""
        character = self.get_character(project_id, character_id)
        project_root = self.projects.project_dir(project_id)
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Reference image not found: {source}")

        try:
            relative = to_project_relative(source, project_root)
        except PathValidationError:
            destination_dir = project_root / "assets" / "characters" / character_id
            destination_dir.mkdir(parents=True, exist_ok=True)
            suffix = source.suffix.lower() or ".png"
            stem = "".join(
                char if char.isalnum() or char in "-_" else "-" for char in source.stem
            )[:32] or "reference"
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:8]
            destination = destination_dir / f"{stem}-{digest}{suffix}"
            if not destination.is_file():
                shutil.copy2(source, destination)
            relative = to_project_relative(destination, project_root)

        references = list(character.reference_images)
        if relative not in references:
            references.append(relative)
        return self.update_character(
            project_id, character_id, {"reference_images": references}
        )

    def get_references(self, project_id: str, character_id: str) -> list[str]:
        """Return project-relative paths of a character's reference images."""
        return list(self.get_character(project_id, character_id).reference_images)

    def missing_characters(
        self, project_id: str, character_ids: Sequence[str]
    ) -> list[str]:
        """Return IDs that are not present in the character registry."""
        known = {item.character_id for item in self.load_characters(project_id)}
        return [item for item in character_ids if item not in known]

    def validate_characters(self, project_id: str) -> ValidationResult:
        """Validate the character manifest against its reference files."""
        path = self.manifest_path(project_id)
        if not path.is_file():
            return ValidationResult()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result = ValidationResult()
            result.add_error(
                "manifest.invalid_json",
                f"Could not read {path.name} as JSON: {exc}",
                path.name,
            )
            return result
        return validate_character_manifest(
            payload, project_root=self.projects.project_dir(project_id)
        )

    def resolve_reference_paths(
        self, project_id: str, character_ids: Sequence[str]
    ) -> dict[str, list[Path]]:
        """Return absolute paths of existing reference images by character ID."""
        root = self.projects.project_dir(project_id)
        resolved: dict[str, list[Path]] = {}
        for character_id in character_ids:
            try:
                character = self.get_character(project_id, character_id)
            except CharacterNotFoundError:
                continue
            paths: list[Path] = []
            for reference in character.reference_images:
                try:
                    path = resolve_within_project(root, reference, "reference_images")
                except PathValidationError:
                    continue
                if path.is_file():
                    paths.append(path)
            resolved[character_id] = paths
        return resolved
