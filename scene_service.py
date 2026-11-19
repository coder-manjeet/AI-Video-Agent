"""Scene management backed by ``script/scenes.json``.

The manifest is the single source of truth for scene order and production
state.  Scene numbers are always kept sequential, updates preserve unrelated
fields, and deleting a scene never deletes generated assets.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from manifest_validation import SCENES_RELATIVE_PATH, validate_scene_manifest
from project_models import DEFAULT_SCENE_STATUS, SCENE_STATUSES, SCHEMA_VERSION, Scene
from project_paths import to_project_relative, validate_identifier, write_json_atomic
from project_service import ProjectService

logger = logging.getLogger(__name__)

ASSET_FIELDS: dict[str, str] = {
    "image": "image_path",
    "thumbnail": "thumbnail_path",
    "generated_video": "generated_video_path",
    "narration_audio": "narration_audio_path",
    "sfx": "sfx_paths",
}
IMMUTABLE_FIELDS = frozenset(
    {"scene_id", "scene_number", "timeline_start", "timeline_end", "schema_version"}
)
NON_NULLABLE_FIELDS = frozenset(
    {
        "title",
        "description",
        "duration",
        "narration",
        "image_prompt",
        "animation_prompt",
        "negative_prompt",
        "status",
        "generation_provider",
    }
)


class SceneServiceError(RuntimeError):
    """Base error for scene service failures."""


class SceneManifestError(SceneServiceError):
    """Raised when a scenes.json file is missing or corrupted."""


class SceneNotFoundError(SceneServiceError):
    """Raised when a scene ID does not exist in the manifest."""


class DuplicateSceneError(SceneServiceError):
    """Raised when a caller supplies a scene ID that is already in use."""


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:32] or "scene"


def _renumber(scenes: list[Scene]) -> list[Scene]:
    for number, scene in enumerate(scenes, start=1):
        scene.scene_number = number
    return scenes


def _require_valid(scenes: list[Scene]) -> None:
    result = validate_scene_manifest(scenes)
    if not result.valid:
        raise ValueError(
            "Invalid scene manifest: " + "; ".join(issue.message for issue in result.errors)
        )


class SceneService:
    """Create, update, delete and reorder scenes for a project."""

    def __init__(self, project_service: ProjectService) -> None:
        self.projects = project_service

    def manifest_path(self, project_id: str) -> Path:
        """Return the validated absolute path of ``script/scenes.json``."""
        return self.projects.project_dir(project_id) / SCENES_RELATIVE_PATH

    def load_scenes(self, project_id: str) -> list[Scene]:
        """Load scenes; a missing manifest means an empty project."""
        path = self.manifest_path(project_id)
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneManifestError(f"Could not read scene manifest {path}: {exc}") from exc

        if isinstance(payload, Mapping):
            raw_scenes = payload.get("scenes", [])
        elif isinstance(payload, list):
            raw_scenes = payload
        else:
            raise SceneManifestError(f"Scene manifest must be an object or array: {path}")
        if not isinstance(raw_scenes, list):
            raise SceneManifestError(f"scenes must be an array in {path}")

        scenes: list[Scene] = []
        for index, item in enumerate(raw_scenes):
            if not isinstance(item, Mapping):
                raise SceneManifestError(f"Scene entry {index} in {path} must be an object")
            try:
                scenes.append(Scene.from_dict(item))
            except ValueError as exc:
                raise SceneManifestError(f"Invalid scene entry {index} in {path}: {exc}") from exc
        return scenes

    def save_scenes(self, project_id: str, scenes: Sequence[Scene]) -> None:
        """Atomically save a full scene manifest, creating parent directories."""
        if not self.projects.exists(project_id):
            raise SceneServiceError(f"Project not found: {project_id}")
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "scenes": [scene.to_dict() for scene in scenes],
        }
        write_json_atomic(self.manifest_path(project_id), payload)

    def ensure_manifest(self, project_id: str) -> list[Scene]:
        """Create an empty manifest only when one does not already exist."""
        if self.manifest_path(project_id).is_file():
            return self.load_scenes(project_id)
        scenes: list[Scene] = []
        self.save_scenes(project_id, scenes)
        return scenes

    def list_scenes(self, project_id: str) -> list[Scene]:
        """Return scenes in manifest order."""
        return self.load_scenes(project_id)

    def get_scene(self, project_id: str, scene_id: str) -> Scene:
        """Return one scene by ID."""
        scenes = self.load_scenes(project_id)
        for scene in scenes:
            if scene.scene_id == scene_id:
                return scene
        raise SceneNotFoundError(f"Scene not found: {scene_id}")

    def _new_scene_id(self, existing: list[Scene], title: str, scene_id: str | None) -> str:
        taken = {scene.scene_id for scene in existing}
        if scene_id is not None:
            validate_identifier(scene_id, "scene_id")
            if scene_id in taken:
                raise DuplicateSceneError(f"Scene ID already exists: {scene_id}")
            return scene_id
        while True:
            candidate = f"{_slug(title)}-{uuid.uuid4().hex[:8]}"
            if candidate not in taken:
                return candidate

    def add_scene(
        self,
        project_id: str,
        *,
        title: str,
        duration: float = 5.0,
        scene_id: str | None = None,
        description: str = "",
        narration: str = "",
        image_prompt: str = "",
        animation_prompt: str = "",
        negative_prompt: str = "",
        character_references: Sequence[str] | None = None,
        status: str = DEFAULT_SCENE_STATUS,
    ) -> Scene:
        """Append a new scene and renumber the manifest."""
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        if status not in SCENE_STATUSES:
            raise ValueError(f"status must be one of {sorted(SCENE_STATUSES)}")
        scenes = self.load_scenes(project_id)
        scene = Scene(
            scene_id=self._new_scene_id(scenes, title, scene_id),
            scene_number=len(scenes) + 1,
            title=title,
            description=description,
            duration=duration,
            narration=narration,
            image_prompt=image_prompt,
            animation_prompt=animation_prompt,
            negative_prompt=negative_prompt,
            character_references=list(character_references or []),
            status=status,
        )
        scenes.append(scene)
        _renumber(scenes)
        _require_valid(scenes)
        self.save_scenes(project_id, scenes)
        logger.info("Added scene %s to project %s", scene.scene_id, project_id)
        return scene

    def update_scene(
        self, project_id: str, scene_id: str, changes: Mapping[str, Any]
    ) -> Scene:
        """Update supplied fields while preserving all unrelated data."""
        updates = dict(changes)
        conflicts = IMMUTABLE_FIELDS.intersection(updates)
        if conflicts:
            raise ValueError(
                "Fields cannot be updated directly: " + ", ".join(sorted(conflicts))
            )
        for key in NON_NULLABLE_FIELDS:
            if key in updates and updates[key] is None:
                raise ValueError(f"{key} cannot be null")

        scenes = self.load_scenes(project_id)
        index = self._index(scenes, scene_id)
        merged = scenes[index].to_dict()
        merged.update(updates)
        try:
            updated = Scene.from_dict(merged)
        except ValueError as exc:
            raise SceneServiceError(f"Invalid scene data: {exc}") from exc
        scenes[index] = updated
        _require_valid(scenes)
        self.save_scenes(project_id, scenes)
        return updated

    def delete_scene(self, project_id: str, scene_id: str) -> Scene:
        """Remove a scene and renumber; referenced asset files are left alone."""
        scenes = self.load_scenes(project_id)
        index = self._index(scenes, scene_id)
        removed = scenes.pop(index)
        _renumber(scenes)
        _require_valid(scenes)
        self.save_scenes(project_id, scenes)
        logger.info("Deleted scene %s from project %s", scene_id, project_id)
        return removed

    def reorder_scenes(self, project_id: str, scene_order: Sequence[str]) -> list[Scene]:
        """Replace the scene order with *scene_order* and renumber from 1."""
        scenes = self.load_scenes(project_id)
        current = [scene.scene_id for scene in scenes]
        requested = list(scene_order)
        if len(requested) != len(current) or set(requested) != set(current):
            raise ValueError("scene_order must contain every scene ID exactly once")
        by_id = {scene.scene_id: scene for scene in scenes}
        ordered = [by_id[scene_id] for scene_id in requested]
        _renumber(ordered)
        _require_valid(ordered)
        self.save_scenes(project_id, ordered)
        return ordered

    def move_scene(self, project_id: str, scene_id: str, new_index: int) -> list[Scene]:
        """Move one scene to a zero-based index and renumber the manifest."""
        scenes = self.load_scenes(project_id)
        index = self._index(scenes, scene_id)
        if not 0 <= new_index < len(scenes):
            raise ValueError(f"new_index must be in [0, {len(scenes) - 1}]")
        order = [scene.scene_id for scene in scenes]
        order.pop(index)
        order.insert(new_index, scene_id)
        return self.reorder_scenes(project_id, order)

    def update_scene_status(self, project_id: str, scene_id: str, status: str) -> Scene:
        """Set a scene status after validating it against the allowed values."""
        if status not in SCENE_STATUSES:
            raise ValueError(f"status must be one of {sorted(SCENE_STATUSES)}")
        return self.update_scene(project_id, scene_id, {"status": status})

    def register_asset(
        self, project_id: str, scene_id: str, asset_kind: str, path: str | Path
    ) -> Scene:
        """Record a generated asset using a safe project-relative path."""
        if asset_kind not in ASSET_FIELDS:
            raise ValueError(
                f"asset_kind must be one of {sorted(ASSET_FIELDS)}"
            )
        relative = to_project_relative(path, self.projects.project_dir(project_id))
        field = ASSET_FIELDS[asset_kind]
        if field == "sfx_paths":
            current = self.get_scene(project_id, scene_id)
            paths = list(current.sfx_paths)
            if relative not in paths:
                paths.append(relative)
            return self.update_scene(project_id, scene_id, {field: paths})
        return self.update_scene(project_id, scene_id, {field: relative})

    def apply_timeline(
        self, project_id: str, positions: Mapping[str, tuple[float, float]]
    ) -> list[Scene]:
        """Store calculated timeline positions without changing scene order."""
        scenes = self.load_scenes(project_id)
        known = {scene.scene_id for scene in scenes}
        unknown = set(positions) - known
        if unknown:
            raise SceneNotFoundError(f"Unknown scene IDs in timeline: {sorted(unknown)}")
        for scene in scenes:
            if scene.scene_id in positions:
                start, end = positions[scene.scene_id]
                scene.timeline_start = start
                scene.timeline_end = end
        _require_valid(scenes)
        self.save_scenes(project_id, scenes)
        return scenes

    @staticmethod
    def _index(scenes: list[Scene], scene_id: str) -> int:
        for index, scene in enumerate(scenes):
            if scene.scene_id == scene_id:
                return index
        raise SceneNotFoundError(f"Scene not found: {scene_id}")
