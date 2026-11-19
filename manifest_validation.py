"""Structured validation for project manifests, scenes and referenced assets.

Validation never mutates manifests.  Problems that make a manifest invalid are
reported as errors; missing generated assets are reported as warnings so that
draft scenes remain valid.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from character_models import (
    CHARACTER_OVERRIDE_FIELDS,
    VISUAL_IDENTITY_FIELDS,
    Character,
)
from project_models import (
    PROJECT_STATUSES,
    SCENE_STATUSES,
    Project,
    Scene,
)
from project_paths import (
    PathValidationError,
    is_iso8601_timestamp,
    is_valid_aspect_ratio,
    is_valid_language,
    resolve_within_project,
    validate_identifier,
)

PROJECT_FILENAME = "project.json"
SCENES_RELATIVE_PATH = Path("script") / "scenes.json"
CHARACTERS_RELATIVE_PATH = Path("assets") / "characters" / "characters.json"

ASSET_FIELDS = (
    "image_path",
    "thumbnail_path",
    "generated_video_path",
    "narration_audio_path",
)
PATH_LIST_FIELDS = ("character_references", "sfx_paths")
TEXT_FIELDS = (
    "description",
    "narration",
    "image_prompt",
    "animation_prompt",
    "negative_prompt",
)


@dataclass(frozen=True)
class ValidationIssue:
    """A single structured problem found during validation."""

    code: str
    message: str
    location: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
        }


@dataclass
class ValidationResult:
    """Errors make a manifest invalid; warnings do not."""

    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def add_error(self, code: str, message: str, location: str = "") -> None:
        self.errors.append(ValidationIssue(code, message, location))

    def add_warning(self, code: str, message: str, location: str = "") -> None:
        self.warnings.append(ValidationIssue(code, message, location))

    def merge(self, other: ValidationResult) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_text_fields(
    raw: Mapping[str, Any], location: str, result: ValidationResult
) -> None:
    for key in TEXT_FIELDS:
        if key in raw and raw[key] is not None and not isinstance(raw[key], str):
            result.add_error(
                "scene.invalid_text",
                f"{key} must be a string when present",
                location,
            )


def _validate_asset(
    value: Any,
    key: str,
    location: str,
    project_root: Path | None,
    result: ValidationResult,
) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        result.add_error(
            "scene.invalid_asset_path",
            f"{key} must be a non-empty project-relative path",
            location,
        )
        return
    if project_root is None:
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            result.add_error(
                "scene.invalid_asset_path",
                f"{key} must be relative to the project directory: {value!r}",
                location,
            )
        return
    try:
        resolved = resolve_within_project(project_root, value, f"{location}.{key}")
    except PathValidationError as exc:
        result.add_error("scene.invalid_asset_path", str(exc), location)
        return
    if not resolved.is_file():
        result.add_warning(
            "scene.missing_asset",
            f"{key} references a missing file: {value}",
            location,
        )


def validate_project_metadata(data: Project | Mapping[str, Any]) -> ValidationResult:
    """Validate project-level metadata and render settings."""
    result = ValidationResult()
    raw = data.to_dict() if isinstance(data, Project) else dict(data)

    for key in ("project_id", "title", "status", "language", "created_at", "updated_at"):
        if key not in raw:
            result.add_error("project.missing_field", f"Missing required field: {key}", key)
    for key in ("project_id", "title"):
        if key in raw and (not isinstance(raw[key], str) or not raw[key].strip()):
            result.add_error(
                "project.invalid_field", f"{key} must be a non-empty string", key
            )

    project_id = raw.get("project_id")
    if isinstance(project_id, str) and project_id.strip():
        try:
            validate_identifier(project_id, "project_id")
        except PathValidationError as exc:
            result.add_error("project.invalid_id", str(exc), "project_id")

    status = raw.get("status")
    if status is not None and status not in PROJECT_STATUSES:
        result.add_error(
            "project.invalid_status",
            f"status must be one of {sorted(PROJECT_STATUSES)}",
            "status",
        )

    language = raw.get("language")
    if language is not None and not is_valid_language(language):
        result.add_error(
            "project.invalid_language",
            f"language must be a language code such as 'en' or 'en-US': {language!r}",
            "language",
        )

    for key in ("created_at", "updated_at"):
        if key in raw and not is_iso8601_timestamp(raw[key]):
            result.add_error(
                "project.invalid_timestamp",
                f"{key} must be an ISO 8601 timestamp with a timezone",
                key,
            )

    settings = raw.get("settings")
    if not isinstance(settings, Mapping):
        result.add_error("project.invalid_settings", "settings must be an object", "settings")
    else:
        for key in ("width", "height", "fps"):
            value = settings.get(key)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                result.add_error(
                    "project.invalid_setting",
                    f"settings.{key} must be a positive integer",
                    f"settings.{key}",
                )
        target_duration = settings.get("target_duration")
        if (
            not isinstance(target_duration, (int, float))
            or isinstance(target_duration, bool)
            or not math.isfinite(float(target_duration))
            or target_duration < 0
        ):
            result.add_error(
                "project.invalid_setting",
                "settings.target_duration must be a non-negative number",
                "settings.target_duration",
            )
        for key in ("video_format", "narrator_voice", "visual_style"):
            value = settings.get(key)
            if not isinstance(value, str) or not value.strip():
                result.add_error(
                    "project.invalid_setting",
                    f"settings.{key} must be a non-empty string",
                    f"settings.{key}",
                )
        if not is_valid_aspect_ratio(settings.get("aspect_ratio")):
            result.add_error(
                "project.invalid_setting",
                "settings.aspect_ratio must use the W:H format with positive integers",
                "settings.aspect_ratio",
            )

    identity = raw.get("visual_identity")
    if identity is not None:
        if not isinstance(identity, Mapping):
            result.add_error(
                "project.invalid_visual_identity",
                "visual_identity must be an object or null",
                "visual_identity",
            )
        else:
            for key in VISUAL_IDENTITY_FIELDS:
                value = identity.get(key)
                if value is not None and not isinstance(value, str):
                    result.add_error(
                        "project.invalid_visual_identity",
                        f"visual_identity.{key} must be a string",
                        f"visual_identity.{key}",
                    )
    return result


def validate_scene_manifest(
    data: Mapping[str, Any] | Sequence[Scene | Mapping[str, Any]],
    *,
    project_root: str | Path | None = None,
) -> ValidationResult:
    """Validate scene order, fields and (optionally) referenced asset files."""
    result = ValidationResult()
    root = Path(project_root).expanduser().resolve() if project_root is not None else None

    if isinstance(data, Mapping):
        schema_version = data.get("schema_version")
        if schema_version is not None and (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version <= 0
        ):
            result.add_error(
                "manifest.invalid_schema_version",
                "schema_version must be a positive integer",
                "schema_version",
            )
        raw_scenes = data.get("scenes", [])
    else:
        raw_scenes = data

    if not isinstance(raw_scenes, (list, tuple)):
        result.add_error("manifest.invalid_scenes", "scenes must be an array", "scenes")
        return result

    seen_ids: set[str] = set()
    for index, item in enumerate(raw_scenes):
        location = f"scenes[{index}]"
        if isinstance(item, Scene):
            raw: Mapping[str, Any] = item.to_dict()
        elif isinstance(item, Mapping):
            raw = dict(item)
        else:
            result.add_error(
                "scene.invalid_entry", "each scene must be an object", location
            )
            continue

        scene_id = raw.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            result.add_error(
                "scene.missing_id", "scene_id must be a non-empty string", location
            )
        else:
            try:
                validate_identifier(scene_id, "scene_id")
            except PathValidationError as exc:
                result.add_error("scene.invalid_id", str(exc), location)
            if scene_id in seen_ids:
                result.add_error(
                    "scene.duplicate_id", f"duplicate scene_id: {scene_id}", location
                )
            seen_ids.add(scene_id)

        title = raw.get("title")
        if not isinstance(title, str) or not title.strip():
            result.add_error(
                "scene.invalid_title", "title must be a non-empty string", location
            )

        number = raw.get("scene_number")
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number != index + 1
        ):
            result.add_error(
                "scene.invalid_order",
                f"scene_number must be {index + 1} for scene at index {index}",
                location,
            )

        duration = raw.get("duration")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or duration <= 0
        ):
            result.add_error(
                "scene.invalid_duration", "duration must be a positive number", location
            )

        status = raw.get("status", "draft")
        if status not in SCENE_STATUSES:
            result.add_error(
                "scene.invalid_status",
                f"status must be one of {sorted(SCENE_STATUSES)}",
                location,
            )

        _validate_text_fields(raw, location, result)

        for key in PATH_LIST_FIELDS:
            value = raw.get(key, [])
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(entry, str) for entry in value
            ):
                result.add_error(
                    "scene.invalid_path_list",
                    f"{key} must be an array of strings",
                    location,
                )
                continue
            if key == "sfx_paths":
                for entry in value:
                    _validate_asset(entry, key, location, root, result)

        character_ids = raw.get("character_ids", [])
        if not isinstance(character_ids, (list, tuple)) or not all(
            isinstance(entry, str) for entry in character_ids
        ):
            result.add_error(
                "scene.invalid_character_ids",
                "character_ids must be an array of strings",
                f"{location}.character_ids",
            )
        else:
            for entry in character_ids:
                try:
                    validate_identifier(entry, "character_id")
                except PathValidationError as exc:
                    result.add_error(
                        "scene.invalid_character_ids",
                        str(exc),
                        f"{location}.character_ids",
                    )

        overrides = raw.get("character_overrides", {})
        if not isinstance(overrides, Mapping) or not all(
            isinstance(key, str) and isinstance(value, Mapping)
            for key, value in overrides.items()
        ):
            result.add_error(
                "scene.invalid_character_overrides",
                "character_overrides must be an object keyed by character_id",
                f"{location}.character_overrides",
            )
        else:
            for key, value in overrides.items():
                for field_name, field_value in value.items():
                    if field_name in CHARACTER_OVERRIDE_FIELDS and not isinstance(
                        field_value, str
                    ):
                        result.add_error(
                            "scene.invalid_character_overrides",
                            f"character_overrides.{key}.{field_name} must be a string",
                            f"{location}.character_overrides.{key}",
                        )

        visual_overrides = raw.get("visual_identity_overrides", {})
        if not isinstance(visual_overrides, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in visual_overrides.items()
        ):
            result.add_error(
                "scene.invalid_visual_identity_overrides",
                "visual_identity_overrides must be an object of string values",
                f"{location}.visual_identity_overrides",
            )

        for key in ASSET_FIELDS:
            _validate_asset(raw.get(key), key, location, root, result)

        start = raw.get("timeline_start", 0.0)
        end = raw.get("timeline_end", 0.0)
        if not _is_number(start) or float(start) < 0:
            result.add_error(
                "scene.invalid_timeline",
                "timeline_start must be a non-negative number",
                location,
            )
        if not _is_number(end) or float(end) < 0:
            result.add_error(
                "scene.invalid_timeline",
                "timeline_end must be a non-negative number",
                location,
            )
        if (
            isinstance(start, (int, float))
            and not isinstance(start, bool)
            and isinstance(end, (int, float))
            and not isinstance(end, bool)
            and isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and abs((float(end) - float(start)) - float(duration)) > 1e-6
        ):
            result.add_warning(
                "scene.timeline_stale",
                "timeline_start/timeline_end do not match duration; "
                "recalculate with the timeline planner",
                location,
            )

        if status in {"ready", "completed"}:
            for key in ("narration", "image_prompt"):
                value = raw.get(key)
                if isinstance(value, str) and not value.strip():
                    result.add_warning(
                        "scene.missing_content",
                        f"{key} is empty for a {status} scene",
                        location,
                    )
    return result


def validate_character_manifest(
    data: Mapping[str, Any] | Sequence[Character | Mapping[str, Any]],
    *,
    project_root: str | Path | None = None,
) -> ValidationResult:
    """Validate a characters manifest and optionally its reference files."""
    result = ValidationResult()
    root = Path(project_root).expanduser().resolve() if project_root is not None else None

    if isinstance(data, Mapping):
        schema_version = data.get("schema_version")
        if schema_version is not None and (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version <= 0
        ):
            result.add_error(
                "character_manifest.invalid_schema_version",
                "schema_version must be a positive integer",
                "schema_version",
            )
        raw_characters = data.get("characters", [])
    else:
        raw_characters = data

    if not isinstance(raw_characters, (list, tuple)):
        result.add_error(
            "character_manifest.invalid_characters",
            "characters must be an array",
            "characters",
        )
        return result

    seen_ids: set[str] = set()
    for index, item in enumerate(raw_characters):
        location = f"characters[{index}]"
        if isinstance(item, Character):
            raw: Mapping[str, Any] = item.to_dict()
        elif isinstance(item, Mapping):
            raw = dict(item)
        else:
            result.add_error(
                "character.invalid_entry", "each character must be an object", location
            )
            continue

        character_id = raw.get("character_id")
        if not isinstance(character_id, str) or not character_id.strip():
            result.add_error(
                "character.missing_id",
                "character_id must be a non-empty string",
                location,
            )
        else:
            try:
                validate_identifier(character_id, "character_id")
            except PathValidationError as exc:
                result.add_error("character.invalid_id", str(exc), location)
            if character_id in seen_ids:
                result.add_error(
                    "character.duplicate_id",
                    f"duplicate character_id: {character_id}",
                    location,
                )
            seen_ids.add(character_id)

        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            result.add_error(
                "character.invalid_name", "name must be a non-empty string", location
            )

        for key in (
            "description",
            "age_description",
            "gender_description",
            "physical_appearance",
            "hairstyle",
            "clothing",
            "distinguishing_features",
            "personality",
            "visual_prompt",
            "negative_prompt",
        ):
            value = raw.get(key)
            if value is not None and not isinstance(value, str):
                result.add_error(
                    "character.invalid_text",
                    f"{key} must be a string",
                    f"{location}.{key}",
                )

        references = raw.get("reference_images", [])
        if not isinstance(references, (list, tuple)) or not all(
            isinstance(entry, str) for entry in references
        ):
            result.add_error(
                "character.invalid_reference_images",
                "reference_images must be an array of strings",
                f"{location}.reference_images",
            )
        else:
            for entry in references:
                _validate_asset(entry, "reference_images", location, root, result)

        for key in ("created_at", "updated_at"):
            value = raw.get(key)
            if value is not None and not is_iso8601_timestamp(value):
                result.add_error(
                    "character.invalid_timestamp",
                    f"{key} must be an ISO 8601 timestamp with a timezone",
                    f"{location}.{key}",
                )
    return result


def validate_project_directory(project_root: str | Path) -> ValidationResult:
    """Load and validate both manifests under *project_root*."""
    root = Path(project_root).expanduser().resolve()
    result = ValidationResult()

    for relative in (Path(PROJECT_FILENAME), SCENES_RELATIVE_PATH):
        path = root / relative
        if not path.is_file():
            result.add_error(
                "manifest.missing_file",
                f"Required manifest is missing: {relative.as_posix()}",
            )

    project_path = root / PROJECT_FILENAME
    if project_path.is_file():
        result.merge(_validate_json_file(project_path, "project", validate_project_metadata))

    scenes_path = root / SCENES_RELATIVE_PATH
    if scenes_path.is_file():
        payload = _read_json(scenes_path, result)
        if payload is not None:
            result.merge(validate_scene_manifest(payload, project_root=root))

    characters_path = root / CHARACTERS_RELATIVE_PATH
    if characters_path.is_file():
        payload = _read_json(characters_path, result)
        if payload is not None:
            result.merge(validate_character_manifest(payload, project_root=root))
    return result


def _read_json(path: Path, result: ValidationResult) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.add_error(
            "manifest.invalid_json",
            f"Could not read {path.name} as JSON: {exc}",
            path.name,
        )
        return None


def _validate_json_file(
    path: Path,
    label: str,
    validator: Callable[[Mapping[str, Any]], ValidationResult],
) -> ValidationResult:
    result = ValidationResult()
    payload = _read_json(path, result)
    if payload is None:
        return result
    if not isinstance(payload, Mapping):
        result.add_error(
            "manifest.invalid_root",
            f"{label} manifest must be a JSON object",
            path.name,
        )
        return result
    return validator(payload)
