"""Serializable data models for scene-based video projects.

The models are deliberately provider-agnostic: no LTX or ComfyUI field names
appear here.  Unknown manifest keys are retained in ``extra`` so future
workflows can extend the format without losing data on update.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from character_models import CharacterSceneOverride, VisualIdentity

SCHEMA_VERSION = 1

PROJECT_STATUSES = frozenset({"draft", "active", "archived"})
SCENE_STATUSES = frozenset(
    {
        "draft",
        "ready",
        "generating_image",
        "image_generated",
        "generating_video",
        "video_generated",
        "completed",
        "failed",
    }
)
VIDEO_FORMATS = frozenset({"mp4", "mov", "webm"})

DEFAULT_LANGUAGE = "en"
DEFAULT_PROJECT_STATUS = "active"
DEFAULT_SCENE_STATUS = "draft"
DEFAULT_NARRATOR_VOICE = "chatterbox-default-narrator"
DEFAULT_VISUAL_STYLE = "cinematic"
DEFAULT_ASPECT_RATIO = "16:9"


def _known_names(model: Any) -> set[str]:
    return {item.name for item in fields(model)}


def _string(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str)]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _extras(raw: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value) for key, value in raw.items() if key not in known
    }


@dataclass
class ProjectSettings:
    """Render defaults for a project; all fields are configurable."""

    width: int = 768
    height: int = 512
    fps: int = 24
    target_duration: float = 60.0
    video_format: str = "mp4"
    narrator_voice: str = DEFAULT_NARRATOR_VOICE
    visual_style: str = DEFAULT_VISUAL_STYLE
    aspect_ratio: str = DEFAULT_ASPECT_RATIO
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        known = _known_names(self)
        result = {
            name: copy.deepcopy(getattr(self, name))
            for name in sorted(known - {"extra"})
        }
        result.update(copy.deepcopy(self.extra))
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProjectSettings:
        raw = dict(data)
        known = _known_names(cls)
        settings = cls(
            width=_integer(raw.get("width"), 768),
            height=_integer(raw.get("height"), 512),
            fps=_integer(raw.get("fps"), 24),
            target_duration=_number(raw.get("target_duration"), 60.0),
            video_format=_string(raw.get("video_format"), "mp4"),
            narrator_voice=_string(raw.get("narrator_voice"), DEFAULT_NARRATOR_VOICE),
            visual_style=_string(raw.get("visual_style"), DEFAULT_VISUAL_STYLE),
            aspect_ratio=_string(raw.get("aspect_ratio"), DEFAULT_ASPECT_RATIO),
        )
        settings.extra = _extras(raw, known)
        return settings


@dataclass
class Project:
    """Project-level metadata, kept separate from scene data."""

    project_id: str
    title: str
    description: str = ""
    status: str = DEFAULT_PROJECT_STATUS
    language: str = DEFAULT_LANGUAGE
    created_at: str = ""
    updated_at: str = ""
    settings: ProjectSettings = field(default_factory=ProjectSettings)
    visual_identity: VisualIdentity | None = None
    schema_version: int = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        known = _known_names(self)
        result = {
            name: copy.deepcopy(getattr(self, name))
            for name in sorted(known - {"extra", "settings", "visual_identity"})
        }
        result["settings"] = self.settings.to_dict()
        result["visual_identity"] = (
            self.visual_identity.to_dict() if self.visual_identity is not None else None
        )
        result.update(copy.deepcopy(self.extra))
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Project:
        raw = dict(data)
        project_id = raw.get("project_id")
        title = raw.get("title")
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("project.json: 'project_id' must be a non-empty string")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("project.json: 'title' must be a non-empty string")
        known = _known_names(cls)
        project = cls(
            project_id=project_id,
            title=title,
            description=_string(raw.get("description")),
            status=_string(raw.get("status"), DEFAULT_PROJECT_STATUS),
            language=_string(raw.get("language"), DEFAULT_LANGUAGE),
            created_at=_string(raw.get("created_at")),
            updated_at=_string(raw.get("updated_at")),
            settings=ProjectSettings.from_dict(_mapping(raw.get("settings", {}))),
            visual_identity=(
                VisualIdentity.from_dict(identity)
                if isinstance(identity := raw.get("visual_identity"), Mapping)
                else None
            ),
            schema_version=_integer(raw.get("schema_version"), SCHEMA_VERSION),
        )
        project.extra = _extras(raw, known)
        return project


@dataclass
class Scene:
    """One independent production unit; durations are planned, not rendered."""

    scene_id: str
    scene_number: int
    title: str
    description: str = ""
    duration: float = 5.0
    narration: str = ""
    image_prompt: str = ""
    animation_prompt: str = ""
    negative_prompt: str = ""
    character_references: list[str] = field(default_factory=list)
    character_ids: list[str] = field(default_factory=list)
    character_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    visual_identity_overrides: dict[str, Any] = field(default_factory=dict)
    image_path: str | None = None
    generated_video_path: str | None = None
    narration_audio_path: str | None = None
    sfx_paths: list[str] = field(default_factory=list)
    status: str = DEFAULT_SCENE_STATUS
    timeline_start: float = 0.0
    timeline_end: float = 0.0
    thumbnail_path: str | None = None
    generation_provider: str = "default"
    provider_options: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        known = _known_names(self)
        result = {
            name: copy.deepcopy(getattr(self, name))
            for name in sorted(known - {"extra"})
        }
        result.update(copy.deepcopy(self.extra))
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Scene:
        raw = dict(data)
        scene_id = raw.get("scene_id")
        title = raw.get("title")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene: 'scene_id' must be a non-empty string")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("scene: 'title' must be a non-empty string")
        known = _known_names(cls)
        scene = cls(
            scene_id=scene_id,
            scene_number=_integer(raw.get("scene_number"), 1),
            title=title,
            description=_string(raw.get("description")),
            duration=_number(raw.get("duration"), 5.0),
            narration=_string(raw.get("narration")),
            image_prompt=_string(raw.get("image_prompt")),
            animation_prompt=_string(raw.get("animation_prompt")),
            negative_prompt=_string(raw.get("negative_prompt")),
            character_references=_string_list(raw.get("character_references")),
            character_ids=_string_list(raw.get("character_ids")),
            character_overrides={
                key: CharacterSceneOverride.from_dict(_mapping(value)).to_dict()
                for key, value in _mapping(raw.get("character_overrides")).items()
                if isinstance(key, str)
            },
            visual_identity_overrides={
                key: _string(value)
                for key, value in _mapping(
                    raw.get("visual_identity_overrides")
                ).items()
                if isinstance(key, str)
            },
            image_path=_optional_string(raw.get("image_path")),
            generated_video_path=_optional_string(raw.get("generated_video_path")),
            narration_audio_path=_optional_string(raw.get("narration_audio_path")),
            sfx_paths=_string_list(raw.get("sfx_paths")),
            status=_string(raw.get("status"), DEFAULT_SCENE_STATUS),
            timeline_start=_number(raw.get("timeline_start"), 0.0),
            timeline_end=_number(raw.get("timeline_end"), 0.0),
            thumbnail_path=_optional_string(raw.get("thumbnail_path")),
            generation_provider=_string(raw.get("generation_provider"), "default"),
            provider_options=_mapping(raw.get("provider_options")),
            schema_version=_integer(raw.get("schema_version"), SCHEMA_VERSION),
        )
        scene.extra = _extras(raw, known)
        return scene
