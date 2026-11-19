"""Typed models for character identity and project-wide visual identity.

These dataclasses are intentionally independent of MCP, ComfyUI and file I/O.
Unknown JSON keys are preserved in ``extra`` so manifests survive round-trips.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from project_paths import is_iso8601_timestamp

TEXT_FIELDS: tuple[str, ...] = (
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
)

VISUAL_IDENTITY_FIELDS: tuple[str, ...] = (
    "illustration_style",
    "color_palette",
    "lighting_style",
    "rendering_style",
    "environment_style",
    "camera_language",
    "composition_guidelines",
    "negative_prompt",
    "general_visual_guidelines",
)

CHARACTER_OVERRIDE_FIELDS: tuple[str, ...] = (
    "appearance",
    "clothing",
    "expression",
    "action",
    "pose",
)


def _string(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


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
class VisualIdentity:
    """Project-wide visual direction inherited by scenes."""

    illustration_style: str = ""
    color_palette: str = ""
    lighting_style: str = ""
    rendering_style: str = ""
    environment_style: str = ""
    camera_language: str = ""
    composition_guidelines: str = ""
    negative_prompt: str = ""
    general_visual_guidelines: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-ready mapping with extra keys flattened."""
        result = {
            "illustration_style": self.illustration_style,
            "color_palette": self.color_palette,
            "lighting_style": self.lighting_style,
            "rendering_style": self.rendering_style,
            "environment_style": self.environment_style,
            "camera_language": self.camera_language,
            "composition_guidelines": self.composition_guidelines,
            "negative_prompt": self.negative_prompt,
            "general_visual_guidelines": self.general_visual_guidelines,
        }
        result.update(copy.deepcopy(self.extra))
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> VisualIdentity:
        """Build a visual identity, coercing missing or invalid fields safely."""
        identity = cls()
        for name in VISUAL_IDENTITY_FIELDS:
            setattr(identity, name, _string(raw.get(name)))
        known = {field.name for field in fields(cls)}
        identity.extra = _extras(raw, known)
        return identity

    def overrides(self, values: Mapping[str, Any] | None) -> VisualIdentity:
        """Return a copy with sparse scene-specific overrides applied."""
        merged = VisualIdentity.from_dict(self.to_dict())
        for key, value in (values or {}).items():
            if key in VISUAL_IDENTITY_FIELDS:
                setattr(merged, key, _string(value))
            else:
                merged.extra[key] = copy.deepcopy(value)
        return merged


@dataclass
class CharacterSceneOverride:
    """Scene-specific changes to a character's appearance or behaviour."""

    appearance: str = ""
    clothing: str = ""
    expression: str = ""
    action: str = ""
    pose: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-ready mapping with extra keys flattened."""
        result = {
            "appearance": self.appearance,
            "clothing": self.clothing,
            "expression": self.expression,
            "action": self.action,
            "pose": self.pose,
        }
        result.update(copy.deepcopy(self.extra))
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CharacterSceneOverride:
        """Build scene overrides, preserving unknown keys."""
        override = cls()
        for name in CHARACTER_OVERRIDE_FIELDS:
            setattr(override, name, _string(raw.get(name)))
        known = {item.name for item in fields(cls)}
        override.extra = _extras(raw, known)
        return override


@dataclass
class Character:
    """Reusable identity record for one character in a project."""

    character_id: str
    name: str
    description: str = ""
    age_description: str = ""
    gender_description: str = ""
    physical_appearance: str = ""
    hairstyle: str = ""
    clothing: str = ""
    distinguishing_features: str = ""
    personality: str = ""
    visual_prompt: str = ""
    reference_images: list[str] = field(default_factory=list)
    negative_prompt: str = ""
    created_at: str = ""
    updated_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-ready mapping with extra keys flattened."""
        result = {
            "character_id": self.character_id,
            "name": self.name,
            "description": self.description,
            "age_description": self.age_description,
            "gender_description": self.gender_description,
            "physical_appearance": self.physical_appearance,
            "hairstyle": self.hairstyle,
            "clothing": self.clothing,
            "distinguishing_features": self.distinguishing_features,
            "personality": self.personality,
            "visual_prompt": self.visual_prompt,
            "reference_images": copy.deepcopy(self.reference_images),
            "negative_prompt": self.negative_prompt,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        result.update(copy.deepcopy(self.extra))
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Character:
        """Build a character from a manifest entry."""
        character = cls(
            character_id=_string(raw.get("character_id")),
            name=_string(raw.get("name")),
        )
        for name in TEXT_FIELDS:
            setattr(character, name, _string(raw.get(name)))
        character.reference_images = _string_list(raw.get("reference_images"))
        character.created_at = (
            value
            if is_iso8601_timestamp(value := raw.get("created_at", ""))
            else ""
        )
        character.updated_at = (
            value
            if is_iso8601_timestamp(value := raw.get("updated_at", ""))
            else ""
        )
        known = {item.name for item in fields(cls)}
        character.extra = _extras(raw, known)
        return character
