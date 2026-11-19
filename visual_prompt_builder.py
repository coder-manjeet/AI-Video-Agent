"""Deterministic, structured image-prompt composition.

The builder is a pure service: it performs no file I/O and calls no external
systems.  Callers supply already-loaded project/character data in a
:class:`PromptContext`; the result keeps the original user-written prompt and
exposes each section separately for debugging.

Prompt composition improves the *chance* of visual consistency, but it cannot
guarantee identity preservation in the generated image.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from character_models import Character, CharacterSceneOverride, VisualIdentity

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "es", "fr", "de", "hi", "pt", "zh")

# Section labels are localised where practical; technical field labels
# (style, palette, lighting, ...) stay in English across languages.
_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "visual_identity": "Visual identity",
        "characters": "Characters",
        "scene": "Scene",
        "character_actions": "Character actions",
        "prompt": "Prompt",
        "negative": "Negative prompt",
        "expression": "expression",
        "action": "action",
        "pose": "pose",
    },
    "es": {
        "visual_identity": "Identidad visual",
        "characters": "Personajes",
        "scene": "Escena",
        "character_actions": "Acciones de personajes",
        "prompt": "Prompt",
        "negative": "Prompt negativo",
        "expression": "expresión",
        "action": "acción",
        "pose": "pose",
    },
    "fr": {
        "visual_identity": "Identité visuelle",
        "characters": "Personnages",
        "scene": "Scène",
        "character_actions": "Actions des personnages",
        "prompt": "Prompt",
        "negative": "Prompt négatif",
        "expression": "expression",
        "action": "action",
        "pose": "pose",
    },
    "de": {
        "visual_identity": "Visuelle Identität",
        "characters": "Charaktere",
        "scene": "Szene",
        "character_actions": "Charakteraktionen",
        "prompt": "Prompt",
        "negative": "Negativer Prompt",
        "expression": "Ausdruck",
        "action": "Aktion",
        "pose": "Pose",
    },
    "hi": {
        "visual_identity": "दृश्य पहचान",
        "characters": "पात्र",
        "scene": "दृश्य",
        "character_actions": "पात्रों की क्रियाएँ",
        "prompt": "प्रॉम्प्ट",
        "negative": "नकारात्मक प्रॉम्प्ट",
        "expression": "भाव",
        "action": "क्रिया",
        "pose": "मुद्रा",
    },
    "pt": {
        "visual_identity": "Identidade visual",
        "characters": "Personagens",
        "scene": "Cena",
        "character_actions": "Ações dos personagens",
        "prompt": "Prompt",
        "negative": "Prompt negativo",
        "expression": "expressão",
        "action": "ação",
        "pose": "pose",
    },
    "zh": {
        "visual_identity": "视觉识别",
        "characters": "角色",
        "scene": "场景",
        "character_actions": "角色动作",
        "prompt": "提示词",
        "negative": "负面提示词",
        "expression": "表情",
        "action": "动作",
        "pose": "姿势",
    },
}

_IDENTITY_LABELS: tuple[tuple[str, str], ...] = (
    ("illustration_style", "style"),
    ("color_palette", "palette"),
    ("lighting_style", "lighting"),
    ("rendering_style", "rendering"),
    ("environment_style", "environment"),
    ("camera_language", "camera"),
    ("composition_guidelines", "composition"),
    ("general_visual_guidelines", "guidelines"),
)

_RENDER_ORDER: tuple[str, ...] = (
    "visual_identity",
    "characters",
    "scene",
    "character_actions",
    "prompt",
)


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def _split_fragments(value: str) -> list[str]:
    """Split a free-form prompt into comparable fragments for de-duplication."""
    result: list[str] = []
    for line in value.replace(";", ",").replace("\n", ",").split(","):
        fragment = line.strip()
        if fragment and fragment not in result:
            result.append(fragment)
    return result


@dataclass(frozen=True)
class PromptCharacter:
    """One character plus its optional scene-specific overrides."""

    character: Character
    override: CharacterSceneOverride | None = None


@dataclass
class PromptContext:
    """All inputs needed to compose one scene image prompt."""

    base_prompt: str = ""
    scene_description: str = ""
    base_negative_prompt: str = ""
    project_identity: VisualIdentity | None = None
    characters: Sequence[PromptCharacter] = ()
    visual_identity_overrides: Mapping[str, str] = field(default_factory=dict)
    language: str = "en"


@dataclass(frozen=True)
class PromptResult:
    """Structured output of prompt composition."""

    positive_prompt: str
    negative_prompt: str
    original_positive_prompt: str
    original_negative_prompt: str
    character_ids: tuple[str, ...]
    language: str
    sections: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable debug representation."""
        return {
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "original_positive_prompt": self.original_positive_prompt,
            "original_negative_prompt": self.original_negative_prompt,
            "character_ids": list(self.character_ids),
            "language": self.language,
            "sections": {key: list(value) for key, value in self.sections.items()},
        }


class VisualPromptBuilder:
    """Compose project identity, characters and scene data into prompts."""

    def build(self, context: PromptContext) -> PromptResult:
        """Compose a deterministic positive/negative prompt pair."""
        language = context.language if context.language in _LABELS else "en"
        labels = _LABELS[language]
        sections: dict[str, list[str]] = {}
        seen_positive: set[str] = set()

        def add_positive(section: str, text: str) -> None:
            fragment = text.strip()
            if not fragment:
                return
            key = _normalise(fragment)
            if key in seen_positive:
                return
            seen_positive.add(key)
            sections.setdefault(section, []).append(fragment)

        identity = context.project_identity
        if identity is not None:
            merged = identity.overrides(context.visual_identity_overrides)
            for attribute, label in _IDENTITY_LABELS:
                value = getattr(merged, attribute, "")
                if isinstance(value, str) and value.strip():
                    add_positive("visual_identity", f"{label}: {value.strip()}")
            for key, value in merged.extra.items():
                if isinstance(value, str) and value.strip():
                    add_positive("visual_identity", f"{key}: {value.strip()}")

        character_ids: list[str] = []
        for entry in context.characters:
            character = entry.character
            override = entry.override
            character_ids.append(character.character_id)

            details: list[str] = []
            if character.name.strip():
                details.append(character.name.strip())
            for attribute in (
                "description",
                "age_description",
                "gender_description",
            ):
                value = getattr(character, attribute)
                if value.strip():
                    details.append(value.strip())
            appearance = (
                override.appearance
                if override is not None and override.appearance.strip()
                else character.physical_appearance
            )
            if appearance.strip():
                details.append(appearance.strip())
            if character.hairstyle.strip():
                details.append(character.hairstyle.strip())
            clothing = (
                override.clothing
                if override is not None and override.clothing.strip()
                else character.clothing
            )
            if clothing.strip():
                details.append(clothing.strip())
            for attribute in ("distinguishing_features", "personality", "visual_prompt"):
                value = getattr(character, attribute)
                if value.strip():
                    details.append(value.strip())
            if details:
                add_positive("characters", ", ".join(details))

            actions: list[str] = []
            if override is not None:
                for attribute in ("expression", "action", "pose"):
                    value = getattr(override, attribute)
                    if value.strip():
                        actions.append(f"{labels[attribute]}: {value.strip()}")
            if actions:
                prefix = f"{character.name.strip()}: " if character.name.strip() else ""
                add_positive("character_actions", prefix + "; ".join(actions))

        add_positive("scene", context.scene_description)
        add_positive("prompt", context.base_prompt)

        negative_parts: list[str] = []
        seen_negative: set[str] = set()

        def add_negative(value: str) -> None:
            for fragment in _split_fragments(value):
                key = _normalise(fragment)
                if key in seen_negative or key in seen_positive:
                    continue
                seen_negative.add(key)
                negative_parts.append(fragment)

        if identity is not None:
            add_negative(identity.negative_prompt)
        add_negative(context.base_negative_prompt)
        for entry in context.characters:
            add_negative(entry.character.negative_prompt)

        positive_prompt = self._render_positive(sections, labels)
        return PromptResult(
            positive_prompt=positive_prompt,
            negative_prompt=", ".join(negative_parts),
            original_positive_prompt=context.base_prompt,
            original_negative_prompt=context.base_negative_prompt,
            character_ids=tuple(character_ids),
            language=language,
            sections={
                key: tuple(fragments) for key, fragments in sections.items()
            },
        )

    @staticmethod
    def _render_positive(
        sections: Mapping[str, Sequence[str]], labels: Mapping[str, str]
    ) -> str:
        """Render collected fragments in a stable, labelled order."""
        lines: list[str] = []
        for key in _RENDER_ORDER:
            fragments = sections.get(key)
            if not fragments:
                continue
            label = labels[key]
            if key == "characters":
                lines.append(
                    f"{label}:\n" + "\n".join(f"- {fragment}" for fragment in fragments)
                )
            else:
                lines.append(f"{label}: " + "; ".join(fragments))
        return "\n".join(lines)
