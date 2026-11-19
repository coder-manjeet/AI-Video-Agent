"""Tests for deterministic, structured prompt composition."""

from __future__ import annotations

import json

from character_models import Character, CharacterSceneOverride, VisualIdentity
from visual_prompt_builder import (
    PromptCharacter,
    PromptContext,
    VisualPromptBuilder,
)


def _identity() -> VisualIdentity:
    return VisualIdentity(
        illustration_style="cinematic",
        color_palette="warm amber",
        camera_language="35mm lens",
        negative_prompt="text, watermark",
        extra={"custom_style": "ink wash"},
    )


def _aria() -> Character:
    return Character(
        character_id="aria",
        name="Aria",
        description="a cartographer",
        physical_appearance="tall with silver hair",
        hairstyle="long braid",
        clothing="leather coat",
        visual_prompt="soft portrait lighting",
        negative_prompt="extra fingers",
    )


def test_build_combines_identity_character_scene_and_original_prompt() -> None:
    context = PromptContext(
        base_prompt="Aria walks home",
        scene_description="A rainy street at dusk",
        base_negative_prompt="blurry",
        project_identity=_identity(),
        characters=[
            PromptCharacter(
                character=_aria(),
                override=CharacterSceneOverride(
                    appearance="rain-soaked",
                    clothing="dark hood",
                    expression="determined",
                    action="running",
                    pose="leaning forward",
                ),
            )
        ],
    )

    result = VisualPromptBuilder().build(context)

    assert result.character_ids == ("aria",)
    assert result.original_positive_prompt == "Aria walks home"
    assert result.original_negative_prompt == "blurry"
    assert result.negative_prompt == "text, watermark, blurry, extra fingers"
    assert "style: cinematic" in result.positive_prompt
    assert "palette: warm amber" in result.positive_prompt
    assert "camera: 35mm lens" in result.positive_prompt
    assert "custom_style: ink wash" in result.positive_prompt
    assert "rain-soaked" in result.positive_prompt
    assert "dark hood" in result.positive_prompt
    assert "tall with silver hair" not in result.positive_prompt
    assert "leather coat" not in result.positive_prompt
    assert "expression: determined" in result.positive_prompt
    assert "action: running" in result.positive_prompt
    assert "pose: leaning forward" in result.positive_prompt
    assert "A rainy street at dusk" in result.positive_prompt
    assert result.positive_prompt.endswith("Prompt: Aria walks home")
    assert set(result.sections) == {
        "visual_identity",
        "characters",
        "character_actions",
        "scene",
        "prompt",
    }


def test_multiple_characters_are_all_included() -> None:
    boman = Character(
        character_id="boman",
        name="Boman",
        description="a guard",
        distinguishing_features="scar on chin",
    )
    result = VisualPromptBuilder().build(
        PromptContext(
            base_prompt="Two people argue",
            project_identity=_identity(),
            characters=[PromptCharacter(_aria()), PromptCharacter(boman)],
        )
    )

    assert result.character_ids == ("aria", "boman")
    assert "a cartographer" in result.positive_prompt
    assert "a guard" in result.positive_prompt
    assert result.positive_prompt.count("\n- ") == 2


def test_scene_without_characters_is_supported() -> None:
    result = VisualPromptBuilder().build(
        PromptContext(
            base_prompt="An empty room",
            scene_description="Dust in a sunbeam",
            project_identity=_identity(),
        )
    )

    assert result.character_ids == ()
    assert "characters" not in result.sections
    assert "character_actions" not in result.sections
    assert "An empty room" in result.positive_prompt
    assert "Dust in a sunbeam" in result.positive_prompt


def test_scene_identity_overrides_replace_instead_of_duplicate() -> None:
    result = VisualPromptBuilder().build(
        PromptContext(
            base_prompt="A portrait",
            project_identity=_identity(),
            visual_identity_overrides={"color_palette": "cold blue"},
        )
    )

    assert "palette: cold blue" in result.positive_prompt
    assert "warm amber" not in result.positive_prompt
    assert result.positive_prompt.count("palette:") == 1


def test_duplicate_fragments_are_removed_from_positive_and_negative() -> None:
    identity = VisualIdentity(color_palette="warm amber", negative_prompt="blurry")
    result = VisualPromptBuilder().build(
        PromptContext(
            base_prompt="warm amber",
            base_negative_prompt="warm amber, blurry, watermark",
            project_identity=identity,
        )
    )

    assert result.positive_prompt.count("warm amber") == 2  # style fragment + base prompt
    assert result.negative_prompt == "blurry, watermark"


def test_language_labels_and_unknown_language_fallback() -> None:
    context = PromptContext(
        base_prompt="A street",
        scene_description="Night",
        characters=[PromptCharacter(_aria())],
        language="hi",
    )
    hindi = VisualPromptBuilder().build(context)

    assert "पात्र:" in hindi.positive_prompt
    assert "दृश्य:" in hindi.positive_prompt
    assert hindi.language == "hi"

    fallback = VisualPromptBuilder().build(
        PromptContext(base_prompt="A street", language="xx")
    )
    assert fallback.language == "en"
    assert "Characters" not in fallback.positive_prompt


def test_to_dict_is_json_serialisable_and_keeps_originals() -> None:
    result = VisualPromptBuilder().build(
        PromptContext(
            base_prompt="Original prompt",
            base_negative_prompt="Original negative",
            project_identity=_identity(),
            characters=[PromptCharacter(_aria())],
        )
    )

    payload = result.to_dict()

    assert payload["original_positive_prompt"] == "Original prompt"
    assert payload["original_negative_prompt"] == "Original negative"
    assert payload["character_ids"] == ["aria"]
    assert isinstance(payload["sections"]["characters"], list)
    assert json.loads(json.dumps(payload)) == payload
