"""Tests for character and visual identity dataclass serialization."""

from __future__ import annotations

from character_models import Character, CharacterSceneOverride, VisualIdentity


def test_character_roundtrip_preserves_all_fields_and_extra() -> None:
    character = Character(
        character_id="aria",
        name="Aria",
        description="A cartographer",
        age_description="young adult",
        gender_description="woman",
        physical_appearance="tall, silver hair",
        hairstyle="long braid",
        clothing="leather coat",
        distinguishing_features="scar on left cheek",
        personality="curious",
        visual_prompt="soft cinematic portrait",
        reference_images=["assets/characters/aria/ref.png"],
        negative_prompt="blurry, extra fingers",
        created_at="2026-11-20T00:00:00+00:00",
        updated_at="2026-11-20T01:00:00+00:00",
        extra={"future_field": {"nested": True}},
    )

    payload = character.to_dict()
    restored = Character.from_dict(payload)

    assert restored == character
    assert payload["future_field"] == {"nested": True}
    assert restored.extra["future_field"] == {"nested": True}


def test_character_optional_fields_are_not_forced() -> None:
    character = Character.from_dict({"character_id": "hero", "name": "Hero"})

    assert character.character_id == "hero"
    assert character.name == "Hero"
    assert character.description == ""
    assert character.reference_images == []
    assert character.created_at == ""
    assert character.extra == {}


def test_visual_identity_roundtrip_and_sparse_overrides() -> None:
    identity = VisualIdentity(
        illustration_style="cinematic",
        color_palette="warm amber",
        camera_language="35mm lens",
        negative_prompt="text, watermark",
        extra={"custom_style": "ink wash"},
    )

    merged = identity.overrides({"color_palette": "cold blue", "new_field": "kept"})

    assert merged.illustration_style == "cinematic"
    assert merged.color_palette == "cold blue"
    assert merged.negative_prompt == "text, watermark"
    assert merged.extra["custom_style"] == "ink wash"
    assert merged.extra["new_field"] == "kept"
    assert identity.color_palette == "warm amber"
    assert VisualIdentity.from_dict(identity.to_dict()) == identity


def test_character_scene_override_roundtrip() -> None:
    override = CharacterSceneOverride(
        appearance="wet cloak",
        expression="determined",
        action="running",
        pose="leaning forward",
        extra={"glow": "faint blue"},
    )

    restored = CharacterSceneOverride.from_dict(override.to_dict())

    assert restored == override
