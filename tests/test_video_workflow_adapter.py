"""Tests for the video workflow registry and offline patching."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import workflow_adapter as wa

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _video_registry() -> wa.WorkflowRegistry:
    return wa.load_workflow_registry(PROJECT_DIR / "video_workflows.json")


def _video_workflow() -> dict[str, Any]:
    payload = json.loads(
        (PROJECT_DIR / "ltx_image_to_video_api.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def test_video_registry_matches_the_verified_ltx_workflow() -> None:
    registry = _video_registry()
    spec = registry.default

    assert registry.default_name == "ltx_image_to_video"
    assert spec.workflow_file == "ltx_image_to_video_api.json"
    assert spec.output.images_key == "gifs"
    assert spec.output.check_input is False
    assert spec.image is not None and spec.image.node == "78"
    assert spec.frames is not None and spec.frames.node == "95"
    assert [binding.node for binding in spec.fps] == ["69", "103"]
    assert spec.model_inputs == ("38.clip_name", "44.ckpt_name")
    assert wa.validate_workflow(spec, _video_workflow()) == []


def test_video_validation_detects_wrong_model_or_missing_image_node() -> None:
    spec = _video_registry().default
    workflow = _video_workflow()
    workflow["44"]["inputs"]["ckpt_name"] = "unknown.safetensors"
    del workflow["78"]

    errors = wa.validate_workflow(spec, workflow)

    assert any("44.ckpt_name" in error for error in errors)
    assert any("image: missing node 78" in error for error in errors)


def test_video_patch_applies_all_bindings_without_mutating_source() -> None:
    spec = _video_registry().default
    original = _video_workflow()
    saved = copy.deepcopy(original)
    options = wa.GenerationOptions(
        positive_prompt="Gentle motion",
        negative_prompt="Flicker",
        width=512,
        height=384,
        frames=49,
        fps=12,
        seed=7,
    )

    patched = wa.patch_workflow(
        spec,
        original,
        options,
        filename_prefix="scene-001-opening",
        image_name="approved.png",
    )

    assert patched["6"]["inputs"]["text"] == "Gentle motion"
    assert patched["7"]["inputs"]["text"] == "Flicker"
    assert patched["78"]["inputs"]["image"] == "approved.png"
    assert patched["95"]["inputs"]["width"] == 512
    assert patched["95"]["inputs"]["height"] == 384
    assert patched["95"]["inputs"]["length"] == 49
    assert patched["69"]["inputs"]["frame_rate"] == 12
    assert patched["103"]["inputs"]["frame_rate"] == 12
    assert patched["102"]["inputs"]["noise_seed"] == 7
    assert patched["103"]["inputs"]["filename_prefix"] == "scene-001-opening"
    assert original == saved


def test_extract_output_entry_finds_video_in_gifs() -> None:
    spec = _video_registry().default
    video = {"filename": "ltxv_00001.mp4", "subfolder": "", "type": "output"}
    history = {
        "p1": {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {"103": {"gifs": [video]}},
        }
    }

    assert wa.extract_output_entry(spec, history, "p1") == video
    assert wa.extract_output_entry(spec, {}, "p1") is None


def test_image_workflow_registry_remains_unchanged() -> None:
    registry = wa.load_workflow_registry(PROJECT_DIR / "image_workflows.json")
    spec = registry.default

    assert spec.workflow_file == "z_image_turbo_text_to_image_api.json"
    assert spec.output.images_key == "images"
    assert spec.output.check_input is True
    assert spec.fps == ()
    assert spec.model_inputs == ()
    assert wa.validate_workflow(spec, json.loads(
        (PROJECT_DIR / spec.workflow_file).read_text(encoding="utf-8")
    )) == []


@pytest.mark.parametrize("value", [None, 0, -1, 1.5])
def test_invalid_check_input_is_rejected(tmp_path: Path, value: Any) -> None:
    payload = json.loads((PROJECT_DIR / "video_workflows.json").read_text(encoding="utf-8"))
    payload["workflows"]["ltx_image_to_video"]["output"]["check_input"] = value
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(wa.WorkflowConfigError, match="check_input"):
        wa.load_workflow_registry(registry_file, PROJECT_DIR)
