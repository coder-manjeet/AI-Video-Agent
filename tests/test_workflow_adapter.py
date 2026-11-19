"""Tests for the offline workflow registry, validator and patcher."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

import workflow_adapter as wa

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _registry() -> wa.WorkflowRegistry:
    return wa.load_workflow_registry(PROJECT_DIR / "image_workflows.json")


def _workflow() -> dict:
    payload = json.loads(
        (PROJECT_DIR / "z_image_turbo_text_to_image_api.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(payload, dict)
    return payload


def test_registry_loads_default_and_lists_workflows() -> None:
    registry = _registry()

    assert registry.default_name == "z_image_turbo_text_to_image"
    assert registry.default.name == "z_image_turbo_text_to_image"
    assert "z_image_turbo_text_to_image" in registry.names()
    assert registry.workflow_path(registry.default).is_file()
    assert registry.default.capabilities["negative_prompt"] is False
    assert registry.default.defaults["width"] == 768


def test_registry_rejects_unknown_workflow() -> None:
    with pytest.raises(wa.WorkflowConfigError, match="Unknown workflow"):
        _registry().get("missing")


def test_registry_rejects_workflow_file_outside_base_dir(tmp_path: Path) -> None:
    registry = _registry()
    escaping = replace(registry.default, workflow_file="../outside.json")
    bad = wa.WorkflowRegistry(
        {escaping.name: escaping},
        registry.default_name,
        tmp_path,
    )
    with pytest.raises(wa.WorkflowConfigError, match="parent segment"):
        bad.workflow_path(escaping)


def test_verified_workflow_passes_validation() -> None:
    registry = _registry()
    spec = registry.default

    assert wa.validate_workflow(spec, _workflow()) == []
    assert registry.workflow_path(spec).name == "z_image_turbo_text_to_image_api.json"


def test_validation_detects_incorrect_sd3_clip_configuration() -> None:
    workflow = _workflow()
    workflow["2"]["inputs"]["type"] = "sd3"

    errors = wa.validate_workflow(_registry().default, workflow)

    assert any("required input 2.type" in error for error in errors)


def test_validation_detects_missing_node_and_input() -> None:
    workflow = _workflow()
    del workflow["10"]["inputs"]["filename_prefix"]
    workflow.pop("9")

    errors = wa.validate_workflow(_registry().default, workflow)

    assert any("node 9" in error for error in errors)
    assert any("filename_prefix" in error for error in errors)


def test_validation_detects_wrong_output_class() -> None:
    workflow = _workflow()
    workflow["10"]["class_type"] = "PreviewImage"

    errors = wa.validate_workflow(_registry().default, workflow)

    assert any("output" in error and "PreviewImage" in error for error in errors)


def test_patch_workflow_applies_options_without_mutating_source() -> None:
    registry = _registry()
    spec = registry.default
    original = _workflow()
    saved = copy.deepcopy(original)
    options = wa.GenerationOptions(
        positive_prompt="A red apple on a white table.",
        negative_prompt="blurry",
        width=512,
        height=512,
        seed=42,
        steps=8,
        cfg=1.0,
    )

    patched = wa.patch_workflow(
        spec, original, options, filename_prefix="scene-001-opening"
    )

    assert patched["4"]["inputs"]["text"] == "A red apple on a white table."
    assert patched["6"]["inputs"]["width"] == 512
    assert patched["6"]["inputs"]["height"] == 512
    assert patched["8"]["inputs"]["seed"] == 42
    assert patched["8"]["inputs"]["steps"] == 8
    assert patched["10"]["inputs"]["filename_prefix"] == "scene-001-opening"
    assert original == saved


def test_patch_workflow_rejects_incompatible_workflow() -> None:
    workflow = _workflow()
    workflow["2"]["inputs"]["type"] = "sd3"
    options = wa.GenerationOptions("a cat")

    with pytest.raises(wa.WorkflowValidationError, match="incompatible"):
        wa.patch_workflow(_registry().default, workflow, options)


def test_unsupported_options_report_negative_prompt() -> None:
    spec = _registry().default
    options = wa.GenerationOptions("a cat", negative_prompt="blurry", seed=7)

    warnings = wa.unsupported_options(spec, options)

    assert len(warnings) == 1
    assert "negative prompt" in warnings[0].lower()
    assert wa.unsupported_options(spec, wa.GenerationOptions("a cat", width=512)) == []


def test_job_status_maps_history_states() -> None:
    assert wa.job_status({}, "p1") == "pending"
    assert wa.job_status({"p1": {"status": {"completed": False}}}, "p1") == "running"
    assert wa.job_status({"p1": {"status": {"completed": True}}}, "p1") == "completed"
    assert (
        wa.job_status({"p1": {"status": {"status_str": "error"}}}, "p1") == "error"
    )


def test_extract_output_image_success_pending_and_error() -> None:
    spec = _registry().default
    image = {"filename": "out.png", "subfolder": "", "type": "output"}
    history = {
        "p1": {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {"10": {"images": [image]}},
        }
    }

    assert wa.extract_output_image(spec, history, "p1") == image
    assert wa.extract_output_image(spec, {}, "p1") is None

    failed = {
        "p1": {
            "status": {"status_str": "error", "messages": ["out of memory"]},
            "outputs": {},
        }
    }
    with pytest.raises(wa.ComfyUIJobError, match="out of memory"):
        wa.extract_output_image(spec, failed, "p1")
