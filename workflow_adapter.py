"""Configuration-driven adapter for ComfyUI image-generation workflows.

Node IDs live in ``image_workflows.json`` instead of the generation service so
workflows can be swapped without touching business logic.  This module performs
pure, offline compatibility checks and prompt patching; it never talks to
ComfyUI itself.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from project_paths import PathValidationError, resolve_within_project

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = PROJECT_DIR / "image_workflows.json"


class WorkflowConfigError(ValueError):
    """Raised when the workflow registry itself is invalid."""


class WorkflowValidationError(ValueError):
    """Raised when a workflow file is incompatible with its specification."""


class ComfyUIJobError(RuntimeError):
    """Raised when ComfyUI reports a failed generation job."""


@dataclass(frozen=True)
class NodeBinding:
    """A node ID and input name that a generation option maps to."""

    node: str
    input: str
    expected_class: str | None = None


@dataclass(frozen=True)
class OutputBinding:
    """Where to find the generated image in a ComfyUI history entry."""

    node: str
    images_key: str = "images"
    expected_class: str | None = None


@dataclass(frozen=True)
class GenerationOptions:
    """Normalised generation parameters passed to a workflow."""

    positive_prompt: str
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None


@dataclass(frozen=True)
class WorkflowSpec:
    """Complete, validated description of one API-format image workflow."""

    name: str
    display_name: str
    description: str
    workflow_file: str
    expected_classes: dict[str, str]
    required_inputs: dict[str, Any]
    positive_prompt: NodeBinding
    negative_prompt: NodeBinding | None
    width: NodeBinding | None
    height: NodeBinding | None
    seed: NodeBinding | None
    steps: NodeBinding | None
    cfg: NodeBinding | None
    filename_prefix: NodeBinding | None
    output: OutputBinding
    capabilities: dict[str, bool]
    defaults: dict[str, Any]
    limits: dict[str, int]
    notes: tuple[str, ...]


class WorkflowRegistry:
    """An immutable collection of workflow specifications."""

    def __init__(
        self,
        specs: Mapping[str, WorkflowSpec],
        default_name: str,
        base_dir: str | Path,
    ) -> None:
        if not specs:
            raise WorkflowConfigError("workflow registry contains no workflows")
        if default_name not in specs:
            raise WorkflowConfigError(
                f"default workflow {default_name!r} is not in the registry"
            )
        self._specs = dict(specs)
        self.default_name = default_name
        self.base_dir = Path(base_dir).expanduser().resolve()

    @property
    def default(self) -> WorkflowSpec:
        """Return the configured default workflow specification."""
        return self._specs[self.default_name]

    def get(self, name: str | None = None) -> WorkflowSpec:
        """Return a specification by name, or the default when name is None."""
        if name is None:
            return self.default
        try:
            return self._specs[name]
        except KeyError as exc:
            raise WorkflowConfigError(f"Unknown workflow: {name!r}") from exc

    def names(self) -> list[str]:
        """Return the available workflow names in insertion order."""
        return list(self._specs)

    def workflow_path(self, spec: WorkflowSpec) -> Path:
        """Resolve a workflow file to a path inside the registry directory."""
        try:
            return resolve_within_project(
                self.base_dir, spec.workflow_file, "workflow_file"
            )
        except PathValidationError as exc:
            raise WorkflowConfigError(str(exc)) from exc


def _binding(raw: Any, label: str) -> NodeBinding | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise WorkflowConfigError(f"nodes.{label} must be an object or null")
    node = raw.get("node")
    input_name = raw.get("input")
    if not isinstance(node, str) or not node:
        raise WorkflowConfigError(f"nodes.{label}.node must be a non-empty string")
    if not isinstance(input_name, str) or not input_name:
        raise WorkflowConfigError(f"nodes.{label}.input must be a non-empty string")
    expected_class = raw.get("expected_class")
    if expected_class is not None and not isinstance(expected_class, str):
        raise WorkflowConfigError(
            f"nodes.{label}.expected_class must be a string or null"
        )
    return NodeBinding(node, input_name, expected_class)


def _parse_spec(name: str, raw: Any) -> WorkflowSpec:
    if not isinstance(raw, Mapping):
        raise WorkflowConfigError(f"workflows.{name} must be an object")
    workflow_file = raw.get("workflow_file")
    if not isinstance(workflow_file, str) or not workflow_file:
        raise WorkflowConfigError(f"workflows.{name}.workflow_file is required")

    expected_classes = raw.get("expected_classes", {})
    required_inputs = raw.get("required_inputs", {})
    capabilities = raw.get("capabilities", {})
    defaults = raw.get("defaults", {})
    limits = raw.get("limits", {})
    notes = raw.get("notes", [])
    for label, value in (
        ("expected_classes", expected_classes),
        ("required_inputs", required_inputs),
        ("capabilities", capabilities),
        ("defaults", defaults),
        ("limits", limits),
    ):
        if not isinstance(value, Mapping):
            raise WorkflowConfigError(f"workflows.{name}.{label} must be an object")
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        raise WorkflowConfigError(f"workflows.{name}.notes must be a list of strings")

    nodes = raw.get("nodes", {})
    if not isinstance(nodes, Mapping):
        raise WorkflowConfigError(f"workflows.{name}.nodes must be an object")

    output_raw = raw.get("output")
    if not isinstance(output_raw, Mapping):
        raise WorkflowConfigError(f"workflows.{name}.output must be an object")
    output_node = output_raw.get("node")
    images_key = output_raw.get("images_key", "images")
    output_class = output_raw.get("expected_class")
    if not isinstance(output_node, str) or not output_node:
        raise WorkflowConfigError(f"workflows.{name}.output.node is required")
    if not isinstance(images_key, str) or not images_key:
        raise WorkflowConfigError(f"workflows.{name}.output.images_key is required")
    if output_class is not None and not isinstance(output_class, str):
        raise WorkflowConfigError(
            f"workflows.{name}.output.expected_class must be a string or null"
        )

    positive = _binding(nodes.get("positive_prompt"), "positive_prompt")
    if positive is None:
        raise WorkflowConfigError(
            f"workflows.{name}.nodes.positive_prompt is required"
        )

    return WorkflowSpec(
        name=name,
        display_name=str(raw.get("name", name)),
        description=str(raw.get("description", "")),
        workflow_file=workflow_file,
        expected_classes=dict(expected_classes),
        required_inputs=dict(required_inputs),
        positive_prompt=positive,
        negative_prompt=_binding(nodes.get("negative_prompt"), "negative_prompt"),
        width=_binding(nodes.get("width"), "width"),
        height=_binding(nodes.get("height"), "height"),
        seed=_binding(nodes.get("seed"), "seed"),
        steps=_binding(nodes.get("steps"), "steps"),
        cfg=_binding(nodes.get("cfg"), "cfg"),
        filename_prefix=_binding(
            nodes.get("filename_prefix"), "filename_prefix"
        ),
        output=OutputBinding(output_node, images_key, output_class),
        capabilities=dict(capabilities),
        defaults=dict(defaults),
        limits=dict(limits),
        notes=tuple(notes),
    )


def load_workflow_registry(
    path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> WorkflowRegistry:
    """Load and validate ``image_workflows.json`` from disk."""
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    registry_path = registry_path.expanduser().resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(f"Workflow registry not found: {registry_path}")
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkflowConfigError(
            f"Could not read workflow registry {registry_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise WorkflowConfigError("workflow registry must be a JSON object")

    raw_workflows = payload.get("workflows")
    if not isinstance(raw_workflows, Mapping) or not raw_workflows:
        raise WorkflowConfigError("workflow registry must define 'workflows'")
    specs = {
        str(name): _parse_spec(str(name), raw)
        for name, raw in raw_workflows.items()
    }
    default_name = str(payload.get("default_workflow", next(iter(specs))))
    return WorkflowRegistry(
        specs, default_name, base_dir if base_dir is not None else registry_path.parent
    )


def validate_workflow(
    spec: WorkflowSpec, workflow: Mapping[str, Any]
) -> list[str]:
    """Return a list of compatibility problems; an empty list means valid."""
    errors: list[str] = []
    if not isinstance(workflow, Mapping):
        return ["workflow must be a JSON object"]

    for node_id, node in workflow.items():
        if not isinstance(node, Mapping):
            errors.append(f"node {node_id}: not an object")
            continue
        if not isinstance(node.get("class_type"), str):
            errors.append(f"node {node_id}: missing class_type")
        if not isinstance(node.get("inputs"), Mapping):
            errors.append(f"node {node_id}: missing inputs")

    for node_id, expected_class in spec.expected_classes.items():
        node = workflow.get(node_id)
        if node is None:
            errors.append(f"node {node_id}: missing (expected {expected_class})")
            continue
        actual = node.get("class_type")
        if actual != expected_class:
            errors.append(
                f"node {node_id}: class_type is {actual!r}, expected {expected_class!r}"
            )

    for path, expected_value in spec.required_inputs.items():
        node_id, _, input_name = path.rpartition(".")
        node = workflow.get(node_id)
        if node is None:
            errors.append(f"required input {path}: missing node {node_id}")
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping) or input_name not in inputs:
            errors.append(f"required input {path}: missing input")
            continue
        if inputs[input_name] != expected_value:
            errors.append(
                f"required input {path}: got {inputs[input_name]!r}, "
                f"expected {expected_value!r}"
            )

    bindings: list[tuple[str, NodeBinding | None]] = [
        ("positive_prompt", spec.positive_prompt),
        ("negative_prompt", spec.negative_prompt),
        ("width", spec.width),
        ("height", spec.height),
        ("seed", spec.seed),
        ("steps", spec.steps),
        ("cfg", spec.cfg),
        ("filename_prefix", spec.filename_prefix),
    ]
    for label, binding in bindings:
        if binding is None:
            continue
        node = workflow.get(binding.node)
        if node is None:
            errors.append(f"{label}: missing node {binding.node}")
            continue
        if (
            binding.expected_class is not None
            and node.get("class_type") != binding.expected_class
        ):
            errors.append(
                f"{label}: node {binding.node} class_type is "
                f"{node.get('class_type')!r}, expected {binding.expected_class!r}"
            )
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping) or binding.input not in inputs:
            errors.append(f"{label}: input {binding.node}.{binding.input} is missing")

    output_node = workflow.get(spec.output.node)
    if output_node is None:
        errors.append(f"output: missing node {spec.output.node}")
    else:
        if (
            spec.output.expected_class is not None
            and output_node.get("class_type") != spec.output.expected_class
        ):
            errors.append(
                f"output: node {spec.output.node} class_type is "
                f"{output_node.get('class_type')!r}, "
                f"expected {spec.output.expected_class!r}"
            )
        inputs = output_node.get("inputs")
        if not isinstance(inputs, Mapping) or spec.output.images_key not in inputs:
            errors.append(
                f"output: input {spec.output.node}.{spec.output.images_key} is missing"
            )

    return errors


def assert_workflow_valid(
    spec: WorkflowSpec, workflow: Mapping[str, Any]
) -> None:
    """Raise :class:`WorkflowValidationError` when a workflow is incompatible."""
    errors = validate_workflow(spec, workflow)
    if errors:
        raise WorkflowValidationError(
            f"Workflow {spec.name!r} is incompatible: " + "; ".join(errors)
        )


def unsupported_options(
    spec: WorkflowSpec, options: GenerationOptions
) -> list[str]:
    """Describe requested options that this workflow cannot honour."""
    warnings: list[str] = []
    if options.negative_prompt is not None and not spec.capabilities.get(
        "negative_prompt", False
    ):
        warnings.append(
            f"Workflow {spec.name!r} does not support negative prompts; "
            "the supplied negative prompt was ignored."
        )
    for label in ("width", "height", "seed", "steps", "cfg"):
        value = getattr(options, label)
        if value is not None and not spec.capabilities.get(label, False):
            warnings.append(
                f"Workflow {spec.name!r} does not support the {label!r} option; "
                "the supplied value was ignored."
            )
    return warnings


def patch_workflow(
    spec: WorkflowSpec,
    workflow: Mapping[str, Any],
    options: GenerationOptions,
    *,
    filename_prefix: str | None = None,
) -> dict[str, Any]:
    """Return a fresh workflow copy with generation options applied."""
    errors = validate_workflow(spec, workflow)
    if errors:
        raise WorkflowValidationError(
            f"Workflow {spec.name!r} is incompatible: " + "; ".join(errors)
        )
    if not isinstance(options.positive_prompt, str) or not options.positive_prompt:
        raise ValueError("positive_prompt must be a non-empty string")

    patched = copy.deepcopy(dict(workflow))

    def set_input(binding: NodeBinding | None, value: Any) -> None:
        if binding is not None:
            patched[binding.node]["inputs"][binding.input] = value

    set_input(spec.positive_prompt, options.positive_prompt)
    if spec.negative_prompt is not None and options.negative_prompt is not None:
        set_input(spec.negative_prompt, options.negative_prompt)
    set_input(spec.width, options.width)
    set_input(spec.height, options.height)
    set_input(spec.seed, options.seed)
    set_input(spec.steps, options.steps)
    set_input(spec.cfg, options.cfg)
    if filename_prefix is not None:
        set_input(spec.filename_prefix, filename_prefix)
    return patched


def job_status(history: Mapping[str, Any], prompt_id: str) -> str:
    """Map a ComfyUI history entry to ``pending``/``running``/``completed``/``error``."""
    job = history.get(prompt_id) if isinstance(history, Mapping) else None
    if not isinstance(job, Mapping):
        return "pending"
    status = job.get("status", {})
    if not isinstance(status, Mapping):
        return "running"
    if status.get("status_str") == "error":
        return "error"
    if status.get("completed"):
        return "completed"
    return "running"


def extract_output_image(
    spec: WorkflowSpec, history: Mapping[str, Any], prompt_id: str
) -> dict[str, Any] | None:
    """Return the generated image entry, or None while the job is unfinished."""
    status = job_status(history, prompt_id)
    if status == "pending" or status == "running":
        return None
    job = history[prompt_id]
    if status == "error":
        messages = job.get("status", {}).get("messages", [])
        raise ComfyUIJobError(
            f"ComfyUI job {prompt_id} failed: {str(messages)[:1000]}"
        )

    outputs = job.get("outputs", {})
    if not isinstance(outputs, Mapping):
        return None
    preferred = outputs.get(spec.output.node)
    ordered = [preferred] if isinstance(preferred, Mapping) else []
    ordered.extend(
        node_output
        for node_id, node_output in outputs.items()
        if node_id != spec.output.node and isinstance(node_output, Mapping)
    )
    for node_output in ordered:
        images = node_output.get(spec.output.images_key, [])
        if isinstance(images, list):
            for image in images:
                if isinstance(image, Mapping) and image.get("filename"):
                    return dict(image)
    return None
