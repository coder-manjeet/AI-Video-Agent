"""Independent, sequential AI image generation for project scenes.

The service owns no MCP endpoints and performs no video work.  ComfyUI calls
are injectable so unit tests never need a GPU.  Scene manifests are written
exactly once per scene, after the downloaded image has been verified; failures
are recorded in the in-memory job registry only.
"""

from __future__ import annotations

import asyncio
import logging
import re
import struct
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

import comfyui_client
from character_models import CharacterSceneOverride
from character_service import CharacterNotFoundError, CharacterService
from project_paths import (
    PathValidationError,
    resolve_within_project,
    to_project_relative,
)
from project_service import ProjectService
from scene_service import SceneService
from visual_prompt_builder import (
    PromptCharacter,
    PromptContext,
    PromptResult,
    VisualPromptBuilder,
)
from workflow_adapter import (
    ComfyUIJobError,
    GenerationOptions,
    WorkflowSpec,
    WorkflowValidationError,
    extract_output_image,
    job_status,
    load_workflow_registry,
    patch_workflow,
    unsupported_options,
)
from workflow_adapter import validate_workflow as validate_workflow_payload

logger = logging.getLogger(__name__)

IMAGES_RELATIVE_DIR = Path("assets") / "images"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SUBMIT_ERRORS = (httpx.HTTPError, KeyError, ValueError, TypeError, OSError, RuntimeError)


class SceneImageServiceError(RuntimeError):
    """Base error for scene image generation failures."""


class ImageOutputError(SceneImageServiceError):
    """Raised when a ComfyUI output image is missing or invalid."""


class GenerationTimeout(SceneImageServiceError):
    """Raised when a job does not finish within the configured timeout."""


class WorkflowError(SceneImageServiceError):
    """Raised when the configured image workflow cannot be used."""


class JobNotFoundError(SceneImageServiceError):
    """Raised when a job ID is unknown."""


class ComfyClient(Protocol):
    """The subset of :mod:`comfyui_client` used by this service."""

    def load_workflow_file(self, path: str | Path) -> dict[str, Any]: ...

    async def submit_workflow(self, workflow: Mapping[str, Any]) -> str: ...

    async def get_prompt_history(self, prompt_id: str) -> dict[str, Any]: ...

    async def download_output_image(
        self,
        image: Mapping[str, Any],
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path: ...


@dataclass
class ImageJob:
    """Mutable record of one generation attempt."""

    job_id: str
    project_id: str
    scene_id: str
    workflow: str
    status: str
    created_at: str
    updated_at: str
    prompt_id: str | None = None
    attempt: int = 1
    error: str | None = None
    image_path: str | None = None
    filename: str | None = None
    width: int | None = None
    height: int | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _scene_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:32] or "scene"


def _verify_png(path: Path, expected: tuple[int, int] | None) -> tuple[int, int]:
    """Verify a PNG signature and its IHDR dimensions."""
    with path.open("rb") as file:
        header = file.read(24)
    if len(header) < 24 or header[:8] != PNG_SIGNATURE:
        raise ImageOutputError(f"Generated file is not a readable PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ImageOutputError(f"Generated PNG has invalid dimensions: {path}")
    if expected is not None and (width, height) != expected:
        raise ImageOutputError(
            f"Generated PNG is {width}x{height}, expected {expected[0]}x{expected[1]}"
        )
    return width, height


class SceneImageService:
    """Generate and track scene images with sequential ComfyUI jobs."""

    def __init__(
        self,
        project_service: ProjectService,
        scene_service: SceneService,
        *,
        client: ComfyClient | None = None,
        registry_path: str | Path | None = None,
        character_service: CharacterService | None = None,
        poll_interval_s: float = 1.0,
        timeout_s: float = 300.0,
        max_retries: int = 1,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.projects = project_service
        self.scenes = scene_service
        self.characters = (
            character_service
            if character_service is not None
            else CharacterService(project_service)
        )
        self.prompt_builder = VisualPromptBuilder()
        self.client: ComfyClient = client if client is not None else comfyui_client
        self.registry = load_workflow_registry(registry_path)
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._jobs: dict[str, ImageJob] = {}
        self._lock = asyncio.Lock()

    # --- introspection -------------------------------------------------

    def list_workflows(self) -> list[dict[str, Any]]:
        """Return the available workflow specifications as JSON-safe dicts."""
        return [
            {
                "name": name,
                "display_name": self.registry.get(name).display_name,
                "capabilities": dict(self.registry.get(name).capabilities),
                "defaults": dict(self.registry.get(name).defaults),
            }
            for name in self.registry.names()
        ]

    def validate_workflow(self, workflow: str | None = None) -> dict[str, Any]:
        """Validate a configured workflow file without starting a GPU job."""
        spec = self.registry.get(workflow)
        path = self.registry.workflow_path(spec)
        try:
            payload = self.client.load_workflow_file(path)
        except (OSError, ValueError, TypeError) as exc:
            raise WorkflowError(f"Could not load workflow file {path}: {exc}") from exc
        errors = validate_workflow_payload(spec, payload)
        return {
            "valid": not errors,
            "workflow": spec.name,
            "workflow_file": path.name,
            "errors": errors,
        }

    def list_missing_images(self, project_id: str) -> list[str]:
        """Return scene IDs whose image is unset or whose file is missing."""
        missing: list[str] = []
        project_root = self.projects.project_dir(project_id)
        for scene in self.scenes.load_scenes(project_id):
            if not scene.image_path:
                missing.append(scene.scene_id)
                continue
            try:
                path = resolve_within_project(project_root, scene.image_path, "image_path")
            except PathValidationError:
                missing.append(scene.scene_id)
                continue
            if not path.is_file():
                missing.append(scene.scene_id)
        return missing

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a job record by ID, or None when unknown."""
        job = self._jobs.get(job_id)
        return asdict(job) if job is not None else None

    def list_jobs(
        self,
        *,
        project_id: str | None = None,
        scene_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return job records, optionally filtered by project or scene."""
        return [
            asdict(job)
            for job in self._jobs.values()
            if (project_id is None or job.project_id == project_id)
            and (scene_id is None or job.scene_id == scene_id)
        ]

    # --- generation ----------------------------------------------------

    async def generate_scene_image(
        self,
        project_id: str,
        scene_id: str,
        *,
        workflow: str | None = None,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        regenerate: bool = False,
        character_references: Sequence[str] | None = None,
        visual_style: str | None = None,
        color_palette: str | None = None,
        character_appearance: str | None = None,
        compose_prompt: bool = False,
        prompt_language: str | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> dict[str, Any]:
        """Generate one scene image; never overwrite another scene's image.

        ``compose_prompt`` is opt-in: when false, the supplied or scene prompt
        is used verbatim, preserving the original direct-generation behaviour.
        When true, project visual identity, character identities and scene
        overrides are composed into the prompt (prompt-level consistency only).
        """
        async with self._lock:
            spec = self.registry.get(workflow)
            scene = self.scenes.get_scene(project_id, scene_id)
            project_root = self.projects.project_dir(project_id)

            composition: PromptResult | None = None
            if compose_prompt:
                composition = self._compose_prompt(
                    project_id,
                    scene,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    language=prompt_language,
                )
                effective_prompt: str | None = composition.positive_prompt
                effective_negative: str | None = composition.negative_prompt or None
            else:
                effective_prompt = positive_prompt
                effective_negative = negative_prompt

            options = self._build_options(
                spec,
                scene,
                positive_prompt=effective_prompt,
                negative_prompt=effective_negative,
                width=width,
                height=height,
                seed=seed,
                steps=steps,
                cfg=cfg,
            )

            metadata = {
                "character_references": list(character_references or []),
                "visual_style": visual_style,
                "color_palette": color_palette,
                "character_appearance": character_appearance,
                "compose_prompt": compose_prompt,
                "prompt_composition": (
                    composition.to_dict() if composition is not None else None
                ),
            }
            warnings = self._collect_warnings(spec, options, metadata)

            existing = self._existing_image(project_root, scene.image_path)
            if existing is not None and not regenerate:
                job = self._new_job(project_id, scene_id, spec, status="skipped")
                job.warnings = warnings + [
                    (
                        "Scene already has a verified image; pass regenerate=True "
                        "to create a new version."
                    )
                ]
                job.error = "image_exists"
                self._jobs[job.job_id] = job
                return self._job_result(
                    job,
                    status="skipped",
                    error="image_exists",
                    image_path=scene.image_path,
                    warnings=job.warnings,
                )

            images_dir = project_root / IMAGES_RELATIVE_DIR
            used_paths = {
                item.image_path
                for item in self.scenes.load_scenes(project_id)
                if item.image_path
            }
            filename = self._choose_filename(scene, images_dir, used_paths)
            destination = images_dir / filename
            relative_path = to_project_relative(destination, project_root)

            job = self._new_job(project_id, scene_id, spec, status="queued")
            job.metadata = metadata
            job.options = {
                "workflow": spec.name,
                "positive_prompt": options.positive_prompt,
                "negative_prompt": options.negative_prompt,
                "width": options.width,
                "height": options.height,
                "seed": options.seed,
                "steps": options.steps,
                "cfg": options.cfg,
                "regenerate": regenerate,
                # Preserve consistency metadata across retries.  The composed
                # prompt is already resolved into positive_prompt, so retries
                # must not compose a second time.
                "character_references": list(character_references or []),
                "visual_style": visual_style,
                "color_palette": color_palette,
                "character_appearance": character_appearance,
                "compose_prompt": False,
                "prompt_language": prompt_language,
            }
            self._jobs[job.job_id] = job

            try:
                payload = self._load_payload(spec)
                patched = patch_workflow(
                    spec,
                    payload,
                    options,
                    filename_prefix=Path(filename).stem,
                )
            except (WorkflowValidationError, WorkflowError, ValueError, OSError) as exc:
                return self._fail_job(job, status="failed", error=str(exc), warnings=warnings)

            prompt_id = await self._submit_with_retry(job, patched, warnings)
            if prompt_id is None:
                return self._job_result(
                    job, status=job.status, error=job.error, warnings=warnings
                )

            job.prompt_id = prompt_id
            job.status = "running"
            job.updated_at = _utc_now()

            effective_timeout = timeout_s if timeout_s is not None else self.timeout_s
            effective_poll = (
                poll_interval_s if poll_interval_s is not None else self.poll_interval_s
            )
            try:
                image = await self._wait_for_image(
                    spec,
                    prompt_id,
                    timeout_s=effective_timeout,
                    poll_interval_s=effective_poll,
                )
                await self.client.download_output_image(image, destination)
                expected = (
                    (options.width, options.height)
                    if options.width is not None and options.height is not None
                    else None
                )
                width_result, height_result = _verify_png(destination, expected)
            except GenerationTimeout as exc:
                return self._fail_job(job, status="timeout", error=str(exc), warnings=warnings)
            except (ComfyUIJobError, ImageOutputError, OSError, ValueError) as exc:
                return self._fail_job(job, status="failed", error=str(exc), warnings=warnings)

            self.scenes.update_scene(
                project_id,
                scene_id,
                {"image_path": relative_path, "status": "image_generated"},
            )
            job.status = "completed"
            job.updated_at = _utc_now()
            job.error = None
            job.image_path = relative_path
            job.filename = filename
            job.width = width_result
            job.height = height_result
            job.warnings = warnings
            logger.info(
                "Generated image for scene %s in project %s: %s",
                scene_id,
                project_id,
                relative_path,
            )
            return self._job_result(job, status="completed", warnings=warnings)

    async def generate_scene_images(
        self,
        project_id: str,
        *,
        scene_ids: Sequence[str] | None = None,
        workflow: str | None = None,
        regenerate: bool = False,
        **options: Any,
    ) -> list[dict[str, Any]]:
        """Generate images sequentially; continue past individual failures."""
        targets = (
            list(scene_ids)
            if scene_ids is not None
            else self.list_missing_images(project_id)
        )
        results: list[dict[str, Any]] = []
        for scene_id in targets:
            results.append(
                await self.generate_scene_image(
                    project_id,
                    scene_id,
                    workflow=workflow,
                    regenerate=regenerate,
                    **options,
                )
            )
        return results

    async def retry_job(
        self,
        job_id: str,
        *,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> dict[str, Any]:
        """Retry a failed or timed-out job using its recorded options."""
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Unknown image job: {job_id}")
        if job.status not in {"failed", "timeout"}:
            raise SceneImageServiceError(
                f"Job {job_id} cannot be retried from status {job.status!r}"
            )
        options = dict(job.options)
        workflow = options.pop("workflow", None)
        return await self.generate_scene_image(
            job.project_id,
            job.scene_id,
            workflow=workflow,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            **options,
        )

    async def retry_scene_image(
        self,
        project_id: str,
        scene_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Retry the most recent failed or timed-out job for one scene."""
        candidates = [
            job
            for job in reversed(list(self._jobs.values()))
            if job.project_id == project_id
            and job.scene_id == scene_id
            and job.status in {"failed", "timeout"}
        ]
        if not candidates:
            raise SceneImageServiceError(
                f"No failed image job for scene {scene_id!r} in project {project_id!r}"
            )
        return await self.retry_job(candidates[0].job_id, **kwargs)

    # --- internals -----------------------------------------------------

    def _compose_prompt(
        self,
        project_id: str,
        scene: Any,
        *,
        positive_prompt: str | None,
        negative_prompt: str | None,
        language: str | None,
    ) -> PromptResult:
        """Load identity data and build a structured prompt for one scene."""
        project = self.projects.load_project(project_id)
        characters: list[PromptCharacter] = []
        for character_id in scene.character_ids:
            try:
                character = self.characters.get_character(project_id, character_id)
            except CharacterNotFoundError:
                logger.warning(
                    "Scene %s references missing character %s in project %s",
                    scene.scene_id,
                    character_id,
                    project_id,
                )
                continue
            override_data = scene.character_overrides.get(character_id)
            override = (
                CharacterSceneOverride.from_dict(override_data)
                if override_data is not None
                else None
            )
            characters.append(PromptCharacter(character=character, override=override))

        context = PromptContext(
            base_prompt=(
                positive_prompt if positive_prompt is not None else scene.image_prompt
            ),
            scene_description=scene.description,
            base_negative_prompt=(
                negative_prompt
                if negative_prompt is not None
                else scene.negative_prompt
            ),
            project_identity=project.visual_identity,
            characters=characters,
            visual_identity_overrides=scene.visual_identity_overrides,
            language=language if language is not None else project.language,
        )
        return self.prompt_builder.build(context)

    def _build_options(
        self,
        spec: WorkflowSpec,
        scene: Any,
        *,
        positive_prompt: str | None,
        negative_prompt: str | None,
        width: int | None,
        height: int | None,
        seed: int | None,
        steps: int | None,
        cfg: float | None,
    ) -> GenerationOptions:
        prompt = positive_prompt if positive_prompt is not None else scene.image_prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Scene {scene.scene_id!r} has no image prompt; supply positive_prompt"
            )
        negative = negative_prompt
        if negative is None:
            negative = scene.negative_prompt or None
        values: dict[str, Any] = {
            "width": width if width is not None else spec.defaults.get("width"),
            "height": height if height is not None else spec.defaults.get("height"),
            "seed": seed if seed is not None else spec.defaults.get("seed"),
            "steps": steps if steps is not None else spec.defaults.get("steps"),
            "cfg": cfg if cfg is not None else spec.defaults.get("cfg"),
        }
        for label in ("width", "height", "steps"):
            value = values[label]
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{label} must be an integer")
            minimum = spec.limits.get(f"min_{label}")
            maximum = spec.limits.get(f"max_{label}")
            if (minimum is not None and value < minimum) or (
                maximum is not None and value > maximum
            ):
                raise ValueError(f"{label} must be between {minimum} and {maximum}")
        seed_value = values["seed"]
        if not isinstance(seed_value, int) or isinstance(seed_value, bool):
            raise TypeError("seed must be an integer")
        cfg_value = values["cfg"]
        if not isinstance(cfg_value, (int, float)) or isinstance(cfg_value, bool):
            raise TypeError("cfg must be a number")
        return GenerationOptions(
            positive_prompt=prompt,
            negative_prompt=negative,
            width=values["width"],
            height=values["height"],
            seed=seed_value,
            steps=values["steps"],
            cfg=float(cfg_value),
        )

    def _collect_warnings(
        self,
        spec: WorkflowSpec,
        options: GenerationOptions,
        metadata: Mapping[str, Any],
    ) -> list[str]:
        warnings = unsupported_options(spec, options)
        if any(metadata["character_references"]):
            warnings.append(
                "Character reference images were recorded as metadata only; this "
                "workflow does not condition on reference images and does not "
                "guarantee character consistency."
            )
        for key, label in (
            ("visual_style", "visual style"),
            ("color_palette", "color palette"),
            ("character_appearance", "character appearance"),
        ):
            if metadata[key]:
                warnings.append(
                    f"The supplied {label} was recorded as metadata only and did "
                    "not affect the generated image."
                )
        return warnings

    def _existing_image(self, project_root: Path, image_path: str | None) -> Path | None:
        if not image_path:
            return None
        try:
            path = resolve_within_project(project_root, image_path, "image_path")
        except PathValidationError:
            return None
        return path if path.is_file() else None

    def _choose_filename(
        self,
        scene: Any,
        images_dir: Path,
        used_paths: set[str],
    ) -> str:
        base = f"scene-{scene.scene_number:03d}-{_scene_slug(scene.title)}"
        for version in range(1, 10000):
            stem = base if version == 1 else f"{base}-v{version}"
            filename = f"{stem}.png"
            relative = (IMAGES_RELATIVE_DIR / filename).as_posix()
            if relative not in used_paths and not (images_dir / filename).exists():
                return filename
        raise ImageOutputError("Could not allocate a unique image filename")

    def _new_job(
        self,
        project_id: str,
        scene_id: str,
        spec: WorkflowSpec,
        *,
        status: str,
    ) -> ImageJob:
        now = _utc_now()
        return ImageJob(
            job_id=uuid.uuid4().hex,
            project_id=project_id,
            scene_id=scene_id,
            workflow=spec.name,
            status=status,
            created_at=now,
            updated_at=now,
        )

    def _load_payload(self, spec: WorkflowSpec) -> dict[str, Any]:
        path = self.registry.workflow_path(spec)
        try:
            return self.client.load_workflow_file(path)
        except (OSError, ValueError, TypeError) as exc:
            raise WorkflowError(f"Could not load workflow file {path}: {exc}") from exc

    async def _submit_with_retry(
        self,
        job: ImageJob,
        workflow: dict[str, Any],
        warnings: list[str],
    ) -> str | None:
        last_error: str | None = None
        for attempt in range(1, self.max_retries + 2):
            job.attempt = attempt
            try:
                prompt_id = await self.client.submit_workflow(workflow)
            except SUBMIT_ERRORS as exc:
                last_error = str(exc)
                logger.warning(
                    "Submission attempt %d for job %s failed: %s",
                    attempt,
                    job.job_id,
                    last_error,
                )
                continue
            if not isinstance(prompt_id, str) or not prompt_id:
                last_error = "ComfyUI returned an empty prompt_id"
                continue
            return prompt_id
        self._fail_job(job, status="failed", error=last_error or "submission failed", warnings=warnings)
        return None

    async def _wait_for_image(
        self,
        spec: WorkflowSpec,
        prompt_id: str,
        *,
        timeout_s: float,
        poll_interval_s: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            history = await self.client.get_prompt_history(prompt_id)
            image = extract_output_image(spec, history, prompt_id)
            if image is not None:
                return image
            if job_status(history, prompt_id) == "error":
                raise ComfyUIJobError(f"ComfyUI job {prompt_id} failed")
            if time.monotonic() >= deadline:
                raise GenerationTimeout(
                    f"Timed out after {timeout_s:g}s waiting for image job {prompt_id}"
                )
            await asyncio.sleep(poll_interval_s)

    def _fail_job(
        self,
        job: ImageJob,
        *,
        status: str,
        error: str | None,
        warnings: list[str],
    ) -> dict[str, Any]:
        job.status = status
        job.error = error
        job.warnings = warnings
        job.updated_at = _utc_now()
        logger.warning("Image job %s %s: %s", job.job_id, status, error)
        return self._job_result(job, status=status, error=error, warnings=warnings)

    def _job_result(
        self,
        job: ImageJob,
        *,
        status: str,
        error: str | None = None,
        image_path: str | None = None,
        warnings: list[str],
    ) -> dict[str, Any]:
        return {
            "status": status,
            "job_id": job.job_id,
            "prompt_id": job.prompt_id,
            "project_id": job.project_id,
            "scene_id": job.scene_id,
            "workflow": job.workflow,
            "attempt": job.attempt,
            "image_path": image_path if image_path is not None else job.image_path,
            "filename": job.filename,
            "width": job.width,
            "height": job.height,
            "error": error,
            "warnings": list(warnings),
        }
