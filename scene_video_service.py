"""Independent, sequential animation generation for project scenes.

Each scene is animated from its approved image and animation prompt using the
verified LTX image-to-video workflow.  ComfyUI and FFmpeg calls are injected,
so unit tests need neither a GPU nor a running server.  A scene manifest is
written only after the downloaded MP4 has been verified; failures leave the
previous successful path and all unrelated scene data untouched.

This module owns no MCP endpoints and performs no final video assembly.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

import httpx

import comfyui_client
import media_service
from project_paths import (
    PathValidationError,
    resolve_within_project,
    to_project_relative,
)
from project_service import ProjectService
from scene_service import SceneNotFoundError, SceneService
from workflow_adapter import (
    ComfyUIJobError,
    GenerationOptions,
    WorkflowSpec,
    WorkflowValidationError,
    extract_output_entry,
    job_status,
    load_workflow_registry,
    patch_workflow,
    unsupported_options,
)
from workflow_adapter import validate_workflow as validate_workflow_payload

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_REGISTRY_PATH = PROJECT_DIR / "video_workflows.json"
VIDEOS_RELATIVE_DIR = Path("video") / "scenes"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"

# Only used when a caller explicitly opts in with allow_default_prompt=True.
DEFAULT_ANIMATION_PROMPT = (
    "Subtle, continuous motion. The subject moves naturally and smoothly while "
    "keeping the same identity, clothing, proportions and background; stable "
    "camera, consistent lighting and no sudden changes."
)
SUBMIT_ERRORS = (httpx.HTTPError, KeyError, ValueError, TypeError, OSError, RuntimeError)

ProbeCallable = Callable[[str | Path], dict[str, Any]]


class SceneVideoServiceError(RuntimeError):
    """Base error for scene animation failures."""


class SceneEligibilityError(SceneVideoServiceError):
    """Raised when a scene cannot be animated in its current state."""


class VideoOutputError(SceneVideoServiceError):
    """Raised when a generated video is missing or invalid."""


class GenerationTimeout(SceneVideoServiceError):
    """Raised when a job does not finish within the configured timeout."""


class WorkflowError(SceneVideoServiceError):
    """Raised when the configured animation workflow cannot be used."""


class ModelAvailabilityError(SceneVideoServiceError):
    """Raised when a required model is not installed in ComfyUI."""


class JobNotFoundError(SceneVideoServiceError):
    """Raised when a job ID is unknown."""


class ComfyClient(Protocol):
    """The subset of :mod:`comfyui_client` used by this service."""

    def load_workflow_file(self, path: str | Path) -> dict[str, Any]: ...

    async def upload_image(self, image_path: str) -> str: ...

    async def submit_workflow(self, workflow: Mapping[str, Any]) -> str: ...

    async def get_prompt_history(self, prompt_id: str) -> dict[str, Any]: ...

    async def download_output_file(
        self,
        output: Mapping[str, Any],
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path: ...


@dataclass
class VideoJob:
    """Mutable record of one animation attempt."""

    job_id: str
    project_id: str
    scene_id: str
    workflow: str
    status: str
    submitted_at: str
    completed_at: str | None = None
    updated_at: str = ""
    prompt_id: str | None = None
    attempt: int = 1
    error: str | None = None
    output_path: str | None = None
    filename: str | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _scene_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:32] or "scene"


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _finite_number(value: object, label: str) -> float:
    """Return a finite numeric probe value or raise a structured error."""
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    raise VideoOutputError(f"Generated video has invalid {label}: {value!r}")


class SceneVideoService:
    """Generate and track scene animations with sequential ComfyUI jobs."""

    def __init__(
        self,
        project_service: ProjectService,
        scene_service: SceneService,
        *,
        client: ComfyClient | None = None,
        registry_path: str | Path | None = None,
        probe_media: ProbeCallable | None = None,
        poll_interval_s: float = 1.0,
        timeout_s: float = 600.0,
        max_retries: int = 1,
        default_animation_prompt: str | None = None,
        allow_default_prompt: bool = False,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if default_animation_prompt is not None and not default_animation_prompt.strip():
            raise ValueError("default_animation_prompt must not be blank")
        self.projects = project_service
        self.scenes = scene_service
        self.client: ComfyClient = client if client is not None else comfyui_client
        self.probe_media: ProbeCallable = (
            probe_media if probe_media is not None else media_service.probe_media
        )
        self.registry = load_workflow_registry(
            registry_path if registry_path is not None else DEFAULT_VIDEO_REGISTRY_PATH
        )
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.allow_default_prompt = allow_default_prompt
        self.default_animation_prompt = (
            default_animation_prompt
            if default_animation_prompt is not None
            else (DEFAULT_ANIMATION_PROMPT if allow_default_prompt else None)
        )
        self._jobs: dict[str, VideoJob] = {}
        self._lock = asyncio.Lock()

    # --- introspection and preflight -----------------------------------

    def list_workflows(self) -> list[dict[str, Any]]:
        """Return animation workflow specifications as JSON-safe dicts."""
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
        """Check node structure and model configuration without a GPU job."""
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

    async def verify_models(self, workflow: str | None = None) -> dict[str, Any]:
        """Check that configured model files are available in ComfyUI."""
        spec = self.registry.get(workflow)
        payload = self._load_payload(spec)
        errors = validate_workflow_payload(spec, payload)
        if errors:
            return {
                "valid": False,
                "workflow": spec.name,
                "errors": ["incompatible workflow: " + "; ".join(errors)],
            }
        try:
            errors = await self._check_model_availability(spec, payload)
        except ModelAvailabilityError as exc:
            errors = [str(exc)]
        return {"valid": not errors, "workflow": spec.name, "errors": errors}

    def validate_scene_eligibility(
        self,
        project_id: str,
        scene_id: str,
        *,
        animation_prompt: str | None = None,
        allow_default_prompt: bool | None = None,
    ) -> dict[str, Any]:
        """Return actionable reasons why a scene cannot be animated."""
        allow_default = (
            self.allow_default_prompt
            if allow_default_prompt is None
            else allow_default_prompt
        )
        errors: list[str] = []
        image_path: Path | None = None
        prompt: str | None = None
        if not self.projects.exists(project_id):
            errors.append(f"Project not found: {project_id!r}")
        try:
            scene = self.scenes.get_scene(project_id, scene_id)
        except (SceneNotFoundError, ValueError, OSError) as exc:
            scene = None
            errors.append(f"Could not load scene {scene_id!r}: {exc}")
        if scene is not None:
            image_path, image_error = self._readable_image(
                self.projects.project_dir(project_id), scene.image_path
            )
            if image_error is not None:
                errors.append(image_error)
            prompt = self._resolve_prompt(scene, animation_prompt, allow_default)
            if prompt is None:
                errors.append(
                    f"Scene {scene_id!r} has no animation prompt; supply "
                    "animation_prompt or allow_default_prompt=True"
                )
        return {
            "valid": not errors,
            "errors": errors,
            "image_path": str(image_path) if image_path is not None else None,
            "animation_prompt": prompt,
        }

    def list_missing_videos(self, project_id: str) -> list[str]:
        """Return scene IDs whose generated video is unset or unreadable."""
        missing: list[str] = []
        project_root = self.projects.project_dir(project_id)
        for scene in self.scenes.load_scenes(project_id):
            path, _ = self._readable_video(project_root, scene.generated_video_path)
            if path is None:
                missing.append(scene.scene_id)
        return missing

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return one job record by ID, or None when unknown."""
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

    async def generate_scene_video(
        self,
        project_id: str,
        scene_id: str,
        *,
        workflow: str | None = None,
        animation_prompt: str | None = None,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        fps: float | None = None,
        seed: int | None = None,
        regenerate: bool = False,
        allow_default_prompt: bool = False,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> dict[str, Any]:
        """Animate one scene without touching any other scene's video."""
        async with self._lock:
            return await self._generate_one(
                project_id,
                scene_id,
                job=None,
                workflow=workflow,
                animation_prompt=animation_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                frames=frames,
                fps=fps,
                seed=seed,
                regenerate=regenerate,
                allow_default_prompt=allow_default_prompt or self.allow_default_prompt,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
            )

    async def generate_scene_videos(
        self,
        project_id: str,
        *,
        scene_ids: Sequence[str] | None = None,
        workflow: str | None = None,
        regenerate: bool = False,
        **options: Any,
    ) -> list[dict[str, Any]]:
        """Animate scenes one at a time; continue past individual failures."""
        targets = (
            list(scene_ids)
            if scene_ids is not None
            else self.list_missing_videos(project_id)
        )
        results: list[dict[str, Any]] = []
        for scene_id in targets:
            results.append(
                await self.generate_scene_video(
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
        """Retry a failed, timed-out or cancelled job with its saved options."""
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Unknown video job: {job_id}")
        if job.status not in {"failed", "timeout", "cancelled"}:
            raise SceneVideoServiceError(
                f"Job {job_id} cannot be retried from status {job.status!r}"
            )
        options = dict(job.options)
        workflow = options.pop("workflow", None)
        async with self._lock:
            return await self._generate_one(
                job.project_id,
                job.scene_id,
                job=job,
                workflow=workflow,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
                **options,
            )

    async def retry_scene_video(
        self,
        project_id: str,
        scene_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Retry the most recent unsuccessful job for one scene."""
        candidates = [
            job
            for job in reversed(list(self._jobs.values()))
            if job.project_id == project_id
            and job.scene_id == scene_id
            and job.status in {"failed", "timeout", "cancelled"}
        ]
        if not candidates:
            raise SceneVideoServiceError(
                f"No failed video job for scene {scene_id!r} in project {project_id!r}"
            )
        return await self.retry_job(candidates[0].job_id, **kwargs)

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Ask ComfyUI to remove a queued job, if the API supports it."""
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Unknown video job: {job_id}")
        canceller = getattr(self.client, "cancel_prompt", None)
        if canceller is None:
            return {
                "status": job.status,
                "supported": False,
                "error": "ComfyUI client does not support prompt cancellation",
            }
        if job.status not in {"queued", "running", "timeout"} or job.prompt_id is None:
            return {
                "status": job.status,
                "supported": True,
                "error": f"Job is not queued or running: {job.status!r}",
            }
        try:
            await canceller(job.prompt_id)
        except SUBMIT_ERRORS as exc:
            return {"status": job.status, "supported": True, "error": str(exc)}
        job.status = "cancelled"
        job.updated_at = _utc_now()
        logger.info("Cancelled video job %s (prompt %s)", job_id, job.prompt_id)
        return {"status": "cancelled", "supported": True, "error": None}

    # --- internals -----------------------------------------------------

    async def _generate_one(
        self,
        project_id: str,
        scene_id: str,
        *,
        job: VideoJob | None,
        workflow: str | None,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
        **raw_options: Any,
    ) -> dict[str, Any]:
        spec = self.registry.get(workflow)
        warnings: list[str] = []
        eligibility = self.validate_scene_eligibility(
            project_id,
            scene_id,
            animation_prompt=raw_options.get("animation_prompt"),
            allow_default_prompt=bool(raw_options.get("allow_default_prompt", False)),
        )
        if not eligibility["valid"]:
            error = "; ".join(str(item) for item in eligibility["errors"])
            if job is None:
                job = self._new_job(project_id, scene_id, spec, status="failed")
                self._jobs[job.job_id] = job
            return self._fail_job(
                job, status="failed", error=error, warnings=warnings
            )

        scene = self.scenes.get_scene(project_id, scene_id)
        project_root = self.projects.project_dir(project_id)
        assert isinstance(eligibility["image_path"], str)
        assert isinstance(eligibility["animation_prompt"], str)
        image_path = Path(str(eligibility["image_path"]))
        animation_prompt = str(eligibility["animation_prompt"])

        previous, _ = self._readable_video(
            project_root, scene.generated_video_path
        )
        regenerate = bool(raw_options.get("regenerate", False))
        if previous is not None and not regenerate:
            if job is None:
                job = self._new_job(project_id, scene_id, spec, status="skipped")
            job.error = "video_exists"
            job.updated_at = _utc_now()
            self._jobs[job.job_id] = job
            return self._job_result(
                job,
                status="skipped",
                error="video_exists",
                output_path=scene.generated_video_path,
                warnings=[
                    (
                        "Scene already has a verified video; pass regenerate=True "
                        "to create a new version."
                    )
                ],
            )

        try:
            options = self._build_options(
                spec,
                animation_prompt=animation_prompt,
                negative_prompt=raw_options.get("negative_prompt"),
                scene_negative_prompt=scene.negative_prompt,
                width=raw_options.get("width"),
                height=raw_options.get("height"),
                frames=raw_options.get("frames"),
                fps=raw_options.get("fps"),
                seed=raw_options.get("seed"),
            )
        except (TypeError, ValueError) as exc:
            if job is None:
                job = self._new_job(project_id, scene_id, spec, status="failed")
                self._jobs[job.job_id] = job
            return self._fail_job(
                job, status="failed", error=str(exc), warnings=warnings
            )
        warnings.extend(unsupported_options(spec, options))

        videos_dir = project_root / VIDEOS_RELATIVE_DIR
        used_paths = {
            item.generated_video_path
            for item in self.scenes.load_scenes(project_id)
            if item.generated_video_path
        }
        filename = self._choose_filename(scene, videos_dir, used_paths)
        destination = videos_dir / filename
        relative_path = to_project_relative(destination, project_root)

        if job is None:
            job = self._new_job(project_id, scene_id, spec, status="queued")
        else:
            job.attempt += 1
            job.status = "queued"
            job.error = None
            job.prompt_id = None
            job.output_path = None
            job.filename = None
            job.duration_s = None
            job.completed_at = None
            job.updated_at = _utc_now()
        job.warnings = warnings
        job.options = {
            "workflow": spec.name,
            "animation_prompt": options.positive_prompt,
            "negative_prompt": options.negative_prompt,
            "width": options.width,
            "height": options.height,
            "frames": options.frames,
            "fps": options.fps,
            "seed": options.seed,
            "regenerate": regenerate,
        }
        self._jobs[job.job_id] = job

        try:
            payload = self._load_payload(spec)
            errors = validate_workflow_payload(spec, payload)
            if errors:
                raise WorkflowError(
                    f"Workflow {spec.name!r} is incompatible: " + "; ".join(errors)
                )
            model_errors = await self._check_model_availability(spec, payload)
            if model_errors:
                raise ModelAvailabilityError("; ".join(model_errors))
            image_name = await self.client.upload_image(str(image_path))
            if not isinstance(image_name, str) or not image_name:
                raise VideoOutputError("ComfyUI returned an empty uploaded image name")
            patched = patch_workflow(
                spec,
                payload,
                options,
                filename_prefix=Path(filename).stem,
                image_name=image_name,
            )
        except (
            WorkflowValidationError,
            WorkflowError,
            ModelAvailabilityError,
            ValueError,
            OSError,
            httpx.HTTPError,
        ) as exc:
            return self._fail_job(job, status="failed", error=str(exc), warnings=warnings)

        prompt_id = await self._submit_with_retry(
            job, patched, warnings, start_attempt=job.attempt
        )
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
            entry = await self._wait_for_video(
                spec,
                prompt_id,
                timeout_s=effective_timeout,
                poll_interval_s=effective_poll,
            )
            await self.client.download_output_file(entry, destination)
            expected_duration = (
                options.frames / options.fps
                if options.frames is not None
                and options.fps is not None
                and options.fps > 0
                else None
            )
            verified = self._verify_video(
                destination,
                expected_resolution=(
                    (options.width, options.height)
                    if options.width is not None and options.height is not None
                    else None
                ),
                expected_fps=options.fps,
                expected_duration_s=expected_duration,
            )
        except GenerationTimeout as exc:
            self._discard(destination)
            return self._fail_job(
                job, status="timeout", error=str(exc), warnings=warnings
            )
        except (
            ComfyUIJobError,
            VideoOutputError,
            OSError,
            ValueError,
            TypeError,
            httpx.HTTPError,
        ) as exc:
            self._discard(destination)
            return self._fail_job(job, status="failed", error=str(exc), warnings=warnings)

        warnings.extend(verified["warnings"])
        current = self.scenes.get_scene(project_id, scene_id)
        provider_options = dict(current.provider_options)
        metadata: dict[str, Any] = {
            "job_id": job.job_id,
            "prompt_id": prompt_id,
            "workflow": spec.name,
            "attempt": job.attempt,
            "submitted_at": job.submitted_at,
            "completed_at": _utc_now(),
            "animation_prompt": options.positive_prompt,
            "negative_prompt": options.negative_prompt,
            "width": options.width,
            "height": options.height,
            "frames": options.frames,
            "fps": options.fps,
            "seed": options.seed,
            "actual_duration_s": verified["duration_s"],
            "output_resolution": verified["resolution"],
            "output_fps": verified["fps"],
            "video_codec": verified["video_codec"],
            "size_bytes": verified["size_bytes"],
        }
        provider_options["video_generation"] = metadata
        self.scenes.update_scene(
            project_id,
            scene_id,
            {
                "generated_video_path": relative_path,
                "status": "video_generated",
                "provider_options": provider_options,
            },
        )
        completed_at = str(metadata["completed_at"])
        job.status = "completed"
        job.completed_at = completed_at
        job.updated_at = completed_at
        job.error = None
        job.output_path = relative_path
        job.filename = filename
        job.duration_s = (
            float(verified["duration_s"])
            if isinstance(verified["duration_s"], (int, float))
            else None
        )
        job.width = (
            int(verified["width"])
            if isinstance(verified["width"], (int, float))
            else None
        )
        job.height = (
            int(verified["height"])
            if isinstance(verified["height"], (int, float))
            else None
        )
        job.fps = (
            float(verified["fps"])
            if isinstance(verified["fps"], (int, float))
            else None
        )
        job.warnings = warnings
        job.metadata = metadata
        logger.info(
            "Generated video for scene %s in project %s: %s",
            scene_id,
            project_id,
            relative_path,
        )
        return self._job_result(job, status="completed", warnings=warnings)

    def _build_options(
        self,
        spec: WorkflowSpec,
        *,
        animation_prompt: str,
        negative_prompt: str | None,
        scene_negative_prompt: str,
        width: object,
        height: object,
        frames: object,
        fps: object,
        seed: object,
    ) -> GenerationOptions:
        if not animation_prompt.strip():
            raise ValueError("animation_prompt must be a non-empty string")
        values: dict[str, Any] = {
            "width": width if width is not None else spec.defaults.get("width"),
            "height": height if height is not None else spec.defaults.get("height"),
            "frames": frames if frames is not None else spec.defaults.get("frames"),
            "fps": fps if fps is not None else spec.defaults.get("fps"),
            "seed": seed if seed is not None else spec.defaults.get("seed"),
        }
        for label in ("width", "height", "frames", "seed"):
            value = values[label]
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{label} must be an integer")
            minimum = spec.limits.get(f"min_{label}")
            maximum = spec.limits.get(f"max_{label}")
            if (minimum is not None and value < minimum) or (
                maximum is not None and value > maximum
            ):
                raise ValueError(f"{label} must be between {minimum} and {maximum}")
        fps_value = values["fps"]
        if not _is_finite_number(fps_value):
            raise TypeError("fps must be a finite number")
        minimum_fps = spec.limits.get("min_fps")
        maximum_fps = spec.limits.get("max_fps")
        if (minimum_fps is not None and fps_value < minimum_fps) or (
            maximum_fps is not None and fps_value > maximum_fps
        ):
            raise ValueError(f"fps must be between {minimum_fps} and {maximum_fps}")
        negative = negative_prompt if negative_prompt is not None else scene_negative_prompt
        return GenerationOptions(
            positive_prompt=animation_prompt,
            negative_prompt=negative or None,
            width=int(values["width"]),
            height=int(values["height"]),
            frames=int(values["frames"]),
            fps=float(fps_value),
            seed=int(values["seed"]),
        )

    def _resolve_prompt(
        self,
        scene: Any,
        supplied: str | None,
        allow_default: bool,
    ) -> str | None:
        candidates = (supplied, getattr(scene, "animation_prompt", None))
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        if allow_default:
            return self.default_animation_prompt or DEFAULT_ANIMATION_PROMPT
        return None

    def _readable_image(
        self, project_root: Path, image_path: str | None
    ) -> tuple[Path | None, str | None]:
        if not image_path:
            return None, "Scene has no approved image_path; generate or register an image first"
        try:
            path = resolve_within_project(project_root, image_path, "image_path")
        except PathValidationError as exc:
            return None, str(exc)
        if not path.is_file():
            return None, f"Scene image does not exist: {image_path}"
        try:
            with path.open("rb") as handle:
                header = handle.read(12)
        except OSError as exc:
            return None, f"Scene image is not readable: {image_path}: {exc}"
        if len(header) < 12 or (
            header[:8] != PNG_SIGNATURE and header[:3] != JPEG_SIGNATURE
        ):
            return None, f"Scene image is not a readable PNG/JPEG: {image_path}"
        return path, None

    def _readable_video(
        self, project_root: Path, video_path: str | None
    ) -> tuple[Path | None, str | None]:
        if not video_path:
            return None, "Scene has no generated_video_path"
        try:
            path = resolve_within_project(project_root, video_path, "generated_video_path")
        except PathValidationError:
            return None, f"Unsafe generated_video_path: {video_path}"
        return (path, None) if path.is_file() and path.stat().st_size > 0 else (None, None)

    def _choose_filename(self, scene: Any, videos_dir: Path, used_paths: set[str]) -> str:
        base = f"scene-{scene.scene_number:03d}-{_scene_slug(scene.title)}"
        for version in range(1, 10000):
            stem = base if version == 1 else f"{base}-v{version}"
            filename = f"{stem}.mp4"
            relative = (VIDEOS_RELATIVE_DIR / filename).as_posix()
            if relative not in used_paths and not (videos_dir / filename).exists():
                return filename
        raise VideoOutputError("Could not allocate a unique video filename")

    def _new_job(
        self,
        project_id: str,
        scene_id: str,
        spec: WorkflowSpec,
        *,
        status: str,
    ) -> VideoJob:
        now = _utc_now()
        return VideoJob(
            job_id=uuid.uuid4().hex,
            project_id=project_id,
            scene_id=scene_id,
            workflow=spec.name,
            status=status,
            submitted_at=now,
            updated_at=now,
        )

    def _load_payload(self, spec: WorkflowSpec) -> dict[str, Any]:
        path = self.registry.workflow_path(spec)
        try:
            return self.client.load_workflow_file(path)
        except (OSError, ValueError, TypeError) as exc:
            raise WorkflowError(f"Could not load workflow file {path}: {exc}") from exc

    async def _check_model_availability(
        self, spec: WorkflowSpec, payload: Mapping[str, Any]
    ) -> list[str]:
        getter = getattr(self.client, "get_object_info", None)
        if getter is None:
            return []
        object_info = await getter()
        if not isinstance(object_info, Mapping):
            raise ModelAvailabilityError("ComfyUI /object_info returned invalid data")
        errors: list[str] = []
        for path in spec.model_inputs:
            node_id, _, input_name = path.rpartition(".")
            node = payload.get(node_id)
            if not isinstance(node, Mapping):
                errors.append(f"model input {path}: missing workflow node")
                continue
            value = node.get("inputs", {}).get(input_name)
            class_type = node.get("class_type")
            node_info = object_info.get(class_type)
            if not isinstance(node_info, Mapping):
                errors.append(f"ComfyUI has no node type {class_type!r} required by {path}")
                continue
            inputs = node_info.get("input", {})
            if not isinstance(inputs, Mapping):
                continue
            definition = inputs.get("required", {}).get(input_name)
            if definition is None and isinstance(inputs.get("optional"), Mapping):
                definition = inputs["optional"].get(input_name)
            allowed = definition[0] if isinstance(definition, (list, tuple)) and definition else None
            if isinstance(allowed, list) and value not in allowed:
                errors.append(f"Model {value!r} for {path} is not available in ComfyUI")
        return errors

    async def _submit_with_retry(
        self,
        job: VideoJob,
        workflow: Mapping[str, Any],
        warnings: list[str],
        *,
        start_attempt: int = 1,
    ) -> str | None:
        last_error: str | None = None
        for offset in range(self.max_retries + 1):
            attempt = start_attempt + offset
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
            job.submitted_at = _utc_now()
            return prompt_id
        self._fail_job(
            job,
            status="failed",
            error=last_error or "submission failed",
            warnings=warnings,
        )
        return None

    async def _wait_for_video(
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
            entry = extract_output_entry(spec, history, prompt_id)
            if entry is not None:
                return entry
            if job_status(history, prompt_id) == "error":
                raise ComfyUIJobError(f"ComfyUI job {prompt_id} failed")
            if time.monotonic() >= deadline:
                raise GenerationTimeout(
                    f"Timed out after {timeout_s:g}s waiting for video job {prompt_id}"
                )
            await asyncio.sleep(poll_interval_s)

    def _verify_video(
        self,
        path: Path,
        *,
        expected_resolution: tuple[int, int] | None,
        expected_fps: float | None,
        expected_duration_s: float | None,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise VideoOutputError(f"Generated video does not exist: {path}")
        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            raise VideoOutputError(f"Generated video is empty: {path}")
        info = self.probe_media(path)
        if not isinstance(info, Mapping) or info.get("status") != "ok":
            raise VideoOutputError(f"Generated video is not readable: {path}")
        video_codec = info.get("video_codec")
        if not isinstance(video_codec, str) or not video_codec:
            raise VideoOutputError(f"Generated video has no video stream: {path}")
        format_name = info.get("format_name")
        if not isinstance(format_name, str) or "mp4" not in format_name.lower():
            raise VideoOutputError(
                f"Generated video is not an MP4 container: {format_name!r}"
            )
        width = int(_finite_number(info.get("width"), "width"))
        height = int(_finite_number(info.get("height"), "height"))
        fps = _finite_number(info.get("fps"), "frame rate")
        duration = _finite_number(info.get("duration_s"), "duration")
        if width <= 0 or height <= 0:
            raise VideoOutputError(f"Generated video has invalid resolution: {path}")
        if duration <= 0:
            raise VideoOutputError(f"Generated video has invalid duration: {path}")
        if fps <= 0:
            raise VideoOutputError(f"Generated video has invalid frame rate: {path}")
        if expected_resolution is not None and (width, height) != expected_resolution:
            raise VideoOutputError(
                f"Generated video is {width}x{height}, expected "
                f"{expected_resolution[0]}x{expected_resolution[1]}"
            )
        warnings: list[str] = []
        if expected_fps is not None and abs(fps - expected_fps) > 0.05:
            warnings.append(
                f"Generated FPS {fps:g} differs from configured {expected_fps:g}"
            )
        if expected_duration_s is not None:
            delta = abs(duration - expected_duration_s)
            if delta > 0.5:
                warnings.append(
                    f"Generated duration {duration:.3f}s differs from "
                    f"frames/fps estimate {expected_duration_s:.3f}s"
                )
        return {
            "duration_s": duration,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "video_codec": video_codec,
            "size_bytes": size_bytes,
            "warnings": warnings,
        }

    @staticmethod
    def _discard(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove invalid video output: %s", path)

    def _fail_job(
        self,
        job: VideoJob,
        *,
        status: str,
        error: str | None,
        warnings: list[str],
    ) -> dict[str, Any]:
        job.status = status
        job.error = error
        job.warnings = warnings
        job.updated_at = _utc_now()
        logger.warning("Video job %s %s: %s", job.job_id, status, error)
        return self._job_result(job, status=status, error=error, warnings=warnings)

    def _job_result(
        self,
        job: VideoJob,
        *,
        status: str,
        error: str | None = None,
        output_path: str | None = None,
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
            "output_path": output_path if output_path is not None else job.output_path,
            "filename": job.filename,
            "duration_s": job.duration_s,
            "width": job.width,
            "height": job.height,
            "fps": job.fps,
            "submitted_at": job.submitted_at,
            "completed_at": job.completed_at,
            "error": error,
            "warnings": list(warnings),
        }
