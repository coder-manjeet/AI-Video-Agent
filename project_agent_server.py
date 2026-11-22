"""FastMCP server ``local-project-agent`` for project and scene production.

This process runs under ``.venv`` and exposes thirteen thin tools over stdio.
All project, scene, image and animation logic stays in the existing MCP-independent
services; this module only validates arguments, starts background jobs, tracks
submissions and normalises responses into JSON-compatible dicts.

Generation is deliberately non-blocking: a tool returns a ``submission_id``
immediately while an ``asyncio`` task runs the existing service coroutine. The
existing service job registry remains authoritative for job status, output
paths, errors and warnings.

A single ``asyncio.Lock`` serialises image and video generation **within this
server only**. It does not coordinate GPU work from other MCP processes,
including ``local-tts-agent`` or the legacy ``local-video-agent`` server.
Cross-process locking is intentionally out of scope.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from character_service import CharacterService
from project_paths import validate_identifier
from project_service import ProjectError, ProjectService
from scene_image_service import SceneImageService, SceneImageServiceError
from scene_service import SceneService, SceneServiceError
from scene_video_service import SceneVideoService, SceneVideoServiceError

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECTS_ROOT_ENV = "PROJECT_AGENT_PROJECTS_ROOT"
MAX_TRACKED_SUBMISSIONS = 500

_EXPECTED_ERRORS = (
    ProjectError,
    SceneServiceError,
    SceneImageServiceError,
    SceneVideoServiceError,
    ValueError,
    TypeError,
    OSError,
    KeyError,
)

_IMAGE_FIELDS = (
    "status",
    "job_id",
    "prompt_id",
    "project_id",
    "scene_id",
    "workflow",
    "attempt",
    "image_path",
    "filename",
    "width",
    "height",
    "error",
    "warnings",
)
_VIDEO_FIELDS = (
    "status",
    "job_id",
    "prompt_id",
    "project_id",
    "scene_id",
    "workflow",
    "attempt",
    "output_path",
    "filename",
    "duration_s",
    "width",
    "height",
    "fps",
    "submitted_at",
    "completed_at",
    "error",
    "warnings",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _error(exc: BaseException) -> dict[str, object]:
    """Convert an expected failure into a JSON-serializable tool result."""
    LOGGER.warning("Project operation failed: %s", exc)
    return {"status": "error", "error": _format_error(exc)}


def _reject_empty(**values: object) -> str | None:
    """Return an error message for the first empty/non-string argument."""
    for label, value in values.items():
        if not isinstance(value, str) or not value.strip():
            return f"{label} must be a non-empty string"
    return None


def _validate_ids(project_id: str, scene_id: str | None = None) -> str | None:
    """Apply the existing identifier validation at the tool boundary."""
    for label, value in (("project_id", project_id), ("scene_id", scene_id)):
        if value is None:
            continue
        problem = _reject_empty(**{label: value})
        if problem is not None:
            return problem
        try:
            validate_identifier(value, label)
        except ValueError as exc:
            return str(exc)
    return None


# --- service wiring -----------------------------------------------------


@dataclass(frozen=True)
class _Services:
    projects: ProjectService
    scenes: SceneService
    characters: CharacterService
    image: SceneImageService
    video: SceneVideoService


_SERVICES: _Services | None = None


def _projects_root() -> Path:
    """Return the absolute projects root, with an optional env override."""
    configured = os.getenv(PROJECTS_ROOT_ENV)
    root = Path(configured) if configured else PROJECT_ROOT / "projects"
    return root.expanduser().resolve()


def _get_services() -> _Services:
    """Build the shared service graph once, on first use."""
    global _SERVICES
    if _SERVICES is None:
        projects = ProjectService(_projects_root())
        scenes = SceneService(projects)
        characters = CharacterService(projects)
        image = SceneImageService(projects, scenes, character_service=characters)
        video = SceneVideoService(projects, scenes)
        _SERVICES = _Services(projects, scenes, characters, image, video)
        LOGGER.info("Project services initialised")
    return _SERVICES


# --- submission tracking ------------------------------------------------


@dataclass
class _Submission:
    submission_id: str
    kind: str
    project_id: str
    scene_id: str
    status: str = "submitted"
    job_id: str | None = None
    result: dict[str, object] | None = None
    error: str | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)


_SUBMISSIONS: dict[str, _Submission] = {}
_TASKS: dict[str, asyncio.Task[None]] = {}
_GPU_LOCK: asyncio.Lock | None = None


def _gpu_lock() -> asyncio.Lock:
    """Return the shared in-server GPU lock, created lazily."""
    global _GPU_LOCK
    if _GPU_LOCK is None:
        _GPU_LOCK = asyncio.Lock()
    return _GPU_LOCK


def _prune_submissions() -> None:
    """Drop the oldest finished submissions once the registry is full."""
    if len(_SUBMISSIONS) < MAX_TRACKED_SUBMISSIONS:
        return
    for submission_id in list(_SUBMISSIONS):
        if submission_id in _TASKS:
            continue
        del _SUBMISSIONS[submission_id]
        if len(_SUBMISSIONS) < MAX_TRACKED_SUBMISSIONS:
            return


def _submission_view(sub: _Submission) -> dict[str, object]:
    return {
        "status": "submitted",
        "submission_id": sub.submission_id,
        "job_id": sub.job_id,
        "project_id": sub.project_id,
        "scene_id": sub.scene_id,
        "error": sub.error,
    }


def _task_done(sub: _Submission, task: asyncio.Task[None]) -> None:
    if _TASKS.get(sub.submission_id) is task:
        _TASKS.pop(sub.submission_id, None)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        LOGGER.error(
            "Unexpected error in %s submission %s: %s",
            sub.kind,
            sub.submission_id,
            _format_error(exc),
        )


async def _run_submission(
    sub: _Submission,
    invoke: Callable[[], Coroutine[Any, Any, dict[str, object]]],
) -> None:
    """Run one service generation coroutine under the shared GPU lock."""
    try:
        async with _gpu_lock():
            sub.status = "running"
            sub.updated_at = _utc_now()
            LOGGER.info("Starting %s submission %s", sub.kind, sub.submission_id)
            result = await invoke()
        if not isinstance(result, Mapping):
            raise TypeError("generation service returned a non-object result")
        sub.result = dict(result)
        job_id = result.get("job_id")
        if isinstance(job_id, str) and job_id:
            sub.job_id = job_id
        status = result.get("status")
        sub.status = status if isinstance(status, str) else "completed"
        if sub.status == "error":
            sub.error = str(result.get("error") or "generation failed")
        sub.updated_at = _utc_now()
        LOGGER.info(
            "%s submission %s finished with status %s",
            sub.kind,
            sub.submission_id,
            sub.status,
        )
    except asyncio.CancelledError:
        sub.status = "cancelled"
        sub.error = None
        sub.updated_at = _utc_now()
        LOGGER.info("%s submission %s was cancelled", sub.kind, sub.submission_id)
        raise
    except Exception as exc:  # noqa: BLE001 - any service failure must become structured JSON
        sub.status = "error"
        sub.error = _format_error(exc)
        sub.updated_at = _utc_now()
        LOGGER.warning("%s submission %s failed: %s", sub.kind, sub.submission_id, exc)


def _known_job_ids(service: Any, project_id: str, scene_id: str) -> set[str]:
    try:
        jobs = service.list_jobs(project_id=project_id, scene_id=scene_id)
    except Exception as exc:  # noqa: BLE001 - job inspection must not break launch
        LOGGER.warning("Could not inspect job registry: %s", exc)
        return set()
    return {
        str(job["job_id"])
        for job in jobs
        if isinstance(job, Mapping) and isinstance(job.get("job_id"), str)
    }


def _link_service_job(sub: _Submission, service: Any, known: set[str]) -> None:
    """Best-effort early link to the service job created by a running task."""
    try:
        jobs = service.list_jobs(project_id=sub.project_id, scene_id=sub.scene_id)
    except Exception as exc:  # noqa: BLE001 - completion still links the job
        LOGGER.warning("Could not inspect job registry: %s", exc)
        return
    for job in reversed(list(jobs)):
        job_id = job.get("job_id")
        if isinstance(job_id, str) and job_id and job_id not in known:
            sub.job_id = job_id
            return


def _submit(
    sub: _Submission,
    invoke: Callable[[], Coroutine[Any, Any, dict[str, object]]],
) -> None:
    """Register a submission and keep a strong reference to its task."""
    _SUBMISSIONS[sub.submission_id] = sub
    _prune_submissions()
    task = asyncio.create_task(
        _run_submission(sub, invoke),
        name=f"project-agent-{sub.kind}-{sub.submission_id}",
    )
    _TASKS[sub.submission_id] = task
    task.add_done_callback(lambda done: _task_done(sub, done))


def _find_submission(
    kind: str,
    submission_id: str | None = None,
    job_id: str | None = None,
) -> _Submission | None:
    if submission_id is not None:
        sub = _SUBMISSIONS.get(submission_id)
        return sub if sub is not None and sub.kind == kind else None
    if job_id is None:
        return None
    for sub in reversed(list(_SUBMISSIONS.values())):
        if sub.kind == kind and sub.job_id == job_id:
            return sub
    return None


def _job_view(
    kind: str,
    sub: _Submission | None,
    record: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Merge submission state with the authoritative service job record."""
    if sub is not None and sub.result is not None:
        source: Mapping[str, Any] = sub.result
    elif record is not None:
        source = record
    else:
        source = {}
    fields = _IMAGE_FIELDS if kind == "image" else _VIDEO_FIELDS
    view: dict[str, object] = {
        key: source[key] for key in fields if key in source and source[key] is not None
    }
    view["submission_id"] = sub.submission_id if sub is not None else None
    view["job_id"] = source.get("job_id") or (sub.job_id if sub is not None else None)
    view["project_id"] = source.get("project_id") or (sub.project_id if sub else None)
    view["scene_id"] = source.get("scene_id") or (sub.scene_id if sub else None)
    status = source.get("status") or (sub.status if sub is not None else "unknown")
    # A task-level exception is authoritative when the service registry did not
    # already record a failure; otherwise preserve the service's own status.
    if (
        sub is not None
        and sub.status in {"error", "cancelled"}
        and status not in {"failed", "error", "timeout", "cancelled"}
    ):
        status = sub.status
    view["status"] = status
    warnings = source.get("warnings")
    view["warnings"] = list(warnings) if isinstance(warnings, (list, tuple)) else []
    error = source.get("error") or (sub.error if sub is not None else None)
    if error is not None:
        view["error"] = error
    return view


async def _shutdown_submissions() -> None:
    """Cancel and drain generation tasks when the MCP server shuts down."""
    active = [(sid, task) for sid, task in _TASKS.items() if not task.done()]
    for sid, task in active:
        task.cancel()
        sub = _SUBMISSIONS.get(sid)
        if sub is not None:
            sub.status = "cancelled"
            sub.updated_at = _utc_now()
    if active:
        await asyncio.gather(*(task for _, task in active), return_exceptions=True)
    LOGGER.info("Shut down %d active generation task(s)", len(active))


@asynccontextmanager
async def _lifespan(_server: FastMCP[None]) -> AsyncIterator[None]:
    yield
    await _shutdown_submissions()


mcp = FastMCP("local-project-agent", lifespan=_lifespan)


# --- helpers shared by tools -------------------------------------------


def _project_summaries(service: ProjectService) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for project_id in service.list_projects():
        try:
            project = service.load_project(project_id)
        except Exception as exc:  # noqa: BLE001 - one bad manifest must not hide the rest
            summaries.append(
                {
                    "project_id": project_id,
                    "status": "error",
                    "error": _format_error(exc),
                }
            )
            continue
        summaries.append(
            {
                "project_id": project_id,
                "title": project.title,
                "status": project.status,
            }
        )
    return summaries


def _scene_options(
    *,
    title: str,
    duration: float,
    scene_id: str | None,
    description: str,
    narration: str,
    image_prompt: str,
    animation_prompt: str,
    negative_prompt: str,
    character_references: list[str] | None,
    status: str,
) -> dict[str, Any]:
    return {
        "title": title,
        "duration": duration,
        "scene_id": scene_id,
        "description": description,
        "narration": narration,
        "image_prompt": image_prompt,
        "animation_prompt": animation_prompt,
        "negative_prompt": negative_prompt,
        "character_references": character_references,
        "status": status,
    }


def _image_options(
    *,
    workflow: str | None,
    positive_prompt: str | None,
    negative_prompt: str | None,
    width: int | None,
    height: int | None,
    seed: int | None,
    steps: int | None,
    cfg: float | None,
    regenerate: bool,
    compose_prompt: bool,
    prompt_language: str | None,
    character_references: list[str] | None,
    visual_style: str | None,
    color_palette: str | None,
    character_appearance: str | None,
    timeout_s: float | None,
    poll_interval_s: float | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "workflow": workflow,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "regenerate": regenerate,
        "compose_prompt": compose_prompt,
        "prompt_language": prompt_language,
        "character_references": character_references,
        "visual_style": visual_style,
        "color_palette": color_palette,
        "character_appearance": character_appearance,
    }
    if timeout_s is not None:
        options["timeout_s"] = timeout_s
    if poll_interval_s is not None:
        options["poll_interval_s"] = poll_interval_s
    return options


def _video_options(
    *,
    workflow: str | None,
    animation_prompt: str | None,
    negative_prompt: str | None,
    width: int | None,
    height: int | None,
    frames: int | None,
    fps: float | None,
    seed: int | None,
    regenerate: bool,
    allow_default_prompt: bool,
    timeout_s: float | None,
    poll_interval_s: float | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "workflow": workflow,
        "animation_prompt": animation_prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "seed": seed,
        "regenerate": regenerate,
        "allow_default_prompt": allow_default_prompt,
    }
    if timeout_s is not None:
        options["timeout_s"] = timeout_s
    if poll_interval_s is not None:
        options["poll_interval_s"] = poll_interval_s
    return options


def _retry_options(timeout_s: float | None, poll_interval_s: float | None) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if timeout_s is not None:
        options["timeout_s"] = timeout_s
    if poll_interval_s is not None:
        options["poll_interval_s"] = poll_interval_s
    return options


# --- project tools ------------------------------------------------------


@mcp.tool()
async def create_project(
    project_id: str,
    title: str,
    description: str = "",
    language: str = "en",
    status: str = "active",
    settings: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Create a new project with its directory scaffold and ``project.json``.

    Returns ``status="created"`` with the project ID and full manifest. Existing
    projects are never overwritten; expected failures use ``status="error"``.
    """
    problem = _validate_ids(project_id)
    if problem is not None:
        return _error(ValueError(problem))
    if not isinstance(title, str) or not title.strip():
        return _error(ValueError("title must be a non-empty string"))
    try:
        service = _get_services().projects
        project = await asyncio.to_thread(
            service.create_project,
            project_id,
            title,
            description=description,
            language=language,
            status=status,
            settings=settings,
        )
        return {
            "status": "created",
            "project_id": project.project_id,
            "project": project.to_dict(),
        }
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


@mcp.tool()
async def list_projects() -> dict[str, object]:
    """List projects that have a valid ``project.json`` manifest.

    Returns ``status="ok"`` and a ``projects`` array with IDs, titles and
    statuses. A corrupt manifest is reported per item without hiding others.
    """
    try:
        service = _get_services().projects
        summaries = await asyncio.to_thread(_project_summaries, service)
        return {"status": "ok", "projects": summaries}
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


@mcp.tool()
async def get_project(project_id: str) -> dict[str, object]:
    """Load one project manifest by ID.

    Returns ``status="ok"`` with the full project manifest, or a structured
    error when the project is missing or its manifest is invalid.
    """
    problem = _validate_ids(project_id)
    if problem is not None:
        return _error(ValueError(problem))
    try:
        service = _get_services().projects
        project = await asyncio.to_thread(service.load_project, project_id)
        return {"status": "ok", "project_id": project.project_id, "project": project.to_dict()}
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


# --- scene tools --------------------------------------------------------


@mcp.tool()
async def add_scene(
    project_id: str,
    title: str,
    duration: float = 5.0,
    scene_id: str | None = None,
    description: str = "",
    narration: str = "",
    image_prompt: str = "",
    animation_prompt: str = "",
    negative_prompt: str = "",
    character_references: list[str] | None = None,
    status: str = "draft",
) -> dict[str, object]:
    """Append a scene to a project and renumber the scene manifest.

    An omitted ``scene_id`` is generated by the scene service. Returns
    ``status="created"`` with the new scene ID and full scene record.
    """
    problem = _validate_ids(project_id, scene_id)
    if problem is not None:
        return _error(ValueError(problem))
    if not isinstance(title, str) or not title.strip():
        return _error(ValueError("title must be a non-empty string"))
    try:
        service = _get_services().scenes
        options = _scene_options(
            title=title,
            duration=duration,
            scene_id=scene_id,
            description=description,
            narration=narration,
            image_prompt=image_prompt,
            animation_prompt=animation_prompt,
            negative_prompt=negative_prompt,
            character_references=character_references,
            status=status,
        )
        scene = await asyncio.to_thread(service.add_scene, project_id, **options)
        return {
            "status": "created",
            "scene_id": scene.scene_id,
            "scene": scene.to_dict(),
        }
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


@mcp.tool()
async def list_scenes(project_id: str) -> dict[str, object]:
    """List a project's scenes in manifest order.

    Returns ``status="ok"`` and a ``scenes`` array of full scene records.
    """
    problem = _validate_ids(project_id)
    if problem is not None:
        return _error(ValueError(problem))
    try:
        service = _get_services().scenes
        scenes = await asyncio.to_thread(service.list_scenes, project_id)
        return {
            "status": "ok",
            "project_id": project_id,
            "scenes": [scene.to_dict() for scene in scenes],
        }
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


@mcp.tool()
async def update_scene(
    project_id: str,
    scene_id: str,
    title: str | None = None,
    description: str | None = None,
    duration: float | None = None,
    narration: str | None = None,
    image_prompt: str | None = None,
    animation_prompt: str | None = None,
    negative_prompt: str | None = None,
    status: str | None = None,
    character_ids: list[str] | None = None,
    character_references: list[str] | None = None,
) -> dict[str, object]:
    """Update supplied scene fields while preserving all other scene data.

    Only non-null arguments are applied. Immutable fields such as
    ``scene_number`` and ``timeline_start`` are not exposed. Returns
    ``status="updated"`` with the full updated scene record.
    """
    problem = _validate_ids(project_id, scene_id)
    if problem is not None:
        return _error(ValueError(problem))
    changes = {
        key: value
        for key, value in (
            ("title", title),
            ("description", description),
            ("duration", duration),
            ("narration", narration),
            ("image_prompt", image_prompt),
            ("animation_prompt", animation_prompt),
            ("negative_prompt", negative_prompt),
            ("status", status),
            ("character_ids", character_ids),
            ("character_references", character_references),
        )
        if value is not None
    }
    if not changes:
        return _error(ValueError("at least one scene field must be supplied"))
    try:
        service = _get_services().scenes
        scene = await asyncio.to_thread(service.update_scene, project_id, scene_id, changes)
        return {
            "status": "updated",
            "project_id": project_id,
            "scene_id": scene.scene_id,
            "scene": scene.to_dict(),
        }
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


# --- image generation tools --------------------------------------------


async def _require_scene(project_id: str, scene_id: str) -> dict[str, object] | None:
    try:
        await asyncio.to_thread(_get_services().scenes.get_scene, project_id, scene_id)
    except _EXPECTED_ERRORS as exc:
        return _error(exc)
    return None


@mcp.tool()
async def generate_scene_image(
    project_id: str,
    scene_id: str,
    workflow: str | None = None,
    positive_prompt: str | None = None,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    regenerate: bool = False,
    compose_prompt: bool = False,
    prompt_language: str | None = None,
    character_references: list[str] | None = None,
    visual_style: str | None = None,
    color_palette: str | None = None,
    character_appearance: str | None = None,
    timeout_s: float | None = None,
    poll_interval_s: float | None = None,
) -> dict[str, object]:
    """Submit one scene image generation job without blocking the caller.

    Returns immediately with ``status="submitted"`` and a ``submission_id``.
    Poll :func:`get_scene_image_job` with that ID (or with the service
    ``job_id``) for status, output path, errors and warnings. Generation is
    serialised with video generation inside this server.
    """
    problem = _validate_ids(project_id, scene_id)
    if problem is not None:
        return _error(ValueError(problem))
    invalid = await _require_scene(project_id, scene_id)
    if invalid is not None:
        return invalid
    try:
        service = _get_services().image
        options = _image_options(
            workflow=workflow,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            regenerate=regenerate,
            compose_prompt=compose_prompt,
            prompt_language=prompt_language,
            character_references=character_references,
            visual_style=visual_style,
            color_palette=color_palette,
            character_appearance=character_appearance,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
        sub = _Submission(
            submission_id=uuid.uuid4().hex,
            kind="image",
            project_id=project_id,
            scene_id=scene_id,
        )
        known = _known_job_ids(service, project_id, scene_id)
        _submit(
            sub,
            lambda: service.generate_scene_image(project_id, scene_id, **options),
        )
        # Yield once so the task can register its service job, then link that
        # job ID for live polling. This is best-effort; completion always links.
        await asyncio.sleep(0)
        _link_service_job(sub, service, known)
        return _submission_view(sub)
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


@mcp.tool()
async def get_scene_image_job(
    submission_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, object]:
    """Return the current state of one scene image generation job.

    Provide a ``submission_id`` from ``generate_scene_image`` /
    ``retry_scene_image`` or an existing service ``job_id``. Unknown IDs return
    a structured error. No generation is started by this tool.
    """
    if not submission_id and not job_id:
        return _error(ValueError("submission_id or job_id is required"))
    sub = _find_submission("image", submission_id, job_id)
    effective_job_id = job_id or (sub.job_id if sub is not None else None)
    record = None
    if effective_job_id:
        record = _get_services().image.get_job(effective_job_id)
    if sub is None and record is None:
        return _error(LookupError(f"Unknown image job or submission: {job_id or submission_id}"))
    return _job_view("image", sub, record)


@mcp.tool()
async def retry_scene_image(
    project_id: str,
    scene_id: str,
    timeout_s: float | None = None,
    poll_interval_s: float | None = None,
) -> dict[str, object]:
    """Retry the most recent failed or timed-out image job for one scene.

    Returns immediately with a new ``submission_id``. The image service creates
    a **new** service job ID for each retry; the response of
    :func:`get_scene_image_job` reports the current one.
    """
    problem = _validate_ids(project_id, scene_id)
    if problem is not None:
        return _error(ValueError(problem))
    invalid = await _require_scene(project_id, scene_id)
    if invalid is not None:
        return invalid
    try:
        service = _get_services().image
        candidates = [
            job
            for job in service.list_jobs(project_id=project_id, scene_id=scene_id)
            if job.get("status") in {"failed", "timeout"}
        ]
        if not candidates:
            return _error(
                LookupError(
                    f"No failed image job for scene {scene_id!r} in project {project_id!r}"
                )
            )
        options = _retry_options(timeout_s, poll_interval_s)
        sub = _Submission(
            submission_id=uuid.uuid4().hex,
            kind="image",
            project_id=project_id,
            scene_id=scene_id,
        )
        known = _known_job_ids(service, project_id, scene_id)
        _submit(
            sub,
            lambda: service.retry_scene_image(project_id, scene_id, **options),
        )
        await asyncio.sleep(0)
        _link_service_job(sub, service, known)
        return _submission_view(sub)
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


# --- video generation tools --------------------------------------------


@mcp.tool()
async def generate_scene_video(
    project_id: str,
    scene_id: str,
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
) -> dict[str, object]:
    """Submit one scene animation job without blocking the caller.

    Returns immediately with ``status="submitted"`` and a ``submission_id``.
    Poll :func:`get_scene_video_job` for status, output path, errors and
    warnings. Generation is serialised with image generation inside this
    server; other MCP processes are not coordinated by this lock.
    """
    problem = _validate_ids(project_id, scene_id)
    if problem is not None:
        return _error(ValueError(problem))
    invalid = await _require_scene(project_id, scene_id)
    if invalid is not None:
        return invalid
    try:
        service = _get_services().video
        options = _video_options(
            workflow=workflow,
            animation_prompt=animation_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            seed=seed,
            regenerate=regenerate,
            allow_default_prompt=allow_default_prompt,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
        sub = _Submission(
            submission_id=uuid.uuid4().hex,
            kind="video",
            project_id=project_id,
            scene_id=scene_id,
        )
        known = _known_job_ids(service, project_id, scene_id)
        _submit(
            sub,
            lambda: service.generate_scene_video(project_id, scene_id, **options),
        )
        await asyncio.sleep(0)
        _link_service_job(sub, service, known)
        return _submission_view(sub)
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


@mcp.tool()
async def get_scene_video_job(
    submission_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, object]:
    """Return the current state of one scene animation job.

    Provide a ``submission_id`` from ``generate_scene_video`` /
    ``retry_scene_video`` or an existing service ``job_id``. Unknown IDs return
    a structured error. No generation is started by this tool.
    """
    if not submission_id and not job_id:
        return _error(ValueError("submission_id or job_id is required"))
    sub = _find_submission("video", submission_id, job_id)
    effective_job_id = job_id or (sub.job_id if sub is not None else None)
    record = None
    if effective_job_id:
        record = _get_services().video.get_job(effective_job_id)
    if sub is None and record is None:
        return _error(LookupError(f"Unknown video job or submission: {job_id or submission_id}"))
    return _job_view("video", sub, record)


@mcp.tool()
async def retry_scene_video(
    project_id: str,
    scene_id: str,
    timeout_s: float | None = None,
    poll_interval_s: float | None = None,
) -> dict[str, object]:
    """Retry the most recent failed, timed-out or cancelled animation job.

    Returns immediately with a new ``submission_id``. The video service reuses
    the existing service job ID and increments its ``attempt`` counter.
    """
    problem = _validate_ids(project_id, scene_id)
    if problem is not None:
        return _error(ValueError(problem))
    invalid = await _require_scene(project_id, scene_id)
    if invalid is not None:
        return invalid
    try:
        service = _get_services().video
        candidates = [
            job
            for job in service.list_jobs(project_id=project_id, scene_id=scene_id)
            if job.get("status") in {"failed", "timeout", "cancelled"}
        ]
        if not candidates:
            return _error(
                LookupError(
                    f"No failed video job for scene {scene_id!r} in project {project_id!r}"
                )
            )
        options = _retry_options(timeout_s, poll_interval_s)
        sub = _Submission(
            submission_id=uuid.uuid4().hex,
            kind="video",
            project_id=project_id,
            scene_id=scene_id,
        )
        known = _known_job_ids(service, project_id, scene_id)
        _submit(
            sub,
            lambda: service.retry_scene_video(project_id, scene_id, **options),
        )
        await asyncio.sleep(0)
        _link_service_job(sub, service, known)
        return _submission_view(sub)
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


@mcp.tool()
async def cancel_scene_video(
    job_id: str | None = None,
    submission_id: str | None = None,
) -> dict[str, object]:
    """Request cancellation of a scene animation job.

    Accepts a service ``job_id`` or a ``submission_id``. Before a service job
    exists, cancelling the submission cancels the tracked task. Once a job
    exists, the service's ComfyUI cancellation is used and its ``supported``
    and ``error`` fields are preserved.
    """
    if not job_id and not submission_id:
        return _error(ValueError("job_id or submission_id is required"))
    sub = _find_submission("video", submission_id, job_id)
    if submission_id is not None and sub is None:
        return _error(LookupError(f"Unknown video submission: {submission_id}"))
    effective_job_id = job_id or (sub.job_id if sub is not None else None)
    if effective_job_id is None:
        if sub is None:
            return _error(LookupError(f"Unknown video job: {job_id}"))
        task = _TASKS.get(sub.submission_id)
        if task is None or task.done():
            return {
                "status": sub.status,
                "supported": True,
                "job_id": None,
                "submission_id": sub.submission_id,
                "error": "Job is no longer active",
            }
        task.cancel()
        sub.status = "cancelled"
        sub.updated_at = _utc_now()
        return {
            "status": "cancelled",
            "supported": True,
            "job_id": None,
            "submission_id": sub.submission_id,
            "error": None,
        }
    try:
        service = _get_services().video
        result = await service.cancel_job(effective_job_id)
        response = dict(result)
        response["job_id"] = effective_job_id
        if sub is not None:
            response["submission_id"] = sub.submission_id
        return response
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


if __name__ == "__main__":
    mcp.run(transport="stdio")
