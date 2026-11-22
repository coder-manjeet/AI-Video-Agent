"""FastMCP server ``local-render-agent`` for final render jobs.

All production work lives in MCP-independent services. This module only
validates arguments, starts a persistent job in a worker thread and exposes
polling, cancellation, report retrieval and read-only output validation. A job
creation call returns immediately with a job ID; FFmpeg and QA never run on the
MCP event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

import final_qa
from project_paths import PathValidationError, validate_identifier
from project_service import ProjectService
from render_job_models import RenderConfig, RenderJobError
from render_job_store import RenderJobStore
from render_orchestrator import RenderOrchestrationError, RenderOrchestrator

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECTS_ROOT_ENV = "PROJECT_AGENT_PROJECTS_ROOT"
STORE_ROOT_ENV = "RENDER_JOB_STORE_ROOT"

_EXPECTED_ERRORS = (
    RenderJobError,
    RenderOrchestrationError,
    PathValidationError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
)


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _error(exc: BaseException) -> dict[str, object]:
    LOGGER.warning("Render tool call failed: %s", exc)
    payload: dict[str, object] = {
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
    }
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        payload["code"] = code
    return payload


def _validate_job_id(job_id: str) -> str | None:
    if not isinstance(job_id, str) or not job_id.strip():
        return "job_id must be a non-empty string"
    try:
        validate_identifier(job_id, "job_id")
    except PathValidationError as exc:
        return str(exc)
    return None


# --- service wiring ------------------------------------------------------


@dataclass(frozen=True)
class _Services:
    store: RenderJobStore
    orchestrator: RenderOrchestrator
    projects: ProjectService


_SERVICES: _Services | None = None
_TASKS: dict[str, asyncio.Task[None]] = {}
_VALIDATED: dict[str, dict[str, object]] = {}


def _projects_root() -> Path:
    configured = os.getenv(PROJECTS_ROOT_ENV)
    root = Path(configured) if configured else PROJECT_ROOT / "projects"
    return root.expanduser().resolve()


def _get_services() -> _Services:
    global _SERVICES
    if _SERVICES is None:
        projects = ProjectService(_projects_root())
        store = RenderJobStore()
        orchestrator = RenderOrchestrator(projects, store=store)
        _SERVICES = _Services(store=store, orchestrator=orchestrator, projects=projects)
        LOGGER.info("Render services initialised")
    return _SERVICES


# --- task lifecycle ------------------------------------------------------


def _task_done(job_id: str, task: asyncio.Task[None]) -> None:
    if _TASKS.get(job_id) is task:
        _TASKS.pop(job_id, None)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        LOGGER.error("Unexpected render task failure for %s: %s", job_id, exc)


def _start_job(orchestrator: RenderOrchestrator, job_id: str) -> None:
    """Start one render in a worker thread and keep a strong task reference."""
    if job_id in _TASKS:
        return

    async def _worker() -> None:
        await asyncio.to_thread(orchestrator.execute_job, job_id)

    task = asyncio.create_task(_worker(), name=f"render-job-{job_id}")
    _TASKS[job_id] = task
    task.add_done_callback(lambda done: _task_done(job_id, done))


async def _shutdown_jobs() -> None:
    """Request cooperative cancellation for in-flight jobs on server exit."""
    services = _get_services()
    if services is not None:
        for job_id in list(_TASKS):
            services.store.request_cancel(job_id)
    active = [task for task in _TASKS.values() if not task.done()]
    for task in active:
        task.cancel()
    if active:
        await asyncio.gather(*active, return_exceptions=True)
    _TASKS.clear()
    LOGGER.info("Shut down %d active render task(s)", len(active))


@asynccontextmanager
async def _lifespan(_server: FastMCP[None]) -> AsyncIterator[None]:
    yield
    await _shutdown_jobs()


mcp = FastMCP("local-render-agent", lifespan=_lifespan)


# --- tools ---------------------------------------------------------------


@mcp.tool()
async def create_render_job(
    project_id: str,
    config: dict[str, Any] | None = None,
    render_id: str | None = None,
    auto_start: bool = True,
) -> dict[str, object]:
    """Create a persistent render job and optionally start it in the background."""
    if not isinstance(project_id, str) or not project_id.strip():
        return {"status": "error", "error": "project_id must be a non-empty string"}
    try:
        validate_identifier(project_id, "project_id")
        if render_id is not None:
            validate_identifier(render_id, "render_id")
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("config must be an object")
        if not isinstance(auto_start, bool):
            raise TypeError("auto_start must be a boolean")
        services = _get_services()
        result = services.orchestrator.create_job_status(project_id, config, render_id=render_id)
        if result.get("status") != "ok":
            return result
        job = result["job"]
        if not isinstance(job, Mapping):
            raise TypeError("orchestrator returned an invalid job")
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise TypeError("orchestrator returned a job without an ID")
        if auto_start:
            _start_job(services.orchestrator, job_id)
    except _EXPECTED_ERRORS as exc:
        return _error(exc)
    return {"status": "queued", "job": dict(job), "job_id": job_id}


@mcp.tool()
async def get_render_job(job_id: str) -> dict[str, object]:
    """Return the current persistent state of one render job."""
    problem = _validate_job_id(job_id)
    if problem is not None:
        return {"status": "error", "error": problem}
    try:
        job = _get_services().store.get_job(job_id)
        if job is None:
            return _error(RenderJobError("render.job.not_found", f"Unknown job: {job_id}", job_id))
        return {"status": "ok", "job": job.to_dict()}
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


@mcp.tool()
async def list_render_jobs(project_id: str | None = None) -> dict[str, object]:
    """List render jobs, optionally filtered by project."""
    if project_id is not None:
        problem = _validate_job_id(project_id)
        if problem is not None:
            return {"status": "error", "error": problem}
    try:
        jobs = _get_services().store.list_jobs(project_id=project_id)
        return {"status": "ok", "jobs": [job.to_dict() for job in jobs]}
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


@mcp.tool()
async def cancel_render_job(job_id: str) -> dict[str, object]:
    """Request cooperative cancellation; safety is checked between stages."""
    problem = _validate_job_id(job_id)
    if problem is not None:
        return {"status": "error", "error": problem}
    try:
        cancelled = _get_services().store.request_cancel(job_id)
        if not cancelled:
            job = _get_services().store.get_job(job_id)
            if job is None:
                return _error(
                    RenderJobError("render.job.not_found", f"Unknown job: {job_id}", job_id)
                )
            return {
                "status": "ok",
                "cancelled": False,
                "reason": "job is already terminal",
                "job": job.to_dict(),
            }
        return {"status": "ok", "cancelled": True, "job_id": job_id}
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


@mcp.tool()
async def get_render_report(job_id: str) -> dict[str, object]:
    """Read the machine-readable production report for a render job."""
    problem = _validate_job_id(job_id)
    if problem is not None:
        return {"status": "error", "error": problem}
    try:
        services = _get_services()
        job = services.store.get_job(job_id)
        if job is None:
            return _error(RenderJobError("render.job.not_found", f"Unknown job: {job_id}", job_id))
        relative = job.output_paths.get("report_json")
        if not isinstance(relative, str) or not relative:
            return {
                "status": "error",
                "error": "render.report.not_ready",
                "detail": "No production report was generated for this job",
            }
        root = services.projects.project_dir(job.project_id)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return {
                "status": "error",
                "error": "render.report.missing",
                "detail": f"Production report is missing: {relative}",
            }
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("production report must contain a JSON object")
        return {"status": "ok", "report": dict(payload)}
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


@mcp.tool()
async def validate_render_output(job_id: str) -> dict[str, object]:
    """Run a read-only QA pass on the finished MP4 off the MCP event loop."""
    problem = _validate_job_id(job_id)
    if problem is not None:
        return {"status": "error", "error": problem}
    try:
        services = _get_services()
        job = services.store.get_job(job_id)
        if job is None:
            return _error(RenderJobError("render.job.not_found", f"Unknown job: {job_id}", job_id))
        relative_video = job.output_paths.get("video")
        if not isinstance(relative_video, str) or not relative_video:
            return {
                "status": "error",
                "error": "render.output.not_ready",
                "detail": "This job has no validated final video",
            }
        root = services.projects.project_dir(job.project_id)
        video = (root / relative_video).resolve()
        if not video.is_relative_to(root) or not video.is_file():
            return {
                "status": "error",
                "error": "render.output.missing",
                "detail": f"Final video is missing: {relative_video}",
            }
        config = RenderConfig.from_dict(job.config)
        subtitles = [
            root / path
            for key in ("subtitle_srt", "subtitle_vtt")
            if isinstance(path := job.output_paths.get(key), str) and path
        ]
        expected: dict[str, object] = {
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "video_codec": "h264",
            "pixel_format": config.pixel_format,
            "audio_codec": config.audio_codec,
            "sample_rate": config.audio_sample_rate,
            "channels": config.audio_channels,
            "language": config.subtitle_language,
            "burn_in_verified": False,
        }
        thresholds = final_qa.QAThresholds.from_dict(
            {
                "duration_tolerance_s": config.duration_tolerance_s,
                "sync_tolerance_s": config.sync_tolerance_s,
                "fps_tolerance": config.fps_tolerance,
                "peak_limit": config.peak_limit,
                "min_silence_s": config.min_silence_s,
                "black_min_duration_s": config.black_min_duration_s,
                "ffmpeg_bin": config.ffmpeg_bin,
                "ffprobe_bin": config.ffprobe_bin,
                "ffmpeg_timeout_s": config.ffmpeg_timeout_s,
                "run_detectors": False,
            }
        )

        def _validate() -> dict[str, object]:
            validator = final_qa.FinalQAValidator()
            report = validator.validate(
                video,
                expected=expected,
                subtitle_paths=subtitles,
                burn_in=config.burn_subtitles,
                thresholds=thresholds,
            )
            return report.to_dict()

        report_payload = await asyncio.to_thread(_validate)
        _VALIDATED[job_id] = report_payload
        return {"status": "ok", "validation": report_payload, "validated_at": _utc_now()}
    except _EXPECTED_ERRORS as exc:
        return _error(exc)


if __name__ == "__main__":
    mcp.run(transport="stdio")
