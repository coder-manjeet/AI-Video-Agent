"""MCP-independent orchestration of a complete production render.

The orchestrator never re-implements media work: it validates the project,
reuses the existing render planner and audio production coordinator, generates
subtitles, merges the final MP4, runs final QA and writes the production report.
It updates a persistent render job after each stage so MCP clients can poll a
stable job ID instead of blocking on a long-running request.

Cancellation is cooperative and only observed between stages. An FFmpeg process
already started is never killed from here; intermediates and source assets are
preserved so a failure or cancellation can be diagnosed safely. This service
does not start ComfyUI or TTS jobs, so it performs no GPU generation itself.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import final_qa
import final_renderer
import media_service
import render_report
import subtitle_renderer
import subtitle_service
from audio_production import AudioProductionCoordinator
from manifest_validation import validate_project_metadata, validate_scene_manifest
from project_paths import PathValidationError, to_project_relative, validate_identifier
from project_service import ProjectService
from render_job_models import (
    RENDER_STATUSES,
    RenderConfig,
    RenderJob,
    RenderJobError,
    RenderProgress,
    utc_now,
)
from render_job_store import RenderJobStore
from render_models import RenderPlan, RenderPolicy, RenderValidationSettings
from render_planner import RenderPlanner, RenderPlanningError
from scene_service import SceneService

logger = logging.getLogger(__name__)

PROJECTS_ROOT_ENV = "PROJECT_AGENT_PROJECTS_ROOT"
_DEFAULT_RENDER_ROOT = "video/renders"
_STAGE_PERCENTS: dict[str, float] = {
    "validating": 5.0,
    "planning": 15.0,
    "preparing_video": 25.0,
    "preparing_audio": 45.0,
    "mixing_audio": 55.0,
    "generating_subtitles": 70.0,
    "merging": 82.0,
    "validating_output": 94.0,
}


class RenderOrchestrationError(RuntimeError):
    """Structured failure with a stable code and stage."""

    def __init__(
        self,
        code: str,
        message: str,
        location: str = "",
        *,
        stage: str = "render",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location
        self.stage = stage

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "stage": self.stage,
        }


class _CachedPlanner:
    """Return a plan that was already built, avoiding a second probe pass."""

    def __init__(self, plan: RenderPlan) -> None:
        self.plan = plan

    def build_plan(
        self, project_id: str, policy: RenderPolicy | None = None
    ) -> RenderPlan:
        """Return the cached plan for the audio production coordinator."""
        if project_id != self.plan.project_id:
            raise RenderOrchestrationError(
                "render.plan.project_mismatch",
                "Cached render plan belongs to a different project",
                "project_id",
                stage="planning",
            )
        return self.plan


class RenderOrchestrator:
    """Coordinate one production render and persist its lifecycle."""

    def __init__(
        self,
        projects: ProjectService | None = None,
        scenes: SceneService | None = None,
        *,
        store: RenderJobStore | None = None,
        planner: RenderPlanner | None = None,
        audio: AudioProductionCoordinator | None = None,
        subtitles: subtitle_service.SubtitleService | None = None,
        final_renderer_service: final_renderer.FinalRenderer | None = None,
        qa: final_qa.FinalQAValidator | None = None,
        command_runner: Any | None = None,
        probe: Any | None = None,
    ) -> None:
        if projects is None:
            configured = os.getenv(PROJECTS_ROOT_ENV)
            root = Path(configured) if configured else Path(__file__).resolve().parent / "projects"
            projects = ProjectService(root)
        self.projects = projects
        self.scenes = scenes if scenes is not None else SceneService(projects)
        self.store = store if store is not None else RenderJobStore()
        self.planner = planner
        self.command_runner = command_runner
        self.probe = probe
        self.audio = audio
        self._plan_cache: _CachedPlanner | None = None
        self.subtitles = subtitles or subtitle_service.SubtitleService(
            self.projects, self.scenes
        )
        self.final_renderer = (
            final_renderer_service
            or final_renderer.FinalRenderer(
                self.projects,
                command_runner=command_runner,
                probe=probe,
                check_executables=command_runner is None and probe is None,
            )
        )
        self.qa = qa or final_qa.FinalQAValidator(
            probe=probe, command_runner=command_runner
        )

    # --- public API ------------------------------------------------------

    def create_job(
        self,
        project_id: str,
        config: RenderConfig | Mapping[str, object] | None = None,
        *,
        render_id: str | None = None,
    ) -> RenderJob:
        """Create a queued render job with a validated configuration snapshot."""
        effective = RenderConfig.from_dict(config)
        try:
            return self.store.create_job(project_id, effective, render_id=render_id)
        except PathValidationError as exc:
            raise RenderJobError(
                "render.job.invalid_project", str(exc), str(project_id)
            ) from exc

    def create_job_status(
        self,
        project_id: str,
        config: RenderConfig | Mapping[str, object] | None = None,
        *,
        render_id: str | None = None,
    ) -> dict[str, object]:
        """Return a JSON-compatible result for job creation."""
        try:
            job = self.create_job(project_id, config, render_id=render_id)
        except (RenderJobError, PathValidationError, TypeError, ValueError) as exc:
            return {"status": "error", "error": self._error_payload(exc)}
        return {"status": "ok", "job": job.to_dict()}

    def execute_job(self, job_id: str) -> dict[str, object]:
        """Run one job to completion; safe to call from a worker thread."""
        job = self.store.get_job(job_id)
        if job is None:
            return {
                "status": "error",
                "error": self._error_payload(
                    RenderJobError("render.job.not_found", f"Unknown render job: {job_id}", job_id)
                ),
            }
        if not job.is_active:
            return {"status": job.status, "job": job.to_dict()}

        try:
            validate_identifier(job.job_id, "job_id")
        except PathValidationError as exc:
            return {"status": "error", "error": self._error_payload(exc)}

        with self.store.lock(job.project_id):
            job = self.store.get_job(job_id)
            if job is None or not job.is_active:
                return {"status": job.status if job else "error", "job": job.to_dict() if job else {}}
            job = self.store.save_job(
                RenderJob(
                    **{
                        **job.__dict__,
                        "status": "validating",
                        "stage": "validating",
                        "started_at": job.started_at or utc_now(),
                        "updated_at": utc_now(),
                        "attempt": job.attempt + 1,
                        "error": None,
                    }
                )
            )
            self._record_stage(job, "validating", "started")
            try:
                self._run(job_id)
            except RenderCancelled as exc:
                job = self._finish_cancelled(job_id, str(exc))
            except RenderOrchestrationError as exc:
                job = self._finish_failed(job_id, exc)
            except (RenderPlanningError, PathValidationError) as exc:
                error = RenderOrchestrationError(
                    "render.pipeline.invalid",
                    str(exc),
                    stage=self._current_stage(job_id),
                )
                job = self._finish_failed(job_id, error)
            except Exception as exc:
                logger.exception("Unexpected render failure for job %s", job_id)
                error = RenderOrchestrationError(
                    "render.pipeline.unexpected",
                    f"{type(exc).__name__}: {exc}",
                    stage=self._current_stage(job_id),
                )
                job = self._finish_failed(job_id, error)
        final = self._require_job(job_id)
        return {"status": final.status, "job": final.to_dict()}

    def recover_interrupted(self) -> list[RenderJob]:
        """Mark jobs interrupted by a previous process as terminal."""
        return self.store.recover_interrupted()

    # --- stage pipeline --------------------------------------------------

    def _run(self, job_id: str) -> None:
        job = self._require_job(job_id)
        config = RenderConfig.from_dict(job.config)
        root = self.projects.project_dir(job.project_id)

        self._begin_stage(job_id, "validating")
        project = self.projects.load_project(job.project_id)
        manifest_check = validate_project_metadata(project)
        scene_check = validate_scene_manifest(
            self.scenes.list_scenes(job.project_id), project_root=root
        )
        problems = [issue.message for issue in manifest_check.errors + scene_check.errors]
        if problems:
            raise RenderOrchestrationError(
                "render.project.invalid",
                "; ".join(problems),
                "project",
                stage="validating",
            )
        if not self.scenes.list_scenes(job.project_id):
            raise RenderOrchestrationError(
                "render.project.no_scenes", "Project has no scenes to render", "scenes", stage="validating"
            )

        self._begin_stage(job_id, "planning")
        policy = RenderPolicy(
            width=config.width,
            height=config.height,
            fps=config.fps,
            duration_tolerance_s=config.duration_tolerance_s,
            validation=RenderValidationSettings(fps_tolerance=config.fps_tolerance),
            require_audio=False,
        )
        planner = self.planner or RenderPlanner(
            self.projects,
            self.scenes,
            probe=self._planner_probe(config),
        )
        try:
            plan = planner.build_plan(job.project_id, policy)
        except RenderPlanningError as exc:
            raise RenderOrchestrationError(
                exc.code, exc.message, exc.location, stage="planning"
            ) from exc
        self._plan_cache = _CachedPlanner(plan)
        self._complete_stage(job_id, "planning", f"{plan.scene_count} scenes planned")

        # AudioProductionCoordinator performs scene holds, silent concatenation,
        # narration arrangement, mixing and its own validation. Its sub-stages
        # are reported coarsely: we cannot honestly report FFmpeg percentages.
        self._begin_stage(job_id, "preparing_video")
        self._begin_stage(job_id, "preparing_audio")
        self._begin_stage(job_id, "mixing_audio")
        audio_config = self._audio_config(config, job)
        audio = self._audio_coordinator(config)
        try:
            audio_report = audio.produce(
                job.project_id,
                audio_config,
                render_id=job.render_id,
                include_registered=True,
            )
        except Exception as exc:
            raise RenderOrchestrationError(
                getattr(exc, "code", "render.audio.failed"),
                str(exc),
                getattr(exc, "location", ""),
                stage=getattr(exc, "stage", "mixing_audio"),
            ) from exc
        render_dir = self._resolve_render_dir(root, audio_report.render_dir, job)
        self._complete_stage(
            job_id,
            "mixing_audio",
            f"Silent video and audio master prepared ({audio_report.scene_count if hasattr(audio_report, 'scene_count') else len(audio_report.scene_order)} scenes)",
        )

        self._check_cancel(job_id)
        self._begin_stage(job_id, "generating_subtitles")
        subtitle_config: dict[str, object] = {
            "language": config.subtitle_language,
            "formats": list(config.subtitle_formats),
            "granularity": config.subtitle_granularity,
            "check_fonts": True,
        }
        extra_subtitle = config.extra.get("subtitle")
        if isinstance(extra_subtitle, Mapping):
            subtitle_config.update(extra_subtitle)
        subtitle_result = self.subtitles.generate(
            job.project_id,
            audio_report.timeline,
            subtitle_config,
            render_dir=render_dir,
            render_id=audio_report.render_id,
        )
        subtitle_paths = [
            Path(path) if Path(path).is_absolute() else root / path
            for path in subtitle_result.files.values()
        ]
        self._complete_stage(
            job_id,
            "generating_subtitles",
            f"{subtitle_result.track.segment_count} subtitle cues",
        )

        burn_in_filter: str | None = None
        if config.burn_subtitles:
            primary = subtitle_result.files.get("vtt") or subtitle_result.files.get("srt")
            if primary is None:
                raise RenderOrchestrationError(
                    "render.subtitle.no_file",
                    "Subtitle burn-in was requested but no subtitle file was generated",
                    "subtitles",
                    stage="generating_subtitles",
                )
            burn_in_filter = subtitle_renderer.build_burn_in_filter(
                root / primary, subtitle_result.track.style
            )
            if burn_in_filter is None:  # defensive: renderer returns a string on success
                raise RenderOrchestrationError(
                    "render.subtitle.burn_in_unavailable",
                    "Subtitle burn-in is unavailable in this environment",
                    "subtitles",
                    stage="generating_subtitles",
                )

        self._check_cancel(job_id)
        self._begin_stage(job_id, "merging")
        final_config = final_renderer.FinalRenderConfig.from_dict(
            {
                "width": config.width,
                "height": config.height,
                "fps": config.fps,
                "video_codec": config.video_codec,
                "pixel_format": config.pixel_format,
                "audio_codec": config.audio_codec,
                "audio_bitrate_kbps": config.audio_bitrate_kbps,
                "audio_sample_rate": config.audio_sample_rate,
                "audio_channels": config.audio_channels,
                "container": config.container,
                "preset_x264": config.preset_x264,
                "crf": config.crf,
                "movflags": config.movflags,
                "subtitle_burn_in": config.burn_subtitles,
                "burn_in_filter": burn_in_filter,
                "ffmpeg_bin": config.ffmpeg_bin,
                "ffprobe_bin": config.ffprobe_bin,
                "ffmpeg_timeout_s": config.ffmpeg_timeout_s,
                "ffprobe_timeout_s": config.ffprobe_timeout_s,
                "duration_tolerance_s": config.duration_tolerance_s,
            }
        )
        try:
            final_result = self.final_renderer.render(
                job.project_id,
                render_dir,
                audio_report.silent_video_path,
                audio_report.audio_master_path,
                render_id=audio_report.render_id,
                config=final_config,
                subtitle_path=subtitle_paths[0] if subtitle_paths else None,
                burn_in_filter=burn_in_filter,
            )
        except Exception as exc:
            raise RenderOrchestrationError(
                getattr(exc, "code", "render.merge.failed"),
                str(exc),
                getattr(exc, "location", ""),
                stage=getattr(exc, "stage", "merging"),
            ) from exc
        self._complete_stage(job_id, "merging", final_result.output_path)

        self._check_cancel(job_id)
        self._begin_stage(job_id, "validating_output")
        qa_thresholds = final_qa.QAThresholds.from_dict(
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
            }
        )
        final_path = render_dir / "final.mp4"
        expected: dict[str, object] = {
            "width": final_result.width,
            "height": final_result.height,
            "fps": final_result.fps,
            "video_codec": "h264",
            "pixel_format": config.pixel_format,
            "audio_codec": "aac",
            "sample_rate": config.audio_sample_rate,
            "channels": config.audio_channels,
            "duration_s": float(final_result.duration_s),
            "language": config.subtitle_language,
            "burn_in_verified": False,
        }
        qa_report = self.qa.validate(
            final_path,
            expected=expected,
            timeline=audio_report.timeline,
            subtitle_paths=subtitle_paths,
            burn_in=config.burn_subtitles,
            thresholds=qa_thresholds,
            peak=float(audio_report.mix.peak),
        )
        if not qa_report.valid:
            self.store.save_job(
                self._with_updates(
                    job_id,
                    validation_summary=qa_report.to_dict(),
                    output_paths={"video": final_result.output_path},
                )
            )
            details = "; ".join(issue.message for issue in qa_report.errors)
            raise RenderOrchestrationError(
                "render.qa.fatal",
                f"Final validation failed: {details}",
                final_result.output_path,
                stage="validating_output",
            )
        self._complete_stage(job_id, "validating_output", "QA passed")

        report_payload = render_report.build_render_report(
            project=project,
            config=job["config"] if isinstance(job, Mapping) else job.config,
            job=self._require_job(job_id).to_dict(),
            audio_report=audio_report,
            subtitle_result=subtitle_result,
            final_result=final_result,
            qa_report=qa_report,
            project_root=root,
            plan=plan,
            timeline=audio_report.timeline,
            stages=self._require_job(job_id).stages,
        )
        json_path, markdown_path = render_report.write_render_report(render_dir, report_payload)
        final_job = self.store.save_job(
            self._with_updates(
                job_id,
                status="completed",
                stage="completed",
                progress=RenderProgress(stage="completed", percent=100.0, label="Render complete"),
                completed_at=utc_now(),
                error=None,
                output_paths={
                    "video": final_result.output_path,
                    "subtitle_srt": subtitle_result.files.get("srt", ""),
                    "subtitle_vtt": subtitle_result.files.get("vtt", ""),
                    "audio_master": to_project_relative(
                        root / audio_report.audio_master_path, root
                    ),
                    "silent_video": to_project_relative(
                        root / audio_report.silent_video_path, root
                    ),
                    "report_json": to_project_relative(json_path, root),
                    "report_md": to_project_relative(markdown_path, root),
                },
                validation_summary=qa_report.to_dict(),
            )
        )
        self._record_stage(final_job, "completed", "finished")

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _error_payload(exc: BaseException) -> dict[str, object]:
        code = getattr(exc, "code", type(exc).__name__)
        return {
            "code": code,
            "message": str(exc),
            "type": type(exc).__name__,
        }

    def _require_job(self, job_id: str) -> RenderJob:
        job = self.store.get_job(job_id)
        if job is None:
            raise RenderJobError("render.job.not_found", f"Unknown render job: {job_id}", job_id)
        return job

    def _current_stage(self, job_id: str) -> str:
        job = self.store.get_job(job_id)
        return job.stage if job else "render"

    def _check_cancel(self, job_id: str) -> None:
        job = self._require_job(job_id)
        if job.cancel_requested:
            raise RenderCancelled(job_id)

    def _begin_stage(self, job_id: str, stage: str, label: str = "") -> None:
        self._check_cancel(job_id)
        job = self.store.save_job(
            self._with_updates(
                job_id,
                status=stage,
                stage=stage,
                progress=RenderProgress(
                    stage=stage,
                    percent=_STAGE_PERCENTS.get(stage),
                    label=label or stage.replace("_", " "),
                ),
            )
        )
        self._record_stage(job, stage, "started", label=label)

    def _complete_stage(self, job_id: str, stage: str, label: str = "") -> None:
        job = self.store.save_job(
            self._with_updates(
                job_id,
                stage=stage,
                progress=RenderProgress(
                    stage=stage,
                    percent=_STAGE_PERCENTS.get(stage),
                    label=label or f"{stage} complete",
                ),
            )
        )
        self._record_stage(job, stage, "completed", label=label)

    def _record_stage(
        self, job: RenderJob, stage: str, state: str, *, label: str = ""
    ) -> None:
        entry: dict[str, object] = {
            "stage": stage,
            "state": state,
            "status": job.status,
            "label": label,
            "at": utc_now(),
        }
        stages = list(job.stages) + [entry]
        self.store.save_job(self._with_updates(job.job_id, stages=stages))

    def _with_updates(self, job_id: str, **changes: object) -> RenderJob:
        job = self._require_job(job_id)
        payload: dict[str, object] = dict(job.__dict__)
        payload.update(changes)
        payload["updated_at"] = utc_now()
        return RenderJob(**payload)  # type: ignore[arg-type]

    def _finish_cancelled(self, job_id: str, message: str) -> RenderJob:
        job = self._require_job(job_id)
        error = {"code": "render.job.cancelled", "message": message, "stage": job.stage}
        return self.store.save_job(
            self._with_updates(
                job_id,
                status="cancelled",
                stage="cancelled",
                progress=RenderProgress(stage="cancelled", percent=None, label=message),
                completed_at=utc_now(),
                error=error,
            )
        )

    def _finish_failed(self, job_id: str, error: RenderOrchestrationError) -> RenderJob:
        job = self._require_job(job_id)
        payload = error.to_dict()
        stage = error.stage if error.stage in RENDER_STATUSES else job.stage
        return self.store.save_job(
            self._with_updates(
                job_id,
                status="failed",
                stage=stage,
                progress=RenderProgress(
                    stage=job.stage, percent=None, label="Render failed"
                ),
                completed_at=utc_now(),
                error=payload,
            )
        )

    def _audio_config(
        self, config: RenderConfig, job: RenderJob
    ) -> dict[str, object]:
        return {
            "preparation": {
                "width": config.width,
                "height": config.height,
                "fps": config.fps,
                "video_codec": config.video_codec,
                "pixel_format": config.pixel_format,
                "container": config.container,
                "preset": config.preset_x264,
                "crf": config.crf,
                "movflags": config.movflags,
                "render_root": config.render_root,
                "duration_tolerance_s": config.duration_tolerance_s,
                "ffmpeg_bin": config.ffmpeg_bin,
                "ffprobe_bin": config.ffprobe_bin,
                "ffmpeg_timeout_s": config.ffmpeg_timeout_s,
                "ffprobe_timeout_s": config.ffprobe_timeout_s,
                "policy": {
                    "width": config.width,
                    "height": config.height,
                    "fps": config.fps,
                    "duration_tolerance_s": config.duration_tolerance_s,
                    "validation": {"fps_tolerance": config.fps_tolerance},
                },
            },
            "timeline": {
                "missing_narration_policy": config.missing_narration_policy,
                "duration_tolerance_s": config.duration_tolerance_s,
            },
            "mix": {
                "sample_rate": config.audio_sample_rate,
                "channels": config.audio_channels,
                "duration_tolerance_s": config.duration_tolerance_s,
                "render_root": config.render_root,
                "render_id": job.render_id,
                "ffmpeg_bin": config.ffmpeg_bin,
                "ffprobe_bin": config.ffprobe_bin,
                "ffmpeg_timeout_s": config.ffmpeg_timeout_s,
                "ffprobe_timeout_s": config.ffprobe_timeout_s,
            },
        }

    def _audio_coordinator(self, config: RenderConfig) -> AudioProductionCoordinator:
        if self.audio is not None:
            return self.audio
        probe = self._planner_probe(config)
        runner = self.command_runner
        return AudioProductionCoordinator(
            self.projects,
            self.scenes,
            planner=self._plan_cache,  # type: ignore[arg-type]  # duck-typed cached planner
            command_runner=runner,
            probe=probe,
            check_executables=runner is None and probe is None,
        )

    def _planner_probe(self, config: RenderConfig) -> Any:
        if self.probe is not None:
            return self.probe
        return lambda path, *, timeout_s=None: media_service.probe_media(
            path,
            timeout_s=config.ffprobe_timeout_s if timeout_s is None else timeout_s,
        )

    @staticmethod
    def _resolve_render_dir(root: Path, relative: str, job: RenderJob) -> Path:
        raw = Path(relative)
        target = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
        if not target.is_relative_to(root):
            raise RenderOrchestrationError(
                "render.path.unsafe",
                f"Render directory escapes the project: {relative}",
                "render_dir",
                stage="merging",
            )
        if not target.is_dir():
            raise RenderOrchestrationError(
                "render.render_dir.missing",
                f"Render directory does not exist: {relative}",
                "render_dir",
                stage="merging",
            )
        return target


class RenderCancelled(RuntimeError):
    """Cooperative cancellation requested between pipeline stages."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Render job {job_id} was cancelled")
        self.job_id = job_id
