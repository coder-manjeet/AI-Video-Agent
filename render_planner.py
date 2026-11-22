"""MCP-independent render planning from measured scene media.

The planner reads project and scene manifests but never writes them. Every
generated video is probed with the injected media prober, and the render
timeline is calculated independently from the planned ``timeline_start`` /
``timeline_end`` fields. Missing narration is reported as a warning; no audio is
generated, and planned durations are never used as a fallback for a missing
measurement.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path

import media_service
from manifest_validation import ValidationIssue
from project_paths import PathValidationError, resolve_within_project
from project_service import ProjectService
from render_models import (
    RenderMediaInfo,
    RenderPlan,
    RenderPolicy,
    RenderScene,
    coerce_decimal,
)
from scene_service import SceneService

logger = logging.getLogger(__name__)

ProbeCallable = Callable[[Path], Mapping[str, object]]
PROBE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
)


class RenderPlanningError(ValueError):
    """Structured failure raised when a render plan cannot be built."""

    def __init__(
        self,
        code: str,
        message: str,
        location: str = "",
        *,
        issues: Sequence[ValidationIssue] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location
        self.issue = ValidationIssue(code, message, location)
        self.issues = tuple(issues) if issues else (self.issue,)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class RenderPlanner:
    """Build an independent render timeline from existing scene assets."""

    def __init__(
        self,
        projects: ProjectService,
        scenes: SceneService,
        probe: ProbeCallable | None = None,
    ) -> None:
        self.projects = projects
        self.scenes = scenes
        self.probe: ProbeCallable = (
            probe if probe is not None else media_service.probe_media
        )

    def build_plan(
        self,
        project_id: str,
        policy: RenderPolicy | None = None,
    ) -> RenderPlan:
        """Probe every scene and return a read-only render plan."""
        project = self.projects.load_project(project_id)
        root = self.projects.project_dir(project_id)
        scenes = self.scenes.list_scenes(project_id)
        effective_policy = policy if policy is not None else RenderPolicy()

        warnings: list[ValidationIssue] = []
        entries: list[RenderScene] = []
        cursor = Decimal(0)
        planned_total = Decimal(0)

        for index, scene in enumerate(scenes):
            location = f"scenes[{index}] ({scene.scene_id})"
            planned = coerce_decimal(scene.duration, f"{location}.duration", positive=True)
            planned_total += planned

            video_relative = self._require_asset(
                root,
                scene.generated_video_path,
                "generated_video_path",
                location,
            )
            video_path = self._require_nonempty_file(video_relative, "video", location)
            video_info = RenderMediaInfo.from_probe(video_path, self._probe(video_path, "video", location))
            self._validate_video_metadata(video_info, location)

            narration_path = self._optional_asset(
                root, scene.narration_audio_path, "narration_audio_path", location
            )
            narration_duration, narration_info = self._probe_narration(
                narration_path,
                has_narration_text=bool(scene.narration and scene.narration.strip()),
                location=location,
                warnings=warnings,
            )

            measured = video_info.duration_s
            assert measured is not None
            if narration_duration is None:
                effective = measured
                hold = Decimal(0)
                silent_tail = Decimal(0)
                narration_delta = None
            else:
                effective = max(measured, narration_duration)
                hold = max(Decimal(0), narration_duration - measured)
                silent_tail = max(Decimal(0), measured - narration_duration)
                narration_delta = narration_duration - measured

            start = cursor
            end = start + effective
            cursor = end

            planned_start = coerce_decimal(
                scene.timeline_start, f"{location}.timeline_start", non_negative=True
            )
            planned_end = coerce_decimal(
                scene.timeline_end, f"{location}.timeline_end", non_negative=True
            )
            delta = measured - planned

            entries.append(
                RenderScene(
                    scene_id=scene.scene_id,
                    scene_number=scene.scene_number,
                    title=scene.title,
                    status=scene.status,
                    planned_duration_s=planned,
                    measured_duration_s=measured,
                    effective_duration_s=effective,
                    video_hold_duration_s=hold,
                    silent_tail_duration_s=silent_tail,
                    timeline_start_s=start,
                    timeline_end_s=end,
                    planned_timeline_start_s=planned_start,
                    planned_timeline_end_s=planned_end,
                    duration_delta_s=delta,
                    narration_duration_s=narration_duration,
                    narration_delta_s=narration_delta,
                    narration_audio_path=scene.narration_audio_path,
                    narration_text=scene.narration,
                    generated_video_path=str(scene.generated_video_path),
                    video=video_info,
                    narration=narration_info,
                )
            )

            if delta != 0:
                sign = "+" if delta > 0 else ""
                warnings.append(
                    ValidationIssue(
                        "render.duration.differs_from_plan",
                        "Measured video duration "
                        f"{sign}{self._format(delta)}s differs from planned "
                        f"{self._format(planned)}s",
                        f"{location}.duration",
                    )
                )
            if hold > 0:
                warnings.append(
                    ValidationIssue(
                        "render.duration.video_hold_required",
                        f"Hold the final video frame for {self._format(hold)}s "
                        "so the full narration is heard",
                        f"{location}.narration_audio_path",
                    )
                )
            if silent_tail > 0:
                warnings.append(
                    ValidationIssue(
                        "render.duration.silent_tail",
                        f"Narration ends {self._format(silent_tail)}s before the "
                        "video; neither stream was truncated",
                        f"{location}.narration_audio_path",
                    )
                )

        output_width = self._positive_setting(project.settings.width, "width")
        output_height = self._positive_setting(project.settings.height, "height")
        output_fps = self._positive_setting(project.settings.fps, "fps")

        logger.info(
            "Built render plan for project %s: %d scenes, %.6fs",
            project_id,
            len(entries),
            float(cursor),
        )
        return RenderPlan(
            project_id=project_id,
            policy=effective_policy,
            scenes=tuple(entries),
            warnings=tuple(warnings),
            total_planned_duration_s=planned_total,
            total_render_duration_s=cursor,
            output_width=output_width,
            output_height=output_height,
            output_fps=float(output_fps),
        )

    @staticmethod
    def _format(value: Decimal) -> str:
        return format(value, "f")

    @staticmethod
    def _positive_setting(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RenderPlanningError(
                "render.project.invalid_settings",
                f"project.settings.{label} must be a positive integer",
                "settings",
            )
        return value

    @staticmethod
    def _fail(code: str, message: str, location: str) -> RenderPlanningError:
        return RenderPlanningError(code, message, location)

    def _require_asset(
        self,
        root: Path,
        value: object,
        field: str,
        location: str,
    ) -> Path:
        if value is None or (isinstance(value, str) and not value.strip()):
            raise self._fail(
                f"render.{field}.missing",
                f"Scene has no registered {field}; generate and register it first",
                f"{location}.{field}",
            )
        if not isinstance(value, str):
            raise self._fail(
                f"render.{field}.invalid",
                f"{field} must be a project-relative path string",
                f"{location}.{field}",
            )
        try:
            return resolve_within_project(root, value, field)
        except PathValidationError as exc:
            raise self._fail(
                "render.asset.unsafe_path",
                str(exc),
                f"{location}.{field}",
            ) from exc

    def _optional_asset(
        self,
        root: Path,
        value: object,
        field: str,
        location: str,
    ) -> Path | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if not isinstance(value, str):
            raise self._fail(
                f"render.{field}.invalid",
                f"{field} must be a project-relative path string",
                f"{location}.{field}",
            )
        try:
            return resolve_within_project(root, value, field)
        except PathValidationError as exc:
            raise self._fail(
                "render.asset.unsafe_path",
                str(exc),
                f"{location}.{field}",
            ) from exc

    def _require_nonempty_file(self, path: Path, label: str, location: str) -> Path:
        if not path.is_file():
            raise self._fail(
                f"render.{label}.missing_file",
                f"Generated {label} file does not exist: {path}",
                location,
            )
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise self._fail(
                f"render.{label}.unreadable",
                f"Could not stat {label} file {path}: {exc}",
                location,
            ) from exc
        if size <= 0:
            raise self._fail(
                f"render.{label}.empty_file",
                f"Generated {label} file is empty: {path}",
                location,
            )
        return path

    def _probe(self, path: Path, label: str, location: str) -> Mapping[str, object]:
        try:
            info = self.probe(path)
        except PROBE_ERRORS as exc:
            raise self._fail(
                f"render.{label}.unreadable",
                f"Could not probe {label} media {path}: {exc}",
                location,
            ) from exc
        if not isinstance(info, Mapping):
            raise self._fail(
                f"render.{label}.unreadable",
                f"{label} probe returned a non-mapping result: {type(info).__name__}",
                location,
            )
        status = info.get("status")
        if status is not None and status != "ok":
            detail = info.get("error")
            suffix = f": {detail}" if isinstance(detail, str) and detail else ""
            raise self._fail(
                f"render.{label}.unreadable",
                f"{label} probe returned status {status!r}{suffix}",
                location,
            )
        return info

    def _validate_video_metadata(self, info: RenderMediaInfo, location: str) -> None:
        if not info.video_codec:
            raise self._fail(
                "render.video.missing_stream",
                "Generated video has no video stream",
                location,
            )
        if info.width is None or info.height is None:
            raise self._fail(
                "render.video.invalid_resolution",
                "Generated video has missing or invalid width/height",
                location,
            )
        if info.fps is None:
            raise self._fail(
                "render.video.invalid_fps",
                "Generated video has missing or invalid FPS",
                location,
            )
        if info.duration_s is None:
            raise self._fail(
                "render.video.invalid_duration",
                "Measured video duration is missing or invalid; planned duration "
                "is not a fallback",
                f"{location}.duration",
            )

    def _probe_narration(
        self,
        path: Path | None,
        *,
        has_narration_text: bool,
        location: str,
        warnings: list[ValidationIssue],
    ) -> tuple[Decimal | None, RenderMediaInfo | None]:
        if path is None:
            code = "render.narration.audio_missing" if has_narration_text else "render.narration.none"
            detail = (
                "Scene has narration text but no registered narration_audio_path; "
                "no audio will be fabricated"
                if has_narration_text
                else "Scene has no narration; the render timeline will be silent"
            )
            warnings.append(ValidationIssue(code, detail, f"{location}.narration_audio_path"))
            return None, None
        if not path.is_file():
            warnings.append(
                ValidationIssue(
                    "render.narration.missing_file",
                    f"Registered narration audio file does not exist: {path}",
                    f"{location}.narration_audio_path",
                )
            )
            return None, None
        try:
            size = path.stat().st_size
        except OSError as exc:
            warnings.append(
                ValidationIssue(
                    "render.narration.unreadable",
                    f"Could not stat narration audio {path}: {exc}",
                    f"{location}.narration_audio_path",
                )
            )
            return None, None
        if size <= 0:
            warnings.append(
                ValidationIssue(
                    "render.narration.empty_file",
                    f"Registered narration audio file is empty: {path}",
                    f"{location}.narration_audio_path",
                )
            )
            return None, None
        try:
            info = RenderMediaInfo.from_probe(path, self._probe(path, "narration", location))
        except RenderPlanningError as exc:
            warnings.append(
                ValidationIssue("render.narration.unreadable", exc.message, exc.location)
            )
            return None, None
        if not info.audio_codec:
            warnings.append(
                ValidationIssue(
                    "render.narration.no_audio_stream",
                    "Registered narration file has no audio stream",
                    f"{location}.narration_audio_path",
                )
            )
            return None, info
        if info.duration_s is None:
            warnings.append(
                ValidationIssue(
                    "render.narration.invalid_duration",
                    "Registered narration has no measurable duration",
                    f"{location}.narration_audio_path",
                )
            )
            return None, info
        return info.duration_s, info
