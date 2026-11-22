"""Additive coordinator for silent video holds and the synchronized WAV master.

This module composes the existing render planner, the new audio timeline, the
asset arrangement/mixing services and the pure FFmpeg helpers from
``scene_preparation_service``. It never edits source videos, project manifests
or existing render directories: every run uses a new versioned directory and
returns a structured report with timing and validation details.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import scene_preparation_service as sps
from audio_arrangement import AudioArrangementService, AudioAssetRegistry
from audio_mixer import AudioMixer
from audio_models import (
    AudioMixConfig,
    AudioPipelineError,
    AudioProductionReport,
    AudioTimelinePolicy,
    MusicSettings,
    SfxSettings,
    ValidationIssue,
    coerce_decimal,
    decimal_text,
)
from audio_timeline import AudioTimeline, AudioTimelinePlanner
from project_paths import (
    PathValidationError,
    resolve_within_project,
    validate_identifier,
    write_json_atomic,
)
from project_service import ProjectService
from render_models import RenderExpectations, RenderPlan, RenderPolicy
from render_planner import RenderPlanner, RenderPlanningError
from render_validation import RenderValidator
from scene_service import SceneService

logger = logging.getLogger(__name__)

ProbeCallable = Callable[..., Mapping[str, object]]
DEFAULT_AUDIO_RENDER_PREFIX = "audio-"
DEFAULT_DURATION_TOLERANCE_S = Decimal("0.5")


@dataclass(frozen=True)
class AudioProductionConfig:
    """Bundled configuration for one coordinated production run."""

    preparation: sps.ScenePreparationConfig
    timeline: AudioTimelinePolicy
    mix: AudioMixConfig

    def __post_init__(self) -> None:
        if not isinstance(self.preparation, sps.ScenePreparationConfig):
            raise TypeError("preparation must be a ScenePreparationConfig")
        if not isinstance(self.timeline, AudioTimelinePolicy):
            raise TypeError("timeline must be an AudioTimelinePolicy")
        if not isinstance(self.mix, AudioMixConfig):
            raise TypeError("mix must be an AudioMixConfig")

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | AudioProductionConfig | None = None,
    ) -> AudioProductionConfig:
        if isinstance(data, AudioProductionConfig):
            return data
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise TypeError("production configuration must be an object")
        return cls(
            preparation=sps.ScenePreparationConfig.from_dict(data.get("preparation")),
            timeline=AudioTimelinePolicy.from_dict(data.get("timeline")),
            mix=AudioMixConfig.from_dict(data.get("mix")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "preparation": self.preparation.to_dict(),
            "timeline": self.timeline.to_dict(),
            "mix": self.mix.to_dict(),
        }


def _fail(
    code: str,
    message: str,
    location: str = "",
    *,
    stage: str = "production",
) -> AudioPipelineError:
    return AudioPipelineError(code, message, location, stage=stage)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AudioProductionCoordinator:
    """Produce held scene videos, a silent concat and the WAV audio master."""

    def __init__(
        self,
        projects: ProjectService | None = None,
        scenes: SceneService | None = None,
        *,
        planner: RenderPlanner | None = None,
        timeline_planner: AudioTimelinePlanner | None = None,
        arrangement: AudioArrangementService | None = None,
        mixer: AudioMixer | None = None,
        validator: RenderValidator | None = None,
        command_runner: sps.CommandRunner | None = None,
        probe: ProbeCallable | None = None,
        check_executables: bool | None = None,
    ) -> None:
        self.projects = projects if projects is not None else ProjectService()
        self.scenes = scenes if scenes is not None else SceneService(self.projects)
        self.planner = planner
        self.timeline_planner = timeline_planner or AudioTimelinePlanner()
        self.command_runner: sps.CommandRunner = (
            command_runner if command_runner is not None else sps.run_ffmpeg
        )
        self.probe: ProbeCallable = (
            probe
            if probe is not None
            else lambda path, *, timeout_s: sps.probe_video(
                path, ffprobe_bin=sps.DEFAULT_FFPROBE_BIN, timeout_s=timeout_s
            )
        )
        self.check_executables = (
            check_executables
            if check_executables is not None
            else command_runner is None and probe is None
        )
        self.validator = validator
        self.arrangement = arrangement or AudioArrangementService(
            self.projects,
            AudioAssetRegistry(self.projects, self.scenes),
            probe=lambda path, *, timeout_s: self.probe(path, timeout_s=timeout_s),
        )
        def run_mixer_command(command: Sequence[str], timeout_s: float) -> None:
            self.command_runner(command, timeout_s)

        self.mixer = mixer or AudioMixer(
            self.projects,
            command_runner=run_mixer_command,
            probe=lambda path, *, timeout_s: self.probe(path, timeout_s=timeout_s),
            check_executables=False,
        )

    def produce_status(
        self,
        project_id: str,
        config: AudioProductionConfig | Mapping[str, Any] | None = None,
        *,
        render_id: str | None = None,
        music: MusicSettings | None = None,
        sfx: Sequence[SfxSettings] = (),
        include_registered: bool = True,
    ) -> dict[str, object]:
        """Return a JSON-serializable status, report or structured error."""
        try:
            report = self.produce(
                project_id,
                config,
                render_id=render_id,
                music=music,
                sfx=sfx,
                include_registered=include_registered,
            )
        except AudioPipelineError as exc:
            return {"status": "error", "error": exc.to_dict()}
        return {"status": "ok", "result": report.to_dict()}

    def produce(
        self,
        project_id: str,
        config: AudioProductionConfig | Mapping[str, Any] | None = None,
        *,
        render_id: str | None = None,
        music: MusicSettings | None = None,
        sfx: Sequence[SfxSettings] = (),
        include_registered: bool = True,
    ) -> AudioProductionReport:
        """Run phases B-E for one project and return the validated report."""
        try:
            settings = AudioProductionConfig.from_dict(config)
        except (TypeError, ValueError) as exc:
            raise _fail("audio.production.config_invalid", str(exc), stage="config") from exc
        try:
            validate_identifier(project_id, "project_id")
        except PathValidationError as exc:
            raise _fail("audio.identifier.invalid", str(exc), "project_id") from exc
        if not self.projects.exists(project_id):
            raise _fail(
                "audio.project.not_found",
                f"Project not found: {project_id}",
                "project_id",
                stage="project",
            )
        self._check_executables(settings.preparation)
        root = self.projects.project_dir(project_id)
        project_manifest = root / "project.json"
        scenes_manifest = root / "script" / "scenes.json"
        hashes_before = {
            "project.json": _sha256(project_manifest),
            "script/scenes.json": _sha256(scenes_manifest),
        }

        render_dir, effective_render_id = self._create_render_dir(
            root, settings.preparation, render_id
        )
        plan = self._build_plan(project_id, settings.preparation)
        timeline = self.timeline_planner.build(plan, settings.timeline)
        arrangement = self.arrangement.build(
            project_id,
            timeline,
            music,
            sfx,
            include_registered=include_registered,
        )

        warnings: list[ValidationIssue] = list(timeline.warnings)
        warnings.extend(arrangement.warnings)
        prepared_paths: list[Path] = []
        hold_durations: dict[str, Decimal] = {}
        for order, scene in enumerate(timeline.scenes):
            prepared = self._prepare_scene(
                root, render_dir, order, scene, settings.preparation, hold_durations
            )
            prepared_paths.append(prepared)

        final_video = self._concatenate(
            render_dir, prepared_paths, timeline, settings.preparation, warnings
        )
        final_video_duration = coerce_decimal(
            self._probe(final_video, settings.preparation).get("duration_s"),
            "final video duration",
            positive=True,
        )
        video_verdict = self._validate_video(
            final_video,
            final_video_duration,
            settings.preparation,
        )
        warnings.extend(video_verdict.warnings)
        if not video_verdict.valid:
            details = "; ".join(issue.message for issue in video_verdict.errors)
            raise _fail(
                "audio.production.video_invalid",
                f"Prepared silent video failed validation: {details}",
                str(final_video),
                stage="video",
            )

        mix_config = self._mix_config(settings, effective_render_id)
        mix = self.mixer.mix(project_id, timeline, arrangement, mix_config)
        warnings.extend(mix.warnings)
        audio_duration = mix.master_duration_s
        delta = abs(final_video_duration - audio_duration)
        if delta > settings.preparation.duration_tolerance_s:
            raise _fail(
                "audio.production.timeline_mismatch",
                f"Silent video duration {decimal_text(final_video_duration)}s "
                f"and audio master duration {decimal_text(audio_duration)}s "
                f"differ by {decimal_text(delta)}s",
                str(render_dir),
                stage="validate",
            )

        hashes_after = {
            "project.json": _sha256(project_manifest),
            "script/scenes.json": _sha256(scenes_manifest),
        }
        if hashes_before != hashes_after:
            raise _fail(
                "audio.production.manifest_mutated",
                "A project or scene manifest changed during production",
                str(root),
                stage="validate",
            )

        validation: dict[str, object] = {
            "video": video_verdict.to_dict(),
            "audio": {
                "codec": "pcm_s16le",
                "sample_rate": mix.sample_rate,
                "channels": mix.channels,
                "duration_s": decimal_text(audio_duration),
                "peak": mix.peak,
                "clipped_samples": mix.clipped_samples,
            },
            "duration_delta_s": decimal_text(delta),
            "manifest_sha256_before": hashes_before,
            "manifest_sha256_after": hashes_after,
            "manifest_preserved": True,
            "timeline_consistency": "ok",
        }
        report = AudioProductionReport(
            project_id=project_id,
            render_id=effective_render_id,
            render_dir=str(render_dir),
            silent_video_path=str(final_video),
            silent_video_duration_s=final_video_duration,
            audio_master_path=mix.master_path,
            audio_duration_s=audio_duration,
            scene_order=tuple(scene.scene_id for scene in timeline.scenes),
            hold_durations_s=hold_durations,
            timeline=timeline,
            mix=mix,
            validation=validation,
            warnings=tuple(warnings),
        )
        report_path = render_dir / "report.json"
        if report_path.exists():
            raise _fail(
                "audio.production.report_collision",
                f"Report already exists: {report_path}",
                str(report_path),
                stage="report",
            )
        try:
            write_json_atomic(report_path, report.to_dict())
        except OSError as exc:
            raise _fail(
                "audio.production.report_failed",
                f"Could not write production report {report_path}: {exc}",
                str(report_path),
                stage="report",
            ) from exc
        logger.info(
            "Completed audio production %s for project %s", effective_render_id, project_id
        )
        return report

    @staticmethod
    def _mix_config(
        settings: AudioProductionConfig, render_id: str
    ) -> AudioMixConfig:
        return replace(
            settings.mix,
            render_root=settings.preparation.render_root,
            render_id=render_id,
            render_dir=(Path(settings.preparation.render_root) / render_id).as_posix(),
            ffmpeg_bin=settings.preparation.ffmpeg_bin,
            ffprobe_bin=settings.preparation.ffprobe_bin,
            ffmpeg_timeout_s=settings.preparation.ffmpeg_timeout_s,
            ffprobe_timeout_s=settings.preparation.ffprobe_timeout_s,
        )

    def _check_executables(self, config: sps.ScenePreparationConfig) -> None:
        if not self.check_executables:
            return
        if shutil.which(config.ffmpeg_bin) is None:
            raise _fail(
                "audio.ffmpeg.not_installed",
                f"FFmpeg executable not found: {config.ffmpeg_bin}",
                stage="ffmpeg",
            )
        if shutil.which(config.ffprobe_bin) is None:
            raise _fail(
                "audio.ffprobe.not_installed",
                f"FFprobe executable not found: {config.ffprobe_bin}",
                stage="ffprobe",
            )

    def _probe(
        self, path: Path, config: sps.ScenePreparationConfig
    ) -> Mapping[str, object]:
        try:
            info = self.probe(path, timeout_s=config.ffprobe_timeout_s)
        except sps.ScenePreparationError as exc:
            raise _fail(
                "audio.production.probe_failed",
                f"Could not probe {path}: {exc.message}",
                str(path),
                stage="probe",
            ) from exc
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise _fail(
                "audio.production.probe_failed",
                f"Could not probe {path}: {exc}",
                str(path),
                stage="probe",
            ) from exc
        if not isinstance(info, Mapping):
            raise _fail(
                "audio.production.probe_failed",
                f"Probe returned {type(info).__name__} instead of a mapping",
                str(path),
                stage="probe",
            )
        return info

    def _build_plan(
        self, project_id: str, config: sps.ScenePreparationConfig
    ) -> RenderPlan:
        planner = self.planner or RenderPlanner(
            self.projects,
            self.scenes,
            probe=lambda path: self._probe(path, config),
        )
        policy = RenderPolicy(
            duration_mode=config.policy.duration_mode,
            require_audio=False,
            width=config.width,
            height=config.height,
            fps=config.fps,
            duration_tolerance_s=config.duration_tolerance_s,
            validation=config.policy.validation,
        )
        try:
            return planner.build_plan(project_id, policy)
        except RenderPlanningError as exc:
            raise _fail(
                "audio.production.plan_failed",
                exc.message,
                exc.location,
                stage="plan",
            ) from exc
        except PathValidationError as exc:
            raise _fail("audio.production.unsafe_path", str(exc), stage="plan") from exc

    def _create_render_dir(
        self,
        root: Path,
        config: sps.ScenePreparationConfig,
        render_id: str | None,
    ) -> tuple[Path, str]:
        suffix = render_id or uuid.uuid4().hex
        suffix = suffix.removeprefix(DEFAULT_AUDIO_RENDER_PREFIX)
        try:
            identifier = validate_identifier(suffix, "render_id")
            effective = validate_identifier(
                f"{DEFAULT_AUDIO_RENDER_PREFIX}{identifier}", "render_id"
            )
        except PathValidationError as exc:
            raise _fail("audio.render.unsafe_path", str(exc), stage="render") from exc
        try:
            target = resolve_within_project(
                root, Path(config.render_root) / effective, "audio render directory"
            )
        except PathValidationError as exc:
            raise _fail("audio.render.unsafe_path", str(exc), stage="render") from exc
        if target.exists():
            raise _fail(
                "audio.render.collision",
                f"Render directory already exists: {target}",
                str(target),
                stage="render",
            )
        try:
            target.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise _fail(
                "audio.render.create_failed",
                f"Could not create render directory {target}: {exc}",
                str(target),
                stage="render",
            ) from exc
        return target, effective

    @staticmethod
    def _new_temp_file(directory: Path, stem: str) -> Path:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{stem}-", suffix=".tmp.mp4", dir=str(directory)
        )
        os.close(descriptor)
        return Path(name)

    def _prepare_scene(
        self,
        root: Path,
        render_dir: Path,
        order: int,
        scene: Any,
        config: sps.ScenePreparationConfig,
        hold_durations: dict[str, Decimal],
    ) -> Path:
        location = f"scenes[{order}] ({scene.scene_id})"
        try:
            source = resolve_within_project(
                root, scene.generated_video_path, "generated_video_path"
            )
        except PathValidationError as exc:
            raise _fail(
                "audio.scene.unsafe_path",
                str(exc),
                f"{location}.generated_video_path",
                stage="scene",
            ) from exc
        if not source.is_file() or source.stat().st_size <= 0:
            raise _fail(
                "audio.scene.missing_video",
                f"Scene video is missing or empty: {source}",
                str(source),
                stage="scene",
            )
        hold = scene.final_frame_hold_s
        if hold > 0:
            hold_durations[scene.scene_id] = hold
        command = sps.build_prepare_command(source, render_dir / "unused.mp4", config)
        filter_index = command.index("-vf") + 1
        video_filter = sps.scale_filter(config)
        if hold > 0:
            video_filter += f",tpad=stop_mode=clone:stop_duration={hold:.6f}"
        command[filter_index] = video_filter
        temp_output = self._new_temp_file(render_dir, f"scene_{order:04d}_{scene.scene_id}")
        command[-1] = str(temp_output)
        logger.info(
            "Preparing scene %s with %.6fs final-frame hold", scene.scene_id, float(hold)
        )
        try:
            self.command_runner(command, config.ffmpeg_timeout_s)
        except sps.ScenePreparationError as exc:
            raise _fail(
                "audio.production.video_prepare_failed",
                f"Could not prepare scene video {scene.scene_id}: {exc.message}",
                str(source),
                stage="scene",
            ) from exc
        info = self._probe(temp_output, config)
        prepared_duration = coerce_decimal(
            info.get("duration_s"), f"{location}.prepared_duration", positive=True
        )
        if info.get("video_codec") is None:
            raise _fail(
                "audio.production.video_stream_missing",
                f"Prepared scene has no video stream: {temp_output}",
                str(temp_output),
                stage="scene",
            )
        if abs(prepared_duration - scene.effective_duration_s) > config.duration_tolerance_s:
            raise _fail(
                "audio.production.scene_duration_mismatch",
                f"Prepared duration {decimal_text(prepared_duration)}s does not match "
                f"effective duration {decimal_text(scene.effective_duration_s)}s "
                f"for scene {scene.scene_id}",
                str(temp_output),
                stage="scene",
            )
        prepared_path = render_dir / f"scene_{order:04d}_{scene.scene_id}.mp4"
        if prepared_path.exists():
            raise _fail(
                "audio.production.scene_collision",
                f"Prepared scene video already exists: {prepared_path}",
                str(prepared_path),
                stage="scene",
            )
        temp_output.replace(prepared_path)
        return prepared_path

    def _concatenate(
        self,
        render_dir: Path,
        prepared: Sequence[Path],
        timeline: AudioTimeline,
        config: sps.ScenePreparationConfig,
        warnings: list[ValidationIssue],
    ) -> Path:
        if len(prepared) != timeline.scene_count:
            raise _fail(
                "audio.production.scene_count_mismatch",
                "Prepared scene count differs from the audio timeline",
                str(render_dir),
                stage="concat",
            )
        concat_list = render_dir / "concat_list.txt"
        if concat_list.exists():
            raise _fail(
                "audio.production.concat_collision",
                f"Concat list already exists: {concat_list}",
                str(concat_list),
                stage="concat",
            )
        lines = [f"file '{sps.escape_concat_path(path)}'" for path in prepared]
        concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temp_output = self._new_temp_file(render_dir, "final_silent")
        command = sps.build_concat_command(
            concat_list, temp_output, config, reencode=False
        )
        try:
            self.command_runner(command, config.ffmpeg_timeout_s)
        except sps.ScenePreparationError as exc:
            raise _fail(
                "audio.production.concat_failed",
                f"Silent concatenation failed: {exc.message}",
                str(concat_list),
                stage="concat",
            ) from exc
        final_path = render_dir / "final_silent.mp4"
        if final_path.exists():
            raise _fail(
                "audio.production.concat_collision",
                f"Final silent video already exists: {final_path}",
                str(final_path),
                stage="concat",
            )
        info = self._probe(temp_output, config)
        if info.get("audio_codec") is not None:
            raise _fail(
                "audio.production.unexpected_audio",
                "Concatenated video unexpectedly contains an audio stream",
                str(temp_output),
                stage="concat",
            )
        temp_output.replace(final_path)
        if final_path.stat().st_size <= 0:
            raise _fail(
                "audio.production.empty_video",
                f"Final silent video is empty: {final_path}",
                str(final_path),
                stage="concat",
            )
        warnings.append(
            ValidationIssue(
                "audio.production.scene_order_preserved",
                f"Concatenated {len(prepared)} scenes in render-plan order",
                "scene_order",
            )
        )
        return final_path

    def _validate_video(
        self,
        path: Path,
        duration: Decimal,
        config: sps.ScenePreparationConfig,
    ) -> Any:
        expectations = RenderExpectations(
            width=config.width,
            height=config.height,
            fps=config.fps,
            expected_duration_s=duration,
            duration_tolerance_s=config.duration_tolerance_s,
            fps_tolerance=config.policy.validation.fps_tolerance,
            require_video_stream=True,
            require_positive_duration=True,
        )
        validator = self.validator or RenderValidator(
            probe=lambda target: self._probe(Path(target), config)
        )
        return validator.validate_final(path, expectations, require_audio=False)
