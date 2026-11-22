"""MCP-independent music and SFX registration, validation and arrangement.

Assets are user-provided, project-confined files; this service never downloads
or invents audio. Existing scene ``sfx_paths`` remain authoritative for simple
registrations, while richer placement settings live in compatible ``extra``
fields on the project and scene manifests. Arrangement validates every file and
produces absolute timeline placements before any FFmpeg mixing starts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path

import media_service
from audio_models import (
    AUDIO_ASSET_EXTENSIONS,
    DEFAULT_FFPROBE_TIMEOUT_S,
    AudioArrangement,
    AudioPipelineError,
    MusicSettings,
    ResolvedAudioAsset,
    SfxSettings,
    ValidationIssue,
    coerce_duration,
    decimal_text,
    update_scene_extra,
)
from audio_timeline import AudioTimeline
from project_paths import (
    PathValidationError,
    resolve_within_project,
    write_json_atomic,
)
from project_service import ProjectService
from scene_service import SceneService, SceneServiceError

logger = logging.getLogger(__name__)

MUSIC_EXTRA_KEY = "audio_music"
SFX_EXTRA_KEY = "audio_sfx"
ProbeCallable = Callable[..., Mapping[str, object]]


def _fail(
    code: str,
    message: str,
    location: str = "",
    *,
    stage: str = "arrange",
) -> AudioPipelineError:
    return AudioPipelineError(code, message, location, stage=stage)


def probe_audio(path: str | Path, *, timeout_s: float) -> Mapping[str, object]:
    """Default audio prober backed by the existing media service."""
    return media_service.probe_media(path, timeout_s=timeout_s)


def _valid_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"audio.{label}.invalid_path", f"{label} must be a non-empty path", label)
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise _fail(
            f"audio.{label}.unsafe_path",
            f"{label} must be a project-relative path: {value!r}",
            label,
        )
    return raw.as_posix()


class AudioAssetRegistry:
    """Store project/scene audio settings compatibly in manifest ``extra``."""

    def __init__(
        self,
        projects: ProjectService | None = None,
        scenes: SceneService | None = None,
    ) -> None:
        self.projects = projects if projects is not None else ProjectService()
        self.scenes = scenes if scenes is not None else SceneService(self.projects)

    def set_music(self, project_id: str, settings: MusicSettings) -> dict[str, object]:
        """Register project-level music settings without changing known fields."""
        if not isinstance(settings, MusicSettings):
            raise TypeError("settings must be MusicSettings")
        _valid_relative_path(settings.path, "music.path")
        path = self.projects.project_file(project_id)
        if not path.is_file():
            raise _fail(
                "audio.project.not_found",
                f"Project manifest not found: {path}",
                "project_id",
                stage="project",
            )
        try:
            payload: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise _fail(
                "audio.project.unreadable",
                f"Could not read project manifest {path}: {exc}",
                "project_id",
                stage="project",
            ) from exc
        if not isinstance(payload, dict):
            raise _fail(
                "audio.project.invalid_manifest",
                "Project manifest must be a JSON object",
                "project_id",
                stage="project",
            )
        extra = payload.get("extra")
        if not isinstance(extra, dict):
            extra = {}
        extra[MUSIC_EXTRA_KEY] = settings.to_dict()
        payload["extra"] = extra
        write_json_atomic(path, payload)
        logger.info("Registered project music asset %s", settings.path)
        return settings.to_dict()

    def add_sfx(self, project_id: str, scene_id: str, settings: SfxSettings) -> dict[str, object]:
        """Register one scene SFX via SceneService plus its placement settings."""
        if not isinstance(settings, SfxSettings):
            raise TypeError("settings must be SfxSettings")
        _valid_relative_path(settings.path, "sfx.path")
        effective = SfxSettings.from_dict({**settings.to_dict(), "scene_id": scene_id})
        try:
            self.scenes.register_asset(project_id, scene_id, "sfx", effective.path)
        except (SceneServiceError, ValueError, PathValidationError) as exc:
            raise _fail(
                "audio.sfx.registration_failed",
                f"Could not register SFX with scene {scene_id}: {exc}",
                f"scene:{scene_id}.sfx_paths",
                stage="register",
            ) from exc
        self.scenes.get_scene(project_id, scene_id)
        metadata = effective.to_dict()

        def store_sfx_extra(extra: dict[str, object]) -> None:
            raw_entries = extra.get(SFX_EXTRA_KEY, [])
            if not isinstance(raw_entries, list):
                raw_entries = []
            entries = [
                entry
                for entry in raw_entries
                if isinstance(entry, dict) and entry.get("path") != effective.path
            ]
            entries.append(metadata)
            extra[SFX_EXTRA_KEY] = entries

        try:
            update_scene_extra(self.projects, project_id, scene_id, store_sfx_extra)
        except (OSError, ValueError, TypeError, KeyError, PathValidationError) as exc:
            raise _fail(
                "audio.sfx.metadata_failed",
                f"Could not store SFX settings for scene {scene_id}: {exc}",
                f"scene:{scene_id}.extra.{SFX_EXTRA_KEY}",
                stage="register",
            ) from exc
        logger.info("Registered scene SFX %s for %s", effective.path, scene_id)
        return effective.to_dict()

    def load(
        self, project_id: str
    ) -> tuple[MusicSettings | None, tuple[SfxSettings, ...]]:
        """Load registered music and SFX settings from an existing project."""
        path = self.projects.project_file(project_id)
        if not path.is_file():
            raise _fail(
                "audio.project.not_found",
                f"Project manifest not found: {path}",
                "project_id",
                stage="project",
            )
        try:
            payload: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise _fail(
                "audio.project.unreadable",
                f"Could not read project manifest {path}: {exc}",
                "project_id",
                stage="project",
            ) from exc
        if not isinstance(payload, dict):
            raise _fail(
                "audio.project.invalid_manifest",
                "Project manifest must be a JSON object",
                "project_id",
                stage="project",
            )
        extra = payload.get("extra")
        music: MusicSettings | None = None
        if isinstance(extra, dict) and isinstance(extra.get(MUSIC_EXTRA_KEY), Mapping):
            try:
                music = MusicSettings.from_dict(extra[MUSIC_EXTRA_KEY])
            except (TypeError, ValueError) as exc:
                raise _fail(
                    "audio.music.invalid_settings",
                    f"Invalid project music settings: {exc}",
                    f"extra.{MUSIC_EXTRA_KEY}",
                    stage="config",
                ) from exc

        sfx: list[SfxSettings] = []
        for scene in self.scenes.list_scenes(project_id):
            scene_extra = scene.extra
            configured: list[SfxSettings] = []
            raw_entries = scene_extra.get(SFX_EXTRA_KEY, [])
            if isinstance(raw_entries, list):
                for entry in raw_entries:
                    if not isinstance(entry, Mapping):
                        continue
                    try:
                        configured.append(
                            SfxSettings.from_dict(
                                {**dict(entry), "scene_id": scene.scene_id}
                            )
                        )
                    except (TypeError, ValueError) as exc:
                        raise _fail(
                            "audio.sfx.invalid_settings",
                            f"Invalid SFX settings for {scene.scene_id}: {exc}",
                            f"scene:{scene.scene_id}.extra.{SFX_EXTRA_KEY}",
                            stage="config",
                        ) from exc
            configured_paths = {entry.path for entry in configured}
            for path_value in scene.sfx_paths:
                if path_value in configured_paths:
                    continue
                try:
                    configured.append(
                        SfxSettings(path=path_value, scene_id=scene.scene_id)
                    )
                except (TypeError, ValueError):
                    continue
            sfx.extend(configured)
        return music, tuple(sfx)


class AudioArrangementService:
    """Validate and place music/SFX assets on an :class:`AudioTimeline`."""

    def __init__(
        self,
        projects: ProjectService | None = None,
        registry: AudioAssetRegistry | None = None,
        *,
        probe: ProbeCallable | None = None,
        ffprobe_timeout_s: float = DEFAULT_FFPROBE_TIMEOUT_S,
    ) -> None:
        self.projects = projects if projects is not None else ProjectService()
        self.registry = registry if registry is not None else AudioAssetRegistry(self.projects)
        self.probe: ProbeCallable = probe if probe is not None else probe_audio
        if isinstance(ffprobe_timeout_s, bool) or not isinstance(
            ffprobe_timeout_s, (int, float)
        ):
            raise TypeError("ffprobe_timeout_s must be a number")
        if ffprobe_timeout_s <= 0:
            raise ValueError("ffprobe_timeout_s must be positive")
        self.ffprobe_timeout_s = float(ffprobe_timeout_s)

    def build(
        self,
        project_id: str,
        timeline: AudioTimeline,
        music: MusicSettings | None = None,
        sfx: Sequence[SfxSettings] = (),
        *,
        include_registered: bool = True,
    ) -> AudioArrangement:
        """Resolve all assets and verify their timeline placement."""
        if not isinstance(timeline, AudioTimeline):
            raise TypeError("timeline must be an AudioTimeline")
        root = self.projects.project_dir(project_id)
        registered_music: MusicSettings | None = None
        registered_sfx: tuple[SfxSettings, ...] = ()
        if include_registered:
            registered_music, registered_sfx = self.registry.load(project_id)
        effective_music = music if music is not None else registered_music
        sfx_settings = list(registered_sfx) + list(sfx)

        warnings: list[ValidationIssue] = []
        resolved_music: ResolvedAudioAsset | None = None
        if effective_music is not None:
            resolved_music = self._resolve(
                root,
                effective_music,
                stage_label="music",
                location="audio_music",
            )
            available = timeline.total_duration_s - resolved_music.start_s
            usable = resolved_music.trimmed_duration_s
            if not effective_music.loop and usable < available:
                warnings.append(
                    ValidationIssue(
                        "audio.arrangement.music_short",
                        f"Music covers {decimal_text(usable)}s of "
                        f"{decimal_text(available)}s; silent padding will be added "
                        "at the tail (enable loop to repeat the asset)",
                        "audio_music",
                    )
                )
            if effective_music.ducking is None and timeline.narration_scenes:
                warnings.append(
                    ValidationIssue(
                        "audio.arrangement.ducking_absent",
                        "Narration is present but music ducking is not configured",
                        "audio_music.ducking",
                    )
                )

        resolved_sfx: list[ResolvedAudioAsset] = []
        scene_by_id = {scene.scene_id: scene for scene in timeline.scenes}
        for index, settings in enumerate(sfx_settings):
            if not isinstance(settings, SfxSettings):
                raise TypeError("sfx entries must be SfxSettings")
            scene = scene_by_id.get(settings.scene_id)
            if scene is None:
                raise _fail(
                    "audio.sfx.unknown_scene",
                    f"SFX references unknown scene {settings.scene_id!r}",
                    f"sfx[{index}].scene_id",
                )
            placed = SfxSettings.from_dict(
                {
                    **settings.to_dict(),
                    "start_offset_s": decimal_text(settings.start_offset_s),
                }
            )
            resolved = self._resolve(
                root,
                placed,
                stage_label="sfx",
                location=f"sfx[{index}] ({settings.scene_id})",
            )
            absolute_start = scene.video_start_s + settings.start_offset_s
            end = absolute_start + resolved.trimmed_duration_s
            if end > timeline.total_duration_s and (
                end - timeline.total_duration_s
            ) > Decimal("0.05"):
                raise _fail(
                    "audio.sfx.exceeds_timeline",
                    f"SFX ends at {decimal_text(end)}s, after the timeline ends at "
                    f"{decimal_text(timeline.total_duration_s)}s",
                    f"sfx[{index}].start_offset_s",
                )
            resolved_sfx.append(
                ResolvedAudioAsset(
                    settings=resolved.settings,
                    relative_path=resolved.relative_path,
                    absolute_path=resolved.absolute_path,
                    duration_s=resolved.duration_s,
                    start_s=absolute_start,
                    trimmed_duration_s=resolved.trimmed_duration_s,
                    sample_rate=resolved.sample_rate,
                    channels=resolved.channels,
                    warnings=resolved.warnings,
                )
            )

        logger.info(
            "Arranged audio for project %s: music=%s sfx=%d",
            project_id,
            resolved_music is not None,
            len(resolved_sfx),
        )
        return AudioArrangement(
            project_id=project_id,
            total_duration_s=timeline.total_duration_s,
            music=resolved_music,
            sfx=tuple(resolved_sfx),
            warnings=tuple(warnings),
        )

    def _resolve(
        self,
        root: Path,
        settings: AudioAssetSettingsType,
        *,
        stage_label: str,
        location: str,
    ) -> ResolvedAudioAsset:
        try:
            path = resolve_within_project(root, settings.path, f"{stage_label}.path")
        except PathValidationError as exc:
            raise _fail(
                "audio.asset.unsafe_path",
                str(exc),
                f"{location}.path",
            ) from exc
        if not path.is_file():
            raise _fail(
                "audio.asset.missing_file",
                f"Audio file does not exist: {path}",
                f"{location}.path",
            )
        if path.stat().st_size <= 0:
            raise _fail(
                "audio.asset.empty_file",
                f"Audio file is empty: {path}",
                f"{location}.path",
            )
        if path.suffix.lower() not in AUDIO_ASSET_EXTENSIONS:
            raise _fail(
                "audio.asset.invalid_format",
                f"Unsupported audio extension {path.suffix!r}; supported: "
                f"{', '.join(sorted(AUDIO_ASSET_EXTENSIONS))}",
                f"{location}.path",
            )
        try:
            info = self.probe(str(path), timeout_s=self.ffprobe_timeout_s)
        except Exception as exc:
            raise _fail(
                "audio.asset.unreadable",
                f"Could not probe audio asset {path}: {exc}",
                f"{location}.path",
            ) from exc
        if not isinstance(info, Mapping):
            raise _fail(
                "audio.asset.unreadable",
                f"Probe returned {type(info).__name__} instead of a mapping",
                f"{location}.path",
            )
        status = info.get("status")
        if status is not None and status != "ok":
            detail = info.get("error")
            raise _fail(
                "audio.asset.unreadable",
                f"Audio probe returned status {status!r}: {detail}",
                f"{location}.path",
            )
        if info.get("audio_codec") is None:
            raise _fail(
                "audio.asset.no_audio_stream",
                f"Audio asset has no audio stream: {path}",
                f"{location}.path",
            )
        raw_duration = info.get("duration_s")
        if isinstance(raw_duration, bool) or not isinstance(raw_duration, (int, float)):
            raise _fail(
                "audio.asset.invalid_duration",
                f"Audio asset has no measurable duration: {path}",
                f"{location}.duration_s",
            )
        duration = coerce_duration(str(raw_duration), "audio duration")
        if duration <= 0:
            raise _fail(
                "audio.asset.invalid_duration",
                f"Audio asset duration must be positive: {path}",
                f"{location}.duration_s",
            )
        trimmed = resolve_trimmed_duration(settings, duration, location)
        sample_rate = info.get("sample_rate")
        channels = info.get("channels")
        return ResolvedAudioAsset(
            settings=settings,
            relative_path=settings.path,
            absolute_path=str(path),
            duration_s=duration,
            start_s=settings.start_offset_s,
            trimmed_duration_s=trimmed,
            sample_rate=(
                int(sample_rate)
                if isinstance(sample_rate, int) and not isinstance(sample_rate, bool)
                else None
            ),
            channels=(
                int(channels)
                if isinstance(channels, int) and not isinstance(channels, bool)
                else None
            ),
        )


AudioAssetSettingsType = MusicSettings | SfxSettings


def resolve_trimmed_duration(
    settings: AudioAssetSettingsType,
    duration_s: Decimal,
    location: str,
) -> Decimal:
    """Validate source trims and return the usable duration."""
    trim_start = settings.trim_start_s
    trim_end = settings.trim_end_s
    if trim_start + trim_end >= duration_s:
        raise _fail(
            "audio.asset.invalid_trim",
            f"Trim start {decimal_text(trim_start)}s plus trim end "
            f"{decimal_text(trim_end)}s must be shorter than the source "
            f"({decimal_text(duration_s)}s)",
            f"{location}.trim",
        )
    usable = duration_s - trim_start - trim_end
    if settings.fade_in_s > usable or settings.fade_out_s > usable:
        raise _fail(
            "audio.asset.invalid_fade",
            "Fade durations must not exceed the usable audio duration",
            f"{location}.fade",
        )
    if settings.fade_in_s + settings.fade_out_s > usable:
        raise _fail(
            "audio.asset.invalid_fade",
            "Fade-in and fade-out overlap; reduce one or both durations",
            f"{location}.fade",
        )
    return usable
