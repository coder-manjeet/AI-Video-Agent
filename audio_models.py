"""Immutable data models for the scene audio production pipeline.

These models are shared by narration generation, timeline planning, music/SFX
arrangement, mixing and final coordination. They deliberately import no MCP,
ComfyUI or Chatterbox runtime code: business logic stays in dedicated services.
Durations are :class:`~decimal.Decimal` in memory and decimal strings in JSON,
matching the existing render models.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from project_service import ProjectService

from manifest_validation import ValidationIssue
from render_models import RenderValidationIssue, coerce_decimal
from tts_service import DEFAULT_VOICE

AudioValidationIssue = RenderValidationIssue

DEFAULT_NARRATION_LANGUAGE = "en"
NARRATION_LANGUAGES = frozenset({"en", "hi", "hinglish"})
MISSING_NARRATION_POLICIES = frozenset({"allow", "error"})
DEFAULT_NARRATION_OUTPUT_ROOT = "assets/audio/narration"
DEFAULT_NARRATION_TIMEOUT_S = float(os.getenv("TTS_SUBPROCESS_TIMEOUT_S", "300"))
DEFAULT_MASTER_SAMPLE_RATE = 48_000
DEFAULT_MASTER_CHANNELS = 2
DEFAULT_MASTER_SAMPLE_WIDTH_BYTES = 2
DEFAULT_AUDIO_DURATION_TOLERANCE_S = Decimal("0.05")
DEFAULT_AUDIO_RENDER_ROOT = "audio/renders"
DEFAULT_FFMPEG_BIN = os.getenv("MEDIA_FFMPEG_BIN", "ffmpeg")
DEFAULT_FFPROBE_BIN = os.getenv("MEDIA_FFPROBE_BIN", "ffprobe")
DEFAULT_FFMPEG_TIMEOUT_S = float(os.getenv("MEDIA_FFMPEG_TIMEOUT_S", "180"))
DEFAULT_FFPROBE_TIMEOUT_S = float(os.getenv("MEDIA_FFPROBE_TIMEOUT_S", "30"))
DEFAULT_LIMITER_LIMIT = 0.95
AUDIO_ASSET_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg", ".oga", ".opus"}
)


class AudioPipelineError(RuntimeError):
    """Structured failure with a stable code, stage and location."""

    def __init__(
        self,
        code: str,
        message: str,
        location: str = "",
        *,
        stage: str = "audio",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location
        self.stage = stage
        self.issue = ValidationIssue(code, message, location)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "stage": self.stage,
            "issues": [self.issue.to_dict()],
        }


def decimal_text(value: Decimal) -> str:
    """Serialize a Decimal without scientific notation or precision loss."""
    return format(value, "f")


def update_scene_extra(
    projects: ProjectService,
    project_id: str,
    scene_id: str,
    updater: Callable[[dict[str, object]], None],
) -> None:
    """Mutate one scene's raw unknown keys, exposed as :attr:`Scene.extra`.

    ``Scene.from_dict`` collects unknown top-level manifest keys into
    ``Scene.extra`` and drops a literal ``extra`` key. Audio metadata must
    survive ``SceneService`` round-trips, so this helper edits the raw scene
    manifest atomically while preserving every other manifest key verbatim.
    """
    from project_paths import validate_identifier, write_json_atomic

    validate_identifier(project_id, "project_id")
    validate_identifier(scene_id, "scene_id")
    path = projects.project_dir(project_id) / "script" / "scenes.json"
    if not path.is_file():
        raise FileNotFoundError(f"Scene manifest not found: {path}")
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read scene manifest {path}: {exc}") from exc
    if isinstance(payload, Mapping):
        raw_scenes: object = payload.get("scenes", [])
    elif isinstance(payload, list):
        raw_scenes = payload
    else:
        raise TypeError(f"Scene manifest must be a JSON object or array: {path}")
    if not isinstance(raw_scenes, list):
        raise TypeError(f"scenes must be a JSON array in {path}")
    for entry in raw_scenes:
        if not isinstance(entry, dict) or entry.get("scene_id") != scene_id:
            continue
        updater(entry)
        write_json_atomic(path, payload)
        return
    raise KeyError(f"Scene {scene_id!r} not found in {path}")


def coerce_float(value: object, label: str, *, positive: bool = False) -> float:
    """Return a finite float with explicit bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def coerce_gain(value: object, label: str) -> float:
    """Return a finite, non-negative gain."""
    result = coerce_float(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def coerce_duration(value: object, label: str) -> Decimal:
    """Return a non-negative Decimal duration."""
    return coerce_decimal(value, label, non_negative=True)


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_ducking(value: object, label: str) -> DuckingConfig | None:
    if value is None:
        return None
    if isinstance(value, DuckingConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object or null")
    return DuckingConfig.from_dict(value)


@dataclass(frozen=True)
class NarrationConfig:
    """TTS request settings supported by the installed Chatterbox service."""

    language: str = DEFAULT_NARRATION_LANGUAGE
    voice: str = DEFAULT_VOICE
    expressiveness: float | None = None
    speed: float = 1.0
    reference_wav: str | None = None
    output_root: str = DEFAULT_NARRATION_OUTPUT_ROOT
    timeout_s: float = DEFAULT_NARRATION_TIMEOUT_S

    def __post_init__(self) -> None:
        language = _nonempty(self.language, "language").strip().lower()
        if language not in NARRATION_LANGUAGES:
            raise ValueError(
                f"Unsupported language {language!r}; supported: "
                f"{', '.join(sorted(NARRATION_LANGUAGES))}"
            )
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "voice", _nonempty(self.voice, "voice"))
        if self.expressiveness is not None:
            expressiveness = coerce_gain(self.expressiveness, "expressiveness")
            object.__setattr__(self, "expressiveness", expressiveness)
        speed = coerce_float(self.speed, "speed", positive=True)
        if speed != 1.0:
            # The installed ChatterboxService has no speech-rate parameter.
            # Reject the request instead of silently time-stretching narration.
            raise ValueError(
                "The installed Chatterbox service has no speech-rate parameter; "
                "speed must be 1.0. Narration is never time-stretched."
            )
        object.__setattr__(self, "speed", speed)
        if self.reference_wav is not None:
            object.__setattr__(
                self,
                "reference_wav",
                _nonempty(self.reference_wav, "reference_wav"),
            )
        root = _nonempty(self.output_root, "output_root")
        from pathlib import Path

        raw_root = Path(root)
        if raw_root.is_absolute() or ".." in raw_root.parts:
            raise ValueError("output_root must be relative to the project")
        object.__setattr__(self, "output_root", raw_root.as_posix())
        object.__setattr__(
            self, "timeout_s", coerce_float(self.timeout_s, "timeout_s", positive=True)
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | NarrationConfig | None) -> NarrationConfig:
        if data is None:
            return cls()
        if isinstance(data, NarrationConfig):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("narration configuration must be an object")
        return cls(
            language=data.get("language", DEFAULT_NARRATION_LANGUAGE),
            voice=data.get("voice", DEFAULT_VOICE),
            expressiveness=data.get("expressiveness"),
            speed=data.get("speed", 1.0),
            reference_wav=data.get("reference_wav"),
            output_root=data.get("output_root", DEFAULT_NARRATION_OUTPUT_ROOT),
            timeout_s=data.get("timeout_s", DEFAULT_NARRATION_TIMEOUT_S),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "voice": self.voice,
            "expressiveness": self.expressiveness,
            "speed": self.speed,
            "reference_wav": self.reference_wav,
            "output_root": self.output_root,
            "timeout_s": self.timeout_s,
        }


@dataclass(frozen=True)
class NarrationResult:
    """One generated, verified and registered scene narration WAV."""

    project_id: str
    scene_id: str
    wav_path: str
    version: int
    duration_s: Decimal
    sample_rate: int
    channels: int
    sample_width_bytes: int
    samples: int
    file_size_bytes: int
    language: str
    effective_language: str
    voice: str
    expressiveness: float | None
    reference_used: bool
    text_sha256: str
    generated_at: str
    peak: float = 0.0
    rms: float = 0.0
    model_load_s: float = 0.0
    generation_s: float = 0.0
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.project_id, "project_id")
        _nonempty(self.scene_id, "scene_id")
        _nonempty(self.wav_path, "wav_path")
        object.__setattr__(self, "version", int(self.version))
        if self.version < 1:
            raise ValueError("version must be positive")
        object.__setattr__(
            self, "duration_s", coerce_decimal(self.duration_s, "duration_s", positive=True)
        )
        for name in ("sample_rate", "channels", "sample_width_bytes", "samples"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.sample_width_bytes != 2:
            raise ValueError("Narration WAV must be 16-bit PCM")
        if self.file_size_bytes <= 0:
            raise ValueError("file_size_bytes must be positive")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for issue in self.warnings:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("warnings must contain ValidationIssue entries")

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "scene_id": self.scene_id,
            "wav_path": self.wav_path,
            "version": self.version,
            "duration_s": decimal_text(self.duration_s),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width_bytes,
            "samples": self.samples,
            "file_size_bytes": self.file_size_bytes,
            "language": self.language,
            "effective_language": self.effective_language,
            "voice": self.voice,
            "expressiveness": self.expressiveness,
            "reference_used": self.reference_used,
            "text_sha256": self.text_sha256,
            "generated_at": self.generated_at,
            "peak": self.peak,
            "rms": self.rms,
            "model_load_s": self.model_load_s,
            "generation_s": self.generation_s,
            "warnings": [issue.to_dict() for issue in self.warnings],
        }

    def to_manifest_metadata(self) -> dict[str, object]:
        """Return portable metadata suitable for ``scene.extra`` storage."""
        payload = self.to_dict()
        payload.pop("warnings", None)
        return payload


@dataclass(frozen=True)
class AudioTimelinePolicy:
    """Controls how missing narration is treated during timeline planning."""

    missing_narration_policy: str = "allow"
    duration_tolerance_s: Decimal = DEFAULT_AUDIO_DURATION_TOLERANCE_S

    def __post_init__(self) -> None:
        policy = _nonempty(self.missing_narration_policy, "missing_narration_policy")
        if policy not in MISSING_NARRATION_POLICIES:
            raise ValueError(
                "missing_narration_policy must be one of "
                f"{sorted(MISSING_NARRATION_POLICIES)}"
            )
        object.__setattr__(self, "missing_narration_policy", policy)
        object.__setattr__(
            self,
            "duration_tolerance_s",
            coerce_duration(self.duration_tolerance_s, "duration_tolerance_s"),
        )

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | AudioTimelinePolicy | None
    ) -> AudioTimelinePolicy:
        if data is None:
            return cls()
        if isinstance(data, AudioTimelinePolicy):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("audio timeline policy must be an object")
        return cls(
            missing_narration_policy=data.get("missing_narration_policy", "allow"),
            duration_tolerance_s=data.get(
                "duration_tolerance_s", DEFAULT_AUDIO_DURATION_TOLERANCE_S
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "missing_narration_policy": self.missing_narration_policy,
            "duration_tolerance_s": decimal_text(self.duration_tolerance_s),
        }


@dataclass(frozen=True)
class DuckingConfig:
    """FFmpeg sidechain-compression settings for narration-aware music."""

    level: float = 0.25
    attack_ms: int = 100
    release_ms: int = 400
    threshold: float = 0.02

    def __post_init__(self) -> None:
        level = coerce_float(self.level, "level", positive=True)
        if level > 1:
            raise ValueError("ducking level must be in (0, 1]")
        object.__setattr__(self, "level", level)
        if isinstance(self.attack_ms, bool) or not isinstance(self.attack_ms, int):
            raise TypeError("attack_ms must be an integer")
        if isinstance(self.release_ms, bool) or not isinstance(self.release_ms, int):
            raise TypeError("release_ms must be an integer")
        if self.attack_ms < 1 or self.release_ms < 1:
            raise ValueError("attack_ms and release_ms must be positive")
        if self.attack_ms > 5000 or self.release_ms > 10000:
            raise ValueError("attack_ms and release_ms are unreasonably large")
        threshold = coerce_float(self.threshold, "threshold")
        if not 0 <= threshold <= 1:
            raise ValueError("ducking threshold must be in [0, 1]")
        object.__setattr__(self, "threshold", threshold)

    @property
    def ratio(self) -> float:
        """Compressor ratio corresponding to the requested duck level."""
        return max(1.0, 1.0 / self.level)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | DuckingConfig) -> DuckingConfig:
        if isinstance(data, DuckingConfig):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("ducking configuration must be an object")
        return cls(
            level=data.get("level", 0.25),
            attack_ms=data.get("attack_ms", 100),
            release_ms=data.get("release_ms", 400),
            threshold=data.get("threshold", 0.02),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class AudioAssetSettings:
    """Common placement, gain, trim and fade settings for one audio asset."""

    path: str
    volume: float = 1.0
    start_offset_s: Decimal = Decimal(0)
    trim_start_s: Decimal = Decimal(0)
    trim_end_s: Decimal = Decimal(0)
    fade_in_s: Decimal = Decimal(0)
    fade_out_s: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _nonempty(self.path, "path"))
        object.__setattr__(self, "volume", coerce_gain(self.volume, "volume"))
        for name in (
            "start_offset_s",
            "trim_start_s",
            "trim_end_s",
            "fade_in_s",
            "fade_out_s",
        ):
            object.__setattr__(
                self, name, coerce_duration(getattr(self, name), name)
            )

    @classmethod
    def _common_kwargs(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "path": data.get("path"),
            "volume": data.get("volume", 1.0),
            "start_offset_s": data.get("start_offset_s", Decimal(0)),
            "trim_start_s": data.get("trim_start_s", Decimal(0)),
            "trim_end_s": data.get("trim_end_s", Decimal(0)),
            "fade_in_s": data.get("fade_in_s", Decimal(0)),
            "fade_out_s": data.get("fade_out_s", Decimal(0)),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "volume": self.volume,
            "start_offset_s": decimal_text(self.start_offset_s),
            "trim_start_s": decimal_text(self.trim_start_s),
            "trim_end_s": decimal_text(self.trim_end_s),
            "fade_in_s": decimal_text(self.fade_in_s),
            "fade_out_s": decimal_text(self.fade_out_s),
        }


@dataclass(frozen=True)
class MusicSettings(AudioAssetSettings):
    """Project-level music or ambient bed settings."""

    loop: bool = False
    ducking: DuckingConfig | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.loop, bool):
            raise TypeError("loop must be a boolean")
        object.__setattr__(self, "ducking", _optional_ducking(self.ducking, "ducking"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | MusicSettings) -> MusicSettings:
        if isinstance(data, MusicSettings):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("music settings must be an object")
        return cls(
            **cls._common_kwargs(data),
            loop=data.get("loop", False),
            ducking=_optional_ducking(data.get("ducking"), "ducking"),
        )

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload.update(
            {
                "loop": self.loop,
                "ducking": self.ducking.to_dict() if self.ducking is not None else None,
            }
        )
        return payload


@dataclass(frozen=True)
class SfxSettings(AudioAssetSettings):
    """Scene-specific sound effect settings; offsets are scene-relative."""

    scene_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "scene_id", _nonempty(self.scene_id, "scene_id"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | SfxSettings) -> SfxSettings:
        if isinstance(data, SfxSettings):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("SFX settings must be an object")
        return cls(
            **cls._common_kwargs(data),
            scene_id=data.get("scene_id", ""),
        )

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["scene_id"] = self.scene_id
        return payload


@dataclass(frozen=True)
class ResolvedAudioAsset:
    """A validated, project-confined audio asset placed on the timeline."""

    settings: AudioAssetSettings
    relative_path: str
    absolute_path: str
    duration_s: Decimal
    start_s: Decimal
    trimmed_duration_s: Decimal
    sample_rate: int | None = None
    channels: int | None = None
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.settings, AudioAssetSettings):
            raise TypeError("settings must be an AudioAssetSettings")
        _nonempty(self.relative_path, "relative_path")
        _nonempty(self.absolute_path, "absolute_path")
        object.__setattr__(
            self, "duration_s", coerce_decimal(self.duration_s, "duration_s", positive=True)
        )
        object.__setattr__(self, "start_s", coerce_duration(self.start_s, "start_s"))
        object.__setattr__(
            self,
            "trimmed_duration_s",
            coerce_decimal(self.trimmed_duration_s, "trimmed_duration_s", positive=True),
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for issue in self.warnings:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("warnings must contain ValidationIssue entries")

    @property
    def end_s(self) -> Decimal:
        return self.start_s + self.trimmed_duration_s

    def to_dict(self) -> dict[str, object]:
        return {
            "settings": self.settings.to_dict(),
            "relative_path": self.relative_path,
            "absolute_path": self.absolute_path,
            "duration_s": decimal_text(self.duration_s),
            "start_s": decimal_text(self.start_s),
            "trimmed_duration_s": decimal_text(self.trimmed_duration_s),
            "end_s": decimal_text(self.end_s),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class AudioArrangement:
    """Validated music and SFX arrangement for one timeline."""

    project_id: str
    total_duration_s: Decimal
    music: ResolvedAudioAsset | None = None
    sfx: tuple[ResolvedAudioAsset, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.project_id, "project_id")
        object.__setattr__(
            self,
            "total_duration_s",
            coerce_decimal(self.total_duration_s, "total_duration_s", positive=True),
        )
        if self.music is not None and not isinstance(self.music, ResolvedAudioAsset):
            raise TypeError("music must be a ResolvedAudioAsset or None")
        object.__setattr__(self, "sfx", tuple(self.sfx))
        for asset in self.sfx:
            if not isinstance(asset, ResolvedAudioAsset):
                raise TypeError("sfx must contain ResolvedAudioAsset entries")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for issue in self.warnings:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("warnings must contain ValidationIssue entries")

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "total_duration_s": decimal_text(self.total_duration_s),
            "music": self.music.to_dict() if self.music is not None else None,
            "sfx": [asset.to_dict() for asset in self.sfx],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class AudioCommand:
    """One FFmpeg/FFprobe invocation recorded in a mixing report."""

    command: tuple[str, ...]
    returncode: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", tuple(str(part) for part in self.command))
        if not self.command:
            raise ValueError("command must not be empty")
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise TypeError("returncode must be an integer")

    def to_dict(self) -> dict[str, object]:
        return {"command": list(self.command), "returncode": self.returncode}


@dataclass(frozen=True)
class AudioMixConfig:
    """Runtime and output profile for one audio master render."""

    sample_rate: int = DEFAULT_MASTER_SAMPLE_RATE
    channels: int = DEFAULT_MASTER_CHANNELS
    sample_width_bytes: int = DEFAULT_MASTER_SAMPLE_WIDTH_BYTES
    render_root: str = DEFAULT_AUDIO_RENDER_ROOT
    render_id: str | None = None
    render_dir: str | None = None
    master_filename: str = "audio_master.wav"
    ffmpeg_bin: str = DEFAULT_FFMPEG_BIN
    ffprobe_bin: str = DEFAULT_FFPROBE_BIN
    ffmpeg_timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S
    ffprobe_timeout_s: float = DEFAULT_FFPROBE_TIMEOUT_S
    duration_tolerance_s: Decimal = DEFAULT_AUDIO_DURATION_TOLERANCE_S
    limiter_limit: float = DEFAULT_LIMITER_LIMIT

    def __post_init__(self) -> None:
        from pathlib import Path

        for name in ("sample_rate", "channels", "sample_width_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_width_bytes != 2:
            raise ValueError("only 16-bit PCM master output is supported")
        object.__setattr__(
            self, "render_root", _nonempty(self.render_root, "render_root")
        )
        raw_root = Path(self.render_root)
        if raw_root.is_absolute() or ".." in raw_root.parts:
            raise ValueError("render_root must be relative to the project")
        object.__setattr__(self, "render_root", raw_root.as_posix())
        if self.render_id is not None:
            from project_paths import validate_identifier

            validate_identifier(_nonempty(self.render_id, "render_id"), "render_id")
        if self.render_dir is not None:
            raw_dir = Path(_nonempty(self.render_dir, "render_dir"))
            if raw_dir.is_absolute() or ".." in raw_dir.parts:
                raise ValueError("render_dir must be relative to the project")
            object.__setattr__(self, "render_dir", raw_dir.as_posix())
        object.__setattr__(
            self, "master_filename", _nonempty(self.master_filename, "master_filename")
        )
        if Path(self.master_filename).name != self.master_filename:
            raise ValueError("master_filename must be a plain filename")
        object.__setattr__(self, "ffmpeg_bin", _nonempty(self.ffmpeg_bin, "ffmpeg_bin"))
        object.__setattr__(self, "ffprobe_bin", _nonempty(self.ffprobe_bin, "ffprobe_bin"))
        object.__setattr__(
            self,
            "ffmpeg_timeout_s",
            coerce_float(self.ffmpeg_timeout_s, "ffmpeg_timeout_s", positive=True),
        )
        object.__setattr__(
            self,
            "ffprobe_timeout_s",
            coerce_float(self.ffprobe_timeout_s, "ffprobe_timeout_s", positive=True),
        )
        object.__setattr__(
            self,
            "duration_tolerance_s",
            coerce_duration(self.duration_tolerance_s, "duration_tolerance_s"),
        )
        limit = coerce_float(self.limiter_limit, "limiter_limit", positive=True)
        if not 0 < limit <= 1:
            raise ValueError("limiter_limit must be in (0, 1]")
        object.__setattr__(self, "limiter_limit", limit)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | AudioMixConfig | None
    ) -> AudioMixConfig:
        if data is None:
            return cls()
        if isinstance(data, AudioMixConfig):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("audio mix configuration must be an object")
        return cls(
            sample_rate=data.get("sample_rate", DEFAULT_MASTER_SAMPLE_RATE),
            channels=data.get("channels", DEFAULT_MASTER_CHANNELS),
            sample_width_bytes=data.get(
                "sample_width_bytes", DEFAULT_MASTER_SAMPLE_WIDTH_BYTES
            ),
            render_root=data.get("render_root", DEFAULT_AUDIO_RENDER_ROOT),
            render_id=data.get("render_id"),
            render_dir=data.get("render_dir"),
            master_filename=data.get("master_filename", "audio_master.wav"),
            ffmpeg_bin=data.get("ffmpeg_bin", DEFAULT_FFMPEG_BIN),
            ffprobe_bin=data.get("ffprobe_bin", DEFAULT_FFPROBE_BIN),
            ffmpeg_timeout_s=data.get("ffmpeg_timeout_s", DEFAULT_FFMPEG_TIMEOUT_S),
            ffprobe_timeout_s=data.get("ffprobe_timeout_s", DEFAULT_FFPROBE_TIMEOUT_S),
            duration_tolerance_s=data.get(
                "duration_tolerance_s", DEFAULT_AUDIO_DURATION_TOLERANCE_S
            ),
            limiter_limit=data.get("limiter_limit", DEFAULT_LIMITER_LIMIT),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width_bytes,
            "render_root": self.render_root,
            "render_id": self.render_id,
            "render_dir": self.render_dir,
            "master_filename": self.master_filename,
            "ffmpeg_bin": self.ffmpeg_bin,
            "ffprobe_bin": self.ffprobe_bin,
            "ffmpeg_timeout_s": self.ffmpeg_timeout_s,
            "ffprobe_timeout_s": self.ffprobe_timeout_s,
            "duration_tolerance_s": decimal_text(self.duration_tolerance_s),
            "limiter_limit": self.limiter_limit,
        }


@dataclass(frozen=True)
class AudioMixResult:
    """Verified master WAV and the staged commands that produced it."""

    project_id: str
    render_id: str
    render_dir: str
    master_path: str
    master_duration_s: Decimal
    sample_rate: int
    channels: int
    sample_width_bytes: int
    narration_inputs: int
    music_inputs: int
    sfx_inputs: int
    ducking_applied: bool
    peak: float
    clipped_samples: int
    commands: tuple[AudioCommand, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.project_id, "project_id")
        _nonempty(self.render_id, "render_id")
        _nonempty(self.render_dir, "render_dir")
        _nonempty(self.master_path, "master_path")
        object.__setattr__(
            self,
            "master_duration_s",
            coerce_decimal(self.master_duration_s, "master_duration_s", positive=True),
        )
        for name in ("sample_rate", "channels", "sample_width_bytes"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("narration_inputs", "music_inputs", "sfx_inputs", "clipped_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.ducking_applied, bool):
            raise TypeError("ducking_applied must be a boolean")
        object.__setattr__(self, "commands", tuple(self.commands))
        for command in self.commands:
            if not isinstance(command, AudioCommand):
                raise TypeError("commands must contain AudioCommand entries")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for issue in self.warnings:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("warnings must contain ValidationIssue entries")

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "render_id": self.render_id,
            "render_dir": self.render_dir,
            "master_path": self.master_path,
            "master_duration_s": decimal_text(self.master_duration_s),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width_bytes,
            "narration_inputs": self.narration_inputs,
            "music_inputs": self.music_inputs,
            "sfx_inputs": self.sfx_inputs,
            "ducking_applied": self.ducking_applied,
            "peak": self.peak,
            "clipped_samples": self.clipped_samples,
            "commands": [command.to_dict() for command in self.commands],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class AudioProductionReport:
    """Final Milestone 3D report: video, audio, timing and validation."""

    project_id: str
    render_id: str
    render_dir: str
    silent_video_path: str
    silent_video_duration_s: Decimal
    audio_master_path: str
    audio_duration_s: Decimal
    scene_order: tuple[str, ...]
    hold_durations_s: Mapping[str, Decimal]
    timeline: Any
    mix: AudioMixResult
    validation: Mapping[str, object]
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.project_id, "project_id")
        _nonempty(self.render_id, "render_id")
        _nonempty(self.render_dir, "render_dir")
        _nonempty(self.silent_video_path, "silent_video_path")
        _nonempty(self.audio_master_path, "audio_master_path")
        object.__setattr__(
            self,
            "silent_video_duration_s",
            coerce_decimal(
                self.silent_video_duration_s, "silent_video_duration_s", positive=True
            ),
        )
        object.__setattr__(
            self,
            "audio_duration_s",
            coerce_decimal(self.audio_duration_s, "audio_duration_s", positive=True),
        )
        object.__setattr__(self, "scene_order", tuple(self.scene_order))
        object.__setattr__(
            self,
            "hold_durations_s",
            {str(key): coerce_duration(value, "hold_duration_s") for key, value in self.hold_durations_s.items()},
        )
        if not isinstance(self.mix, AudioMixResult):
            raise TypeError("mix must be an AudioMixResult")
        if not isinstance(self.validation, Mapping):
            raise TypeError("validation must be a mapping")
        object.__setattr__(self, "validation", dict(self.validation))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for issue in self.warnings:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("warnings must contain ValidationIssue entries")

    def to_dict(self) -> dict[str, object]:
        timeline = self.timeline
        return {
            "project_id": self.project_id,
            "render_id": self.render_id,
            "render_dir": self.render_dir,
            "silent_video_path": self.silent_video_path,
            "silent_video_duration_s": decimal_text(self.silent_video_duration_s),
            "audio_master_path": self.audio_master_path,
            "audio_duration_s": decimal_text(self.audio_duration_s),
            "scene_order": list(self.scene_order),
            "hold_durations_s": {
                key: decimal_text(value) for key, value in self.hold_durations_s.items()
            },
            "timeline": timeline.to_dict() if hasattr(timeline, "to_dict") else timeline,
            "mix": self.mix.to_dict(),
            "validation": dict(self.validation),
            "warnings": [issue.to_dict() for issue in self.warnings],
        }
