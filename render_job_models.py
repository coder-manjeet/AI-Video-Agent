"""Immutable models for persistent, resumable render jobs.

The models deliberately import no MCP, ComfyUI, TTS or FFmpeg runtime code.
Durations and tolerances are :class:`~decimal.Decimal` in memory and decimal
strings in JSON; every other value is JSON-serializable without absolute
machine paths.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from project_paths import is_iso8601_timestamp, validate_identifier
from render_models import coerce_decimal

logger = logging.getLogger(__name__)

RENDER_STATUSES = frozenset(
    {
        "queued",
        "validating",
        "planning",
        "preparing_video",
        "preparing_audio",
        "generating_subtitles",
        "mixing_audio",
        "merging",
        "validating_output",
        "completed",
        "failed",
        "cancelled",
    }
)
TERMINAL_RENDER_STATUSES = frozenset({"completed", "failed", "cancelled"})
ACTIVE_RENDER_STATUSES = frozenset(RENDER_STATUSES - TERMINAL_RENDER_STATUSES)
RENDER_PRESETS = frozenset({"720p", "1080p", "custom"})

PRESET_DIMENSIONS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}
DEFAULT_PRESET = "1080p"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 24.0
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_SUBTITLE_FORMATS: tuple[str, ...] = ("srt", "vtt")

ALLOWED_VIDEO_CODECS = frozenset({"libx264", "h264", "libx265", "hevc"})
ALLOWED_PIXEL_FORMATS = frozenset(
    {"yuv420p", "yuv422p", "yuv444p", "yuv420p10le"}
)
ALLOWED_AUDIO_CODECS = frozenset(
    {"aac", "libfdk_aac", "libmp3lame", "mp3", "libopus", "opus", "flac", "alac"}
)
ALLOWED_CONTAINERS = frozenset({"mp4"})
ALLOWED_SUBTITLE_FORMATS = frozenset({"srt", "vtt", "ass", "ssa"})
ALLOWED_SUBTITLE_GRANULARITIES = frozenset({"scene", "fixed", "sentence", "word"})
ALLOWED_X264_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }
)
MISSING_NARRATION_POLICIES = frozenset({"allow", "error"})


class RenderJobError(RuntimeError):
    """Structured failure with a stable code, message and location."""

    def __init__(self, code: str, message: str, location: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
        }


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string (``Z`` suffix)."""
    return (
        datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def preset_values(name: str) -> tuple[int, int]:
    """Return the ``(width, height)`` pair for a named resolution preset."""
    if not isinstance(name, str):
        raise TypeError("preset must be a string")
    key = name.strip().lower()
    if key not in PRESET_DIMENSIONS:
        raise ValueError(
            f"Unsupported preset {name!r}; supported: {', '.join(sorted(RENDER_PRESETS))}"
        )
    return PRESET_DIMENSIONS[key]


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _coerce_int(
    value: object,
    label: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
    even: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    if non_negative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    if even and value % 2 != 0:
        raise ValueError(f"{label} must be even")
    return value


def _coerce_float(
    value: object,
    label: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    if non_negative and result < 0:
        raise ValueError(f"{label} must be non-negative")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must not exceed {maximum}")
    return result


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _validate_timestamp(value: object, label: str) -> str:
    """Validate an optional ISO-8601 timestamp, requiring UTC when present."""
    if value is None or value == "":
        return ""
    if not is_iso8601_timestamp(value):
        raise ValueError(f"{label} must be an ISO-8601 timestamp with a timezone")
    text = str(value).strip()
    candidate = f"{text[:-1]}+00:00" if text.endswith(("Z", "z")) else text
    parsed = datetime.fromisoformat(candidate)
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must use UTC")
    return str(value)


def _string_set(value: object, allowed: frozenset[str], label: str) -> str:
    text = _nonempty(value, label).strip().lower()
    if text not in allowed:
        raise ValueError(f"Unsupported {label} {text!r}; supported: {sorted(allowed)}")
    return text


def _require_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"render job {key} must be a string")
    return value


def _mapping_copy(value: object, label: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _path_mapping_copy(value: object, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError(f"{label} must map strings to strings")
        result[key] = item
    return result


def _event_list_copy(value: object, label: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be an array")
    result: list[dict[str, object]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise TypeError(f"{label} must contain objects")
        result.append({str(key): item for key, item in entry.items()})
    return result


@dataclass(frozen=True)
class RenderConfig:
    """Frozen render profile shared by orchestration and validation."""

    preset: str = DEFAULT_PRESET
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: float = DEFAULT_FPS
    video_codec: str = DEFAULT_VIDEO_CODEC
    pixel_format: str = DEFAULT_PIXEL_FORMAT
    audio_codec: str = DEFAULT_AUDIO_CODEC
    audio_bitrate_kbps: int = 192
    audio_sample_rate: int = 48_000
    audio_channels: int = 2
    container: str = "mp4"
    preset_x264: str = "medium"
    crf: int = 18
    movflags: str = "+faststart"
    render_root: str = "video/renders"
    burn_subtitles: bool = False
    subtitle_language: str = "en"
    subtitle_formats: tuple[str, ...] = DEFAULT_SUBTITLE_FORMATS
    subtitle_granularity: str = "scene"
    duration_tolerance_s: Decimal = Decimal("0.5")
    sync_tolerance_s: Decimal = Decimal("0.5")
    fps_tolerance: float = 0.05
    peak_limit: float = 0.99
    min_silence_s: float = 1.0
    black_min_duration_s: float = 2.0
    missing_narration_policy: str = "allow"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    ffmpeg_timeout_s: float = 180.0
    ffprobe_timeout_s: float = 30.0
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        preset = _string_set(self.preset, RENDER_PRESETS, "preset")
        object.__setattr__(self, "preset", preset)

        width = _coerce_int(self.width, "width", positive=True, even=True)
        height = _coerce_int(self.height, "height", positive=True, even=True)
        if preset in PRESET_DIMENSIONS:
            preset_width, preset_height = PRESET_DIMENSIONS[preset]
            if width == DEFAULT_WIDTH:
                width = preset_width
            if height == DEFAULT_HEIGHT:
                height = preset_height
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

        object.__setattr__(self, "fps", _coerce_float(self.fps, "fps", positive=True))
        object.__setattr__(
            self,
            "video_codec",
            _string_set(self.video_codec, ALLOWED_VIDEO_CODECS, "video_codec"),
        )
        object.__setattr__(
            self,
            "pixel_format",
            _string_set(self.pixel_format, ALLOWED_PIXEL_FORMATS, "pixel_format"),
        )
        object.__setattr__(
            self,
            "audio_codec",
            _string_set(self.audio_codec, ALLOWED_AUDIO_CODECS, "audio_codec"),
        )
        for label in ("audio_bitrate_kbps", "audio_sample_rate", "audio_channels"):
            object.__setattr__(
                self,
                label,
                _coerce_int(getattr(self, label), label, positive=True),
            )
        object.__setattr__(
            self, "container", _string_set(self.container, ALLOWED_CONTAINERS, "container")
        )
        object.__setattr__(
            self,
            "preset_x264",
            _string_set(self.preset_x264, ALLOWED_X264_PRESETS, "preset_x264"),
        )
        crf = _coerce_int(self.crf, "crf", non_negative=True)
        if crf > 51:
            raise ValueError("crf must be between 0 and 51")
        object.__setattr__(self, "crf", crf)
        object.__setattr__(self, "movflags", _nonempty(self.movflags, "movflags"))

        root = Path(_nonempty(self.render_root, "render_root"))
        if root.is_absolute() or ".." in root.parts:
            raise ValueError("render_root must be relative to the project")
        object.__setattr__(self, "render_root", root.as_posix())

        if not isinstance(self.burn_subtitles, bool):
            raise TypeError("burn_subtitles must be a boolean")
        object.__setattr__(self, "subtitle_language", _nonempty(self.subtitle_language, "subtitle_language"))
        formats = tuple(self.subtitle_formats)
        for item in formats:
            if not isinstance(item, str) or item.strip().lower() not in ALLOWED_SUBTITLE_FORMATS:
                raise ValueError(f"Unsupported subtitle format: {item!r}")
        object.__setattr__(
            self, "subtitle_formats", tuple(item.strip().lower() for item in formats)
        )
        object.__setattr__(
            self,
            "subtitle_granularity",
            _string_set(
                self.subtitle_granularity,
                ALLOWED_SUBTITLE_GRANULARITIES,
                "subtitle_granularity",
            ),
        )
        object.__setattr__(
            self,
            "duration_tolerance_s",
            coerce_decimal(
                self.duration_tolerance_s, "duration_tolerance_s", non_negative=True
            ),
        )
        object.__setattr__(
            self,
            "sync_tolerance_s",
            coerce_decimal(self.sync_tolerance_s, "sync_tolerance_s", non_negative=True),
        )
        object.__setattr__(
            self,
            "fps_tolerance",
            _coerce_float(self.fps_tolerance, "fps_tolerance", non_negative=True),
        )
        peak_limit = _coerce_float(self.peak_limit, "peak_limit", positive=True)
        if peak_limit > 1:
            raise ValueError("peak_limit must be in (0, 1]")
        object.__setattr__(self, "peak_limit", peak_limit)
        for label in ("min_silence_s", "black_min_duration_s"):
            object.__setattr__(
                self, label, _coerce_float(getattr(self, label), label, non_negative=True)
            )
        object.__setattr__(
            self,
            "missing_narration_policy",
            _string_set(
                self.missing_narration_policy,
                MISSING_NARRATION_POLICIES,
                "missing_narration_policy",
            ),
        )
        object.__setattr__(self, "ffmpeg_bin", _nonempty(self.ffmpeg_bin, "ffmpeg_bin"))
        object.__setattr__(self, "ffprobe_bin", _nonempty(self.ffprobe_bin, "ffprobe_bin"))
        for label in ("ffmpeg_timeout_s", "ffprobe_timeout_s"):
            object.__setattr__(
                self, label, _coerce_float(getattr(self, label), label, positive=True)
            )
        extra = _mapping_copy(self.extra, "extra")
        object.__setattr__(self, "extra", extra)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | RenderConfig | None = None
    ) -> RenderConfig:
        if isinstance(data, RenderConfig):
            return data
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError("render configuration must be an object")

        raw = dict(data)
        known = {item.name for item in fields(cls)}
        extra: dict[str, object] = {}
        nested_extra = raw.pop("extra", None)
        if isinstance(nested_extra, Mapping):
            extra.update({str(key): value for key, value in nested_extra.items()})
        extra.update({key: value for key, value in raw.items() if key not in known})

        preset = _string_set(raw.get("preset", DEFAULT_PRESET), RENDER_PRESETS, "preset")
        has_width = "width" in raw
        has_height = "height" in raw
        if preset == "custom" and not (has_width and has_height):
            raise ValueError("custom preset requires both width and height")
        if preset in PRESET_DIMENSIONS:
            preset_width, preset_height = PRESET_DIMENSIONS[preset]
            width = raw["width"] if has_width else preset_width
            height = raw["height"] if has_height else preset_height
        else:
            width = raw.get("width", DEFAULT_WIDTH)
            height = raw.get("height", DEFAULT_HEIGHT)

        return cls(
            preset=preset,
            width=width,
            height=height,
            fps=raw.get("fps", DEFAULT_FPS),
            video_codec=raw.get("video_codec", DEFAULT_VIDEO_CODEC),
            pixel_format=raw.get("pixel_format", DEFAULT_PIXEL_FORMAT),
            audio_codec=raw.get("audio_codec", DEFAULT_AUDIO_CODEC),
            audio_bitrate_kbps=raw.get("audio_bitrate_kbps", 192),
            audio_sample_rate=raw.get("audio_sample_rate", 48_000),
            audio_channels=raw.get("audio_channels", 2),
            container=raw.get("container", "mp4"),
            preset_x264=raw.get("preset_x264", "medium"),
            crf=raw.get("crf", 18),
            movflags=raw.get("movflags", "+faststart"),
            render_root=raw.get("render_root", "video/renders"),
            burn_subtitles=raw.get("burn_subtitles", False),
            subtitle_language=raw.get("subtitle_language", "en"),
            subtitle_formats=raw.get("subtitle_formats", DEFAULT_SUBTITLE_FORMATS),
            subtitle_granularity=raw.get("subtitle_granularity", "scene"),
            duration_tolerance_s=raw.get("duration_tolerance_s", Decimal("0.5")),
            sync_tolerance_s=raw.get("sync_tolerance_s", Decimal("0.5")),
            fps_tolerance=raw.get("fps_tolerance", 0.05),
            peak_limit=raw.get("peak_limit", 0.99),
            min_silence_s=raw.get("min_silence_s", 1.0),
            black_min_duration_s=raw.get("black_min_duration_s", 2.0),
            missing_narration_policy=raw.get("missing_narration_policy", "allow"),
            ffmpeg_bin=raw.get("ffmpeg_bin", "ffmpeg"),
            ffprobe_bin=raw.get("ffprobe_bin", "ffprobe"),
            ffmpeg_timeout_s=raw.get("ffmpeg_timeout_s", 180.0),
            ffprobe_timeout_s=raw.get("ffprobe_timeout_s", 30.0),
            extra=extra,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "preset": self.preset,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "video_codec": self.video_codec,
            "pixel_format": self.pixel_format,
            "audio_codec": self.audio_codec,
            "audio_bitrate_kbps": self.audio_bitrate_kbps,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "container": self.container,
            "preset_x264": self.preset_x264,
            "crf": self.crf,
            "movflags": self.movflags,
            "render_root": self.render_root,
            "burn_subtitles": self.burn_subtitles,
            "subtitle_language": self.subtitle_language,
            "subtitle_formats": list(self.subtitle_formats),
            "subtitle_granularity": self.subtitle_granularity,
            "duration_tolerance_s": _decimal_text(self.duration_tolerance_s),
            "sync_tolerance_s": _decimal_text(self.sync_tolerance_s),
            "fps_tolerance": self.fps_tolerance,
            "peak_limit": self.peak_limit,
            "min_silence_s": self.min_silence_s,
            "black_min_duration_s": self.black_min_duration_s,
            "missing_narration_policy": self.missing_narration_policy,
            "ffmpeg_bin": self.ffmpeg_bin,
            "ffprobe_bin": self.ffprobe_bin,
            "ffmpeg_timeout_s": self.ffmpeg_timeout_s,
            "ffprobe_timeout_s": self.ffprobe_timeout_s,
        }
        payload.update({key: value for key, value in self.extra.items() if key not in payload})
        return payload


@dataclass(frozen=True)
class RenderProgress:
    """Snapshot of one render stage; ``percent=None`` is indeterminate."""

    stage: str
    percent: float | None = None
    label: str = ""
    completed: int = 0
    total: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _nonempty(self.stage, "stage"))
        if self.percent is not None:
            percent = _coerce_float(self.percent, "percent")
            if not 0 <= percent <= 100:
                raise ValueError("percent must be between 0 and 100")
            object.__setattr__(self, "percent", percent)
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        for name in ("completed", "total"):
            object.__setattr__(
                self,
                name,
                _coerce_int(getattr(self, name), name, non_negative=True),
            )

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | RenderProgress | None = None
    ) -> RenderProgress:
        if isinstance(data, RenderProgress):
            return data
        if data is None:
            return cls(stage="queued")
        if not isinstance(data, Mapping):
            raise TypeError("render progress must be an object")
        return cls(
            stage=str(data.get("stage", "queued")),
            percent=data.get("percent"),
            label=str(data.get("label", "")),
            completed=data.get("completed", 0),
            total=data.get("total", 0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "percent": self.percent,
            "label": self.label,
            "completed": self.completed,
            "total": self.total,
        }


@dataclass(frozen=True)
class RenderJob:
    """Persistent state of one queued or running final render."""

    job_id: str
    project_id: str
    render_id: str
    status: str = "queued"
    stage: str = "queued"
    progress: RenderProgress = field(
        default_factory=lambda: RenderProgress(stage="queued")
    )
    config: dict[str, object] = field(default_factory=dict)
    created_at: str = ""
    started_at: str | None = None
    updated_at: str = ""
    completed_at: str | None = None
    error: dict[str, object] | None = None
    output_paths: dict[str, str] = field(default_factory=dict)
    validation_summary: dict[str, object] = field(default_factory=dict)
    stages: list[dict[str, object]] = field(default_factory=list)
    cancel_requested: bool = False
    attempt: int = 0
    warnings: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_identifier(self.job_id, "job_id")
        validate_identifier(self.project_id, "project_id")
        validate_identifier(self.render_id, "render_id")
        if self.status not in RENDER_STATUSES:
            raise ValueError(
                f"Unsupported status {self.status!r}; supported: {sorted(RENDER_STATUSES)}"
            )
        if self.stage not in RENDER_STATUSES:
            raise ValueError(
                f"Unsupported stage {self.stage!r}; supported: {sorted(RENDER_STATUSES)}"
            )
        if not isinstance(self.progress, RenderProgress):
            raise TypeError("progress must be a RenderProgress")
        _validate_timestamp(self.created_at, "created_at")
        _validate_timestamp(self.updated_at, "updated_at")
        for name in ("started_at", "completed_at"):
            if getattr(self, name) is not None:
                _validate_timestamp(getattr(self, name), name)
        object.__setattr__(self, "config", _mapping_copy(self.config, "config"))
        if self.error is not None:
            object.__setattr__(self, "error", _mapping_copy(self.error, "error"))
        object.__setattr__(
            self, "output_paths", _path_mapping_copy(self.output_paths, "output_paths")
        )
        object.__setattr__(
            self,
            "validation_summary",
            _mapping_copy(self.validation_summary, "validation_summary"),
        )
        object.__setattr__(self, "stages", _event_list_copy(self.stages, "stages"))
        object.__setattr__(self, "warnings", _event_list_copy(self.warnings, "warnings"))
        if not isinstance(self.cancel_requested, bool):
            raise TypeError("cancel_requested must be a boolean")
        object.__setattr__(
            self, "attempt", _coerce_int(self.attempt, "attempt", non_negative=True)
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_RENDER_STATUSES

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_RENDER_STATUSES

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | RenderJob | None = None) -> RenderJob:
        if isinstance(data, RenderJob):
            return data
        if data is None:
            raise TypeError("render job data must be an object")
        if not isinstance(data, Mapping):
            raise TypeError("render job data must be an object")
        raw = dict(data)
        progress_data = raw.get("progress")
        if progress_data is None:
            progress = RenderProgress(stage=str(raw.get("stage", "queued")))
        elif isinstance(progress_data, (RenderProgress, Mapping)):
            progress = RenderProgress.from_dict(progress_data)
        else:
            raise TypeError("render progress must be an object")
        return cls(
            job_id=_require_string(raw, "job_id"),
            project_id=_require_string(raw, "project_id"),
            render_id=_require_string(raw, "render_id"),
            status=raw.get("status", "queued"),
            stage=raw.get("stage", "queued"),
            progress=progress,
            config=raw.get("config", {}),
            created_at=raw.get("created_at", ""),
            started_at=raw.get("started_at"),
            updated_at=raw.get("updated_at", ""),
            completed_at=raw.get("completed_at"),
            error=raw.get("error"),
            output_paths=raw.get("output_paths", {}),
            validation_summary=raw.get("validation_summary", {}),
            stages=raw.get("stages", []),
            cancel_requested=raw.get("cancel_requested", False),
            attempt=raw.get("attempt", 0),
            warnings=raw.get("warnings", []),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "project_id": self.project_id,
            "render_id": self.render_id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress.to_dict(),
            "config": dict(self.config),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": dict(self.error) if self.error is not None else None,
            "output_paths": dict(self.output_paths),
            "validation_summary": dict(self.validation_summary),
            "stages": [dict(event) for event in self.stages],
            "cancel_requested": self.cancel_requested,
            "attempt": self.attempt,
            "warnings": [dict(entry) for entry in self.warnings],
        }

    def evolve(self, **changes: object) -> RenderJob:
        """Return a new job with *changes* applied and a fresh ``updated_at``."""
        allowed = {item.name for item in fields(type(self))}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown render job fields: {sorted(unknown)}")
        immutable = {"job_id", "project_id", "render_id"} & set(changes)
        if immutable:
            raise ValueError(f"Render job identity fields cannot change: {sorted(immutable)}")
        updates = dict(changes)
        progress = updates.pop("progress", None)
        if isinstance(progress, RenderProgress):
            updates["progress"] = progress.to_dict()
        elif isinstance(progress, Mapping):
            updates["progress"] = RenderProgress.from_dict(progress).to_dict()
        elif progress is not None:
            raise TypeError("progress must be a RenderProgress or object")
        payload = self.to_dict()
        payload.update(updates)
        if "updated_at" not in updates:
            payload["updated_at"] = utc_now()
        return RenderJob.from_dict(payload)

    def with_status(
        self,
        status: str,
        *,
        stage: str | None = None,
        progress: RenderProgress | Mapping[str, Any] | None = None,
        updated_at: str | None = None,
        **changes: object,
    ) -> RenderJob:
        """Return a new job at *status* with optional stage/progress updates."""
        updates: dict[str, object] = {"status": status}
        if stage is not None:
            updates["stage"] = stage
        if progress is not None:
            updates["progress"] = progress
        if updated_at is not None:
            updates["updated_at"] = updated_at
        updates.update(changes)
        return self.evolve(**updates)
