"""Immutable models for render planning and validation.

Timeline and duration values are :class:`~decimal.Decimal` in memory. JSON
serialization emits them as decimal strings so no precision is lost; callers can
restore them with ``Decimal(value)``. Validation issues reuse the structured
:class:`~manifest_validation.ValidationIssue` type. This module imports no MCP,
ComfyUI, TTS or FFmpeg code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from manifest_validation import ValidationIssue

# Render plans reuse the project's structured validation issue type.
RenderValidationIssue = ValidationIssue

DEFAULT_DURATION_MODE = "actual_measured_extend_narration"
DURATION_MODES = frozenset({DEFAULT_DURATION_MODE})
DEFAULT_DURATION_TOLERANCE_S = Decimal("0.5")
DEFAULT_FPS_TOLERANCE = 0.05

_POSITIVE_DURATION_FIELDS = (
    "planned_duration_s",
    "measured_duration_s",
    "effective_duration_s",
)
_NON_NEGATIVE_DURATION_FIELDS = (
    "video_hold_duration_s",
    "silent_tail_duration_s",
    "timeline_start_s",
    "timeline_end_s",
    "planned_timeline_start_s",
    "planned_timeline_end_s",
)
_SIGNED_DURATION_FIELDS = ("duration_delta_s", "narration_delta_s")


def coerce_decimal(
    value: object,
    label: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> Decimal:
    """Return *value* as a finite :class:`Decimal` with explicit bounds."""
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, float, str)):
        raise TypeError(f"{label} must be a decimal number")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be a decimal number: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    if non_negative and result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _decimal_or_none(value: object, label: str, *, positive: bool = False) -> Decimal | None:
    if value is None:
        return None
    return coerce_decimal(value, label, positive=positive)


def _probe_decimal(value: object, *, positive: bool = False) -> Decimal | None:
    try:
        return _decimal_or_none(value, "probe duration", positive=positive)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _probe_int(value: object, *, positive: bool = False) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if not positive or value > 0 else None


def _coerce_float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _probe_float(value: object, *, positive: bool = False) -> float | None:
    try:
        return _coerce_float(value, "probe number", positive=positive)
    except (TypeError, ValueError):
        return None


def _probe_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _optional_decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else _decimal_text(value)


@dataclass(frozen=True)
class RenderValidationSettings:
    """Tunables shared by both render validation profiles."""

    fps_tolerance: float = DEFAULT_FPS_TOLERANCE
    require_video_stream: bool = True
    require_positive_duration: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fps_tolerance", _coerce_float(self.fps_tolerance, "fps_tolerance")
        )
        if self.fps_tolerance < 0:
            raise ValueError("fps_tolerance must be non-negative")
        for name in ("require_video_stream", "require_positive_duration"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "fps_tolerance": self.fps_tolerance,
            "require_video_stream": self.require_video_stream,
            "require_positive_duration": self.require_positive_duration,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | RenderValidationSettings) -> RenderValidationSettings:
        if isinstance(data, RenderValidationSettings):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("validation settings must be an object")
        return cls(
            fps_tolerance=data.get("fps_tolerance", DEFAULT_FPS_TOLERANCE),
            require_video_stream=data.get("require_video_stream", True),
            require_positive_duration=data.get("require_positive_duration", True),
        )


@dataclass(frozen=True)
class RenderPolicy:
    """How a render plan measures time and validates its outputs."""

    duration_mode: str = DEFAULT_DURATION_MODE
    require_audio: bool = True
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_tolerance_s: Decimal = DEFAULT_DURATION_TOLERANCE_S
    validation: RenderValidationSettings = field(default_factory=RenderValidationSettings)

    def __post_init__(self) -> None:
        if self.duration_mode not in DURATION_MODES:
            raise ValueError(
                f"Unsupported duration_mode {self.duration_mode!r}; "
                f"supported modes: {', '.join(sorted(DURATION_MODES))}"
            )
        if not isinstance(self.require_audio, bool):
            raise TypeError("require_audio must be a boolean")
        object.__setattr__(self, "width", self._optional_int(self.width, "width"))
        object.__setattr__(self, "height", self._optional_int(self.height, "height"))
        object.__setattr__(self, "fps", self._optional_float(self.fps, "fps"))
        object.__setattr__(
            self,
            "duration_tolerance_s",
            coerce_decimal(self.duration_tolerance_s, "duration_tolerance_s", non_negative=True),
        )
        object.__setattr__(
            self, "validation", RenderValidationSettings.from_dict(self.validation)
        )

    @staticmethod
    def _optional_int(value: object, label: str) -> int | None:
        if value is None:
            return None
        return _coerce_int(value, label, positive=True)

    @staticmethod
    def _optional_float(value: object, label: str) -> float | None:
        if value is None:
            return None
        return _coerce_float(value, label, positive=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "duration_mode": self.duration_mode,
            "require_audio": self.require_audio,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_tolerance_s": _decimal_text(self.duration_tolerance_s),
            "validation": self.validation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | RenderPolicy | None = None) -> RenderPolicy:
        if data is None:
            return cls()
        if isinstance(data, RenderPolicy):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("render policy must be an object")
        return cls(
            duration_mode=data.get("duration_mode", DEFAULT_DURATION_MODE),
            require_audio=data.get("require_audio", True),
            width=data.get("width"),
            height=data.get("height"),
            fps=data.get("fps"),
            duration_tolerance_s=data.get(
                "duration_tolerance_s", DEFAULT_DURATION_TOLERANCE_S
            ),
            validation=RenderValidationSettings.from_dict(data.get("validation", {})),
        )


@dataclass(frozen=True)
class RenderExpectations:
    """Optional checks applied to a scene video or final video file."""

    width: int | None = None
    height: int | None = None
    fps: float | None = None
    expected_duration_s: Decimal | None = None
    duration_tolerance_s: Decimal = DEFAULT_DURATION_TOLERANCE_S
    fps_tolerance: float = DEFAULT_FPS_TOLERANCE
    require_video_stream: bool = True
    require_positive_duration: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", RenderPolicy._optional_int(self.width, "width"))
        object.__setattr__(self, "height", RenderPolicy._optional_int(self.height, "height"))
        object.__setattr__(self, "fps", RenderPolicy._optional_float(self.fps, "fps"))
        object.__setattr__(
            self,
            "expected_duration_s",
            _decimal_or_none(self.expected_duration_s, "expected_duration_s", positive=True),
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
            "fps_tolerance",
            _coerce_float(self.fps_tolerance, "fps_tolerance"),
        )
        if self.fps_tolerance < 0:
            raise ValueError("fps_tolerance must be non-negative")
        for name in ("require_video_stream", "require_positive_duration"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

    @classmethod
    def from_policy(
        cls,
        policy: RenderPolicy,
        *,
        duration_s: Decimal | None = None,
    ) -> RenderExpectations:
        """Derive expectations from a policy, optionally adding a duration."""
        return cls(
            width=policy.width,
            height=policy.height,
            fps=policy.fps,
            expected_duration_s=duration_s,
            duration_tolerance_s=policy.duration_tolerance_s,
            fps_tolerance=policy.validation.fps_tolerance,
            require_video_stream=policy.validation.require_video_stream,
            require_positive_duration=policy.validation.require_positive_duration,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "expected_duration_s": _optional_decimal_text(self.expected_duration_s),
            "duration_tolerance_s": _decimal_text(self.duration_tolerance_s),
            "fps_tolerance": self.fps_tolerance,
            "require_video_stream": self.require_video_stream,
            "require_positive_duration": self.require_positive_duration,
        }


@dataclass(frozen=True)
class RenderMediaInfo:
    """Typed view of one ``probe_media`` result for render planning/validation."""

    path: str
    duration_s: Decimal | None = None
    size_bytes: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    format_name: str | None = None

    @classmethod
    def from_probe(cls, path: str | Path, info: Mapping[str, object]) -> RenderMediaInfo:
        """Convert a probe dictionary into typed fields; invalid values become None."""
        return cls(
            path=str(path),
            duration_s=_probe_decimal(info.get("duration_s"), positive=True),
            size_bytes=_probe_int(info.get("size_bytes"), positive=True),
            width=_probe_int(info.get("width"), positive=True),
            height=_probe_int(info.get("height"), positive=True),
            fps=_probe_float(info.get("fps"), positive=True),
            video_codec=_probe_string(info.get("video_codec")),
            audio_codec=_probe_string(info.get("audio_codec")),
            sample_rate=_probe_int(info.get("sample_rate"), positive=True),
            channels=_probe_int(info.get("channels"), positive=True),
            format_name=_probe_string(info.get("format_name")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "duration_s": _optional_decimal_text(self.duration_s),
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format_name": self.format_name,
        }


@dataclass(frozen=True)
class RenderScene:
    """One scene's measured timing, planned timing and media metadata."""

    scene_id: str
    scene_number: int
    title: str
    status: str
    planned_duration_s: Decimal
    measured_duration_s: Decimal
    effective_duration_s: Decimal
    video_hold_duration_s: Decimal
    silent_tail_duration_s: Decimal
    timeline_start_s: Decimal
    timeline_end_s: Decimal
    planned_timeline_start_s: Decimal
    planned_timeline_end_s: Decimal
    duration_delta_s: Decimal
    narration_duration_s: Decimal | None
    narration_delta_s: Decimal | None
    narration_audio_path: str | None
    narration_text: str
    generated_video_path: str
    video: RenderMediaInfo
    narration: RenderMediaInfo | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        object.__setattr__(self, "scene_number", _coerce_int(self.scene_number, "scene_number", positive=True))
        for name in _POSITIVE_DURATION_FIELDS:
            object.__setattr__(self, name, coerce_decimal(getattr(self, name), name, positive=True))
        for name in _NON_NEGATIVE_DURATION_FIELDS:
            object.__setattr__(
                self, name, coerce_decimal(getattr(self, name), name, non_negative=True)
            )
        for name in _SIGNED_DURATION_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, coerce_decimal(value, name))
        if not isinstance(self.video, RenderMediaInfo):
            raise TypeError("video must be a RenderMediaInfo")
        if self.narration is not None and not isinstance(self.narration, RenderMediaInfo):
            raise TypeError("narration must be a RenderMediaInfo or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "scene_number": self.scene_number,
            "title": self.title,
            "status": self.status,
            "planned_duration_s": _decimal_text(self.planned_duration_s),
            "measured_duration_s": _decimal_text(self.measured_duration_s),
            "effective_duration_s": _decimal_text(self.effective_duration_s),
            "video_hold_duration_s": _decimal_text(self.video_hold_duration_s),
            "silent_tail_duration_s": _decimal_text(self.silent_tail_duration_s),
            "timeline_start_s": _decimal_text(self.timeline_start_s),
            "timeline_end_s": _decimal_text(self.timeline_end_s),
            "planned_timeline_start_s": _decimal_text(self.planned_timeline_start_s),
            "planned_timeline_end_s": _decimal_text(self.planned_timeline_end_s),
            "duration_delta_s": _decimal_text(self.duration_delta_s),
            "narration_duration_s": _optional_decimal_text(self.narration_duration_s),
            "narration_delta_s": _optional_decimal_text(self.narration_delta_s),
            "narration_audio_path": self.narration_audio_path,
            "narration_text": self.narration_text,
            "generated_video_path": self.generated_video_path,
            "video": self.video.to_dict(),
            "narration": self.narration.to_dict() if self.narration is not None else None,
        }


@dataclass(frozen=True)
class RenderPlan:
    """A complete, read-only render timeline derived from measured media."""

    project_id: str
    policy: RenderPolicy
    scenes: tuple[RenderScene, ...]
    warnings: tuple[ValidationIssue, ...]
    total_planned_duration_s: Decimal
    total_render_duration_s: Decimal
    output_width: int
    output_height: int
    output_fps: float

    def __post_init__(self) -> None:
        if not isinstance(self.project_id, str) or not self.project_id.strip():
            raise ValueError("project_id must be a non-empty string")
        if not isinstance(self.policy, RenderPolicy):
            raise TypeError("policy must be a RenderPolicy")
        object.__setattr__(self, "scenes", tuple(self.scenes))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for scene in self.scenes:
            if not isinstance(scene, RenderScene):
                raise TypeError("scenes must contain RenderScene entries")
        for issue in self.warnings:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("warnings must contain ValidationIssue entries")
        object.__setattr__(
            self,
            "total_planned_duration_s",
            coerce_decimal(
                self.total_planned_duration_s, "total_planned_duration_s", non_negative=True
            ),
        )
        object.__setattr__(
            self,
            "total_render_duration_s",
            coerce_decimal(
                self.total_render_duration_s, "total_render_duration_s", non_negative=True
            ),
        )
        object.__setattr__(self, "output_width", _coerce_int(self.output_width, "output_width", positive=True))
        object.__setattr__(self, "output_height", _coerce_int(self.output_height, "output_height", positive=True))
        object.__setattr__(self, "output_fps", _coerce_float(self.output_fps, "output_fps", positive=True))

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "policy": self.policy.to_dict(),
            "scene_count": self.scene_count,
            "total_planned_duration_s": _decimal_text(self.total_planned_duration_s),
            "total_render_duration_s": _decimal_text(self.total_render_duration_s),
            "output_width": self.output_width,
            "output_height": self.output_height,
            "output_fps": self.output_fps,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }
