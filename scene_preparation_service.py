"""MCP-independent scene video preparation and silent concatenation.

This service turns an existing :class:`~render_models.RenderPlan` into one
silent MP4.  It never writes project or scene manifests, never modifies source
videos, and never creates audio.  Planning stays in ``render_planner`` and
final validation stays in ``render_validation``; this module only normalizes
video streams, concatenates them, and reports structured outcomes.

FFmpeg and FFprobe are invoked through argument lists (never shell strings).
All generated files live in a unique directory under the project, and existing
render directories or final videos are never overwritten.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import media_service
from manifest_validation import ValidationIssue
from project_paths import (
    PathValidationError,
    resolve_within_project,
    validate_identifier,
)
from project_service import ProjectService
from render_models import (
    RenderPlan,
    RenderPolicy,
    RenderScene,
    coerce_decimal,
)
from render_planner import RenderPlanner, RenderPlanningError
from render_validation import RenderExpectations, RenderValidator
from scene_service import SceneService

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 24.0
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"
DEFAULT_CONTAINER = "mp4"
DEFAULT_PRESET = "medium"
DEFAULT_CRF = 18
DEFAULT_MOVFLAGS = "+faststart"
DEFAULT_RENDER_ROOT = "video/renders"
DEFAULT_DURATION_TOLERANCE_S = Decimal("0.5")
DEFAULT_FFMPEG_TIMEOUT_S = float(os.getenv("MEDIA_FFMPEG_TIMEOUT_S", "180"))
DEFAULT_FFPROBE_TIMEOUT_S = float(os.getenv("MEDIA_FFPROBE_TIMEOUT_S", "30"))
DEFAULT_FFMPEG_BIN = os.getenv("MEDIA_FFMPEG_BIN", "ffmpeg")
DEFAULT_FFPROBE_BIN = os.getenv("MEDIA_FFPROBE_BIN", "ffprobe")

SUPPORTED_PRESETS = frozenset(
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
CONCAT_MODES = frozenset({"auto", "copy", "reencode"})
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_STDERR_TAIL_CHARS = 1000

ProbeCallable = Callable[..., Mapping[str, object]]


class ScenePreparationError(RuntimeError):
    """Structured failure raised for any unsupported preparation outcome."""

    def __init__(
        self,
        code: str,
        message: str,
        location: str = "",
        *,
        stage: str = "prepare",
        stderr_tail: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location
        self.stage = stage
        self.stderr_tail = stderr_tail
        self.issue = ValidationIssue(code, message, location)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "stage": self.stage,
            "stderr_tail": self.stderr_tail,
            "issues": [self.issue.to_dict()],
        }


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if any(character in value for character in "\r\n\x00"):
        raise ValueError(f"{label} must not contain control characters")
    return value


def _coerce_int(value: object, label: str, *, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _coerce_float(value: object, label: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _coerce_token(value: object, label: str) -> str:
    token = _require_nonempty_string(value, label)
    if _SAFE_TOKEN_RE.fullmatch(token) is None:
        raise ValueError(f"{label} contains unsupported characters: {token!r}")
    return token


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one FFmpeg/FFprobe subprocess invocation."""

    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        command = tuple(str(part) for part in self.command)
        if not command:
            raise ValueError("command must not be empty")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "returncode", _coerce_int(self.returncode, "returncode", positive=False))
        for name in ("stdout", "stderr"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")

    @property
    def stderr_tail(self) -> str:
        return self.stderr[-_STDERR_TAIL_CHARS:]

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "stderr_tail": self.stderr_tail,
        }


CommandRunner = Callable[[Sequence[str], float], CommandResult]


@dataclass(frozen=True)
class ScenePreparationConfig:
    """Output profile and runtime settings for one silent preparation run."""

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: float = DEFAULT_FPS
    video_codec: str = DEFAULT_VIDEO_CODEC
    pixel_format: str = DEFAULT_PIXEL_FORMAT
    container: str = DEFAULT_CONTAINER
    preset: str = DEFAULT_PRESET
    crf: int = DEFAULT_CRF
    movflags: str = DEFAULT_MOVFLAGS
    concat_mode: str = "auto"
    duration_tolerance_s: Decimal = DEFAULT_DURATION_TOLERANCE_S
    render_root: str = DEFAULT_RENDER_ROOT
    render_id: str | None = None
    ffmpeg_bin: str = DEFAULT_FFMPEG_BIN
    ffprobe_bin: str = DEFAULT_FFPROBE_BIN
    ffmpeg_timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S
    ffprobe_timeout_s: float = DEFAULT_FFPROBE_TIMEOUT_S
    policy: RenderPolicy = field(default_factory=RenderPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", _coerce_int(self.width, "width"))
        object.__setattr__(self, "height", _coerce_int(self.height, "height"))
        if self.width % 2 or self.height % 2:
            raise ValueError("width and height must be even for yuv420p output")
        object.__setattr__(self, "fps", _coerce_float(self.fps, "fps"))
        object.__setattr__(self, "video_codec", _coerce_token(self.video_codec, "video_codec"))
        object.__setattr__(self, "pixel_format", _coerce_token(self.pixel_format, "pixel_format"))
        object.__setattr__(self, "container", _coerce_token(self.container, "container"))
        if self.container.lower() != "mp4":
            raise ValueError("container must be 'mp4' in this milestone")
        preset = _coerce_token(self.preset, "preset").lower()
        if preset not in SUPPORTED_PRESETS:
            raise ValueError(f"unsupported preset {preset!r}; expected one of {sorted(SUPPORTED_PRESETS)}")
        object.__setattr__(self, "preset", preset)
        object.__setattr__(self, "crf", _coerce_int(self.crf, "crf", positive=False))
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be in [0, 51]")
        object.__setattr__(self, "movflags", _coerce_token(self.movflags, "movflags"))
        mode = _coerce_token(self.concat_mode, "concat_mode").lower()
        if mode not in CONCAT_MODES:
            raise ValueError(f"unsupported concat_mode {mode!r}; expected one of {sorted(CONCAT_MODES)}")
        object.__setattr__(self, "concat_mode", mode)
        object.__setattr__(
            self,
            "duration_tolerance_s",
            coerce_decimal(self.duration_tolerance_s, "duration_tolerance_s", non_negative=True),
        )
        root = _require_nonempty_string(self.render_root, "render_root")
        raw_root = Path(root)
        if raw_root.is_absolute() or ".." in raw_root.parts:
            raise ValueError("render_root must be a relative path inside the project")
        object.__setattr__(self, "render_root", raw_root.as_posix())
        if self.render_id is not None:
            identifier = _require_nonempty_string(self.render_id, "render_id")
            validate_identifier(identifier, "render_id")
            object.__setattr__(self, "render_id", identifier)
        object.__setattr__(self, "ffmpeg_bin", _require_nonempty_string(self.ffmpeg_bin, "ffmpeg_bin"))
        object.__setattr__(self, "ffprobe_bin", _require_nonempty_string(self.ffprobe_bin, "ffprobe_bin"))
        object.__setattr__(
            self,
            "ffmpeg_timeout_s",
            _coerce_float(self.ffmpeg_timeout_s, "ffmpeg_timeout_s"),
        )
        object.__setattr__(
            self,
            "ffprobe_timeout_s",
            _coerce_float(self.ffprobe_timeout_s, "ffprobe_timeout_s"),
        )
        object.__setattr__(self, "policy", RenderPolicy.from_dict(self.policy))

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "video_codec": self.video_codec,
            "pixel_format": self.pixel_format,
            "container": self.container,
            "preset": self.preset,
            "crf": self.crf,
            "movflags": self.movflags,
            "concat_mode": self.concat_mode,
            "duration_tolerance_s": _decimal_text(self.duration_tolerance_s),
            "render_root": self.render_root,
            "render_id": self.render_id,
            "ffmpeg_bin": self.ffmpeg_bin,
            "ffprobe_bin": self.ffprobe_bin,
            "ffmpeg_timeout_s": self.ffmpeg_timeout_s,
            "ffprobe_timeout_s": self.ffprobe_timeout_s,
            "policy": self.policy.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | ScenePreparationConfig | None = None
    ) -> ScenePreparationConfig:
        if isinstance(data, ScenePreparationConfig):
            return data
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError("render configuration must be an object")
        return cls(
            width=data.get("width", DEFAULT_WIDTH),
            height=data.get("height", DEFAULT_HEIGHT),
            fps=data.get("fps", DEFAULT_FPS),
            video_codec=data.get("video_codec", DEFAULT_VIDEO_CODEC),
            pixel_format=data.get("pixel_format", DEFAULT_PIXEL_FORMAT),
            container=data.get("container", DEFAULT_CONTAINER),
            preset=data.get("preset", DEFAULT_PRESET),
            crf=data.get("crf", DEFAULT_CRF),
            movflags=data.get("movflags", DEFAULT_MOVFLAGS),
            concat_mode=data.get("concat_mode", "auto"),
            duration_tolerance_s=data.get(
                "duration_tolerance_s", DEFAULT_DURATION_TOLERANCE_S
            ),
            render_root=data.get("render_root", DEFAULT_RENDER_ROOT),
            render_id=data.get("render_id"),
            ffmpeg_bin=data.get("ffmpeg_bin", DEFAULT_FFMPEG_BIN),
            ffprobe_bin=data.get("ffprobe_bin", DEFAULT_FFPROBE_BIN),
            ffmpeg_timeout_s=data.get("ffmpeg_timeout_s", DEFAULT_FFMPEG_TIMEOUT_S),
            ffprobe_timeout_s=data.get("ffprobe_timeout_s", DEFAULT_FFPROBE_TIMEOUT_S),
            policy=RenderPolicy.from_dict(data.get("policy")),
        )


@dataclass(frozen=True)
class FinalValidation:
    """Read-only view of the final silent-video validation verdict."""

    valid: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a boolean")
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for name, issues in (("errors", self.errors), ("warnings", self.warnings)):
            for issue in issues:
                if not isinstance(issue, ValidationIssue):
                    raise TypeError(f"{name} must contain ValidationIssue entries")

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class PreparedScene:
    """Preparation outcome for one scene in render-plan order."""

    scene_id: str
    scene_number: int
    order: int
    title: str
    source_path: str
    prepared_path: str
    video_codec: str
    pixel_format: str
    input_width: int
    input_height: int
    input_fps: float
    output_width: int
    output_height: int
    output_fps: float
    source_duration_s: Decimal
    prepared_duration_s: Decimal
    hold_duration_s: Decimal
    command: CommandResult
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.scene_id, "scene_id")
        object.__setattr__(self, "scene_number", _coerce_int(self.scene_number, "scene_number"))
        object.__setattr__(self, "order", _coerce_int(self.order, "order", positive=False))
        object.__setattr__(self, "title", _require_nonempty_string(self.title, "title"))
        object.__setattr__(self, "source_path", _require_nonempty_string(self.source_path, "source_path"))
        object.__setattr__(
            self, "prepared_path", _require_nonempty_string(self.prepared_path, "prepared_path")
        )
        object.__setattr__(self, "video_codec", _coerce_token(self.video_codec, "video_codec"))
        object.__setattr__(self, "pixel_format", _coerce_token(self.pixel_format, "pixel_format"))
        for name in ("input_width", "input_height", "output_width", "output_height"):
            object.__setattr__(self, name, _coerce_int(getattr(self, name), name))
        for name in ("input_fps", "output_fps"):
            object.__setattr__(self, name, _coerce_float(getattr(self, name), name))
        for name in ("source_duration_s", "prepared_duration_s"):
            object.__setattr__(self, name, coerce_decimal(getattr(self, name), name, positive=True))
        object.__setattr__(
            self,
            "hold_duration_s",
            coerce_decimal(self.hold_duration_s, "hold_duration_s", non_negative=True),
        )
        if not isinstance(self.command, CommandResult):
            raise TypeError("command must be a CommandResult")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for issue in self.warnings:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("warnings must contain ValidationIssue entries")

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "scene_number": self.scene_number,
            "order": self.order,
            "title": self.title,
            "source_path": self.source_path,
            "prepared_path": self.prepared_path,
            "video_codec": self.video_codec,
            "pixel_format": self.pixel_format,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "input_fps": self.input_fps,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "output_fps": self.output_fps,
            "source_duration_s": _decimal_text(self.source_duration_s),
            "prepared_duration_s": _decimal_text(self.prepared_duration_s),
            "hold_duration_s": _decimal_text(self.hold_duration_s),
            "command": self.command.to_dict(),
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class PreparationResult:
    """Overall outcome of one silent preparation and concatenation run."""

    project_id: str
    render_id: str
    render_dir: str
    final_video_path: str
    final_duration_s: Decimal
    final_width: int
    final_height: int
    final_fps: float
    scene_order: tuple[str, ...]
    prepared_scenes: tuple[PreparedScene, ...]
    concat_mode: str
    concat_command: CommandResult
    final_validation: FinalValidation
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.project_id, "project_id")
        validate_identifier(self.render_id, "render_id")
        for name in ("render_dir", "final_video_path"):
            object.__setattr__(self, name, _require_nonempty_string(getattr(self, name), name))
        object.__setattr__(
            self,
            "final_duration_s",
            coerce_decimal(self.final_duration_s, "final_duration_s", positive=True),
        )
        for name in ("final_width", "final_height"):
            object.__setattr__(self, name, _coerce_int(getattr(self, name), name))
        object.__setattr__(self, "final_fps", _coerce_float(self.final_fps, "final_fps"))
        object.__setattr__(self, "scene_order", tuple(self.scene_order))
        object.__setattr__(self, "prepared_scenes", tuple(self.prepared_scenes))
        for scene in self.prepared_scenes:
            if not isinstance(scene, PreparedScene):
                raise TypeError("prepared_scenes must contain PreparedScene entries")
        if tuple(scene.scene_id for scene in self.prepared_scenes) != self.scene_order:
            raise ValueError("scene_order must match prepared_scenes order")
        object.__setattr__(self, "concat_mode", _coerce_token(self.concat_mode, "concat_mode"))
        if not isinstance(self.concat_command, CommandResult):
            raise TypeError("concat_command must be a CommandResult")
        if not isinstance(self.final_validation, FinalValidation):
            raise TypeError("final_validation must be a FinalValidation")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for issue in self.warnings:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("warnings must contain ValidationIssue entries")

    @property
    def prepared_paths(self) -> tuple[str, ...]:
        return tuple(scene.prepared_path for scene in self.prepared_scenes)

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "render_id": self.render_id,
            "render_dir": self.render_dir,
            "final_video_path": self.final_video_path,
            "final_duration_s": _decimal_text(self.final_duration_s),
            "final_width": self.final_width,
            "final_height": self.final_height,
            "final_fps": self.final_fps,
            "scene_order": list(self.scene_order),
            "prepared_paths": list(self.prepared_paths),
            "prepared_scenes": [scene.to_dict() for scene in self.prepared_scenes],
            "concat_mode": self.concat_mode,
            "concat_command": self.concat_command.to_dict(),
            "final_validation": self.final_validation.to_dict(),
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def _stderr_detail(result: CommandResult) -> str:
    """Keep a bounded head and tail so actionable prefixes are not lost."""
    text = (result.stderr or result.stdout).strip()
    if len(text) <= _STDERR_TAIL_CHARS:
        return text
    half = (_STDERR_TAIL_CHARS - 40) // 2
    return f"{text[:half]}\n... [truncated] ...\n{text[-half:]}"


def _run_command(
    command: Sequence[str],
    timeout_s: float,
    *,
    label: str,
    tool: str,
) -> CommandResult:
    """Execute an FFmpeg-family command without a shell and return its outcome."""
    values = [str(part) for part in command]
    if not values:
        raise ValueError("command must not be empty")
    try:
        completed = subprocess.run(
            values,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ScenePreparationError(
            f"preparation.{tool}.not_installed",
            f"{label} executable not found: {values[0]}",
            stage=tool,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ScenePreparationError(
            f"preparation.{tool}.timeout",
            f"{label} timed out after {timeout_s:.0f}s",
            stage=tool,
        ) from exc
    result = CommandResult(
        tuple(values),
        completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
    if completed.returncode != 0:
        detail = _stderr_detail(result)
        message = f"{label} failed (exit {completed.returncode})"
        if detail:
            message = f"{message}: {detail}"
        raise ScenePreparationError(
            f"preparation.{tool}.failed", message, stage=tool, stderr_tail=detail
        )
    return result


def run_ffmpeg(command: Sequence[str], timeout_s: float) -> CommandResult:
    """Run an FFmpeg argument list with a timeout and structured failures."""
    return _run_command(command, timeout_s, label="FFmpeg", tool="ffmpeg")


def run_ffprobe(command: Sequence[str], timeout_s: float) -> CommandResult:
    """Run an FFprobe argument list with a timeout and structured failures."""
    return _run_command(command, timeout_s, label="FFprobe", tool="ffprobe")


def _parse_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _parse_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _parse_rate(value: object) -> float | None:
    if isinstance(value, str):
        numerator, separator, denominator = value.partition("/")
        try:
            top = float(numerator)
            bottom = float(denominator) if separator else 1.0
        except ValueError:
            return None
        if bottom == 0:
            return None
        return top / bottom if math.isfinite(top / bottom) and top / bottom > 0 else None
    return _parse_float(value)


def probe_video(
    path: str | Path,
    *,
    ffprobe_bin: str = DEFAULT_FFPROBE_BIN,
    timeout_s: float = DEFAULT_FFPROBE_TIMEOUT_S,
) -> dict[str, object]:
    """Probe one video file with FFprobe, including pixel format."""
    target = Path(path).expanduser().resolve()
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(target),
    ]
    result = run_ffprobe(command, timeout_s)
    try:
        decoded: object = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ScenePreparationError(
            "preparation.ffprobe.invalid_json",
            "FFprobe returned invalid JSON",
            str(target),
            stage="ffprobe",
        ) from exc
    if not isinstance(decoded, dict):
        raise ScenePreparationError(
            "preparation.ffprobe.invalid_json",
            "FFprobe returned an unexpected payload",
            str(target),
            stage="ffprobe",
        )
    streams = decoded.get("streams")
    if not isinstance(streams, list):
        raise ScenePreparationError(
            "preparation.ffprobe.missing_streams",
            "FFprobe output has no stream list",
            str(target),
            stage="ffprobe",
        )
    video = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        None,
    )
    audio = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ),
        None,
    )
    container = decoded.get("format")
    if not isinstance(container, dict):
        container = {}
    rate = video.get("avg_frame_rate") if video is not None else None
    if not rate and video is not None:
        rate = video.get("r_frame_rate")
    duration = _parse_float(container.get("duration"))
    if duration is None and video is not None:
        duration = _parse_float(video.get("duration"))
    if duration is None and audio is not None:
        duration = _parse_float(audio.get("duration"))
    size = _parse_int(container.get("size"))
    if size is None and target.is_file():
        size = target.stat().st_size
    codec = video.get("codec_name") if video is not None else None
    audio_codec = audio.get("codec_name") if audio is not None else None
    pixel_format = video.get("pix_fmt") if video is not None else None
    return {
        "status": "ok",
        "path": str(target),
        "duration_s": duration,
        "size_bytes": size,
        "width": _parse_int(video.get("width")) if video is not None else None,
        "height": _parse_int(video.get("height")) if video is not None else None,
        "fps": _parse_rate(rate),
        "video_codec": codec if isinstance(codec, str) and codec.strip() else None,
        "audio_codec": (
            audio_codec if isinstance(audio_codec, str) and audio_codec.strip() else None
        ),
        "pixel_format": (
            pixel_format if isinstance(pixel_format, str) and pixel_format.strip() else None
        ),
        "sample_rate": _parse_int(audio.get("sample_rate")) if audio is not None else None,
        "channels": _parse_int(audio.get("channels")) if audio is not None else None,
        "format_name": container.get("format_name"),
    }


def escape_concat_path(path: str | Path) -> str:
    """Escape one path for an FFmpeg concat demuxer ``file`` directive."""
    text = _require_nonempty_string(str(path), "concat path")
    normalized = text.replace("\\", "/")
    return normalized.replace("'", "'\\''")


def expected_probe_codec(video_codec: str) -> str:
    """Return the FFprobe ``codec_name`` produced by an encoder name."""
    return "h264" if video_codec in {"h264", "libx264"} else video_codec


def scale_filter(config: ScenePreparationConfig) -> str:
    """Build the aspect-preserving scale/pad/FPS filter chain."""
    return (
        f"scale={config.width}:{config.height}:force_original_aspect_ratio=decrease,"
        f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,fps={config.fps:g}"
    )


def build_prepare_command(
    source: str | Path,
    output: str | Path,
    config: ScenePreparationConfig,
) -> list[str]:
    """Build explicit arguments that normalize one source to the output profile."""
    return [
        config.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        scale_filter(config),
        "-c:v",
        config.video_codec,
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        "-pix_fmt",
        config.pixel_format,
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-movflags",
        config.movflags,
        str(output),
    ]


def build_concat_command(
    concat_list: str | Path,
    output: str | Path,
    config: ScenePreparationConfig,
    *,
    reencode: bool,
) -> list[str]:
    """Build concat-demuxer arguments, optionally re-encoding the final stream."""
    command = [
        config.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
    ]
    if reencode:
        command.extend(
            [
                "-c:v",
                config.video_codec,
                "-preset",
                config.preset,
                "-crf",
                str(config.crf),
                "-pix_fmt",
                config.pixel_format,
            ]
        )
    else:
        command.extend(["-c", "copy"])
    command.extend(["-movflags", config.movflags, str(output)])
    return command


def _new_temp_file(directory: Path, stem: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{stem}-", suffix=".tmp.mp4", dir=str(directory)
    )
    os.close(descriptor)
    return Path(name)


class ScenePreparationService:
    """Prepare render-plan scenes and concatenate them into one silent MP4."""

    def __init__(
        self,
        projects: ProjectService | None = None,
        scenes: SceneService | None = None,
        *,
        planner: RenderPlanner | None = None,
        validator: RenderValidator | None = None,
        command_runner: CommandRunner | None = None,
        probe: ProbeCallable | None = None,
        check_executables: bool | None = None,
    ) -> None:
        self.projects = projects if projects is not None else ProjectService()
        self.scenes = scenes if scenes is not None else SceneService(self.projects)
        self.planner = planner
        self.validator = validator
        self.command_runner: CommandRunner = (
            command_runner if command_runner is not None else run_ffmpeg
        )
        self.probe: ProbeCallable = probe if probe is not None else probe_video
        self.check_executables = (
            check_executables
            if check_executables is not None
            else command_runner is None or probe is None
        )

    def prepare(
        self,
        project_id: str,
        config: ScenePreparationConfig | Mapping[str, Any] | None = None,
    ) -> dict[str, object]:
        """Return a JSON-serializable status, result or structured error."""
        try:
            result = self.prepare_scene_videos(project_id, config)
        except ScenePreparationError as exc:
            return {"status": "error", "error": exc.to_dict()}
        return {"status": "ok", "result": result.to_dict()}

    def prepare_scene_videos(
        self,
        project_id: str,
        config: ScenePreparationConfig | Mapping[str, Any] | None = None,
    ) -> PreparationResult:
        """Prepare every planned scene and return the silent final video result."""
        effective_config = self._config(config)
        try:
            validate_identifier(project_id, "project_id")
        except PathValidationError as exc:
            raise ScenePreparationError(
                "preparation.project.invalid_id",
                str(exc),
                "project_id",
                stage="project",
            ) from exc
        self._check_executables(effective_config)

        plan = self._build_plan(project_id, effective_config)
        root = self.projects.project_dir(project_id)
        render_dir, render_id = self._create_render_dir(root, effective_config)
        logger.info(
            "Preparing %d scenes for project %s in %s",
            plan.scene_count,
            project_id,
            render_dir,
        )

        warnings: list[ValidationIssue] = list(plan.warnings)
        if (
            plan.output_width != effective_config.width
            or plan.output_height != effective_config.height
            or abs(plan.output_fps - effective_config.fps)
            > effective_config.policy.validation.fps_tolerance
        ):
            warnings.append(
                ValidationIssue(
                    "preparation.profile.differs_from_plan",
                    "Render plan output profile "
                    f"{plan.output_width}x{plan.output_height}@{plan.output_fps:g} "
                    "differs from the preparation profile "
                    f"{effective_config.width}x{effective_config.height}"
                    f"@{effective_config.fps:g}; using the preparation profile",
                    "render_plan.output",
                )
            )

        prepared: list[PreparedScene] = []
        for order, scene in enumerate(plan.scenes):
            prepared.append(
                self._prepare_scene(
                    root,
                    render_dir,
                    scene,
                    order,
                    effective_config,
                    warnings,
                )
            )

        reencode = self._requires_reencode(prepared, effective_config, warnings)
        concat_list = self._write_concat_list(render_dir, prepared)
        final_path = render_dir / "final_silent.mp4"
        if final_path.exists():
            raise ScenePreparationError(
                "preparation.final.collision",
                f"Final video already exists: {final_path}",
                str(final_path),
                stage="final",
            )
        temp_final = _new_temp_file(render_dir, "final_silent")
        concat_command = self._execute(
            build_concat_command(
                concat_list,
                temp_final,
                effective_config,
                reencode=reencode,
            ),
            effective_config,
        )
        if final_path.exists():
            raise ScenePreparationError(
                "preparation.final.collision",
                f"Final video appeared during concatenation: {final_path}. No file was replaced.",
                str(final_path),
                stage="final",
            )
        temp_final.replace(final_path)
        self._require_nonempty_output(final_path, "final video")

        final_info = self._probe(final_path, effective_config)
        if final_info.get("audio_codec") is not None:
            raise ScenePreparationError(
                "preparation.final.unexpected_audio",
                "Final video contains an audio stream; no audio is created in this milestone",
                str(final_path),
                stage="final",
            )
        final_duration = coerce_decimal(
            final_info.get("duration_s"), "final duration", positive=True
        )
        final_width = _coerce_int(final_info.get("width"), "final width")
        final_height = _coerce_int(final_info.get("height"), "final height")
        final_fps = _coerce_float(final_info.get("fps"), "final FPS")

        expected_duration = sum(
            (scene.prepared_duration_s for scene in prepared), Decimal(0)
        )
        expectations = RenderExpectations(
            width=effective_config.width,
            height=effective_config.height,
            fps=effective_config.fps,
            expected_duration_s=expected_duration,
            duration_tolerance_s=effective_config.duration_tolerance_s,
            fps_tolerance=effective_config.policy.validation.fps_tolerance,
            require_video_stream=True,
            require_positive_duration=True,
        )
        validator = self.validator or RenderValidator(
            probe=lambda path: self._probe(path, effective_config)
        )
        verdict = validator.validate_final(
            final_path, expectations, require_audio=False
        )
        final_validation = FinalValidation(
            valid=verdict.valid,
            errors=tuple(verdict.errors),
            warnings=tuple(verdict.warnings),
        )
        warnings.extend(final_validation.warnings)
        if not final_validation.valid:
            details = "; ".join(issue.message for issue in final_validation.errors)
            raise ScenePreparationError(
                "preparation.final.invalid",
                f"Final silent video failed validation: {details}",
                str(final_path),
                stage="final",
            )
        self._require_nonempty_output(final_path, "validated final video")

        result = PreparationResult(
            project_id=project_id,
            render_id=render_id,
            render_dir=str(render_dir),
            final_video_path=str(final_path),
            final_duration_s=final_duration,
            final_width=final_width,
            final_height=final_height,
            final_fps=final_fps,
            scene_order=tuple(scene.scene_id for scene in prepared),
            prepared_scenes=tuple(prepared),
            concat_mode=("reencode" if reencode else "copy"),
            concat_command=concat_command,
            final_validation=final_validation,
            warnings=tuple(warnings),
        )
        logger.info("Prepared silent render %s for project %s", render_id, project_id)
        return result

    @staticmethod
    def _config(
        value: ScenePreparationConfig | Mapping[str, Any] | None,
    ) -> ScenePreparationConfig:
        try:
            return ScenePreparationConfig.from_dict(value)
        except (TypeError, ValueError) as exc:
            raise ScenePreparationError(
                "preparation.config.invalid", str(exc), stage="config"
            ) from exc

    def _check_executables(self, config: ScenePreparationConfig) -> None:
        if not self.check_executables:
            return
        if shutil.which(config.ffmpeg_bin) is None:
            raise ScenePreparationError(
                "preparation.ffmpeg.not_installed",
                f"FFmpeg executable not found: {config.ffmpeg_bin}",
                stage="ffmpeg",
            )
        if shutil.which(config.ffprobe_bin) is None:
            raise ScenePreparationError(
                "preparation.ffprobe.not_installed",
                f"FFprobe executable not found: {config.ffprobe_bin}",
                stage="ffprobe",
            )

    def _probe(
        self, path: Path, config: ScenePreparationConfig
    ) -> Mapping[str, object]:
        try:
            info = self.probe(path, timeout_s=config.ffprobe_timeout_s)
        except ScenePreparationError:
            raise
        except media_service.MediaError as exc:
            raise ScenePreparationError(
                "preparation.ffprobe.failed",
                f"Could not probe {path}: {exc}",
                str(path),
                stage="ffprobe",
            ) from exc
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise ScenePreparationError(
                "preparation.ffprobe.failed",
                f"Could not probe {path}: {exc}",
                str(path),
                stage="ffprobe",
            ) from exc
        if not isinstance(info, Mapping):
            raise ScenePreparationError(
                "preparation.ffprobe.failed",
                f"Probe returned {type(info).__name__} instead of a mapping",
                str(path),
                stage="ffprobe",
            )
        return info

    def _build_plan(
        self, project_id: str, config: ScenePreparationConfig
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
            code = (
                exc.code.replace("render.", "preparation.", 1)
                if exc.code.startswith("render.")
                else "preparation.plan.failed"
            )
            lowered = exc.message.lower()
            if "ffprobe executable not found" in lowered:
                code = "preparation.ffprobe.not_installed"
            elif "timed out" in lowered:
                code = "preparation.ffprobe.timeout"
            raise ScenePreparationError(
                code, exc.message, exc.location, stage="plan"
            ) from exc
        except PathValidationError as exc:
            raise ScenePreparationError(
                "preparation.project.unsafe_path",
                str(exc),
                "project_id",
                stage="project",
            ) from exc

    def _create_render_dir(
        self, root: Path, config: ScenePreparationConfig
    ) -> tuple[Path, str]:
        suffix = config.render_id
        if suffix is None:
            suffix = uuid.uuid4().hex
        elif suffix.startswith("render-"):
            suffix = suffix[len("render-") :]
        try:
            identifier = validate_identifier(suffix, "render_id")
            render_id = validate_identifier(f"render-{identifier}", "render_id")
        except PathValidationError as exc:
            raise ScenePreparationError(
                "preparation.render.unsafe_path", str(exc), stage="render"
            ) from exc
        relative = Path(config.render_root) / render_id
        try:
            target = resolve_within_project(root, relative, "render directory")
        except PathValidationError as exc:
            raise ScenePreparationError(
                "preparation.render.unsafe_path", str(exc), stage="render"
            ) from exc
        if target.exists():
            raise ScenePreparationError(
                "preparation.render.collision",
                f"Render directory already exists: {target}",
                str(target),
                stage="render",
            )
        try:
            target.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise ScenePreparationError(
                "preparation.render.create_failed",
                f"Could not create render directory {target}: {exc}",
                str(target),
                stage="render",
            ) from exc
        return target, render_id

    def _prepare_scene(
        self,
        root: Path,
        render_dir: Path,
        scene: RenderScene,
        order: int,
        config: ScenePreparationConfig,
        warnings: list[ValidationIssue],
    ) -> PreparedScene:
        location = f"scenes[{order}] ({scene.scene_id})"
        try:
            source = resolve_within_project(
                root, scene.generated_video_path, "generated_video_path"
            )
        except PathValidationError as exc:
            raise ScenePreparationError(
                "preparation.asset.unsafe_path",
                str(exc),
                f"{location}.generated_video_path",
                stage="scene",
            ) from exc
        self._require_nonempty_output(source, f"scene {scene.scene_id} source")

        source_duration = coerce_decimal(
            scene.measured_duration_s, f"{location}.measured_duration_s", positive=True
        )
        input_width = _coerce_int(scene.video.width, f"{location}.width")
        input_height = _coerce_int(scene.video.height, f"{location}.height")
        input_fps = _coerce_float(scene.video.fps, f"{location}.fps")

        prepared_path = render_dir / f"scene_{order:04d}_{scene.scene_id}.mp4"
        if prepared_path.exists():
            raise ScenePreparationError(
                "preparation.scene.collision",
                f"Prepared video already exists: {prepared_path}",
                str(prepared_path),
                stage="scene",
            )
        temp_output = _new_temp_file(render_dir, prepared_path.stem)
        command = self._execute(
            build_prepare_command(source, temp_output, config), config
        )
        temp_output.replace(prepared_path)
        self._require_nonempty_output(prepared_path, f"prepared video for {scene.scene_id}")

        info = self._probe(prepared_path, config)
        prepared_duration = coerce_decimal(
            info.get("duration_s"), f"{location}.prepared_duration_s", positive=True
        )
        codec = info.get("video_codec")
        if not isinstance(codec, str) or not codec.strip():
            raise ScenePreparationError(
                "preparation.scene.missing_video_stream",
                f"Prepared video has no video stream: {prepared_path}",
                str(prepared_path),
                stage="scene",
            )
        pixel_format = info.get("pixel_format")
        if not isinstance(pixel_format, str) or not pixel_format.strip():
            pixel_format = config.pixel_format
        output_fps = _coerce_float(info.get("fps"), f"{location}.output_fps")
        output_width = _coerce_int(info.get("width"), f"{location}.output_width")
        output_height = _coerce_int(info.get("height"), f"{location}.output_height")

        scene_warnings: list[ValidationIssue] = []
        if scene.video_hold_duration_s > 0:
            hold = coerce_decimal(scene.video_hold_duration_s, "hold_duration_s", non_negative=True)
            scene_warnings.append(
                ValidationIssue(
                    "preparation.scene.hold_recorded",
                    f"Planned hold of {_decimal_text(hold)}s recorded; "
                    "video holding and narration are not implemented here",
                    f"{location}.video_hold_duration_s",
                )
            )
        drift = abs(prepared_duration - source_duration)
        if drift > config.duration_tolerance_s:
            scene_warnings.append(
                ValidationIssue(
                    "preparation.scene.duration_drift",
                    f"Prepared duration differs from measured source by "
                    f"{_decimal_text(drift)}s after FPS normalization",
                    f"{location}.duration",
                )
            )
        warnings.extend(scene_warnings)
        return PreparedScene(
            scene_id=scene.scene_id,
            scene_number=scene.scene_number,
            order=order,
            title=scene.title,
            source_path=str(source),
            prepared_path=str(prepared_path),
            video_codec=codec,
            pixel_format=pixel_format,
            input_width=input_width,
            input_height=input_height,
            input_fps=input_fps,
            output_width=output_width,
            output_height=output_height,
            output_fps=output_fps,
            source_duration_s=source_duration,
            prepared_duration_s=prepared_duration,
            hold_duration_s=scene.video_hold_duration_s,
            command=command,
            warnings=tuple(scene_warnings),
        )

    def _execute(
        self, command: Sequence[str], config: ScenePreparationConfig
    ) -> CommandResult:
        logger.debug("Running FFmpeg command with %d arguments", len(command))
        return self.command_runner(list(command), config.ffmpeg_timeout_s)

    def _requires_reencode(
        self,
        prepared: Sequence[PreparedScene],
        config: ScenePreparationConfig,
        warnings: list[ValidationIssue],
    ) -> bool:
        problems: list[str] = []
        codec_name = expected_probe_codec(config.video_codec)
        for scene in prepared:
            if scene.video_codec != codec_name:
                problems.append(f"{scene.scene_id}: codec {scene.video_codec}")
            if scene.pixel_format != config.pixel_format:
                problems.append(f"{scene.scene_id}: pixel format {scene.pixel_format}")
            if scene.output_width != config.width or scene.output_height != config.height:
                problems.append(
                    f"{scene.scene_id}: {scene.output_width}x{scene.output_height}"
                )
            if (
                abs(scene.output_fps - config.fps)
                > config.policy.validation.fps_tolerance
            ):
                problems.append(f"{scene.scene_id}: {scene.output_fps:g} FPS")
        incompatible = bool(problems)
        if config.concat_mode == "copy" and incompatible:
            raise ScenePreparationError(
                "preparation.concat.incompatible_streams",
                "Prepared streams are incompatible with stream copy: "
                + "; ".join(problems),
                stage="concat",
            )
        if config.concat_mode == "reencode":
            warnings.append(
                ValidationIssue(
                    "preparation.concat.reencode_requested",
                    "Final concatenation re-encoded with the configured H.264 profile",
                    "concat_mode",
                )
            )
            return True
        if incompatible:
            warnings.append(
                ValidationIssue(
                    "preparation.concat.reencode_fallback",
                    "Prepared streams differ from the output profile; "
                    "final concatenation was re-encoded: " + "; ".join(problems),
                    "concat_mode",
                )
            )
            return True
        return False

    def _write_concat_list(
        self,
        render_dir: Path,
        prepared: Sequence[PreparedScene],
    ) -> Path:
        path = render_dir / "concat_list.txt"
        if path.exists():
            raise ScenePreparationError(
                "preparation.concat.collision",
                f"Concat list already exists: {path}",
                str(path),
                stage="concat",
            )
        lines = [f"file '{escape_concat_path(scene.prepared_path)}'" for scene in prepared]
        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            raise ScenePreparationError(
                "preparation.concat.list_write_failed",
                f"Could not write concat list {path}: {exc}",
                str(path),
                stage="concat",
            ) from exc
        return path

    @staticmethod
    def _require_nonempty_output(path: Path, label: str) -> None:
        if not path.is_file():
            raise ScenePreparationError(
                "preparation.output.missing",
                f"{label} does not exist: {path}",
                str(path),
                stage="output",
            )
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ScenePreparationError(
                "preparation.output.unreadable",
                f"Could not stat {label} {path}: {exc}",
                str(path),
                stage="output",
            ) from exc
        if size <= 0:
            raise ScenePreparationError(
                "preparation.output.empty",
                f"{label} is empty: {path}",
                str(path),
                stage="output",
            )
