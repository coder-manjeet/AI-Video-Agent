"""MCP-independent final MP4 assembly for one silent video plus one audio master.

This module muxes the validated silent video from scene preparation with the
WAV master from audio production.  The video timeline is authoritative: audio
is never trimmed, padded or replaced.  When narration runs longer than the
video, the final frame is held with ``tpad``; when it ends early, the resulting
silent tail is recorded and left untouched.

FFmpeg and FFprobe are invoked through argument lists (never shell strings).
All work happens inside one project render directory, existing final MP4s are
never overwritten, and source or intermediate media is never modified.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import media_service
import scene_preparation_service as sps
from manifest_validation import ValidationIssue
from project_paths import (
    PathValidationError,
    resolve_within_project,
    to_project_relative,
    validate_identifier,
)
from project_service import ProjectService
from render_models import coerce_decimal

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 24.0
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_AUDIO_BITRATE_KBPS = 192
DEFAULT_AUDIO_SAMPLE_RATE = 48000
DEFAULT_AUDIO_CHANNELS = 2
DEFAULT_CONTAINER = "mp4"
DEFAULT_PRESET_X264 = "medium"
DEFAULT_CRF = 18
DEFAULT_MOVFLAGS = "+faststart"
DEFAULT_FFMPEG_BIN = "ffmpeg"
DEFAULT_FFPROBE_BIN = "ffprobe"
DEFAULT_FFMPEG_TIMEOUT_S = 180.0
DEFAULT_FFPROBE_TIMEOUT_S = 30.0
DEFAULT_DURATION_TOLERANCE_S = Decimal("0.5")
DEFAULT_OUTPUT_FILENAME = "final.mp4"

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
FPS_TOLERANCE = 0.05
TEMP_PREFIX = ".final-"
TEMP_SUFFIX = ".tmp.mp4"

_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.mp4$", re.IGNORECASE)

ProbeCallable = Callable[..., Mapping[str, object]]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if any(character in value for character in "\r\n\x00"):
        raise ValueError(f"{label} must not contain control characters")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, label)


def _require_int(value: object, label: str, *, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _require_float(value: object, label: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _require_token(value: object, label: str) -> str:
    token = _require_text(value, label)
    if _SAFE_TOKEN_RE.fullmatch(token) is None:
        raise ValueError(f"{label} contains unsupported characters: {token!r}")
    return token


def _require_output_filename(value: object, label: str = "output_filename") -> str:
    name = _require_text(value, label)
    if _SAFE_FILENAME_RE.fullmatch(name) is None:
        raise ValueError(f"{label} must be a safe .mp4 filename: {name!r}")
    if any(part in {"", ".", ".."} for part in Path(name).parts):
        raise ValueError(f"{label} must not contain path segments: {name!r}")
    return name


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _json_safe(value: object) -> object:
    """Convert probe data into strictly JSON-serializable builtin values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _json_safe_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return one mapping as a JSON-safe dictionary."""
    return cast("dict[str, object]", _json_safe(value))


def _probe_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _probe_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _probe_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalRenderConfig:
    """Output profile and runtime settings for one final MP4 render."""

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: float = DEFAULT_FPS
    video_codec: str = DEFAULT_VIDEO_CODEC
    pixel_format: str = DEFAULT_PIXEL_FORMAT
    audio_codec: str = DEFAULT_AUDIO_CODEC
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    audio_channels: int = DEFAULT_AUDIO_CHANNELS
    container: str = DEFAULT_CONTAINER
    preset_x264: str = DEFAULT_PRESET_X264
    crf: int = DEFAULT_CRF
    movflags: str = DEFAULT_MOVFLAGS
    subtitle_burn_in: bool = False
    burn_in_filter: str | None = None
    ffmpeg_bin: str = DEFAULT_FFMPEG_BIN
    ffprobe_bin: str = DEFAULT_FFPROBE_BIN
    ffmpeg_timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S
    ffprobe_timeout_s: float = DEFAULT_FFPROBE_TIMEOUT_S
    duration_tolerance_s: Decimal = DEFAULT_DURATION_TOLERANCE_S
    output_filename: str = DEFAULT_OUTPUT_FILENAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", _require_int(self.width, "width"))
        object.__setattr__(self, "height", _require_int(self.height, "height"))
        if self.width % 2 or self.height % 2:
            raise ValueError("width and height must be even for yuv420p output")
        object.__setattr__(self, "fps", _require_float(self.fps, "fps"))
        for name in ("video_codec", "pixel_format", "audio_codec"):
            value = _require_token(getattr(self, name), name).lower()
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "audio_bitrate_kbps",
            _require_int(self.audio_bitrate_kbps, "audio_bitrate_kbps"),
        )
        object.__setattr__(
            self,
            "audio_sample_rate",
            _require_int(self.audio_sample_rate, "audio_sample_rate"),
        )
        object.__setattr__(
            self,
            "audio_channels",
            _require_int(self.audio_channels, "audio_channels"),
        )
        container = _require_token(self.container, "container").lower()
        if container != "mp4":
            raise ValueError("container must be 'mp4' in this milestone")
        object.__setattr__(self, "container", container)
        preset = _require_token(self.preset_x264, "preset_x264").lower()
        if preset not in SUPPORTED_PRESETS:
            raise ValueError(
                f"unsupported preset_x264 {preset!r}; expected one of {sorted(SUPPORTED_PRESETS)}"
            )
        object.__setattr__(self, "preset_x264", preset)
        object.__setattr__(self, "crf", _require_int(self.crf, "crf", positive=False))
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be in [0, 51]")
        object.__setattr__(self, "movflags", _require_token(self.movflags, "movflags"))
        object.__setattr__(
            self,
            "subtitle_burn_in",
            _require_bool(self.subtitle_burn_in, "subtitle_burn_in"),
        )
        object.__setattr__(
            self,
            "burn_in_filter",
            _optional_text(self.burn_in_filter, "burn_in_filter"),
        )
        object.__setattr__(
            self,
            "ffmpeg_bin",
            _require_text(self.ffmpeg_bin, "ffmpeg_bin"),
        )
        object.__setattr__(
            self,
            "ffprobe_bin",
            _require_text(self.ffprobe_bin, "ffprobe_bin"),
        )
        object.__setattr__(
            self,
            "ffmpeg_timeout_s",
            _require_float(self.ffmpeg_timeout_s, "ffmpeg_timeout_s"),
        )
        object.__setattr__(
            self,
            "ffprobe_timeout_s",
            _require_float(self.ffprobe_timeout_s, "ffprobe_timeout_s"),
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
            "output_filename",
            _require_output_filename(self.output_filename),
        )

    def to_dict(self) -> dict[str, object]:
        return {
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
            "subtitle_burn_in": self.subtitle_burn_in,
            "burn_in_filter": self.burn_in_filter,
            "ffmpeg_bin": self.ffmpeg_bin,
            "ffprobe_bin": self.ffprobe_bin,
            "ffmpeg_timeout_s": self.ffmpeg_timeout_s,
            "ffprobe_timeout_s": self.ffprobe_timeout_s,
            "duration_tolerance_s": _decimal_text(self.duration_tolerance_s),
            "output_filename": self.output_filename,
        }

    @classmethod
    def from_dict(
        cls, data: FinalRenderConfig | Mapping[str, Any] | None = None
    ) -> FinalRenderConfig:
        if isinstance(data, FinalRenderConfig):
            return data
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError("final render configuration must be an object")
        return cls(
            width=data.get("width", DEFAULT_WIDTH),
            height=data.get("height", DEFAULT_HEIGHT),
            fps=data.get("fps", DEFAULT_FPS),
            video_codec=data.get("video_codec", DEFAULT_VIDEO_CODEC),
            pixel_format=data.get("pixel_format", DEFAULT_PIXEL_FORMAT),
            audio_codec=data.get("audio_codec", DEFAULT_AUDIO_CODEC),
            audio_bitrate_kbps=data.get(
                "audio_bitrate_kbps", DEFAULT_AUDIO_BITRATE_KBPS
            ),
            audio_sample_rate=data.get(
                "audio_sample_rate", DEFAULT_AUDIO_SAMPLE_RATE
            ),
            audio_channels=data.get("audio_channels", DEFAULT_AUDIO_CHANNELS),
            container=data.get("container", DEFAULT_CONTAINER),
            preset_x264=data.get("preset_x264", DEFAULT_PRESET_X264),
            crf=data.get("crf", DEFAULT_CRF),
            movflags=data.get("movflags", DEFAULT_MOVFLAGS),
            subtitle_burn_in=data.get("subtitle_burn_in", False),
            burn_in_filter=data.get("burn_in_filter"),
            ffmpeg_bin=data.get("ffmpeg_bin", DEFAULT_FFMPEG_BIN),
            ffprobe_bin=data.get("ffprobe_bin", DEFAULT_FFPROBE_BIN),
            ffmpeg_timeout_s=data.get("ffmpeg_timeout_s", DEFAULT_FFMPEG_TIMEOUT_S),
            ffprobe_timeout_s=data.get("ffprobe_timeout_s", DEFAULT_FFPROBE_TIMEOUT_S),
            duration_tolerance_s=data.get(
                "duration_tolerance_s", DEFAULT_DURATION_TOLERANCE_S
            ),
            output_filename=data.get("output_filename", DEFAULT_OUTPUT_FILENAME),
        )


class FinalRenderError(RuntimeError):
    """Structured failure raised for any unsupported final render outcome."""

    def __init__(
        self,
        code: str,
        message: str,
        location: str = "",
        *,
        stage: str = "merge",
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


@dataclass(frozen=True)
class FinalRenderResult:
    """Validated outcome of one final mux, with project-relative paths."""

    project_id: str
    render_id: str
    render_dir: str
    output_path: str
    size_bytes: int
    duration_s: Decimal
    width: int
    height: int
    fps: float
    video_codec: str
    pixel_format: str | None
    audio_codec: str
    sample_rate: int
    channels: int
    subtitle_path: str | None
    burn_in: bool
    duration_decisions: tuple[str, ...]
    command: tuple[str, ...]
    probe: dict[str, object]
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.project_id, "project_id")
        validate_identifier(self.render_id, "render_id")
        for name in ("render_dir", "output_path"):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "size_bytes",
            _require_int(self.size_bytes, "size_bytes", positive=False),
        )
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        object.__setattr__(
            self,
            "duration_s",
            coerce_decimal(self.duration_s, "duration_s", positive=True),
        )
        for name in ("width", "height"):
            object.__setattr__(self, name, _require_int(getattr(self, name), name))
        object.__setattr__(self, "fps", _require_float(self.fps, "fps"))
        for name in ("video_codec", "audio_codec"):
            object.__setattr__(self, name, _require_token(getattr(self, name), name))
        if self.pixel_format is not None:
            object.__setattr__(
                self,
                "pixel_format",
                _require_token(self.pixel_format, "pixel_format"),
            )
        for name in ("sample_rate", "channels"):
            object.__setattr__(self, name, _require_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "subtitle_path",
            _optional_text(self.subtitle_path, "subtitle_path"),
        )
        object.__setattr__(
            self,
            "burn_in",
            _require_bool(self.burn_in, "burn_in"),
        )
        decisions = tuple(self.duration_decisions)
        for decision in decisions:
            _require_text(decision, "duration_decisions entry")
        object.__setattr__(self, "duration_decisions", decisions)
        command = tuple(str(part) for part in self.command)
        if not command:
            raise ValueError("command must not be empty")
        for part in command:
            _require_text(part, "command argument")
        object.__setattr__(self, "command", command)
        if not isinstance(self.probe, Mapping):
            raise TypeError("probe must be a mapping")
        object.__setattr__(self, "probe", _json_safe_mapping(dict(self.probe)))
        warnings = tuple(self.warnings)
        for issue in warnings:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("warnings must contain ValidationIssue entries")
        object.__setattr__(self, "warnings", warnings)

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "render_id": self.render_id,
            "render_dir": self.render_dir,
            "output_path": self.output_path,
            "size_bytes": self.size_bytes,
            "duration_s": _decimal_text(self.duration_s),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "video_codec": self.video_codec,
            "pixel_format": self.pixel_format,
            "audio_codec": self.audio_codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "subtitle_path": self.subtitle_path,
            "burn_in": self.burn_in,
            "duration_decisions": list(self.duration_decisions),
            "command": list(self.command),
            "probe": _json_safe_mapping(dict(self.probe)),
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def _fail(
    code: str,
    message: str,
    location: str = "",
    *,
    stage: str = "merge",
    stderr_tail: str = "",
) -> FinalRenderError:
    return FinalRenderError(
        code, message, location, stage=stage, stderr_tail=stderr_tail
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class FinalRenderer:
    """Mux a silent video and an audio master into one validated final MP4."""

    def __init__(
        self,
        projects: ProjectService | None = None,
        *,
        command_runner: sps.CommandRunner | None = None,
        probe: Callable[..., Mapping[str, object]] | None = None,
        check_executables: bool | None = None,
    ) -> None:
        self.projects = projects if projects is not None else ProjectService()
        self.command_runner: sps.CommandRunner = (
            command_runner if command_runner is not None else sps.run_ffmpeg
        )
        self._default_probe = probe is None
        self.probe: Callable[..., Mapping[str, object]] = (
            probe
            if probe is not None
            else lambda path, *, timeout_s, ffprobe_bin=DEFAULT_FFPROBE_BIN: sps.probe_video(
                path, ffprobe_bin=ffprobe_bin, timeout_s=timeout_s
            )
        )
        self.check_executables = (
            check_executables
            if check_executables is not None
            else command_runner is None and probe is None
        )

    # -- public API ---------------------------------------------------------

    def render(
        self,
        project_id: str,
        render_dir: str | Path,
        silent_video_path: str | Path,
        audio_master_path: str | Path,
        *,
        render_id: str,
        config: FinalRenderConfig | Mapping[str, Any] | None = None,
        subtitle_path: str | Path | None = None,
        burn_in_filter: str | None = None,
    ) -> FinalRenderResult:
        """Render and verify one final MP4 without touching source media."""
        try:
            settings = FinalRenderConfig.from_dict(config)
        except (TypeError, ValueError) as exc:
            raise _fail("final.config.invalid", str(exc), stage="config") from exc
        try:
            validate_identifier(project_id, "project_id")
            validate_identifier(render_id, "render_id")
        except PathValidationError as exc:
            raise _fail("final.project.invalid_id", str(exc), stage="project") from exc
        self._check_executables(settings)

        root = self.projects.project_dir(project_id)
        target_dir = self._within_project(root, render_dir, "render_dir")
        if not target_dir.is_dir():
            raise _fail(
                "final.render.missing",
                f"Render directory does not exist: {target_dir}",
                str(target_dir),
                stage="render",
            )
        video = self._within_project(root, silent_video_path, "silent_video_path")
        audio = self._within_project(root, audio_master_path, "audio_master_path")
        self._require_source(video, "final.video.missing", "silent video")
        self._require_source(audio, "final.audio.missing", "audio master")

        subtitle: Path | None = None
        if subtitle_path is not None:
            subtitle = self._within_project(root, subtitle_path, "subtitle_path")
            self._require_source(subtitle, "final.subtitle.missing", "subtitle file")

        output = self._within_project(
            root, target_dir / settings.output_filename, "final output path"
        )
        if output.exists():
            raise _fail(
                "final.output.collision",
                f"Final MP4 already exists: {output}",
                str(output),
                stage="output",
            )

        video_info = self._probe(video, settings, label="silent video")
        audio_info = self._probe(audio, settings, label="audio master")
        video_duration = self._duration(video_info, "video", video)
        audio_duration = self._duration(audio_info, "audio", audio)

        effective_filter = (
            burn_in_filter if burn_in_filter is not None else settings.burn_in_filter
        )
        if effective_filter is not None:
            try:
                effective_filter = _require_text(effective_filter, "burn_in_filter")
            except (TypeError, ValueError) as exc:
                raise _fail(
                    "final.subtitle.filter_invalid",
                    str(exc),
                    "burn_in_filter",
                    stage="subtitle",
                ) from exc
        burn_in = False
        warnings: list[ValidationIssue] = []
        video_filters: list[str] = []
        decisions: list[str] = []
        extension = Decimal(0)

        delta = audio_duration - video_duration
        if delta > settings.duration_tolerance_s:
            extension = Decimal(f"{delta:.6f}")
            if extension > 0:
                video_filters.append(
                    f"tpad=stop_mode=clone:stop_duration={extension:.6f}"
                )
            decisions.append(
                f"audio_exceeds_video_by_{_decimal_text(delta)}s;"
                f" final frame held for {_decimal_text(extension)}s"
            )
            warnings.append(
                ValidationIssue(
                    "final.duration.video_extended",
                    "Audio master exceeds the silent video; the final frame was "
                    f"held for {_decimal_text(extension)}s",
                    "audio_master_path",
                )
            )
        elif delta < -settings.duration_tolerance_s:
            shortfall = -delta
            decisions.append(
                f"audio_ends_early_by_{_decimal_text(shortfall)}s;"
                " silent tail retained, no audio padding"
            )
            warnings.append(
                ValidationIssue(
                    "final.duration.silent_tail",
                    f"Audio master ends {_decimal_text(shortfall)}s before the "
                    "video; the silent tail was retained",
                    "audio_master_path",
                )
            )
        else:
            decisions.append(
                f"duration_delta_{_decimal_text(delta)}s"
                " within tolerance; no stream changed"
            )

        if settings.subtitle_burn_in:
            if subtitle is None:
                raise _fail(
                    "final.subtitle.missing",
                    "subtitle_path is required when subtitle_burn_in is enabled",
                    "subtitle_path",
                    stage="subtitle",
                )
            if effective_filter is None:
                raise _fail(
                    "final.subtitle.filter_missing",
                    "A burn-in filter string is required when subtitle_burn_in is enabled",
                    "burn_in_filter",
                    stage="subtitle",
                )
            video_filters.append(effective_filter)
            burn_in = True
            decisions.append("subtitle burn-in enabled")
        elif effective_filter is not None:
            decisions.append("burn_in_filter ignored because subtitle_burn_in is disabled")

        expected_duration = video_duration + extension
        reencode = bool(video_filters)
        temp_output = self._new_temp_file(target_dir)
        command = self._build_command(
            video,
            audio,
            temp_output,
            settings,
            video_filter=",".join(video_filters) if video_filters else None,
            reencode=reencode,
        )
        command_result = self._execute(command, settings)
        if output.exists():
            raise _fail(
                "final.output.collision",
                f"Final MP4 appeared during rendering: {output}. No file was replaced. "
                f"The temporary render is retained at {temp_output}.",
                str(output),
                stage="output",
            )
        self._require_source(temp_output, "final.output.empty", "rendered MP4")
        try:
            temp_output.replace(output)
        except OSError as exc:
            raise _fail(
                "final.output.promote_failed",
                f"Could not promote temporary MP4 {temp_output} to {output}: {exc}",
                str(output),
                stage="output",
            ) from exc

        final_info = self._probe(output, settings, label="final MP4")
        self._verify_output(output, final_info, settings, expected_duration)
        size_bytes = _probe_int(final_info.get("size_bytes"))
        if size_bytes is None or size_bytes <= 0:
            try:
                size_bytes = output.stat().st_size
            except OSError as exc:
                raise _fail(
                    "final.output.unreadable",
                    f"Could not stat final MP4 {output}: {exc}",
                    str(output),
                    stage="output",
                ) from exc

        # Metadata types have already been verified by ``_verify_output``.
        width = cast("int", final_info["width"])
        height = cast("int", final_info["height"])
        fps = cast("float", final_info["fps"])
        video_codec = cast("str", final_info["video_codec"])
        pixel_format = cast("str | None", final_info.get("pixel_format"))
        audio_codec = cast("str", final_info["audio_codec"])
        sample_rate = cast("int", final_info["sample_rate"])
        channels = cast("int", final_info["channels"])
        result = FinalRenderResult(
            project_id=project_id,
            render_id=render_id,
            render_dir=to_project_relative(target_dir, root),
            output_path=to_project_relative(output, root),
            size_bytes=size_bytes,
            duration_s=expected_duration,
            width=width,
            height=height,
            fps=fps,
            video_codec=video_codec,
            pixel_format=pixel_format,
            audio_codec=audio_codec,
            sample_rate=sample_rate,
            channels=channels,
            subtitle_path=(
                to_project_relative(subtitle, root) if subtitle is not None else None
            ),
            burn_in=burn_in,
            duration_decisions=tuple(decisions),
            command=command_result.command,
            probe=dict(final_info),
            warnings=tuple(warnings),
        )
        logger.info(
            "Rendered final MP4 %s for project %s (%s bytes)",
            render_id,
            project_id,
            size_bytes,
        )
        return result

    def render_status(
        self,
        project_id: str,
        render_dir: str | Path,
        silent_video_path: str | Path,
        audio_master_path: str | Path,
        *,
        render_id: str,
        config: FinalRenderConfig | Mapping[str, Any] | None = None,
        subtitle_path: str | Path | None = None,
        burn_in_filter: str | None = None,
    ) -> dict[str, object]:
        """Return a JSON-serializable status, result or structured error."""
        try:
            result = self.render(
                project_id,
                render_dir,
                silent_video_path,
                audio_master_path,
                render_id=render_id,
                config=config,
                subtitle_path=subtitle_path,
                burn_in_filter=burn_in_filter,
            )
        except FinalRenderError as exc:
            return {"status": "error", "error": exc.to_dict()}
        return {"status": "ok", "result": result.to_dict()}

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _within_project(root: Path, value: str | Path, label: str) -> Path:
        try:
            return resolve_within_project(root, value, label)
        except (PathValidationError, TypeError, ValueError, OSError) as exc:
            raise _fail("final.path.unsafe", str(exc), label, stage="path") from exc

    @staticmethod
    def _require_source(path: Path, code: str, label: str) -> None:
        if not path.is_file():
            raise _fail(code, f"{label.capitalize()} is missing: {path}", str(path), stage="input")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise _fail(
                code,
                f"Could not read {label} {path}: {exc}",
                str(path),
                stage="input",
            ) from exc
        if size <= 0:
            raise _fail(code, f"{label.capitalize()} is empty: {path}", str(path), stage="input")

    def _check_executables(self, config: FinalRenderConfig) -> None:
        if not self.check_executables:
            return
        if shutil.which(config.ffmpeg_bin) is None:
            raise _fail(
                "final.ffmpeg.not_installed",
                f"FFmpeg executable not found: {config.ffmpeg_bin}",
                stage="ffmpeg",
            )
        if shutil.which(config.ffprobe_bin) is None:
            raise _fail(
                "final.ffprobe.not_installed",
                f"FFprobe executable not found: {config.ffprobe_bin}",
                stage="ffprobe",
            )

    def _probe(
        self, path: Path, config: FinalRenderConfig, *, label: str
    ) -> Mapping[str, object]:
        options: dict[str, Any] = {"timeout_s": config.ffprobe_timeout_s}
        if self._default_probe:
            options["ffprobe_bin"] = config.ffprobe_bin
        try:
            info = self.probe(path, **options)
        except sps.ScenePreparationError as exc:
            raise _fail(
                "final.probe.invalid",
                f"Could not probe {label} {path}: {exc.message}",
                str(path),
                stage="probe",
            ) from exc
        except media_service.MediaError as exc:
            raise _fail(
                "final.probe.invalid",
                f"Could not probe {label} {path}: {exc}",
                str(path),
                stage="probe",
            ) from exc
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise _fail(
                "final.probe.invalid",
                f"Could not probe {label} {path}: {exc}",
                str(path),
                stage="probe",
            ) from exc
        if not isinstance(info, Mapping):
            raise _fail(
                "final.probe.invalid",
                f"Probe of {label} returned {type(info).__name__} instead of a mapping",
                str(path),
                stage="probe",
            )
        return info

    @staticmethod
    def _duration(info: Mapping[str, object], label: str, path: Path) -> Decimal:
        try:
            return coerce_decimal(info.get("duration_s"), f"{label} duration", positive=True)
        except (TypeError, ValueError) as exc:
            raise _fail(
                "final.probe.invalid",
                f"Could not determine {label} duration: {exc}",
                str(path),
                stage="probe",
            ) from exc

    @staticmethod
    def _new_temp_file(directory: Path) -> Path:
        descriptor, name = tempfile.mkstemp(
            prefix=TEMP_PREFIX, suffix=TEMP_SUFFIX, dir=str(directory)
        )
        os.close(descriptor)
        return Path(name)

    @staticmethod
    def _build_command(
        video: Path,
        audio: Path,
        temp_output: Path,
        config: FinalRenderConfig,
        *,
        video_filter: str | None,
        reencode: bool,
    ) -> list[str]:
        command = [
            config.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
        ]
        if video_filter:
            command.extend(["-vf", video_filter])
        command.extend(["-map", "0:v:0", "-map", "1:a:0"])
        if reencode:
            command.extend(
                [
                    "-c:v",
                    config.video_codec,
                    "-preset",
                    config.preset_x264,
                    "-crf",
                    str(config.crf),
                    "-pix_fmt",
                    config.pixel_format,
                ]
            )
        else:
            command.extend(["-c:v", "copy"])
        command.extend(
            [
                "-c:a",
                config.audio_codec,
                "-b:a",
                f"{config.audio_bitrate_kbps}k",
                "-ar",
                str(config.audio_sample_rate),
                "-ac",
                str(config.audio_channels),
                "-movflags",
                config.movflags,
                "-map_metadata",
                "-1",
                str(temp_output),
            ]
        )
        return command

    def _execute(
        self, command: Sequence[str], config: FinalRenderConfig
    ) -> sps.CommandResult:
        logger.debug("Running final FFmpeg command with %d arguments", len(command))
        code = "final.ffmpeg.failed"
        try:
            result = self.command_runner(list(command), config.ffmpeg_timeout_s)
        except sps.ScenePreparationError as exc:
            lowered = exc.code.lower()
            if "not_installed" in lowered:
                code = "final.ffmpeg.not_installed"
            elif "timeout" in lowered:
                code = "final.ffmpeg.timeout"
            raise _fail(
                code,
                exc.message,
                exc.location,
                stage=exc.stage or "ffmpeg",
                stderr_tail=exc.stderr_tail,
            ) from exc
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise _fail(
                code,
                f"FFmpeg render failed: {exc}",
                stage="ffmpeg",
            ) from exc
        if isinstance(result, sps.CommandResult):
            recorded_command = result.command
            returncode = result.returncode
            stderr_tail = result.stderr_tail
        elif isinstance(result, Mapping):
            raw_command = result.get("command", command)
            recorded_command = tuple(str(part) for part in raw_command)  # type: ignore[union-attr]
            returncode = _probe_int(result.get("returncode")) or 0
            stderr_tail = str(result.get("stderr_tail") or result.get("stderr") or "")
        else:
            raise _fail(
                "final.ffmpeg.invalid_result",
                f"Command runner returned {type(result).__name__} instead of a CommandResult",
                stage="ffmpeg",
            )
        if returncode != 0:
            detail = stderr_tail[-1000:]
            message = f"FFmpeg failed (exit {returncode})"
            if detail:
                message = f"{message}: {detail}"
            raise _fail(
                "final.ffmpeg.failed", message, stage="ffmpeg", stderr_tail=detail
            )
        return sps.CommandResult(recorded_command, returncode, stderr=stderr_tail)

    @staticmethod
    def _verify_output(
        path: Path,
        info: Mapping[str, object],
        config: FinalRenderConfig,
        expected_duration: Decimal,
    ) -> None:
        video_codec = _probe_text(info.get("video_codec"))
        if video_codec is None:
            raise _fail(
                "final.output.missing_video_stream",
                f"Rendered MP4 has no video stream: {path}",
                str(path),
                stage="validate",
            )
        audio_codec = _probe_text(info.get("audio_codec"))
        if audio_codec is None:
            raise _fail(
                "final.output.missing_audio_stream",
                f"Rendered MP4 has no audio stream: {path}",
                str(path),
                stage="validate",
            )

        problems: list[str] = []
        expected_video_codec = sps.expected_probe_codec(config.video_codec)
        if video_codec.lower() != expected_video_codec.lower():
            problems.append(
                f"video codec {video_codec!r} != expected {expected_video_codec!r}"
            )
        if audio_codec.lower() != config.audio_codec.lower():
            problems.append(
                f"audio codec {audio_codec!r} != expected {config.audio_codec!r}"
            )
        width = _probe_int(info.get("width"))
        height = _probe_int(info.get("height"))
        if width != config.width or height != config.height:
            problems.append(
                f"resolution {width}x{height} != expected "
                f"{config.width}x{config.height}"
            )
        fps = _probe_float(info.get("fps"))
        if fps is None or abs(fps - config.fps) > FPS_TOLERANCE:
            problems.append(f"fps {fps!r} differs from expected {config.fps:g}")
        pixel_format = _probe_text(info.get("pixel_format"))
        if pixel_format is None or pixel_format.lower() != config.pixel_format.lower():
            problems.append(
                f"pixel format {pixel_format!r} != expected {config.pixel_format!r}"
            )
        sample_rate = _probe_int(info.get("sample_rate"))
        channels = _probe_int(info.get("channels"))
        if sample_rate != config.audio_sample_rate:
            problems.append(
                f"sample rate {sample_rate!r} != expected {config.audio_sample_rate}"
            )
        if channels != config.audio_channels:
            problems.append(f"channels {channels!r} != expected {config.audio_channels}")
        try:
            duration = coerce_decimal(
                info.get("duration_s"), "output duration", positive=True
            )
        except (TypeError, ValueError) as exc:
            raise _fail(
                "final.probe.invalid",
                f"Rendered MP4 has an invalid duration: {exc}",
                str(path),
                stage="validate",
            ) from exc
        if abs(duration - expected_duration) > config.duration_tolerance_s:
            problems.append(
                f"duration {_decimal_text(duration)}s differs from expected "
                f"{_decimal_text(expected_duration)}s"
            )
        if problems:
            raise _fail(
                "final.output.metadata_mismatch",
                "Rendered MP4 failed verification: " + "; ".join(problems),
                str(path),
                stage="validate",
            )
