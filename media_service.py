"""Reusable, MCP-independent FFmpeg media operations.

The service owns all ``ffmpeg``/``ffprobe`` subprocess handling and returns
plain, JSON-serializable dictionaries. Business logic never imports FastMCP,
so it can be used from scripts or a different transport later. All blocking
work is synchronous here; MCP adapters are responsible for moving calls off
the event loop.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)

FFMPEG_BIN = os.getenv("MEDIA_FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("MEDIA_FFPROBE_BIN", "ffprobe")
DEFAULT_FFMPEG_TIMEOUT_S = float(os.getenv("MEDIA_FFMPEG_TIMEOUT_S", "180"))
DEFAULT_FFPROBE_TIMEOUT_S = float(os.getenv("MEDIA_FFPROBE_TIMEOUT_S", "30"))
DEFAULT_DURATION_TOLERANCE_S = 0.5
DURATION_EPSILON_S = 0.05
AAC_BITRATE = "192k"
VALIDATION_VIDEO_CRF = "18"
VALIDATION_VIDEO_PRESET = "medium"
DURATION_POLICIES = frozenset({"trim_audio", "extend_video"})


class MediaError(RuntimeError):
    """Base class for expected media processing failures."""


class MediaProbeError(MediaError):
    """Raised when ffprobe cannot inspect a file or returns unusable data."""


class MediaProcessingError(MediaError):
    """Raised when an FFmpeg render or its validation fails."""


def _resolve_path(value: str | Path, label: str) -> Path:
    """Resolve a non-empty path without requiring it to exist."""
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    path = Path(value).expanduser().resolve()
    return path


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} file is empty: {path}")


def _normalize_timeout(timeout_s: float | None, default: float) -> float:
    value = default if timeout_s is None else float(timeout_s)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_s must be a finite, positive number")
    return value


def _normalize_volume(volume: float) -> float:
    value = float(volume)
    if not math.isfinite(value) or value < 0:
        raise ValueError("volume must be a finite, non-negative number")
    return value


def _run_command(command: list[str], *, timeout_s: float, label: str) -> str:
    """Run an FFmpeg-family command with a timeout and actionable errors."""
    LOGGER.debug("Running %s on %s", label, command[-1])
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(
            f"{label} executable not found on PATH: {command[0]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(
            f"{label} timed out after {timeout_s:.0f}s"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-1000:]
        message = f"{label} failed (exit {completed.returncode})"
        if detail:
            message = f"{message}: {detail}"
        raise MediaError(message)
    return completed.stdout


def _parse_duration(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _parse_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _parse_frame_rate(value: object) -> float | None:
    if isinstance(value, str):
        numerator, separator, denominator = value.partition("/")
        try:
            top = float(numerator)
            bottom = float(denominator) if separator else 1.0
        except ValueError:
            return None
        if bottom == 0:
            return None
        rate = top / bottom
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rate = float(value)
    else:
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def _first_duration(*sources: object) -> float | None:
    for source in sources:
        if isinstance(source, dict):
            duration = _parse_duration(source.get("duration"))
            if duration is not None:
                return duration
    return None


def _require_duration(info: dict[str, object], label: str) -> float:
    duration = info.get("duration_s")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or duration <= 0
    ):
        raise MediaError(f"Could not determine {label} duration")
    return float(duration)


def probe_media(
    media_path: str | Path, *, timeout_s: float | None = None
) -> dict[str, object]:
    """Inspect a media file with ffprobe.

    Returns a dictionary with ``status="ok"`` plus duration, resolution, FPS,
    codec, sample rate, channel count and file size. Missing files raise
    ``FileNotFoundError``; unreadable or malformed media raises
    ``MediaProbeError`` so callers can report a structured error.
    """
    path = _resolve_path(media_path, "media_path")
    _require_file(path, "media")
    effective_timeout = _normalize_timeout(
        timeout_s, DEFAULT_FFPROBE_TIMEOUT_S
    )
    try:
        output = _run_command(
            [
                FFPROBE_BIN,
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            timeout_s=effective_timeout,
            label="ffprobe",
        )
    except MediaError as exc:
        raise MediaProbeError(str(exc)) from exc
    try:
        decoded: object = json.loads(output)
    except json.JSONDecodeError as exc:
        raise MediaProbeError("ffprobe returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise MediaProbeError("ffprobe returned an unexpected payload")

    streams = decoded.get("streams")
    if not isinstance(streams, list):
        raise MediaProbeError("ffprobe output has no stream list")
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

    width = _parse_int(video.get("width")) if video is not None else None
    height = _parse_int(video.get("height")) if video is not None else None
    raw_rate = video.get("avg_frame_rate") if video is not None else None
    if not raw_rate and video is not None:
        raw_rate = video.get("r_frame_rate")
    sample_rate = (
        _parse_int(audio.get("sample_rate")) if audio is not None else None
    )
    channels = _parse_int(audio.get("channels")) if audio is not None else None
    size_bytes = _parse_int(container.get("size"))
    if size_bytes is None:
        size_bytes = path.stat().st_size

    return {
        "status": "ok",
        "path": str(path),
        "duration_s": _first_duration(container, video, audio),
        "size_bytes": size_bytes,
        "resolution": (
            f"{width}x{height}" if width is not None and height is not None else None
        ),
        "width": width,
        "height": height,
        "fps": _parse_frame_rate(raw_rate),
        "video_codec": video.get("codec_name") if video is not None else None,
        "audio_codec": audio.get("codec_name") if audio is not None else None,
        "sample_rate": sample_rate,
        "channels": channels,
        "format_name": container.get("format_name"),
    }


def validate_media(
    media_path: str | Path,
    *,
    expected_duration_s: float | None = None,
    duration_tolerance_s: float = DEFAULT_DURATION_TOLERANCE_S,
    timeout_s: float | None = None,
) -> dict[str, object]:
    """Return a structured verdict on a media file's basic streams.

    Unlike :func:`probe_media`, this function does not raise for a missing or
    unreadable file: it returns ``valid=False`` with the reasons in ``errors``.
    """
    path = _resolve_path(media_path, "media_path")
    tolerance = float(duration_tolerance_s)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("duration_tolerance_s must be finite and non-negative")
    expected: float | None = None
    if expected_duration_s is not None:
        expected = float(expected_duration_s)
        if not math.isfinite(expected) or expected < 0:
            raise ValueError("expected_duration_s must be finite and non-negative")

    checks: dict[str, bool] = {}
    errors: list[str] = []
    info: dict[str, object] | None = None
    duration_delta: float | None = None

    if not path.is_file():
        checks["exists"] = False
        errors.append(f"File not found: {path}")
    else:
        checks["exists"] = True
        if path.stat().st_size == 0:
            checks["readable"] = False
            errors.append(f"File is empty: {path}")
        else:
            try:
                info = probe_media(path, timeout_s=timeout_s)
            except MediaError as exc:
                checks["readable"] = False
                errors.append(str(exc))
            else:
                checks["readable"] = True

    if info is not None:
        has_video = info.get("video_codec") is not None
        has_audio = info.get("audio_codec") is not None
        checks["has_video"] = has_video
        checks["has_audio"] = has_audio
        if not has_video:
            errors.append("No video stream found")
        if not has_audio:
            errors.append("No audio stream found")

        resolution_ok = all(
            isinstance(info.get(key), int) and int(str(info[key])) > 0
            for key in ("width", "height")
        )
        checks["resolution"] = resolution_ok
        if not resolution_ok:
            errors.append("Video resolution is missing or invalid")

        fps = info.get("fps")
        fps_ok = isinstance(fps, (int, float)) and not isinstance(fps, bool) and fps > 0
        checks["frame_rate"] = fps_ok
        if not fps_ok:
            errors.append("Video frame rate is missing or invalid")

        sample_rate = info.get("sample_rate")
        channels = info.get("channels")
        audio_metadata_ok = (
            isinstance(sample_rate, int)
            and sample_rate > 0
            and isinstance(channels, int)
            and channels > 0
        )
        checks["audio_metadata"] = audio_metadata_ok
        if not audio_metadata_ok:
            errors.append("Audio sample rate or channel count is missing")

        actual_duration = info.get("duration_s")
        if expected is None:
            duration_ok = isinstance(actual_duration, (int, float)) and actual_duration > 0
        else:
            duration_ok = (
                isinstance(actual_duration, (int, float))
                and abs(float(actual_duration) - expected) <= tolerance
            )
            if isinstance(actual_duration, (int, float)):
                duration_delta = abs(float(actual_duration) - expected)
                if not duration_ok:
                    errors.append(
                        f"Duration {float(actual_duration):.3f}s differs from "
                        f"expected {expected:.3f}s by more than {tolerance:.3f}s"
                    )
        checks["duration"] = duration_ok

    valid = bool(checks) and all(checks.values()) and not errors
    return {
        "status": "ok" if valid else "error",
        "valid": valid,
        "path": str(path),
        "info": info,
        "checks": checks,
        "errors": errors,
        "duration_delta_s": duration_delta,
    }


def _duration_report(
    policy: str, video_duration: float, narration_duration: float
) -> tuple[float, float, float, float, list[str]]:
    """Compute explicit duration deltas and human-readable warnings."""
    trimmed = 0.0
    padded = 0.0
    extended = 0.0
    early = 0.0
    warnings: list[str] = []
    if policy == "trim_audio":
        if narration_duration > video_duration + DURATION_EPSILON_S:
            trimmed = narration_duration - video_duration
            warnings.append(
                f"Narration was trimmed by {trimmed:.3f}s to match the video"
            )
        elif narration_duration < video_duration - DURATION_EPSILON_S:
            padded = video_duration - narration_duration
            warnings.append(
                f"Silence of {padded:.3f}s was added after the narration "
                "to match the video"
            )
    else:
        if narration_duration > video_duration + DURATION_EPSILON_S:
            extended = narration_duration - video_duration
            warnings.append(
                f"The final video frame was frozen for {extended:.3f}s "
                "so the full narration is heard"
            )
        elif narration_duration < video_duration - DURATION_EPSILON_S:
            early = video_duration - narration_duration
            warnings.append(
                f"Narration ends {early:.3f}s before the video; "
                "neither stream was truncated"
            )
    return trimmed, padded, extended, early, warnings


def _build_merge_command(
    video: Path,
    narration: Path,
    temp_output: Path,
    *,
    policy: str,
    volume: float,
    video_duration: float,
    extension_s: float,
) -> list[str]:
    command = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-i",
        str(narration),
    ]
    if policy == "trim_audio":
        graph = (
            f"[1:a]volume={volume:.6f},"
            f"apad=whole_dur={video_duration:.6f},"
            f"atrim=start=0:end={video_duration:.6f},"
            "asetpts=PTS-STARTPTS[a]"
        )
        return command + [
            "-filter_complex",
            graph,
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            AAC_BITRATE,
            "-movflags",
            "+faststart",
            str(temp_output),
        ]

    if extension_s > 0:
        graph = (
            f"[0:v]tpad=stop_mode=clone:stop_duration={extension_s:.6f}[v];"
            f"[1:a]volume={volume:.6f}[a]"
        )
        return command + [
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-crf",
            VALIDATION_VIDEO_CRF,
            "-preset",
            VALIDATION_VIDEO_PRESET,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            AAC_BITRATE,
            "-movflags",
            "+faststart",
            str(temp_output),
        ]

    graph = f"[1:a]volume={volume:.6f}[a]"
    return command + [
        "-filter_complex",
        graph,
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        AAC_BITRATE,
        "-movflags",
        "+faststart",
        str(temp_output),
    ]


def merge_narration(
    video_path: str | Path,
    narration_path: str | Path,
    output_path: str | Path,
    *,
    volume: float = 1.0,
    duration_policy: str = "trim_audio",
    overwrite: bool = False,
    timeout_s: float | None = None,
) -> dict[str, object]:
    """Combine a video and narration WAV into a validated MP4.

    ``duration_policy="trim_audio"`` matches the audio to the video (stream
    copying the video); ``"extend_video"`` freezes the last video frame until
    the narration ends. Every trim, pad or gap is reported in ``warnings`` and
    timing fields, never applied silently. Existing outputs are refused unless
    ``overwrite=True``. FFmpeg renders to a temporary file which is promoted
    only after ffprobe validation succeeds.
    """
    video = _resolve_path(video_path, "video_path")
    narration = _resolve_path(narration_path, "narration_path")
    output = _resolve_path(output_path, "output_path")
    _require_file(video, "video")
    _require_file(narration, "narration")
    if narration.suffix.lower() != ".wav":
        raise ValueError("narration_path must end in .wav")
    if output.suffix.lower() != ".mp4":
        raise ValueError("output_path must end in .mp4")
    if output.is_dir():
        raise ValueError(f"output_path must be a file, not a directory: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output MP4 already exists: {output}. "
            "Pass overwrite=True to explicitly replace it."
        )
    policy = duration_policy.strip().lower()
    if policy not in DURATION_POLICIES:
        raise ValueError(
            f"Unsupported duration_policy '{duration_policy}'. "
            f"Supported policies: {', '.join(sorted(DURATION_POLICIES))}"
        )
    effective_volume = _normalize_volume(volume)
    effective_timeout = _normalize_timeout(
        timeout_s, DEFAULT_FFMPEG_TIMEOUT_S
    )

    video_info = probe_media(video, timeout_s=effective_timeout)
    narration_info = probe_media(narration, timeout_s=effective_timeout)
    if video_info.get("video_codec") is None:
        raise MediaError(f"Video file has no video stream: {video}")
    if narration_info.get("audio_codec") is None:
        raise MediaError(f"Narration file has no audio stream: {narration}")
    video_duration = _require_duration(video_info, "video")
    narration_duration = _require_duration(narration_info, "narration")

    trimmed, padded, extended, early, warnings = _duration_report(
        policy, video_duration, narration_duration
    )
    expected_duration = (
        video_duration if policy == "trim_audio" else max(video_duration, narration_duration)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output.stem}-",
        suffix=".tmp.mp4",
        dir=str(output.parent),
    )
    os.close(descriptor)
    temp_output = Path(temp_name)
    command = _build_merge_command(
        video,
        narration,
        temp_output,
        policy=policy,
        volume=effective_volume,
        video_duration=video_duration,
        extension_s=extended,
    )
    try:
        _run_command(command, timeout_s=effective_timeout, label="ffmpeg")
        validation = validate_media(
            temp_output,
            expected_duration_s=expected_duration,
            timeout_s=effective_timeout,
        )
        if not validation.get("valid"):
            validation_errors = validation.get("errors")
            details = (
                [str(item) for item in validation_errors]
                if isinstance(validation_errors, list)
                else []
            )
            raise MediaProcessingError(
                "Rendered MP4 failed validation: " + "; ".join(details)
            )
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"Output MP4 appeared during rendering: {output}. "
                "No file was replaced."
            )
        temp_output.replace(output)
    finally:
        if temp_output.exists():
            temp_output.unlink(missing_ok=True)

    return {
        "status": "merged",
        "path": str(output),
        "duration_policy": policy,
        "volume": effective_volume,
        "video_duration_s": video_duration,
        "narration_duration_s": narration_duration,
        "output_duration_s": expected_duration,
        "audio_trimmed_s": trimmed,
        "audio_padded_s": padded,
        "video_extended_s": extended,
        "audio_ends_early_s": early,
        "warnings": warnings,
        "validation": validation,
    }
