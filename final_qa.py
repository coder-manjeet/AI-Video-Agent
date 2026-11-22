"""MCP-independent final QA validation for rendered video.

The validator combines strict metadata checks (fatal errors) with conservative
heuristic detectors (warnings only).  It never mutates media, manifests or any
other project file.  FFmpeg is invoked through explicit argument lists and an
injectable command runner so the module is fully testable without FFmpeg.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from render_models import coerce_decimal
from scene_preparation_service import CommandResult, probe_video, run_ffmpeg

logger = logging.getLogger(__name__)

SEVERITIES = frozenset({"error", "warning"})
HINDI_LANGUAGES = frozenset({"hi", "hinglish"})
DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F

PROBE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
)

CommandRunner = Callable[[Sequence[str], float], CommandResult]
ProbeCallable = Callable[..., Mapping[str, object]]

_TIME_RE = re.compile(
    r"(?:(?P<h>\d{1,3}):)?(?P<m>\d{1,2}):(?P<s>\d{2})[.,](?P<ms>\d{1,3})"
)
_BLACK_RE = re.compile(
    r"black_start:\s*(?P<start>-?\d+(?:\.\d+)?)"
    r".*?black_end:\s*(?P<end>-?\d+(?:\.\d+)?)",
    re.DOTALL,
)
_SILENCE_RE = re.compile(
    r"silence_start:\s*(?P<start>-?\d+(?:\.\d+)?)"
    r".*?silence_end:\s*(?P<end>-?\d+(?:\.\d+)?)",
    re.DOTALL,
)
_ABS_TEXT_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/]|/)[^\s\"'<>|]+"
)


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _coerce_float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _coerce_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _coerce_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_decimal(value: object, *, positive: bool = True) -> Decimal | None:
    try:
        return coerce_decimal(value, "duration", positive=positive)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class QAThresholds:
    """Validated tunables for one final QA pass."""

    duration_tolerance_s: Decimal = Decimal("0.5")
    sync_tolerance_s: Decimal = Decimal("0.5")
    fps_tolerance: float = 0.05
    peak_limit: float = 0.99
    min_silence_s: float = 1.0
    black_min_duration_s: float = 2.0
    black_pixel_threshold: float = 0.98
    black_area_threshold: float = 0.98
    run_detectors: bool = True
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    ffmpeg_timeout_s: float = 180.0

    def __post_init__(self) -> None:
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
            self, "fps_tolerance", _coerce_float(self.fps_tolerance, "fps_tolerance")
        )
        if self.fps_tolerance < 0:
            raise ValueError("fps_tolerance must be non-negative")
        peak_limit = _coerce_float(self.peak_limit, "peak_limit", positive=True)
        if peak_limit > 1:
            raise ValueError("peak_limit must be in (0, 1]")
        object.__setattr__(self, "peak_limit", peak_limit)
        min_silence = _coerce_float(self.min_silence_s, "min_silence_s")
        if min_silence < 0:
            raise ValueError("min_silence_s must be non-negative")
        object.__setattr__(self, "min_silence_s", min_silence)
        black_min = _coerce_float(self.black_min_duration_s, "black_min_duration_s")
        if black_min < 0:
            raise ValueError("black_min_duration_s must be non-negative")
        object.__setattr__(self, "black_min_duration_s", black_min)
        for name in ("black_pixel_threshold", "black_area_threshold"):
            value = _coerce_float(getattr(self, name), name, positive=True)
            if value > 1:
                raise ValueError(f"{name} must be in (0, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "run_detectors", _coerce_bool(self.run_detectors, "run_detectors")
        )
        object.__setattr__(self, "ffmpeg_bin", _coerce_text(self.ffmpeg_bin, "ffmpeg_bin"))
        object.__setattr__(
            self, "ffprobe_bin", _coerce_text(self.ffprobe_bin, "ffprobe_bin")
        )
        object.__setattr__(
            self,
            "ffmpeg_timeout_s",
            _coerce_float(self.ffmpeg_timeout_s, "ffmpeg_timeout_s", positive=True),
        )

    @classmethod
    def from_dict(
        cls, data: QAThresholds | Mapping[str, Any] | None = None
    ) -> QAThresholds:
        """Build thresholds from a mapping, rejecting invalid ranges."""
        if data is None:
            return cls()
        if isinstance(data, QAThresholds):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("QA thresholds must be an object")
        defaults = cls()
        return cls(
            duration_tolerance_s=data.get(
                "duration_tolerance_s", defaults.duration_tolerance_s
            ),
            sync_tolerance_s=data.get("sync_tolerance_s", defaults.sync_tolerance_s),
            fps_tolerance=data.get("fps_tolerance", defaults.fps_tolerance),
            peak_limit=data.get("peak_limit", defaults.peak_limit),
            min_silence_s=data.get("min_silence_s", defaults.min_silence_s),
            black_min_duration_s=data.get(
                "black_min_duration_s", defaults.black_min_duration_s
            ),
            black_pixel_threshold=data.get(
                "black_pixel_threshold", defaults.black_pixel_threshold
            ),
            black_area_threshold=data.get(
                "black_area_threshold", defaults.black_area_threshold
            ),
            run_detectors=data.get("run_detectors", defaults.run_detectors),
            ffmpeg_bin=data.get("ffmpeg_bin", defaults.ffmpeg_bin),
            ffprobe_bin=data.get("ffprobe_bin", defaults.ffprobe_bin),
            ffmpeg_timeout_s=data.get("ffmpeg_timeout_s", defaults.ffmpeg_timeout_s),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot (Decimals become strings)."""
        return {
            "duration_tolerance_s": _decimal_text(self.duration_tolerance_s),
            "sync_tolerance_s": _decimal_text(self.sync_tolerance_s),
            "fps_tolerance": self.fps_tolerance,
            "peak_limit": self.peak_limit,
            "min_silence_s": self.min_silence_s,
            "black_min_duration_s": self.black_min_duration_s,
            "black_pixel_threshold": self.black_pixel_threshold,
            "black_area_threshold": self.black_area_threshold,
            "run_detectors": self.run_detectors,
            "ffmpeg_bin": self.ffmpeg_bin,
            "ffprobe_bin": self.ffprobe_bin,
            "ffmpeg_timeout_s": self.ffmpeg_timeout_s,
        }


@dataclass(frozen=True)
class QAIssue:
    """One QA finding with an honest severity and stable code."""

    severity: str
    code: str
    message: str
    location: str = ""
    timestamp_s: Decimal | None = None
    scene_id: str | None = None
    recommendation: str = ""

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(SEVERITIES)}: {self.severity!r}"
            )
        _coerce_text(self.code, "code")
        _coerce_text(self.message, "message")
        if not isinstance(self.location, str):
            raise TypeError("location must be a string")
        if not isinstance(self.recommendation, str):
            raise TypeError("recommendation must be a string")
        if self.scene_id is not None and not isinstance(self.scene_id, str):
            raise TypeError("scene_id must be a string or None")
        if self.timestamp_s is not None:
            object.__setattr__(
                self,
                "timestamp_s",
                coerce_decimal(self.timestamp_s, "timestamp_s"),
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe issue; Decimal timestamps become strings."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "timestamp_s": (
                _decimal_text(self.timestamp_s) if self.timestamp_s is not None else None
            ),
            "scene_id": self.scene_id,
            "recommendation": self.recommendation,
        }


def _json_value(value: object) -> object:
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class QAReport:
    """Structured QA verdict; only ``error`` issues make a render invalid."""

    issues: tuple[QAIssue, ...] = ()
    checks: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        for issue in self.issues:
            if not isinstance(issue, QAIssue):
                raise TypeError("issues must contain QAIssue entries")
        object.__setattr__(
            self, "checks", {str(key): _json_value(value) for key, value in self.checks.items()}
        )
        object.__setattr__(
            self,
            "metrics",
            {str(key): _json_value(value) for key, value in self.metrics.items()},
        )

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> tuple[QAIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[QAIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, object]:
        """Return the machine-readable verdict for a production report."""
        return {
            "valid": self.valid,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
        }


class DetectorError(RuntimeError):
    """A detector could not be executed or its output could not be parsed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def build_black_command(
    video_path: str | Path, thresholds: QAThresholds
) -> list[str]:
    """Build an explicit FFmpeg blackdetect argument list (no shell)."""
    filter_value = (
        f"blackdetect=d={thresholds.black_min_duration_s:.3f}"
        f":pix_th={thresholds.black_pixel_threshold:.3f}"
        f":pic_th={thresholds.black_area_threshold:.3f}"
    )
    return [
        thresholds.ffmpeg_bin,
        "-hide_banner",
        "-nostats",
        "-i",
        str(video_path),
        "-vf",
        filter_value,
        "-an",
        "-f",
        "null",
        "-",
    ]


def build_silence_command(
    video_path: str | Path, thresholds: QAThresholds
) -> list[str]:
    """Build an explicit FFmpeg silencedetect argument list (no shell)."""
    filter_value = f"silencedetect=noise=-60dB:d={thresholds.min_silence_s:.3f}"
    return [
        thresholds.ffmpeg_bin,
        "-hide_banner",
        "-nostats",
        "-i",
        str(video_path),
        "-af",
        filter_value,
        "-vn",
        "-f",
        "null",
        "-",
    ]


def _run_detector(
    command: Sequence[str],
    thresholds: QAThresholds,
    command_runner: CommandRunner | None,
    *,
    label: str,
) -> CommandResult:
    runner = command_runner if command_runner is not None else run_ffmpeg
    try:
        result = runner(list(command), thresholds.ffmpeg_timeout_s)
    except Exception as exc:  # detector failures are never fatal
        raise DetectorError(
            f"qa.detector.{label}.unavailable", f"{label} could not run: {exc}"
        ) from exc
    if not isinstance(result, CommandResult):
        raise DetectorError(
            f"qa.detector.{label}.unavailable",
            f"{label} returned {type(result).__name__} instead of CommandResult",
        )
    if result.returncode != 0:
        detail = result.stderr_tail.strip()
        suffix = f": {detail}" if detail else ""
        raise DetectorError(
            f"qa.detector.{label}.failed",
            f"{label} exited with {result.returncode}{suffix}",
        )
    return result


def _intervals(
    stderr: str, pattern: re.Pattern[str], *, min_duration_s: float
) -> list[dict[str, float]]:
    intervals: list[dict[str, float]] = []
    for match in pattern.finditer(stderr):
        start = float(match.group("start"))
        end = float(match.group("end"))
        duration = end - start
        if end <= start or duration < min_duration_s:
            continue
        intervals.append(
            {"start_s": start, "end_s": end, "duration_s": duration}
        )
    return intervals


def detect_black(
    video_path: str | Path,
    *,
    thresholds: QAThresholds | Mapping[str, object] | None = None,
    command_runner: CommandRunner | None = None,
) -> list[dict[str, float]]:
    """Return blackdetect intervals as ``start_s``/``end_s``/``duration_s`` dicts."""
    settings = QAThresholds.from_dict(thresholds)
    result = _run_detector(
        build_black_command(video_path, settings),
        settings,
        command_runner,
        label="blackdetect",
    )
    return _intervals(
        result.stderr_tail,
        _BLACK_RE,
        min_duration_s=settings.black_min_duration_s,
    )


def detect_silence(
    video_path: str | Path,
    *,
    thresholds: QAThresholds | Mapping[str, object] | None = None,
    command_runner: CommandRunner | None = None,
) -> list[dict[str, float]]:
    """Return silencedetect intervals as ``start_s``/``end_s``/``duration_s`` dicts."""
    settings = QAThresholds.from_dict(thresholds)
    result = _run_detector(
        build_silence_command(video_path, settings),
        settings,
        command_runner,
        label="silencedetect",
    )
    return _intervals(
        result.stderr_tail,
        _SILENCE_RE,
        min_duration_s=settings.min_silence_s,
    )


@dataclass(frozen=True)
class _Cue:
    start_s: float
    end_s: float
    text: str


def _cue_seconds(match: re.Match[str]) -> float:
    if match.group("h") is not None:
        hours = int(match.group("h"))
        minutes = int(match.group("m"))
        seconds = int(match.group("s"))
    else:
        hours = 0
        minutes = int(match.group("m"))
        seconds = int(match.group("s"))
    fraction = match.group("ms") or "0"
    return hours * 3600 + minutes * 60 + seconds + int(fraction) / (10 ** len(fraction))


def _parse_cues(text: str) -> tuple[list[_Cue], list[str]]:
    """Parse simple SRT/VTT timestamp lines; return cues and error codes."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[_Cue] = []
    errors: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if "-->" not in line:
            index += 1
            continue
        times = list(_TIME_RE.finditer(line))
        if len(times) < 2:
            errors.append("qa.subtitle.invalid_timestamp")
            index += 1
            continue
        start = _cue_seconds(times[0])
        end = _cue_seconds(times[1])
        index += 1
        cue_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            cue_lines.append(lines[index].strip())
            index += 1
        cues.append(_Cue(start_s=start, end_s=end, text=" ".join(cue_lines)))
    return cues, errors


def _get(obj: object, key: str) -> object:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _scene_items(timeline: object) -> list[object]:
    raw = _get(timeline, "scenes")
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return []


def _first(obj: object, *keys: str) -> object:
    for key in keys:
        value = _get(obj, key)
        if value is not None:
            return value
    return None


def _scrub_text(text: str) -> str:
    """Replace absolute paths inside free text with their basenames."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(0).rstrip(".,;:")
        suffix = match.group(0)[len(token) :]
        return Path(token).name + suffix

    return _ABS_TEXT_RE.sub(replace, text)


class FinalQAValidator:
    """Validate a final render with strict metadata checks and soft detectors."""

    def __init__(
        self,
        probe: ProbeCallable | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.probe: ProbeCallable = probe if probe is not None else probe_video
        self.command_runner: CommandRunner | None = command_runner

    def detect_black(
        self,
        video_path: str | Path,
        *,
        thresholds: QAThresholds | Mapping[str, object] | None = None,
    ) -> list[dict[str, float]]:
        """Delegate to :func:`detect_black` with this validator's runner."""
        return detect_black(
            video_path, thresholds=thresholds, command_runner=self.command_runner
        )

    def detect_silence(
        self,
        video_path: str | Path,
        *,
        thresholds: QAThresholds | Mapping[str, object] | None = None,
    ) -> list[dict[str, float]]:
        """Delegate to :func:`detect_silence` with this validator's runner."""
        return detect_silence(
            video_path, thresholds=thresholds, command_runner=self.command_runner
        )

    def validate(
        self,
        final_video_path: str | Path,
        *,
        expected: Mapping[str, object],
        timeline: object | None = None,
        subtitle_paths: Sequence[str | Path] = (),
        burn_in: bool = False,
        thresholds: QAThresholds | Mapping[str, object] | None = None,
        probe_info: Mapping[str, object] | None = None,
        peak: float | None = None,
    ) -> QAReport:
        """Return a complete QA report for one final video."""
        settings = QAThresholds.from_dict(thresholds)
        if not isinstance(expected, Mapping):
            raise TypeError("expected must be a mapping")
        issues: list[QAIssue] = []
        checks: dict[str, object] = {
            "file_exists": False,
            "file_nonempty": False,
            "probe_ok": False,
            "video_stream": False,
            "audio_stream": False,
            "metadata_matches": False,
            "duration_matches": False,
            "av_sync": False,
            "peak_checked": False,
            "narration_complete": False,
            "timeline_checked": timeline is not None,
            "subtitles_checked": bool(subtitle_paths),
            "burn_in_verified": False,
            "detectors_run": False,
        }
        metrics: dict[str, object] = {}
        raw_path = Path(final_video_path).expanduser()
        location = raw_path.name

        def add(
            severity: str,
            code: str,
            message: str,
            *,
            timestamp_s: Decimal | None = None,
            scene_id: str | None = None,
            recommendation: str = "",
        ) -> None:
            issues.append(
                QAIssue(
                    severity=severity,
                    code=code,
                    message=_scrub_text(message),
                    location=location,
                    timestamp_s=timestamp_s,
                    scene_id=scene_id,
                    recommendation=recommendation,
                )
            )

        try:
            target = raw_path.resolve()
        except OSError as exc:
            add("error", "qa.file.unreadable", f"Could not resolve video path: {exc}")
            return QAReport(tuple(issues), checks, metrics)
        if not target.is_file():
            add("error", "qa.file.missing", "Final video does not exist")
            return QAReport(tuple(issues), checks, metrics)
        try:
            size = target.stat().st_size
        except OSError as exc:
            add("error", "qa.file.unreadable", f"Could not stat final video: {exc}")
            return QAReport(tuple(issues), checks, metrics)
        metrics["size_bytes"] = size
        if size == 0:
            add("error", "qa.file.empty", "Final video is empty")
            return QAReport(tuple(issues), checks, metrics)
        checks["file_exists"] = True
        checks["file_nonempty"] = True

        if probe_info is not None:
            if not isinstance(probe_info, Mapping):
                add("error", "qa.probe.invalid", "probe_info must be a mapping")
                return QAReport(tuple(issues), checks, metrics)
            info: Mapping[str, object] = probe_info
        else:
            try:
                info = self._call_probe(target, settings)
            except PROBE_ERRORS as exc:
                add("error", "qa.probe.failed", f"Could not probe final video: {exc}")
                return QAReport(tuple(issues), checks, metrics)
        status = info.get("status")
        if status is not None and status != "ok":
            detail = info.get("error")
            suffix = f": {detail}" if isinstance(detail, str) and detail else ""
            add("error", "qa.probe.failed", f"Probe reported status {status!r}{suffix}")
            return QAReport(tuple(issues), checks, metrics)
        checks["probe_ok"] = True

        self._check_media(info, expected, settings, add, checks, metrics)
        video_duration = self._video_duration(info, expected)
        audio_duration = self._audio_duration(info, expected)
        self._check_sync(video_duration, audio_duration, settings, add, checks, metrics)
        self._check_peak(info, expected, peak, settings, add, checks, metrics)
        self._check_timeline(
            timeline, audio_duration, settings, add, checks, metrics
        )
        self._check_subtitles(
            subtitle_paths, burn_in, expected, video_duration, settings, add, checks, metrics
        )
        self._check_detectors(
            target, settings, timeline, expected, add, checks, metrics
        )

        logger.debug(
            "Final QA for %s: valid=%s errors=%d warnings=%d",
            location,
            not issues or all(i.severity != "error" for i in issues),
            sum(i.severity == "error" for i in issues),
            sum(i.severity == "warning" for i in issues),
        )
        return QAReport(tuple(issues), checks, metrics)

    def _call_probe(self, target: Path, settings: QAThresholds) -> Mapping[str, object]:
        """Support probes with or without an explicit ``timeout_s`` keyword."""
        try:
            result = self.probe(target, timeout_s=settings.ffmpeg_timeout_s)
        except TypeError as exc:
            if "timeout_s" not in str(exc):
                raise
            result = self.probe(target)
        if not isinstance(result, Mapping):
            raise TypeError("probe returned a non-mapping result")
        return result

    @staticmethod
    def _video_duration(
        info: Mapping[str, object], expected: Mapping[str, object]
    ) -> Decimal | None:
        for key in ("duration_s", "video_duration_s"):
            value = _optional_decimal(info.get(key))
            if value is not None:
                return value
        return _optional_decimal(expected.get("duration_s"))

    @staticmethod
    def _audio_duration(
        info: Mapping[str, object], expected: Mapping[str, object]
    ) -> Decimal | None:
        for source in (info, expected):
            for key in ("audio_duration_s", "master_duration_s"):
                value = _optional_decimal(source.get(key))
                if value is not None:
                    return value
            nested = source.get("audio")
            if isinstance(nested, Mapping):
                for key in ("duration_s", "audio_duration_s"):
                    value = _optional_decimal(nested.get(key))
                    if value is not None:
                        return value
        return None

    def _check_media(
        self,
        info: Mapping[str, object],
        expected: Mapping[str, object],
        settings: QAThresholds,
        add: Callable[..., None],
        checks: dict[str, object],
        metrics: dict[str, object],
    ) -> None:
        duration = self._video_duration(info, expected)
        metrics["duration_s"] = _decimal_text(duration) if duration else None
        video_codec = info.get("video_codec")
        if not isinstance(video_codec, str) or not video_codec.strip():
            add("error", "qa.video.missing_stream", "Final video has no video stream")
        else:
            checks["video_stream"] = True
            metrics["video_codec"] = video_codec
        audio_codec = info.get("audio_codec")
        if not isinstance(audio_codec, str) or not audio_codec.strip():
            add("error", "qa.audio.missing_stream", "Final video has no audio stream")
        else:
            checks["audio_stream"] = True
            metrics["audio_codec"] = audio_codec

        if duration is None:
            add("error", "qa.video.invalid_duration", "Video duration is missing or invalid")
            checks["duration_matches"] = False
        else:
            wanted_duration = _optional_decimal(expected.get("duration_s"))
            metrics["expected_duration_s"] = (
                _decimal_text(wanted_duration) if wanted_duration else None
            )
            if wanted_duration is None or abs(duration - wanted_duration) <= settings.duration_tolerance_s:
                checks["duration_matches"] = True
            else:
                delta = abs(duration - wanted_duration)
                add(
                    "error",
                    "qa.video.duration_mismatch",
                    f"Duration {_decimal_text(duration)}s differs from expected "
                    f"{_decimal_text(wanted_duration)}s by {_decimal_text(delta)}s",
                    recommendation="Re-render or re-mux the final video with the expected duration.",
                )
                metrics["duration_delta_s"] = _decimal_text(delta)

        metadata_ok = True
        wanted_video_codec = expected.get("video_codec")
        if (
            isinstance(wanted_video_codec, str)
            and wanted_video_codec.strip()
            and (
                not isinstance(video_codec, str)
                or video_codec.strip().lower() != wanted_video_codec.strip().lower()
            )
        ):
            metadata_ok = False
            add(
                "error",
                "qa.video.codec_mismatch",
                f"Video codec is {video_codec!r}, expected {wanted_video_codec!r}",
            )
        wanted_audio_codec = expected.get("audio_codec")
        if (
            isinstance(wanted_audio_codec, str)
            and wanted_audio_codec.strip()
            and (
                not isinstance(audio_codec, str)
                or audio_codec.strip().lower() != wanted_audio_codec.strip().lower()
            )
        ):
            metadata_ok = False
            add(
                "error",
                "qa.audio.codec_mismatch",
                f"Audio codec is {audio_codec!r}, expected {wanted_audio_codec!r}",
            )
        for key, label in (("width", "width"), ("height", "height")):
            wanted = expected.get(key)
            if isinstance(wanted, int) and not isinstance(wanted, bool) and wanted > 0:
                actual = info.get(key)
                if actual != wanted:
                    metadata_ok = False
                    add(
                        "error",
                        "qa.video.resolution_mismatch",
                        f"Video {label} is {actual!r}, expected {wanted}",
                    )
        metrics["resolution"] = f"{info.get('width')}x{info.get('height')}"
        wanted_fps = _optional_float(expected.get("fps"))
        actual_fps = _optional_float(info.get("fps"))
        metrics["fps"] = actual_fps
        if wanted_fps is not None and (
            actual_fps is None or abs(actual_fps - wanted_fps) > settings.fps_tolerance
        ):
            metadata_ok = False
            add(
                "error",
                "qa.video.fps_mismatch",
                f"FPS is {actual_fps!r}, expected {wanted_fps:g} "
                f"(tolerance {settings.fps_tolerance:g})",
            )
        wanted_pixel = expected.get("pixel_format")
        if isinstance(wanted_pixel, str) and wanted_pixel.strip():
            actual_pixel = info.get("pixel_format")
            metrics["pixel_format"] = actual_pixel
            if not isinstance(actual_pixel, str) or (
                actual_pixel.strip().lower() != wanted_pixel.strip().lower()
            ):
                metadata_ok = False
                add(
                    "error",
                    "qa.video.pixel_format_mismatch",
                    f"Pixel format is {actual_pixel!r}, expected {wanted_pixel!r}",
                )
        for key, code, label in (
            ("sample_rate", "qa.audio.sample_rate_mismatch", "sample rate"),
            ("channels", "qa.audio.channels_mismatch", "channel count"),
        ):
            wanted = expected.get(key)
            if isinstance(wanted, int) and not isinstance(wanted, bool) and wanted > 0:
                actual = info.get(key)
                metrics[key] = actual
                if actual != wanted:
                    metadata_ok = False
                    add(
                        "error",
                        code,
                        f"Audio {label} is {actual!r}, expected {wanted}",
                    )
        checks["metadata_matches"] = metadata_ok

    @staticmethod
    def _check_sync(
        video_duration: Decimal | None,
        audio_duration: Decimal | None,
        settings: QAThresholds,
        add: Callable[..., None],
        checks: dict[str, object],
        metrics: dict[str, object],
    ) -> None:
        if video_duration is None or audio_duration is None:
            metrics["sync_delta_s"] = None
            return
        delta = abs(audio_duration - video_duration)
        metrics["audio_duration_s"] = _decimal_text(audio_duration)
        metrics["sync_delta_s"] = _decimal_text(delta)
        if delta <= settings.sync_tolerance_s:
            checks["av_sync"] = True
        else:
            add(
                "error",
                "qa.av.sync_mismatch",
                f"Audio duration {_decimal_text(audio_duration)}s differs from "
                f"video duration {_decimal_text(video_duration)}s by "
                f"{_decimal_text(delta)}s",
                recommendation="Align the audio master with the video timeline before delivery.",
            )

    @staticmethod
    def _check_peak(
        info: Mapping[str, object],
        expected: Mapping[str, object],
        peak: float | None,
        settings: QAThresholds,
        add: Callable[..., None],
        checks: dict[str, object],
        metrics: dict[str, object],
    ) -> None:
        candidates = (peak, expected.get("peak"), info.get("peak"))
        measured = next((value for value in candidates if value is not None), None)
        level = _optional_float(measured)
        metrics["peak"] = level
        if level is None:
            return
        checks["peak_checked"] = True
        if level > settings.peak_limit:
            severity = "error" if level > 1.0 else "warning"
            add(
                severity,
                "qa.audio.clipping",
                f"Measured peak {level:.4f} exceeds the configured limit "
                f"{settings.peak_limit:.4f}",
                recommendation="Re-mix with more headroom before publishing.",
            )

    @staticmethod
    def _check_timeline(
        timeline: object | None,
        audio_duration: Decimal | None,
        settings: QAThresholds,
        add: Callable[..., None],
        checks: dict[str, object],
        metrics: dict[str, object],
    ) -> None:
        if timeline is None:
            metrics["timeline_total_duration_s"] = None
            return
        scenes = _scene_items(timeline)
        metrics["timeline_scene_count"] = len(scenes)
        holds: dict[str, object] = {}
        silent_tails: list[dict[str, object]] = []
        total_hold = Decimal(0)
        cursor: Decimal | None = None
        order_ok = True
        continuity_ok = True

        for index, scene in enumerate(scenes):
            scene_id_value = _first(scene, "scene_id", "id")
            scene_id = scene_id_value if isinstance(scene_id_value, str) else f"scenes[{index}]"
            order_value = _first(scene, "order", "scene_number")
            expected_order = index if isinstance(order_value, int) else index + 1
            if isinstance(order_value, int) and not isinstance(order_value, bool):
                if order_value != expected_order:
                    order_ok = False
                    add(
                        "error",
                        "qa.timeline.scene_order",
                        f"Scene at index {index} has order {order_value}, expected {expected_order}",
                        timestamp_s=cursor,
                        scene_id=scene_id,
                    )
            elif order_value is not None:
                order_ok = False
                add(
                    "error",
                    "qa.timeline.scene_order",
                    f"Scene at index {index} has a non-integer order {order_value!r}",
                    scene_id=scene_id,
                )
            start = _optional_decimal(_first(scene, "video_start_s", "timeline_start_s", "start_s"))
            end = _optional_decimal(_first(scene, "video_end_s", "timeline_end_s", "end_s"))
            if start is not None and cursor is not None:
                gap = start - cursor
                if abs(gap) > settings.duration_tolerance_s:
                    continuity_ok = False
                    code = "qa.timeline.overlap" if gap < 0 else "qa.timeline.gap"
                    add(
                        "error",
                        code,
                        f"Scene {scene_id} starts at {_decimal_text(start)}s after "
                        f"previous end {_decimal_text(cursor)}s "
                        f"(delta {_decimal_text(gap)}s)",
                        timestamp_s=start,
                        scene_id=scene_id,
                    )
            elif start is not None and cursor is None and start > settings.duration_tolerance_s:
                continuity_ok = False
                add(
                    "error",
                    "qa.timeline.gap",
                    f"First scene starts at {_decimal_text(start)}s instead of 0s",
                    timestamp_s=start,
                    scene_id=scene_id,
                )
            if end is not None:
                cursor = end

            hold = _optional_decimal(
                _first(scene, "final_frame_hold_s", "video_hold_duration_s", "hold_duration_s"),
                positive=False,
            )
            if hold is not None and hold > 0:
                holds[scene_id] = hold
                total_hold += hold

            tail = _optional_decimal(
                _first(scene, "silent_tail_s", "silent_tail_duration_s"),
                positive=False,
            )
            narration_present = _get(scene, "narration_present")
            narration_end = _optional_decimal(_get(scene, "narration_end_s"))
            if tail is not None and tail > 0:
                tail_start = narration_end if narration_end is not None else (
                    (end - tail) if end is not None else None
                )
                silent_tails.append(
                    {
                        "scene_id": scene_id,
                        "start_s": _decimal_text(tail_start) if tail_start else None,
                        "end_s": _decimal_text(end) if end else None,
                        "duration_s": _decimal_text(tail),
                        "kind": "silent_tail",
                    }
                )
            elif narration_present is False and start is not None and end is not None:
                silent_tails.append(
                    {
                        "scene_id": scene_id,
                        "start_s": _decimal_text(start),
                        "end_s": _decimal_text(end),
                        "duration_s": _decimal_text(end - start),
                        "kind": "no_narration",
                    }
                )

        checks["timeline_scene_order"] = order_ok
        checks["timeline_continuity"] = continuity_ok
        checks["final_frame_holds"] = bool(holds)
        metrics["final_frame_hold_s"] = _decimal_text(total_hold)
        metrics["final_frame_holds_by_scene"] = holds
        metrics["intentional_silent_tails"] = silent_tails
        checks["intentional_silence_reported"] = True
        if silent_tails:
            add(
                "warning",
                "qa.audio.intentional_silent_tail",
                f"Timeline records {len(silent_tails)} intentional silent interval(s); "
                "these are not narration truncation",
                recommendation="Confirm the silent tails are intentional before delivery.",
            )

        timeline_total = _optional_decimal(_get(timeline, "total_duration_s"))
        metrics["timeline_total_duration_s"] = (
            _decimal_text(timeline_total) if timeline_total else None
        )
        if timeline_total is not None:
            metrics["timeline_audio_delta_s"] = (
                _decimal_text(abs(audio_duration - timeline_total))
                if audio_duration is not None
                else None
            )
            if audio_duration is not None and (
                audio_duration + settings.duration_tolerance_s < timeline_total
            ):
                add(
                    "error",
                    "qa.audio.narration_truncated",
                    f"Audio duration {_decimal_text(audio_duration)}s is shorter than "
                    f"timeline total {_decimal_text(timeline_total)}s",
                    recommendation="Regenerate the audio master without truncating narration.",
                )
                checks["narration_complete"] = False
            else:
                checks["narration_complete"] = audio_duration is not None
        else:
            checks["narration_complete"] = False

    @staticmethod
    def _check_subtitles(
        subtitle_paths: Sequence[str | Path],
        burn_in: bool,
        expected: Mapping[str, object],
        video_duration: Decimal | None,
        settings: QAThresholds,
        add: Callable[..., None],
        checks: dict[str, object],
        metrics: dict[str, object],
    ) -> None:
        details: list[dict[str, object]] = []
        subtitles_ok = True
        burn_in_verified = bool(expected.get("burn_in_verified") is True)
        cues_all: list[_Cue] = []
        for index, raw_path in enumerate(subtitle_paths):
            path = Path(raw_path).expanduser()
            label = path.name
            entry: dict[str, object] = {"path": label, "exists": False}
            if not path.is_file():
                subtitles_ok = False
                entry["valid"] = False
                add("error", "qa.subtitle.missing", f"Subtitle file is missing: {label}")
                details.append(entry)
                continue
            try:
                size = path.stat().st_size
                text = path.read_text(encoding="utf-8-sig", errors="replace")
            except (OSError, UnicodeError) as exc:
                subtitles_ok = False
                entry["valid"] = False
                add(
                    "error",
                    "qa.subtitle.unreadable",
                    f"Could not read subtitle file {label}: {exc}",
                )
                details.append(entry)
                continue
            entry["exists"] = True
            entry["size_bytes"] = size
            if size == 0:
                subtitles_ok = False
                entry["valid"] = False
                add("error", "qa.subtitle.empty", f"Subtitle file is empty: {label}")
                details.append(entry)
                continue
            is_vtt = path.suffix.lower() == ".vtt" or text.lstrip().startswith("WEBVTT")
            entry["format"] = "vtt" if is_vtt else "srt"
            if is_vtt and not text.lstrip().startswith("WEBVTT"):
                subtitles_ok = False
                entry["valid"] = False
                add(
                    "error",
                    "qa.subtitle.invalid_vtt_header",
                    f"VTT subtitle file does not start with WEBVTT: {label}",
                )
            cues, parse_errors = _parse_cues(text)
            entry["cue_count"] = len(cues)
            cues_all.extend(cues)
            if not cues:
                subtitles_ok = False
                entry["valid"] = False
                add("error", "qa.subtitle.no_cues", f"No subtitle cues found in {label}")
            for code in sorted(set(parse_errors)):
                subtitles_ok = False
                entry["valid"] = False
                add("error", code, f"Unparseable subtitle timestamp in {label}")
            previous_start = -1.0
            for cue in cues:
                if cue.end_s <= cue.start_s:
                    subtitles_ok = False
                    entry["valid"] = False
                    add(
                        "error",
                        "qa.subtitle.invalid_timestamp",
                        f"Subtitle cue ends before it starts in {label}",
                        timestamp_s=Decimal(str(cue.start_s)),
                    )
                if cue.start_s < previous_start:
                    subtitles_ok = False
                    entry["valid"] = False
                    add(
                        "error",
                        "qa.subtitle.out_of_order",
                        f"Subtitle cues are out of order in {label}",
                        timestamp_s=Decimal(str(cue.start_s)),
                    )
                previous_start = cue.start_s
                if not cue.text:
                    subtitles_ok = False
                    entry["valid"] = False
                    add(
                        "error",
                        "qa.subtitle.empty_cue",
                        f"Subtitle cue has no text in {label}",
                        timestamp_s=Decimal(str(cue.start_s)),
                    )
                if video_duration is not None and cue.end_s > float(video_duration) + float(
                    settings.duration_tolerance_s
                ):
                    subtitles_ok = False
                    entry["valid"] = False
                    add(
                        "error",
                        "qa.subtitle.beyond_duration",
                        f"Subtitle cue ends after the video ends in {label}",
                        timestamp_s=Decimal(str(cue.start_s)),
                    )
            entry.setdefault("valid", True)
            details.append(entry)

        language = expected.get("language")
        if isinstance(language, str) and language.strip().lower() in HINDI_LANGUAGES and cues_all:
            has_devanagari = any(
                DEVANAGARI_START <= ord(character) <= DEVANAGARI_END
                for cue in cues_all
                for character in cue.text
            )
            if not has_devanagari:
                severity = "error" if language.strip().lower() == "hi" else "warning"
                add(
                    severity,
                    "qa.subtitle.hindi_text_lost",
                    "No Devanagari characters found in Hindi/Hinglish subtitles",
                    recommendation="Regenerate subtitles using a Devanagari-capable renderer.",
                )
        if burn_in and not burn_in_verified:
            add(
                "warning",
                "qa.subtitle.burn_in_unverified",
                "Burn-in was requested but cannot be verified at pixel level in this milestone",
                recommendation="Inspect sample frames manually before delivery.",
            )
        checks["subtitles_valid"] = subtitles_ok
        checks["burn_in_verified"] = burn_in_verified
        metrics["subtitle_count"] = len(subtitle_paths)
        metrics["subtitle_cue_count"] = len(cues_all)
        metrics["subtitles"] = details

    def _check_detectors(
        self,
        target: Path,
        settings: QAThresholds,
        timeline: object | None,
        expected: Mapping[str, object],
        add: Callable[..., None],
        checks: dict[str, object],
        metrics: dict[str, object],
    ) -> None:
        raw_tails = metrics.get("intentional_silent_tails")
        intentional_silence = (
            [entry for entry in raw_tails if isinstance(entry, Mapping)]
            if isinstance(raw_tails, (list, tuple))
            else []
        )
        raw_black = expected.get("intentional_black")
        intentional_black = (
            [entry for entry in raw_black if isinstance(entry, Mapping)]
            if isinstance(raw_black, (list, tuple))
            else []
        )

        if not settings.run_detectors:
            checks["detectors_run"] = False
            add(
                "warning",
                "qa.video.black_detect_disabled",
                "Black-frame detection was disabled by configuration",
            )
            add(
                "warning",
                "qa.audio.silence_detect_disabled",
                "Silence detection was disabled by configuration",
            )
            metrics["black_intervals"] = []
            metrics["silence_intervals"] = []
            return

        black: list[dict[str, float]] = []
        silence: list[dict[str, float]] = []
        try:
            black = self.detect_black(target, thresholds=settings)
        except DetectorError as exc:
            add("warning", exc.code, exc.message)
        try:
            silence = self.detect_silence(target, thresholds=settings)
        except DetectorError as exc:
            add("warning", exc.code, exc.message)
        checks["detectors_run"] = True
        metrics["black_intervals"] = black
        metrics["silence_intervals"] = silence

        for interval in silence:
            if self._is_intentional(interval, intentional_silence):
                continue
            add(
                "warning",
                "qa.audio.unexpected_silence",
                f"Unexpected silence from {interval['start_s']:.3f}s to "
                f"{interval['end_s']:.3f}s ({interval['duration_s']:.3f}s)",
                timestamp_s=Decimal(str(interval["start_s"])),
                recommendation="Compare with the intentional silent tails in the timeline.",
            )
        for interval in black:
            if self._is_intentional(interval, intentional_black):
                continue
            if interval["duration_s"] < settings.black_min_duration_s:
                continue
            add(
                "warning",
                "qa.video.unexpected_black",
                f"Unexpected black frame run from {interval['start_s']:.3f}s to "
                f"{interval['end_s']:.3f}s ({interval['duration_s']:.3f}s)",
                timestamp_s=Decimal(str(interval["start_s"])),
                recommendation="Review scene transitions and intentional black scenes.",
            )

    @staticmethod
    def _is_intentional(
        interval: Mapping[str, float], intentional: Sequence[Mapping[object, object]]
    ) -> bool:
        start = float(interval["start_s"])
        end = float(interval["end_s"])
        for entry in intentional:
            raw_start = _first(entry, "start_s", "start", "begin_s")
            raw_end = _first(entry, "end_s", "end", "finish_s")
            start_value = _optional_float(raw_start)
            end_value = _optional_float(raw_end)
            if start_value is None or end_value is None:
                continue
            if start + 0.5 >= start_value and end - 0.5 <= end_value:
                return True
        return False


def qa_status(
    final_video_path: str | Path, *, expected: Mapping[str, object], **kwargs: Any
) -> dict[str, object]:
    """Convenience wrapper returning ``{"status": ..., "result"|"error": ...}``."""
    try:
        report = FinalQAValidator().validate(final_video_path, expected=expected, **kwargs)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "result": report.to_dict()}
