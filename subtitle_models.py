"""Immutable subtitle models, timing helpers and serializers.

Timeline values are :class:`~decimal.Decimal` in memory and decimal strings in
JSON. Validation issues reuse the structured
:class:`~manifest_validation.ValidationIssue` type. This module is pure: it has
no MCP, FFmpeg, TTS or filesystem dependencies and never fabricates word-level
timings.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from manifest_validation import ValidationIssue
from render_models import coerce_decimal

SUBTITLE_LANGUAGES = frozenset({"en", "hi", "hinglish"})
SUBTITLE_FORMATS = frozenset({"srt", "vtt"})
DEFAULT_LANGUAGE = "en"
DEFAULT_FORMAT = "srt"
DEFAULT_FONT_FAMILY = "Arial"
DEFAULT_FONT_SIZE = 36
DEFAULT_MAX_CHARS_PER_LINE = 42
DEFAULT_MAX_LINES = 2
DEFAULT_OFFSET_S = Decimal("0")  # noqa: FURB157 - contract uses the decimal string

_POSITIONS = frozenset({"top", "middle", "bottom"})
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SENTENCE_CHARS = frozenset(".!?।")


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _normalize_color(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX_COLOR_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a #RRGGBB hex color: {value!r}")
    return value.upper()


def _issues_tuple(values: object, label: str) -> tuple[ValidationIssue, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of ValidationIssue entries")
    result = tuple(values)
    for issue in result:
        if not isinstance(issue, ValidationIssue):
            raise TypeError(f"{label} must contain ValidationIssue entries")
    return result


def ass_color(hex_color: str) -> str:
    """Convert a ``#RRGGBB`` color to an opaque ASS ``&HAABBGGRR`` color."""
    normalized = _normalize_color(hex_color, "hex_color")
    red = int(normalized[1:3], 16)
    green = int(normalized[3:5], 16)
    blue = int(normalized[5:7], 16)
    return f"&H00{blue:02X}{green:02X}{red:02X}"


@dataclass(frozen=True)
class SubtitleStyle:
    """Visual style shared by external subtitle files and burn-in filters."""

    font_family: str = DEFAULT_FONT_FAMILY
    font_size: int = DEFAULT_FONT_SIZE
    font_color: str = "#FFFFFF"
    outline: int = 2
    outline_color: str = "#000000"
    background: bool = False
    background_color: str = "#000000"
    background_opacity: int = 64
    shadow: int = 0
    position: str = "bottom"
    margin_l: int = 40
    margin_r: int = 40
    margin_v: int = 40
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE
    max_lines: int = DEFAULT_MAX_LINES
    bold: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "font_family", _require_nonempty_string(self.font_family, "font_family")
        )
        for name in (
            "font_size",
            "outline",
            "background_opacity",
            "shadow",
            "margin_l",
            "margin_r",
            "margin_v",
            "max_chars_per_line",
            "max_lines",
        ):
            minimum = 1 if name in {"font_size", "max_chars_per_line", "max_lines"} else 0
            object.__setattr__(self, name, _require_int(getattr(self, name), name, minimum=minimum))
        if self.background_opacity > 100:
            raise ValueError("background_opacity must be a percentage between 0 and 100")
        for name in ("font_color", "outline_color", "background_color"):
            object.__setattr__(self, name, _normalize_color(getattr(self, name), name))
        if self.position not in _POSITIONS:
            raise ValueError(f"position must be one of {sorted(_POSITIONS)}")
        for name in ("background", "bold"):
            object.__setattr__(self, name, _require_bool(getattr(self, name), name))

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | SubtitleStyle | None = None
    ) -> SubtitleStyle:
        """Build a style from a mapping, ignoring unknown keys."""
        if data is None:
            return cls()
        if isinstance(data, SubtitleStyle):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("subtitle style must be an object")
        keys = (
            "font_family",
            "font_size",
            "font_color",
            "outline",
            "outline_color",
            "background",
            "background_color",
            "background_opacity",
            "shadow",
            "position",
            "margin_l",
            "margin_r",
            "margin_v",
            "max_chars_per_line",
            "max_lines",
            "bold",
        )
        return cls(**{key: data[key] for key in keys if key in data})

    def to_dict(self) -> dict[str, object]:
        return {
            "font_family": self.font_family,
            "font_size": self.font_size,
            "font_color": self.font_color,
            "outline": self.outline,
            "outline_color": self.outline_color,
            "background": self.background,
            "background_color": self.background_color,
            "background_opacity": self.background_opacity,
            "shadow": self.shadow,
            "position": self.position,
            "margin_l": self.margin_l,
            "margin_r": self.margin_r,
            "margin_v": self.margin_v,
            "max_chars_per_line": self.max_chars_per_line,
            "max_lines": self.max_lines,
            "bold": self.bold,
        }


@dataclass(frozen=True)
class SubtitleSegment:
    """One timed text cue on the render timeline."""

    index: int
    start_s: Decimal
    end_s: Decimal
    text: str
    scene_id: str = ""
    language: str = DEFAULT_LANGUAGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _require_int(self.index, "index", minimum=1))
        object.__setattr__(
            self, "start_s", coerce_decimal(self.start_s, "start_s", non_negative=True)
        )
        object.__setattr__(
            self, "end_s", coerce_decimal(self.end_s, "end_s", non_negative=True)
        )
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        object.__setattr__(self, "text", _require_nonempty_string(self.text, "text"))
        if not isinstance(self.scene_id, str):
            raise TypeError("scene_id must be a string")
        object.__setattr__(self, "scene_id", self.scene_id)
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty string")

    @property
    def duration_s(self) -> Decimal:
        return self.end_s - self.start_s

    def format_timestamp(self, fmt: str, offset_s: Decimal = DEFAULT_OFFSET_S) -> str:
        """Format the segment start as an SRT/VTT timestamp with *offset_s*."""
        if fmt not in SUBTITLE_FORMATS:
            raise ValueError(f"Unsupported subtitle format: {fmt!r}")
        shifted = self.start_s + coerce_decimal(offset_s, "offset_s")
        return _format_timestamp_value(shifted, fmt)

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start_s": _decimal_text(self.start_s),
            "end_s": _decimal_text(self.end_s),
            "duration_s": _decimal_text(self.duration_s),
            "text": self.text,
            "scene_id": self.scene_id,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | SubtitleSegment) -> SubtitleSegment:
        if isinstance(data, SubtitleSegment):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("subtitle segment must be an object")
        return cls(
            index=data.get("index", 1),
            start_s=data.get("start_s", Decimal(0)),
            end_s=data.get("end_s", Decimal(0)),
            text=data.get("text", ""),
            scene_id=data.get("scene_id", ""),
            language=data.get("language", DEFAULT_LANGUAGE),
        )


def _format_timestamp_value(value: Decimal, fmt: str) -> str:
    if value < 0:
        raise ValueError(f"Subtitle timestamp cannot be negative: {value}")
    total_ms = int((value * 1000).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    separator = "," if fmt == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


@dataclass(frozen=True)
class SubtitleTrack:
    """A complete subtitle track for one render and language."""

    project_id: str
    render_id: str
    language: str
    formats: tuple[str, ...]
    segments: tuple[SubtitleSegment, ...]
    style: SubtitleStyle
    timing_source: str
    total_duration_s: Decimal
    offset_s: Decimal = DEFAULT_OFFSET_S
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_id", _require_nonempty_string(self.project_id, "project_id")
        )
        object.__setattr__(self, "render_id", _require_nonempty_string(self.render_id, "render_id"))
        if self.language not in SUBTITLE_LANGUAGES:
            raise ValueError(
                f"Unsupported subtitle language {self.language!r}; "
                f"supported: {sorted(SUBTITLE_LANGUAGES)}"
            )
        formats = tuple(self.formats)
        if not formats or any(fmt not in SUBTITLE_FORMATS for fmt in formats):
            raise ValueError(f"formats must be a non-empty subset of {sorted(SUBTITLE_FORMATS)}")
        object.__setattr__(self, "formats", formats)
        segments = tuple(self.segments)
        for segment in segments:
            if not isinstance(segment, SubtitleSegment):
                raise TypeError("segments must contain SubtitleSegment entries")
        object.__setattr__(self, "segments", segments)
        if not isinstance(self.style, SubtitleStyle):
            raise TypeError("style must be a SubtitleStyle")
        object.__setattr__(
            self,
            "total_duration_s",
            coerce_decimal(self.total_duration_s, "total_duration_s", non_negative=True),
        )
        object.__setattr__(self, "offset_s", coerce_decimal(self.offset_s, "offset_s"))
        object.__setattr__(
            self, "timing_source", _require_nonempty_string(self.timing_source, "timing_source")
        )
        object.__setattr__(self, "warnings", _issues_tuple(self.warnings, "warnings"))

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "render_id": self.render_id,
            "language": self.language,
            "formats": list(self.formats),
            "segment_count": self.segment_count,
            "segments": [segment.to_dict() for segment in self.segments],
            "style": self.style.to_dict(),
            "timing_source": self.timing_source,
            "total_duration_s": _decimal_text(self.total_duration_s),
            "offset_s": _decimal_text(self.offset_s),
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class SubtitleGenerationResult:
    """External subtitle files plus an optional, informational burn-in filter."""

    project_id: str
    render_id: str
    track: SubtitleTrack
    files: dict[str, str]
    burn_in_filter: str | None
    capabilities: dict[str, object]
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_id", _require_nonempty_string(self.project_id, "project_id")
        )
        object.__setattr__(self, "render_id", _require_nonempty_string(self.render_id, "render_id"))
        if not isinstance(self.track, SubtitleTrack):
            raise TypeError("track must be a SubtitleTrack")
        if not isinstance(self.files, Mapping) or not self.files:
            raise ValueError("files must be a non-empty format-to-path mapping")
        for fmt, path in self.files.items():
            if fmt not in SUBTITLE_FORMATS or not isinstance(path, str) or not path.strip():
                raise ValueError("files must map 'srt'/'vtt' to non-empty paths")
        object.__setattr__(self, "files", dict(self.files))
        if self.burn_in_filter is not None and not isinstance(self.burn_in_filter, str):
            raise TypeError("burn_in_filter must be a string or None")
        if not isinstance(self.capabilities, Mapping):
            raise TypeError("capabilities must be a mapping")
        object.__setattr__(self, "capabilities", dict(self.capabilities))
        object.__setattr__(self, "warnings", _issues_tuple(self.warnings, "warnings"))

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "render_id": self.render_id,
            "track": self.track.to_dict(),
            "files": dict(self.files),
            "burn_in_filter": self.burn_in_filter,
            "capabilities": dict(self.capabilities),
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


class SubtitleGenerationError(RuntimeError):
    """Structured subtitle generation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        location: str = "",
        *,
        stage: str = "subtitle",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location
        self.stage = stage

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "stage": self.stage,
        }


def split_sentences(text: str, language: str) -> list[str]:
    """Split text on ``.!?।`` and newlines without altering its characters.

    The terminator stays attached to its sentence. Devanagari text is preserved
    codepoint-for-codepoint; *language* is accepted for API symmetry and does
    not change the splitting rules.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(language, str):
        raise TypeError("language must be a string")
    sentences: list[str] = []
    current: list[str] = []
    for char in text:
        if char == "\n":
            candidate = "".join(current).strip()
            if candidate:
                sentences.append(candidate)
            current = []
            continue
        current.append(char)
        if char in _SENTENCE_CHARS:
            candidate = "".join(current).strip()
            if candidate:
                sentences.append(candidate)
            current = []
    candidate = "".join(current).strip()
    if candidate:
        sentences.append(candidate)
    return sentences


def validate_segments(
    segments: Sequence[SubtitleSegment],
    video_duration_s: Decimal | int | str,
) -> list[ValidationIssue]:
    """Return ordering, overlap, timestamp and bounds issues; never mutate."""
    duration = coerce_decimal(video_duration_s, "video_duration_s", non_negative=True)
    issues: list[ValidationIssue] = []
    previous: SubtitleSegment | None = None
    for position, segment in enumerate(segments):
        if not isinstance(segment, SubtitleSegment):
            raise TypeError("segments must contain SubtitleSegment entries")
        location = f"segments[{position}] ({segment.index})"
        if segment.start_s < 0 or segment.end_s <= segment.start_s:
            issues.append(
                ValidationIssue(
                    "subtitle.segment.timestamp",
                    "start_s must be non-negative and end_s must exceed start_s",
                    location,
                )
            )
        if segment.index <= 0:
            issues.append(
                ValidationIssue("subtitle.segment.index", "index must be positive", location)
            )
        if segment.end_s > duration:
            issues.append(
                ValidationIssue(
                    "subtitle.segment.out_of_bounds",
                    f"Segment ends after the video duration ({_decimal_text(duration)}s)",
                    location,
                )
            )
        if previous is not None:
            if segment.start_s < previous.start_s:
                issues.append(
                    ValidationIssue(
                        "subtitle.segment.order",
                        "Segments must be ordered by start time",
                        location,
                    )
                )
            elif segment.start_s < previous.end_s:
                issues.append(
                    ValidationIssue(
                        "subtitle.segment.overlap",
                        "Segment overlaps the preceding segment",
                        location,
                    )
                )
        previous = segment
    return issues


def _wrap_text(text: str, max_chars_per_line: int, max_lines: int) -> list[str]:
    """Wrap at word boundaries; line count wins if both limits cannot hold."""
    if len(text) <= max_chars_per_line and "\n" not in text:
        return [text]
    words = " ".join(text.split()).split(" ")
    words = [word for word in words if word]
    if not words:
        return []
    if max_lines >= len(words):
        return words

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if not current or len(candidate) <= max_chars_per_line:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) <= max_lines:
        return lines

    total_chars = sum(len(word) for word in words)
    boundaries: list[int] = []
    for line_number in range(1, max_lines):
        ideal = total_chars * line_number / max_lines
        lower = (boundaries[-1] if boundaries else 0) + 1
        upper = len(words) - (max_lines - line_number)
        candidates = range(lower, upper + 1)
        cumulative = 0
        ends: dict[int, int] = {}
        for word_index, word in enumerate(words, start=1):
            cumulative += len(word) + (1 if word_index > 1 else 0)
            ends[word_index] = cumulative
        best = min(
            candidates,
            key=lambda index: (abs(ends[index] - ideal), index),
        )
        boundaries.append(best)

    redistributed: list[str] = []
    start = 0
    for end in (*boundaries, len(words)):
        redistributed.append(" ".join(words[start:end]))
        start = end
    return redistributed


def serialize_subtitles(track: SubtitleTrack, fmt: str) -> str:
    """Serialize a track to SRT or WebVTT, applying the track offset."""
    if not isinstance(track, SubtitleTrack):
        raise TypeError("track must be a SubtitleTrack")
    if fmt not in SUBTITLE_FORMATS:
        raise ValueError(f"Unsupported subtitle format: {fmt!r}")

    blocks: list[str] = []
    for position, segment in enumerate(track.segments, start=1):
        start = _format_timestamp_value(segment.start_s + track.offset_s, fmt)
        end = _format_timestamp_value(segment.end_s + track.offset_s, fmt)
        lines = _wrap_text(
            segment.text, track.style.max_chars_per_line, track.style.max_lines
        )
        blocks.append(f"{position}\n{start} --> {end}\n" + "\n".join(lines))
    body = "\n\n".join(blocks)
    if fmt == "vtt":
        return f"WEBVTT\n\n{body}\n" if body else "WEBVTT\n\n"
    return f"{body}\n" if body else ""
