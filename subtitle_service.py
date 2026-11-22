"""Generate external subtitle tracks from an existing audio timeline.

Generation is read-only with respect to project and scene manifests: narration
text is read from the timeline scene or, when absent there, from the scene
manifest through :class:`~scene_service.SceneService`. Subtitle files are
written under a render directory inside the project, never over existing ones.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

from manifest_validation import ValidationIssue
from project_paths import (
    PathValidationError,
    resolve_within_project,
    to_project_relative,
    validate_identifier,
)
from project_service import ProjectService
from render_models import coerce_decimal
from scene_service import SceneService
from subtitle_models import (
    DEFAULT_FORMAT,
    DEFAULT_LANGUAGE,
    SUBTITLE_FORMATS,
    SUBTITLE_LANGUAGES,
    SubtitleGenerationError,
    SubtitleGenerationResult,
    SubtitleSegment,
    SubtitleStyle,
    SubtitleTrack,
    serialize_subtitles,
    split_sentences,
    validate_segments,
)
from subtitle_renderer import (
    SubtitleCapabilities,
    build_burn_in_filter,
    check_capabilities,
    write_subtitle_file,
)

logger = logging.getLogger(__name__)

TIMING_SOURCE = "audio_timeline"
_EMPTY_MANUAL_SEGMENTS: Mapping[str, Sequence[Mapping[str, object]]] = MappingProxyType({})


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _normalize_formats(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("formats must be a sequence of subtitle format names")
    formats = tuple(value)
    if not formats:
        raise ValueError("formats must not be empty")
    for fmt in formats:
        if not isinstance(fmt, str) or fmt.lower() not in SUBTITLE_FORMATS:
            raise ValueError(
                f"Unsupported subtitle format {fmt!r}; supported: {sorted(SUBTITLE_FORMATS)}"
            )
    return tuple(dict.fromkeys(fmt.lower() for fmt in formats))


def _normalize_manual_segments(
    value: object,
) -> Mapping[str, tuple[dict[str, object], ...]]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("manual_segments must be an object keyed by scene_id")
    normalized: dict[str, tuple[dict[str, object], ...]] = {}
    for scene_id, entries in value.items():
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("manual_segments keys must be non-empty scene IDs")
        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise TypeError(f"manual_segments[{scene_id!r}] must be a sequence")
        cleaned: list[dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise TypeError(f"manual_segments[{scene_id!r}] entries must be objects")
            for key in ("start_s", "end_s", "text"):
                if key not in entry:
                    raise ValueError(f"manual_segments[{scene_id!r}] requires {key}")
            start = coerce_decimal(
                entry["start_s"], f"manual_segments[{scene_id!r}].start_s", non_negative=True
            )
            end = coerce_decimal(
                entry["end_s"], f"manual_segments[{scene_id!r}].end_s", non_negative=True
            )
            if end <= start:
                raise ValueError(f"manual_segments[{scene_id!r}] requires end_s > start_s")
            text = entry["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"manual_segments[{scene_id!r}].text must be non-empty")
            cleaned.append({"start_s": start, "end_s": end, "text": text})
        normalized[scene_id] = tuple(cleaned)
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class SubtitleConfig:
    """Validated options for one subtitle generation call."""

    language: str = DEFAULT_LANGUAGE
    formats: tuple[str, ...] = ("srt", "vtt")
    granularity: str = "scene"
    offset_s: Decimal = Decimal(0)
    style: SubtitleStyle = field(default_factory=SubtitleStyle)
    manual_segments: Mapping[str, Sequence[Mapping[str, object]]] = field(
        default_factory=lambda: _EMPTY_MANUAL_SEGMENTS
    )
    check_fonts: bool = True
    subtitle_dir_name: str = "subtitles"

    def __post_init__(self) -> None:
        if self.language not in SUBTITLE_LANGUAGES:
            raise ValueError(
                f"Unsupported subtitle language {self.language!r}; "
                f"supported: {sorted(SUBTITLE_LANGUAGES)}"
            )
        object.__setattr__(self, "formats", _normalize_formats(self.formats))
        if self.granularity not in {"scene", "sentence"}:
            raise ValueError("granularity must be 'scene' or 'sentence'")
        object.__setattr__(
            self, "offset_s", coerce_decimal(self.offset_s, "offset_s")
        )
        if not isinstance(self.style, SubtitleStyle):
            object.__setattr__(self, "style", SubtitleStyle.from_dict(self.style))
        object.__setattr__(
            self, "manual_segments", _normalize_manual_segments(self.manual_segments)
        )
        if not isinstance(self.check_fonts, bool):
            raise TypeError("check_fonts must be a boolean")
        name = self.subtitle_dir_name
        if (
            not isinstance(name, str)
            or not name.strip()
            or name in {".", ".."}
            or Path(name).name != name
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("subtitle_dir_name must be a single safe directory name")

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | SubtitleConfig | None = None
    ) -> SubtitleConfig:
        if data is None:
            return cls()
        if isinstance(data, SubtitleConfig):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("subtitle config must be an object")
        return cls(
            language=data.get("language", DEFAULT_LANGUAGE),
            formats=data.get("formats", ("srt", "vtt")),
            granularity=data.get("granularity", "scene"),
            offset_s=data.get("offset_s", Decimal(0)),
            style=SubtitleStyle.from_dict(data.get("style")),
            manual_segments=data.get("manual_segments", {}),
            check_fonts=data.get("check_fonts", True),
            subtitle_dir_name=data.get("subtitle_dir_name", "subtitles"),
        )

    def to_dict(self) -> dict[str, object]:
        manual: dict[str, list[dict[str, object]]] = {}
        for scene_id, entries in self.manual_segments.items():
            manual[scene_id] = [
                {
                    "start_s": _decimal_text(entry["start_s"]),  # type: ignore[arg-type]
                    "end_s": _decimal_text(entry["end_s"]),  # type: ignore[arg-type]
                    "text": entry["text"],
                }
                for entry in entries
            ]
        return {
            "language": self.language,
            "formats": list(self.formats),
            "granularity": self.granularity,
            "offset_s": _decimal_text(self.offset_s),
            "style": self.style.to_dict(),
            "manual_segments": manual,
            "check_fonts": self.check_fonts,
            "subtitle_dir_name": self.subtitle_dir_name,
        }


class SubtitleService:
    """Build subtitle tracks from a timeline without touching manifests."""

    def __init__(
        self,
        projects: ProjectService | None = None,
        scenes: SceneService | None = None,
        capabilities: SubtitleCapabilities | None = None,
        command_runner: Any | None = None,
    ) -> None:
        self.projects = projects if projects is not None else ProjectService()
        self.scenes = scenes if scenes is not None else SceneService(self.projects)
        self.capabilities = capabilities
        self.command_runner = command_runner

    def _narration_text(self, project_id: str, scene: Any) -> str:
        for attr in ("narration_text", "narration"):
            value = getattr(scene, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        scene_id = getattr(scene, "scene_id", "")
        if self.scenes is not None and isinstance(scene_id, str) and scene_id:
            try:
                stored = self.scenes.get_scene(project_id, scene_id)
            except Exception as exc:  # noqa: BLE001 - a bad scene must not abort generation
                logger.debug("Could not read narration for scene %s: %s", scene_id, exc)
                return ""
            value = getattr(stored, "narration", "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _capabilities_for(self, config: SubtitleConfig) -> SubtitleCapabilities:
        if self.capabilities is not None:
            return self.capabilities
        if not config.check_fonts:
            return SubtitleCapabilities(
                ffmpeg_bin="ffmpeg",
                ffprobe_bin="ffprobe",
                libass=False,
                burn_in_supported=False,
                fonts=(),
                devanagari_font=None,
                notes=("Font and FFmpeg checks were disabled by configuration",),
            )
        return check_capabilities(command_runner=self.command_runner)

    def _scene_window(
        self,
        scene: Any,
        total_duration_s: Decimal,
        location: str,
        warnings: list[ValidationIssue],
    ) -> tuple[Decimal, Decimal, Decimal] | None:
        """Return (start, end, narration duration) clamped to the timeline."""
        try:
            start = coerce_decimal(
                getattr(scene, "narration_start_s", Decimal(0)),
                "narration_start_s",
                non_negative=True,
            )
            effective = getattr(scene, "effective_duration_s", None)
            if effective is None:
                effective = getattr(scene, "narration_duration_s", Decimal(0))
            effective_duration = coerce_decimal(
                effective, "effective_duration_s", non_negative=True
            )
            narration_duration = getattr(scene, "narration_duration_s", None)
            narration_duration_s = (
                None
                if narration_duration is None
                else coerce_decimal(
                    narration_duration, "narration_duration_s", non_negative=True
                )
            )
            video_end = getattr(scene, "video_end_s", None)
            scene_end = (
                coerce_decimal(video_end, "video_end_s", non_negative=True)
                if video_end is not None
                else start + effective_duration
            )
        except (TypeError, ValueError) as exc:
            warnings.append(
                ValidationIssue("subtitle.timing.invalid", str(exc), location)
            )
            return None
        if narration_duration_s is None:
            narration_duration_s = effective_duration
            warnings.append(
                ValidationIssue(
                    "subtitle.timing.invalid",
                    "narration_duration_s is missing; scene duration was used",
                    f"{location}.narration_duration_s",
                )
            )
        if start < 0:
            warnings.append(
                ValidationIssue(
                    "subtitle.timing.invalid",
                    "Negative narration start was clamped to zero",
                    f"{location}.narration_start_s",
                )
            )
            start = Decimal(0)
        end = min(scene_end, start + effective_duration) if effective_duration > 0 else scene_end
        end = min(end, start + narration_duration_s, total_duration_s)
        if end <= start:
            warnings.append(
                ValidationIssue(
                    "subtitle.timing.invalid",
                    "Scene has no usable subtitle window after clamping",
                    location,
                )
            )
            return None
        return start, end, narration_duration_s

    def _sentence_segments(
        self,
        text: str,
        start: Decimal,
        end: Decimal,
        scene_id: str,
        language: str,
    ) -> list[SubtitleSegment]:
        """Estimate sentence timings from character counts.

        This is sentence-level estimation, NOT word alignment: it never
        fabricates word timestamps and never subdivides below sentence granularity.
        """
        sentences = split_sentences(text, language)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [SubtitleSegment(1, start, end, sentences[0], scene_id, language)]
        total_chars = sum(len(sentence) for sentence in sentences)
        start_ms = int((start * 1000).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        end_ms = int((end * 1000).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        duration_ms = end_ms - start_ms
        segments: list[SubtitleSegment] = []
        cumulative_chars = 0
        previous_ms = start_ms
        for position, sentence in enumerate(sentences):
            cumulative_chars += len(sentence)
            if position == len(sentences) - 1:
                boundary_ms = end_ms
            else:
                offset = (
                    Decimal(duration_ms * cumulative_chars) / Decimal(total_chars)
                ).quantize(Decimal(1), rounding=ROUND_HALF_UP)
                boundary_ms = start_ms + int(offset)
            boundary_ms = max(previous_ms, min(boundary_ms, end_ms))
            if boundary_ms > previous_ms:
                segments.append(
                    SubtitleSegment(
                        index=position + 1,
                        start_s=Decimal(previous_ms) / 1000,
                        end_s=Decimal(boundary_ms) / 1000,
                        text=sentence,
                        scene_id=scene_id,
                        language=language,
                    )
                )
            previous_ms = boundary_ms
        return segments

    def generate(
        self,
        project_id: str,
        timeline: Any,
        config: SubtitleConfig | Mapping[str, Any] | None,
        *,
        render_dir: str | Path,
        render_id: str,
    ) -> SubtitleGenerationResult:
        """Generate external subtitle files for one render; manifests stay untouched."""
        validate_identifier(project_id, "project_id")
        if not isinstance(render_id, str) or not render_id.strip():
            raise SubtitleGenerationError(
                "subtitle.generation.invalid", "render_id must be a non-empty string", "render_id"
            )
        effective_config = SubtitleConfig.from_dict(config)
        project_root = self.projects.project_dir(project_id)
        try:
            resolved_render_dir = resolve_within_project(project_root, render_dir, "render_dir")
            subtitle_dir = resolve_within_project(
                project_root,
                resolved_render_dir / effective_config.subtitle_dir_name,
                "subtitle_dir",
            )
        except PathValidationError as exc:
            raise SubtitleGenerationError(
                "subtitle.path.unsafe", str(exc), "render_dir"
            ) from exc

        total_value = getattr(timeline, "total_duration_s", None)
        if total_value is None:
            raise SubtitleGenerationError(
                "subtitle.timing.invalid",
                "timeline.total_duration_s is required",
                "timeline.total_duration_s",
            )
        try:
            total_duration_s = coerce_decimal(
                total_value, "timeline.total_duration_s", non_negative=True
            )
        except (TypeError, ValueError) as exc:
            raise SubtitleGenerationError(
                "subtitle.timing.invalid", str(exc), "timeline.total_duration_s"
            ) from exc

        warnings: list[ValidationIssue] = list(getattr(timeline, "warnings", ()) or ())
        candidates: list[SubtitleSegment] = []
        timeline_scenes = getattr(timeline, "scenes", ())
        if timeline_scenes is None or isinstance(timeline_scenes, (str, bytes)):
            raise SubtitleGenerationError(
                "subtitle.timing.invalid", "timeline.scenes must be a sequence", "timeline.scenes"
            )

        for position, scene in enumerate(timeline_scenes):
            scene_id = getattr(scene, "scene_id", "")
            if not isinstance(scene_id, str) or not scene_id:
                warnings.append(
                    ValidationIssue(
                        "subtitle.narration.missing",
                        "Timeline scene has no scene_id; subtitles were not generated",
                        f"scenes[{position}]",
                    )
                )
                continue
            location = f"scenes[{position}] ({scene_id})"
            window = self._scene_window(scene, total_duration_s, location, warnings)
            if window is None:
                continue
            start, end, _narration_duration = window

            manual = effective_config.manual_segments.get(scene_id)
            if manual:
                for manual_index, entry in enumerate(manual):
                    manual_start = coerce_decimal(
                        entry["start_s"], "manual start_s", non_negative=True
                    )
                    manual_end = coerce_decimal(
                        entry["end_s"], "manual end_s", non_negative=True
                    )
                    if manual_start < start or manual_end > end or manual_end <= manual_start:
                        warnings.append(
                            ValidationIssue(
                                "subtitle.segment.invalid",
                                "Manual segment must be ordered and inside the scene window",
                                f"{location}.manual_segments[{manual_index}]",
                            )
                        )
                        continue
                    candidates.append(
                        SubtitleSegment(
                            index=len(candidates) + 1,
                            start_s=manual_start,
                            end_s=manual_end,
                            text=str(entry["text"]),
                            scene_id=scene_id,
                            language=effective_config.language,
                        )
                    )
                continue

            narration_present = bool(getattr(scene, "narration_present", False))
            narration_text = self._narration_text(project_id, scene)
            if not narration_present or not narration_text:
                warnings.append(
                    ValidationIssue(
                        "subtitle.narration.missing",
                        f"Scene {scene_id} has no usable narration; no speech was fabricated",
                        f"{location}.narration",
                    )
                )
                continue
            if effective_config.granularity == "sentence":
                candidates.extend(
                    self._sentence_segments(
                        narration_text, start, end, scene_id, effective_config.language
                    )
                )
            else:
                candidates.append(
                    SubtitleSegment(
                        index=len(candidates) + 1,
                        start_s=start,
                        end_s=end,
                        text=narration_text,
                        scene_id=scene_id,
                        language=effective_config.language,
                    )
                )

        candidates.sort(key=lambda segment: (segment.start_s, segment.end_s))
        clean: list[SubtitleSegment] = []
        for segment in candidates:
            if clean and segment.start_s < clean[-1].end_s:
                warnings.append(
                    ValidationIssue(
                        "subtitle.segment.overlap",
                        f"Segment for {segment.scene_id} overlaps the previous cue",
                        f"segment ({segment.index})",
                    )
                )
                continue
            clean.append(segment)
        segments = tuple(replace(segment, index=index) for index, segment in enumerate(clean, 1))
        warnings.extend(validate_segments(segments, total_duration_s))

        track = SubtitleTrack(
            project_id=project_id,
            render_id=render_id,
            language=effective_config.language,
            formats=effective_config.formats,
            segments=segments,
            style=effective_config.style,
            timing_source=TIMING_SOURCE,
            total_duration_s=total_duration_s,
            offset_s=effective_config.offset_s,
            warnings=tuple(warnings),
        )

        files: dict[str, str] = {}
        paths: dict[str, Path] = {}
        for fmt in effective_config.formats:
            path = subtitle_dir / f"subtitles.{effective_config.language}.{fmt}"
            content = serialize_subtitles(track, fmt)
            if path.exists():
                raise SubtitleGenerationError(
                    "subtitle.file.collision",
                    f"Subtitle file already exists: {path}",
                    str(path),
                )
            write_subtitle_file(path, content)
            files[fmt] = to_project_relative(path, project_root)
            paths[fmt] = path

        capabilities = self._capabilities_for(effective_config)
        result = self.build_result(
            project_id=project_id,
            render_id=render_id,
            track=track,
            files=files,
            paths=paths,
            capabilities=capabilities,
            warnings=warnings,
        )
        logger.info(
            "Generated %d subtitle cues for %s/%s in %d formats",
            result.track.segment_count,
            project_id,
            render_id,
            len(files),
        )
        return result

    def build_result(
        self,
        *,
        project_id: str,
        render_id: str,
        track: SubtitleTrack,
        files: Mapping[str, str],
        paths: Mapping[str, Path],
        capabilities: SubtitleCapabilities,
        warnings: Sequence[ValidationIssue] = (),
        include_burn_in: bool = True,
    ) -> SubtitleGenerationResult:
        """Assemble a result; burn-in is informational and opt-out downstream."""
        issues = list(warnings)
        burn_in_filter: str | None = None
        primary = track.formats[0] if track.formats else DEFAULT_FORMAT
        primary_path = paths.get(primary)
        if include_burn_in and track.language in {"hi", "hinglish"} and not capabilities.devanagari_font:
            issues.append(
                ValidationIssue(
                    "subtitle.font.devanagari_missing",
                    "No Devanagari-capable font was verified; external subtitles were "
                    "written but burn-in was disabled",
                    "style.font_family",
                )
            )
        elif include_burn_in and not capabilities.burn_in_supported:
            issues.append(
                ValidationIssue(
                    "subtitle.burn_in.unsupported",
                    "FFmpeg subtitle burn-in filters were not verified",
                    "capabilities.burn_in_supported",
                )
            )
        elif include_burn_in and primary_path is not None:
            style = track.style
            if (
                track.language in {"hi", "hinglish"}
                and capabilities.devanagari_font
                and style.font_family not in capabilities.fonts
            ):
                # The configured default font cannot render Devanagari; use the
                # verified font for burn-in only and leave the track untouched.
                style = replace(style, font_family=capabilities.devanagari_font)
            burn_in_filter = build_burn_in_filter(primary_path, style)
        return SubtitleGenerationResult(
            project_id=project_id,
            render_id=render_id,
            track=track,
            files=dict(files),
            burn_in_filter=burn_in_filter,
            capabilities=capabilities.to_dict(),
            warnings=tuple(issues),
        )

    def generate_status(
        self,
        project_id: str,
        timeline: Any,
        config: SubtitleConfig | Mapping[str, Any] | None,
        *,
        render_dir: str | Path,
        render_id: str,
    ) -> dict[str, object]:
        """MCP-friendly wrapper returning a serializable status dictionary."""
        try:
            result = self.generate(
                project_id,
                timeline,
                config,
                render_dir=render_dir,
                render_id=render_id,
            )
        except SubtitleGenerationError as exc:
            return {"status": "error", "error": exc.to_dict()}
        except (TypeError, ValueError, OSError) as exc:
            error = SubtitleGenerationError(
                "subtitle.generation.failed", str(exc), "subtitle"
            )
            return {"status": "error", "error": error.to_dict()}
        return {"status": "ok", "result": result.to_dict()}
