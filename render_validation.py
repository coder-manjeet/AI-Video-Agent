"""MCP-independent validation profiles for scene videos and final videos.

Both profiles share one probing path and reuse
:class:`~manifest_validation.ValidationResult`; they never mutate files. The
scene profile accepts silent LTX clips, while the final profile requires an
audio stream by default and checks optional output expectations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import media_service
from manifest_validation import ValidationResult
from render_models import RenderExpectations, RenderMediaInfo

logger = logging.getLogger(__name__)

ProbeCallable = Callable[[Path], Mapping[str, object]]
PROBE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
)


class RenderValidator:
    """Validate generated media with scene-video and final-video profiles."""

    def __init__(self, probe: ProbeCallable | None = None) -> None:
        self.probe: ProbeCallable = (
            probe if probe is not None else media_service.probe_media
        )

    def validate_scene_video(
        self,
        path: str | Path,
        expected: RenderExpectations | None = None,
    ) -> ValidationResult:
        """Validate one scene clip; a missing audio stream is not an error."""
        return self._validate(
            path,
            profile="scene_video",
            expected=expected,
            require_audio=False,
        )

    def validate_final(
        self,
        path: str | Path,
        expected: RenderExpectations | None = None,
        require_audio: bool = True,
    ) -> ValidationResult:
        """Validate a final render, requiring audio unless explicitly disabled."""
        return self._validate(
            path,
            profile="final",
            expected=expected,
            require_audio=require_audio,
        )

    def _validate(
        self,
        path: str | Path,
        *,
        profile: str,
        expected: RenderExpectations | None,
        require_audio: bool,
    ) -> ValidationResult:
        result = ValidationResult()
        expectations = expected if expected is not None else RenderExpectations()
        target = Path(path).expanduser().resolve()

        if not target.is_file():
            result.add_error("render.file.missing", f"File does not exist: {target}")
            return result
        try:
            size = target.stat().st_size
        except OSError as exc:
            result.add_error("render.file.unreadable", f"Could not stat {target}: {exc}")
            return result
        if size == 0:
            result.add_error("render.file.empty", f"File is empty: {target}")
            return result

        try:
            info = self.probe(target)
        except PROBE_ERRORS as exc:
            result.add_error(
                "render.media.unreadable", f"Could not probe {target}: {exc}"
            )
            return result
        if not isinstance(info, Mapping):
            result.add_error(
                "render.media.unreadable",
                f"Probe returned a non-mapping result: {type(info).__name__}",
            )
            return result
        status = info.get("status")
        if status is not None and status != "ok":
            detail = info.get("error")
            suffix = f": {detail}" if isinstance(detail, str) and detail else ""
            result.add_error(
                "render.media.unreadable", f"Probe returned status {status!r}{suffix}"
            )
            return result

        media = RenderMediaInfo.from_probe(target, info)
        if not media.video_codec:
            if expectations.require_video_stream:
                result.add_error("render.video.missing_stream", "No video stream found")
            else:
                result.add_warning("render.video.missing_stream", "No video stream found")
            return result

        if media.duration_s is None:
            if expectations.require_positive_duration:
                result.add_error(
                    "render.video.invalid_duration",
                    "Video duration is missing, non-positive or invalid",
                )
                return result
            result.add_warning(
                "render.video.invalid_duration",
                "Video duration is missing, non-positive or invalid",
            )
        if media.width is None or media.height is None:
            result.add_error(
                "render.video.invalid_resolution",
                "Video width or height is missing or invalid",
            )
        if media.fps is None:
            result.add_error("render.video.invalid_fps", "Video FPS is missing or invalid")

        has_audio = bool(media.audio_codec)
        if require_audio:
            if not has_audio:
                result.add_error(
                    "render.final.missing_audio",
                    "Final video has no audio stream",
                )
            elif media.sample_rate is None or media.channels is None:
                result.add_error(
                    "render.final.invalid_audio_metadata",
                    "Audio sample rate or channel count is missing or invalid",
                )
        elif has_audio and (media.sample_rate is None or media.channels is None):
            result.add_warning(
                "render.scene.invalid_audio_metadata",
                "Audio stream has incomplete sample rate or channel metadata",
            )

        self._check_expectations(result, media, expectations)
        logger.debug(
            "Validated %s profile for %s: valid=%s", profile, target, result.valid
        )
        return result

    @staticmethod
    def _check_expectations(
        result: ValidationResult,
        media: RenderMediaInfo,
        expectations: RenderExpectations,
    ) -> None:
        resolution_problems: list[str] = []
        if expectations.width is not None and media.width != expectations.width:
            resolution_problems.append(
                f"width is {media.width}, expected {expectations.width}"
            )
        if expectations.height is not None and media.height != expectations.height:
            resolution_problems.append(
                f"height is {media.height}, expected {expectations.height}"
            )
        if resolution_problems:
            result.add_error(
                "render.video.resolution_mismatch", "; ".join(resolution_problems)
            )

        if (
            expectations.fps is not None
            and media.fps is not None
            and abs(media.fps - expectations.fps) > expectations.fps_tolerance
        ):
            result.add_error(
                "render.video.fps_mismatch",
                f"FPS is {media.fps:g}, expected {expectations.fps:g} "
                f"(tolerance {expectations.fps_tolerance:g})",
            )

        if expectations.expected_duration_s is not None and media.duration_s is not None:
            delta = abs(media.duration_s - expectations.expected_duration_s)
            if delta > expectations.duration_tolerance_s:
                result.add_error(
                    "render.video.duration_mismatch",
                    f"Duration {media.duration_s:f}s differs from "
                    f"expected {expectations.expected_duration_s:f}s "
                    f"by {delta:f}s (tolerance "
                    f"{expectations.duration_tolerance_s:f}s)",
                )
