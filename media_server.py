"""FastMCP server ``local-media-agent`` for local FFmpeg media operations.

This is a third, independent stdio server registered alongside
``local-video-agent`` and ``local-tts-agent``. Tool bodies stay thin: they
validate strings, move blocking FFmpeg work to a worker thread, and convert
expected failures into structured ``status="error"`` results. All media logic
lives in :mod:`media_service` and uses only the ``.venv`` environment.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from mcp.server.fastmcp import FastMCP

import media_service

LOGGER = logging.getLogger(__name__)

mcp = FastMCP("local-media-agent")


def _error(exc: Exception) -> dict[str, object]:
    """Convert an expected failure into a JSON-serializable tool result."""
    LOGGER.warning("Media operation failed: %s", exc)
    return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _reject_empty(**values: object) -> str | None:
    """Return an error message for the first empty/non-string argument."""
    for label, value in values.items():
        if not isinstance(value, str) or not value.strip():
            return f"{label} must be a non-empty string"
    return None


async def _invoke(
    operation: Callable[..., dict[str, object]], **kwargs: object
) -> dict[str, object]:
    """Run a blocking service operation off the event loop."""
    try:
        return await asyncio.to_thread(operation, **kwargs)
    except (media_service.MediaError, ValueError, OSError) as exc:
        return _error(exc)


@mcp.tool()
async def probe_media(
    path: str, timeout_s: float | None = None
) -> dict[str, object]:
    """Inspect a video or audio file with ffprobe.

    Returns ``status="ok"`` with duration, resolution, FPS, codecs, sample
    rate, channel count and file size. Missing or invalid files return
    ``status="error"`` with an explanatory ``error`` field.
    """
    problem = _reject_empty(path=path)
    if problem is not None:
        return _error(ValueError(problem))
    return await _invoke(media_service.probe_media, media_path=path, timeout_s=timeout_s)


@mcp.tool()
async def merge_narration(
    video_path: str,
    narration_path: str,
    output_path: str,
    volume: float = 1.0,
    duration_policy: str = "trim_audio",
    overwrite: bool = False,
    timeout_s: float | None = None,
) -> dict[str, object]:
    """Combine video with narration WAV and produce a validated MP4.

    ``duration_policy`` is ``trim_audio`` (match narration to the video) or
    ``extend_video`` (freeze the final frame until narration ends). ``volume``
    scales narration loudness. Existing outputs are refused unless
    ``overwrite=True``. Returns timing deltas, warnings and validation details;
    expected failures use ``status="error"``.
    """
    problem = _reject_empty(
        video_path=video_path,
        narration_path=narration_path,
        output_path=output_path,
    )
    if problem is not None:
        return _error(ValueError(problem))
    return await _invoke(
        media_service.merge_narration,
        video_path=video_path,
        narration_path=narration_path,
        output_path=output_path,
        volume=volume,
        duration_policy=duration_policy,
        overwrite=overwrite,
        timeout_s=timeout_s,
    )


@mcp.tool()
async def validate_media(
    path: str,
    expected_duration_s: float | None = None,
    duration_tolerance_s: float = media_service.DEFAULT_DURATION_TOLERANCE_S,
    timeout_s: float | None = None,
) -> dict[str, object]:
    """Check that a media file is readable and has valid video/audio streams.

    Returns ``valid`` plus per-check results, errors and stream metadata. An
    unreadable or incomplete file yields ``valid=False`` instead of raising.
    """
    problem = _reject_empty(path=path)
    if problem is not None:
        return _error(ValueError(problem))
    return await _invoke(
        media_service.validate_media,
        media_path=path,
        expected_duration_s=expected_duration_s,
        duration_tolerance_s=duration_tolerance_s,
        timeout_s=timeout_s,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
