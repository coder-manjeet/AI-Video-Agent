"""Unit tests for the ``local-media-agent`` FastMCP tool boundary.

All service calls are monkeypatched; no FFmpeg subprocess or real media file
is required for these tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import media_server
import media_service


def test_server_name() -> None:
    assert media_server.mcp.name == "local-media-agent"


@pytest.mark.asyncio
async def test_registered_tool_names() -> None:
    tools = await media_server.mcp.list_tools()

    assert {tool.name for tool in tools} == {
        "probe_media",
        "merge_narration",
        "validate_media",
    }


@pytest.mark.asyncio
async def test_probe_tool_forwards_arguments_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_probe(media_path: str, *, timeout_s: float | None) -> dict[str, object]:
        captured.update(media_path=media_path, timeout_s=timeout_s)
        return {"status": "ok", "duration_s": 1.5, "video_codec": "h264"}

    monkeypatch.setattr(media_service, "probe_media", fake_probe)

    result = await media_server.probe_media("clip.mp4", timeout_s=7.0)

    assert result["status"] == "ok"
    assert result["duration_s"] == 1.5
    assert captured == {"media_path": "clip.mp4", "timeout_s": 7.0}


@pytest.mark.asyncio
async def test_probe_tool_reports_missing_file_as_structured_error(
    tmp_path: Path,
) -> None:
    result = await media_server.probe_media(str(tmp_path / "missing.mp4"))

    assert result["status"] == "error"
    assert "file not found" in str(result["error"]).lower()


@pytest.mark.asyncio
async def test_merge_tool_forwards_all_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_merge(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "merged", "path": kwargs["output_path"]}

    monkeypatch.setattr(media_service, "merge_narration", fake_merge)

    result = await media_server.merge_narration(
        "clip.mp4",
        "voice.wav",
        "final.mp4",
        volume=0.6,
        duration_policy="extend_video",
        overwrite=True,
        timeout_s=42.0,
    )

    assert result["status"] == "merged"
    assert captured["video_path"] == "clip.mp4"
    assert captured["narration_path"] == "voice.wav"
    assert captured["output_path"] == "final.mp4"
    assert captured["volume"] == 0.6
    assert captured["duration_policy"] == "extend_video"
    assert captured["overwrite"] is True
    assert captured["timeout_s"] == 42.0


@pytest.mark.asyncio
async def test_merge_tool_reports_existing_output_without_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "clip.mp4"
    narration = tmp_path / "voice.wav"
    output = tmp_path / "final.mp4"
    video.write_bytes(b"video")
    narration.write_bytes(b"audio")
    output.write_bytes(b"existing")

    def forbidden_probe(*_args: object, **_kwargs: object) -> dict[str, object]:
        pytest.fail("probe_media must not run when the output exists")

    monkeypatch.setattr(media_service, "probe_media", forbidden_probe)

    result = await media_server.merge_narration(
        str(video), str(narration), str(output)
    )

    assert result["status"] == "error"
    assert "already exists" in str(result["error"])


@pytest.mark.asyncio
async def test_validate_tool_returns_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_validate(media_path: str, **kwargs: object) -> dict[str, object]:
        assert media_path == "final.mp4"
        assert kwargs["expected_duration_s"] == 3.0
        assert kwargs["duration_tolerance_s"] == 0.2
        return {"status": "error", "valid": False, "errors": ["No audio stream"]}

    monkeypatch.setattr(media_service, "validate_media", fake_validate)

    result = await media_server.validate_media(
        "final.mp4", expected_duration_s=3.0, duration_tolerance_s=0.2
    )

    assert result["valid"] is False
    assert result["errors"] == ["No audio stream"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        lambda: media_server.probe_media("  "),
        lambda: media_server.validate_media(""),
        lambda: media_server.merge_narration("clip.mp4", "", "final.mp4"),
    ],
)
async def test_tools_reject_empty_paths(call: Any) -> None:
    result = await call()

    assert result["status"] == "error"
    assert "non-empty string" in str(result["error"])
