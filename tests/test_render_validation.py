"""Deterministic tests for scene and final render validation profiles."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

from manifest_validation import ValidationResult
from render_models import RenderExpectations
from render_validation import RenderValidator


def _probe_info(
    *,
    duration_s: object = 0.4,
    has_video: bool = True,
    has_audio: bool = False,
    width: int | None = 64,
    height: int | None = 64,
    fps: float | None = 10.0,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "status": status,
        "path": "fixture",
        "duration_s": duration_s,
        "size_bytes": 1024,
        "width": width,
        "height": height,
        "fps": fps,
        "video_codec": "h264" if has_video else None,
        "audio_codec": "aac" if has_audio else None,
        "sample_rate": 48000 if has_audio else None,
        "channels": 2 if has_audio else None,
        "format_name": "mov,mp4",
    }


def _validator(info: Mapping[str, object]) -> RenderValidator:
    def fake_probe(_path: Path) -> Mapping[str, object]:
        return info

    return RenderValidator(probe=fake_probe)


def _clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    return path


def _codes(result: ValidationResult) -> set[str]:
    return {issue.code for issue in result.errors}


def test_silent_scene_video_passes_scene_validation(tmp_path: Path) -> None:
    path = _clip(tmp_path)

    result = _validator(_probe_info(has_audio=False)).validate_scene_video(path)

    assert result.valid is True
    assert result.errors == []
    assert result.to_dict()["valid"] is True


def test_video_with_audio_passes_scene_validation(tmp_path: Path) -> None:
    path = _clip(tmp_path)

    result = _validator(_probe_info(has_audio=True)).validate_scene_video(path)

    assert result.valid is True
    assert result.errors == []


def test_missing_video_stream_fails_scene_validation(tmp_path: Path) -> None:
    path = _clip(tmp_path)

    result = _validator(_probe_info(has_video=False)).validate_scene_video(path)

    assert result.valid is False
    assert "render.video.missing_stream" in _codes(result)


def test_invalid_duration_fails_scene_validation(tmp_path: Path) -> None:
    path = _clip(tmp_path)

    result = _validator(_probe_info(duration_s=None)).validate_scene_video(path)

    assert result.valid is False
    assert "render.video.invalid_duration" in _codes(result)


def test_missing_audio_fails_final_validation_by_default(tmp_path: Path) -> None:
    path = _clip(tmp_path)

    result = _validator(_probe_info(has_audio=False)).validate_final(path)

    assert result.valid is False
    assert "render.final.missing_audio" in _codes(result)


def test_final_validation_can_explicitly_allow_silent_video(tmp_path: Path) -> None:
    path = _clip(tmp_path)

    result = _validator(_probe_info(has_audio=False)).validate_final(
        path, require_audio=False
    )

    assert result.valid is True


def test_final_video_with_valid_audio_passes(tmp_path: Path) -> None:
    path = _clip(tmp_path)

    result = _validator(_probe_info(has_audio=True)).validate_final(path)

    assert result.valid is True
    assert result.errors == []


def test_resolution_mismatch_is_reported(tmp_path: Path) -> None:
    path = _clip(tmp_path)
    expected = RenderExpectations(width=1920, height=1080)

    result = _validator(_probe_info()).validate_final(path, expected)

    assert result.valid is False
    assert "render.video.resolution_mismatch" in _codes(result)


def test_fps_mismatch_is_reported(tmp_path: Path) -> None:
    path = _clip(tmp_path)
    expected = RenderExpectations(fps=30.0)

    result = _validator(_probe_info()).validate_final(path, expected)

    assert result.valid is False
    assert "render.video.fps_mismatch" in _codes(result)


def test_fps_tolerance_is_honoured(tmp_path: Path) -> None:
    path = _clip(tmp_path)
    expected = RenderExpectations(fps=30.0, fps_tolerance=25.0)

    result = _validator(_probe_info(fps=10.0, has_audio=True)).validate_final(
        path, expected
    )

    assert result.valid is True


def test_duration_outside_tolerance_is_reported(tmp_path: Path) -> None:
    path = _clip(tmp_path)
    expected = RenderExpectations(
        expected_duration_s=Decimal("2.0"),
        duration_tolerance_s=Decimal("0.05"),
    )

    result = _validator(_probe_info(duration_s=0.4)).validate_final(path, expected)

    assert result.valid is False
    assert "render.video.duration_mismatch" in _codes(result)


def test_missing_file_is_handled_cleanly(tmp_path: Path) -> None:
    result = _validator(_probe_info()).validate_final(tmp_path / "missing.mp4")

    assert result.valid is False
    assert "render.file.missing" in _codes(result)


def test_empty_file_is_handled_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "empty.mp4"
    path.touch()

    result = _validator(_probe_info()).validate_scene_video(path)

    assert result.valid is False
    assert "render.file.empty" in _codes(result)


def test_probe_failure_is_reported_as_structured_issue(tmp_path: Path) -> None:
    def broken_probe(_path: Path) -> Mapping[str, object]:
        raise RuntimeError("ffprobe missing")

    result = RenderValidator(probe=broken_probe).validate_scene_video(_clip(tmp_path))

    assert result.valid is False
    assert "render.media.unreadable" in _codes(result)


def test_unexpected_probe_status_is_reported(tmp_path: Path) -> None:
    result = _validator(_probe_info(status="error")).validate_final(_clip(tmp_path))

    assert result.valid is False
    assert "render.media.unreadable" in _codes(result)
