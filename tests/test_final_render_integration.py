"""Real-binary smoke tests for final MP4 assembly.

Every generated file stays under ``tmp_path``.  Tests skip cleanly when
FFmpeg/FFprobe or the optional libass filter are unavailable.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

import media_service
from final_renderer import FinalRenderConfig, FinalRenderer
from project_service import ProjectService

PROJECT_ID = "demo"
RENDER_ID = "render-integration"
RENDER_RELATIVE = "video/renders/render-integration"
WIDTH = 64
HEIGHT = 48
FPS = 24.0
DURATION_S = 0.5

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe are not on PATH"
)


def _has_subtitles_filter() -> bool:
    if not FFMPEG_AVAILABLE:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and " subtitles " in result.stdout


requires_libass = pytest.mark.skipif(
    not _has_subtitles_filter(), reason="FFmpeg was built without the subtitles filter"
)


def _config(**overrides: object) -> FinalRenderConfig:
    payload: dict[str, object] = {
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "audio_sample_rate": 48000,
        "audio_channels": 2,
        "duration_tolerance_s": "0.5",
    }
    payload.update(overrides)
    return FinalRenderConfig.from_dict(payload)


def _make_project(tmp_path: Path) -> tuple[ProjectService, Path, Path, Path, Path]:
    projects = ProjectService(tmp_path / "projects")
    projects.create_project(PROJECT_ID, "Integration Demo")
    root = projects.project_dir(PROJECT_ID)
    render_dir = root / "video" / "renders" / "render-integration"
    render_dir.mkdir(parents=True, exist_ok=True)
    video = root / "video" / "final" / "silent.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    audio = root / "audio" / "master.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    return projects, root, render_dir, video, audio


def _make_silent_video(path: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={WIDTH}x{HEIGHT}:rate={FPS:g}:duration={DURATION_S}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{FPS:g}",
        str(path),
    ]
    subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=True,
    )


def _make_wav(path: Path) -> None:
    sample_rate = 48000
    channels = 2
    frame_count = int(sample_rate * DURATION_S)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            value = int(6000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            packed = value.to_bytes(2, "little", signed=True)
            frames.extend(packed * channels)
        handle.writeframes(bytes(frames))


def _assert_streams(path: Path, *, expected_duration: float) -> dict[str, object]:
    info = media_service.probe_media(path, timeout_s=30.0)
    assert info["status"] == "ok"
    assert info["video_codec"] == "h264"
    assert info["audio_codec"] == "aac"
    assert info["width"] == WIDTH
    assert info["height"] == HEIGHT
    assert info["resolution"] == f"{WIDTH}x{HEIGHT}"
    assert isinstance(info["fps"], float)
    assert abs(info["fps"] - FPS) <= 0.05
    assert info["sample_rate"] == 48000
    assert info["channels"] == 2
    assert isinstance(info["duration_s"], float)
    assert abs(info["duration_s"] - expected_duration) <= 0.5
    return info


@requires_ffmpeg
def test_final_render_with_real_ffmpeg(tmp_path: Path) -> None:
    projects, _root, _render_dir, video, audio = _make_project(tmp_path)
    _make_silent_video(video)
    _make_wav(audio)
    renderer = FinalRenderer(projects)

    result = renderer.render(
        PROJECT_ID,
        RENDER_RELATIVE,
        "video/final/silent.mp4",
        "audio/master.wav",
        render_id=RENDER_ID,
        config=_config(),
    )

    output = _root_output(tmp_path)
    assert output.is_file()
    assert result.output_path == f"{RENDER_RELATIVE}/final.mp4"
    assert result.size_bytes == output.stat().st_size
    assert result.burn_in is False
    assert result.duration_s == pytest.approx(DURATION_S)
    _assert_streams(output, expected_duration=DURATION_S)


def _root_output(tmp_path: Path) -> Path:
    return tmp_path / "projects" / PROJECT_ID / "video" / "renders" / RENDER_ID / "final.mp4"


@requires_ffmpeg
@requires_libass
def test_final_render_with_srt_burn_in(tmp_path: Path) -> None:
    projects, root, _render_dir, video, audio = _make_project(tmp_path)
    _make_silent_video(video)
    _make_wav(audio)
    subtitle = root / "subtitles" / "burn.srt"
    subtitle.parent.mkdir(parents=True, exist_ok=True)
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:00,400\nBURN-IN TEST\n",
        encoding="utf-8",
    )
    escaped = str(subtitle).replace("\\", "/").replace(":", "\\:")
    burn_in_filter = f"subtitles=filename='{escaped}'"
    renderer = FinalRenderer(projects)

    result = renderer.render(
        PROJECT_ID,
        RENDER_RELATIVE,
        "video/final/silent.mp4",
        "audio/master.wav",
        render_id=RENDER_ID,
        config=_config(subtitle_burn_in=True),
        subtitle_path="subtitles/burn.srt",
        burn_in_filter=burn_in_filter,
    )

    output = _root_output(tmp_path)
    assert result.burn_in is True
    assert result.subtitle_path == "subtitles/burn.srt"
    assert "-vf" in result.command
    assert burn_in_filter in result.command
    assert output.is_file()
    _assert_streams(output, expected_duration=DURATION_S)
