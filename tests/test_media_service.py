"""Unit and small-fixture tests for the FFmpeg media service.

Real FFmpeg fixtures are tiny (64x64, sub-second) and skipped when the
binaries are unavailable. No GPU, model weights or AI workflow is involved.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import wave
from array import array
from pathlib import Path

import pytest

import media_service

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe are not on PATH"
)


def _write_sine_wav(path: Path, duration_s: float = 0.7) -> Path:
    """Write a short mono PCM WAV with only the standard library."""
    sample_rate = 24_000
    frames = int(sample_rate * duration_s)
    pcm = array(
        "h",
        (
            int(8000 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
            for index in range(frames)
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return path


def _make_av_fixture(path: Path, duration_s: float = 0.4) -> Path:
    """Render a tiny MP4 with one video and one audio stream."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x64:rate=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440",
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    return path


def _probe_result(
    duration_s: float, *, has_video: bool = True, has_audio: bool = True
) -> dict[str, object]:
    return {
        "status": "ok",
        "path": "fixture",
        "duration_s": duration_s,
        "size_bytes": 1024,
        "resolution": "64x64",
        "width": 64,
        "height": 64,
        "fps": 10.0,
        "video_codec": "h264" if has_video else None,
        "audio_codec": "aac" if has_audio else None,
        "sample_rate": 48_000 if has_audio else None,
        "channels": 2 if has_audio else None,
        "format_name": "mov,mp4",
    }


def _fake_validation(path: Path) -> dict[str, object]:
    return {
        "status": "ok",
        "valid": True,
        "path": str(path),
        "info": {},
        "checks": {"has_video": True, "has_audio": True},
        "errors": [],
        "duration_delta_s": 0.0,
    }


def _stub_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    video = tmp_path / "video.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "output.mp4"
    video.write_bytes(b"video")
    _write_sine_wav(narration, duration_s=0.1)
    return video, narration, output


# --- probe_media ----------------------------------------------------------


def test_probe_media_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="media file not found"):
        media_service.probe_media(tmp_path / "missing.mp4")


def test_probe_media_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.wav"
    empty.touch()
    with pytest.raises(ValueError, match="media file is empty"):
        media_service.probe_media(empty)


@requires_ffmpeg
def test_probe_media_rejects_invalid_media(tmp_path: Path) -> None:
    invalid = tmp_path / "broken.mp4"
    invalid.write_bytes(b"this is not a media file")
    with pytest.raises(media_service.MediaProbeError):
        media_service.probe_media(invalid)


@requires_ffmpeg
def test_probe_media_returns_stream_metadata(tmp_path: Path) -> None:
    video = _make_av_fixture(tmp_path / "clip.mp4")

    info = media_service.probe_media(video)

    assert info["status"] == "ok"
    assert info["path"] == str(video)
    assert isinstance(info["duration_s"], float) and info["duration_s"] > 0
    assert info["resolution"] == "64x64"
    assert info["fps"] == 10.0
    assert info["video_codec"] == "h264"
    assert info["audio_codec"] == "aac"
    assert info["sample_rate"] == 44_100 or info["sample_rate"] == 48_000
    assert info["channels"] == 1 or info["channels"] == 2
    assert isinstance(info["size_bytes"], int) and info["size_bytes"] > 0


# --- validate_media -------------------------------------------------------


def test_validate_media_reports_missing_file(tmp_path: Path) -> None:
    result = media_service.validate_media(tmp_path / "missing.mp4")

    assert result["status"] == "error"
    assert result["valid"] is False
    assert result["checks"] == {"exists": False}
    assert any("File not found" in str(item) for item in result["errors"])


@requires_ffmpeg
def test_validate_media_reports_invalid_media(tmp_path: Path) -> None:
    invalid = tmp_path / "broken.mp4"
    invalid.write_bytes(b"garbage")

    result = media_service.validate_media(invalid)

    assert result["valid"] is False
    assert result["checks"]["exists"] is True
    assert result["checks"]["readable"] is False
    assert result["errors"]


@requires_ffmpeg
def test_validate_media_accepts_good_file_and_checks_duration(
    tmp_path: Path,
) -> None:
    video = _make_av_fixture(tmp_path / "clip.mp4", duration_s=0.4)

    result = media_service.validate_media(
        video, expected_duration_s=0.4, duration_tolerance_s=0.1
    )

    assert result["status"] == "ok"
    assert result["valid"] is True
    assert result["checks"] == {
        "exists": True,
        "readable": True,
        "has_video": True,
        "has_audio": True,
        "resolution": True,
        "frame_rate": True,
        "audio_metadata": True,
        "duration": True,
    }
    assert isinstance(result["info"], dict)
    assert isinstance(result["duration_delta_s"], float)


@requires_ffmpeg
def test_validate_media_rejects_duration_mismatch(tmp_path: Path) -> None:
    video = _make_av_fixture(tmp_path / "clip.mp4", duration_s=0.4)

    result = media_service.validate_media(
        video, expected_duration_s=2.0, duration_tolerance_s=0.05
    )

    assert result["valid"] is False
    assert result["checks"]["duration"] is False
    assert any("differs from expected" in str(item) for item in result["errors"])


# --- merge_narration: validation -----------------------------------------


def test_merge_refuses_existing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, narration, output = _stub_inputs(tmp_path)
    output.write_bytes(b"finished video")

    def forbidden_probe(*_args: object, **_kwargs: object) -> dict[str, object]:
        pytest.fail("probe_media must not run when the output exists")

    monkeypatch.setattr(media_service, "probe_media", forbidden_probe)

    with pytest.raises(FileExistsError, match="already exists"):
        media_service.merge_narration(video, narration, output)


def test_merge_allows_explicit_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands, video, narration, output = _stub_merge(
        monkeypatch, tmp_path, video_duration=1.0, narration_duration=1.0
    )
    output.write_bytes(b"previous render")

    result = media_service.merge_narration(video, narration, output, overwrite=True)

    assert result["status"] == "merged"
    assert commands
    assert output.read_bytes() != b"previous render"


def test_merge_requires_both_input_files(tmp_path: Path) -> None:
    video, narration, output = _stub_inputs(tmp_path)

    with pytest.raises(FileNotFoundError, match="video file not found"):
        media_service.merge_narration(tmp_path / "missing.mp4", narration, output)
    with pytest.raises(FileNotFoundError, match="narration file not found"):
        media_service.merge_narration(video, tmp_path / "missing.wav", output)


def test_merge_rejects_invalid_output_and_options(tmp_path: Path) -> None:
    video, narration, output = _stub_inputs(tmp_path)

    with pytest.raises(ValueError, match="output_path must end in .mp4"):
        media_service.merge_narration(video, narration, output.with_suffix(".mov"))
    bad_narration = tmp_path / "narration.mp3"
    bad_narration.write_bytes(b"audio")
    with pytest.raises(ValueError, match="narration_path must end in .wav"):
        media_service.merge_narration(video, bad_narration, output)
    with pytest.raises(ValueError, match="Unsupported duration_policy"):
        media_service.merge_narration(
            video, narration, output, duration_policy="crossfade"
        )
    with pytest.raises(ValueError, match="volume must be"):
        media_service.merge_narration(video, narration, output, volume=-0.1)


def test_merge_requires_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, narration, output = _stub_inputs(tmp_path)
    results = iter(
        [_probe_result(1.0, has_video=False), _probe_result(1.0, has_audio=False)]
    )
    monkeypatch.setattr(
        media_service, "probe_media", lambda *_args, **_kwargs: next(results)
    )

    with pytest.raises(media_service.MediaError, match="no video stream"):
        media_service.merge_narration(video, narration, output)


# --- merge_narration: duration policies ----------------------------------


def _stub_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, video_duration: float,
    narration_duration: float,
) -> tuple[list[list[str]], Path, Path, Path]:
    video, narration, output = _stub_inputs(tmp_path)
    results = iter([_probe_result(video_duration), _probe_result(narration_duration)])
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, timeout_s: float, label: str
    ) -> str:
        assert label == "ffmpeg"
        assert timeout_s > 0
        commands.append(command)
        return ""

    monkeypatch.setattr(
        media_service, "probe_media", lambda *_args, **_kwargs: next(results)
    )
    monkeypatch.setattr(media_service, "_run_command", fake_run)
    monkeypatch.setattr(
        media_service,
        "validate_media",
        lambda path, **_kwargs: _fake_validation(Path(path)),
    )
    return commands, video, narration, output


def test_trim_audio_policy_trims_longer_narration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands, video, narration, output = _stub_merge(
        monkeypatch, tmp_path, video_duration=2.0, narration_duration=3.0
    )

    result = media_service.merge_narration(video, narration, output)

    assert result["status"] == "merged"
    assert result["audio_trimmed_s"] == pytest.approx(1.0)
    assert result["video_extended_s"] == 0.0
    assert any("trimmed by 1.000s" in str(item) for item in result["warnings"])
    graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "atrim=start=0:end=2.000000" in graph
    assert commands[0][commands[0].index("-c:v") + 1] == "copy"
    assert output.is_file()


def test_trim_audio_policy_pads_shorter_narration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands, video, narration, output = _stub_merge(
        monkeypatch, tmp_path, video_duration=3.0, narration_duration=2.0
    )

    result = media_service.merge_narration(video, narration, output)

    assert result["audio_padded_s"] == pytest.approx(1.0)
    assert result["audio_trimmed_s"] == 0.0
    assert any("Silence of 1.000s" in str(item) for item in result["warnings"])
    graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "apad=whole_dur=3.000000" in graph


def test_extend_video_policy_freezes_final_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands, video, narration, output = _stub_merge(
        monkeypatch, tmp_path, video_duration=2.0, narration_duration=3.0
    )

    result = media_service.merge_narration(
        video, narration, output, duration_policy="extend_video", volume=0.75
    )

    assert result["video_extended_s"] == pytest.approx(1.0)
    assert result["output_duration_s"] == 3.0
    assert any("frozen for 1.000s" in str(item) for item in result["warnings"])
    graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "tpad=stop_mode=clone:stop_duration=1.000000" in graph
    assert "volume=0.750000" in graph
    assert commands[0][commands[0].index("-c:v") + 1] == "libx264"
    assert output.is_file()


def test_extend_video_policy_reports_early_narration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands, video, narration, output = _stub_merge(
        monkeypatch, tmp_path, video_duration=3.0, narration_duration=2.0
    )

    result = media_service.merge_narration(
        video, narration, output, duration_policy="extend_video"
    )

    assert result["audio_ends_early_s"] == pytest.approx(1.0)
    assert result["video_extended_s"] == 0.0
    assert any("ends 1.000s before" in str(item) for item in result["warnings"])
    assert commands[0][commands[0].index("-c:v") + 1] == "copy"


# --- FFmpeg subprocess failures ------------------------------------------


def test_run_command_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=12)

    monkeypatch.setattr(media_service.subprocess, "run", timed_out)

    with pytest.raises(media_service.MediaError, match="timed out after 12s"):
        media_service._run_command(["ffmpeg"], timeout_s=12, label="ffmpeg")


def test_run_command_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=1, stdout="", stderr="Invalid filter graph"
        )

    monkeypatch.setattr(media_service.subprocess, "run", failed)

    with pytest.raises(media_service.MediaError, match="Invalid filter graph"):
        media_service._run_command(["ffmpeg"], timeout_s=5, label="ffmpeg")


def test_run_command_reports_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(media_service.subprocess, "run", missing)

    with pytest.raises(media_service.MediaError, match="not found on PATH"):
        media_service._run_command(["ffmpeg"], timeout_s=5, label="ffmpeg")


# --- End-to-end with tiny fixtures ---------------------------------------


@requires_ffmpeg
@pytest.mark.parametrize(
    ("policy", "expected_duration"),
    [("trim_audio", 0.4), ("extend_video", 0.7)],
)
def test_merge_produces_validated_mp4(
    tmp_path: Path, policy: str, expected_duration: float
) -> None:
    video = _make_av_fixture(tmp_path / "clip.mp4", duration_s=0.4)
    narration = _write_sine_wav(tmp_path / "narration.wav", duration_s=0.7)
    output = tmp_path / "nested" / "merged.mp4"

    result = media_service.merge_narration(
        video, narration, output, duration_policy=policy, volume=0.8
    )

    assert result["status"] == "merged"
    assert output.is_file()
    assert result["validation"]["valid"] is True
    assert result["warnings"], "duration difference must be reported explicitly"
    probe = media_service.probe_media(output)
    assert probe["video_codec"] == "h264"
    assert probe["audio_codec"] == "aac"
    assert isinstance(probe["duration_s"], float)
    assert probe["duration_s"] == pytest.approx(expected_duration, abs=0.25)
