"""Small real-FFmpeg integration test for the complete audio production flow."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

import media_service
from audio_arrangement import AudioAssetRegistry
from audio_models import DuckingConfig, MusicSettings, SfxSettings
from audio_production import AudioProductionCoordinator
from project_service import ProjectService
from scene_service import SceneService

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe are not on PATH"
)


def _make_tiny_clip(path: Path, *, seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=10",
        "-t",
        f"{seconds:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    return path


def _write_wav(
    path: Path,
    *,
    frames: int,
    sample_rate: int,
    channels: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames * channels)
    return path


@requires_ffmpeg
def test_integration_produces_held_video_and_synchronized_master(tmp_path: Path) -> None:
    projects = ProjectService(tmp_path / "projects")
    projects.create_project("integration", "Integration")
    scenes = SceneService(projects)
    root = projects.project_dir("integration")

    originals: dict[str, bytes] = {}
    for scene_id, video_seconds, narration_seconds in (
        ("one", 0.5, 0.8),
        ("two", 0.8, 0.4),
    ):
        scenes.add_scene(
            "integration",
            title=scene_id,
            scene_id=scene_id,
            duration=video_seconds,
            narration="Narration text",
        )
        clip = _make_tiny_clip(
            root / "video" / "scenes" / f"{scene_id}.mp4",
            seconds=video_seconds,
        )
        narration = _write_wav(
            root / "assets" / "audio" / "narration" / f"{scene_id}.wav",
            frames=int(24000 * narration_seconds),
            sample_rate=24000,
            channels=1,
        )
        scenes.update_scene(
            "integration",
            scene_id,
            {
                "generated_video_path": f"video/scenes/{clip.name}",
                "narration_audio_path": (
                    f"assets/audio/narration/{narration.name}"
                ),
            },
        )
        originals[f"video/scenes/{clip.name}"] = clip.read_bytes()

    music = _write_wav(
        root / "assets" / "audio" / "music.wav",
        frames=48000,
        sample_rate=48000,
        channels=2,
    )
    sfx = _write_wav(
        root / "assets" / "sfx" / "chime.wav",
        frames=4800,
        sample_rate=24000,
        channels=1,
    )
    registry = AudioAssetRegistry(projects, scenes)
    registry.set_music(
        "integration",
        MusicSettings(
            path="assets/audio/music.wav",
            volume=0.15,
            loop=True,
            ducking=DuckingConfig.from_dict({"level": 0.25}),
        ),
    )
    registry.add_sfx(
        "integration",
        "two",
        SfxSettings(path="assets/sfx/chime.wav", scene_id="two", start_offset_s="0.1"),
    )

    manifest_before = {
        "project.json": (root / "project.json").read_bytes(),
        "script/scenes.json": (root / "script" / "scenes.json").read_bytes(),
    }
    config: dict[str, object] = {
        "preparation": {
            "width": 64,
            "height": 48,
            "fps": 10,
            "preset": "ultrafast",
            "crf": 28,
            "render_root": "audio/renders",
            "ffmpeg_timeout_s": 60,
            "ffprobe_timeout_s": 30,
        },
        "timeline": {"missing_narration_policy": "allow"},
        "mix": {"duration_tolerance_s": "0.5"},
    }
    coordinator = AudioProductionCoordinator(projects, scenes)

    report = coordinator.produce("integration", config, render_id="integration")

    assert report.render_id == "audio-integration"
    assert report.scene_order == ("one", "two")
    assert float(report.hold_durations_s["one"]) == pytest.approx(0.3, abs=0.15)
    assert report.mix.ducking_applied is True
    assert any("sidechaincompress" in command.command[command.command.index("-filter_complex") + 1] for command in report.mix.commands if "-filter_complex" in command.command)

    render_dir = root / "audio/renders/audio-integration"
    silent_video = Path(report.silent_video_path)
    master = Path(report.audio_master_path)
    report_file = render_dir / "report.json"
    assert silent_video == render_dir / "final_silent.mp4"
    assert master == render_dir / "audio_master.wav"
    assert silent_video.is_file() and silent_video.stat().st_size > 0
    assert master.is_file() and master.stat().st_size > 0
    assert report_file.is_file()

    video_info = media_service.probe_media(silent_video)
    assert video_info["width"] == 64
    assert video_info["height"] == 48
    assert video_info["fps"] == pytest.approx(10.0, abs=0.05)
    assert video_info["video_codec"] == "h264"
    assert video_info["audio_codec"] is None
    assert isinstance(video_info["duration_s"], float)
    assert video_info["duration_s"] == pytest.approx(
        float(report.silent_video_duration_s), abs=0.1
    )

    prepared_one = render_dir / "scene_0000_one.mp4"
    prepared_info = media_service.probe_media(prepared_one)
    source_info = media_service.probe_media(root / "video/scenes/one.mp4")
    assert isinstance(prepared_info["duration_s"], float)
    assert isinstance(source_info["duration_s"], float)
    assert prepared_info["duration_s"] > source_info["duration_s"]

    audio_info = media_service.probe_media(master)
    assert audio_info["audio_codec"] == "pcm_s16le"
    assert audio_info["sample_rate"] == 48000
    assert audio_info["channels"] == 2
    assert isinstance(audio_info["duration_s"], float)
    assert audio_info["duration_s"] == pytest.approx(
        float(report.audio_duration_s), abs=0.05
    )
    assert audio_info["duration_s"] == pytest.approx(
        video_info["duration_s"], abs=0.5
    )

    assert report.validation["manifest_preserved"] is True
    assert report.validation["timeline_consistency"] == "ok"
    assert (root / "project.json").read_bytes() == manifest_before["project.json"]
    assert (root / "script" / "scenes.json").read_bytes() == manifest_before["script/scenes.json"]
    for relative, original in originals.items():
        assert (root / relative).read_bytes() == original

    payload = report.to_dict()
    assert payload["scene_order"] == ["one", "two"]
    assert payload["mix"]["ducking_applied"] is True
    assert music.is_file() and sfx.is_file()
