"""Deterministic tests for staged audio mixing and master validation."""

from __future__ import annotations

import wave
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from audio_mixer import AudioMixer
from audio_models import (
    AudioArrangement,
    MusicSettings,
    ResolvedAudioAsset,
    SfxSettings,
)
from audio_timeline import AudioSceneTiming, AudioTimeline, AudioTimelinePolicy
from project_service import ProjectService


def _write_wav(
    path: Path, *, frames: int = 48000, sample_rate: int = 48000, channels: int = 2
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames * channels)


def _fixture(tmp_path: Path) -> tuple[ProjectService, Path]:
    projects = ProjectService(tmp_path / "projects")
    projects.create_project("demo", "Demo")
    root = projects.project_dir("demo")
    _write_wav(root / "assets/audio/narration/one.wav", frames=24000, sample_rate=24000, channels=1)
    return projects, root


def _timeline(*, narration: bool = True) -> AudioTimeline:
    timing = AudioSceneTiming(
        scene_id="one",
        order=0,
        title="One",
        video_start_s=Decimal(0),
        video_end_s=Decimal(1),
        video_duration_s=Decimal(1),
        narration_start_s=Decimal(0),
        narration_end_s=Decimal(1) if narration else Decimal(0),
        narration_duration_s=Decimal(1) if narration else None,
        narration_overflow_s=Decimal(0),
        final_frame_hold_s=Decimal(0),
        silent_tail_s=Decimal(0),
        effective_duration_s=Decimal(1),
        narration_present=narration,
        narration_path="assets/audio/narration/one.wav" if narration else None,
    )
    return AudioTimeline(
        project_id="demo",
        policy=AudioTimelinePolicy(),
        scenes=(timing,),
        total_duration_s=Decimal(1),
    )


def _arrangement(root: Path, *, assets: bool = False) -> AudioArrangement:
    if not assets:
        return AudioArrangement(project_id="demo", total_duration_s=Decimal(1))

    music_path = root / "assets/audio/bed.wav"
    sfx_path = root / "assets/sfx/whoosh.wav"
    _write_wav(music_path, frames=24000)
    _write_wav(sfx_path, frames=48000)
    music_settings = MusicSettings.from_dict(
        {
            "path": "assets/audio/bed.wav",
            "volume": 0.2,
            "loop": True,
            "ducking": {"level": 0.25, "attack_ms": 100, "release_ms": 400},
        }
    )
    sfx_settings = SfxSettings.from_dict(
        {
            "path": "assets/sfx/whoosh.wav",
            "scene_id": "one",
            "start_offset_s": "0.2",
            "volume": 0.7,
        }
    )
    music = ResolvedAudioAsset(
        settings=music_settings,
        relative_path="assets/audio/bed.wav",
        absolute_path=str(music_path),
        duration_s=Decimal(1),
        start_s=Decimal(0),
        trimmed_duration_s=Decimal(1),
        sample_rate=48000,
        channels=2,
    )
    sfx = ResolvedAudioAsset(
        settings=sfx_settings,
        relative_path="assets/sfx/whoosh.wav",
        absolute_path=str(sfx_path),
        duration_s=Decimal(1),
        start_s=Decimal("0.2"),
        trimmed_duration_s=Decimal("0.8"),
        sample_rate=24000,
        channels=1,
    )
    return AudioArrangement(
        project_id="demo", total_duration_s=Decimal(1), music=music, sfx=(sfx,)
    )


def _runner(commands: list[list[str]]) -> Any:
    def run(command: list[str], timeout_s: float) -> None:
        commands.append(list(command))
        output = Path(command[-1])
        _write_wav(output)

    return run


def _probe(path: str | Path, *, timeout_s: float) -> Mapping[str, object]:
    return {
        "status": "ok",
        "path": str(path),
        "audio_codec": "pcm_s16le",
        "sample_rate": 48000,
        "channels": 2,
        "duration_s": 1.0,
        "size_bytes": 192000,
    }


def _mixer(
    projects: ProjectService,
    commands: list[list[str]],
    *,
    probe: Any = _probe,
) -> AudioMixer:
    return AudioMixer(
        projects,
        command_runner=_runner(commands),
        probe=probe,
        check_executables=False,
    )


def test_mix_builds_narration_and_final_buses(tmp_path: Path) -> None:
    projects, root = _fixture(tmp_path)
    commands: list[list[str]] = []
    mixer = _mixer(projects, commands)

    result = mixer.mix(
        "demo",
        _timeline(),
        _arrangement(root),
        {"render_id": "basic"},
    )

    assert result.render_id == "mix-basic"
    master_path = Path(result.master_path)
    assert master_path.parent.name == "mix-basic"
    assert master_path.name == "audio_master.wav"
    assert master_path.is_file()
    assert result.master_duration_s == Decimal(1)
    assert result.sample_rate == 48000
    assert result.channels == 2
    assert result.sample_width_bytes == 2
    assert result.narration_inputs == 1
    assert result.music_inputs == 0
    assert result.sfx_inputs == 0
    assert result.ducking_applied is False
    assert len(result.commands) == 2

    narration_command = commands[0]
    assert "narration_bus.wav" in narration_command[-1]
    assert "adelay=0:all=1" in narration_command[narration_command.index("-filter_complex") + 1]
    final_command = commands[1]
    assert "alimiter=" in final_command[final_command.index("-filter_complex") + 1]
    assert "apad=whole_dur=1.000000" in final_command[final_command.index("-filter_complex") + 1]
    assert "atrim=end=1.000000" in final_command[final_command.index("-filter_complex") + 1]
    assert list(Path(result.master_path).parent.glob(".audio_master.*.tmp.wav")) == []


def test_mix_without_narration_reports_no_fabricated_speech(tmp_path: Path) -> None:
    projects, _root = _fixture(tmp_path)
    commands: list[list[str]] = []
    mixer = _mixer(projects, commands)

    result = mixer.mix("demo", _timeline(narration=False), _arrangement(_root), {})

    assert len(commands) == 1
    assert any("anullsrc" in part for part in commands[0])
    assert result.narration_inputs == 0
    assert result.master_path.endswith("audio_master.wav")
    assert {issue.code for issue in result.warnings} == {"audio.mix.no_narration"}


def test_mix_builds_ducked_music_and_sfx_buses(tmp_path: Path) -> None:
    projects, root = _fixture(tmp_path)
    commands: list[list[str]] = []
    mixer = _mixer(projects, commands)

    result = mixer.mix(
        "demo",
        _timeline(),
        _arrangement(root, assets=True),
        {"render_id": "ducked"},
    )

    assert len(commands) == 4
    assert result.music_inputs == 1
    assert result.sfx_inputs == 1
    assert result.ducking_applied is True

    music_filter = commands[1][commands[1].index("-filter_complex") + 1]
    assert "-stream_loop" in commands[1]
    assert "sidechaincompress=threshold=0.0200:ratio=4.000" in music_filter
    assert "attack=100.00:release=400.00:mix=1" in music_filter
    assert "[1:a]aresample=48000,apad=whole_dur=1.000000,atrim=end=1.000000" in music_filter

    sfx_filter = commands[2][commands[2].index("-filter_complex") + 1]
    assert "adelay=200:all=1" in sfx_filter
    assert commands[3].count("-i") == 3


def test_mix_rejects_render_directory_collision(tmp_path: Path) -> None:
    projects, root = _fixture(tmp_path)
    mixer = _mixer(projects, [])

    mixer.mix("demo", _timeline(), _arrangement(root), {"render_id": "reuse"})

    with pytest.raises(Exception) as raised:
        mixer.mix("demo", _timeline(), _arrangement(root), {"render_id": "reuse"})

    assert getattr(raised.value, "code", None) == "audio.render.collision"


def test_mix_rejects_invalid_master_without_promoting_it(tmp_path: Path) -> None:
    projects, root = _fixture(tmp_path)

    def bad_probe(path: str | Path, *, timeout_s: float) -> Mapping[str, object]:
        return {**_probe(path, timeout_s=timeout_s), "sample_rate": 44100}

    mixer = _mixer(projects, [], probe=bad_probe)

    with pytest.raises(Exception) as raised:
        mixer.mix("demo", _timeline(), _arrangement(root), {"render_id": "badmaster"})

    assert getattr(raised.value, "code", None) == "audio.mix.invalid_output"
    render_dir = root / "audio/renders/mix-badmaster"
    assert render_dir.is_dir()
    assert not (render_dir / "audio_master.wav").exists()


def test_mix_render_wraps_errors_in_status(tmp_path: Path) -> None:
    projects, root = _fixture(tmp_path)
    mixer = _mixer(projects, [], probe=_probe)

    payload = mixer.render("demo", _timeline(), _arrangement(root), {"render_id": "status"})

    assert payload["status"] == "ok"
    result = payload["result"]
    assert isinstance(result, dict)
    assert result["project_id"] == "demo"

    empty_projects = ProjectService(tmp_path / "empty-projects")
    empty_mixer = _mixer(empty_projects, [], probe=_probe)
    payload = empty_mixer.render("demo", _timeline(), _arrangement(root), {})

    assert payload["status"] == "error"
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "audio.project.not_found"
