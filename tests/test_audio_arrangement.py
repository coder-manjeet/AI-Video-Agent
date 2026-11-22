"""Deterministic tests for music/SFX registration and timeline arrangement."""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from audio_arrangement import AudioArrangementService, AudioAssetRegistry
from audio_models import (
    DuckingConfig,
    MusicSettings,
    SfxSettings,
    ValidationIssue,
)
from audio_timeline import AudioSceneTiming, AudioTimeline, AudioTimelinePolicy
from project_service import ProjectService
from scene_service import SceneService


def _timeline() -> AudioTimeline:
    scenes = (
        AudioSceneTiming(
            scene_id="one",
            order=0,
            title="One",
            video_start_s=Decimal(0),
            video_end_s=Decimal(5),
            video_duration_s=Decimal(5),
            narration_start_s=Decimal(0),
            narration_end_s=Decimal(5),
            narration_duration_s=Decimal(5),
            narration_overflow_s=Decimal(0),
            final_frame_hold_s=Decimal(0),
            silent_tail_s=Decimal(0),
            effective_duration_s=Decimal(5),
            narration_present=True,
            narration_path="assets/audio/narration/one.wav",
        ),
        AudioSceneTiming(
            scene_id="two",
            order=1,
            title="Two",
            video_start_s=Decimal(5),
            video_end_s=Decimal(10),
            video_duration_s=Decimal(5),
            narration_start_s=Decimal(5),
            narration_end_s=Decimal(10),
            narration_duration_s=Decimal(5),
            narration_overflow_s=Decimal(0),
            final_frame_hold_s=Decimal(0),
            silent_tail_s=Decimal(0),
            effective_duration_s=Decimal(5),
            narration_present=True,
            narration_path="assets/audio/narration/two.wav",
        ),
    )
    return AudioTimeline(
        project_id="demo",
        policy=AudioTimelinePolicy(),
        scenes=scenes,
        total_duration_s=Decimal(10),
    )


def _fixture(tmp_path: Path) -> tuple[ProjectService, SceneService, Path, AudioAssetRegistry]:
    projects = ProjectService(tmp_path / "projects")
    projects.create_project("demo", "Demo")
    scenes = SceneService(projects)
    for scene_id in ("one", "two"):
        scenes.add_scene("demo", title=scene_id, scene_id=scene_id, duration=5.0)
    root = projects.project_dir("demo")
    for relative in ("assets/audio/bed.wav", "assets/sfx/whoosh.wav", "assets/audio/bed.txt"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")
    return projects, scenes, root, AudioAssetRegistry(projects, scenes)


def _probe(path: str | Path, *, timeout_s: float) -> Mapping[str, object]:
    name = Path(path).name
    durations = {"bed.wav": 8.0, "whoosh.wav": 2.0, "bed.txt": 8.0}
    return {
        "status": "ok",
        "audio_codec": "pcm_s16le",
        "duration_s": durations.get(name, 1.0),
        "sample_rate": 48000 if name == "bed.wav" else 24000,
        "channels": 2 if name == "bed.wav" else 1,
    }


def _music(**overrides: object) -> MusicSettings:
    payload: dict[str, object] = {
        "path": "assets/audio/bed.wav",
        "volume": 0.2,
        "start_offset_s": "1",
        "trim_start_s": "0.5",
        "trim_end_s": "0.5",
        "fade_in_s": "0.1",
        "fade_out_s": "0.2",
        "loop": True,
        "ducking": DuckingConfig.from_dict({"level": 0.25}),
    }
    payload.update(overrides)
    return MusicSettings.from_dict(payload)


def _sfx(**overrides: object) -> SfxSettings:
    payload: dict[str, object] = {
        "path": "assets/sfx/whoosh.wav",
        "scene_id": "two",
        "start_offset_s": "0.5",
        "volume": 0.8,
        "trim_start_s": "0.1",
        "fade_in_s": "0.05",
    }
    payload.update(overrides)
    return SfxSettings.from_dict(payload)


def test_registry_writes_and_loads_project_and_scene_settings(tmp_path: Path) -> None:
    _projects, scenes, root, registry = _fixture(tmp_path)

    registry.set_music("demo", _music())
    registry.add_sfx("demo", "two", _sfx())

    project_payload = json.loads((root / "project.json").read_text(encoding="utf-8"))
    extra = project_payload["extra"]
    assert isinstance(extra, dict)
    music_payload = extra["audio_music"]
    assert isinstance(music_payload, dict)
    assert music_payload["loop"] is True

    scene = scenes.get_scene("demo", "two")
    assert "assets/sfx/whoosh.wav" in scene.sfx_paths
    sfx_entries = scene.extra["audio_sfx"]
    assert isinstance(sfx_entries, list)
    assert sfx_entries[0]["start_offset_s"] == "0.5"

    loaded_music, loaded_sfx = registry.load("demo")
    assert loaded_music is not None
    assert loaded_music.path == "assets/audio/bed.wav"
    assert len(loaded_sfx) == 1
    assert loaded_sfx[0].scene_id == "two"


def test_build_resolves_music_and_scene_relative_sfx(tmp_path: Path) -> None:
    projects, _scenes, _root, registry = _fixture(tmp_path)
    service = AudioArrangementService(projects, registry, probe=_probe)

    arrangement = service.build(
        "demo",
        _timeline(),
        _music(),
        [_sfx()],
        include_registered=False,
    )

    assert arrangement.project_id == "demo"
    assert arrangement.total_duration_s == Decimal(10)
    assert arrangement.music is not None
    assert arrangement.music.start_s == Decimal(1)
    assert arrangement.music.trimmed_duration_s == Decimal(7)
    assert arrangement.music.sample_rate == 48000
    assert len(arrangement.sfx) == 1
    assert arrangement.sfx[0].start_s == Decimal("5.5")
    assert arrangement.sfx[0].trimmed_duration_s == Decimal("1.9")
    assert arrangement.warnings == ()

    payload = arrangement.to_dict()
    assert payload["music"]["trimmed_duration_s"] == "7.0"
    assert payload["sfx"][0]["start_s"] == "5.5"


def test_build_warns_for_short_unlooped_music_without_ducking(tmp_path: Path) -> None:
    projects, _scenes, _root, registry = _fixture(tmp_path)
    service = AudioArrangementService(projects, registry, probe=_probe)

    arrangement = service.build(
        "demo",
        _timeline(),
        _music(loop=False, ducking=None),
        include_registered=False,
    )

    codes = {issue.code for issue in arrangement.warnings}
    assert "audio.arrangement.music_short" in codes
    assert "audio.arrangement.ducking_absent" in codes


@pytest.mark.parametrize(
    ("music", "sfx", "code"),
    [
        (_music(path="assets/audio/missing.wav"), (), "audio.asset.missing_file"),
        (_music(path="../bed.wav"), (), "audio.asset.unsafe_path"),
        (_music(path="assets/audio/bed.txt"), (), "audio.asset.invalid_format"),
        (
            _music(trim_start_s="4", trim_end_s="4"),
            (),
            "audio.asset.invalid_trim",
        ),
        (
            _music(fade_in_s="4", fade_out_s="4"),
            (),
            "audio.asset.invalid_fade",
        ),
        (None, (_sfx(scene_id="missing"),), "audio.sfx.unknown_scene"),
        (
            None,
            (_sfx(start_offset_s="9.5"),),
            "audio.sfx.exceeds_timeline",
        ),
    ],
)
def test_build_rejects_invalid_assets(
    tmp_path: Path,
    music: MusicSettings | None,
    sfx: tuple[SfxSettings, ...],
    code: str,
) -> None:
    projects, _scenes, _root, registry = _fixture(tmp_path)
    service = AudioArrangementService(projects, registry, probe=_probe)

    with pytest.raises(Exception) as raised:
        service.build("demo", _timeline(), music, sfx, include_registered=False)

    assert getattr(raised.value, "code", None) == code


def test_build_rejects_probe_without_audio_stream(tmp_path: Path) -> None:
    projects, _scenes, _root, registry = _fixture(tmp_path)

    def bad_probe(path: str | Path, *, timeout_s: float) -> dict[str, object]:
        return {"status": "ok", "audio_codec": None, "duration_s": 8.0}

    service = AudioArrangementService(projects, registry, probe=bad_probe)

    with pytest.raises(Exception) as raised:
        service.build("demo", _timeline(), _music(), include_registered=False)

    assert getattr(raised.value, "code", None) == "audio.asset.no_audio_stream"


def test_resolve_trimmed_duration_rejects_overlapping_fades() -> None:
    settings = _music(trim_start_s="0", trim_end_s="0", fade_in_s="4", fade_out_s="5")

    with pytest.raises(Exception) as raised:
        from audio_arrangement import resolve_trimmed_duration

        resolve_trimmed_duration(settings, Decimal(8), "audio_music")

    assert getattr(raised.value, "code", None) == "audio.asset.invalid_fade"


def test_arrangement_rejects_invalid_warning_entries() -> None:
    with pytest.raises(TypeError, match="ValidationIssue"):
        from audio_models import AudioArrangement

        AudioArrangement(
            project_id="demo",
            total_duration_s=Decimal(1),
            warnings=("bad",),  # type: ignore[arg-type]
        )

    assert isinstance(ValidationIssue("code", "message").to_dict()["code"], str)


def test_service_rejects_non_numeric_probe_timeout(tmp_path: Path) -> None:
    projects, _scenes, _root, registry = _fixture(tmp_path)

    with pytest.raises(TypeError, match="ffprobe_timeout_s"):
        AudioArrangementService(projects, registry, probe=_probe, ffprobe_timeout_s="bad")  # type: ignore[arg-type]


def test_probe_callable_receives_timeout(tmp_path: Path) -> None:
    projects, _scenes, _root, registry = _fixture(tmp_path)
    calls: list[dict[str, Any]] = []

    def recording_probe(path: str | Path, *, timeout_s: float) -> Mapping[str, object]:
        calls.append({"path": str(path), "timeout_s": timeout_s})
        return _probe(path, timeout_s=timeout_s)

    service = AudioArrangementService(
        projects, registry, probe=recording_probe, ffprobe_timeout_s=12.5
    )
    service.build("demo", _timeline(), _music(), include_registered=False)

    assert calls and calls[0]["timeout_s"] == 12.5
