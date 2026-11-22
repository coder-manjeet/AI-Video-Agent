"""Deterministic tests for the audio/video production coordinator."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import scene_preparation_service as sps
from audio_models import (
    AudioArrangement,
    AudioMixConfig,
    AudioMixResult,
    AudioProductionReport,
)
from audio_production import AudioProductionCoordinator
from project_service import ProjectService
from render_models import (
    RenderMediaInfo,
    RenderPlan,
    RenderPolicy,
    RenderScene,
)
from scene_service import SceneService

PROJECT_ID = "demo"


def _media_info(path: str, duration: Decimal) -> RenderMediaInfo:
    return RenderMediaInfo(
        path=path,
        duration_s=duration,
        size_bytes=2048,
        width=64,
        height=48,
        fps=10.0,
        video_codec="h264",
        audio_codec=None,
        format_name="mov,mp4",
    )


def _scene(
    scene_id: str,
    *,
    number: int,
    measured: Decimal,
    narration: Decimal,
    start: Decimal,
    end: Decimal,
) -> RenderScene:
    hold = max(Decimal(0), narration - measured)
    effective = max(measured, narration)
    video_path = f"video/scenes/{scene_id}.mp4"
    return RenderScene(
        scene_id=scene_id,
        scene_number=number,
        title=scene_id.title(),
        status="done",
        planned_duration_s=measured,
        measured_duration_s=measured,
        effective_duration_s=effective,
        video_hold_duration_s=hold,
        silent_tail_duration_s=max(Decimal(0), measured - narration),
        timeline_start_s=start,
        timeline_end_s=end,
        planned_timeline_start_s=start,
        planned_timeline_end_s=end,
        duration_delta_s=Decimal(0),
        narration_duration_s=narration,
        narration_delta_s=narration - measured,
        narration_audio_path=f"assets/audio/narration/{scene_id}.wav",
        narration_text="Narration text",
        generated_video_path=video_path,
        video=_media_info(video_path, measured),
        narration=RenderMediaInfo(
            path=f"assets/audio/narration/{scene_id}.wav",
            duration_s=narration,
            audio_codec="pcm_s16le",
            sample_rate=24000,
            channels=1,
        ),
    )


def _plan() -> RenderPlan:
    scenes = (
        _scene(
            "one",
            number=1,
            measured=Decimal(1),
            narration=Decimal(1),
            start=Decimal(0),
            end=Decimal(1),
        ),
        _scene(
            "two",
            number=2,
            measured=Decimal(1),
            narration=Decimal("1.5"),
            start=Decimal(1),
            end=Decimal("2.5"),
        ),
    )
    return RenderPlan(
        project_id=PROJECT_ID,
        policy=RenderPolicy(require_audio=False),
        scenes=scenes,
        warnings=(),
        total_planned_duration_s=Decimal(2),
        total_render_duration_s=Decimal("2.5"),
        output_width=64,
        output_height=48,
        output_fps=10.0,
    )


class _Planner:
    def __init__(self, plan: RenderPlan) -> None:
        self.plan = plan

    def build_plan(self, project_id: str, policy: RenderPolicy) -> RenderPlan:
        assert project_id == PROJECT_ID
        return self.plan


class _ArrangementService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def build(
        self,
        project_id: str,
        timeline: Any,
        music: Any,
        sfx: Sequence[Any],
        *,
        include_registered: bool,
    ) -> AudioArrangement:
        self.calls.append(
            {
                "project_id": project_id,
                "timeline": timeline,
                "include_registered": include_registered,
            }
        )
        return AudioArrangement(
            project_id=project_id,
            total_duration_s=timeline.total_duration_s,
        )


class _Mixer:
    def __init__(self, root: Path, *, duration: Decimal = Decimal("2.5")) -> None:
        self.root = root
        self.duration = duration
        self.calls: list[AudioMixConfig] = []

    def mix(
        self,
        project_id: str,
        timeline: Any,
        arrangement: AudioArrangement,
        config: AudioMixConfig | Mapping[str, Any],
    ) -> AudioMixResult:
        assert isinstance(config, AudioMixConfig)
        self.calls.append(config)
        master = (
            self.root
            / config.render_root
            / str(config.render_id)
            / config.master_filename
        )
        return AudioMixResult(
            project_id=project_id,
            render_id=str(config.render_id),
            render_dir=str(master.parent),
            master_path=str(master),
            master_duration_s=self.duration,
            sample_rate=config.sample_rate,
            channels=config.channels,
            sample_width_bytes=config.sample_width_bytes,
            narration_inputs=2,
            music_inputs=0,
            sfx_inputs=0,
            ducking_applied=False,
            peak=0.1,
            clipped_samples=0,
        )


class _Validator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def validate_final(
        self,
        path: str | Path,
        expectations: Any,
        require_audio: bool,
    ) -> Any:
        from manifest_validation import ValidationResult

        self.calls.append(
            {
                "path": Path(path),
                "expectations": expectations,
                "require_audio": require_audio,
            }
        )
        return ValidationResult()


def _config() -> dict[str, object]:
    return {
        "preparation": {
            "width": 64,
            "height": 48,
            "fps": 10,
            "render_root": "audio/renders",
            "ffmpeg_bin": "ffmpeg",
            "ffprobe_bin": "ffprobe",
        },
        "timeline": {"missing_narration_policy": "allow"},
        "mix": {"render_root": "audio/renders"},
    }


def _harness(
    tmp_path: Path,
    *,
    mix_duration: Decimal = Decimal("2.5"),
    mutate_manifest: bool = False,
) -> tuple[AudioProductionCoordinator, Path, dict[str, bytes], list[list[str]], _Mixer, _Validator]:
    projects = ProjectService(tmp_path / "projects")
    projects.create_project(PROJECT_ID, "Demo")
    scenes = SceneService(projects)
    for scene_id in ("one", "two"):
        scenes.add_scene(PROJECT_ID, title=scene_id, scene_id=scene_id, duration=1.0)
    root = projects.project_dir(PROJECT_ID)
    for scene_id in ("one", "two"):
        video = root / "video" / "scenes" / f"{scene_id}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{scene_id}".encode())
        narration = root / "assets" / "audio" / "narration" / f"{scene_id}.wav"
        narration.parent.mkdir(parents=True, exist_ok=True)
        narration.write_bytes(f"audio-{scene_id}".encode())
    originals = {
        "project.json": (root / "project.json").read_bytes(),
        "script/scenes.json": (root / "script" / "scenes.json").read_bytes(),
        "video/scenes/one.mp4": (root / "video/scenes/one.mp4").read_bytes(),
        "video/scenes/two.mp4": (root / "video/scenes/two.mp4").read_bytes(),
    }
    commands: list[list[str]] = []
    mutated = {"done": False}

    def command_runner(command: Sequence[str], timeout_s: float) -> sps.CommandResult:
        commands.append(list(command))
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"prepared")
        if mutate_manifest and not mutated["done"]:
            manifest = root / "script" / "scenes.json"
            with manifest.open("ab") as handle:
                handle.write(b"\n")
            mutated["done"] = True
        return sps.CommandResult(tuple(command), 0)

    def probe(path: str | Path, *, timeout_s: float) -> Mapping[str, object]:
        name = Path(path).name
        if name.startswith(".scene_0000_"):
            duration = 1.0
        elif name.startswith(".scene_0001_"):
            duration = 1.5
        else:
            duration = 2.5
        return {
            "status": "ok",
            "path": str(path),
            "duration_s": duration,
            "size_bytes": 2048,
            "width": 64,
            "height": 48,
            "fps": 10.0,
            "video_codec": "h264",
            "audio_codec": None,
            "pixel_format": "yuv420p",
        }

    mixer = _Mixer(root, duration=mix_duration)
    validator = _Validator()
    coordinator = AudioProductionCoordinator(
        projects,
        scenes,
        planner=_Planner(_plan()),
        arrangement=_ArrangementService(),
        mixer=mixer,
        validator=validator,
        command_runner=command_runner,
        probe=probe,
        check_executables=False,
    )
    return coordinator, root, originals, commands, mixer, validator


def test_produce_creates_report_and_preserves_sources(tmp_path: Path) -> None:
    coordinator, root, originals, commands, mixer, validator = _harness(tmp_path)

    report = coordinator.produce(PROJECT_ID, _config(), render_id="unit")

    assert isinstance(report, AudioProductionReport)
    assert report.render_id == "audio-unit"
    assert report.scene_order == ("one", "two")
    assert report.hold_durations_s["two"] == Decimal("0.5")
    assert report.silent_video_duration_s == Decimal("2.5")
    assert report.audio_duration_s == Decimal("2.5")
    assert report.validation["manifest_preserved"] is True
    assert report.validation["timeline_consistency"] == "ok"
    assert len(commands) == 3
    assert "tpad=stop_mode=clone:stop_duration=0.500000" in commands[1][commands[1].index("-vf") + 1]
    assert "-f" in commands[2] and "concat" in commands[2]

    report_path = root / "audio/renders/audio-unit/report.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["render_id"] == "audio-unit"
    assert payload["hold_durations_s"]["two"] == "0.5"
    assert (root / "audio/renders/audio-unit/final_silent.mp4").is_file()
    assert (root / "audio/renders/audio-unit/scene_0000_one.mp4").is_file()
    assert (root / "audio/renders/audio-unit/scene_0001_two.mp4").is_file()
    assert mixer.calls and mixer.calls[0].render_id == "audio-unit"
    assert len(validator.calls) == 1

    for relative, original in originals.items():
        assert (root / relative).read_bytes() == original


def test_produce_rejects_render_id_reuse(tmp_path: Path) -> None:
    coordinator, _root, _originals, _commands, _mixer, _validator = _harness(tmp_path)

    coordinator.produce(PROJECT_ID, _config(), render_id="reuse")
    with pytest.raises(Exception) as raised:
        coordinator.produce(PROJECT_ID, _config(), render_id="reuse")

    assert getattr(raised.value, "code", None) == "audio.render.collision"


def test_produce_rejects_audio_video_duration_mismatch(tmp_path: Path) -> None:
    coordinator, _root, _originals, _commands, _mixer, _validator = _harness(
        tmp_path, mix_duration=Decimal(9)
    )

    with pytest.raises(Exception) as raised:
        coordinator.produce(PROJECT_ID, _config(), render_id="mismatch")

    assert getattr(raised.value, "code", None) == "audio.production.timeline_mismatch"


def test_produce_detects_manifest_mutation(tmp_path: Path) -> None:
    coordinator, _root, _originals, _commands, _mixer, _validator = _harness(
        tmp_path, mutate_manifest=True
    )

    with pytest.raises(Exception) as raised:
        coordinator.produce(PROJECT_ID, _config(), render_id="mutated")

    assert getattr(raised.value, "code", None) == "audio.production.manifest_mutated"


def test_produce_status_returns_structured_error(tmp_path: Path) -> None:
    projects = ProjectService(tmp_path / "projects")
    coordinator = AudioProductionCoordinator(projects, check_executables=False)

    payload = coordinator.produce_status("missing", _config())

    assert payload["status"] == "error"
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "audio.project.not_found"
