"""Deterministic tests for additive audio timeline planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from audio_models import AudioTimelinePolicy
from audio_timeline import AudioTimelineError, AudioTimelinePlanner
from project_service import ProjectService
from render_models import RenderPlan
from render_planner import RenderPlanner
from scene_service import SceneService

ProbeMap = dict[str, Mapping[str, object]]


def _video_info(duration_s: object) -> dict[str, object]:
    return {
        "status": "ok",
        "path": "fixture.mp4",
        "duration_s": duration_s,
        "size_bytes": 1024,
        "resolution": "64x48",
        "width": 64,
        "height": 48,
        "fps": 10.0,
        "video_codec": "h264",
        "audio_codec": None,
        "sample_rate": None,
        "channels": None,
        "format_name": "mov,mp4",
    }


def _audio_info(duration_s: object) -> dict[str, object]:
    return {
        "status": "ok",
        "path": "fixture.wav",
        "duration_s": duration_s,
        "size_bytes": 1024,
        "resolution": None,
        "width": None,
        "height": None,
        "fps": None,
        "video_codec": None,
        "audio_codec": "pcm_s16le",
        "sample_rate": 24000,
        "channels": 1,
        "format_name": "wav",
    }


def _add_scene(
    scenes: SceneService,
    root: Path,
    probes: ProbeMap,
    *,
    scene_id: str,
    planned: float,
    measured: float,
    narration: float | None,
    planned_timeline: tuple[float, float],
) -> None:
    scenes.add_scene(
        "demo",
        title=f"Scene {scene_id}",
        scene_id=scene_id,
        duration=planned,
        narration="Narration text" if narration is not None else "",
    )
    video = root / "video" / "scenes" / f"{scene_id}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    probes[video.name] = _video_info(measured)
    changes: dict[str, object] = {"generated_video_path": f"video/scenes/{video.name}"}

    if narration is not None:
        wav = root / "audio" / "narration" / f"{scene_id}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"audio")
        probes[wav.name] = _audio_info(narration)
        changes["narration_audio_path"] = f"audio/narration/{wav.name}"

    scenes.update_scene("demo", scene_id, changes)
    scenes.apply_timeline("demo", {scene_id: planned_timeline})


def _plan(tmp_path: Path) -> tuple[SceneService, RenderPlan]:
    projects = ProjectService(tmp_path)
    projects.create_project("demo", "Demo")
    scenes = SceneService(projects)
    root = projects.project_dir("demo")
    probes: ProbeMap = {}

    def fake_probe(path: Path) -> Mapping[str, object]:
        return probes[path.name]

    planner = RenderPlanner(projects, scenes, probe=fake_probe)
    _add_scene(
        scenes, root, probes, scene_id="one", planned=4, measured=5,
        narration=5, planned_timeline=(10, 14),
    )
    _add_scene(
        scenes, root, probes, scene_id="two", planned=2, measured=2,
        narration=3, planned_timeline=(20, 22),
    )
    _add_scene(
        scenes, root, probes, scene_id="three", planned=3, measured=4,
        narration=2, planned_timeline=(30, 33),
    )
    return scenes, planner.build_plan("demo")


def _codes(messages: object) -> set[str]:
    assert isinstance(messages, list)
    return {item["code"] for item in messages if isinstance(item, dict)}


def test_timeline_uses_measured_durations_and_preserves_manifest_timing(
    tmp_path: Path,
) -> None:
    scenes, plan = _plan(tmp_path)

    timeline = AudioTimelinePlanner().build(plan)

    assert timeline.project_id == "demo"
    assert timeline.scene_count == 3
    assert [scene.scene_id for scene in timeline.scenes] == ["one", "two", "three"]
    assert [scene.video_start_s for scene in timeline.scenes] == [Decimal(0), Decimal(5), Decimal(8)]
    assert [scene.video_end_s for scene in timeline.scenes] == [Decimal(5), Decimal(8), Decimal(12)]
    assert timeline.total_duration_s == Decimal(12)
    assert [scene.effective_duration_s for scene in timeline.scenes] == [
        Decimal(5), Decimal(3), Decimal(4)
    ]
    assert [scene.narration_present for scene in timeline.scenes] == [True, True, True]
    assert "audio.timeline.hold_required" in _codes(timeline.scenes[1].to_dict()["warnings"])
    assert "audio.timeline.silent_tail" in _codes(timeline.scenes[2].to_dict()["warnings"])
    assert timeline.scenes[1].final_frame_hold_s == Decimal(1)
    assert timeline.scenes[1].narration_overflow_s == Decimal(1)
    assert timeline.scenes[2].silent_tail_s == Decimal(2)

    for scene_id, timeline_range in (
        ("one", (10.0, 14.0)),
        ("two", (20.0, 22.0)),
        ("three", (30.0, 33.0)),
    ):
        scene = scenes.get_scene("demo", scene_id)
        assert (scene.timeline_start, scene.timeline_end) == timeline_range


def test_missing_narration_is_allowed_by_default_and_can_be_rejected(
    tmp_path: Path,
) -> None:
    projects = ProjectService(tmp_path)
    projects.create_project("demo", "Demo")
    scenes = SceneService(projects)
    root = projects.project_dir("demo")
    probes: ProbeMap = {}

    def fake_probe(path: Path) -> Mapping[str, object]:
        return probes[path.name]

    _add_scene(
        scenes, root, probes, scene_id="silent", planned=2, measured=2,
        narration=None, planned_timeline=(0, 2),
    )
    plan = RenderPlanner(projects, scenes, probe=fake_probe).build_plan("demo")
    planner = AudioTimelinePlanner()

    timeline = planner.build(plan)

    assert timeline.narration_scenes == ()
    assert "audio.timeline.narration_missing" in _codes(timeline.to_dict()["warnings"])
    assert timeline.scenes[0].narration_present is False
    assert timeline.scenes[0].narration_duration_s is None

    with pytest.raises(AudioTimelineError) as raised:
        planner.build(plan, AudioTimelinePolicy(missing_narration_policy="error"))
    assert raised.value.code == "audio.timeline.narration_missing"


def test_sequence_gap_and_total_mismatch_are_reported(tmp_path: Path) -> None:
    _scenes, plan = _plan(tmp_path)
    planner = AudioTimelinePlanner()

    shifted_scene = replace(
        plan.scenes[1],
        timeline_start_s=Decimal(6),
        timeline_end_s=Decimal(9),
    )
    shifted = replace(
        plan,
        scenes=(*plan.scenes[:1], shifted_scene, *plan.scenes[2:]),
    )
    with pytest.raises(AudioTimelineError) as raised:
        planner.build(shifted)
    assert raised.value.code == "audio.timeline.sequence_gap"

    wrong_total = replace(plan, total_render_duration_s=plan.total_render_duration_s + Decimal(1))
    with pytest.raises(AudioTimelineError) as raised:
        planner.build(wrong_total)
    assert raised.value.code == "audio.timeline.total_mismatch"


def test_timeline_rejects_non_plan_input() -> None:
    with pytest.raises(TypeError, match="RenderPlan"):
        AudioTimelinePlanner().build(object())  # type: ignore[arg-type]
