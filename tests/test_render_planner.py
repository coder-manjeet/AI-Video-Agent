"""Deterministic, fully mocked tests for render-plan construction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest

from project_service import ProjectService
from render_models import RenderPlan
from render_planner import RenderPlanner, RenderPlanningError
from scene_service import SceneService

ProbeMap = dict[str, Mapping[str, object]]


def _video_info(duration_s: object) -> dict[str, object]:
    return {
        "status": "ok",
        "path": "fixture.mp4",
        "duration_s": duration_s,
        "size_bytes": 167200,
        "resolution": "768x512",
        "width": 768,
        "height": 512,
        "fps": 24.0,
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
        "size_bytes": 48000,
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


def _fixture(
    tmp_path: Path,
) -> tuple[ProjectService, SceneService, Path, ProbeMap, RenderPlanner]:
    projects = ProjectService(tmp_path)
    projects.create_project("demo", "Demo")
    root = projects.project_dir("demo")
    scenes = SceneService(projects)
    probes: ProbeMap = {}

    def fake_probe(path: Path) -> Mapping[str, object]:
        return probes[path.name]

    planner = RenderPlanner(projects, scenes, probe=fake_probe)
    return projects, scenes, root, probes, planner


def _add_scene(
    scenes: SceneService,
    root: Path,
    probes: ProbeMap,
    *,
    scene_id: str,
    planned_duration: float,
    measured_duration: object,
    narration_duration: float | None = None,
    narration_text: str = "",
    planned_timeline: tuple[float, float] = (0.0, 0.0),
) -> None:
    scenes.add_scene(
        "demo",
        title=f"Scene {scene_id}",
        scene_id=scene_id,
        duration=planned_duration,
        narration=narration_text,
    )
    video = root / "video" / "scenes" / f"{scene_id}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    probes[video.name] = _video_info(measured_duration)
    changes: dict[str, object] = {"generated_video_path": f"video/scenes/{video.name}"}

    if narration_duration is not None:
        wav = root / "audio" / "narration" / f"{scene_id}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"audio")
        probes[wav.name] = _audio_info(narration_duration)
        changes["narration_audio_path"] = f"audio/narration/{wav.name}"

    scenes.update_scene("demo", scene_id, changes)
    scenes.apply_timeline("demo", {scene_id: planned_timeline})


def _plan_for(
    tmp_path: Path,
) -> tuple[SceneService, Path, ProbeMap, RenderPlanner]:
    _projects, scenes, root, probes, planner = _fixture(tmp_path)
    return scenes, root, probes, planner


def _codes(plan: RenderPlan) -> set[str]:
    return {issue.code for issue in plan.warnings}


def test_scene_order_follows_manifest_order(tmp_path: Path) -> None:
    scenes, root, probes, planner = _plan_for(tmp_path)
    for scene_id in ("one", "two", "three"):
        _add_scene(
            scenes,
            root,
            probes,
            scene_id=scene_id,
            planned_duration=2.0,
            measured_duration=2.0,
        )
    scenes.reorder_scenes("demo", ["three", "one", "two"])

    plan = planner.build_plan("demo")

    assert [entry.scene_id for entry in plan.scenes] == ["three", "one", "two"]
    assert [entry.scene_number for entry in plan.scenes] == [1, 2, 3]


def test_measured_duration_drives_timeline_and_planned_timing_is_preserved(
    tmp_path: Path,
) -> None:
    scenes, root, probes, planner = _plan_for(tmp_path)
    _add_scene(
        scenes,
        root,
        probes,
        scene_id="one",
        planned_duration=4.0,
        measured_duration=6.0,
        planned_timeline=(10.0, 14.0),
    )

    plan = planner.build_plan("demo")
    entry = plan.scenes[0]

    assert entry.planned_duration_s == Decimal("4.0")
    assert entry.measured_duration_s == Decimal("6.0")
    assert entry.effective_duration_s == Decimal("6.0")
    assert entry.timeline_start_s == Decimal(0)
    assert entry.timeline_end_s == Decimal("6.0")
    assert entry.planned_timeline_start_s == Decimal("10.0")
    assert entry.planned_timeline_end_s == Decimal("14.0")
    assert plan.total_render_duration_s == Decimal("6.0")
    assert scenes.get_scene("demo", "one").duration == 4.0
    assert (
        scenes.get_scene("demo", "one").timeline_start,
        scenes.get_scene("demo", "one").timeline_end,
    ) == (10.0, 14.0)


def test_sequential_timeline_uses_decimal_arithmetic(tmp_path: Path) -> None:
    scenes, root, probes, planner = _plan_for(tmp_path)
    _add_scene(scenes, root, probes, scene_id="one", planned_duration=1.0, measured_duration=1.5)
    _add_scene(scenes, root, probes, scene_id="two", planned_duration=2.0, measured_duration=2.25)
    _add_scene(scenes, root, probes, scene_id="three", planned_duration=3.0, measured_duration=3.0)

    payload = planner.build_plan("demo").to_dict()
    timing = [
        (entry["timeline_start_s"], entry["timeline_end_s"])
        for entry in payload["scenes"]
    ]

    assert timing == [("0", "1.5"), ("1.5", "3.75"), ("3.75", "6.75")]
    assert payload["total_render_duration_s"] == "6.75"
    assert payload["total_planned_duration_s"] == "6.0"


def test_narration_longer_than_video_produces_video_hold(tmp_path: Path) -> None:
    scenes, root, probes, planner = _plan_for(tmp_path)
    _add_scene(
        scenes,
        root,
        probes,
        scene_id="one",
        planned_duration=2.0,
        measured_duration=2.0,
        narration_duration=3.5,
        narration_text="Long narration",
    )

    entry = planner.build_plan("demo").scenes[0]

    assert entry.video_hold_duration_s == Decimal("1.5")
    assert entry.silent_tail_duration_s == Decimal(0)
    assert entry.narration_delta_s == Decimal("1.5")
    assert entry.effective_duration_s == Decimal("3.5")
    assert "render.duration.video_hold_required" in _codes(planner.build_plan("demo"))


def test_narration_shorter_than_video_produces_silent_tail(tmp_path: Path) -> None:
    scenes, root, probes, planner = _plan_for(tmp_path)
    _add_scene(
        scenes,
        root,
        probes,
        scene_id="one",
        planned_duration=4.0,
        measured_duration=4.0,
        narration_duration=2.5,
        narration_text="Short narration",
    )

    plan = planner.build_plan("demo")
    entry = plan.scenes[0]

    assert entry.video_hold_duration_s == Decimal(0)
    assert entry.silent_tail_duration_s == Decimal("1.5")
    assert entry.narration_delta_s == Decimal("-1.5")
    assert entry.effective_duration_s == Decimal("4.0")
    assert "render.duration.silent_tail" in _codes(plan)


def test_missing_optional_narration_is_explicit_warning(tmp_path: Path) -> None:
    scenes, root, probes, planner = _plan_for(tmp_path)
    _add_scene(
        scenes,
        root,
        probes,
        scene_id="text_only",
        planned_duration=2.0,
        measured_duration=2.0,
        narration_text="Words without audio",
    )
    _add_scene(
        scenes,
        root,
        probes,
        scene_id="silent",
        planned_duration=2.0,
        measured_duration=2.0,
    )

    plan = planner.build_plan("demo")
    codes = _codes(plan)

    assert "render.narration.audio_missing" in codes
    assert "render.narration.none" in codes
    assert all(entry.narration_duration_s is None for entry in plan.scenes)
    assert all(entry.video_hold_duration_s == 0 for entry in plan.scenes)
    assert all(entry.silent_tail_duration_s == 0 for entry in plan.scenes)


def test_missing_generated_video_path_is_clear_error(tmp_path: Path) -> None:
    _projects, scenes, _root, _probes, planner = _fixture(tmp_path)
    scenes.add_scene(
        "demo", title="No Video", scene_id="novideo", duration=2.0
    )

    with pytest.raises(RenderPlanningError) as excinfo:
        planner.build_plan("demo")

    assert excinfo.value.code == "render.generated_video_path.missing"
    assert "generated_video_path" in str(excinfo.value)


def test_missing_video_file_is_clear_error(tmp_path: Path) -> None:
    scenes, root, probes, planner = _plan_for(tmp_path)
    _add_scene(scenes, root, probes, scene_id="one", planned_duration=2.0, measured_duration=2.0)
    (root / "video" / "scenes" / "one.mp4").unlink()

    with pytest.raises(RenderPlanningError) as excinfo:
        planner.build_plan("demo")

    assert excinfo.value.code == "render.video.missing_file"
    assert "does not exist" in str(excinfo.value)


def test_unreadable_video_is_clear_error(tmp_path: Path) -> None:
    scenes, root, probes, planner = _plan_for(tmp_path)
    _add_scene(scenes, root, probes, scene_id="one", planned_duration=2.0, measured_duration=2.0)
    probes["one.mp4"] = {"status": "error", "error": "ffprobe failed"}

    with pytest.raises(RenderPlanningError) as excinfo:
        planner.build_plan("demo")

    assert excinfo.value.code == "render.video.unreadable"
    assert "ffprobe failed" in str(excinfo.value)


def test_missing_measured_duration_never_falls_back_to_planned(tmp_path: Path) -> None:
    scenes, root, probes, planner = _plan_for(tmp_path)
    _add_scene(scenes, root, probes, scene_id="one", planned_duration=4.0, measured_duration=None)

    with pytest.raises(RenderPlanningError) as excinfo:
        planner.build_plan("demo")

    assert excinfo.value.code == "render.video.invalid_duration"
    assert "planned duration" in str(excinfo.value)


def test_path_traversal_outside_project_root_is_rejected(tmp_path: Path) -> None:
    scenes, _root, probes, planner = _plan_for(tmp_path)
    _add_scene(scenes, _root, probes, scene_id="one", planned_duration=2.0, measured_duration=2.0)
    scene = scenes.get_scene("demo", "one").to_dict()
    scene["generated_video_path"] = "../outside.mp4"
    scenes.manifest_path("demo").write_text(
        json.dumps({"schema_version": 1, "scenes": [scene]}), encoding="utf-8"
    )

    with pytest.raises(RenderPlanningError) as excinfo:
        planner.build_plan("demo")

    assert excinfo.value.code == "render.asset.unsafe_path"


def test_planning_does_not_modify_project_manifests(tmp_path: Path) -> None:
    projects, scenes, root, probes, planner = _fixture(tmp_path)
    _add_scene(
        scenes,
        root,
        probes,
        scene_id="one",
        planned_duration=4.0,
        measured_duration=4.041667,
        planned_timeline=(0.0, 4.0),
    )
    project_bytes = projects.project_file("demo").read_bytes()
    scenes_bytes = scenes.manifest_path("demo").read_bytes()
    before = scenes.list_scenes("demo")

    plan = planner.build_plan("demo")

    assert plan.scenes[0].measured_duration_s == Decimal("4.041667")
    assert projects.project_file("demo").read_bytes() == project_bytes
    assert scenes.manifest_path("demo").read_bytes() == scenes_bytes
    assert scenes.list_scenes("demo") == before


def test_smoke_timing_regression_measured_duration_drives_render(tmp_path: Path) -> None:
    scenes, root, probes, planner = _plan_for(tmp_path)
    _add_scene(
        scenes,
        root,
        probes,
        scene_id="scene-001",
        planned_duration=4.0,
        measured_duration=4.041667,
    )

    plan = planner.build_plan("demo")
    entry = plan.scenes[0]

    assert entry.planned_duration_s == Decimal("4.0")
    assert entry.measured_duration_s == Decimal("4.041667")
    assert entry.timeline_end_s == Decimal("4.041667")
    assert entry.duration_delta_s == Decimal("0.041667")
    assert "render.duration.differs_from_plan" in _codes(plan)
    assert scenes.get_scene("demo", "scene-001").duration == 4.0
