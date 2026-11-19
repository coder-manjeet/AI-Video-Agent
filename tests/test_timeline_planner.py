"""Tests for sequential timeline planning and editor-ready output."""

from __future__ import annotations

from pathlib import Path

import pytest

from manifest_validation import validate_scene_manifest
from project_models import Scene
from project_service import ProjectService
from scene_service import SceneService
from timeline_planner import TimelinePlanner


def _scenes(durations: list[float]) -> list[Scene]:
    return [
        Scene(
            scene_id=f"s{number}",
            scene_number=number,
            title=f"Scene {number}",
            duration=duration,
        )
        for number, duration in enumerate(durations, start=1)
    ]


def test_sequential_timeline_matches_expected_example() -> None:
    scenes = _scenes([8.0, 12.0, 10.0, 15.0])

    plan = TimelinePlanner().plan(scenes)

    assert [
        (entry.timeline_start_s, entry.timeline_end_s) for entry in plan.scenes
    ] == [(0.0, 8.0), (8.0, 20.0), (20.0, 30.0), (30.0, 45.0)]
    assert [entry.duration_s for entry in plan.scenes] == [8.0, 12.0, 10.0, 15.0]
    assert plan.total_duration_s == 45.0
    assert plan.scene_count == 4
    assert tuple(entry.scene_number for entry in plan.scenes) == (1, 2, 3, 4)


def test_planning_does_not_mutate_input_scenes() -> None:
    scenes = _scenes([8.0, 12.0])

    TimelinePlanner().plan(scenes)

    assert [(scene.timeline_start, scene.timeline_end) for scene in scenes] == [
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    assert [scene.scene_number for scene in scenes] == [1, 2]


def test_empty_project_timeline() -> None:
    plan = TimelinePlanner().plan([])

    assert plan.scenes == ()
    assert plan.total_duration_s == 0.0
    assert plan.scene_count == 0
    assert plan.tracks == {"video": ()}


def test_floating_point_durations_are_accurate() -> None:
    plan = TimelinePlanner().plan(_scenes([0.1, 0.2, 0.3]))

    assert [entry.timeline_start_s for entry in plan.scenes] == [0.0, 0.1, 0.3]
    assert [entry.timeline_end_s for entry in plan.scenes] == [0.1, 0.3, 0.6]
    assert plan.total_duration_s == 0.6


def test_invalid_durations_are_rejected() -> None:
    planner = TimelinePlanner()

    with pytest.raises(ValueError):
        planner.plan(_scenes([0.0]))
    with pytest.raises(ValueError):
        planner.plan(_scenes([-1.0]))
    with pytest.raises(TypeError):
        planner.plan([Scene(scene_id="bad", scene_number=1, title="Bad", duration="x")])


def test_editor_response_contains_scene_cards_and_tracks() -> None:
    plan = TimelinePlanner().plan(_scenes([8.0, 12.0]))
    payload = plan.to_dict()

    assert set(payload) == {"total_duration_s", "scene_count", "scenes", "tracks"}
    assert len(payload["tracks"]["video"]) == 2
    card = payload["scenes"][0]
    assert set(card) >= {
        "scene_id",
        "scene_number",
        "title",
        "status",
        "duration_s",
        "timeline_start_s",
        "timeline_end_s",
        "narration",
        "image_path",
        "thumbnail_path",
        "generated_video_path",
        "narration_audio_path",
        "sfx_paths",
        "preview_path",
        "asset_availability",
    }
    assert card["asset_availability"]["video"] is False


def test_asset_availability_reflects_files_on_disk(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    image = root / "assets" / "images" / "one.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    scenes = [
        Scene(
            scene_id="one",
            scene_number=1,
            title="One",
            duration=5.0,
            image_path="assets/images/one.png",
            generated_video_path="video/scenes/one.mp4",
        )
    ]

    plan = TimelinePlanner().plan(scenes, project_root=root)

    assert plan.scenes[0].asset_availability["image"] is True
    assert plan.scenes[0].asset_availability["video"] is False
    assert plan.scenes[0].preview_path == "video/scenes/one.mp4"


def _project_with_scenes(tmp_path: Path) -> tuple[SceneService, TimelinePlanner]:
    projects = ProjectService(tmp_path)
    projects.create_project("demo", "Demo")
    scenes = SceneService(projects)
    for title, duration in (("One", 8.0), ("Two", 12.0), ("Three", 10.0)):
        scenes.add_scene("demo", title=title, scene_id=title.lower(), duration=duration)
    return scenes, TimelinePlanner(scenes)


def test_timeline_recalculation_after_reorder(tmp_path: Path) -> None:
    scenes, planner = _project_with_scenes(tmp_path)
    before = planner.plan_project("demo")

    assert [entry.scene_id for entry in before.scenes] == ["one", "two", "three"]
    assert before.total_duration_s == 30.0

    scenes.reorder_scenes("demo", ["three", "one", "two"])
    after = planner.plan_project("demo")

    assert [entry.scene_id for entry in after.scenes] == ["three", "one", "two"]
    assert [entry.timeline_start_s for entry in after.scenes] == [0.0, 10.0, 18.0]
    assert after.total_duration_s == 30.0


def test_timeline_recalculation_after_deletion(tmp_path: Path) -> None:
    scenes, planner = _project_with_scenes(tmp_path)

    scenes.delete_scene("demo", "two")
    after = planner.plan_project("demo")

    assert [entry.timeline_end_s for entry in after.scenes] == [8.0, 18.0]
    assert after.total_duration_s == 18.0


def test_plan_and_persist_writes_timeline_fields(tmp_path: Path) -> None:
    scenes, planner = _project_with_scenes(tmp_path)

    plan = planner.plan_and_persist("demo")

    stored = scenes.list_scenes("demo")
    assert [(scene.timeline_start, scene.timeline_end) for scene in stored] == [
        (entry.timeline_start_s, entry.timeline_end_s) for entry in plan.scenes
    ]
    assert validate_scene_manifest([scene.to_dict() for scene in stored]).valid
