"""Deterministic tests for silent scene preparation and concatenation.

Unit tests mock subprocess execution and FFprobe results. The small integration
test uses real FFmpeg binaries with generated clips in ``tmp_path`` only.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import media_service
import scene_preparation_service as sps
from manifest_validation import ValidationResult
from project_service import ProjectService
from render_models import RenderPlan
from render_planner import RenderPlanner
from render_validation import RenderExpectations
from scene_service import SceneService

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe are not on PATH"
)


def _video_info(
    duration_s: object,
    *,
    width: int = 64,
    height: int = 48,
    fps: float = 10.0,
    codec: str | None = "h264",
    audio: str | None = None,
    pixel_format: str = "yuv420p",
) -> dict[str, object]:
    return {
        "status": "ok",
        "path": "fixture.mp4",
        "duration_s": duration_s,
        "size_bytes": 2048,
        "width": width,
        "height": height,
        "fps": fps,
        "video_codec": codec,
        "audio_codec": audio,
        "pixel_format": pixel_format,
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
        "width": None,
        "height": None,
        "fps": None,
        "video_codec": None,
        "audio_codec": "pcm_s16le",
        "sample_rate": 24000,
        "channels": 1,
        "format_name": "wav",
    }


class _Validator:
    """Record validation calls and return a controllable verdict."""

    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid
        self.calls: list[dict[str, Any]] = []

    def validate_final(
        self,
        path: str | Path,
        expected: RenderExpectations | None = None,
        require_audio: bool = True,
    ) -> ValidationResult:
        self.calls.append(
            {
                "path": Path(path),
                "expected": expected,
                "require_audio": require_audio,
            }
        )
        result = ValidationResult()
        if not self.valid:
            result.add_error("render.final.simulated_failure", "Simulated validation failure")
        return result


@dataclass
class _Harness:
    projects: ProjectService
    scenes: SceneService
    root: Path
    config: sps.ScenePreparationConfig
    commands: list[list[str]]
    validator: _Validator
    service: sps.ScenePreparationService
    source_bytes: dict[str, bytes] = field(default_factory=dict)
    manifest_bytes: bytes = b""
    durations: dict[str, float] = field(default_factory=dict)
    probes: dict[str, Mapping[str, object]] = field(default_factory=dict)

    def prepare(
        self, config: sps.ScenePreparationConfig | Mapping[str, Any] | None = None
    ) -> sps.PreparationResult:
        return self.service.prepare_scene_videos("demo", config or self.config)


def _fake_probe(
    source_probes: Mapping[str, Mapping[str, object]],
    config: sps.ScenePreparationConfig,
    durations: Mapping[str, float],
    *,
    final_audio: str | None = None,
) -> Any:
    def probe(path: Path, *, timeout_s: float | None = None) -> Mapping[str, object]:
        assert timeout_s is not None and timeout_s > 0
        name = Path(path).name
        if name in source_probes:
            return source_probes[name]
        if name.startswith("scene_"):
            scene_id = name.split("_", 2)[2].removesuffix(".mp4")
            return _video_info(
                durations.get(scene_id, 1.0),
                width=config.width,
                height=config.height,
                fps=config.fps,
                pixel_format=config.pixel_format,
            )
        if name.startswith("final_silent"):
            return _video_info(
                sum(durations.values()),
                width=config.width,
                height=config.height,
                fps=config.fps,
                audio=final_audio,
                pixel_format=config.pixel_format,
            )
        raise AssertionError(f"Unexpected probe target: {name}")

    return probe


def _fake_runner(
    commands: list[list[str]],
    *,
    fail_on: Any = None,
    create_final_collision: bool = False,
) -> sps.CommandRunner:
    def run(command: Sequence[str], timeout_s: float) -> sps.CommandResult:
        assert timeout_s > 0
        values = [str(part) for part in command]
        commands.append(values)
        if fail_on is not None and fail_on(values):
            raise sps.ScenePreparationError(
                "preparation.ffmpeg.failed", "Simulated FFmpeg failure", stage="ffmpeg"
            )
        output = Path(values[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mp4")
        if create_final_collision and "concat" in values:
            (output.parent / "final_silent.mp4").write_bytes(b"collision")
        return sps.CommandResult(tuple(values), 0)

    return run


def _harness(
    tmp_path: Path,
    *,
    scene_ids: Sequence[str] = ("one", "two"),
    durations: Sequence[float] = (2.0, 1.5),
    config_overrides: Mapping[str, Any] | None = None,
    hold_for: str | None = None,
    validator: _Validator | None = None,
    runner: sps.CommandRunner | None = None,
    planner: Any = None,
    probe: Any = None,
    check_executables: bool = False,
    final_audio: str | None = None,
) -> _Harness:
    overrides: dict[str, Any] = {"render_id": "test", **(config_overrides or {})}
    config = sps.ScenePreparationConfig.from_dict(overrides)
    projects = ProjectService(tmp_path / "projects")
    projects.create_project("demo", "Demo")
    root = projects.project_dir("demo")
    scenes = SceneService(projects)
    source_probes: dict[str, Mapping[str, object]] = {}
    source_bytes: dict[str, bytes] = {}
    duration_by_id: dict[str, float] = {}

    for scene_id, duration in zip(scene_ids, durations, strict=True):
        narration = "Narration" if scene_id == hold_for else ""
        scenes.add_scene(
            "demo",
            title=f"Scene {scene_id}",
            scene_id=scene_id,
            duration=duration + 1.0,
            narration=narration,
        )
        video = root / "video" / "scenes" / f"{scene_id}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"original-{scene_id}".encode())
        source_bytes[scene_id] = video.read_bytes()
        source_probes[video.name] = _video_info(duration)
        duration_by_id[scene_id] = duration
        changes: dict[str, object] = {
            "generated_video_path": f"video/scenes/{video.name}"
        }
        if scene_id == hold_for:
            wav = root / "audio" / "narration" / f"{scene_id}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(b"audio")
            source_probes[wav.name] = _audio_info(duration + 1.0)
            changes["narration_audio_path"] = f"audio/narration/{wav.name}"
        scenes.update_scene("demo", scene_id, changes)

    commands: list[list[str]] = []
    active_validator = validator or _Validator()
    active_runner = runner or _fake_runner(commands)
    active_probe = probe or _fake_probe(
        source_probes, config, duration_by_id, final_audio=final_audio
    )
    service = sps.ScenePreparationService(
        projects,
        scenes,
        planner=planner,
        validator=active_validator,
        command_runner=active_runner,
        probe=active_probe,
        check_executables=check_executables,
    )
    return _Harness(
        projects=projects,
        scenes=scenes,
        root=root,
        config=config,
        commands=commands,
        validator=active_validator,
        service=service,
        source_bytes=source_bytes,
        manifest_bytes=(root / "script" / "scenes.json").read_bytes(),
        durations=duration_by_id,
        probes=source_probes,
    )


# --- configuration --------------------------------------------------------


def test_config_defaults_match_required_output_profile() -> None:
    config = sps.ScenePreparationConfig()

    assert (config.width, config.height, config.fps) == (1280, 720, 24.0)
    assert config.video_codec == "libx264"
    assert config.pixel_format == "yuv420p"
    assert config.container == "mp4"
    assert config.preset == "medium"
    assert config.concat_mode == "auto"
    assert config.ffmpeg_timeout_s > 0
    assert config.ffprobe_timeout_s > 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"width": 1279}, "even"),
        ({"height": 0}, "positive"),
        ({"fps": 0}, "positive"),
        ({"crf": 52}, "crf"),
        ({"preset": "turbo"}, "preset"),
        ({"concat_mode": "append"}, "concat_mode"),
        ({"render_root": "../outside"}, "relative"),
        ({"video_codec": "libx264; rm -rf"}, "unsupported characters"),
        ({"ffmpeg_timeout_s": 0}, "positive"),
    ],
)
def test_config_rejects_invalid_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        sps.ScenePreparationConfig.from_dict(overrides)


def test_config_round_trip_preserves_policy() -> None:
    config = sps.ScenePreparationConfig(
        width=640, height=360, fps=30, preset="fast", render_id="roundtrip"
    )
    payload = config.to_dict()
    restored = sps.ScenePreparationConfig.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.policy.require_audio is True


def test_service_returns_structured_config_error(tmp_path: Path) -> None:
    harness = _harness(tmp_path)

    payload = harness.service.prepare("demo", {"width": 1001})

    assert payload["status"] == "error"
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "preparation.config.invalid"
    assert "even" in str(error["message"])


# --- planning and scene order --------------------------------------------


def test_prepare_preserves_render_plan_order(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path, scene_ids=("one", "two", "three"), durations=(1.0, 2.0, 3.0)
    )
    harness.scenes.reorder_scenes("demo", ["three", "one", "two"])

    result = harness.prepare()

    assert result.scene_order == ("three", "one", "two")
    assert [scene.scene_number for scene in result.prepared_scenes] == [1, 2, 3]
    assert [scene.order for scene in result.prepared_scenes] == [0, 1, 2]
    assert [scene.source_duration_s for scene in result.prepared_scenes] == [
        Decimal("3.0"),
        Decimal("1.0"),
        Decimal("2.0"),
    ]
    assert [Path(path).name for path in result.prepared_paths] == [
        "scene_0000_three.mp4",
        "scene_0001_one.mp4",
        "scene_0002_two.mp4",
    ]


def test_planner_is_used_with_silent_policy(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    calls: list[tuple[str, object]] = []
    real_service = harness.service
    original_planner = real_service.planner

    fallback = RenderPlanner(
        harness.projects,
        harness.scenes,
        probe=lambda path: real_service.probe(
            path, timeout_s=harness.config.ffprobe_timeout_s
        ),
    )

    class _SpyPlanner:
        def build_plan(self, project_id: str, policy: object = None) -> RenderPlan:
            calls.append((project_id, policy))
            return fallback.build_plan(project_id, policy)  # type: ignore[arg-type]

    assert original_planner is None
    real_service.planner = _SpyPlanner()  # type: ignore[assignment]

    result = real_service.prepare_scene_videos("demo", harness.config)

    assert len(calls) == 1
    assert calls[0][0] == "demo"
    policy = calls[0][1]
    assert policy.require_audio is False
    assert result.final_duration_s == Decimal("3.5")


def test_manifest_and_source_videos_are_unchanged(tmp_path: Path) -> None:
    harness = _harness(tmp_path)

    harness.prepare()

    assert (harness.root / "script" / "scenes.json").read_bytes() == harness.manifest_bytes
    for scene_id, original in harness.source_bytes.items():
        video = harness.root / "video" / "scenes" / f"{scene_id}.mp4"
        assert video.read_bytes() == original


def test_planned_timeline_values_are_never_rewritten(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    before = harness.scenes.list_scenes("demo")
    planned = [(scene.timeline_start, scene.timeline_end) for scene in before]

    harness.prepare()

    after = harness.scenes.list_scenes("demo")
    assert [(scene.timeline_start, scene.timeline_end) for scene in after] == planned


# --- FFmpeg argument construction ----------------------------------------


def test_scale_filter_uses_letterbox_and_fps() -> None:
    config = sps.ScenePreparationConfig(width=1280, height=720, fps=24)
    graph = sps.scale_filter(config)

    assert "scale=1280:720:force_original_aspect_ratio=decrease" in graph
    assert "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black" in graph
    assert "setsar=1,fps=24" in graph


def test_ffmpeg_arguments_normalize_video_and_drop_audio(tmp_path: Path) -> None:
    harness = _harness(tmp_path, config_overrides={"preset": "fast", "crf": 23})

    harness.prepare()

    prepare = harness.commands[0]
    assert prepare[0] == harness.config.ffmpeg_bin
    assert "-i" in prepare and "-vf" in prepare
    graph = prepare[prepare.index("-vf") + 1]
    assert "force_original_aspect_ratio=decrease" in graph
    assert prepare[prepare.index("-c:v") + 1] == "libx264"
    assert prepare[prepare.index("-preset") + 1] == "fast"
    assert prepare[prepare.index("-crf") + 1] == "23"
    assert prepare[prepare.index("-pix_fmt") + 1] == "yuv420p"
    assert "-an" in prepare
    assert "-sn" in prepare
    assert "-dn" in prepare
    assert all("shell" not in part for part in prepare)


def test_fps_and_pixel_format_configuration(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        config_overrides={"fps": 30, "pixel_format": "yuv422p"},
    )

    result = harness.prepare()

    graph = harness.commands[0][harness.commands[0].index("-vf") + 1]
    assert "fps=30" in graph
    assert harness.commands[0][harness.commands[0].index("-pix_fmt") + 1] == "yuv422p"
    assert all(scene.output_fps == 30.0 for scene in result.prepared_scenes)
    assert all(scene.pixel_format == "yuv422p" for scene in result.prepared_scenes)


def test_run_command_uses_argument_list_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sps.subprocess, "run", fake_run)

    result = sps.run_ffmpeg(["ffmpeg", "-hide_banner"], 12.5)

    assert result.returncode == 0
    assert calls[0][0] == ["ffmpeg", "-hide_banner"]
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["stdin"] == subprocess.DEVNULL
    assert "shell" not in calls[0][1]


# --- concat and final validation -----------------------------------------


def test_windows_concat_paths_are_escaped() -> None:
    escaped = sps.escape_concat_path("C:\\Videos\\it's a clip.mp4")

    assert escaped == "C:/Videos/it'\\''s a clip.mp4"


def test_concat_list_is_generated_in_render_directory(tmp_path: Path) -> None:
    harness = _harness(tmp_path)

    result = harness.prepare()

    concat_list = Path(result.render_dir) / "concat_list.txt"
    lines = concat_list.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"file '{sps.escape_concat_path(path)}'" for path in result.prepared_paths
    ]


def test_successful_concatenation_is_silent_and_validated(tmp_path: Path) -> None:
    harness = _harness(tmp_path)

    result = harness.prepare()

    final = Path(result.final_video_path)
    assert final.is_file() and final.stat().st_size > 0
    assert result.concat_mode == "copy"
    assert "-c" in harness.commands[-1]
    assert harness.commands[-1][harness.commands[-1].index("-c") + 1] == "copy"
    assert "-an" in harness.commands[-1]
    assert result.final_validation.valid is True
    assert len(harness.validator.calls) == 1
    call = harness.validator.calls[0]
    assert call["require_audio"] is False
    expected = call["expected"]
    assert isinstance(expected, RenderExpectations)
    assert (expected.width, expected.height, expected.fps) == (1280, 720, 24.0)
    assert expected.expected_duration_s == Decimal("3.5")
    assert list(Path(result.render_dir).glob(".*.tmp.mp4")) == []
    assert not any(
        "-c:a" in command or "aac" in command for command in harness.commands
    )


def test_reencode_concat_mode_uses_h264_profile(tmp_path: Path) -> None:
    harness = _harness(tmp_path, config_overrides={"concat_mode": "reencode"})

    result = harness.prepare()

    assert result.concat_mode == "reencode"
    concat = harness.commands[-1]
    assert concat[concat.index("-c:v") + 1] == "libx264"
    assert concat[concat.index("-pix_fmt") + 1] == "yuv420p"
    assert any(
        issue.code == "preparation.concat.reencode_requested"
        for issue in result.warnings
    )


def test_hold_duration_is_recorded_without_audio_work(tmp_path: Path) -> None:
    harness = _harness(tmp_path, hold_for="one", durations=(2.0, 1.5))

    result = harness.prepare()

    first = result.prepared_scenes[0]
    assert first.hold_duration_s == Decimal("1.0")
    assert any(
        issue.code == "preparation.scene.hold_recorded" for issue in first.warnings
    )
    assert not any("-c:a" in command for command in harness.commands)


def test_final_validation_failure_is_structured(tmp_path: Path) -> None:
    harness = _harness(tmp_path, validator=_Validator(valid=False))

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.final.invalid"
    final = harness.root / "video" / "renders" / "render-test" / "final_silent.mp4"
    assert final.is_file()


def test_unexpected_audio_track_is_rejected(tmp_path: Path) -> None:
    harness = _harness(tmp_path, final_audio="aac")

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.final.unexpected_audio"


def test_previous_render_directory_is_preserved(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    renders = harness.root / "video" / "renders"
    previous = renders / "render-previous"
    previous.mkdir(parents=True)
    sentinel = previous / "final_silent.mp4"
    sentinel.write_bytes(b"previous")

    harness.prepare()

    assert sentinel.read_bytes() == b"previous"


# --- failures -------------------------------------------------------------


def test_missing_input_video_is_structured(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    source = harness.root / "video" / "scenes" / "two.mp4"
    source.unlink()

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.video.missing_file"
    assert "does not exist" in raised.value.message


def test_invalid_input_video_is_structured(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    harness.probes["one.mp4"] = _video_info(2.0, codec=None)

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.video.missing_stream"
    assert "no video stream" in raised.value.message.lower()


def test_invalid_duration_is_structured(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    harness.probes["one.mp4"] = _video_info(None)

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.video.invalid_duration"
    assert "duration" in raised.value.message.lower()


def test_ffmpeg_timeout_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=12)

    monkeypatch.setattr(sps.subprocess, "run", timed_out)

    with pytest.raises(sps.ScenePreparationError) as raised:
        sps.run_ffmpeg(["ffmpeg"], 12)

    assert raised.value.code == "preparation.ffmpeg.timeout"


def test_ffmpeg_nonzero_exit_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["ffmpeg"], 1, "", "Invalid filter graph: " + "x" * 5000
        )

    monkeypatch.setattr(sps.subprocess, "run", failed)

    with pytest.raises(sps.ScenePreparationError) as raised:
        sps.run_ffmpeg(["ffmpeg"], 5)

    assert raised.value.code == "preparation.ffmpeg.failed"
    assert len(raised.value.stderr_tail) <= 1000
    assert "Invalid filter graph" in raised.value.stderr_tail
    assert "... [truncated] ..." in raised.value.stderr_tail


def test_ffmpeg_not_installed_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(sps.subprocess, "run", missing)

    with pytest.raises(sps.ScenePreparationError) as raised:
        sps.run_ffmpeg(["ffmpeg"], 5)

    assert raised.value.code == "preparation.ffmpeg.not_installed"


def test_ffprobe_not_installed_preflight_is_structured(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        config_overrides={"ffprobe_bin": "definitely-missing-ffprobe"},
        check_executables=True,
    )

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.ffprobe.not_installed"


def test_ffmpeg_missing_preflight_is_structured(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        config_overrides={"ffmpeg_bin": "definitely-missing-ffmpeg"},
        check_executables=True,
    )

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.ffmpeg.not_installed"


def test_render_directory_collision_is_structured(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    collision = harness.root / "video" / "renders" / "render-test"
    collision.mkdir(parents=True)
    sentinel = collision / "keep.txt"
    sentinel.write_bytes(b"keep")

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.render.collision"
    assert sentinel.read_bytes() == b"keep"


def test_final_output_collision_is_structured(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        runner=_fake_runner(harness_commands := [], create_final_collision=True),
    )

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.final.collision"
    assert any("concat" in command for command in harness_commands)


def test_project_path_traversal_is_rejected(tmp_path: Path) -> None:
    harness = _harness(tmp_path)

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare({"render_root": "../escape"})

    assert raised.value.code == "preparation.config.invalid"
    assert not (harness.root.parent / "escape").exists()


def test_scene_asset_traversal_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path)
    scenes = harness.scenes.list_scenes("demo")
    scenes[0].generated_video_path = "../outside.mp4"
    monkeypatch.setattr(harness.scenes, "list_scenes", lambda _project_id: scenes)

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.asset.unsafe_path"


def test_failed_scene_preserves_earlier_prepared_video(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fail_second(command: Sequence[str]) -> bool:
        values = [str(part) for part in command]
        return "-i" in values and "scene_0001" in values[-1]

    harness = _harness(tmp_path, runner=_fake_runner(commands, fail_on=fail_second))

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.ffmpeg.failed"
    render_dir = harness.root / "video" / "renders" / "render-test"
    assert (render_dir / "scene_0000_one.mp4").is_file()
    assert not (render_dir / "scene_0001_two.mp4").exists()
    assert render_dir.is_dir()


def test_failed_concat_preserves_prepared_videos(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fail_concat(command: Sequence[str]) -> bool:
        return "concat" in [str(part) for part in command]

    harness = _harness(tmp_path, runner=_fake_runner(commands, fail_on=fail_concat))

    with pytest.raises(sps.ScenePreparationError) as raised:
        harness.prepare()

    assert raised.value.code == "preparation.ffmpeg.failed"
    render_dir = harness.root / "video" / "renders" / "render-test"
    assert (render_dir / "scene_0000_one.mp4").is_file()
    assert (render_dir / "scene_0001_two.mp4").is_file()
    assert not (render_dir / "final_silent.mp4").exists()


# --- integration ----------------------------------------------------------


def _make_tiny_clip(path: Path, *, size: str, seconds: float) -> Path:
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
        f"testsrc=size={size}:rate=12",
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


@requires_ffmpeg
def test_integration_prepares_and_concatenates_tiny_clips(tmp_path: Path) -> None:
    projects = ProjectService(tmp_path / "projects")
    projects.create_project("integration", "Integration")
    root = projects.project_dir("integration")
    scenes = SceneService(projects)
    expected: dict[str, float] = {}
    originals: dict[str, bytes] = {}
    for scene_id, size, duration in (
        ("wide", "64x48", 0.5),
        ("square", "32x32", 0.4),
    ):
        scenes.add_scene("integration", title=scene_id, scene_id=scene_id, duration=2.0)
        clip = _make_tiny_clip(
            root / "video" / "scenes" / f"{scene_id}.mp4",
            size=size,
            seconds=duration,
        )
        scenes.update_scene(
            "integration",
            scene_id,
            {"generated_video_path": f"video/scenes/{clip.name}"},
        )
        expected[scene_id] = duration
        originals[scene_id] = clip.read_bytes()
        manifest_before = (root / "script" / "scenes.json").read_bytes()

    config = sps.ScenePreparationConfig(
        width=128,
        height=72,
        fps=12,
        preset="ultrafast",
        crf=28,
        render_root="video/renders",
    )
    service = sps.ScenePreparationService(projects, scenes)

    result = service.prepare_scene_videos("integration", config)

    assert result.scene_order == ("wide", "square")
    final = Path(result.final_video_path)
    assert final.is_file() and final.stat().st_size > 0
    info = media_service.probe_media(final)
    assert info["width"] == 128
    assert info["height"] == 72
    assert info["fps"] == 12.0
    assert info["video_codec"] == "h264"
    assert info["audio_codec"] is None
    assert isinstance(info["duration_s"], float)
    assert info["duration_s"] == pytest.approx(sum(expected.values()), abs=0.5)
    assert result.final_validation.valid is True
    assert (root / "script" / "scenes.json").read_bytes() == manifest_before
    for scene_id, original in originals.items():
        assert (root / "video" / "scenes" / f"{scene_id}.mp4").read_bytes() == original


def test_escape_rejects_newlines() -> None:
    with pytest.raises(ValueError, match="control characters"):
        sps.escape_concat_path("C:\\bad\\clip\n.mp4")


def test_service_prepare_returns_error_payload(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    harness.probes["one.mp4"] = _video_info(2.0, codec=None)

    payload = harness.service.prepare("demo", harness.config)

    assert payload["status"] == "error"
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "preparation.video.missing_stream"
