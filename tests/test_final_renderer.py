"""Deterministic tests for final MP4 assembly.

FFmpeg and FFprobe are replaced by injected runners and probes; every file is
created under ``tmp_path``.  Real-binary smoke tests live in the companion
integration module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

import scene_preparation_service as sps
from final_renderer import (
    FinalRenderConfig,
    FinalRenderer,
    FinalRenderError,
    FinalRenderResult,
)
from manifest_validation import ValidationIssue
from project_service import ProjectService

PROJECT_ID = "demo"
RENDER_ID = "render-final"
RENDER_RELATIVE = "video/renders/render-final"


def _config(**overrides: object) -> FinalRenderConfig:
    payload: dict[str, object] = {
        "width": 64,
        "height": 48,
        "fps": 24.0,
        "duration_tolerance_s": "0.5",
    }
    payload.update(overrides)
    return FinalRenderConfig.from_dict(payload)


def _video_info(duration_s: object) -> dict[str, object]:
    return {
        "status": "ok",
        "path": "silent.mp4",
        "duration_s": duration_s,
        "size_bytes": 2048,
        "width": 64,
        "height": 48,
        "fps": 24.0,
        "video_codec": "h264",
        "audio_codec": None,
        "pixel_format": "yuv420p",
        "sample_rate": None,
        "channels": None,
        "format_name": "mov,mp4",
    }


def _audio_info(duration_s: object) -> dict[str, object]:
    return {
        "status": "ok",
        "path": "master.wav",
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


def _final_info(duration_s: object, config: FinalRenderConfig) -> dict[str, object]:
    return {
        "status": "ok",
        "path": "final.mp4",
        "duration_s": duration_s,
        "size_bytes": 4096,
        "width": config.width,
        "height": config.height,
        "fps": config.fps,
        "video_codec": "h264",
        "audio_codec": "aac",
        "pixel_format": config.pixel_format,
        "sample_rate": config.audio_sample_rate,
        "channels": config.audio_channels,
        "format_name": "mov,mp4",
    }


def _expected_duration(
    video_duration: float, audio_duration: float, tolerance: Decimal
) -> Decimal:
    delta = Decimal(str(audio_duration)) - Decimal(str(video_duration))
    extension = Decimal(f"{delta:.6f}") if delta > tolerance else Decimal(0)
    return Decimal(str(video_duration)) + extension


class _Probe:
    def __init__(
        self,
        config: FinalRenderConfig,
        *,
        video_duration: float = 2.0,
        audio_duration: float = 2.0,
        final_overrides: Mapping[str, object] | None = None,
        raise_on: str | None = None,
    ) -> None:
        self.config = config
        self.video_duration = video_duration
        self.audio_duration = audio_duration
        self.final_overrides = dict(final_overrides or {})
        self.raise_on = raise_on
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, path: str | Path, *, timeout_s: float, ffprobe_bin: str | None = None
    ) -> Mapping[str, object]:
        target = Path(path)
        name = target.name
        self.calls.append({"path": target, "timeout_s": timeout_s, "ffprobe_bin": ffprobe_bin})
        if self.raise_on is not None and name == self.raise_on:
            raise OSError("simulated probe failure")
        if name == "silent.mp4":
            return _video_info(self.video_duration)
        if name == "master.wav":
            return _audio_info(self.audio_duration)
        if name == "final.mp4":
            info = _final_info(
                float(
                    _expected_duration(
                        self.video_duration,
                        self.audio_duration,
                        self.config.duration_tolerance_s,
                    )
                ),
                self.config,
            )
            info.update(self.final_overrides)
            return info
        raise AssertionError(f"Unexpected probe target: {name}")


def _fake_runner(
    commands: list[list[str]],
    *,
    marker: bytes = b"rendered-mp4",
    timeouts: list[float] | None = None,
    fail: BaseException | None = None,
    result: sps.CommandResult | None = None,
    create_output_collision: bool = False,
) -> sps.CommandRunner:
    def run(command: Sequence[str], timeout_s: float) -> sps.CommandResult:
        values = [str(part) for part in command]
        commands.append(values)
        if timeouts is not None:
            timeouts.append(timeout_s)
        if fail is not None:
            raise fail
        if result is not None:
            return result
        output = Path(values[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(marker)
        if create_output_collision:
            (output.parent / "final.mp4").write_bytes(b"collision")
        return sps.CommandResult(tuple(values), 0)

    return run


@dataclass
class _Harness:
    root: Path
    config: FinalRenderConfig
    renderer: FinalRenderer
    commands: list[list[str]]
    probe: _Probe
    video: Path
    audio: Path
    render_dir: Path
    source_bytes: dict[str, bytes] = field(default_factory=dict)

    def render(
        self,
        **overrides: object,
    ) -> FinalRenderResult:
        args: dict[str, Any] = {
            "project_id": PROJECT_ID,
            "render_dir": RENDER_RELATIVE,
            "silent_video_path": str(self.video.relative_to(self.root)),
            "audio_master_path": str(self.audio.relative_to(self.root)),
            "render_id": RENDER_ID,
            "config": self.config,
        }
        args.update(overrides)
        return self.renderer.render(**args)  # type: ignore[arg-type]


def _harness(
    tmp_path: Path,
    *,
    config: FinalRenderConfig | None = None,
    video_duration: float = 2.0,
    audio_duration: float = 2.0,
    final_overrides: Mapping[str, object] | None = None,
    raise_on_probe: str | None = None,
    runner: sps.CommandRunner | None = None,
    check_executables: bool = False,
) -> _Harness:
    settings = config or _config()
    projects = ProjectService(tmp_path / "projects")
    projects.create_project(PROJECT_ID, "Demo")
    root = projects.project_dir(PROJECT_ID)
    render_dir = root / "video" / "renders" / "render-final"
    render_dir.mkdir(parents=True, exist_ok=True)
    video = root / "video" / "final" / "silent.mp4"
    video.write_bytes(b"original-video")
    audio = root / "audio" / "master.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"original-audio")
    commands: list[list[str]] = []
    probe = _Probe(
        settings,
        video_duration=video_duration,
        audio_duration=audio_duration,
        final_overrides=final_overrides,
        raise_on=raise_on_probe,
    )
    renderer = FinalRenderer(
        projects,
        command_runner=runner if runner is not None else _fake_runner(commands),
        probe=probe,
        check_executables=check_executables,
    )
    return _Harness(
        root=root,
        config=settings,
        renderer=renderer,
        commands=commands,
        probe=probe,
        video=video,
        audio=audio,
        render_dir=render_dir,
        source_bytes={str(video): video.read_bytes(), str(audio): audio.read_bytes()},
    )


def _assert_error(
    call: Any, code: str
) -> FinalRenderError:
    with pytest.raises(FinalRenderError) as exc_info:
        call()
    assert exc_info.value.code == code
    return exc_info.value


# ---------------------------------------------------------------------------
# Configuration and model tests
# ---------------------------------------------------------------------------


def test_config_defaults_match_contract() -> None:
    config = FinalRenderConfig()
    assert config.width == 1920
    assert config.height == 1080
    assert config.fps == 24.0
    assert config.video_codec == "libx264"
    assert config.audio_codec == "aac"
    assert config.audio_bitrate_kbps == 192
    assert config.audio_sample_rate == 48000
    assert config.audio_channels == 2
    assert config.container == "mp4"
    assert config.preset_x264 == "medium"
    assert config.crf == 18
    assert config.movflags == "+faststart"
    assert config.subtitle_burn_in is False
    assert config.burn_in_filter is None
    assert config.duration_tolerance_s == Decimal("0.5")
    assert config.output_filename == "final.mp4"


def test_config_to_dict_emits_decimal_string_and_roundtrips() -> None:
    config = _config(duration_tolerance_s="0.25")
    payload = config.to_dict()
    assert payload["duration_tolerance_s"] == "0.25"
    assert isinstance(payload["duration_tolerance_s"], str)
    assert FinalRenderConfig.from_dict(payload) == config


def test_config_rejects_odd_dimensions() -> None:
    with pytest.raises(ValueError, match="even"):
        FinalRenderConfig(width=1921)


@pytest.mark.parametrize("container", ["mov", "mkv", "MP42"])
def test_config_rejects_non_mp4_container(container: str) -> None:
    with pytest.raises(ValueError, match="mp4"):
        FinalRenderConfig(container=container)


@pytest.mark.parametrize("crf", [-1, 52])
def test_config_rejects_crf_out_of_range(crf: int) -> None:
    with pytest.raises(ValueError, match=r"crf"):
        FinalRenderConfig(crf=crf)


def test_config_rejects_unsupported_preset() -> None:
    with pytest.raises(ValueError, match="preset_x264"):
        FinalRenderConfig(preset_x264="lightning")


def test_config_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValueError, match="ffmpeg_timeout_s"):
        FinalRenderConfig(ffmpeg_timeout_s=0)


def test_config_rejects_unsafe_output_filename() -> None:
    for name in ("../final.mp4", "sub/final.mp4", "final.mov", "final"):
        with pytest.raises(ValueError, match="output_filename"):
            FinalRenderConfig(output_filename=name)


def test_error_to_dict_is_structured() -> None:
    error = FinalRenderError(
        "final.output.collision", "exists", "final.mp4", stage="output", stderr_tail="tail"
    )
    payload = error.to_dict()
    assert payload["code"] == "final.output.collision"
    assert payload["message"] == "exists"
    assert payload["location"] == "final.mp4"
    assert payload["stage"] == "output"
    assert payload["stderr_tail"] == "tail"
    issues = cast("list[dict[str, object]]", payload["issues"])
    assert issues[0]["code"] == "final.output.collision"


def test_result_to_dict_is_json_safe_with_decimal_and_path_probe() -> None:
    config = _config()
    result = FinalRenderResult(
        project_id=PROJECT_ID,
        render_id=RENDER_ID,
        render_dir=RENDER_RELATIVE,
        output_path=f"{RENDER_RELATIVE}/final.mp4",
        size_bytes=4096,
        duration_s=Decimal("2.000"),
        width=config.width,
        height=config.height,
        fps=config.fps,
        video_codec="libx264",
        pixel_format="yuv420p",
        audio_codec="aac",
        sample_rate=48000,
        channels=2,
        subtitle_path=None,
        burn_in=False,
        duration_decisions=("within tolerance",),
        command=("ffmpeg", "-i", "in.mp4"),
        probe={"duration_s": Decimal("2.000"), "path": Path("final.mp4")},
        warnings=(ValidationIssue("final.test.warning", "note"),),
    )
    payload = result.to_dict()
    assert payload["duration_s"] == "2.000"
    probe = cast("dict[str, object]", payload["probe"])
    warnings = cast("list[dict[str, object]]", payload["warnings"])
    assert probe["duration_s"] == "2.000"
    assert probe["path"] == "final.mp4"
    assert warnings[0]["code"] == "final.test.warning"
    assert json.loads(json.dumps(payload))["command"][0] == "ffmpeg"


# ---------------------------------------------------------------------------
# Command construction and duration policy tests
# ---------------------------------------------------------------------------


def test_render_stream_copies_video_when_durations_match(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    result = harness.render()
    command = harness.commands[0]
    assert command[0] == "ffmpeg"
    assert command[:4] == ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert command[command.index("-i") + 1].endswith("silent.mp4")
    assert "-vf" not in command
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-b:a") + 1] == "192k"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-ac") + 1] == "2"
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert Path(command[-1]).name.startswith(".final-")
    assert Path(command[-1]).name.endswith(".tmp.mp4")
    assert result.command == tuple(command)
    assert result.output_path == f"{RENDER_RELATIVE}/final.mp4"
    assert result.render_dir == RENDER_RELATIVE
    assert result.subtitle_path is None
    assert result.burn_in is False
    assert "within tolerance" in result.duration_decisions[0]


def test_render_audio_shorter_records_silent_tail(tmp_path: Path) -> None:
    harness = _harness(tmp_path, video_duration=2.0, audio_duration=1.0)
    result = harness.render()
    assert any("silent tail" in decision for decision in result.duration_decisions)
    assert all("tpad" not in part for part in harness.commands[0])
    assert harness.commands[0][harness.commands[0].index("-c:v") + 1] == "copy"
    assert result.duration_s == Decimal("2.0")


def test_render_audio_longer_adds_tpad_and_reencodes(tmp_path: Path) -> None:
    harness = _harness(tmp_path, video_duration=2.0, audio_duration=2.8)
    result = harness.render()
    command = harness.commands[0]
    video_filter = command[command.index("-vf") + 1]
    assert video_filter == "tpad=stop_mode=clone:stop_duration=0.800000"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-preset") + 1] == "medium"
    assert command[command.index("-crf") + 1] == "18"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert any("0.800000s" in decision for decision in result.duration_decisions)
    assert result.duration_s == Decimal("2.800000")


def test_render_within_tolerance_does_not_pad_or_reencode(tmp_path: Path) -> None:
    harness = _harness(tmp_path, video_duration=2.0, audio_duration=2.2)
    result = harness.render()
    assert "-vf" not in harness.commands[0]
    assert harness.commands[0][harness.commands[0].index("-c:v") + 1] == "copy"
    assert "within tolerance" in result.duration_decisions[0]
    assert result.duration_s == Decimal("2.0")


def test_render_burn_in_applies_filter_and_reencodes(tmp_path: Path) -> None:
    harness = _harness(tmp_path, config=_config(subtitle_burn_in=True))
    subtitle = harness.root / "subtitles" / "burn.srt"
    subtitle.parent.mkdir(parents=True, exist_ok=True)
    subtitle.write_bytes(b"1\n00:00:00,000 --> 00:00:01,000\nHello\n")
    burn_in_filter = "subtitles=filename='subtitles/burn.srt'"
    result = harness.render(
        subtitle_path="subtitles/burn.srt", burn_in_filter=burn_in_filter
    )
    command = harness.commands[0]
    assert command[command.index("-vf") + 1] == burn_in_filter
    assert command[command.index("-c:v") + 1] == "libx264"
    assert result.burn_in is True
    assert result.subtitle_path == "subtitles/burn.srt"
    assert any("burn-in" in decision for decision in result.duration_decisions)


def test_render_burn_in_disabled_ignores_supplied_filter(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    result = harness.render(burn_in_filter="subtitles=filename='missing.srt'")
    assert "-vf" not in harness.commands[0]
    assert result.burn_in is False
    assert result.subtitle_path is None
    assert any("ignored" in decision for decision in result.duration_decisions)


def test_render_burn_in_enabled_requires_subtitle_file(tmp_path: Path) -> None:
    harness = _harness(tmp_path, config=_config(subtitle_burn_in=True))
    error = _assert_error(
        lambda: harness.render(burn_in_filter="subtitles=filename='missing.srt'"),
        "final.subtitle.missing",
    )
    assert error.stage == "subtitle"


def test_render_burn_in_enabled_requires_filter(tmp_path: Path) -> None:
    harness = _harness(tmp_path, config=_config(subtitle_burn_in=True))
    subtitle = harness.root / "subtitles" / "burn.srt"
    subtitle.parent.mkdir(parents=True, exist_ok=True)
    subtitle.write_bytes(b"srt")
    error = _assert_error(
        lambda: harness.render(subtitle_path="subtitles/burn.srt"),
        "final.subtitle.filter_missing",
    )
    assert error.location == "burn_in_filter"


# ---------------------------------------------------------------------------
# Path and source validation tests
# ---------------------------------------------------------------------------


def test_render_refuses_absolute_video_outside_project(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")
    error = _assert_error(
        lambda: harness.render(silent_video_path=str(outside)), "final.path.unsafe"
    )
    assert error.stage == "path"


def test_render_refuses_parent_traversal_in_render_dir(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    error = _assert_error(
        lambda: harness.render(render_dir="../escape"), "final.path.unsafe"
    )
    assert error.location == "render_dir"


def test_render_accepts_absolute_path_inside_project(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    result = harness.render(silent_video_path=str(harness.video))
    assert result.output_path.endswith("final.mp4")


def test_render_rejects_missing_video(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    harness.video.unlink()
    _assert_error(lambda: harness.render(), "final.video.missing")


def test_render_rejects_empty_audio(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    harness.audio.write_bytes(b"")
    _assert_error(lambda: harness.render(), "final.audio.missing")


def test_render_rejects_missing_render_directory(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    error = _assert_error(
        lambda: harness.render(render_dir="video/renders/absent"),
        "final.render.missing",
    )
    assert error.stage == "render"


# ---------------------------------------------------------------------------
# Probe, execution and verification tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["silent.mp4", "master.wav"])
def test_render_missing_duration_is_invalid_probe(tmp_path: Path, target: str) -> None:
    config = _config()
    projects = ProjectService(tmp_path / "projects")
    projects.create_project(PROJECT_ID, "Demo")
    root = projects.project_dir(PROJECT_ID)
    render_dir = root / "video" / "renders" / "render-final"
    render_dir.mkdir(parents=True, exist_ok=True)
    video = root / "video" / "final" / "silent.mp4"
    video.write_bytes(b"v")
    audio = root / "audio" / "master.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"a")
    probe = _Probe(config)
    if target == "silent.mp4":
        probe.video_duration = None  # type: ignore[assignment]
    else:
        probe.audio_duration = None  # type: ignore[assignment]
    renderer = FinalRenderer(
        projects,
        command_runner=_fake_runner([]),
        probe=probe,
        check_executables=False,
    )
    error = _assert_error(
        lambda: renderer.render(
            PROJECT_ID,
            RENDER_RELATIVE,
            "video/final/silent.mp4",
            "audio/master.wav",
            render_id=RENDER_ID,
            config=config,
        ),
        "final.probe.invalid",
    )
    assert "duration" in error.message


def test_render_probe_exception_is_structured(tmp_path: Path) -> None:
    harness = _harness(tmp_path, raise_on_probe="silent.mp4")
    error = _assert_error(lambda: harness.render(), "final.probe.invalid")
    assert "simulated probe failure" in error.message


def test_render_runner_exception_is_structured(tmp_path: Path) -> None:
    failure = sps.ScenePreparationError(
        "preparation.ffmpeg.failed", "simulated ffmpeg failure", stage="ffmpeg", stderr_tail="boom"
    )
    harness = _harness(tmp_path, runner=_fake_runner([], fail=failure))
    error = _assert_error(lambda: harness.render(), "final.ffmpeg.failed")
    assert error.stage == "ffmpeg"
    assert error.stderr_tail == "boom"


def test_render_nonzero_runner_result_is_structured(tmp_path: Path) -> None:
    failed = sps.CommandResult(("ffmpeg",), 1, stderr="exit failure")
    harness = _harness(tmp_path, runner=_fake_runner([], result=failed))
    error = _assert_error(lambda: harness.render(), "final.ffmpeg.failed")
    assert "exit failure" in error.message


def test_render_metadata_mismatch_fails_and_preserves_output(tmp_path: Path) -> None:
    harness = _harness(tmp_path, final_overrides={"fps": 12.0})
    error = _assert_error(lambda: harness.render(), "final.output.metadata_mismatch")
    assert "fps" in error.message
    assert (harness.render_dir / "final.mp4").is_file()


def test_render_duration_mismatch_fails(tmp_path: Path) -> None:
    harness = _harness(tmp_path, final_overrides={"duration_s": 0.5})
    error = _assert_error(lambda: harness.render(), "final.output.metadata_mismatch")
    assert "duration" in error.message


def test_render_missing_output_audio_stream_fails(tmp_path: Path) -> None:
    harness = _harness(tmp_path, final_overrides={"audio_codec": None})
    _assert_error(lambda: harness.render(), "final.output.missing_audio_stream")


def test_render_missing_final_duration_is_structured(tmp_path: Path) -> None:
    harness = _harness(tmp_path, final_overrides={"duration_s": None})
    error = _assert_error(lambda: harness.render(), "final.probe.invalid")
    assert "duration" in error.message


# ---------------------------------------------------------------------------
# Collision, promotion, status and injection tests
# ---------------------------------------------------------------------------


def test_render_refuses_existing_output_before_running(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    output = harness.render_dir / "final.mp4"
    output.write_bytes(b"existing")
    error = _assert_error(lambda: harness.render(), "final.output.collision")
    assert "already exists" in error.message
    assert not harness.commands
    assert output.read_bytes() == b"existing"


def test_render_refuses_output_appearing_during_render(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        runner=_fake_runner([], marker=b"new", create_output_collision=True),
    )
    error = _assert_error(lambda: harness.render(), "final.output.collision")
    assert "appeared during rendering" in error.message
    assert (harness.render_dir / "final.mp4").read_bytes() == b"collision"
    assert list(harness.render_dir.glob(".final-*.tmp.mp4"))


def test_render_promotes_temp_and_preserves_sources(tmp_path: Path) -> None:
    harness = _harness(tmp_path, runner=_fake_runner([], marker=b"promoted"))
    result = harness.render()
    output = harness.render_dir / "final.mp4"
    assert output.read_bytes() == b"promoted"
    assert not list(harness.render_dir.glob(".final-*.tmp.mp4"))
    assert result.size_bytes == 4096
    for path, original in harness.source_bytes.items():
        assert Path(path).read_bytes() == original


def test_render_status_returns_ok_then_error(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    status = harness.renderer.render_status(
        PROJECT_ID,
        RENDER_RELATIVE,
        "video/final/silent.mp4",
        "audio/master.wav",
        render_id=RENDER_ID,
        config=harness.config,
    )
    assert status["status"] == "ok"
    assert isinstance(status["result"], dict)
    assert status["result"]["output_path"] == f"{RENDER_RELATIVE}/final.mp4"

    second = harness.renderer.render_status(
        PROJECT_ID,
        RENDER_RELATIVE,
        "video/final/silent.mp4",
        "audio/master.wav",
        render_id=RENDER_ID,
        config=harness.config,
    )
    assert second["status"] == "error"
    second_error = cast("dict[str, object]", second["error"])
    assert second_error["code"] == "final.output.collision"


def test_render_status_reports_invalid_config(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    status = harness.renderer.render_status(
        PROJECT_ID,
        RENDER_RELATIVE,
        "video/final/silent.mp4",
        "audio/master.wav",
        render_id=RENDER_ID,
        config={"width": 21},
    )
    assert status["status"] == "error"
    error = cast("dict[str, object]", status["error"])
    assert error["code"] == "final.config.invalid"
    assert error["stage"] == "config"


def test_injected_runner_and_probe_receive_configured_timeouts(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    timeouts: list[float] = []
    harness.renderer.command_runner = _fake_runner(harness.commands, timeouts=timeouts)
    harness.render()
    assert timeouts == [harness.config.ffmpeg_timeout_s]
    assert [call["timeout_s"] for call in harness.probe.calls] == [
        harness.config.ffprobe_timeout_s,
        harness.config.ffprobe_timeout_s,
        harness.config.ffprobe_timeout_s,
    ]


def test_default_probe_uses_configured_ffprobe_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path)
    config = _config(ffprobe_bin="ffprobe-custom", ffprobe_timeout_s=12.5)
    captured: list[dict[str, Any]] = []

    def fake_probe(
        path: str | Path, *, ffprobe_bin: str, timeout_s: float
    ) -> Mapping[str, object]:
        captured.append({"path": Path(path), "ffprobe_bin": ffprobe_bin, "timeout_s": timeout_s})
        return _Probe(config)(path, timeout_s=timeout_s)

    monkeypatch.setattr(sps, "probe_video", fake_probe)
    renderer = FinalRenderer(
        harness.renderer.projects,
        command_runner=_fake_runner(harness.commands),
        check_executables=False,
    )
    renderer.render(
        PROJECT_ID,
        RENDER_RELATIVE,
        "video/final/silent.mp4",
        "audio/master.wav",
        render_id=RENDER_ID,
        config=config,
    )
    assert captured
    assert all(call["ffprobe_bin"] == "ffprobe-custom" for call in captured)
    assert all(call["timeout_s"] == 12.5 for call in captured)
