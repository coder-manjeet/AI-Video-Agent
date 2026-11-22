"""Deterministic tests for timeline-driven subtitle generation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from manifest_validation import ValidationIssue
from project_service import ProjectService
from scene_service import SceneService
from subtitle_models import (
    SubtitleGenerationError,
    SubtitleGenerationResult,
)
from subtitle_renderer import SubtitleCapabilities
from subtitle_service import SubtitleConfig, SubtitleService


@dataclass(frozen=True)
class FakeScene:
    scene_id: str = "scene-1"
    title: str = "Scene"
    narration_start_s: Decimal = Decimal(0)
    effective_duration_s: Decimal = Decimal(4)
    narration_duration_s: Decimal | None = Decimal(2)
    narration_present: bool = True
    narration_path: str | None = "audio/narration/scene.wav"
    narration: str = "Hello world."
    video_end_s: Decimal = Decimal(4)


@dataclass(frozen=True)
class FakeTimeline:
    scenes: tuple[FakeScene, ...]
    total_duration_s: Decimal = Decimal(8)
    warnings: tuple[ValidationIssue, ...] = ()


def _capabilities(**changes: object) -> SubtitleCapabilities:
    values: dict[str, object] = {
        "ffmpeg_bin": "ffmpeg",
        "ffprobe_bin": "ffprobe",
        "libass": True,
        "burn_in_supported": True,
        "fonts": (),
        "devanagari_font": None,
        "notes": (),
    }
    values.update(changes)
    return SubtitleCapabilities(**values)  # type: ignore[arg-type]


@pytest.fixture()
def service_env(tmp_path: Path) -> tuple[SubtitleService, ProjectService]:
    projects = ProjectService(tmp_path / "projects")
    projects.create_project("demo", "Demo")
    service = SubtitleService(projects=projects, scenes=SceneService(projects), capabilities=_capabilities())
    return service, projects


def _generate(
    service: SubtitleService,
    timeline: FakeTimeline,
    config: SubtitleConfig | dict[str, Any] | None = None,
    *,
    render_dir: str | Path = "video/renders/r1",
) -> SubtitleGenerationResult:
    return service.generate(
        "demo", timeline, config, render_dir=render_dir, render_id="render-1"
    )


def test_config_defaults_and_roundtrip() -> None:
    config = SubtitleConfig()
    assert config.language == "en"
    assert config.formats == ("srt", "vtt")
    assert config.granularity == "scene"
    assert config.offset_s == Decimal(0)
    assert config.check_fonts is True
    assert SubtitleConfig.from_dict(config.to_dict()) == config


def test_config_rejects_invalid_values() -> None:
    cases: tuple[tuple[dict[str, Any], type[Exception]], ...] = (
        ({"language": "fr"}, ValueError),
        ({"formats": ()}, ValueError),
        ({"formats": ("srt", "ass")}, ValueError),
        ({"formats": "srt"}, TypeError),
        ({"granularity": "word"}, ValueError),
        ({"offset_s": "not-a-number"}, ValueError),
        ({"subtitle_dir_name": "../escape"}, ValueError),
    )
    for changes, expected in cases:
        with pytest.raises(expected):
            SubtitleConfig.from_dict(changes)


def test_config_manual_segments_normalize_decimal_strings() -> None:
    config = SubtitleConfig.from_dict(
        {
            "manual_segments": {
                "scene-1": [
                    {"start_s": "1.5", "end_s": "3", "text": "नमस्ते"},
                ]
            }
        }
    )
    entry = config.manual_segments["scene-1"][0]
    assert entry["start_s"] == Decimal("1.5")
    payload = cast(dict[str, Any], config.to_dict())
    assert payload["manual_segments"]["scene-1"][0]["end_s"] == "3"


def test_generate_scene_english(service_env: tuple[SubtitleService, ProjectService]) -> None:
    service, projects = service_env
    timeline = FakeTimeline((FakeScene(narration="Hello world."),))
    result = _generate(service, timeline)
    assert result.track.segment_count == 1
    segment = result.track.segments[0]
    assert (segment.start_s, segment.end_s) == (Decimal(0), Decimal(2))
    assert segment.text == "Hello world."
    assert set(result.files) == {"srt", "vtt"}
    assert result.files["srt"] == "video/renders/r1/subtitles/subtitles.en.srt"
    assert (projects.project_dir("demo") / result.files["srt"]).is_file()
    assert (projects.project_dir("demo") / result.files["vtt"]).is_file()
    assert result.burn_in_filter and "subtitles=filename=" in result.burn_in_filter
    assert result.warnings == ()


def test_generate_sentence_granularity_estimates_characters(
    service_env: tuple[SubtitleService, ProjectService]
) -> None:
    service, _ = service_env
    timeline = FakeTimeline((FakeScene(narration="One two. Three four."),))
    result = _generate(service, timeline, {"granularity": "sentence", "formats": ("srt",)})
    assert [segment.text for segment in result.track.segments] == ["One two.", "Three four."]
    starts = [segment.start_s for segment in result.track.segments]
    ends = [segment.end_s for segment in result.track.segments]
    # "One two." is 8 of 19 characters, so the boundary is 2s * 8/19 = 0.842s.
    assert starts == [Decimal(0), Decimal("0.842")]
    assert ends == [Decimal("0.842"), Decimal(2)]
    assert all(segment.duration_s > 0 for segment in result.track.segments)


def test_generate_missing_narration_warns_without_fabricating(
    service_env: tuple[SubtitleService, ProjectService]
) -> None:
    service, projects = service_env
    timeline = FakeTimeline((FakeScene(narration_present=False, narration="Ignored"),))
    result = _generate(service, timeline)
    assert result.track.segment_count == 0
    assert any(issue.code == "subtitle.narration.missing" for issue in result.warnings)
    srt = projects.project_dir("demo") / result.files["srt"]
    assert srt.read_text(encoding="utf-8-sig") == ""


def test_generate_manual_segments_replace_automatic(
    service_env: tuple[SubtitleService, ProjectService]
) -> None:
    service, _ = service_env
    scene = FakeScene(scene_id="scene-1", narration="Automatic text")
    timeline = FakeTimeline((scene,))
    config = SubtitleConfig.from_dict(
        {
            "formats": ("srt",),
            "manual_segments": {
                "scene-1": [
                    {"start_s": "0.5", "end_s": "1.5", "text": "Manual one"},
                    {"start_s": "5", "end_s": "6", "text": "Outside scene"},
                ]
            },
        }
    )
    result = _generate(service, timeline, config)
    assert [segment.text for segment in result.track.segments] == ["Manual one"]
    assert any(issue.code == "subtitle.segment.invalid" for issue in result.warnings)


def test_generate_clamps_to_scene_and_timeline_duration(
    service_env: tuple[SubtitleService, ProjectService]
) -> None:
    service, _ = service_env
    scene = FakeScene(
        narration_duration_s=Decimal(12),
        effective_duration_s=Decimal(12),
        video_end_s=Decimal(12),
    )
    timeline = FakeTimeline((scene,), total_duration_s=Decimal(6))
    result = _generate(service, timeline, {"formats": ("srt",)})
    assert result.track.segments[0].end_s == Decimal(6)
    assert result.track.total_duration_s == Decimal(6)


def test_generate_rejects_unsafe_render_dir(
    service_env: tuple[SubtitleService, ProjectService]
) -> None:
    service, _ = service_env
    for render_dir in ("../escape", "C:/Windows/Temp/subtitles"):
        with pytest.raises(SubtitleGenerationError) as exc:
            _generate(service, FakeTimeline(()), render_dir=render_dir)
        assert exc.value.code == "subtitle.path.unsafe"


def test_generate_refuses_existing_subtitle_file(
    service_env: tuple[SubtitleService, ProjectService]
) -> None:
    service, projects = service_env
    existing = projects.project_dir("demo") / "video/renders/r1/subtitles"
    existing.mkdir(parents=True)
    (existing / "subtitles.en.srt").write_text("keep", encoding="utf-8")
    with pytest.raises(SubtitleGenerationError) as exc:
        _generate(service, FakeTimeline((FakeScene(),)))
    assert exc.value.code == "subtitle.file.collision"
    assert (existing / "subtitles.en.srt").read_text(encoding="utf-8") == "keep"


def test_generate_hindi_without_devanagari_font_keeps_external_files(
    service_env: tuple[SubtitleService, ProjectService]
) -> None:
    service, projects = service_env
    timeline = FakeTimeline((FakeScene(narration="यह हिंदी परीक्षण है।"),))
    result = _generate(service, timeline, {"language": "hi", "formats": ("srt",)})
    assert result.burn_in_filter is None
    assert any(issue.code == "subtitle.font.devanagari_missing" for issue in result.warnings)
    srt = projects.project_dir("demo") / result.files["srt"]
    assert "यह हिंदी परीक्षण है।" in srt.read_text(encoding="utf-8-sig")


def test_generate_hinglish_with_font_builds_burn_in(
    service_env: tuple[SubtitleService, ProjectService]
) -> None:
    service, _ = service_env
    service.capabilities = _capabilities(devanagari_font="Nirmala UI", fonts=("Nirmala UI",))
    timeline = FakeTimeline((FakeScene(narration="Namaste दुनिया!"),))
    result = _generate(service, timeline, {"language": "hinglish", "formats": ("srt",)})
    assert result.burn_in_filter and "FontName=Nirmala UI" in result.burn_in_filter
    assert not any(issue.code == "subtitle.font.devanagari_missing" for issue in result.warnings)


def test_build_result_can_disable_burn_in(
    service_env: tuple[SubtitleService, ProjectService]
) -> None:
    service, _ = service_env
    timeline = FakeTimeline((FakeScene(),))
    generated = _generate(service, timeline, {"formats": ("srt",)})
    result = service.build_result(
        project_id="demo",
        render_id="render-1",
        track=generated.track,
        files=generated.files,
        paths={"srt": Path("unused.srt")},
        capabilities=_capabilities(),
        include_burn_in=False,
    )
    assert result.burn_in_filter is None


def test_generate_reads_scene_narration_without_mutating_manifests(
    service_env: tuple[SubtitleService, ProjectService]
) -> None:
    service, projects = service_env
    scenes = SceneService(projects)
    scenes.add_scene("demo", title="Opening", scene_id="scene-1", duration=4, narration="From manifest.")
    manifest = projects.project_dir("demo") / "script" / "scenes.json"
    project_manifest = projects.project_dir("demo") / "project.json"
    before = (manifest.read_bytes(), project_manifest.read_bytes())

    @dataclass(frozen=True)
    class _SceneWithoutNarration:
        scene_id: str
        narration_start_s: Decimal
        effective_duration_s: Decimal
        narration_duration_s: Decimal
        narration_present: bool
        video_end_s: Decimal

    scene = _SceneWithoutNarration(
        "scene-1", Decimal(0), Decimal(4), Decimal(2), True, Decimal(4)
    )
    timeline = FakeTimeline((scene,))  # type: ignore[arg-type]
    result = _generate(service, timeline, {"formats": ("srt",)})
    assert result.track.segments[0].text == "From manifest."
    assert before == (manifest.read_bytes(), project_manifest.read_bytes())


def test_generate_status_ok_and_error(
    service_env: tuple[SubtitleService, ProjectService]
) -> None:
    service, _ = service_env
    ok = service.generate_status(
        "demo", FakeTimeline((FakeScene(),)), {"formats": ("srt",)},
        render_dir="video/renders/r2", render_id="render-2",
    )
    assert ok["status"] == "ok"
    assert isinstance(ok["result"], dict)
    error = service.generate_status(
        "demo", FakeTimeline(()), None, render_dir="../nope", render_id="render-3"
    )
    assert error["status"] == "error"
    assert isinstance(error["error"], dict)
    assert error["error"]["code"] == "subtitle.path.unsafe"
