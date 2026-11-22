"""End-to-end subtitle checks, including one opt-in real FFmpeg burn-in test."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from project_service import ProjectService
from scene_preparation_service import CommandResult
from subtitle_models import SubtitleStyle, serialize_subtitles
from subtitle_renderer import (
    build_burn_in_filter,
    check_capabilities,
    write_subtitle_file,
)
from subtitle_service import SubtitleService

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg"))
requires_ffmpeg = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg is not on PATH")


@dataclass(frozen=True)
class _Runner:
    stdout: str

    def __call__(self, command: Sequence[str], timeout_s: float) -> CommandResult:
        return CommandResult(tuple(command), 0, stdout=self.stdout)


@dataclass(frozen=True)
class _Scene:
    scene_id: str
    narration: str
    narration_start_s: Decimal = Decimal(0)
    effective_duration_s: Decimal = Decimal(4)
    narration_duration_s: Decimal = Decimal(2)
    narration_present: bool = True
    video_end_s: Decimal = Decimal(4)


@dataclass(frozen=True)
class _Timeline:
    scenes: tuple[_Scene, ...]
    total_duration_s: Decimal = Decimal(8)


def _new_service(tmp_path: Path, **kwargs: object) -> SubtitleService:
    projects = ProjectService(tmp_path / "projects")
    projects.create_project("demo", "Demo")
    return SubtitleService(projects=projects, **kwargs)  # type: ignore[arg-type]


def test_pipeline_serialization_matches_written_files(tmp_path: Path) -> None:
    service = _new_service(tmp_path)
    timeline = _Timeline((_Scene("scene-1", "First line. Second line."),))
    result = service.generate(
        "demo", timeline, {"granularity": "sentence", "formats": ("srt", "vtt")},
        render_dir="video/renders/r1", render_id="render-1",
    )
    root = service.projects.project_dir("demo")
    srt = root / result.files["srt"]
    vtt = root / result.files["vtt"]
    assert srt.read_text(encoding="utf-8-sig") == serialize_subtitles(result.track, "srt")
    assert vtt.read_text(encoding="utf-8") == serialize_subtitles(result.track, "vtt")
    filter_text = build_burn_in_filter(srt, result.track.style)
    assert "subtitles=filename=" in filter_text
    assert "First line." in srt.read_text(encoding="utf-8-sig")


def test_service_probes_capabilities_with_injected_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WINDIR", str(tmp_path / "missing-windows"))
    monkeypatch.delenv("RENDER_SUBTITLE_FONT_DIRS", raising=False)
    filters = (
        "  ... subtitles       V->V       Render text subtitles\n"
        "  ... ass             V->V       Render ASS subtitles\n"
    )
    service = _new_service(tmp_path, command_runner=_Runner(filters))
    timeline = _Timeline((_Scene("scene-1", "Hello."),))
    result = service.generate(
        "demo", timeline, None, render_dir="video/renders/r1", render_id="render-1"
    )
    assert result.capabilities["burn_in_supported"] is True
    assert result.burn_in_filter and "force_style='FontName=Arial" in result.burn_in_filter


@requires_ffmpeg
def test_real_ffmpeg_burns_in_english_and_devanagari(tmp_path: Path) -> None:
    capabilities = check_capabilities()
    if not capabilities.burn_in_supported:
        pytest.skip("FFmpeg libass subtitle filters are unavailable")
    if not capabilities.devanagari_font:
        pytest.skip("No Devanagari-capable font was detected")

    video = tmp_path / "source.mp4"
    build = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
            "-t", "1", "-pix_fmt", "yuv420p", str(video),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert build.returncode == 0, build.stderr[-500:]

    subtitle = tmp_path / "sample.srt"
    write_subtitle_file(
        subtitle,
        "1\n00:00:00,000 --> 00:00:00,400\nEnglish and हिंदी\n\n",
    )
    style = SubtitleStyle(font_family=capabilities.devanagari_font, font_size=16)
    burn_in_filter = build_burn_in_filter(subtitle, style)
    output = tmp_path / "burned.mp4"
    run = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-vf", burn_in_filter, "-t", "0.5",
            "-pix_fmt", "yuv420p", str(output),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-1000:]
    assert output.is_file() and output.stat().st_size > 0
