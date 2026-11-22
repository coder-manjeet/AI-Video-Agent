"""Deterministic tests for subtitle capability detection and FFmpeg filters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from scene_preparation_service import CommandResult
from subtitle_models import SubtitleGenerationError, SubtitleStyle
from subtitle_renderer import (
    SubtitleCapabilities,
    build_burn_in_filter,
    check_capabilities,
    escape_filter_path,
    find_font_path,
    write_subtitle_file,
)


@dataclass(frozen=True)
class _Runner:
    stdout: str = ""
    returncode: int = 0

    def __call__(self, command: Sequence[str], timeout_s: float) -> CommandResult:
        return CommandResult(tuple(command), self.returncode, stdout=self.stdout)


@dataclass(frozen=True)
class _FailingRunner:
    def __call__(self, command: Sequence[str], timeout_s: float) -> CommandResult:
        raise OSError("ffmpeg is not installed")


FILTERS_WITH_LIBASS = (
    "  ... scale           V->V       Scale the input video\n"
    "  ... subtitles       V->V       Render text subtitles\n"
    "  ... ass             V->V       Render ASS subtitles\n"
)
FILTERS_WITHOUT_LIBASS = "  ... scale           V->V       Scale the input video\n"


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


@pytest.fixture(autouse=True)
def _isolate_font_scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep real system fonts out of deterministic font-detection tests."""
    monkeypatch.setenv("WINDIR", str(tmp_path / "missing-windows"))
    monkeypatch.delenv("RENDER_SUBTITLE_FONT_DIRS", raising=False)


def test_check_capabilities_detects_libass_filters() -> None:
    capabilities = check_capabilities(command_runner=_Runner(FILTERS_WITH_LIBASS))
    assert capabilities.libass is True
    assert capabilities.burn_in_supported is True
    assert not any("filter was not found" in note for note in capabilities.notes)
    assert capabilities.ffmpeg_bin == "ffmpeg"


def test_check_capabilities_without_libass_reports_notes() -> None:
    capabilities = check_capabilities(command_runner=_Runner(FILTERS_WITHOUT_LIBASS))
    assert capabilities.libass is False
    assert capabilities.burn_in_supported is False
    assert any("subtitles" in note for note in capabilities.notes)


def test_check_capabilities_runner_failure_is_not_fatal() -> None:
    failed = check_capabilities(command_runner=_FailingRunner())
    assert failed.burn_in_supported is False
    assert any("probe failed" in note for note in failed.notes)

    nonzero = check_capabilities(command_runner=_Runner(FILTERS_WITH_LIBASS, returncode=1))
    assert nonzero.libass is True
    assert any("exited with code 1" in note for note in nonzero.notes)


def test_font_detection_via_explicit_directory(tmp_path: Path) -> None:
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    (fonts / "Nirmala.ttf").write_bytes(b"font")
    (fonts / "Mangal.ttf").write_bytes(b"font")
    capabilities = check_capabilities(
        command_runner=_Runner(FILTERS_WITH_LIBASS), font_dirs=[fonts]
    )
    assert capabilities.devanagari_font == "Nirmala UI"
    assert set(capabilities.fonts) == {"Nirmala UI", "Mangal"}


def test_font_detection_via_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fonts = tmp_path / "env-fonts"
    fonts.mkdir()
    (fonts / "NotoSansDevanagari-Regular.ttf").write_bytes(b"font")
    monkeypatch.setenv("RENDER_SUBTITLE_FONT_DIRS", str(fonts))
    capabilities = check_capabilities(command_runner=_Runner(FILTERS_WITH_LIBASS))
    assert capabilities.devanagari_font == "Noto Sans Devanagari"


def test_no_font_found_is_reported_honestly(tmp_path: Path) -> None:
    capabilities = check_capabilities(
        command_runner=_Runner(FILTERS_WITH_LIBASS), font_dirs=[tmp_path]
    )
    assert capabilities.devanagari_font is None
    assert capabilities.fonts == ()
    assert any("No Devanagari" in note for note in capabilities.notes)


def test_find_font_path(tmp_path: Path) -> None:
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    font = fonts / "MANGAL.TTF"
    font.write_bytes(b"font")
    assert find_font_path("Mangal", [fonts]) == font
    assert find_font_path("Nonexistent Family", [fonts]) is None
    with pytest.raises(ValueError, match="font_family"):
        find_font_path("")


def test_capabilities_to_dict() -> None:
    payload = _capabilities(fonts=("Mangal",), devanagari_font="Mangal").to_dict()
    assert payload["fonts"] == ["Mangal"]
    assert payload["devanagari_font"] == "Mangal"
    assert payload["burn_in_supported"] is True


def test_escape_filter_path_windows_quote_and_special_characters() -> None:
    assert escape_filter_path("C:\\Projects\\a'b.srt") == "C\\\\:/Projects/a\\\\\\'b.srt"
    escaped = escape_filter_path("D:/a,b[1]/c'd.srt")
    assert escaped == "D\\\\:/a\\,b\\[1\\]/c\\\\\\'d.srt"
    assert escaped.count("\\") == 8


def test_build_burn_in_filter_default_style() -> None:
    filter_text = build_burn_in_filter("C:\\renders\\sub.srt", SubtitleStyle())
    assert filter_text.startswith("subtitles=filename=C\\\\:/renders/sub.srt:force_style='")
    for fragment in (
        "FontName=Arial",
        "FontSize=36",
        "PrimaryColour=&H00FFFFFF",
        "OutlineColour=&H00000000",
        "Alignment=2",
        "MarginV=40",
        "BorderStyle=1",
    ):
        assert fragment in filter_text


def test_build_burn_in_filter_background_top_and_bold() -> None:
    style = SubtitleStyle(
        font_family="Nirmala UI",
        background=True,
        background_color="#112233",
        background_opacity=64,
        position="top",
        bold=True,
    )
    filter_text = build_burn_in_filter("sub.srt", style)
    assert "BorderStyle=3" in filter_text
    assert "Alignment=8" in filter_text
    assert "BackColour=&H5C332211" in filter_text
    assert "Bold=-1" in filter_text
    assert "FontName=Nirmala UI" in filter_text


def test_write_subtitle_file_encodings_and_unicode(tmp_path: Path) -> None:
    srt = tmp_path / "nested" / "sub.srt"
    vtt = tmp_path / "nested" / "sub.vtt"
    assert write_subtitle_file(srt, "1\n00:00:00,000 --> 00:00:01,000\nनमस्ते\n") == srt
    assert write_subtitle_file(vtt, "WEBVTT\n\nनमस्ते\n") == vtt
    assert srt.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "नमस्ते" in srt.read_text(encoding="utf-8-sig")
    assert not vtt.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "नमस्ते" in vtt.read_text(encoding="utf-8")


def test_write_subtitle_file_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "sub.srt"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(SubtitleGenerationError) as exc:
        write_subtitle_file(target, "replacement")
    assert exc.value.code == "subtitle.file.collision"
    assert target.read_text(encoding="utf-8") == "original"
