"""Deterministic tests for immutable subtitle models and serializers."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any, cast

import pytest

from manifest_validation import ValidationIssue
from subtitle_models import (
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_SIZE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CHARS_PER_LINE,
    DEFAULT_MAX_LINES,
    DEFAULT_OFFSET_S,
    SubtitleGenerationError,
    SubtitleGenerationResult,
    SubtitleSegment,
    SubtitleStyle,
    SubtitleTrack,
    ass_color,
    serialize_subtitles,
    split_sentences,
    validate_segments,
)


def _style(**changes: object) -> SubtitleStyle:
    return SubtitleStyle.from_dict(changes)


def _track(
    segments: tuple[SubtitleSegment, ...] = (),
    *,
    fmt: str = "srt",
    style: SubtitleStyle | None = None,
    offset_s: Decimal = DEFAULT_OFFSET_S,
) -> SubtitleTrack:
    return SubtitleTrack(
        project_id="demo",
        render_id="render-1",
        language="en",
        formats=(fmt,),
        segments=segments,
        style=style if style is not None else SubtitleStyle(),
        timing_source="audio_timeline",
        total_duration_s=Decimal(10),
        offset_s=offset_s,
    )


def test_style_defaults_and_ass_color() -> None:
    style = SubtitleStyle()
    assert style.font_family == DEFAULT_FONT_FAMILY
    assert style.font_size == DEFAULT_FONT_SIZE
    assert style.max_chars_per_line == DEFAULT_MAX_CHARS_PER_LINE
    assert style.max_lines == DEFAULT_MAX_LINES
    assert style.position == "bottom"
    assert ass_color("#FF00AA") == "&H00AA00FF"
    assert ass_color("#000000") == "&H00000000"


def test_style_from_dict_roundtrip() -> None:
    style = _style(font_size=28, color_nonsense="ignored", background=True, position="top")
    assert style.font_size == 28
    assert style.background is True
    assert SubtitleStyle.from_dict(style.to_dict()) == style


def test_style_rejects_invalid_values() -> None:
    cases: tuple[tuple[dict[str, object], type[Exception]], ...] = (
        ({"font_color": "FF0000"}, ValueError),
        ({"position": "left"}, ValueError),
        ({"font_size": 0}, ValueError),
        ({"background_opacity": 101}, ValueError),
        ({"bold": 1}, TypeError),
    )
    for changes, expected in cases:
        with pytest.raises(expected):
            _style(**changes)


def test_segment_validation_and_duration() -> None:
    segment = SubtitleSegment(1, Decimal("1.25"), Decimal("3.5"), "Hello")
    assert segment.duration_s == Decimal("2.25")
    assert segment.scene_id == ""
    assert segment.language == DEFAULT_LANGUAGE
    with pytest.raises(ValueError, match="end_s"):
        SubtitleSegment(1, Decimal(2), Decimal(2), "x")
    with pytest.raises(ValueError, match="non-negative"):
        SubtitleSegment(1, Decimal(-1), Decimal(1), "x")
    with pytest.raises(ValueError, match="text"):
        SubtitleSegment(1, Decimal(0), Decimal(1), "  ")


def test_segment_timestamp_formats_and_offset() -> None:
    segment = SubtitleSegment(1, Decimal("3661.5"), Decimal("3663.25"), "x")
    assert segment.format_timestamp("srt") == "01:01:01,500"
    assert segment.format_timestamp("vtt", Decimal("0.25")) == "01:01:01.750"
    with pytest.raises(ValueError, match="negative"):
        segment.format_timestamp("srt", Decimal(-4000))
    with pytest.raises(ValueError, match="Unsupported"):
        segment.format_timestamp("ass")


def test_segment_dict_roundtrip() -> None:
    segment = SubtitleSegment(2, Decimal("0.5"), Decimal(1), "नमस्ते", "scene-1", "hi")
    payload = segment.to_dict()
    assert payload["start_s"] == "0.5"
    assert payload["end_s"] == "1"
    assert SubtitleSegment.from_dict(payload) == segment


def test_split_sentences_english() -> None:
    assert split_sentences("Hello. World! Again? Yes.", "en") == [
        "Hello.",
        "World!",
        "Again?",
        "Yes.",
    ]


def test_split_sentences_devanagari_preserved() -> None:
    text = "नमस्ते। दुनिया\nफिर मिलेंगे।"
    assert split_sentences(text, "hi") == ["नमस्ते।", "दुनिया", "फिर मिलेंगे।"]
    assert "नमस्ते" in split_sentences(text, "hi")[0]


def test_split_sentences_hinglish_and_empty() -> None:
    assert split_sentences("Namaste दुनिया! Chalo।", "hinglish") == [
        "Namaste दुनिया!",
        "Chalo।",
    ]
    assert split_sentences("  \n ", "en") == []


def test_validate_segments_reports_only_real_issues() -> None:
    first = SubtitleSegment(1, Decimal(0), Decimal(2), "one")
    second = SubtitleSegment(2, Decimal(2), Decimal(4), "two")
    assert validate_segments((first, second), Decimal(4)) == []

    reversed_order = (second, first)
    issues = validate_segments(reversed_order, Decimal(3))
    codes = {issue.code for issue in issues}
    assert codes == {"subtitle.segment.order", "subtitle.segment.out_of_bounds"}
    overlapping = (
        first,
        SubtitleSegment(2, Decimal("1.5"), Decimal(3), "overlap"),
    )
    assert any(
        issue.code == "subtitle.segment.overlap"
        for issue in validate_segments(overlapping, Decimal(4))
    )


def test_serialize_srt_indices_timestamps_and_offset() -> None:
    segments = (
        SubtitleSegment(7, Decimal("0.25"), Decimal(2), "First"),
        SubtitleSegment(9, Decimal(3), Decimal("4.5"), "Second"),
    )
    content = serialize_subtitles(_track(segments, offset_s=Decimal("0.5")), "srt")
    assert content == (
        "1\n00:00:00,750 --> 00:00:02,500\nFirst\n\n"
        "2\n00:00:03,500 --> 00:00:05,000\nSecond\n"
    )


def test_serialize_vtt_header_and_cues() -> None:
    segments = (SubtitleSegment(1, Decimal(0), Decimal(1), "Hello"),)
    content = serialize_subtitles(_track(segments, fmt="vtt"), "vtt")
    assert content == "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHello\n"
    assert "-->" in content and "00:00:00.000" in content
    assert "align:" not in content and "position:" not in content


def test_serialize_wraps_words_without_cutting_or_exceeding_lines() -> None:
    style = _style(max_chars_per_line=10, max_lines=3)
    segments = (SubtitleSegment(1, Decimal(0), Decimal(2), "alpha beta gamma delta"),)
    content = serialize_subtitles(_track(segments, style=style), "srt")
    cue_lines = content.split("\n")[2:5]
    assert cue_lines == ["alpha beta", "gamma", "delta"]
    assert " ".join(cue_lines) == "alpha beta gamma delta"


def test_serialize_long_single_word_kept_intact() -> None:
    style = _style(max_chars_per_line=4, max_lines=1)
    segments = (SubtitleSegment(1, Decimal(0), Decimal(1), "abcde"),)
    content = serialize_subtitles(_track(segments, style=style), "srt")
    assert "abcde" in content


def test_serialize_preserves_hindi_verbatim() -> None:
    segments = (SubtitleSegment(1, Decimal(0), Decimal(2), "यह हिंदी परीक्षण है।"),)
    content = serialize_subtitles(_track(segments), "srt")
    assert "यह हिंदी परीक्षण है।" in content


def test_serialize_rejects_unknown_format_and_offset() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        serialize_subtitles(_track(), "ass")
    segment = SubtitleSegment(1, Decimal(0), Decimal(1), "text")
    with pytest.raises(ValueError, match="negative"):
        serialize_subtitles(_track((segment,), offset_s=Decimal(-10)), "srt")


def test_track_and_result_to_dict() -> None:
    issue = ValidationIssue("subtitle.test", "note", "location")
    segment = SubtitleSegment(1, Decimal(0), Decimal(1), "text")
    track = _track((segment,))
    result = SubtitleGenerationResult(
        project_id="demo",
        render_id="render-1",
        track=track,
        files={"srt": "video/renders/r1/subtitles/subtitles.en.srt"},
        burn_in_filter="subtitles=filename=test.srt",
        capabilities={"libass": True},
        warnings=(issue,),
    )
    payload = cast(dict[str, Any], result.to_dict())
    assert payload["track"]["segment_count"] == 1
    assert payload["track"]["total_duration_s"] == "10"
    assert payload["track"]["warnings"] == []
    assert payload["files"]["srt"].endswith(".srt")
    assert payload["warnings"][0]["code"] == "subtitle.test"
    assert result.burn_in_filter and "subtitles=filename=" in result.burn_in_filter


def test_track_reindex_is_explicit_and_immutable() -> None:
    segment = SubtitleSegment(4, Decimal(0), Decimal(1), "text")
    track = _track((segment,))
    reindexed = replace(track, segments=(replace(segment, index=1),))
    assert reindexed.segment_count == 1
    assert reindexed.segments[0].index == 1
    assert track.segments[0].index == 4


def test_generation_error_to_dict() -> None:
    error = SubtitleGenerationError("subtitle.test.failure", "failed", "here")
    assert error.to_dict() == {
        "code": "subtitle.test.failure",
        "message": "failed",
        "location": "here",
        "stage": "subtitle",
    }
