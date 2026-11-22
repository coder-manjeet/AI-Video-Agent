"""Deterministic tests for final QA validation (no real FFmpeg required)."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from final_qa import (
    FinalQAValidator,
    QAIssue,
    QAReport,
    QAThresholds,
    detect_black,
    detect_silence,
    qa_status,
)
from scene_preparation_service import CommandResult

EXPECTED: dict[str, object] = {
    "width": 1280,
    "height": 720,
    "fps": 24.0,
    "video_codec": "h264",
    "pixel_format": "yuv420p",
    "audio_codec": "aac",
    "sample_rate": 48000,
    "channels": 2,
    "duration_s": "10.0",
    "language": "en",
}


def _expected(**overrides: object) -> dict[str, object]:
    payload = dict(EXPECTED)
    payload.update(overrides)
    return payload


def _probe_info(**overrides: object) -> dict[str, object]:
    info: dict[str, object] = {
        "status": "ok",
        "path": "fixture.mp4",
        "duration_s": 10.0,
        "audio_duration_s": 10.0,
        "size_bytes": 4096,
        "width": 1280,
        "height": 720,
        "fps": 24.0,
        "video_codec": "h264",
        "audio_codec": "aac",
        "pixel_format": "yuv420p",
        "sample_rate": 48000,
        "channels": 2,
    }
    info.update(overrides)
    return info


def _runner(stderr: str = "") -> Any:
    def run(command: Sequence[str], timeout_s: float) -> CommandResult:
        return CommandResult(tuple(command), 0, stderr=stderr)

    return run


def _failing_runner() -> Any:
    def run(command: Sequence[str], timeout_s: float) -> CommandResult:
        raise RuntimeError("ffmpeg missing")

    return run


def _validator(info: Mapping[str, object] | None = None, runner: Any = None) -> FinalQAValidator:
    return FinalQAValidator(
        probe=lambda _path: info if info is not None else _probe_info(),
        command_runner=runner if runner is not None else _runner(),
    )


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "final.mp4"
    path.write_bytes(b"video")
    return path


def _codes(issues: Sequence[QAIssue]) -> set[str]:
    return {issue.code for issue in issues}


class _Timeline:
    def __init__(self, scenes: list[dict[str, object]], total: str = "10.0") -> None:
        self.scenes = scenes
        self.total_duration_s = Decimal(total)


def _scene(
    scene_id: str,
    order: int,
    start: str,
    end: str,
    *,
    hold: str = "0",
    tail: str = "0",
    narration_end: str | None = None,
) -> dict[str, object]:
    narration_started = narration_end is not None
    return {
        "scene_id": scene_id,
        "order": order,
        "title": scene_id,
        "video_start_s": Decimal(start),
        "video_end_s": Decimal(end),
        "final_frame_hold_s": Decimal(hold),
        "silent_tail_s": Decimal(tail),
        "narration_present": narration_started,
        "narration_duration_s": Decimal(3) if narration_started else None,
        "narration_end_s": Decimal(narration_end) if narration_end else None,
    }


def _two_scene_timeline() -> _Timeline:
    return _Timeline(
        [
            _scene("one", 0, "0", "5", tail="1", narration_end="4"),
            _scene("two", 1, "5", "10", hold="0.5"),
        ]
    )


# --- thresholds -----------------------------------------------------------


def test_threshold_defaults_and_round_trip() -> None:
    thresholds = QAThresholds()

    payload = thresholds.to_dict()
    assert payload["duration_tolerance_s"] == "0.5"
    assert payload["sync_tolerance_s"] == "0.5"
    assert QAThresholds.from_dict(payload) == thresholds
    assert QAThresholds.from_dict(thresholds) is thresholds


@pytest.mark.parametrize(
    "payload",
    [
        {"duration_tolerance_s": "-1"},
        {"sync_tolerance_s": "-0.1"},
        {"fps_tolerance": -1},
        {"peak_limit": 1.2},
        {"min_silence_s": -1},
        {"black_min_duration_s": -1},
        {"black_pixel_threshold": 1.5},
        {"black_area_threshold": 0},
        {"ffmpeg_timeout_s": 0},
    ],
)
def test_threshold_from_dict_rejects_invalid_ranges(payload: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        QAThresholds.from_dict(payload)


def test_threshold_from_dict_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="must be an object"):
        QAThresholds.from_dict(["ffmpeg"])  # type: ignore[arg-type]


# --- issues and report ----------------------------------------------------


def test_qa_issue_rejects_unknown_severity() -> None:
    with pytest.raises(ValueError, match="severity"):
        QAIssue(severity="fatal", code="qa.x", message="bad")


def test_qa_report_separates_errors_and_warnings_and_serializes_decimals() -> None:
    error = QAIssue("error", "qa.error", "broken", timestamp_s=Decimal("1.25"))
    warning = QAIssue("warning", "qa.warning", "suspicious")
    report = QAReport((error, warning), {"file_exists": True}, {"peak": 0.5})

    payload = report.to_dict()
    assert report.valid is False
    assert len(report.errors) == 1
    assert len(report.warnings) == 1
    errors = payload["errors"]
    warnings = payload["warnings"]
    assert isinstance(errors, list) and isinstance(errors[0], dict)
    assert isinstance(warnings, list) and isinstance(warnings[0], dict)
    assert errors[0]["timestamp_s"] == "1.25"
    assert warnings[0]["code"] == "qa.warning"
    assert payload["checks"] == {"file_exists": True}
    assert QAReport((warning,)).valid is True


# --- strict metadata checks ----------------------------------------------


def test_valid_final_metadata_passes(tmp_path: Path) -> None:
    report = _validator().validate(_video(tmp_path), expected=_expected())

    assert report.valid is True
    assert report.to_dict()["valid"] is True
    assert report.checks["file_exists"] is True
    assert report.checks["video_stream"] is True
    assert report.checks["audio_stream"] is True
    assert report.checks["metadata_matches"] is True


def test_missing_and_empty_video_are_errors(tmp_path: Path) -> None:
    missing = _validator().validate(tmp_path / "missing.mp4", expected=_expected())
    empty = tmp_path / "empty.mp4"
    empty.touch()
    empty_report = _validator().validate(empty, expected=_expected())

    assert "qa.file.missing" in _codes(missing.errors)
    assert "qa.file.empty" in _codes(empty_report.errors)


def test_probe_status_error_is_reported(tmp_path: Path) -> None:
    info = _probe_info(status="error", error="ffprobe failed")
    report = _validator(info).validate(_video(tmp_path), expected=_expected())

    assert "qa.probe.failed" in _codes(report.errors)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"video_codec": None}, "qa.video.missing_stream"),
        ({"audio_codec": None}, "qa.audio.missing_stream"),
        ({"width": 640}, "qa.video.resolution_mismatch"),
        ({"fps": 30.0}, "qa.video.fps_mismatch"),
        ({"pixel_format": "yuv444p"}, "qa.video.pixel_format_mismatch"),
        ({"duration_s": 12.0}, "qa.video.duration_mismatch"),
        ({"audio_codec": "mp3"}, "qa.audio.codec_mismatch"),
        ({"sample_rate": 44100}, "qa.audio.sample_rate_mismatch"),
        ({"channels": 1}, "qa.audio.channels_mismatch"),
    ],
)
def test_metadata_mismatches_are_errors(
    tmp_path: Path, overrides: dict[str, object], code: str
) -> None:
    info = _probe_info(**overrides)
    report = _validator(info).validate(_video(tmp_path), expected=_expected())

    assert code in _codes(report.errors)
    assert report.valid is False


def test_video_codec_mismatch_is_error(tmp_path: Path) -> None:
    info = _probe_info(video_codec="mpeg4")
    report = _validator(info).validate(_video(tmp_path), expected=_expected())

    assert "qa.video.codec_mismatch" in _codes(report.errors)


def test_audio_video_sync_mismatch_is_error(tmp_path: Path) -> None:
    info = _probe_info(audio_duration_s=8.0)
    report = _validator(info).validate(_video(tmp_path), expected=_expected())

    assert "qa.av.sync_mismatch" in _codes(report.errors)
    assert report.checks["av_sync"] is False


def test_peak_above_limit_is_warning(tmp_path: Path) -> None:
    report = _validator().validate(_video(tmp_path), expected=_expected(), peak=0.995)

    assert report.valid is True
    issue = next(issue for issue in report.warnings if issue.code == "qa.audio.clipping")
    assert issue.severity == "warning"


def test_extreme_peak_is_error(tmp_path: Path) -> None:
    report = _validator().validate(_video(tmp_path), expected=_expected(), peak=1.1)

    issue = next(issue for issue in report.errors if issue.code == "qa.audio.clipping")
    assert issue.severity == "error"
    assert report.valid is False


# --- timeline checks ------------------------------------------------------


def test_timeline_order_and_continuity_pass(tmp_path: Path) -> None:
    report = _validator().validate(
        _video(tmp_path), expected=_expected(), timeline=_two_scene_timeline()
    )

    assert report.checks["timeline_scene_order"] is True
    assert report.checks["timeline_continuity"] is True
    assert report.checks["final_frame_holds"] is True
    assert report.metrics["final_frame_hold_s"] == "0.5"
    assert report.valid is True


def test_timeline_order_error(tmp_path: Path) -> None:
    timeline = _Timeline([_scene("one", 3, "0", "5"), _scene("two", 9, "5", "10")])
    report = _validator().validate(_video(tmp_path), expected=_expected(), timeline=timeline)

    assert "qa.timeline.scene_order" in _codes(report.errors)


@pytest.mark.parametrize(
    ("start", "code"),
    [("6", "qa.timeline.gap"), ("4", "qa.timeline.overlap")],
)
def test_timeline_gap_and_overlap_are_errors(
    tmp_path: Path, start: str, code: str
) -> None:
    timeline = _Timeline(
        [
            _scene("one", 0, "0", "5"),
            _scene("two", 1, start, "10"),
        ]
    )
    report = _validator().validate(_video(tmp_path), expected=_expected(), timeline=timeline)

    assert code in _codes(report.errors)


def test_narration_truncation_is_error(tmp_path: Path) -> None:
    timeline = _Timeline([_scene("one", 0, "0", "10")], total="15.0")
    report = _validator().validate(
        _video(tmp_path), expected=_expected(), timeline=timeline
    )

    assert "qa.audio.narration_truncated" in _codes(report.errors)
    assert report.checks["narration_complete"] is False


# --- subtitles ------------------------------------------------------------


def _srt(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_srt_passes_subtitle_checks(tmp_path: Path) -> None:
    path = _srt(
        tmp_path / "subtitles.srt",
        "1\n00:00:01,000 --> 00:00:02,500\nHello there.\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nSecond cue.\n",
    )
    report = _validator().validate(
        _video(tmp_path), expected=_expected(), subtitle_paths=[path]
    )

    assert report.checks["subtitles_valid"] is True
    assert report.metrics["subtitle_cue_count"] == 2
    assert report.valid is True


@pytest.mark.parametrize(
    "body",
    [
        "1\n00:00:02,000 --> 00:00:01,000\nBad cue.\n",
        "1\n00:00:01,000 --> 00:00:02,000\nFirst.\n\n2\n00:00:00,500 --> 00:00:02,500\nOut of order.\n",
        "1\n00:00:09,000 --> 00:00:12,000\nToo long.\n",
        "1\n00:00:01,000 --> 00:00:02,000\n\n",
    ],
)
def test_invalid_subtitle_cues_are_errors(tmp_path: Path, body: str) -> None:
    path = _srt(tmp_path / "invalid.srt", body)
    report = _validator().validate(
        _video(tmp_path), expected=_expected(), subtitle_paths=[path]
    )

    assert report.valid is False
    assert report.checks["subtitles_valid"] is False


def test_vtt_without_header_is_error(tmp_path: Path) -> None:
    path = tmp_path / "subtitles.vtt"
    path.write_text("00:00:01.000 --> 00:00:02.000\nHello.\n", encoding="utf-8")
    report = _validator().validate(
        _video(tmp_path), expected=_expected(), subtitle_paths=[path]
    )

    assert "qa.subtitle.invalid_vtt_header" in _codes(report.errors)


@pytest.mark.parametrize(
    ("language", "severity"),
    [("hi", "error"), ("hinglish", "warning")],
)
def test_hindi_text_missing_is_reported(
    tmp_path: Path, language: str, severity: str
) -> None:
    path = _srt(
        tmp_path / "subtitles.srt",
        "1\n00:00:01,000 --> 00:00:02,000\nNamaste duniya.\n",
    )
    report = _validator().validate(
        _video(tmp_path),
        expected=_expected(language=language),
        subtitle_paths=[path],
    )

    issue = next(issue for issue in report.issues if issue.code == "qa.subtitle.hindi_text_lost")
    assert issue.severity == severity
    assert report.valid is (severity != "error")


def test_hindi_devanagari_subtitles_pass(tmp_path: Path) -> None:
    path = _srt(
        tmp_path / "subtitles.srt",
        "1\n00:00:01,000 --> 00:00:02,000\nनमस्ते दुनिया।\n",
    )
    report = _validator().validate(
        _video(tmp_path), expected=_expected(language="hi"), subtitle_paths=[path]
    )

    assert "qa.subtitle.hindi_text_lost" not in _codes(report.issues)


def test_burn_in_is_unverified_without_explicit_confirmation(tmp_path: Path) -> None:
    path = _srt(
        tmp_path / "subtitles.srt", "1\n00:00:01,000 --> 00:00:02,000\nHello.\n"
    )
    report = _validator().validate(
        _video(tmp_path), expected=_expected(), subtitle_paths=[path], burn_in=True
    )

    assert "qa.subtitle.burn_in_unverified" in _codes(report.warnings)
    assert report.checks["burn_in_verified"] is False
    assert report.valid is True


def test_burn_in_verified_flag_suppresses_warning(tmp_path: Path) -> None:
    path = _srt(
        tmp_path / "subtitles.srt", "1\n00:00:01,000 --> 00:00:02,000\nHello.\n"
    )
    report = _validator().validate(
        _video(tmp_path),
        expected=_expected(burn_in_verified=True),
        subtitle_paths=[path],
        burn_in=True,
    )

    assert "qa.subtitle.burn_in_unverified" not in _codes(report.issues)
    assert report.checks["burn_in_verified"] is True


def test_missing_subtitle_file_is_error(tmp_path: Path) -> None:
    report = _validator().validate(
        _video(tmp_path), expected=_expected(), subtitle_paths=[tmp_path / "gone.srt"]
    )

    assert "qa.subtitle.missing" in _codes(report.errors)


# --- detectors ------------------------------------------------------------


def test_black_and_silence_detectors_parse_intervals() -> None:
    runner = _runner(
        "[Parsed_blackdetect] black_start:1.000 black_end:4.000 black_duration:3.000\n"
        "[silencedetect] silence_start: 5.000\n"
        "[silencedetect] silence_end: 8.000 | silence_duration: 3.000\n"
    )
    settings = QAThresholds(min_silence_s=1.0, black_min_duration_s=2.0)

    black = detect_black("clip.mp4", thresholds=settings, command_runner=runner)
    silence = detect_silence("clip.mp4", thresholds=settings, command_runner=runner)

    assert black == [{"start_s": 1.0, "end_s": 4.0, "duration_s": 3.0}]
    assert silence == [{"start_s": 5.0, "end_s": 8.0, "duration_s": 3.0}]


def test_detector_commands_are_explicit_argument_lists() -> None:
    seen: list[list[str]] = []

    def run(command: Sequence[str], timeout_s: float) -> CommandResult:
        seen.append(list(command))
        return CommandResult(tuple(command), 0)

    detect_black("clip.mp4", command_runner=run)
    detect_silence("clip.mp4", command_runner=run)

    assert "blackdetect=d=" in seen[0][6]
    assert "silencedetect=noise=" in seen[1][6]
    assert seen[0][-2:] == ["-", ""] or seen[0][-1] == "-"
    assert all(isinstance(part, str) for command in seen for part in command)


def test_unexpected_black_and_silence_are_warnings(tmp_path: Path) -> None:
    stderr = (
        "black_start:1.000 black_end:4.000 black_duration:3.000\n"
        "silence_start: 5.000\nsilence_end: 8.000 | silence_duration: 3.000\n"
    )
    report = _validator(runner=_runner(stderr)).validate(
        _video(tmp_path), expected=_expected()
    )

    assert report.valid is True
    assert "qa.video.unexpected_black" in _codes(report.warnings)
    assert "qa.audio.unexpected_silence" in _codes(report.warnings)
    assert report.checks["detectors_run"] is True
    black_intervals = report.metrics["black_intervals"]
    silence_intervals = report.metrics["silence_intervals"]
    assert isinstance(black_intervals, list) and isinstance(silence_intervals, list)
    assert len(black_intervals) == 1
    assert len(silence_intervals) == 1


def test_intentional_silence_and_black_are_not_flagged(tmp_path: Path) -> None:
    stderr = (
        "black_start:1.000 black_end:4.000 black_duration:3.000\n"
        "silence_start: 8.000\nsilence_end: 9.000 | silence_duration: 1.000\n"
    )
    timeline = _Timeline(
        [
            _scene("one", 0, "0", "10", tail="2", narration_end="8"),
        ],
        total="10.0",
    )
    expected = _expected(
        intentional_black=[{"start_s": 0.0, "end_s": 5.0}]
    )
    report = _validator(runner=_runner(stderr)).validate(
        _video(tmp_path), expected=expected, timeline=timeline
    )

    assert "qa.video.unexpected_black" not in _codes(report.issues)
    assert "qa.audio.unexpected_silence" not in _codes(report.issues)
    assert report.valid is True


def test_detector_failure_is_warning_not_error(tmp_path: Path) -> None:
    report = _validator(runner=_failing_runner()).validate(
        _video(tmp_path), expected=_expected()
    )

    assert report.valid is True
    assert "qa.detector.blackdetect.unavailable" in _codes(report.warnings)
    assert "qa.detector.silencedetect.unavailable" in _codes(report.warnings)


def test_disabled_detectors_are_warnings(tmp_path: Path) -> None:
    report = _validator().validate(
        _video(tmp_path),
        expected=_expected(),
        thresholds={"run_detectors": False},
    )

    assert report.valid is True
    assert "qa.video.black_detect_disabled" in _codes(report.warnings)
    assert "qa.audio.silence_detect_disabled" in _codes(report.warnings)
    assert report.checks["detectors_run"] is False


def test_qa_status_wrapper_reports_errors(tmp_path: Path) -> None:
    status = qa_status(tmp_path / "missing.mp4", expected=_expected())

    assert status["status"] == "ok"
    result = status["result"]
    assert isinstance(result, dict)
    assert result["valid"] is False
    assert any(issue["code"] == "qa.file.missing" for issue in result["errors"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_real_ffmpeg_detectors_return_intervals(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=3:r=5",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo",
        "-t",
        "3",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    intervals = detect_black(path)

    assert any(interval["duration_s"] >= 2.0 for interval in intervals)
