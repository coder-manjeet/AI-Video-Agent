"""Deterministic tests for portable render reports."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from final_qa import QAIssue, QAReport
from render_report import (
    build_render_report,
    render_report_markdown,
    write_render_report,
)


def _payload(root: Path, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "project": {
            "project_id": "demo",
            "title": "Demo film",
            "language": "en",
            "status": "active",
        },
        "config": {
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "narrator_voice": "private-clone",
            "reference_wav": str(root / "assets" / "voice" / "sample.wav"),
            "duration_s": Decimal("12.5"),
        },
        "job": {
            "job_id": "job-1",
            "render_id": "render-1",
            "status": "completed",
            "started_at": "2026-11-22T10:00:00+00:00",
            "completed_at": "2026-11-22T10:01:00+00:00",
            "outputs": {"extra": str(root / "video" / "renders" / "extra.mp4")},
        },
        "audio_report": {
            "audio_master_path": str(root / "audio" / "renders" / "audio_master.wav"),
            "audio_duration_s": Decimal("12.5"),
            "mix": {
                "master_path": str(root / "audio" / "renders" / "audio_master.wav"),
                "master_duration_s": Decimal("12.5"),
                "sample_rate": 48000,
                "channels": 2,
                "narration_inputs": 2,
                "music_inputs": 1,
                "sfx_inputs": 3,
                "ducking_applied": True,
                "peak": 0.82,
                "clipped_samples": 0,
                "commands": [{"command": [str(root / "bin" / "ffmpeg")]}],
            },
            "warnings": [{"code": "audio.warning", "message": "quiet bed"}],
        },
        "subtitle_result": {
            "formats": ["srt", "vtt"],
            "paths": [
                str(root / "video" / "renders" / "subtitles.srt"),
                str(root / "video" / "renders" / "subtitles.vtt"),
            ],
            "language": "en",
            "granularity": "scene",
            "warnings": [{"code": "sub.warn", "message": "review cue 3"}],
        },
        "final_result": {
            "output_path": str(root / "video" / "renders" / "final.mp4"),
            "video_codec": "h264",
            "audio_codec": "aac",
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "pixel_format": "yuv420p",
            "duration_s": Decimal("12.5"),
            "size_bytes": 987654,
        },
        "qa_report": QAReport(
            (
                QAIssue("warning", "qa.audio.clipping", "Peak close to limit"),
            ),
            {"valid_streams": True},
            {"peak": 0.95},
        ),
        "timeline": {
            "total_duration_s": Decimal("12.5"),
            "scenes": [
                {
                    "scene_id": "one",
                    "order": 0,
                    "title": "Opening",
                    "video_duration_s": Decimal(6),
                    "measured_duration_s": Decimal(6),
                    "narration_duration_s": Decimal(5),
                    "final_frame_hold_s": Decimal("0.5"),
                    "silent_tail_s": Decimal(1),
                },
                {
                    "scene_id": "two",
                    "order": 1,
                    "title": "Closing",
                    "video_duration_s": Decimal("6.5"),
                    "measured_duration_s": Decimal("6.5"),
                    "narration_duration_s": Decimal("6.5"),
                    "final_frame_hold_s": Decimal(0),
                    "silent_tail_s": Decimal(0),
                },
            ],
        },
        "stages": [
            {"name": "prepare", "status": "completed"},
            {"name": "mux", "status": "skipped", "reason": "manual review"},
        ],
        "started_at": "2026-11-22T10:00:00+00:00",
        "completed_at": "2026-11-22T10:01:00+00:00",
    }
    payload.update(overrides)
    return payload


def _build(root: Path, **overrides: object) -> dict[str, object]:
    args = _payload(root, **overrides)
    return build_render_report(
        project=args["project"],  # type: ignore[arg-type]
        config=args["config"],  # type: ignore[arg-type]
        job=args["job"],  # type: ignore[arg-type]
        audio_report=args["audio_report"],
        subtitle_result=args["subtitle_result"],
        final_result=args["final_result"],
        qa_report=args["qa_report"],
        project_root=root,
        plan=args.get("plan"),
        timeline=args.get("timeline"),
        stages=args["stages"],  # type: ignore[arg-type]
        started_at=str(args["started_at"]),
        completed_at=str(args["completed_at"]),
    )


def _flatten(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def test_payload_has_contract_keys_and_schema(tmp_path: Path) -> None:
    payload = _build(tmp_path)

    assert payload["schema_version"] == 1
    for key in (
        "generated_at",
        "project",
        "render",
        "scenes",
        "audio",
        "subtitles",
        "video",
        "validation",
        "warnings",
        "failed_or_skipped_stages",
        "outputs",
    ):
        assert key in payload
    assert str(payload["generated_at"]).endswith("+00:00")


def test_project_and_render_sections(tmp_path: Path) -> None:
    render = _build(tmp_path)["render"]
    assert isinstance(render, dict)
    assert render["job_id"] == "job-1"
    assert render["render_id"] == "render-1"
    assert render["status"] == "completed"
    assert render["started_at"] == "2026-11-22T10:00:00+00:00"


def test_paths_inside_project_are_relative_posix(tmp_path: Path) -> None:
    payload = _build(tmp_path)
    video = payload["video"]
    assert isinstance(video, dict)
    assert video["output_path"] == "video/renders/final.mp4"
    audio = payload["audio"]
    assert isinstance(audio, dict)
    assert audio["master_path"] == "audio/renders/audio_master.wav"
    subtitles = payload["subtitles"]
    assert isinstance(subtitles, dict)
    assert subtitles["paths"] == [
        "video/renders/subtitles.srt",
        "video/renders/subtitles.vtt",
    ]


def test_paths_outside_project_are_reduced_to_basename(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside" / "private.mp4"
    payload = _build(
        tmp_path,
        final_result={"output_path": str(outside), "video_codec": "h264"},
    )
    video = payload["video"]
    assert isinstance(video, dict)
    assert video["output_path"] == "private.mp4"


def test_voice_reference_and_machine_paths_are_not_exported(tmp_path: Path) -> None:
    payload = _build(tmp_path)
    text = _flatten(payload)

    assert "private-clone" not in text
    assert "sample.wav" not in text
    assert "reference_wav" not in text
    assert str(tmp_path) not in text
    assert "ffmpeg" not in text


def test_decimals_are_serialized_as_strings(tmp_path: Path) -> None:
    payload = _build(tmp_path)
    text = _flatten(payload)

    assert '"duration_s": "12.5"' in text
    assert '"planned_duration_s": "6"' in text
    assert "Decimal(" not in text


def test_scenes_come_from_timeline_with_expected_fields(tmp_path: Path) -> None:
    scenes = _build(tmp_path)["scenes"]
    assert isinstance(scenes, list)
    assert len(scenes) == 2
    first = scenes[0]
    assert isinstance(first, dict)
    assert first["scene_id"] == "one"
    assert first["order"] == 0
    assert first["scene_order"] == 0
    assert first["planned_duration_s"] == "6"
    assert first["narration_duration_s"] == "5"
    assert first["hold_duration_s"] == "0.5"
    assert first["silent_tail_s"] == "1"


def test_audio_section_has_counts_but_no_command_paths(tmp_path: Path) -> None:
    audio = _build(tmp_path)["audio"]
    assert isinstance(audio, dict)
    assert audio["sample_rate"] == 48000
    assert audio["channels"] == 2
    assert audio["narration_inputs"] == 2
    assert audio["ducking_applied"] is True
    assert "commands" not in audio


def test_validation_uses_qa_report_dict(tmp_path: Path) -> None:
    payload = _build(tmp_path)
    validation = payload["validation"]
    assert isinstance(validation, dict)
    assert validation["valid"] is True
    assert validation["checks"] == {"valid_streams": True}
    assert len(validation["warnings"]) == 1


def test_warnings_are_combined_from_all_sources(tmp_path: Path) -> None:
    warnings = _build(tmp_path)["warnings"]
    assert isinstance(warnings, list)
    codes = {entry.get("code") for entry in warnings if isinstance(entry, dict)}
    assert {"qa.audio.clipping", "audio.warning", "sub.warn"} <= codes


def test_failed_or_skipped_stages_are_derived(tmp_path: Path) -> None:
    stages = _build(tmp_path)["failed_or_skipped_stages"]
    assert isinstance(stages, list)
    assert len(stages) == 1
    assert stages[0]["name"] == "mux"
    assert stages[0]["status"] == "skipped"


def test_markdown_contains_all_required_sections(tmp_path: Path) -> None:
    markdown = render_report_markdown(_build(tmp_path))

    for heading in (
        "# Render report",
        "## Project",
        "## Render config",
        "## Scene order and durations",
        "## Audio",
        "## Subtitles",
        "## Final output",
        "## Validation",
        "## Warnings and limitations",
        "## Failed or skipped stages",
        "## Outputs",
    ):
        assert heading in markdown
    assert "render-1" in markdown
    assert "one" in markdown


def test_write_render_report_creates_both_files(tmp_path: Path) -> None:
    render_dir = tmp_path / "video" / "renders" / "render-1"
    payload = _build(tmp_path)

    json_path, md_path = write_render_report(render_dir, payload)

    assert json_path == render_dir / "render_report.json"
    assert md_path == render_dir / "render_report.md"
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == 1
    assert str(tmp_path) not in md_path.read_text(encoding="utf-8")


def test_write_render_report_refuses_collision(tmp_path: Path) -> None:
    render_dir = tmp_path / "render-1"
    payload = _build(tmp_path)
    write_render_report(render_dir, payload)

    with pytest.raises(FileExistsError, match="render.report.collision"):
        write_render_report(render_dir, payload)


def test_build_render_report_rejects_non_mappings(tmp_path: Path) -> None:
    args = _payload(tmp_path)
    args["config"] = ["not-a-mapping"]
    with pytest.raises(TypeError, match="config must be a mapping"):
        build_render_report(
            project=args["project"],  # type: ignore[arg-type]
            config=args["config"],  # type: ignore[arg-type]
            job=args["job"],  # type: ignore[arg-type]
            audio_report=None,
            subtitle_result=None,
            final_result=None,
            qa_report=None,
            project_root=tmp_path,
        )


def test_report_handles_missing_stage_results(tmp_path: Path) -> None:
    args = _payload(tmp_path)
    payload = build_render_report(
        project=args["project"],  # type: ignore[arg-type]
        config=args["config"],  # type: ignore[arg-type]
        job=args["job"],  # type: ignore[arg-type]
        audio_report=None,
        subtitle_result=None,
        final_result=None,
        qa_report=None,
        project_root=tmp_path,
    )

    assert payload["validation"] == {"valid": None, "errors": [], "warnings": []}
    assert payload["failed_or_skipped_stages"] == []
    assert payload["warnings"] == []
