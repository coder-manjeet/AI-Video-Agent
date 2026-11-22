"""Machine-readable and human-readable production reports for final renders.

The report builder is additive and read-only: it converts existing job, audio,
subtitle, video and QA results into a portable payload.  Absolute machine paths
are reduced to project-relative POSIX strings (or just a basename outside the
project), and voice/reference data is never copied into the report.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from project_paths import PathValidationError, to_project_relative, write_json_atomic

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
REPORT_JSON_NAME = "render_report.json"
REPORT_MARKDOWN_NAME = "render_report.md"

_SENSITIVE_KEY_RE = re.compile(
    r"voice|reference|token|secret|password|api[_-]?key|credential|voiceprint",
    re.IGNORECASE,
)
_ABS_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/]|/)[^\s\"'<>|]+"
)


def _as_mapping(value: object) -> dict[str, object]:
    """Best-effort conversion of a model, mapping or dataclass to a dict."""
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    return {}


def _first(value: object, *keys: str) -> object:
    mapping = _as_mapping(value)
    for key in keys:
        candidate = mapping.get(key)
        if candidate is not None:
            return candidate
    return None


def _text(value: object, *, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return str(value)
    return default


def _sanitize_path(value: object, project_root: Path) -> str:
    """Return a safe, portable representation of *value*."""
    if value is None:
        return ""
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value.strip():
        path = Path(value.strip())
    else:
        return ""
    if ".." in path.parts:
        return path.name
    if path.is_absolute():
        try:
            return to_project_relative(path, project_root)
        except (PathValidationError, ValueError, OSError):
            return path.name
    return path.as_posix()


def _scrub_text(text: str, project_root: Path) -> str:
    """Remove machine paths from free-text messages."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(0).rstrip(".,;:")
        suffix = match.group(0)[len(token) :]
        return _sanitize_path(token, project_root) + suffix

    return _ABS_PATH_RE.sub(replace, text)


def _json_safe(value: object, project_root: Path, *, key: str = "") -> object:
    """Convert arbitrary results into a JSON-safe, privacy-preserving value."""
    if _SENSITIVE_KEY_RE.search(key):
        return None
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Path):
        return _sanitize_path(value, project_root)
    if isinstance(value, str):
        return _scrub_text(value, project_root)
    if isinstance(value, Mapping):
        return {
            str(child_key): result
            for child_key, child_value in value.items()
            if (result := _json_safe(child_value, project_root, key=str(child_key))) is not None
            or not _SENSITIVE_KEY_RE.search(str(child_key))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, project_root) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(_as_mapping(value), project_root, key=key)
    return str(value)


def _warning_entries(values: Sequence[object], project_root: Path) -> list[dict[str, object]]:
    """Normalize QA issues and plain warnings into portable warning entries."""
    result: list[dict[str, object]] = []
    for value in values:
        if isinstance(value, Mapping):
            raw = dict(value)
        elif isinstance(value, str):
            raw = {"message": value}
        else:
            raw = _as_mapping(value)
            if not raw:
                raw = {"message": str(value)}
        entry: dict[str, object] = {}
        for key in ("severity", "code", "message", "location", "timestamp_s", "scene_id"):
            if key in raw and raw[key] is not None:
                entry[key] = _json_safe(raw[key], project_root, key=key)
        if not entry:
            entry["message"] = _scrub_text(str(value), project_root)
        result.append(entry)
    return result


def _collect_warnings(
    sources: Sequence[object], project_root: Path
) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []

    def visit(source: object) -> None:
        if source is None:
            return
        if isinstance(source, (list, tuple)):
            for item in source:
                visit(item)
            return
        mapping = _as_mapping(source)
        if "errors" in mapping or "warnings" in mapping:
            for key in ("errors", "warnings"):
                raw = mapping.get(key)
                if isinstance(raw, (list, tuple)):
                    warnings.extend(_warning_entries(list(raw), project_root))
        elif mapping:
            warnings.extend(_warning_entries([source], project_root))
        else:
            warnings.append({"message": _scrub_text(str(source), project_root)})

    for source in sources:
        visit(source)
    return warnings


def _scene_entries(source: object, project_root: Path) -> list[dict[str, object]]:
    """Return portable per-scene duration entries from a timeline or plan."""
    mapping = _as_mapping(source)
    raw_scenes = mapping.get("scenes")
    if not isinstance(raw_scenes, (list, tuple)):
        return []
    entries: list[dict[str, object]] = []
    for index, raw_scene in enumerate(raw_scenes):
        scene = _as_mapping(raw_scene)
        order = scene.get("order", scene.get("scene_number", index))
        entry: dict[str, object] = {
            "scene_id": _text(scene.get("scene_id"), default=f"scenes[{index}]"),
            "order": _json_safe(order, project_root, key="order"),
            "title": _text(scene.get("title")),
            "planned_duration_s": _json_safe(
                _first(scene, "planned_duration_s", "video_duration_s", "effective_duration_s"),
                project_root,
                key="planned_duration_s",
            ),
            "measured_duration_s": _json_safe(
                _first(scene, "measured_duration_s", "video_duration_s"),
                project_root,
                key="measured_duration_s",
            ),
            "narration_duration_s": _json_safe(
                _first(scene, "narration_duration_s"),
                project_root,
                key="narration_duration_s",
            ),
            "hold_duration_s": _json_safe(
                _first(scene, "final_frame_hold_s", "hold_duration_s", "video_hold_duration_s"),
                project_root,
                key="hold_duration_s",
            ),
            "silent_tail_s": _json_safe(
                _first(scene, "silent_tail_s", "silent_tail_duration_s"),
                project_root,
                key="silent_tail_s",
            ),
            "scene_order": _json_safe(order, project_root, key="scene_order"),
        }
        entries.append(entry)
    return entries


def _audio_entry(audio_report: object | None, project_root: Path) -> dict[str, object]:
    report = _as_mapping(audio_report)
    mix = _as_mapping(_first(report, "mix") or {})
    arrangement = _as_mapping(_first(report, "arrangement") or {})
    sfx_paths_value = arrangement.get("sfx_paths")
    sfx_paths = sfx_paths_value if isinstance(sfx_paths_value, (list, tuple)) else ()
    entry: dict[str, object] = {
        "master_path": _sanitize_path(
            _first(report, "audio_master_path", "master_path") or mix.get("master_path"),
            project_root,
        ),
        "duration_s": _json_safe(
            _first(report, "audio_duration_s", "master_duration_s") or mix.get("master_duration_s"),
            project_root,
            key="duration_s",
        ),
        "sample_rate": _json_safe(
            _first(mix, "sample_rate") or report.get("sample_rate"),
            project_root,
            key="sample_rate",
        ),
        "channels": _json_safe(
            _first(mix, "channels") or report.get("channels"), project_root, key="channels"
        ),
        "music": _json_safe(
            {
                "inputs": _first(mix, "music_inputs", "music_count"),
                "path": _sanitize_path(_first(arrangement, "music_path"), project_root),
            },
            project_root,
            key="music",
        ),
        "sfx": _json_safe(
            {
                "inputs": _first(mix, "sfx_inputs", "sfx_count"),
                "paths": [_sanitize_path(item, project_root) for item in sfx_paths],
            },
            project_root,
            key="sfx",
        ),
        "narration_inputs": _json_safe(
            mix.get("narration_inputs"), project_root, key="narration_inputs"
        ),
        "ducking_applied": _json_safe(
            mix.get("ducking_applied"), project_root, key="ducking_applied"
        ),
        "peak": _json_safe(mix.get("peak"), project_root, key="peak"),
        "clipped_samples": _json_safe(
            mix.get("clipped_samples"), project_root, key="clipped_samples"
        ),
    }
    return entry


def _subtitle_entry(
    subtitle_result: object | None, project_root: Path
) -> dict[str, object]:
    result = _as_mapping(subtitle_result)
    formats = result.get("formats") or (
        [result.get("format")] if result.get("format") is not None else []
    )
    paths = result.get("paths") or (
        [result.get("path")] if result.get("path") is not None else []
    )
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    if not isinstance(formats, (list, tuple)):
        formats = [formats]
    warnings = result.get("warnings") or []
    if not isinstance(warnings, (list, tuple)):
        warnings = [warnings]
    return {
        "formats": [_text(item) for item in formats if item is not None],
        "paths": [_sanitize_path(item, project_root) for item in paths],
        "language": _text(result.get("language")),
        "granularity": _text(result.get("granularity")),
        "warnings": _warning_entries(list(warnings), project_root),
    }


def _video_entry(final_result: object | None, project_root: Path) -> dict[str, object]:
    result = _as_mapping(final_result)
    probe = _as_mapping(_first(result, "probe", "video") or {})
    video = _as_mapping(probe.get("video") or {}) if isinstance(probe, Mapping) else {}
    entry: dict[str, object] = {
        "output_path": _sanitize_path(
            _first(result, "output_path", "final_video_path", "path")
            or probe.get("path"),
            project_root,
        ),
        "video_codec": _json_safe(
            _first(result, "video_codec", "codec") or probe.get("video_codec") or video.get("codec"),
            project_root,
            key="video_codec",
        ),
        "audio_codec": _json_safe(
            _first(result, "audio_codec") or probe.get("audio_codec"),
            project_root,
            key="audio_codec",
        ),
        "width": _json_safe(
            _first(result, "width") or probe.get("width"), project_root, key="width"
        ),
        "height": _json_safe(
            _first(result, "height") or probe.get("height"), project_root, key="height"
        ),
        "resolution": _text(
            _first(result, "resolution")
            or (
                f"{probe.get('width')}x{probe.get('height')}"
                if probe.get("width") and probe.get("height")
                else ""
            )
        ),
        "fps": _json_safe(_first(result, "fps") or probe.get("fps"), project_root, key="fps"),
        "pixel_format": _json_safe(
            _first(result, "pixel_format") or probe.get("pixel_format"),
            project_root,
            key="pixel_format",
        ),
        "duration_s": _json_safe(
            _first(result, "duration_s", "final_duration_s") or probe.get("duration_s"),
            project_root,
            key="duration_s",
        ),
        "size_bytes": _json_safe(
            _first(result, "size_bytes") or probe.get("size_bytes"),
            project_root,
            key="size_bytes",
        ),
    }
    return entry


def _stage_entry(stage: object, project_root: Path) -> dict[str, object]:
    raw = _as_mapping(stage)
    if not raw:
        return {"name": _text(stage), "status": "unknown"}
    return {
        "name": _text(_first(raw, "name", "stage", "id")),
        "status": _text(_first(raw, "status", "state"), default="unknown"),
        "reason": _scrub_text(_text(_first(raw, "reason", "error", "message")), project_root),
    }


def _failed_stages(job: Mapping[str, object], project_root: Path) -> list[dict[str, object]]:
    stages: list[object] = []
    for key in ("failed_or_skipped_stages", "failed_stages", "skipped_stages", "stages"):
        value = job.get(key)
        if isinstance(value, (list, tuple)):
            stages.extend(list(value))
    entries: list[dict[str, object]] = []
    for stage in stages:
        raw = _as_mapping(stage)
        if raw:
            status = _text(_first(raw, "status", "state"), default="").lower()
            failed = status in {"failed", "error", "skipped", "cancelled"}
            failed = failed or any(
                raw.get(key) is True for key in ("failed", "skipped", "cancelled")
            )
            if not failed:
                continue
        elif isinstance(stage, str):
            status = "failed"
        else:
            continue
        entry = _stage_entry(stage, project_root)
        entry.setdefault("status", status)
        entries.append(entry)
    return entries


def build_render_report(
    *,
    project: object,
    config: Mapping[str, object],
    job: Mapping[str, object],
    audio_report: object | None,
    subtitle_result: object | None,
    final_result: object | None,
    qa_report: object | None,
    project_root: str | Path,
    plan: object | None = None,
    timeline: object | None = None,
    stages: Sequence[Mapping[str, object]] = (),
    started_at: str = "",
    completed_at: str = "",
) -> dict[str, object]:
    """Build a portable, machine-readable render report payload."""
    root = Path(project_root).expanduser().resolve()
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(job, Mapping):
        raise TypeError("job must be a mapping")
    project_info = _as_mapping(project)
    job_info = dict(job)
    stage_values: list[object] = list(stages)
    raw_job_stages = job_info.get("stages")
    if isinstance(raw_job_stages, (list, tuple)):
        stage_values.extend(list(raw_job_stages))
    if "stages" not in job_info and stages:
        job_info["stages"] = list(stages)
    timeline_source = timeline if timeline is not None else plan
    qa_payload: object = qa_report
    if qa_report is not None and hasattr(qa_report, "to_dict"):
        qa_payload = qa_report.to_dict()

    warnings = _collect_warnings(
        [
            job_info.get("warnings"),
            _as_mapping(audio_report).get("warnings") if audio_report is not None else None,
            _as_mapping(subtitle_result).get("warnings") if subtitle_result is not None else None,
            _as_mapping(final_result).get("warnings") if final_result is not None else None,
            qa_payload,
            plan,
        ],
        root,
    )
    for stage in stage_values:
        stage_warnings = _as_mapping(stage).get("warnings")
        if isinstance(stage_warnings, (list, tuple)):
            warnings.extend(_warning_entries(list(stage_warnings), root))
    video = _video_entry(final_result, root)
    audio = _audio_entry(audio_report, root)
    subtitles = _subtitle_entry(subtitle_result, root)

    outputs: dict[str, object] = {
        "final_video": video.get("output_path") or "",
        "audio_master": audio.get("master_path") or "",
    }
    for index, path in enumerate(subtitles["paths"] if isinstance(subtitles["paths"], list) else []):
        outputs[f"subtitle_{index + 1}"] = path
    raw_outputs = job_info.get("outputs")
    if isinstance(raw_outputs, Mapping):
        for key, value in raw_outputs.items():
            outputs[str(key)] = _sanitize_path(value, root)
    elif isinstance(raw_outputs, (list, tuple)):
        for index, value in enumerate(raw_outputs):
            outputs[f"output_{index + 1}"] = _sanitize_path(value, root)
    for key in ("report_json", "json_report", "report_path", "report_md", "markdown_report"):
        value = job_info.get(key)
        if value is not None:
            output_key = "report_json" if "json" in key or key == "report_path" else "report_md"
            outputs[output_key] = _sanitize_path(value, root)

    generated_at = datetime.now(UTC).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "project": {
            "project_id": _text(
                _first(project_info, "project_id", "id") or job_info.get("project_id")
            ),
            "title": _text(project_info.get("title")),
            "language": _text(project_info.get("language")),
            "status": _text(project_info.get("status")),
        },
        "render": {
            "job_id": _text(_first(job_info, "job_id", "id")),
            "render_id": _text(
                _first(job_info, "render_id")
                or _as_mapping(audio_report).get("render_id")
            ),
            "status": _text(job_info.get("status"), default="unknown"),
            "config": _json_safe(config, root, key="config"),
            "started_at": _text(started_at or job_info.get("started_at")),
            "completed_at": _text(completed_at or job_info.get("completed_at")),
        },
        "scenes": _scene_entries(timeline_source, root),
        "audio": audio,
        "subtitles": subtitles,
        "video": video,
        "validation": (
            _json_safe(qa_payload, root, key="validation")
            if qa_payload is not None
            else {"valid": None, "errors": [], "warnings": []}
        ),
        "warnings": warnings,
        "failed_or_skipped_stages": _failed_stages(job_info, root),
        "outputs": outputs,
    }


def _markdown_value(value: object) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value, Path.cwd()), ensure_ascii=False, sort_keys=True)
    return str(value)


def render_report_markdown(payload: Mapping[str, object]) -> str:
    """Return a readable Markdown summary of a render report payload."""
    project = _as_mapping(payload.get("project"))
    render = _as_mapping(payload.get("render"))
    video = _as_mapping(payload.get("video"))
    audio = _as_mapping(payload.get("audio"))
    subtitles = _as_mapping(payload.get("subtitles"))
    validation = _as_mapping(payload.get("validation"))
    validation_errors = validation.get("errors")
    validation_warnings = validation.get("warnings")
    error_count = len(validation_errors) if isinstance(validation_errors, (list, tuple)) else 0
    warning_count = (
        len(validation_warnings) if isinstance(validation_warnings, (list, tuple)) else 0
    )
    lines: list[str] = ["# Render report", ""]
    lines.extend(
        [
            f"Generated: {_markdown_value(payload.get('generated_at'))}",
            "",
            "## Project",
            "",
            f"- ID: {_markdown_value(project.get('project_id'))}",
            f"- Title: {_markdown_value(project.get('title'))}",
            f"- Language: {_markdown_value(project.get('language'))}",
            f"- Status: {_markdown_value(project.get('status'))}",
            "",
            "## Render config",
            "",
            f"- Job: {_markdown_value(render.get('job_id'))}",
            f"- Render: {_markdown_value(render.get('render_id'))}",
            f"- Status: {_markdown_value(render.get('status'))}",
            f"- Started: {_markdown_value(render.get('started_at'))}",
            f"- Completed: {_markdown_value(render.get('completed_at'))}",
            f"- Config: {_markdown_value(render.get('config'))}",
            "",
            "## Scene order and durations",
            "",
        ]
    )
    scenes = payload.get("scenes")
    if isinstance(scenes, (list, tuple)) and scenes:
        lines.extend(
            [
                "| Order | Scene | Title | Planned (s) | Measured (s) | Narration (s) | Hold (s) | Silent tail (s) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for raw_scene in scenes:
            scene = _as_mapping(raw_scene)
            lines.append(
                "| {order} | {scene_id} | {title} | {planned} | {measured} | {narration} | {hold} | {tail} |".format(
                    order=_markdown_value(scene.get("order")),
                    scene_id=_markdown_value(scene.get("scene_id")),
                    title=_markdown_value(scene.get("title")),
                    planned=_markdown_value(scene.get("planned_duration_s")),
                    measured=_markdown_value(scene.get("measured_duration_s")),
                    narration=_markdown_value(scene.get("narration_duration_s")),
                    hold=_markdown_value(scene.get("hold_duration_s")),
                    tail=_markdown_value(scene.get("silent_tail_s")),
                )
            )
    else:
        lines.append("No scene timeline recorded.")
    lines.extend(
        [
            "",
            "## Audio",
            "",
            f"- Master: {_markdown_value(audio.get('master_path'))}",
            f"- Duration (s): {_markdown_value(audio.get('duration_s'))}",
            f"- Sample rate: {_markdown_value(audio.get('sample_rate'))}",
            f"- Channels: {_markdown_value(audio.get('channels'))}",
            f"- Music: {_markdown_value(audio.get('music'))}",
            f"- SFX: {_markdown_value(audio.get('sfx'))}",
            f"- Peak: {_markdown_value(audio.get('peak'))}",
            "",
            "## Subtitles",
            "",
            f"- Formats: {_markdown_value(subtitles.get('formats'))}",
            f"- Paths: {_markdown_value(subtitles.get('paths'))}",
            f"- Language: {_markdown_value(subtitles.get('language'))}",
            f"- Granularity: {_markdown_value(subtitles.get('granularity'))}",
            f"- Warnings: {_markdown_value(subtitles.get('warnings'))}",
            "",
            "## Final output",
            "",
            f"- Path: {_markdown_value(video.get('output_path'))}",
            f"- Video codec: {_markdown_value(video.get('video_codec'))}",
            f"- Audio codec: {_markdown_value(video.get('audio_codec'))}",
            f"- Resolution: {_markdown_value(video.get('resolution'))}",
            f"- FPS: {_markdown_value(video.get('fps'))}",
            f"- Pixel format: {_markdown_value(video.get('pixel_format'))}",
            f"- Duration (s): {_markdown_value(video.get('duration_s'))}",
            f"- Size (bytes): {_markdown_value(video.get('size_bytes'))}",
            "",
            "## Validation",
            "",
            f"- Valid: {_markdown_value(validation.get('valid'))}",
            f"- Errors: {error_count}",
            f"- Warnings: {warning_count}",
            f"- Details: {_markdown_value(validation)}",
            "",
            "## Warnings and limitations",
            "",
        ]
    )
    warnings = payload.get("warnings")
    if isinstance(warnings, (list, tuple)) and warnings:
        for warning in warnings:
            lines.append(f"- {_markdown_value(warning)}")
    else:
        lines.append("No warnings recorded.")
    lines.extend(["", "## Failed or skipped stages", ""])
    stage_entries = payload.get("failed_or_skipped_stages")
    if isinstance(stage_entries, (list, tuple)) and stage_entries:
        for stage in stage_entries:
            raw = _as_mapping(stage)
            lines.append(
                f"- {_markdown_value(raw.get('name'))}: {_markdown_value(raw.get('status'))}"
                f" — {_markdown_value(raw.get('reason'))}"
            )
    else:
        lines.append("No failed or skipped stages.")
    lines.extend(["", "## Outputs", ""])
    outputs = _as_mapping(payload.get("outputs"))
    for key, value in outputs.items():
        lines.append(f"- {key}: {_markdown_value(value)}")
    lines.append("")
    return "\n".join(lines)


def _scrub_payload(value: object) -> object:
    """Remove any absolute path that slipped through the report builder."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(0).rstrip(".,;:")
        suffix = match.group(0)[len(token) :]
        return Path(token).name + suffix

    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        return _ABS_PATH_RE.sub(replace, value)
    if isinstance(value, Mapping):
        return {str(key): _scrub_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_payload(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def write_render_report(
    render_dir: str | Path, payload: Mapping[str, object]
) -> tuple[Path, Path]:
    """Atomically write JSON and Markdown reports into one render directory."""
    directory = Path(render_dir).expanduser().resolve()
    json_path = directory / REPORT_JSON_NAME
    md_path = directory / REPORT_MARKDOWN_NAME
    if json_path.exists() or md_path.exists():
        raise FileExistsError(
            "render.report.collision: a completed report already exists in "
            f"{directory.name}; use a unique render directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    safe_payload = _scrub_payload(payload)
    markdown = render_report_markdown(safe_payload) if isinstance(safe_payload, Mapping) else ""
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{REPORT_MARKDOWN_NAME}.", suffix=".tmp", dir=str(directory)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(markdown)
            handle.flush()
            os.fsync(handle.fileno())
        write_json_atomic(json_path, safe_payload)
        if md_path.exists():
            raise FileExistsError(
                f"render.report.collision: {md_path.name} appeared during report write"
            )
        temp_path.replace(md_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    logger.info("Wrote render reports in %s", directory.name)
    return json_path, md_path
