"""Pure path, identifier and safe-write helpers for scene-based projects.

This module has no MCP, ComfyUI, TTS or FFmpeg dependencies.  Every helper
fails explicitly with :class:`PathValidationError` when a value could escape
the project directory.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
LANGUAGE_RE = re.compile(r"^[a-z]{2,8}(?:[-_][A-Za-z]{2,8})*$")
ASPECT_RATIO_RE = re.compile(r"^[1-9][0-9]*:[1-9][0-9]*$")


class PathValidationError(ValueError):
    """Raised when an identifier, timestamp or path is unsafe or invalid."""


def validate_identifier(value: object, label: str = "identifier") -> str:
    """Return *value* when it is a safe, portable identifier."""
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        raise PathValidationError(f"Invalid {label}: {value!r}")
    return value


def resolve_project_root(projects_root: str | Path, project_id: str) -> Path:
    """Resolve ``projects_root / project_id`` after rejecting traversal."""
    validate_identifier(project_id, "project_id")
    root = Path(projects_root).expanduser().resolve()
    candidate = (root / project_id).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise PathValidationError(f"Project path escapes projects root: {project_id!r}")
    return candidate


def resolve_within_project(
    project_root: str | Path,
    relative_path: str | Path,
    label: str = "asset path",
) -> Path:
    """Resolve an asset path, rejecting absolute paths outside the project."""
    if isinstance(relative_path, str) and not relative_path.strip():
        raise PathValidationError(f"Empty {label}")
    raw = Path(relative_path)
    if ".." in raw.parts:
        raise PathValidationError(f"{label} contains a parent segment: {relative_path!r}")
    root = Path(project_root).expanduser().resolve()
    candidate = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise PathValidationError(f"{label} escapes project directory: {relative_path!r}")
    return candidate


def to_project_relative(path: str | Path, project_root: str | Path) -> str:
    """Convert a path inside *project_root* to a portable relative string."""
    root = Path(project_root).expanduser().resolve()
    resolved = resolve_within_project(root, path)
    return resolved.relative_to(root).as_posix()


def is_iso8601_timestamp(value: object) -> bool:
    """Return True for an ISO 8601 timestamp with an explicit timezone."""
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = value.strip()
    if candidate.endswith(("Z", "z")):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def is_valid_language(value: object) -> bool:
    """Accept short language codes such as ``en``, ``en-US`` or ``hinglish``."""
    return isinstance(value, str) and LANGUAGE_RE.fullmatch(value) is not None


def is_valid_aspect_ratio(value: object) -> bool:
    """Accept aspect ratios in ``W:H`` form with positive integers."""
    return isinstance(value, str) and ASPECT_RATIO_RE.fullmatch(value) is not None


def write_json_atomic(path: str | Path, payload: Any) -> None:
    """Write JSON via a same-directory temp file, then promote it atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
