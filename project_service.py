"""Project creation, loading and metadata updates.

Business logic is intentionally independent of MCP and UI layers.  Manifests
are written atomically and existing projects are never overwritten.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from character_models import VISUAL_IDENTITY_FIELDS, VisualIdentity
from manifest_validation import (
    PROJECT_FILENAME,
    validate_project_metadata,
)
from project_models import (
    DEFAULT_LANGUAGE,
    DEFAULT_PROJECT_STATUS,
    PROJECT_STATUSES,
    Project,
    ProjectSettings,
)
from project_paths import resolve_project_root, write_json_atomic

logger = logging.getLogger(__name__)

PROJECT_DIRECTORIES = (
    Path("script"),
    Path("assets") / "images",
    Path("assets") / "characters",
    Path("audio") / "narration",
    Path("audio") / "music",
    Path("audio") / "sfx",
    Path("video") / "scenes",
    Path("video") / "final",
    Path("subtitles"),
    Path("logs"),
)
PROJECT_FILES = (Path("script") / "narration.md",)


class ProjectError(RuntimeError):
    """Base error for project service failures."""


class ProjectExistsError(ProjectError):
    """Raised when a project directory or manifest already exists."""


class ProjectNotFoundError(ProjectError):
    """Raised when a project manifest cannot be found."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_settings(
    settings: ProjectSettings | Mapping[str, Any] | None,
) -> ProjectSettings:
    if settings is None:
        return ProjectSettings()
    if isinstance(settings, ProjectSettings):
        return settings
    if isinstance(settings, Mapping):
        return ProjectSettings.from_dict(settings)
    raise TypeError("settings must be a ProjectSettings instance or a mapping")


class ProjectService:
    """Manage project directories and ``project.json`` manifests."""

    def __init__(self, projects_root: str | Path | None = None) -> None:
        root = Path(projects_root) if projects_root is not None else Path.cwd() / "projects"
        self.projects_root = root.expanduser().resolve()

    def project_dir(self, project_id: str) -> Path:
        """Return the validated absolute path of a project directory."""
        return resolve_project_root(self.projects_root, project_id)

    def project_file(self, project_id: str) -> Path:
        """Return the validated absolute path of ``project.json``."""
        return self.project_dir(project_id) / PROJECT_FILENAME

    def exists(self, project_id: str) -> bool:
        """Return True when the project manifest exists."""
        return self.project_file(project_id).is_file()

    def list_projects(self) -> list[str]:
        """List IDs of projects with a valid ``project.json`` file."""
        if not self.projects_root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self.projects_root.iterdir()
            if entry.is_dir() and (entry / PROJECT_FILENAME).is_file()
        )

    def create_project(
        self,
        project_id: str,
        title: str,
        *,
        description: str = "",
        language: str = DEFAULT_LANGUAGE,
        status: str = DEFAULT_PROJECT_STATUS,
        settings: ProjectSettings | Mapping[str, Any] | None = None,
    ) -> Project:
        """Create a new project and all required directories.

        Raises:
            PathValidationError: If *project_id* is unsafe.
            ProjectExistsError: If the project already has a manifest or files.
            ProjectError: If the metadata is invalid or the manifest cannot be written.
        """
        directory = self.project_dir(project_id)
        manifest = directory / PROJECT_FILENAME
        if manifest.exists():
            raise ProjectExistsError(f"Project already exists: {project_id}")
        if directory.exists() and any(directory.iterdir()):
            raise ProjectExistsError(
                f"Project directory is not empty; refusing to overwrite: {directory}"
            )

        timestamp = _utc_now()
        project = Project(
            project_id=project_id,
            title=title,
            description=description,
            status=status,
            language=language,
            created_at=timestamp,
            updated_at=timestamp,
            settings=_coerce_settings(settings),
        )
        result = validate_project_metadata(project)
        if not result.valid:
            raise ProjectError(
                "Invalid project metadata: "
                + "; ".join(issue.message for issue in result.errors)
            )

        for relative in PROJECT_DIRECTORIES:
            (directory / relative).mkdir(parents=True, exist_ok=True)
        for relative in PROJECT_FILES:
            placeholder = directory / relative
            if not placeholder.exists():
                placeholder.write_text("", encoding="utf-8")
        write_json_atomic(manifest, project.to_dict())
        logger.info("Created project %s at %s", project_id, directory)
        return project

    def load_project(self, project_id: str) -> Project:
        """Load ``project.json`` with explicit errors for missing/corrupt data."""
        manifest = self.project_file(project_id)
        if not manifest.is_file():
            raise ProjectNotFoundError(f"Project manifest not found: {manifest}")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError(f"Could not read project manifest {manifest}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ProjectError(f"Project manifest must be a JSON object: {manifest}")
        try:
            return Project.from_dict(payload)
        except ValueError as exc:
            raise ProjectError(f"Invalid project manifest {manifest}: {exc}") from exc

    def update_project(
        self,
        project_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        language: str | None = None,
        settings: ProjectSettings | Mapping[str, Any] | None = None,
    ) -> Project:
        """Update supplied metadata fields, preserving all other values."""
        project = self.load_project(project_id)

        if title is not None:
            if not isinstance(title, str) or not title.strip():
                raise ValueError("title must be a non-empty string")
            project.title = title
        if description is not None:
            if not isinstance(description, str):
                raise ValueError("description must be a string")
            project.description = description
        if status is not None:
            if status not in PROJECT_STATUSES:
                raise ValueError(f"status must be one of {sorted(PROJECT_STATUSES)}")
            project.status = status
        if language is not None:
            project.language = language
        if settings is not None:
            if isinstance(settings, ProjectSettings):
                project.settings = settings
            elif isinstance(settings, Mapping):
                merged = project.settings.to_dict()
                merged.update(dict(settings))
                project.settings = ProjectSettings.from_dict(merged)
            else:
                raise TypeError("settings must be a ProjectSettings instance or a mapping")

        project.updated_at = _utc_now()
        result = validate_project_metadata(project)
        if not result.valid:
            raise ProjectError(
                "Invalid project metadata: "
                + "; ".join(issue.message for issue in result.errors)
            )
        write_json_atomic(self.project_file(project_id), project.to_dict())
        logger.info("Updated project %s", project_id)
        return project

    def update_visual_identity(
        self,
        project_id: str,
        *,
        identity: VisualIdentity | Mapping[str, Any] | None,
    ) -> Project:
        """Set, merge or clear the project-wide visual identity.

        A mapping is merged into any existing identity so callers can update a
        single field without supplying every other field.  Passing ``None``
        clears the identity.  Unknown keys are preserved by ``VisualIdentity``.
        """
        project = self.load_project(project_id)
        if identity is None:
            project.visual_identity = None
        elif isinstance(identity, VisualIdentity):
            project.visual_identity = identity
        elif isinstance(identity, Mapping):
            invalid = [
                key
                for key, value in identity.items()
                if key in VISUAL_IDENTITY_FIELDS
                and value is not None
                and not isinstance(value, str)
            ]
            if invalid:
                raise ProjectError(
                    "Invalid visual identity: "
                    + "; ".join(f"visual_identity.{key} must be a string" for key in invalid)
                )
            merged = (
                project.visual_identity.to_dict()
                if project.visual_identity is not None
                else {}
            )
            merged.update(dict(identity))
            project.visual_identity = VisualIdentity.from_dict(merged)
        else:
            raise TypeError(
                "identity must be a VisualIdentity instance, a mapping, or None"
            )

        project.updated_at = _utc_now()
        result = validate_project_metadata(project)
        if not result.valid:
            raise ProjectError(
                "Invalid project metadata: "
                + "; ".join(issue.message for issue in result.errors)
            )
        write_json_atomic(self.project_file(project_id), project.to_dict())
        logger.info("Updated visual identity for project %s", project_id)
        return project
