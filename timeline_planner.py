"""Sequential video-scene timeline planning.

Planning is pure arithmetic: it reads scene durations, calculates positions
with :class:`~decimal.Decimal` to avoid floating-point drift, and never probes
or renders media.  A future editor can consume :meth:`TimelinePlan.to_dict`
directly; ``tracks`` is reserved for narration, music, SFX and subtitles.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from project_models import Scene
from project_paths import PathValidationError, resolve_within_project
from scene_service import SceneService

TRACK_VIDEO = "video"
TRACK_NAMES = (TRACK_VIDEO,)


@dataclass(frozen=True)
class TimelineScene:
    """One scene card with its calculated position and asset metadata."""

    scene_id: str
    scene_number: int
    title: str
    status: str
    duration_s: float
    timeline_start_s: float
    timeline_end_s: float
    narration: str
    image_path: str | None
    thumbnail_path: str | None
    generated_video_path: str | None
    narration_audio_path: str | None
    sfx_paths: tuple[str, ...]
    preview_path: str | None
    asset_availability: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_number": self.scene_number,
            "title": self.title,
            "status": self.status,
            "duration_s": self.duration_s,
            "timeline_start_s": self.timeline_start_s,
            "timeline_end_s": self.timeline_end_s,
            "narration": self.narration,
            "image_path": self.image_path,
            "thumbnail_path": self.thumbnail_path,
            "generated_video_path": self.generated_video_path,
            "narration_audio_path": self.narration_audio_path,
            "sfx_paths": list(self.sfx_paths),
            "preview_path": self.preview_path,
            "asset_availability": dict(self.asset_availability),
        }


@dataclass(frozen=True)
class TimelinePlan:
    """Structured timeline data for a future video-editor interface."""

    scenes: tuple[TimelineScene, ...]
    total_duration_s: float
    tracks: dict[str, tuple[TimelineScene, ...]] = field(default_factory=dict)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_duration_s": self.total_duration_s,
            "scene_count": self.scene_count,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "tracks": {
                name: [scene.to_dict() for scene in entries]
                for name, entries in self.tracks.items()
            },
        }


def _duration_as_decimal(scene: Scene) -> Decimal:
    value = scene.duration
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Scene {scene.scene_id!r} has a non-numeric duration")
    duration = Decimal(str(value))
    if not duration.is_finite() or duration <= 0:
        raise ValueError(f"Scene {scene.scene_id!r} duration must be positive and finite")
    return duration


def _file_available(root: Path | None, path: str | None) -> bool:
    if root is None or not path:
        return False
    try:
        return resolve_within_project(root, path).is_file()
    except PathValidationError:
        return False


class TimelinePlanner:
    """Calculate and optionally persist sequential video scene positions."""

    def __init__(self, scene_service: SceneService | None = None) -> None:
        self.scene_service = scene_service

    def plan(
        self,
        scenes: Sequence[Scene],
        *,
        project_root: str | Path | None = None,
    ) -> TimelinePlan:
        """Return a timeline without mutating *scenes* or touching the disk."""
        root = Path(project_root).expanduser().resolve() if project_root is not None else None
        cursor = Decimal(0)
        entries: list[TimelineScene] = []

        for scene in scenes:
            duration = _duration_as_decimal(scene)
            end = cursor + duration
            paths = {
                "image": scene.image_path,
                "thumbnail": scene.thumbnail_path,
                "video": scene.generated_video_path,
                "narration_audio": scene.narration_audio_path,
            }
            availability = {
                key: _file_available(root, path) for key, path in paths.items()
            }
            availability["sfx"] = bool(scene.sfx_paths) and all(
                _file_available(root, path) for path in scene.sfx_paths
            )
            entries.append(
                TimelineScene(
                    scene_id=scene.scene_id,
                    scene_number=scene.scene_number,
                    title=scene.title,
                    status=scene.status,
                    duration_s=float(duration),
                    timeline_start_s=float(cursor),
                    timeline_end_s=float(end),
                    narration=scene.narration,
                    image_path=scene.image_path,
                    thumbnail_path=scene.thumbnail_path,
                    generated_video_path=scene.generated_video_path,
                    narration_audio_path=scene.narration_audio_path,
                    sfx_paths=tuple(scene.sfx_paths),
                    preview_path=scene.generated_video_path or scene.image_path,
                    asset_availability=availability,
                )
            )
            cursor = end

        entries_tuple = tuple(entries)
        return TimelinePlan(
            scenes=entries_tuple,
            total_duration_s=float(cursor),
            tracks={TRACK_VIDEO: entries_tuple},
        )

    def plan_project(self, project_id: str) -> TimelinePlan:
        """Load a project's scenes and plan their timeline."""
        if self.scene_service is None:
            raise RuntimeError("TimelinePlanner requires a SceneService for project planning")
        scenes = self.scene_service.list_scenes(project_id)
        root = self.scene_service.projects.project_dir(project_id)
        return self.plan(scenes, project_root=root)

    def persist(self, project_id: str, plan: TimelinePlan) -> list[Scene]:
        """Write calculated start/end values back into ``scenes.json``."""
        if self.scene_service is None:
            raise RuntimeError("TimelinePlanner requires a SceneService to persist")
        positions = {
            entry.scene_id: (entry.timeline_start_s, entry.timeline_end_s)
            for entry in plan.scenes
        }
        return self.scene_service.apply_timeline(project_id, positions)

    def plan_and_persist(self, project_id: str) -> TimelinePlan:
        """Recalculate from the manifest and persist the resulting positions."""
        plan = self.plan_project(project_id)
        self.persist(project_id, plan)
        return plan
