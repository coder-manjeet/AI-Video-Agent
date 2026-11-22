"""MCP-independent audio timeline planning from an existing RenderPlan.

The planner is additive: it never calls ``RenderPlanner`` differently, never
writes manifests and never changes planned scene durations. Video timing comes
from measured scene durations, narration timing comes from the actual WAV
duration stored in the render plan, and every gap, hold or silent tail is
explicit. Narration is never truncated or time-stretched here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from audio_models import (
    AudioPipelineError,
    AudioTimelinePolicy,
    ValidationIssue,
    coerce_duration,
    decimal_text,
)
from render_models import RenderPlan, coerce_decimal

logger = logging.getLogger(__name__)


class AudioTimelineError(AudioPipelineError):
    """Structured timeline planning failure."""


def _fail(code: str, message: str, location: str = "") -> AudioTimelineError:
    return AudioTimelineError(code, message, location, stage="timeline")


@dataclass(frozen=True)
class AudioSceneTiming:
    """One scene's reconciled video and narration timing."""

    scene_id: str
    order: int
    title: str
    video_start_s: Decimal
    video_end_s: Decimal
    video_duration_s: Decimal
    narration_start_s: Decimal
    narration_end_s: Decimal
    narration_duration_s: Decimal | None
    narration_overflow_s: Decimal
    final_frame_hold_s: Decimal
    silent_tail_s: Decimal
    effective_duration_s: Decimal
    narration_present: bool
    narration_path: str | None
    generated_video_path: str | None = None
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError("order must be a non-negative integer")
        if not isinstance(self.title, str):
            raise TypeError("title must be a string")
        for name in (
            "video_start_s",
            "video_end_s",
            "video_duration_s",
            "narration_start_s",
            "narration_end_s",
            "narration_overflow_s",
            "final_frame_hold_s",
            "silent_tail_s",
            "effective_duration_s",
        ):
            object.__setattr__(self, name, coerce_duration(getattr(self, name), name))
        if self.narration_duration_s is not None:
            object.__setattr__(
                self,
                "narration_duration_s",
                coerce_decimal(self.narration_duration_s, "narration_duration_s", positive=True),
            )
        if not isinstance(self.narration_present, bool):
            raise TypeError("narration_present must be a boolean")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for issue in self.warnings:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("warnings must contain ValidationIssue entries")
        if self.narration_present != (self.narration_duration_s is not None):
            raise ValueError("narration_present must agree with narration_duration_s")

    @property
    def planned_video_start_s(self) -> Decimal:
        """Alias used by reports that prefer the timeline field name."""
        return self.video_start_s

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "order": self.order,
            "title": self.title,
            "video_start_s": decimal_text(self.video_start_s),
            "video_end_s": decimal_text(self.video_end_s),
            "video_duration_s": decimal_text(self.video_duration_s),
            "narration_start_s": decimal_text(self.narration_start_s),
            "narration_end_s": decimal_text(self.narration_end_s),
            "narration_duration_s": (
                decimal_text(self.narration_duration_s)
                if self.narration_duration_s is not None
                else None
            ),
            "narration_overflow_s": decimal_text(self.narration_overflow_s),
            "final_frame_hold_s": decimal_text(self.final_frame_hold_s),
            "silent_tail_s": decimal_text(self.silent_tail_s),
            "effective_duration_s": decimal_text(self.effective_duration_s),
            "narration_present": self.narration_present,
            "narration_path": self.narration_path,
            "generated_video_path": self.generated_video_path,
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class AudioTimeline:
    """A read-only, sequential audio/video timeline for one render plan."""

    project_id: str
    policy: AudioTimelinePolicy
    scenes: tuple[AudioSceneTiming, ...]
    total_duration_s: Decimal
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.project_id, str) or not self.project_id.strip():
            raise ValueError("project_id must be a non-empty string")
        if not isinstance(self.policy, AudioTimelinePolicy):
            raise TypeError("policy must be an AudioTimelinePolicy")
        object.__setattr__(self, "scenes", tuple(self.scenes))
        for scene in self.scenes:
            if not isinstance(scene, AudioSceneTiming):
                raise TypeError("scenes must contain AudioSceneTiming entries")
        object.__setattr__(
            self,
            "total_duration_s",
            coerce_decimal(self.total_duration_s, "total_duration_s", positive=True),
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for issue in self.warnings:
            if not isinstance(issue, ValidationIssue):
                raise TypeError("warnings must contain ValidationIssue entries")

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    @property
    def narration_scenes(self) -> tuple[AudioSceneTiming, ...]:
        return tuple(scene for scene in self.scenes if scene.narration_present)

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "policy": self.policy.to_dict(),
            "scene_count": self.scene_count,
            "total_duration_s": decimal_text(self.total_duration_s),
            "scenes": [scene.to_dict() for scene in self.scenes],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


class AudioTimelinePlanner:
    """Reconcile measured video durations with measured narration durations."""

    def build(
        self,
        plan: RenderPlan,
        policy: AudioTimelinePolicy | None = None,
    ) -> AudioTimeline:
        """Return an additive audio timeline over *plan* without mutating it."""
        if not isinstance(plan, RenderPlan):
            raise TypeError("plan must be a RenderPlan")
        effective_policy = policy if policy is not None else AudioTimelinePolicy()
        if not isinstance(effective_policy, AudioTimelinePolicy):
            raise TypeError("policy must be an AudioTimelinePolicy")
        if plan.scene_count == 0:
            raise _fail("audio.timeline.empty_plan", "Render plan has no scenes", "plan")

        warnings: list[ValidationIssue] = list(plan.warnings)
        entries: list[AudioSceneTiming] = []
        cursor = Decimal(0)
        total = Decimal(0)

        for order, scene in enumerate(plan.scenes):
            location = f"scenes[{order}] ({scene.scene_id})"
            measured = scene.measured_duration_s
            effective = scene.effective_duration_s
            video_start = scene.timeline_start_s
            video_end = scene.timeline_end_s
            if abs((video_end - video_start) - effective) > effective_policy.duration_tolerance_s:
                raise _fail(
                    "audio.timeline.scene_duration_mismatch",
                    f"Render plan scene timing {decimal_text(video_start)}.."
                    f"{decimal_text(video_end)} does not match effective duration "
                    f"{decimal_text(effective)}",
                    location,
                )
            if abs(video_start - cursor) > effective_policy.duration_tolerance_s:
                raise _fail(
                    "audio.timeline.sequence_gap",
                    f"Scene starts at {decimal_text(video_start)}s but the previous "
                    f"scene ends at {decimal_text(cursor)}s",
                    location,
                )

            narration_duration = scene.narration_duration_s
            scene_warnings: list[ValidationIssue] = []
            if narration_duration is None:
                narration_present = False
                narration_end = video_start
                overflow = Decimal(0)
                hold = scene.video_hold_duration_s
                silent_tail = Decimal(0)
                issue = ValidationIssue(
                    "audio.timeline.narration_missing",
                    f"Scene {scene.scene_id} has no usable narration; no speech "
                    "will be fabricated",
                    f"{location}.narration_audio_path",
                )
                if effective_policy.missing_narration_policy == "error":
                    raise _fail(issue.code, issue.message, issue.location)
                warnings.append(issue)
                scene_warnings.append(issue)
            else:
                narration_present = True
                narration_end = video_start + narration_duration
                overflow = max(Decimal(0), narration_duration - measured)
                hold = scene.video_hold_duration_s
                silent_tail = scene.silent_tail_duration_s
                if abs(hold - overflow) > effective_policy.duration_tolerance_s:
                    warnings.append(
                        ValidationIssue(
                            "audio.timeline.hold_differs_from_overflow",
                            f"Planned hold {decimal_text(hold)}s differs from "
                            f"narration overflow {decimal_text(overflow)}s",
                            f"{location}.video_hold_duration_s",
                        )
                    )
                    hold = max(hold, overflow)
                expected_effective = measured + hold
                if (
                    abs(effective - expected_effective)
                    > effective_policy.duration_tolerance_s
                ):
                    raise _fail(
                        "audio.timeline.effective_duration_mismatch",
                        f"Effective duration {decimal_text(effective)}s does not "
                        f"match measured video plus hold "
                        f"({decimal_text(expected_effective)}s)",
                        location,
                    )
                if overflow > 0:
                    scene_warnings.append(
                        ValidationIssue(
                            "audio.timeline.hold_required",
                            f"Narration extends {decimal_text(overflow)}s beyond "
                            "the scene video; the final frame will be held",
                            f"{location}.narration_audio_path",
                        )
                    )
                if silent_tail > 0:
                    scene_warnings.append(
                        ValidationIssue(
                            "audio.timeline.silent_tail",
                            f"Narration ends {decimal_text(silent_tail)}s before "
                            "the scene video; no stream is truncated",
                            f"{location}.narration_audio_path",
                        )
                    )

            entries.append(
                AudioSceneTiming(
                    scene_id=scene.scene_id,
                    order=order,
                    title=scene.title,
                    video_start_s=video_start,
                    video_end_s=video_end,
                    video_duration_s=measured,
                    narration_start_s=video_start,
                    narration_end_s=narration_end,
                    narration_duration_s=narration_duration,
                    narration_overflow_s=overflow,
                    final_frame_hold_s=hold,
                    silent_tail_s=silent_tail,
                    effective_duration_s=effective,
                    narration_present=narration_present,
                    narration_path=scene.narration_audio_path,
                    generated_video_path=scene.generated_video_path,
                    warnings=tuple(scene_warnings),
                )
            )
            total += effective
            cursor = video_end

        if abs(total - plan.total_render_duration_s) > effective_policy.duration_tolerance_s:
            raise _fail(
                "audio.timeline.total_mismatch",
                f"Calculated total {decimal_text(total)}s differs from render plan "
                f"total {decimal_text(plan.total_render_duration_s)}s",
                "plan.total_render_duration_s",
            )
        logger.info(
            "Built audio timeline for project %s: %d scenes, %ss",
            plan.project_id,
            len(entries),
            decimal_text(total),
        )
        return AudioTimeline(
            project_id=plan.project_id,
            policy=effective_policy,
            scenes=tuple(entries),
            total_duration_s=total,
            warnings=tuple(warnings),
        )
