"""Deterministic orchestration tests with every heavy service replaced by a fake."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import render_orchestrator as orch
from project_service import ProjectService
from render_job_models import RenderJobError
from render_job_store import RenderJobStore
from scene_service import SceneService


class FakePlanner:
    def build_plan(self, project_id: str, policy: Any = None) -> Any:
        return SimpleNamespace(scene_count=1, warnings=())


class FakeAudio:
    def __init__(self, root: Path, *, fail: Exception | None = None) -> None:
        self.root = root
        self.fail = fail
        self.calls = 0
        (root / "video" / "renders" / "audio-render").mkdir(parents=True, exist_ok=True)

    def produce(self, project_id: str, config: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return SimpleNamespace(
            render_id="audio-render",
            render_dir="video/renders/audio-render",
            scene_order=("scene-1",),
            silent_video_path="video/renders/audio-render/final_silent.mp4",
            audio_master_path="video/renders/audio-render/audio_master.wav",
            silent_video_duration_s=Decimal("1.0"),
            audio_duration_s=Decimal("1.0"),
            timeline=SimpleNamespace(total_duration_s=Decimal("1.0"), scenes=(), to_dict=dict),
            hold_durations_s={},
            mix=SimpleNamespace(peak=0.25, to_dict=dict),
            validation={},
            warnings=(),
        )


class FakeSubtitleResult:
    def __init__(self) -> None:
        self.files = {
            "srt": "video/renders/audio-render/subtitles/subtitles.en.srt",
            "vtt": "video/renders/audio-render/subtitles/subtitles.en.vtt",
        }
        self.track = SimpleNamespace(segment_count=1, style=SimpleNamespace())
        self.capabilities: dict[str, object] = {}
        self.warnings: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"files": dict(self.files), "track": {"segment_count": 1}}


class FakeSubtitles:
    def __init__(self, *, on_generate: Any = None) -> None:
        self.on_generate = on_generate
        self.calls = 0

    def generate(self, project_id: str, timeline: Any, config: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.on_generate is not None:
            self.on_generate()
        return FakeSubtitleResult()


class FakeFinalRenderer:
    def __init__(self) -> None:
        self.calls = 0

    def render(self, project_id: str, render_dir: Path, video: Any, audio: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(
            output_path="video/renders/audio-render/final.mp4",
            width=64,
            height=48,
            fps=10.0,
            duration_s=Decimal("1.0"),
            to_dict=lambda: {"output_path": "video/renders/audio-render/final.mp4"},
        )


class FakeQAReport:
    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid
        self.errors = () if valid else (SimpleNamespace(message="bad stream"),)

    def to_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "errors": [{"code": "qa.fatal"}] if not self.valid else []}


class FakeQA:
    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid
        self.calls = 0

    def validate(self, *args: Any, **kwargs: Any) -> FakeQAReport:
        self.calls += 1
        return FakeQAReport(valid=self.valid)


@pytest.fixture()
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    projects = ProjectService(tmp_path / "projects")
    projects.create_project("demo", "Demo", settings={"width": 64, "height": 48, "fps": 10})
    scenes = SceneService(projects)
    scenes.add_scene(
        "demo", title="Opening", scene_id="scene-1", duration=0.5, narration="Hello."
    )
    root = projects.project_dir("demo")
    store = RenderJobStore(root=tmp_path / "jobs", recover_on_init=False)
    audio = FakeAudio(root)
    subtitles = FakeSubtitles()
    renderer = FakeFinalRenderer()
    qa = FakeQA()
    orchestrator = orch.RenderOrchestrator(
        projects,
        scenes,
        store=store,
        planner=FakePlanner(),
        audio=audio,
        subtitles=subtitles,  # type: ignore[arg-type]
        final_renderer_service=renderer,  # type: ignore[arg-type]
        qa=qa,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        orch.render_report,
        "build_render_report",
        lambda **kwargs: {"project": {"project_id": kwargs["project"].project_id}},
    )
    monkeypatch.setattr(
        orch.render_report,
        "write_render_report",
        lambda render_dir, payload: (
            render_dir / "render_report.json",
            render_dir / "render_report.md",
        ),
    )
    return SimpleNamespace(
        projects=projects,
        scenes=scenes,
        root=root,
        store=store,
        audio=audio,
        subtitles=subtitles,
        renderer=renderer,
        qa=qa,
        orchestrator=orchestrator,
    )


def _config() -> dict[str, object]:
    return {
        "preset": "custom",
        "width": 64,
        "height": 48,
        "fps": 10,
        "preset_x264": "ultrafast",
        "render_root": "video/renders",
    }


def test_successful_pipeline_records_stages_and_completes(pipeline: Any) -> None:
    job = pipeline.orchestrator.create_job("demo", _config(), render_id="render")

    result = pipeline.orchestrator.execute_job(job.job_id)

    assert result["status"] == "completed", result
    final = pipeline.store.get_job(job.job_id)
    assert final is not None
    assert final.status == "completed"
    assert final.stage == "completed"
    assert final.progress.percent == 100.0
    assert final.completed_at
    assert final.validation_summary["valid"] is True
    assert final.output_paths["video"] == "video/renders/audio-render/final.mp4"
    assert final.output_paths["subtitle_srt"].endswith(".srt")
    assert final.output_paths["report_json"].endswith("render_report.json")
    stages = [entry["stage"] for entry in final.stages]
    for stage in (
        "validating",
        "planning",
        "preparing_video",
        "preparing_audio",
        "mixing_audio",
        "generating_subtitles",
        "merging",
        "validating_output",
        "completed",
    ):
        assert stage in stages
    assert pipeline.subtitles.calls == 1
    assert pipeline.renderer.calls == 1
    assert pipeline.qa.calls == 1


def test_pipeline_fails_when_project_has_no_scenes(pipeline: Any) -> None:
    pipeline.scenes.delete_scene("demo", "scene-1")
    job = pipeline.orchestrator.create_job("demo", _config())

    result = pipeline.orchestrator.execute_job(job.job_id)

    assert result["status"] == "failed", result
    final = pipeline.store.get_job(job.job_id)
    assert final is not None
    assert final.error is not None
    assert final.error["code"] == "render.project.no_scenes"
    assert pipeline.renderer.calls == 0


def test_stage_failure_is_recorded_without_completion(pipeline: Any) -> None:
    pipeline.audio.fail = RuntimeError("audio boom")
    job = pipeline.orchestrator.create_job("demo", _config())

    result = pipeline.orchestrator.execute_job(job.job_id)

    assert result["status"] == "failed", result
    final = pipeline.store.get_job(job.job_id)
    assert final is not None
    assert final.error is not None
    assert "audio boom" in final.error["message"]
    assert final.output_paths == {}
    assert pipeline.subtitles.calls == 0


def test_cancellation_between_stages_is_honest(pipeline: Any) -> None:
    job = pipeline.orchestrator.create_job("demo", _config())
    pipeline.subtitles.on_generate = lambda: pipeline.store.request_cancel(job.job_id)

    result = pipeline.orchestrator.execute_job(job.job_id)

    assert result["status"] == "cancelled", result
    final = pipeline.store.get_job(job.job_id)
    assert final is not None
    assert final.status == "cancelled"
    assert final.error["code"] == "render.job.cancelled"
    assert pipeline.renderer.calls == 0


def test_fatal_qa_result_never_marks_job_completed(pipeline: Any) -> None:
    pipeline.qa.valid = False
    job = pipeline.orchestrator.create_job("demo", _config())

    result = pipeline.orchestrator.execute_job(job.job_id)

    assert result["status"] == "failed", result
    final = pipeline.store.get_job(job.job_id)
    assert final is not None
    assert final.status == "failed"
    assert final.error["code"] == "render.qa.fatal"
    assert final.validation_summary["valid"] is False


def test_terminal_job_is_not_executed_twice(pipeline: Any) -> None:
    job = pipeline.orchestrator.create_job("demo", _config())
    pipeline.orchestrator.execute_job(job.job_id)
    pipeline.audio.calls = 0

    result = pipeline.orchestrator.execute_job(job.job_id)

    assert result["status"] == "completed"
    assert pipeline.audio.calls == 0


def test_duplicate_active_job_is_rejected(pipeline: Any) -> None:
    pipeline.orchestrator.create_job("demo", _config())
    result = pipeline.orchestrator.create_job_status("demo", _config())

    assert result["status"] == "error"
    assert result["error"]["code"] == "render.job.duplicate_active"


def test_interrupted_job_is_recovered_on_restart(pipeline: Any) -> None:
    job = pipeline.orchestrator.create_job("demo", _config())
    pipeline.store.save_job(
        job.__class__(**{**job.__dict__, "status": "merging", "stage": "merging"})
    )

    recovered = RenderJobStore(root=pipeline.store.root, recover_on_init=False).recover_interrupted()

    assert len(recovered) == 1
    assert recovered[0].status == "failed"
    assert recovered[0].error is not None
    assert recovered[0].error["code"] == "render.job.interrupted"


def test_create_job_rejects_invalid_project_id(pipeline: Any) -> None:
    result = pipeline.orchestrator.create_job_status("../escape", _config())

    assert result["status"] == "error"
    with pytest.raises(RenderJobError):
        pipeline.orchestrator.create_job("../escape", _config())


def test_burn_in_request_builds_filter(pipeline: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    filters: list[str] = []
    monkeypatch.setattr(
        orch.subtitle_renderer,
        "build_burn_in_filter",
        lambda path, style: filters.append(str(path)) or "subtitles=fake",
    )
    config = dict(_config(), burn_subtitles=True)
    job = pipeline.orchestrator.create_job("demo", config)

    result = pipeline.orchestrator.execute_job(job.job_id)

    assert result["status"] == "completed", result
    assert len(filters) == 1
