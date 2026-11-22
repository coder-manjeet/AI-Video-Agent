"""Unit tests for the ``local-render-agent`` FastMCP tool boundary."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import render_server as server

pytestmark = pytest.mark.asyncio


class FakeJob:
    def __init__(self, job_id: str, project_id: str, *, status: str = "completed") -> None:
        self.job_id = job_id
        self.project_id = project_id
        self.status = status
        self.config: dict[str, object] = {"width": 64, "height": 48, "fps": 10}
        self.output_paths: dict[str, str] = {}

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "project_id": self.project_id,
            "status": self.status,
            "config": dict(self.config),
            "output_paths": dict(self.output_paths),
        }


class FakeStore:
    def __init__(self) -> None:
        self.jobs: dict[str, FakeJob] = {}

    def create(self, job_id: str, project_id: str) -> FakeJob:
        job = FakeJob(job_id, project_id, status="queued")
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> FakeJob | None:
        return self.jobs.get(job_id)

    def list_jobs(self, project_id: str | None = None) -> list[FakeJob]:
        return [job for job in self.jobs.values() if project_id is None or job.project_id == project_id]

    def request_cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status in {"completed", "failed", "cancelled"}:
            return False
        job.status = "cancelled"
        return True


class FakeProjects:
    def __init__(self, root: Path) -> None:
        self.root = root

    def project_dir(self, project_id: str) -> Path:
        return self.root / project_id


class FakeOrchestrator:
    def __init__(self) -> None:
        self.created: list[tuple[str, Any, str | None]] = []
        self.executed: list[str] = []
        self.block: threading.Event | None = None
        self.started: threading.Event | None = None

    def create_job_status(
        self, project_id: str, config: Any = None, render_id: str | None = None
    ) -> dict[str, object]:
        self.created.append((project_id, config, render_id))
        return {"status": "error", "error": "bad config"} if project_id == "bad" else {
            "status": "ok",
            "job": {"job_id": "job-1", "project_id": project_id, "status": "queued"},
        }

    def execute_job(self, job_id: str) -> dict[str, object]:
        self.executed.append(job_id)
        if self.started is not None:
            self.started.set()
        if self.block is not None:
            self.block.wait(timeout=5)
        return {"status": "completed", "job": {}}


@pytest.fixture(autouse=True)
def reset_state(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(server, "_SERVICES", None)
    server._TASKS.clear()
    server._VALIDATED.clear()
    yield
    for task in list(server._TASKS.values()):
        task.cancel()
    server._TASKS.clear()


def _install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    store = FakeStore()
    orchestrator = FakeOrchestrator()
    services = server._Services(
        store=store,  # type: ignore[arg-type]
        orchestrator=orchestrator,  # type: ignore[arg-type]
        projects=FakeProjects(tmp_path),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(server, "_get_services", lambda: services)
    return SimpleNamespace(store=store, orchestrator=orchestrator, services=services)


async def test_all_six_tools_are_registered() -> None:
    names = {tool.name for tool in await server.mcp.list_tools()}

    assert {
        "create_render_job",
        "get_render_job",
        "list_render_jobs",
        "cancel_render_job",
        "get_render_report",
        "validate_render_output",
    } <= names


async def test_create_job_validates_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _install(monkeypatch, tmp_path)

    assert (await server.create_render_job(""))["status"] == "error"
    assert (await server.create_render_job("../escape"))["status"] == "error"
    assert (await server.create_render_job("demo", config=[1, 2]))["status"] == "error"  # type: ignore[arg-type]
    assert (await server.create_render_job("bad"))["status"] == "error"

    result = await server.create_render_job("demo", {"preset": "1080p"}, auto_start=False)
    assert result["status"] == "queued"
    assert ("demo", {"preset": "1080p"}, None) in fake.orchestrator.created
    assert fake.store.get_job("job-1") is None  # fakes keep storage separate


async def test_create_job_returns_without_waiting_for_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _install(monkeypatch, tmp_path)
    fake.orchestrator.block = threading.Event()
    fake.orchestrator.started = threading.Event()

    result = await server.create_render_job("demo", auto_start=True)
    assert result["status"] == "queued"
    assert await asyncio.to_thread(fake.orchestrator.started.wait, 3) is True
    assert "job-1" in server._TASKS

    fake.orchestrator.block.set()
    await asyncio.gather(*server._TASKS.values())
    assert fake.orchestrator.executed == ["job-1"]


async def test_polling_and_cancellation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _install(monkeypatch, tmp_path)
    fake.store.create("job-1", "demo")

    listed = await server.list_render_jobs("demo")
    assert listed["status"] == "ok"
    assert len(listed["jobs"]) == 1

    polled = await server.get_render_job("job-1")
    assert polled["status"] == "ok"
    assert polled["job"]["job_id"] == "job-1"

    cancelled = await server.cancel_render_job("job-1")
    assert cancelled["status"] == "ok" and cancelled["cancelled"] is True
    again = await server.cancel_render_job("job-1")
    assert again["cancelled"] is False

    assert (await server.get_render_job("missing"))["status"] == "error"
    assert (await server.list_render_jobs("../bad"))["status"] == "error"
    assert (await server.cancel_render_job("bad id"))["status"] == "error"


async def test_report_retrieval_is_scoped_to_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _install(monkeypatch, tmp_path)
    job = fake.store.create("job-1", "demo")
    assert (await server.get_render_report("job-1"))["status"] == "error"

    report = {"project": {"project_id": "demo"}, "render": {"render_id": "r1"}}
    path = tmp_path / "demo" / "video" / "render_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    job.output_paths["report_json"] = "video/render_report.json"

    result = await server.get_render_report("job-1")
    assert result["status"] == "ok"
    assert result["report"] == report


async def test_validate_output_uses_offloop_thread_and_fake_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _install(monkeypatch, tmp_path)
    job = fake.store.create("job-1", "demo")
    video = tmp_path / "demo" / "video" / "final.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"mp4")
    job.output_paths["video"] = "video/final.mp4"

    class FakeValidator:
        def validate(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(to_dict=lambda: {"valid": True, "errors": []})

    monkeypatch.setattr(server.final_qa, "FinalQAValidator", FakeValidator)

    result = await server.validate_render_output("job-1")
    assert result["status"] == "ok"
    assert result["validation"]["valid"] is True
    assert "job-1" in server._VALIDATED

    job.output_paths["video"] = "../outside.mp4"
    assert (await server.validate_render_output("job-1"))["status"] == "error"


async def test_shutdown_requests_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _install(monkeypatch, tmp_path)
    fake.store.create("job-1", "demo")
    server._TASKS["job-1"] = asyncio.create_task(asyncio.sleep(30))
    await asyncio.sleep(0)

    await server._shutdown_jobs()

    assert fake.store.get_job("job-1").status == "cancelled"
    assert server._TASKS == {}
