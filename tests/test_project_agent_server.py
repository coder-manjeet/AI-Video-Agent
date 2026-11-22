"""Unit tests for the ``local-project-agent`` FastMCP tool boundary.

Every service is replaced with a fake: no ComfyUI, GPU, TTS, FFmpeg or real
project directory is used, and no media or manifest is written by these tests.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

import project_agent_server as server
from project_service import ProjectError
from scene_video_service import JobNotFoundError


class FakeProject:
    def __init__(self, project_id: str, title: str = "Demo", status: str = "active") -> None:
        self.project_id = project_id
        self.title = title
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "status": self.status,
            "settings": {"width": 768},
        }


class FakeProjects:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.load_calls: list[str] = []
        self.raise_on_load: dict[str, Exception] = {}

    def create_project(self, project_id: str, title: str, **kwargs: Any) -> FakeProject:
        self.create_calls.append((project_id, title, kwargs))
        return FakeProject(project_id, title, kwargs.get("status", "active"))

    def list_projects(self) -> list[str]:
        return ["alpha", "broken"]

    def load_project(self, project_id: str) -> FakeProject:
        self.load_calls.append(project_id)
        if project_id in self.raise_on_load:
            raise self.raise_on_load[project_id]
        return FakeProject(project_id)


class FakeScene:
    def __init__(self, scene_id: str) -> None:
        self.scene_id = scene_id

    def to_dict(self) -> dict[str, Any]:
        return {"scene_id": self.scene_id, "title": "Opening", "status": "draft"}


class FakeScenes:
    def __init__(self) -> None:
        self.add_calls: list[tuple[str, dict[str, Any]]] = []
        self.update_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.raise_on_get: Exception | None = None

    def add_scene(self, project_id: str, **options: Any) -> FakeScene:
        self.add_calls.append((project_id, options))
        return FakeScene(str(options.get("scene_id") or "scene-1"))

    def list_scenes(self, project_id: str) -> list[FakeScene]:
        return [FakeScene("scene-1")]

    def get_scene(self, project_id: str, scene_id: str) -> FakeScene:
        if self.raise_on_get is not None:
            raise self.raise_on_get
        return FakeScene(scene_id)

    def update_scene(
        self, project_id: str, scene_id: str, changes: dict[str, Any]
    ) -> FakeScene:
        self.update_calls.append((project_id, scene_id, changes))
        return FakeScene(scene_id)


class FakeGPUProbe:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    def enter(self) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)

    def leave(self) -> None:
        self.active -= 1


class FakeGenerator:
    """A generator fake that registers a job before its first await."""

    def __init__(self, kind: str, probe: FakeGPUProbe | None = None) -> None:
        self.kind = kind
        self.probe = probe or FakeGPUProbe()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, tuple[str, str], dict[str, Any]]] = []
        self.release = asyncio.Event()
        self.block = False
        self.fail_with: Exception | None = None
        self.cancel_result: dict[str, Any] = {
            "status": "cancelled",
            "supported": True,
            "error": None,
        }
        self.cancel_calls: list[str] = []
        self._counter = 0

    def _new_job(self, project_id: str, scene_id: str, status: str = "queued") -> dict[str, Any]:
        self._counter += 1
        job_id = f"{self.kind}-job-{self._counter}"
        job: dict[str, Any] = {
            "job_id": job_id,
            "prompt_id": f"prompt-{self._counter}",
            "project_id": project_id,
            "scene_id": scene_id,
            "workflow": f"{self.kind}-workflow",
            "status": status,
            "attempt": 1,
            "warnings": ["test-warning"],
            "error": None,
        }
        self.jobs[job_id] = job
        return job

    def list_jobs(
        self, *, project_id: str | None = None, scene_id: str | None = None
    ) -> list[dict[str, Any]]:
        return [
            job
            for job in self.jobs.values()
            if (project_id is None or job["project_id"] == project_id)
            and (scene_id is None or job["scene_id"] == scene_id)
        ]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        self.cancel_calls.append(job_id)
        if isinstance(self.cancel_result, Exception):
            raise self.cancel_result
        return dict(self.cancel_result)

    async def _generate(
        self,
        method: str,
        project_id: str,
        scene_id: str,
        kwargs: dict[str, Any],
        *,
        retry: bool,
    ) -> dict[str, Any]:
        self.calls.append((method, (project_id, scene_id), kwargs))
        self.probe.enter()
        try:
            if retry:
                candidates = [
                    job
                    for job in self.jobs.values()
                    if job["status"] in {"failed", "timeout", "cancelled"}
                ]
                if not candidates:
                    raise RuntimeError("no retry candidate")
                if self.kind == "image":
                    job = self._new_job(project_id, scene_id)
                else:
                    job = candidates[-1]
                    job["attempt"] += 1
                    job["status"] = "queued"
            else:
                job = self._new_job(project_id, scene_id)
            if self.block:
                await self.release.wait()
            if self.fail_with is not None:
                raise self.fail_with
            job["status"] = "completed"
            path_key = "image_path" if self.kind == "image" else "output_path"
            return {
                "status": "completed",
                "job_id": job["job_id"],
                "prompt_id": job["prompt_id"],
                "project_id": project_id,
                "scene_id": scene_id,
                "workflow": job["workflow"],
                "attempt": job["attempt"],
                path_key: f"assets/{self.kind}s/scene-001.png",
                "filename": "scene-001.png",
                "error": None,
                "warnings": ["test-warning"],
            }
        finally:
            self.probe.leave()

    async def generate_scene_image(
        self, project_id: str, scene_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        return await self._generate(
            "generate_scene_image", project_id, scene_id, kwargs, retry=False
        )

    async def retry_scene_image(
        self, project_id: str, scene_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        return await self._generate(
            "retry_scene_image", project_id, scene_id, kwargs, retry=True
        )

    async def generate_scene_video(
        self, project_id: str, scene_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        return await self._generate(
            "generate_scene_video", project_id, scene_id, kwargs, retry=False
        )

    async def retry_scene_video(
        self, project_id: str, scene_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        return await self._generate(
            "retry_scene_video", project_id, scene_id, kwargs, retry=True
        )


@pytest.fixture(autouse=True)
def reset_server_state(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv(server.PROJECTS_ROOT_ENV, raising=False)
    monkeypatch.setattr(server, "_SERVICES", None)
    monkeypatch.setattr(server, "_GPU_LOCK", None)
    server._SUBMISSIONS.clear()
    server._TASKS.clear()
    yield
    for task in list(server._TASKS.values()):
        task.cancel()
    server._SUBMISSIONS.clear()
    server._TASKS.clear()


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    projects: FakeProjects | None = None,
    scenes: FakeScenes | None = None,
    image: FakeGenerator | None = None,
    video: FakeGenerator | None = None,
) -> Any:
    services = server._Services(
        projects=projects or FakeProjects(),
        scenes=scenes or FakeScenes(),
        characters=object(),
        image=image or FakeGenerator("image"),
        video=video or FakeGenerator("video"),
    )
    monkeypatch.setattr(server, "_get_services", lambda: services)
    return services


async def _wait_for_submission(submission_id: str) -> None:
    task = server._TASKS.get(submission_id)
    if task is not None:
        await asyncio.wait_for(task, timeout=2.0)
    else:
        await asyncio.sleep(0)


# --- registration and wiring -------------------------------------------


def test_server_name() -> None:
    assert server.mcp.name == "local-project-agent"


@pytest.mark.asyncio
async def test_registered_tool_names() -> None:
    tools = await server.mcp.list_tools()
    assert {tool.name for tool in tools} == {
        "create_project",
        "list_projects",
        "get_project",
        "add_scene",
        "list_scenes",
        "update_scene",
        "generate_scene_image",
        "get_scene_image_job",
        "retry_scene_image",
        "generate_scene_video",
        "get_scene_video_job",
        "retry_scene_video",
        "cancel_scene_video",
    }


def test_projects_root_default_and_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert server._projects_root() == (Path(server.__file__).resolve().parent / "projects")
    monkeypatch.setenv(server.PROJECTS_ROOT_ENV, str(tmp_path / "custom"))
    assert server._projects_root() == (tmp_path / "custom").resolve()


def test_lazy_services_share_instances(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(server.PROJECTS_ROOT_ENV, str(tmp_path / "shared"))

    services = server._get_services()
    assert services.projects.projects_root == (tmp_path / "shared").resolve()
    assert services.scenes.projects is services.projects
    assert services.characters.projects is services.projects
    assert services.image.projects is services.projects
    assert services.image.scenes is services.scenes
    assert services.image.characters is services.characters
    assert services.video.projects is services.projects
    assert services.video.scenes is services.scenes
    assert server._get_services() is services


# --- project tools ------------------------------------------------------


@pytest.mark.asyncio
async def test_create_project_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    services = _install_fakes(monkeypatch)
    result = await server.create_project(
        "alpha",
        "Alpha",
        description="desc",
        language="en",
        status="draft",
        settings={"fps": 30},
    )
    assert result["status"] == "created"
    assert result["project_id"] == "alpha"
    assert result["project"]["project_id"] == "alpha"
    assert services.projects.create_calls == [
        (
            "alpha",
            "Alpha",
            {
                "description": "desc",
                "language": "en",
                "status": "draft",
                "settings": {"fps": 30},
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "project_id",
    ["", "   ", "../escape", "bad id"],
)
async def test_create_project_rejects_bad_ids(
    monkeypatch: pytest.MonkeyPatch, project_id: str
) -> None:
    _install_fakes(monkeypatch)
    result = await server.create_project(project_id, "Title")
    assert result["status"] == "error"
    assert "error" in result


@pytest.mark.asyncio
async def test_list_projects_reports_corrupt_manifest_per_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = _install_fakes(monkeypatch)
    services.projects.raise_on_load["broken"] = ValueError("bad json")
    result = await server.list_projects()
    assert result["status"] == "ok"
    assert result["projects"] == [
        {"project_id": "alpha", "title": "Demo", "status": "active"},
        {"project_id": "broken", "status": "error", "error": "ValueError: bad json"},
    ]


@pytest.mark.asyncio
async def test_get_project_returns_manifest_or_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = _install_fakes(monkeypatch)
    result = await server.get_project("alpha")
    assert result["status"] == "ok"
    assert result["project"]["settings"] == {"width": 768}
    services.projects.raise_on_load["alpha"] = ProjectError("missing")
    # Service errors are reported through the expected-error boundary.
    error = await server.get_project("alpha")
    assert error["status"] == "error"


# --- scene tools --------------------------------------------------------


@pytest.mark.asyncio
async def test_add_scene_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    services = _install_fakes(monkeypatch)
    result = await server.add_scene(
        "alpha",
        "Opening",
        duration=7.5,
        scene_id="scene-7",
        description="desc",
        narration="hello",
        image_prompt="a prompt",
        animation_prompt="motion",
        negative_prompt="blur",
        character_references=["hero.png"],
        status="ready",
    )
    assert result["status"] == "created"
    assert result["scene_id"] == "scene-7"
    assert result["scene"]["scene_id"] == "scene-7"
    assert services.scenes.add_calls == [
        (
            "alpha",
            {
                "title": "Opening",
                "duration": 7.5,
                "scene_id": "scene-7",
                "description": "desc",
                "narration": "hello",
                "image_prompt": "a prompt",
                "animation_prompt": "motion",
                "negative_prompt": "blur",
                "character_references": ["hero.png"],
                "status": "ready",
            },
        )
    ]


@pytest.mark.asyncio
async def test_list_scenes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    result = await server.list_scenes("alpha")
    assert result["status"] == "ok"
    assert result["scenes"] == [{"scene_id": "scene-1", "title": "Opening", "status": "draft"}]


@pytest.mark.asyncio
async def test_update_scene_forwards_only_supplied_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = _install_fakes(monkeypatch)
    result = await server.update_scene("alpha", "scene-1", title="New", duration=6.0)
    assert result["status"] == "updated"
    assert services.scenes.update_calls == [
        ("alpha", "scene-1", {"title": "New", "duration": 6.0})
    ]


@pytest.mark.asyncio
async def test_update_scene_requires_a_change(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    result = await server.update_scene("alpha", "scene-1")
    assert result["status"] == "error"
    assert "at least one scene field" in str(result["error"])


def test_update_scene_does_not_expose_immutable_fields() -> None:
    names = set(inspect.signature(server.update_scene).parameters)
    for forbidden in ("scene_number", "timeline_start", "timeline_end", "schema_version"):
        assert forbidden not in names


# --- image generation ---------------------------------------------------


@pytest.mark.asyncio
async def test_generate_scene_image_is_non_blocking_and_links_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = FakeGenerator("image")
    image.block = True
    services = _install_fakes(monkeypatch, image=image)

    started = await server.generate_scene_image("alpha", "scene-1", width=512)
    assert started["status"] == "submitted"
    assert isinstance(started["submission_id"], str)
    assert started["job_id"] == "image-job-1"
    assert image.calls[0][0] == "generate_scene_image"
    assert image.calls[0][2]["width"] == 512
    assert "timeout_s" not in image.calls[0][2]

    submission_id = str(started["submission_id"])
    assert submission_id in server._TASKS
    running = await server.get_scene_image_job(submission_id=submission_id)
    assert running["status"] == "queued"
    assert running["job_id"] == "image-job-1"
    assert running["prompt_id"] == "prompt-1"
    assert running["warnings"] == ["test-warning"]

    image.release.set()
    await _wait_for_submission(submission_id)
    done = await server.get_scene_image_job(job_id="image-job-1")
    assert done["status"] == "completed"
    assert done["image_path"] == "assets/images/scene-001.png"
    assert done["submission_id"] == submission_id
    assert "C:\\" not in str(done)
    assert services.image is image


@pytest.mark.asyncio
async def test_generate_scene_image_forwards_all_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = FakeGenerator("image")
    _install_fakes(monkeypatch, image=image)
    await server.generate_scene_image(
        "alpha",
        "scene-1",
        workflow="wf",
        positive_prompt="prompt",
        negative_prompt="bad",
        width=768,
        height=512,
        seed=7,
        steps=8,
        cfg=1.0,
        regenerate=True,
        compose_prompt=True,
        prompt_language="en",
        character_references=["hero.png"],
        visual_style="cinematic",
        color_palette="warm",
        character_appearance="tall",
        timeout_s=42.0,
        poll_interval_s=0.25,
    )
    await _wait_for_submission(str(next(iter(server._SUBMISSIONS))))
    method, target, kwargs = image.calls[0]
    assert method == "generate_scene_image"
    assert target == ("alpha", "scene-1")
    assert kwargs == {
        "workflow": "wf",
        "positive_prompt": "prompt",
        "negative_prompt": "bad",
        "width": 768,
        "height": 512,
        "seed": 7,
        "steps": 8,
        "cfg": 1.0,
        "regenerate": True,
        "compose_prompt": True,
        "prompt_language": "en",
        "character_references": ["hero.png"],
        "visual_style": "cinematic",
        "color_palette": "warm",
        "character_appearance": "tall",
        "timeout_s": 42.0,
        "poll_interval_s": 0.25,
    }


@pytest.mark.asyncio
async def test_generate_scene_image_rejects_missing_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scene_service import SceneNotFoundError

    services = _install_fakes(monkeypatch)
    services.scenes.raise_on_get = SceneNotFoundError("no scene")
    result = await server.generate_scene_image("alpha", "missing")
    assert result["status"] == "error"
    assert "SceneNotFoundError" in str(result["error"])
    assert not server._TASKS


@pytest.mark.asyncio
async def test_retry_scene_image_creates_new_service_job_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = FakeGenerator("image")
    failed = image._new_job("alpha", "scene-1", status="failed")
    _install_fakes(monkeypatch, image=image)

    result = await server.retry_scene_image("alpha", "scene-1", timeout_s=10.0)
    submission_id = str(result["submission_id"])
    assert result["status"] == "submitted"
    assert image.calls[0][0] == "retry_scene_image"
    assert image.calls[0][2] == {"timeout_s": 10.0}
    await _wait_for_submission(submission_id)
    view = await server.get_scene_image_job(submission_id=submission_id)
    assert view["job_id"] != failed["job_id"]
    assert view["status"] == "completed"


@pytest.mark.asyncio
async def test_retry_scene_image_without_failed_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    result = await server.retry_scene_image("alpha", "scene-1")
    assert result["status"] == "error"
    assert "No failed image job" in str(result["error"])


# --- video generation ---------------------------------------------------


@pytest.mark.asyncio
async def test_generate_scene_video_is_non_blocking_and_links_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = FakeGenerator("video")
    video.block = True
    _install_fakes(monkeypatch, video=video)
    started = await server.generate_scene_video("alpha", "scene-1", frames=97, fps=24.0)
    assert started["status"] == "submitted"
    assert started["job_id"] == "video-job-1"
    submission_id = str(started["submission_id"])
    running = await server.get_scene_video_job(submission_id=submission_id)
    assert running["status"] == "queued"
    assert running["prompt_id"] == "prompt-1"
    video.release.set()
    await _wait_for_submission(submission_id)
    done = await server.get_scene_video_job(submission_id=submission_id)
    assert done["status"] == "completed"
    assert done["output_path"] == "assets/videos/scene-001.png"
    assert done["warnings"] == ["test-warning"]
    assert video.calls[0][2]["frames"] == 97
    assert video.calls[0][2]["fps"] == 24.0
    assert video.calls[0][2]["allow_default_prompt"] is False


@pytest.mark.asyncio
async def test_retry_scene_video_reuses_job_id_and_increments_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = FakeGenerator("video")
    failed = video._new_job("alpha", "scene-1", status="failed")
    _install_fakes(monkeypatch, video=video)
    started = await server.retry_scene_video("alpha", "scene-1", poll_interval_s=0.5)
    submission_id = str(started["submission_id"])
    assert video.calls[0][2] == {"poll_interval_s": 0.5}
    await _wait_for_submission(submission_id)
    view = await server.get_scene_video_job(job_id=failed["job_id"])
    assert view["job_id"] == failed["job_id"]
    assert view["attempt"] == 2
    assert view["status"] == "completed"


@pytest.mark.asyncio
async def test_retry_scene_video_without_failed_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    result = await server.retry_scene_video("alpha", "scene-1")
    assert result["status"] == "error"
    assert "No failed video job" in str(result["error"])


# --- cancellation -------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_video_job_preserves_service_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = FakeGenerator("video")
    job = video._new_job("alpha", "scene-1")
    _install_fakes(monkeypatch, video=video)
    result = await server.cancel_scene_video(job_id=job["job_id"])
    assert result["status"] == "cancelled"
    assert result["supported"] is True
    assert result["error"] is None
    assert result["job_id"] == job["job_id"]
    assert "submission_id" not in result
    assert video.cancel_calls == [job["job_id"]]


@pytest.mark.asyncio
async def test_cancel_video_unsupported_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = FakeGenerator("video")
    job = video._new_job("alpha", "scene-1")
    _install_fakes(monkeypatch, video=video)
    video.cancel_result = {"status": "running", "supported": False, "error": "no API"}
    result = await server.cancel_scene_video(job_id=job["job_id"])
    assert result["status"] == "running"
    assert result["supported"] is False
    assert result["error"] == "no API"

    video.cancel_result = JobNotFoundError("unknown job")
    error = await server.cancel_scene_video(job_id=job["job_id"])
    assert error["status"] == "error"
    assert "JobNotFoundError" in str(error["error"])


@pytest.mark.asyncio
async def test_cancel_submission_before_service_job_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = FakeGenerator("video")
    video.block = True
    _install_fakes(monkeypatch, video=video)
    started = await server.generate_scene_video("alpha", "scene-1")
    # Simulate a submission blocked before the service registers a job.
    video.jobs.clear()
    submission_id = str(started["submission_id"])
    server._SUBMISSIONS[submission_id].job_id = None
    task = server._TASKS[submission_id]
    result = await server.cancel_scene_video(submission_id=submission_id)
    assert result["status"] == "cancelled"
    assert result["job_id"] is None
    await asyncio.wait({task}, timeout=1.0)
    assert task.cancelled()
    view = await server.get_scene_video_job(submission_id=submission_id)
    assert view["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_requires_an_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    result = await server.cancel_scene_video()
    assert result["status"] == "error"


# --- failures, unknown IDs, shutdown ------------------------------------


@pytest.mark.asyncio
async def test_task_exception_becomes_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = FakeGenerator("image")
    image.fail_with = ValueError("generator exploded")
    _install_fakes(monkeypatch, image=image)
    started = await server.generate_scene_image("alpha", "scene-1")
    submission_id = str(started["submission_id"])
    task = server._TASKS[submission_id]
    await asyncio.wait_for(task, timeout=2.0)
    view = await server.get_scene_image_job(submission_id=submission_id)
    assert view["status"] == "error"
    assert view["error"] == "ValueError: generator exploded"


@pytest.mark.asyncio
async def test_unknown_jobs_and_missing_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    image = await server.get_scene_image_job()
    video = await server.get_scene_video_job(job_id="does-not-exist")
    retry = await server.get_scene_video_job(submission_id="does-not-exist")
    assert image["status"] == "error"
    assert video["status"] == "error"
    assert retry["status"] == "error"
    assert "Unknown" in str(video["error"])


@pytest.mark.asyncio
async def test_external_service_job_is_pollable_without_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = FakeGenerator("video")
    job = video._new_job("alpha", "scene-1")
    _install_fakes(monkeypatch, video=video)
    view = await server.get_scene_video_job(job_id=job["job_id"])
    assert view["status"] == "queued"
    assert view["submission_id"] is None
    assert view["job_id"] == job["job_id"]


@pytest.mark.asyncio
async def test_shared_gpu_lock_serializes_image_and_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = FakeGPUProbe()
    image = FakeGenerator("image", probe)
    video = FakeGenerator("video", probe)
    image.block = True
    _install_fakes(monkeypatch, image=image, video=video)

    first = await server.generate_scene_image("alpha", "scene-1")
    second = await server.generate_scene_video("alpha", "scene-1")
    await asyncio.sleep(0)
    assert probe.active == 1
    assert image.calls and not video.calls

    image.release.set()
    await asyncio.wait_for(
        asyncio.gather(
            server._TASKS[str(first["submission_id"])],
            server._TASKS[str(second["submission_id"])],
        ),
        timeout=2.0,
    )
    assert probe.max_active == 1
    assert video.calls


@pytest.mark.asyncio
async def test_shutdown_cancels_active_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    image = FakeGenerator("image")
    image.block = True
    _install_fakes(monkeypatch, image=image)
    started = await server.generate_scene_image("alpha", "scene-1")
    submission_id = str(started["submission_id"])
    await server._shutdown_submissions()
    view = await server.get_scene_image_job(submission_id=submission_id)
    assert view["status"] == "cancelled"
    assert submission_id not in server._TASKS
