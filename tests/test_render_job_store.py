"""Tests for the atomic, file-backed render job store."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from project_paths import write_json_atomic
from render_job_models import RenderConfig, RenderJob, RenderJobError
from render_job_store import RenderJobStore, _default_root


@pytest.fixture()
def store(tmp_path: Path) -> RenderJobStore:
    return RenderJobStore(
        root=tmp_path / "store",
        recover_on_init=False,
        lock_timeout_s=5.0,
    )


def _write_active_job(
    store: RenderJobStore,
    *,
    project_id: str = "demo",
    job_id: str = "job-active",
    render_id: str = "render-active",
    **changes: object,
) -> RenderJob:
    payload: dict[str, object] = {
        "job_id": job_id,
        "project_id": project_id,
        "render_id": render_id,
        "created_at": "2000-01-01T00:00:00Z",
        "updated_at": "2000-01-01T00:00:00Z",
        **changes,
    }
    job = RenderJob.from_dict(payload)
    write_json_atomic(store.jobs_dir / f"{job_id}.json", job.to_dict())
    return job


def test_default_root_follows_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("RENDER_JOB_STORE_ROOT", str(explicit))
    assert _default_root() == explicit.resolve()

    monkeypatch.delenv("RENDER_JOB_STORE_ROOT")
    projects = tmp_path / "projects"
    monkeypatch.setenv("PROJECT_AGENT_PROJECTS_ROOT", str(projects))
    assert _default_root() == tmp_path / ".render-jobs"

    monkeypatch.delenv("PROJECT_AGENT_PROJECTS_ROOT", raising=False)
    assert _default_root() == Path(__file__).resolve().parent.parent / ".render-jobs"


def test_create_job_persists_defaults_without_temporary_files(store: RenderJobStore) -> None:
    job = store.create_job("demo", {})

    assert job.job_id.startswith("job-")
    assert job.render_id.startswith("render-")
    assert job.status == "queued"
    assert job.is_active
    assert job.created_at.endswith("Z")
    assert job.updated_at == job.created_at
    assert store.jobs_dir.is_dir()
    assert store.locks_dir.is_dir()
    assert list(store.locks_dir.glob("*.lock")) == []

    path = store.jobs_dir / f"{job.job_id}.json"
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["config"]["preset"] == "1080p"
    assert payload["stages"][0]["message"] == "job created"
    assert list(store.jobs_dir.glob("*.tmp")) == []
    assert "C:" not in text


def test_create_job_accepts_mapping_and_typed_config(store: RenderJobStore) -> None:
    from_mapping = store.create_job("demo", {"preset": "720p"}, job_id="job-map")
    from_typed = store.create_job(
        "other",
        RenderConfig.from_dict({"preset": "custom", "width": 960, "height": 540}),
        job_id="job-typed",
    )

    assert from_mapping.config["width"] == 1280
    assert from_typed.config["height"] == 540


def test_create_job_rejects_invalid_config(store: RenderJobStore) -> None:
    with pytest.raises(RenderJobError) as caught:
        store.create_job("demo", {"preset": "4k"})

    assert caught.value.code == "render.job.invalid_config"
    assert store.list_jobs() == []


def test_create_job_rejects_existing_job_id(store: RenderJobStore) -> None:
    store.create_job("demo", {}, job_id="job-fixed")

    with pytest.raises(RenderJobError) as caught:
        store.create_job("other", {}, job_id="job-fixed")

    assert caught.value.code == "render.job.duplicate_job_id"


def test_duplicate_active_job_is_rejected(store: RenderJobStore) -> None:
    first = store.create_job("demo", {}, job_id="job-1")

    with pytest.raises(RenderJobError) as caught:
        store.create_job("demo", {}, job_id="job-2")

    assert caught.value.code == "render.job.duplicate_active"
    assert first.job_id in str(caught.value)


def test_terminal_job_allows_new_job_but_render_ids_stay_unique(store: RenderJobStore) -> None:
    first = store.create_job("demo", {}, job_id="job-1", render_id="render-1")
    store.save_job(first.with_status("completed", updated_at="2000-01-01T00:01:00Z"))

    with pytest.raises(RenderJobError) as caught:
        store.create_job("demo", {}, job_id="job-2", render_id="render-1")
    assert caught.value.code == "render.job.duplicate_render_id"

    second = store.create_job("demo", {}, job_id="job-3")
    assert second.is_active


def test_get_job_unknown_and_project_filtering(store: RenderJobStore) -> None:
    store.create_job("demo", {}, job_id="job-1")
    store.create_job("other", {}, job_id="job-2")

    assert store.get_job("missing") is None
    assert [job.job_id for job in store.list_jobs("demo")] == ["job-1"]
    assert {job.job_id for job in store.list_jobs()} == {"job-1", "job-2"}
    with pytest.raises(ValueError):
        store.list_jobs("../escape")


def test_corrupt_job_file_raises_contextual_error(store: RenderJobStore) -> None:
    path = store.jobs_dir / "job-bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RenderJobError) as caught:
        store.get_job("job-bad")

    assert caught.value.code == "render.job.corrupt"
    assert caught.value.location == "job-bad.json"
    assert "Could not read" in str(caught.value)


def test_list_jobs_is_sorted_by_created_at(store: RenderJobStore) -> None:
    first = store.create_job("proj-a", {}, job_id="job-a")
    second = store.create_job("proj-b", {}, job_id="job-b")
    store.save_job(
        first.evolve(created_at="2000-01-02T00:00:00Z", updated_at="2000-01-02T00:00:00Z")
    )
    store.save_job(
        second.evolve(created_at="2000-01-01T00:00:00Z", updated_at="2000-01-01T00:00:00Z")
    )

    assert [job.job_id for job in store.list_jobs()] == ["job-b", "job-a"]


def test_save_job_persists_status_update(store: RenderJobStore) -> None:
    job = store.create_job("demo", {}, job_id="job-1")
    updated = job.with_status(
        "merging", stage="merging", updated_at="2000-01-01T00:01:00Z"
    )

    assert store.save_job(updated) is updated
    persisted = store.get_job("job-1")
    assert persisted is not None
    assert persisted.status == "merging"
    assert persisted.updated_at == "2000-01-01T00:01:00Z"
    assert list(store.jobs_dir.glob("*.tmp")) == []


def test_request_cancel_is_active_only_and_idempotent(store: RenderJobStore) -> None:
    store.create_job("demo", {}, job_id="job-1")

    assert store.request_cancel("job-1") is True
    first = store.get_job("job-1")
    assert first is not None
    assert first.cancel_requested is True
    assert first.stages[-1]["message"] == "cancellation requested"

    assert store.request_cancel("job-1") is True
    second = store.get_job("job-1")
    assert second is not None
    assert second.updated_at == first.updated_at


def test_request_cancel_rejects_terminal_and_unknown_jobs(store: RenderJobStore) -> None:
    job = store.create_job("demo", {}, job_id="job-1")
    store.save_job(job.with_status("completed", updated_at="2000-01-01T00:01:00Z"))

    assert store.request_cancel("job-1") is False
    assert store.request_cancel("missing") is False


def test_recovery_fails_active_jobs_and_preserves_terminal_ones(store: RenderJobStore) -> None:
    _write_active_job(store, job_id="job-active", status="merging")
    done = store.create_job("other", {}, job_id="job-done")
    store.save_job(done.with_status("completed", updated_at="2000-01-01T00:01:00Z"))

    RenderJobStore(root=store.root, recover_on_init=True, lock_timeout_s=5.0)

    failed = store.get_job("job-active")
    assert failed is not None
    assert failed.status == "failed"
    assert failed.stage == "failed"
    assert failed.is_terminal
    assert failed.error is not None
    assert failed.error["code"] == "render.job.interrupted"
    assert failed.stages[-1]["stage"] == "failed"
    assert failed.completed_at is not None

    completed = store.get_job("job-done")
    assert completed is not None
    assert completed.status == "completed"
    assert completed.error is None


def test_recovery_can_be_deferred_and_cancels_requested_jobs(store: RenderJobStore) -> None:
    store.create_job("demo", {}, job_id="job-1")
    store.request_cancel("job-1")

    untouched = RenderJobStore(root=store.root, recover_on_init=False, lock_timeout_s=5.0)
    assert untouched is not None
    pending = store.get_job("job-1")
    assert pending is not None
    assert pending.is_active

    recovered = store.recover_interrupted()
    assert [item.job_id for item in recovered] == ["job-1"]
    cancelled = store.get_job("job-1")
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.error is None


def test_project_lock_serializes_threads(store: RenderJobStore) -> None:
    active = {"count": 0, "max": 0}
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait(timeout=5)
        with store.lock("demo", global_=False):
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            time.sleep(0.02)
            active["count"] -= 1

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert active["max"] == 1
    assert list(store.locks_dir.glob("*.lock")) == []


def test_global_lock_serializes_threads_across_projects(store: RenderJobStore) -> None:
    active = {"count": 0, "max": 0}
    barrier = threading.Barrier(2)

    def worker(project_id: str) -> None:
        barrier.wait(timeout=5)
        with store.lock(project_id):
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            time.sleep(0.02)
            active["count"] -= 1

    threads = [
        threading.Thread(target=worker, args=(project_id,))
        for project_id in ("demo", "other")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert active["max"] == 1


def test_lock_is_reentrant_and_released_on_error(store: RenderJobStore) -> None:
    with store.lock("demo", global_=False), store.lock("demo", global_=False):
        assert (store.locks_dir / "demo.lock").is_file()
    assert list(store.locks_dir.glob("*.lock")) == []

    with pytest.raises(RuntimeError, match="boom"), store.lock("demo", global_=False):
        raise RuntimeError("boom")
    assert list(store.locks_dir.glob("*.lock")) == []


def test_stale_lock_file_is_removed(store: RenderJobStore) -> None:
    path = store.locks_dir / "global.lock"
    path.write_text("stale-owner", encoding="ascii")
    old = time.time() - 10
    os.utime(path, (old, old))

    with store.lock("demo"):
        assert path.is_file()
        assert path.read_text(encoding="ascii") != "stale-owner"

    assert not path.exists()
