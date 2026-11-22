"""File-backed, atomic job store for the final render orchestrator.

Jobs live in ``<root>/jobs/<job_id>.json`` and cross-process lock files in
``<root>/locks/``. Writes go through :func:`project_paths.write_json_atomic`,
so readers never observe a partially written job. Locks combine a process-local
reentrant lock with ``O_CREAT|O_EXCL`` lock files, with stale-file detection.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from project_paths import validate_identifier, write_json_atomic
from render_job_models import (
    RenderConfig,
    RenderJob,
    RenderJobError,
    RenderProgress,
    utc_now,
)

logger = logging.getLogger(__name__)

DEFAULT_LOCK_TIMEOUT_S = 300.0
LOCK_POLL_INTERVAL_S = 0.02
GLOBAL_LOCK_NAME = "global"
RECOVERY_LOCK_NAME = "recovery"

_THREAD_LOCKS: dict[tuple[str, str], Any] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class _LockLocal(threading.local):
    held: dict[tuple[str, str], int]


_LOCAL = _LockLocal()


def _default_root() -> Path:
    """Resolve the store root from the environment or the repository root."""
    configured = os.getenv("RENDER_JOB_STORE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    projects = os.getenv("PROJECT_AGENT_PROJECTS_ROOT", "").strip()
    if projects:
        return Path(projects).expanduser().resolve().parent / ".render-jobs"
    return Path(__file__).resolve().parent / ".render-jobs"


class RenderJobStore:
    """Persistent store for final render jobs."""

    def __init__(
        self,
        root: str | Path | None = None,
        recover_on_init: bool = True,
        *,
        lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    ) -> None:
        self.root = _default_root() if root is None else Path(root).expanduser().resolve()
        self.jobs_dir = self.root / "jobs"
        self.locks_dir = self.root / "locks"
        if isinstance(lock_timeout_s, bool) or not isinstance(lock_timeout_s, (int, float)):
            raise TypeError("lock_timeout_s must be a number")
        self.lock_timeout_s = float(lock_timeout_s)
        if self.lock_timeout_s <= 0:
            raise ValueError("lock_timeout_s must be positive")
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        if recover_on_init:
            self.recover_interrupted()

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{validate_identifier(job_id, 'job_id')}.json"

    def _read_job_file(self, path: Path) -> RenderJob:
        try:
            payload: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RenderJobError(
                "render.job.corrupt",
                f"Could not read render job file {path.name}: {exc}",
                path.name,
            ) from exc
        if not isinstance(payload, Mapping):
            raise RenderJobError(
                "render.job.corrupt",
                f"Render job file must contain an object: {path.name}",
                path.name,
            )
        try:
            return RenderJob.from_dict(payload)
        except (TypeError, ValueError) as exc:
            raise RenderJobError(
                "render.job.corrupt",
                f"Invalid render job file {path.name}: {exc}",
                path.name,
            ) from exc

    def _write_job(self, job: RenderJob) -> RenderJob:
        if not isinstance(job, RenderJob):
            raise TypeError("job must be a RenderJob")
        write_json_atomic(self._job_path(job.job_id), job.to_dict())
        return job

    @staticmethod
    def _thread_lock(key: tuple[str, str]) -> Any:
        with _THREAD_LOCKS_GUARD:
            lock = _THREAD_LOCKS.get(key)
            if lock is None:
                lock = threading.RLock()
                _THREAD_LOCKS[key] = lock
            return lock

    @staticmethod
    def _local_held() -> dict[tuple[str, str], int]:
        held: dict[tuple[str, str], int] | None = getattr(_LOCAL, "held", None)
        if held is None:
            held = {}
            _LOCAL.held = held
        return held

    def _lock_path(self, name: str) -> Path:
        return self.locks_dir / f"{name}.lock"

    def _remove_stale_lock(self, path: Path) -> None:
        try:
            age_s = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Could not stat render lock %s: %s", path.name, exc)
            return
        if age_s <= self.lock_timeout_s:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Could not remove stale render lock %s: %s", path.name, exc)

    def _acquire_file_lock(self, name: str) -> str:
        path = self._lock_path(name)
        deadline = time.monotonic() + self.lock_timeout_s
        token = uuid.uuid4().hex
        while True:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                self._remove_stale_lock(path)
                if time.monotonic() >= deadline:
                    raise RenderJobError(
                        "render.job.lock_timeout",
                        f"Timed out waiting for render lock {name!r}",
                        name,
                    )
                time.sleep(LOCK_POLL_INTERVAL_S)
                continue
            try:
                os.write(descriptor, token.encode("ascii"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return token

    def _release_file_lock(self, name: str, token: str) -> None:
        path = self._lock_path(name)
        try:
            current = path.read_text(encoding="ascii")
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Could not read render lock %s: %s", path.name, exc)
            return
        if current != token:
            logger.warning("Render lock %s no longer belongs to this owner", name)
            return
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Could not remove render lock %s: %s", path.name, exc)

    @contextmanager
    def lock(self, project_id: str, *, global_: bool = True) -> Iterator[None]:
        """Serialize job mutations in-process and across processes."""
        name = (
            GLOBAL_LOCK_NAME
            if global_
            else validate_identifier(project_id, "project_id")
        )
        key = (str(self.root), name)
        thread_lock = self._thread_lock(key)
        thread_lock.acquire()
        token: str | None = None
        held = self._local_held()
        try:
            if key in held:
                held[key] += 1
            else:
                token = self._acquire_file_lock(name)
                held[key] = 1
            yield
        finally:
            if token is not None:
                self._release_file_lock(name, token)
                held.pop(key, None)
            else:
                remaining = held.get(key, 1) - 1
                if remaining > 0:
                    held[key] = remaining
                else:
                    held.pop(key, None)
            thread_lock.release()

    def create_job(
        self,
        project_id: str,
        config: RenderConfig | Mapping[str, object],
        *,
        job_id: str | None = None,
        render_id: str | None = None,
    ) -> RenderJob:
        """Create and persist a queued job for *project_id*."""
        project_id = validate_identifier(project_id, "project_id")
        resolved_job_id = (
            f"job-{uuid.uuid4().hex}"
            if job_id is None
            else validate_identifier(job_id, "job_id")
        )
        resolved_render_id = (
            f"render-{uuid.uuid4().hex}"
            if render_id is None
            else validate_identifier(render_id, "render_id")
        )
        if isinstance(config, RenderConfig):
            render_config = config
        elif isinstance(config, Mapping):
            try:
                render_config = RenderConfig.from_dict(config)
            except (TypeError, ValueError) as exc:
                raise RenderJobError(
                    "render.job.invalid_config",
                    f"Invalid render configuration: {exc}",
                    resolved_job_id,
                ) from exc
        else:
            raise RenderJobError(
                "render.job.invalid_config",
                "config must be a RenderConfig or a mapping",
                resolved_job_id,
            )

        with self.lock(project_id):
            if self._job_path(resolved_job_id).is_file():
                raise RenderJobError(
                    "render.job.duplicate_job_id",
                    f"Render job already exists: {resolved_job_id}",
                    resolved_job_id,
                )
            jobs = self.list_jobs()
            active = [
                job
                for job in jobs
                if job.project_id == project_id and job.is_active
            ]
            if active:
                raise RenderJobError(
                    "render.job.duplicate_active",
                    f"Project {project_id} already has an active render job: "
                    f"{active[0].job_id}",
                    resolved_job_id,
                )
            if any(job.render_id == resolved_render_id for job in jobs):
                raise RenderJobError(
                    "render.job.duplicate_render_id",
                    f"Render id already exists: {resolved_render_id}",
                    resolved_job_id,
                )
            now = utc_now()
            job = RenderJob(
                job_id=resolved_job_id,
                project_id=project_id,
                render_id=resolved_render_id,
                config=render_config.to_dict(),
                created_at=now,
                updated_at=now,
                stages=[
                    {
                        "stage": "queued",
                        "status": "queued",
                        "timestamp": now,
                        "message": "job created",
                    }
                ],
            )
            return self._write_job(job)

    def get_job(self, job_id: str) -> RenderJob | None:
        """Return one job, or ``None`` when it does not exist."""
        path = self._job_path(job_id)
        if not path.is_file():
            return None
        return self._read_job_file(path)

    def list_jobs(self, project_id: str | None = None) -> list[RenderJob]:
        """Return all jobs (optionally for one project), oldest first."""
        selected = validate_identifier(project_id, "project_id") if project_id is not None else None
        jobs: list[RenderJob] = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            job = self._read_job_file(path)
            if selected is None or job.project_id == selected:
                jobs.append(job)
        jobs.sort(key=lambda item: (item.created_at, item.job_id))
        return jobs

    def save_job(self, job: RenderJob) -> RenderJob:
        """Atomically persist the current state of *job*."""
        if not isinstance(job, RenderJob):
            raise TypeError("job must be a RenderJob")
        with self.lock(job.project_id):
            return self._write_job(job)

    def request_cancel(self, job_id: str) -> bool:
        """Flag an active job for cancellation; return False for unknown/terminal jobs."""
        resolved_job_id = validate_identifier(job_id, "job_id")
        with self.lock(resolved_job_id):
            job = self.get_job(resolved_job_id)
            if job is None or job.is_terminal:
                return False
            if job.cancel_requested:
                return True
            now = utc_now()
            stages = list(job.stages)
            stages.append(
                {
                    "stage": job.stage,
                    "status": job.status,
                    "timestamp": now,
                    "message": "cancellation requested",
                }
            )
            self._write_job(
                job.evolve(cancel_requested=True, updated_at=now, stages=stages)
            )
            return True

    def recover_interrupted(self) -> list[RenderJob]:
        """Fail (or cancel) jobs still active when the process restarted."""
        recovered: list[RenderJob] = []
        with self.lock(RECOVERY_LOCK_NAME):
            for job in self.list_jobs():
                if not job.is_active:
                    continue
                now = utc_now()
                stages = list(job.stages)
                if job.cancel_requested:
                    status = "cancelled"
                    error: dict[str, object] | None = None
                    message = "job was cancelled before restart"
                else:
                    status = "failed"
                    error = RenderJobError(
                        "render.job.interrupted",
                        "Render job was interrupted by a restart",
                        job.job_id,
                    ).to_dict()
                    message = "active job marked failed after restart"
                stages.append(
                    {
                        "stage": status,
                        "status": status,
                        "timestamp": now,
                        "message": message,
                    }
                )
                updated = job.evolve(
                    status=status,
                    stage=status,
                    progress=RenderProgress(stage=status),
                    error=error,
                    completed_at=now,
                    updated_at=now,
                    stages=stages,
                )
                self._write_job(updated)
                recovered.append(updated)
        return recovered
