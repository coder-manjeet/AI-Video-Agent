# ADR 0010 — Dedicated Project MCP Server

- Status: Accepted
- Date: 2026-11-22

## Context

Milestones 2A–2C added MCP-independent services for projects, scenes, character
identity, scene image generation and scene animation. Those services had no MCP
tools, so callers had to use Python directly. The three existing MCP servers
(`local-video-agent`, `local-tts-agent`, `local-media-agent`) and their services
were already verified and had to remain unchanged.

A new tool surface needed to expose project and scene management plus image and
video generation without duplicating business logic, adding dependencies,
changing workflows, or starting cross-process GPU coordination. Service job IDs
are created inside long-running coroutines, so a synchronous launch tool cannot
return the real job ID before the task has advanced.

## Decision

- Add `project_agent_server.py`, a thin FastMCP stdio server named
  `local-project-agent` under `.venv`. It exposes exactly thirteen tools:
  `create_project`, `list_projects`, `get_project`, `add_scene`, `list_scenes`,
  `update_scene`, `generate_scene_image`, `get_scene_image_job`,
  `retry_scene_image`, `generate_scene_video`, `get_scene_video_job`,
  `retry_scene_video` and `cancel_scene_video`.
- Instantiate `ProjectService`, `SceneService`, `CharacterService`,
  `SceneImageService` and `SceneVideoService` lazily and share one instance of
  each. The projects root is explicitly
  `Path(__file__).resolve().parent / "projects"`, optionally overridden by
  `PROJECT_AGENT_PROJECTS_ROOT`.
- Keep all validation and generation logic in the existing services. Tools
  validate identifiers with `project_paths.validate_identifier`, move blocking
  service calls to worker threads, and return structured JSON-compatible
  results with `status="error"` and a typed error string for expected failures.
- Generation tools start an `asyncio` task and return immediately with a
  `submission_id` and `status="submitted"`. Strong task references are kept in
  a server registry. A best-effort linkage to the service `job_id` is made after
  yielding once; the service job registry remains authoritative for status,
  output paths, errors and warnings. Poll tools accept either a submission ID
  or a service job ID.
- Image retry calls `retry_scene_image()` and therefore creates a new service
  job ID. Video retry calls `retry_scene_video()`, which reuses the existing
  job ID and increments `attempt`.
- A shared `asyncio.Lock` serialises image and video generation **within this
  server only**. It does not coordinate GPU work from other MCP processes,
  including `local-tts-agent` or the legacy `local-video-agent`. Cross-process
  locking is deliberately out of scope.
- Cancellation uses the video service for known service jobs. If no service job
  exists yet, the tracked submission task is cancelled. Unsupported service
  cancellation and its error fields are preserved.
- Register the server in `opencode.json` as the fourth local MCP entry using
  the existing `.venv` Python and absolute script/cwd paths. The previous
  configuration was backed up and the three existing entries were unchanged.

## Consequences

- Project creation, scene editing, image generation, animation, polling, retry
  and cancellation are now available as MCP tools without modifying any
  existing server or service.
- Submitted generation is non-blocking, but a submission handle is not the same
  as a service job ID; callers poll by either identifier until the job appears.
- The in-server GPU lock prevents this process from overlapping image and video
  work. It cannot prevent a separate MCP process from starting GPU work, so the
  existing one-GPU-job-at-a-time rule still applies manually.
- Service status vocabulary is preserved. A task-level exception is reported
  as an error when the service registry has not already recorded a failure.
- No generated asset is overwritten by the new tools; overwrite decisions stay
  in the existing services.
- The 33 new mocked tests use fake services only: no ComfyUI server, GPU, TTS,
  FFmpeg or real media generation is required. The full suite is 267 tests, and
  Ruff and Mypy pass.
- Final assembly, audio/subtitle integration, UI and cross-process GPU
  coordination remain future work.
