# ADR 0006 — Scene-Based Project and Timeline Foundation

- Status: Accepted
- Date: 2026-11-19

## Context

Milestone 1 required independent scene production units, a single source of truth
for scene order and state, manifest/asset validation, and timeline calculation,
while preserving the three working MCP servers, the tested LTX workflow, TTS,
FFmpeg and 63 passing tests. ADR 0001 deferred any `src/` migration.

## Decision

- Add six flat root modules (`project_models.py`, `project_paths.py`,
  `project_service.py`, `scene_service.py`, `manifest_validation.py`,
  `timeline_planner.py`) instead of creating `src/local_video_agent/`. Existing
  files are not reorganized, and the MCP servers keep no new tools.
- Store project metadata in `project.json` and scene data only in
  `script/scenes.json`. The scene manifest is an object with `schema_version`
  and a `scenes` array; every project-relative asset path is stored as a
  POSIX-style relative string.
- Keep models provider-agnostic: `generation_provider` and `provider_options`
  replace any LTX-specific fields, and unknown JSON keys are preserved in
  `extra` through load/update/save cycles.
- Write manifests atomically using a same-directory temp file and `replace()`.
  Never overwrite an existing project or delete assets when a scene is removed.
- Validate with structured results: invalid identifiers, ordering, numbers,
  status values and escaping paths are errors; missing generated asset files
  are warnings so incomplete draft scenes remain valid.
- Calculate timeline positions from `Decimal(str(duration))`, then convert to
  floats for JSON. Planning is pure; optional persistence writes calculated
  `timeline_start`/`timeline_end` back to the manifest through the scene
  service. `tracks` is reserved for future narration, music, SFX and subtitles.

## Consequences

- The new services are directly importable and unit-testable without MCP, GPU,
  ComfyUI or FFmpeg. No new runtime dependencies were introduced.
- A future editor can consume `TimelinePlan.to_dict()` and scene cards without
  new backend layers; multi-track planning and rendering remain separate.
- Root-module count grows; a future `src/` migration can move these modules
  together without changing their public interfaces.
