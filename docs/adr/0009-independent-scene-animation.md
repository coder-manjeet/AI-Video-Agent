# ADR 0009 — Independent Scene Animation Service

- Status: Accepted
- Date: 2026-11-22

## Context

Milestone 2C required animation for each scene from its approved image and
animation prompt, independently of every other scene. The verified LTX
image-to-video workflow already existed (`ltx_image_to_video_api.json`, 768x512,
97 frames, 24 FPS, about 4.04 s) and had to remain byte-for-byte unchanged.
`comfyui_client.py` already owns ComfyUI HTTP calls; `workflow_adapter.py` had an
image-only configuration layer. `media_service.probe_media()` supplies FFprobe
metadata, but `validate_media()` cannot be used for scene clips because it
requires an audio stream and LTX outputs are silent. Final rendering, audio,
subtitles, MCP servers and UI were out of scope.

## Decision

- Add `video_workflows.json` as a separate declarative registry for the LTX
  workflow. Node IDs, expected classes, verified model values, bindings for
  image/positive/negative prompts, dimensions, frames, FPS, seed and output
  (`gifs` from `VHS_VideoCombine`) live only in that file.
- Extend `workflow_adapter.py` additively with image, frame and FPS bindings,
  an output `check_input` flag for history-only output keys, and an
  `extract_output_entry` alias. The existing image registry and its tests keep
  their behaviour.
- Add generic `download_output_file`, `get_object_info` and `cancel_prompt`
  helpers to `comfyui_client.py`; `download_output_image` delegates to the
  generic downloader. No service implements its own HTTP client.
- `scene_video_service.py` owns generation only. It validates the project,
  scene, image readability and prompt availability; checks workflow structure
  and configured model availability through `/object_info`; uploads the image;
  patches a fresh workflow copy; submits, polls and retrieves the MP4; and
  verifies it with FFprobe (container, codec, duration, resolution, FPS).
  An animation prompt is used only when supplied by the scene or caller, or
  when a default is explicitly permitted.
- Jobs are serialized with an `asyncio.Lock`. The job registry records job,
  project and scene IDs, ComfyUI prompt ID, status, submission and completion
  timestamps, output path, attempt count and errors. Submission retries,
  polling timeouts, cancellation and per-job/per-scene retries are supported.
- Output files are versioned (`scene-001-opening.mp4`, then `-v2`, …) under
  `projects/{project_id}/video/scenes/` and never overwritten. Only after
  verification does a single `update_scene()` write `generated_video_path`,
  `status="video_generated"` and merged generation metadata in
  `provider_options["video_generation"]`. Planned `duration` remains separate
  from measured `actual_duration_s`.
- Failures never change the manifest, and invalid downloads are removed. One
  controlled live job was run against the verified 2A image before completion.

## Consequences

- Scenes can be animated, retrieved, retried and replaced independently; final
  assembly remains a separate future milestone.
- All unit tests use a fake ComfyUI client and a fake FFprobe callable, so no
  GPU or server is required. The full suite is 234 tests; Ruff and Mypy pass.
- Changing frames or FPS is configurable but only the verified defaults were
  smoke-tested; actual duration is always measured rather than inferred from
  the scene's planned duration.
- Cancellation uses ComfyUI's `/queue` deletion endpoint, which only removes
  jobs that have not started; a running job must be inspected separately.
- No new dependencies, models or custom nodes were installed, and no MCP tools
  or UI were added.
