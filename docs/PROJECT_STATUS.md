# Project Status

Last updated: 2026-11-22

## Current milestone

Milestone 2C — Independent Scene Animation — is complete and verified.

Projects now have a structured `project.json`, a scene manifest at `script/scenes.json`, dedicated project/scene/character services, a validation layer, a sequential timeline planner, an independent scene image-generation service, a reusable character identity + visual prompt composition foundation, and an independent sequential scene-animation service. All business logic lives in MCP-independent modules. The three existing MCP servers, their configurations, `opencode.json`, `ltx_image_to_video_api.json`, `z_image_turbo_text_to_image_api.json`, TTS, FFmpeg behaviour, existing tests and previously generated media are unchanged. Final rendering, audio/subtitle integration, MCP tools for the new services and UI were not added.

The complete test suite is **234 passed** (201 previous + 33 new); ruff and mypy pass.

### Milestone 2C work

- Added `video_workflows.json`: a declarative registry for the verified LTX image-to-video workflow. Node IDs, expected classes/model values, image/prompt/frame/FPS bindings, validated limits and the `gifs` output from `VHS_VideoCombine` live in configuration rather than in service code.
- Extended `workflow_adapter.py` additively with image, frame and FPS bindings, an output `check_input` flag for history-only output keys and an `extract_output_entry` alias. Existing image-generation behaviour and tests are unchanged.
- Added generic `download_output_file`, `get_object_info` and `cancel_prompt` helpers to `comfyui_client.py`; `download_output_image` delegates to the generic downloader. No service implements its own HTTP client.
- Added `scene_video_service.py`: validates the project, scene, approved image and prompt; checks workflow structure and configured model availability; uploads the image; patches a fresh workflow copy; submits, polls and downloads the MP4; and verifies it with ffprobe. Jobs run one at a time under an `asyncio.Lock`.
- The job registry records job/project/scene IDs, ComfyUI prompt ID, status, timestamps, output path, attempt count and errors. Submission retries, polling timeouts, cancellation, job retries, scene retries and regeneration are supported; a failed job never replaces the last successful scene path.
- Versioned MP4s are written under `projects/{project_id}/video/scenes/` and are never overwritten. A single scene-manifest write registers `generated_video_path` and `status="video_generated"` only after verification. Planned scene duration stays separate from measured `actual_duration_s`.
- A default animation prompt is available only when explicitly supplied or permitted per call or service construction; otherwise a missing prompt is an actionable eligibility error.
- Added 33 fully mocked tests (`tests/test_video_workflow_adapter.py`, `tests/test_scene_video_service.py`); no GPU, ComfyUI server or live network is required.
- Ran one controlled live GPU job (prompt `35056b30-8ee9-45b9-9106-c241f4c9ba7a`, job `f8cb79b56ccd4f3ab2b247e4f0cc1ac7`) in a temporary project. The downloaded MP4 was 413,338 bytes and ffprobe confirmed 4.041667 s, 768x512, 24 FPS, h264 in an MP4 container; manifest metadata was written only for that verified scene.
- Recorded the architecture decision in [ADR 0009](adr/0009-independent-scene-animation.md).

### Milestone 2B work

- Added `character_models.py`: typed `Character`, `VisualIdentity` and `CharacterSceneOverride` dataclasses. All optional fields are non-forced; unknown JSON keys survive round-trips in `extra`.
- Added `character_service.py`: a project-level character registry at `assets/characters/characters.json` with atomic writes. Supports create/load/update/delete/list/get, reference-image registration (content-addressed copies, idempotent), reference retrieval, validation and missing-character detection. Deleting character metadata **never** deletes reference image files.
- Extended `project_models.py`: `Project.visual_identity` (typed, optional) plus `Scene.character_ids`, `Scene.character_overrides` and `Scene.visual_identity_overrides`. Legacy `character_references` image-path strings are untouched. Older manifests remain valid.
- Extended `manifest_validation.py`: `validate_character_manifest` with structured codes for schema, IDs, names, text, references and timestamps; project `visual_identity` validation; scene character/override validation. Optional manifests mean existing projects are unaffected.
- Extended `project_service.py`: `update_visual_identity(...)` persists, merges and clears project-wide visual identity in `project.json` with validation and atomic writes.
- Added `visual_prompt_builder.py`: a pure, deterministic service that composes project identity, character identities, scene description, character actions/expressions, the original image prompt and negative prompts. It de-duplicates repeated fragments, lets scene overrides replace base appearance/clothing, supports multiple or zero characters, localises section labels for `en/es/fr/de/hi/pt/zh` (English fallback), keeps original prompts available and returns structured debug sections.
- Extended `scene_image_service.py`: opt-in `compose_prompt=True` builds a composed prompt via the builder; the default path is unchanged. Composition debug info is stored in job metadata, and consistency inputs are preserved in `job.options` so retries reuse the same composed prompt without recomposing.
- Surveyed live ComfyUI 0.46.0 (1,596 nodes) and all local model directories. No reference-image conditioning is usable: IP-Adapter core nodes/models, ControlNet models, CLIP-Vision, PhotoMaker, InsightFace and style-model weights are absent (empty combos/directories). Generic image-to-image nodes exist but the verified Z-Image workflow has no image→sampler path. **Prompt composition improves intent, not guaranteed identity preservation.**
- Added 34 mocked tests across four new modules; no GPU, ComfyUI server, live network or new dependencies are required.
- Recorded the architecture decision in [ADR 0008](adr/0008-character-visual-consistency.md).

### Milestone 2A work

- Added `z_image_turbo_text_to_image_api.json`: a new 10-node API-format Z-Image Turbo workflow using `UNETLoader` (`z_image_turbo_bf16.safetensors`), `CLIPLoader` with **`lumina2`** (`qwen_3_4b.safetensors`), `VAELoader` (`ae.safetensors`), `EmptySD3LatentImage`, `ModelSamplingAuraFlow`, `KSampler`, `VAEDecode` and `SaveImage`. The existing LTX workflow was not touched.
- Verified the workflow offline against live ComfyUI `/object_info` (0 errors, all combo values and links valid), then ran **one** controlled 512×512 GPU job (`42ab02e1-a48e-4139-85e7-9bf154fa64e7`) that completed with `success`; the PNG was downloaded and verified as a readable 512×512 image (236,506 bytes).
- Added generic, additive helpers to `comfyui_client.py` (`load_workflow_file`, `submit_workflow`, `get_prompt_history`, `download_output_image`) with overwrite protection. Existing ComfyUI functions were not modified.
- Added `image_workflows.json`: a declarative registry with node bindings, expected classes/model values, capabilities, defaults, limits and notes. Node IDs are not hardcoded in the service.
- Added `workflow_adapter.py`: registry loading, offline compatibility validation (catches the incorrect `sd3` CLIP configuration), safe patching and history output extraction.
- Added `scene_image_service.py`: sequential generation with an `asyncio.Lock`, in-memory job registry, submission retry, status polling, timeout handling, PNG/dimension verification, unique versioned filenames (`scene-001-opening.png`, `scene-001-opening-v2.png`), and a single scene-manifest write **only after** the image is verified. Regenerating one scene never overwrites another scene's image or changes its fields.
- Added 25 mocked unit tests (`tests/test_workflow_adapter.py`, `tests/test_scene_image_service.py`); no GPU, ComfyUI server or live network is required.
- Recorded the architecture decision in [ADR 0007](adr/0007-scene-image-generation.md).

### Milestone 1 work

- Added `project_models.py`: typed, extensible `Project`, `ProjectSettings` and `Scene` dataclasses with JSON round-trips. Unknown manifest keys are preserved in `extra`.
- Added `project_paths.py`: identifier validation, project-root traversal protection, timestamp/language/aspect-ratio helpers and atomic JSON writes via a same-directory temp file.
- Added `project_service.py`: project creation with the full directory structure, load, list and metadata updates; existing projects are never overwritten.
- Added `scene_service.py`: scene add/update/delete/reorder/move, sequential renumbering, status updates, safe asset-path registration and manifest load/save. Deleting a scene never deletes generated assets.
- Added `manifest_validation.py`: structured errors and warnings for project metadata, scene order/status/durations and referenced assets. Missing generated assets are warnings, so draft scenes stay valid; escaping paths are errors.
- Added `timeline_planner.py`: `Decimal`-based sequential video-scene planning with editor-ready scene cards, track scaffolding and optional persistence of `timeline_start`/`timeline_end`.
- Added 79 unit tests across five new test modules; all use `tmp_path` and do not write to `projects/` or `output/`.
- Recorded the architecture decision in [ADR 0006](adr/0006-scene-project-foundation.md).
- Deliberately deferred: MCP tools for the new services, multi-track audio planning, scene generation, final assembly and any UI.

## Completed work (previous milestone)

Milestone 7 — Local FFmpeg Media Service, Phase 1 — was completed and verified end to end.

`media_service.py` provides reusable `probe_media`, `merge_narration` and `validate_media` operations using FFmpeg/ffprobe. A third MCP server, `local-media-agent`, exposes them over stdio. The original `local-video-agent` and `local-tts-agent` servers, their business logic, and the tested LTX workflow are unchanged. All three servers are connected simultaneously under `.venv`.

A live render combined the existing `ltxv_00006.mp4` clip with the verified English narration WAV into a new, validated MP4 using `extend_video` so no narration was cut off.

## Previous milestone work

- Audited the existing project, Git state, MCP integration, LTX workflow and TTS service before implementation.
- Verified FFmpeg and ffprobe are on PATH: `C:\ffmpeg\ffmpeg-9.0.1-essentials_build\bin`, version `9.0.1-essentials_build`. No installation was performed.
- Created `.gitignore` and `README.md`.
- Confirmed ComfyUI MCP status is working.
- Checked official Chatterbox documentation and dependency metadata.
- Proved that normal Chatterbox installation would downgrade Torch/TorchAudio.
- Recorded decisions in:
  - [ADR 0001](adr/0001-preserve-mcp-integration.md)
  - [ADR 0002](adr/0002-chatterbox-dependency-strategy.md)
  - [ADR 0003](adr/0003-windows-tts-dependency-pins.md)
  - [ADR 0004](adr/0004-tts-mcp-subprocess-isolation.md)
  - [ADR 0005](adr/0005-media-service-and-duration-policies.md)
- Pinned dev tools in `requirements-dev.txt`.
- Installed `chatterbox-tts==0.1.7` with `--no-deps` and explicit compatible dependencies; Gradio was not installed.
- Implemented `tts_service.py`, `tts_server.py` and `tts_worker.py`; registered `local-tts-agent` without changing `local-video-agent`.
- Implemented `media_service.py`: ffprobe metadata extraction, structured media validation, ffmpeg merge with AAC audio, configurable volume, two explicit duration policies, timeouts, temp-file render/validate/promote, and overwrite protection.
- Implemented `media_server.py` (`local-media-agent`): three thin async tools that validate inputs, move blocking work to worker threads and return structured `status="error"` results.
- Backed up `opencode.json` to `opencode.json.bak-20261118-000343`, then registered the third MCP server. Both existing registrations were left byte-for-byte unchanged.
- Added 32 new tests for the media layer (22 service tests, 10 MCP tool tests), including missing/invalid files, overwrite refusal and explicit overwrite, both duration policies, FFmpeg failures/timeouts, and successful output validation.
- Updated `AGENTS.md`, `README.md` and both relevant skill documents for the third server and media service.
- No packages were installed, upgraded or downgraded in either environment during this milestone.

## Verification results

| Check | Result |
|---|---|
| `ffmpeg -version` / `ffprobe -version` | `9.0.1-essentials_build`, on PATH |
| `import chatterbox.tts` / `import chatterbox.mtl_tts` | OK |
| `torch` / `torchaudio` | `2.10.0+cu130` |
| `numpy` | `1.26.4` |
| CUDA available | `True` |
| GPU | RTX 5060 Ti, `sm_120` |
| English / Hindi GPU inference (prior milestone) | Passed; quality review still pending for Hindi |
| `python -m ruff check .` | Passed |
| `python -m mypy server.py comfyui_client.py tts_service.py tts_server.py tts_worker.py media_service.py media_server.py workflow_adapter.py scene_video_service.py` | Passed |
| `python -m pytest -q` | 234 passed (201 previous + 33 new) |
| New modules: ruff / mypy | Passed (19 source files) |
| Scene project smoke checks | Directory scaffold, manifest round-trip, traversal rejection and timeline recalculation covered by unit tests |
| Scene animation mocked checks | 33 tests cover eligibility, workflow patching, submission retries, polling, timeout, download verification, versioning, manifest writes, failed-job preservation, retries, regeneration and cancellation |
| Live workflow preflight | LTX registry validation passed; configured CLIP/checkpoint models found through ComfyUI `/object_info` |
| Live scene animation job | Prompt `35056b30-8ee9-45b9-9106-c241f4c9ba7a`, job `f8cb79b56ccd4f3ab2b247e4f0cc1ac7`; completed successfully |
| Live scene MP4 verification | 413,338 bytes; 4.041667 s; 768x512; 24 FPS; h264; MP4 container |
| Live manifest registration | One target scene updated with generation metadata after verification; no other scene changed |
| `opencode mcp list` | `local-media-agent`, `local-tts-agent`, `local-video-agent` all connected |
| MCP `list_tools` (media) | `probe_media`, `merge_narration`, `validate_media` |
| MCP `probe_media` on source video / narration | Correct durations, codecs and stream metadata |
| MCP `merge_narration` with `extend_video` | Rendered and validated new MP4 in 4.25 s |
| MCP `validate_media` on final output | `valid=True`, all checks passed, zero errors |
| Duplicate render without `overwrite` | Rejected with `FileExistsError` message |
| Temp files after render | No `.tmp.mp4` files left in `test-output\` |

Live output created through the `local-media-agent` tool:

```text
C:\AI-Video-Agent\test-output\first_narrated_video.mp4
duration 4.25 s / 768x512 / 24 FPS / h264 video / AAC 24 kHz mono audio
534,698 bytes
```

Inputs and duration handling:

```text
C:\Users\HI User\Documents\ComfyUI\output\ltxv_00006.mp4
4.041667 s / 768x512 / 24 FPS / h264 / no audio stream

C:\AI-Video-Agent\test-output\mcp_english_verified.wav
4.24 s / 24 kHz / mono / pcm_s16le

extend_video: final frame frozen for 0.198 s; narration not truncated
validation duration delta: 0.01 s (within 0.5 s tolerance)
```

## Blockers and open decisions

- Hindi output quality still needs human evaluation; the smoke test only proves the pipeline runs.
- Hinglish has no official Chatterbox code; its script heuristic routes to `hi` or `en` but remains untested.
- Perceptual A/V sync and narration intelligibility are not automatically assessed; validation is stream/metadata level only.
- Large renders are synchronous per request; queueing and concurrency are future work.
- The scene foundation, image-generation service and scene-animation service have no MCP tools and no UI yet; callers use the Python services directly.
- Final video assembly, multi-track audio and subtitles are still future milestones. Scene animation is available as a Python service but is not integrated into final rendering or MCP.
- Custom animation dimensions, frames and FPS are exposed with validated limits, but only the verified LTX defaults (768x512, 97 frames, 24 FPS) were smoke-tested.
- Cancellation deletes queued ComfyUI prompts only; an already-running job cannot be forcibly stopped through the current API helper.
- LTX scene clips are silent and are checked with `probe_media()` plus video-only assertions; `validate_media()` is not suitable because it requires an audio stream.
- The Z-Image Turbo workflow does not support negative prompts or reference-image conditioning; supplied values are recorded as metadata and reported as warnings rather than silently applied.
- Real character identity preservation needs conditioning models/custom nodes that are not installed locally; Milestone 2B provides prompt-level consistency only and makes no guarantee. Enabling reference conditioning requires approval before installing models or nodes, plus a new verified workflow.
- Character registry and visual identity have no MCP tools and no UI yet; callers use the Python services directly.
- ComfyUI retains loaded models between jobs; sequential generation protects VRAM but does not unload models automatically.
- The Git repository has local commits and an `origin` remote; Milestone 2C changes are uncommitted and no remote operations were performed.

## Environment rules

- `.venv` is for MCP and dev tooling; it runs all three MCP servers and has no Torch.
- `.venv-tts` is the only TTS environment and is used by `tts_worker.py`.
- Never mix or recreate either environment without approval.
- Never upgrade, downgrade, or replace Torch, TorchAudio, or CUDA in `.venv-tts`.
- Never run TTS and ComfyUI GPU jobs at the same time.
- An empty ComfyUI queue is not enough: an idle ComfyUI session can retain ~11.5 GB VRAM. Check `GET /system_stats` before a TTS GPU job; if free VRAM is low, call `POST /free` with `{"unload_models":true,"free_memory":true}` and verify free VRAM before proceeding.
- `.env` files are gitignored and not auto-loaded; set variables in the PowerShell session. Relevant overrides: `TTS_SUBPROCESS_TIMEOUT_S`, `MEDIA_FFMPEG_BIN`, `MEDIA_FFPROBE_BIN`, `MEDIA_FFMPEG_TIMEOUT_S`, `MEDIA_FFPROBE_TIMEOUT_S`.
- FFmpeg renders must go to a temporary file and be validated with ffprobe before promotion. Never overwrite finished media without explicit authorization.

## Next action

Milestone 2C is complete. Wait for user confirmation before starting the next milestone. Candidate work includes MCP tools for the scene services, final video assembly, audio/subtitle integration and real reference-image conditioning (which requires model/custom-node approval and a new verified workflow), without changing the three existing MCP servers. Do not start final rendering, subtitles or dependency installs without that confirmation. The Chatterbox model weights are cached; no further Hugging Face download is needed.
