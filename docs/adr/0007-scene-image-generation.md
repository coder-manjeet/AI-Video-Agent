# ADR 0007 — Scene-Based Image Generation Service

- Status: Accepted
- Date: 2026-11-20

## Context

Milestone 2A required independent AI image generation for every scene, built on
the existing scene/project foundation and ComfyUI, without touching video
generation, final rendering, the three MCP servers, TTS, FFmpeg or the tested
LTX workflow. No verified API-format image-generation workflow existed in the
repository: the only repo workflow is LTX image-to-video, and the local UI image
workflows are UI-format and disagree on the text-encoder type (`sd3` versus the
correct `lumina2`). The Z-Image Turbo models were already installed.

## Decision

- Author a new, standalone API-format workflow
  (`z_image_turbo_text_to_image_api.json`) from the official ComfyUI Z-Image
  Turbo template, using `CLIPLoader` type `lumina2` with
  `qwen_3_4b.safetensors`, `z_image_turbo_bf16.safetensors` and `ae.safetensors`.
  Never modify `ltx_image_to_video_api.json`.
- Validate the workflow against live ComfyUI `/object_info` and run exactly one
  small controlled GPU job before writing any service code. Proceed only after
  the image is downloaded and verified.
- Add generic helpers to `comfyui_client.py` only; leave every existing function
  body unchanged. No new runtime dependencies.
- Keep node IDs, expected classes, model values, capabilities, defaults and
  limits in a declarative registry (`image_workflows.json`). `workflow_adapter.py`
  performs offline compatibility checks, safe patching of a fresh workflow copy
  and history output extraction; node IDs are not hardcoded in the service.
- `scene_image_service.py` owns generation only. It injects the ComfyUI client,
  serializes all jobs with an `asyncio.Lock`, keeps an in-memory job registry,
  retries submission failures, polls history, handles timeouts and verifies the
  PNG signature and IHDR dimensions before touching any manifest.
- Write `script/scenes.json` exactly once per successful scene, setting only
  `image_path` and `status="image_generated"`. Failures never modify the
  manifest. Filenames are unique and versioned (`scene-001-opening.png`, then
  `-v2`, `-v3`, …); existing image files are never overwritten automatically.
- Treat negative prompts and consistency inputs (character references, visual
  style, color palette, character appearance) as metadata with explicit warnings
  when the workflow cannot honour them. Do not claim character consistency.

## Consequences

- A future editor can regenerate one scene without touching any other scene,
  and can swap workflows by editing the registry rather than the service.
- All generation logic is unit-testable with a fake ComfyUI client; no GPU or
  network is needed for the test suite (142 previous + 25 new tests pass).
- The Z-Image Turbo workflow cannot apply negative prompts or reference-image
  conditioning, so callers must compose such intent into the positive prompt or
  use a different verified workflow later.
- ComfyUI retains loaded models between jobs; sequential submission protects
  the shared 16 GB VRAM but does not unload models. No MCP tools or UI were added
  in this milestone.
