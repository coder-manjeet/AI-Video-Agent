---
name: comfyui-workflows
description: Use when working with the ComfyUI HTTP API, the LTX workflow JSON, node patching, or video job status.
---

# ComfyUI Workflows

Base URL: `COMFYUI_URL` (default `http://127.0.0.1:8000`). `ltx_image_to_video_api.json` is a tested **API-format** prompt: keys are node IDs, values contain `inputs` and `class_type`. Do not replace it.

## Endpoints
- `GET /system_stats` — health/version/devices.
- `POST /upload/image` — multipart upload, returns `name`.
- `POST /prompt` — body `{"prompt": workflow}`, returns `prompt_id`.
- `GET /history/{prompt_id}` — job result/outputs.

## Patching a workflow safely
1. Load a fresh copy of the JSON for each job; never mutate the saved file.
2. Confirm the node exists and the input key is present before writing.
3. Set node `78.inputs.image` to the uploaded filename; node `6.inputs.text` to the positive prompt.
4. Keep all other nodes, links, and `_meta` untouched (model: `ltxv-2b-0.9.6-distilled-04-25.safetensors`, 768x512, 97 frames, 24 fps).

## Job lifecycle
`prompt_id` → poll `/history` → `pending`/`running` until `status.completed` → read `outputs.*.gifs[0]` for filename, path, and metadata. Use short timeouts; do not busy-poll without a delay in real code.

## Verification and safety
```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/system_stats' -TimeoutSec 5
```
One GPU job at a time (16 GB VRAM shared with TTS). Check VRAM before batching; do not auto-retry failed render jobs — inspect the error first. Never overwrite existing output video files without permission.
