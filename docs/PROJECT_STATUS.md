# Project Status

Last updated: 2026-11-17

## Current milestone

Milestone 6 — TTS core service and TTS MCP integration — is complete and verified end to end.

`tts_service.py` wraps the official Chatterbox API. A second MCP server, `local-tts-agent`, exposes `generate_speech` and `get_audio_metadata`; the original `local-video-agent` and its three ComfyUI tools are unchanged. English is the production target. English and Hindi GPU smoke tests both technically passed; Hindi output quality still requires evaluation.

## Completed work

- Audited the existing project, Git state, MCP integration, and LTX workflow.
- Created `.gitignore` and `README.md`.
- Verified ignore rules in a temporary Git repository outside the project.
- Confirmed ComfyUI MCP status is working.
- Checked official Chatterbox documentation and dependency metadata.
- Proved that normal Chatterbox installation would downgrade Torch/TorchAudio.
- Recorded decisions in:
  - [ADR 0001](adr/0001-preserve-mcp-integration.md)
  - [ADR 0002](adr/0002-chatterbox-dependency-strategy.md)
  - [ADR 0003](adr/0003-windows-tts-dependency-pins.md)
  - [ADR 0004](adr/0004-tts-mcp-subprocess-isolation.md)
- Pinned dev tools in `requirements-dev.txt`.
- Installed `chatterbox-tts==0.1.7` with `--no-deps` and explicit compatible dependencies; Gradio was not installed.
- Restored `numpy==1.26.4`, pinned `ml_dtypes==0.5.4`, and downgraded `scikit-learn` to `1.5.2` after a Windows Application Control DLL block.
- Downloaded the public Chatterbox weights once into the Hugging Face cache (~3.2 GB).
- Loaded `ChatterboxMultilingualTTS.from_pretrained(torch.device("cuda"))` successfully.
- Generated one short English WAV (`language_id="en"`) and one short Hindi WAV (`language_id="hi"`) sequentially.
- Freed ComfyUI VRAM with `POST /free` before the successful retry; ComfyUI remained connected and reloads models on the next render.
- Released the TTS model and cleared the CUDA cache after generation.
- Implemented `tts_service.py`: lazy model loading, language routing (`en`, `hi`, experimental `hinglish`), optional licensed reference WAV, expressiveness mapping, VRAM check/free, 24 kHz WAV output, and model-free metadata extraction.
- Added 31 passing unit tests covering service logic, WAV metadata, tool boundaries, and the worker JSON protocol. No GPU or ComfyUI is required by the default suite.
- Implemented `tts_server.py` (`local-tts-agent`) with `generate_speech` and `get_audio_metadata`, and `tts_worker.py` to keep Torch inside `.venv-tts`.
- Registered `local-tts-agent` in `opencode.json` without changing `local-video-agent`.
- Fixed a Windows MCP integration defect: a child Python inheriting the MCP server's stdio pipe hangs, so the TTS worker now uses `stdin=DEVNULL` and file-based JSON/stdout exchange.
- Added explicit output-path validation: `.wav` extension, existing files refused unless `overwrite=True`.

## Verification results

| Check | Result |
|---|---|
| `import chatterbox.tts` / `import chatterbox.mtl_tts` | OK |
| `torch` / `torchaudio` | `2.10.0+cu130` |
| `numpy` | `1.26.4` |
| CUDA available | `True` |
| GPU | RTX 5060 Ti, `sm_120` |
| CUDA kernel smoke test | Passed |
| `en` / `hi` language codes | Present |
| English GPU inference | Passed — 24 kHz, 3.52 s, peak `0.921326`, RMS `0.106031`, non-silent |
| Hindi GPU inference | Passed — 24 kHz, 4.52 s, peak `0.870392`, RMS `0.113988`, non-silent |
| Hindi output quality | **Not yet evaluated** — technically generated, pronunciation/naturalness unassessed |
| WAV files re-read with `soundfile` | Both valid, matching generated metadata |
| TTS process VRAM after cleanup | `9.12 MiB` allocated; `cuda_after=True` |
| `pip check` | Only expected Chatterbox metadata conflicts |
| `python -m ruff check .` | Passed |
| `python -m mypy server.py comfyui_client.py tts_service.py tts_server.py tts_worker.py` | Passed |
| `python -m pytest -q` | 31 passed |
| `opencode mcp list` | Both `local-video-agent` and `local-tts-agent` connected |
| MCP `list_tools` | Video: `comfyui_status`, `create_video`, `check_video_status`; TTS: `generate_speech`, `get_audio_metadata` |
| MCP `comfyui_status` (original tool) | `connected`, ComfyUI `0.45.3` |
| MCP `get_audio_metadata` before any generation | Returns metadata; no model or GPU needed |
| MCP `generate_speech` (English, live) | Generated `test-output/mcp_english_verified.wav` |
| MCP `get_audio_metadata` after generation | Matches generation metadata exactly |
| Duplicate output path without `overwrite` | Rejected with an error |
| Model load in metadata path | Not called (unit test patches the loader to fail; `.venv` has no Torch) |

English WAV generated through the actual MCP tool:

```text
C:\AI-Video-Agent\test-output\mcp_english_verified.wav
24 kHz / 4.24 s / 101,760 samples / peak 0.746307 / RMS 0.104230
voice: chatterbox-default-narrator
```

Expected `pip check` conflicts:

```text
chatterbox-tts 0.1.7 requires gradio, which is not installed.
chatterbox-tts 0.1.7 requires torch==2.6.0, but you have torch 2.10.0+cu130.
chatterbox-tts 0.1.7 requires torchaudio==2.6.0, but you have torchaudio 2.10.0+cu130.
```

These are intentional. Do not resolve them by downgrading Torch or installing Gradio.

## Blockers and open decisions

- Hindi output quality still needs human evaluation; the smoke test only proves the pipeline runs.
- Hinglish has no official Chatterbox code; its script heuristic routes to `hi` or `en` but remains untested.
- No FFmpeg integration has been started; this milestone stops at TTS and metadata.
- The Git repository exists locally with no commits or remote.

## Environment rules

- `.venv` is for MCP and dev tooling; it runs both MCP servers and has no Torch.
- `.venv-tts` is the only TTS environment and is used by `tts_worker.py`.
- Never mix or recreate either environment without approval.
- Never upgrade, downgrade, or replace Torch, TorchAudio, or CUDA in `.venv-tts`.
- Never run TTS and ComfyUI GPU jobs at the same time.
- An empty ComfyUI queue is not enough: an idle ComfyUI session can retain ~11.5 GB VRAM. Check `GET /system_stats` before a TTS GPU job; if free VRAM is low, call `POST /free` with `{"unload_models":true,"free_memory":true}` and verify free VRAM before proceeding. ComfyUI reloads its models on the next render.
- `.env` files are gitignored and not auto-loaded; set variables in the PowerShell session. `TTS_SUBPROCESS_TIMEOUT_S` can override the worker timeout (default 300 s).

## Next action

Evaluate Hindi and Hinglish output quality, then continue with the scene-based pipeline. Do not begin FFmpeg integration as part of this milestone. The model weights are cached; no further Hugging Face download is needed for subsequent runs.
