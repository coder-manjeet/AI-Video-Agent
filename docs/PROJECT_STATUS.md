# Project Status

Last updated: 2026-11-18

## Current milestone

Milestone 7 — Local FFmpeg Media Service, Phase 1 — is complete and verified end to end.

`media_service.py` provides reusable `probe_media`, `merge_narration` and `validate_media` operations using FFmpeg/ffprobe. A third MCP server, `local-media-agent`, exposes them over stdio. The original `local-video-agent` and `local-tts-agent` servers, their business logic, and the tested LTX workflow are unchanged. All three servers are connected simultaneously under `.venv`.

A live render combined the existing `ltxv_00006.mp4` clip with the verified English narration WAV into a new, validated MP4 using `extend_video` so no narration was cut off.

## Completed work

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
| `python -m mypy server.py comfyui_client.py tts_service.py tts_server.py tts_worker.py media_service.py media_server.py` | Passed |
| `python -m pytest -q` | 63 passed |
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
- The Git repository exists locally with no commits or remote.

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

Continue with the scene-based pipeline (`scenes.json`) and, when needed, add music, SFX and subtitles on top of the validated media service. Do not start multi-scene rendering or subtitle work as part of this milestone. The Chatterbox model weights are cached; no further Hugging Face download is needed.
