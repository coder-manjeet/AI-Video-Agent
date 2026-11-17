# ADR 0004 — TTS MCP server with subprocess isolation

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

The TTS service is implemented in `tts_service.py` and needs two MCP tools. `.venv` owns `mcp==1.30.0` but has no Torch/Chatterbox; `.venv-tts` owns the working CUDA 13.0 / Torch 2.10.0 / Chatterbox stack but has no MCP SDK. The milestone requires using the existing `.venv-tts` environment, following the existing MCP SDK version, and not reinstalling or upgrading packages.

## Decision

Register a second MCP server, `local-tts-agent`, in `opencode.json` while leaving `local-video-agent` unchanged.

- `tts_server.py` runs under `.venv` with FastMCP 1.30.0 and exposes `generate_speech` and `get_audio_metadata`.
- `generate_speech` validates its inputs, then calls `tts_worker.py` through a subprocess started with `.venv-tts\Scripts\python.exe`. The worker calls `TTSService.generate()` and exchanges a JSON request/response via temporary files, so model-library output cannot corrupt the MCP stream.
- `get_audio_metadata` calls `TTSService.get_metadata()` directly in the MCP process. That path needs only audio I/O, never constructs a provider, and cannot load a model (Torch is not installed in `.venv`).
- Existing WAV files are never replaced unless the caller explicitly passes `overwrite=True`.

No packages were installed, reinstalled, or upgraded in either environment.

## Consequences

- Torch and Chatterbox stay confined to `.venv-tts`; the original ComfyUI MCP tools and LTX workflow are untouched.
- Each generation starts a fresh TTS worker, so the model loads once per request rather than once per MCP server process. This is acceptable for the current single-narration workflow.
- The worker has a configurable timeout (`TTS_SUBPROCESS_TIMEOUT_S`, default 300 s) and reports failures as structured JSON.
- The worker gets `stdin=subprocess.DEVNULL` and writes stdout/stderr to a log file. On Windows a child Python that inherits the MCP server's stdio pipe hangs before it can run; file-based exchange plus detached stdin avoids both that hang and noisy progress output on the MCP stream.
- `docs/PROJECT_STATUS.md` records which parts were verified live; GPU inference remains a manual smoke test outside the default pytest suite.
