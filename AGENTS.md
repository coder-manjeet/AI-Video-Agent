# AGENTS.md

Local Video Agent — local-only AI video pipeline. Windows 11, PowerShell only.

## Architecture
- `server.py` — FastMCP stdio tools; `comfyui_client.py` — ComfyUI HTTP client and workflow patching; `ltx_image_to_video_api.json` — tested LTX workflow.
- `.venv` = MCP + dev tooling; `.venv-tts` = CUDA/Torch/TTS. Never mix or recreate either without approval.
- Planned `src/local_video_agent/` migration is deferred; keep business logic out of `@mcp.tool()` bodies.
- Scene-based pipeline with `scenes.json`; never overwrite finished media without permission.

## Commands (PowerShell only)
```powershell
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m mypy server.py comfyui_client.py
& .\.venv\Scripts\python.exe -m pytest -q
```
No bash, no POSIX paths in committed commands.

## Environment and config
- Configure via environment variables (`COMFYUI_URL`). `.env` files are gitignored and **not** auto-loaded; set variables in the PowerShell session.
- Never commit secrets, tokens, personal voice recordings, model weights, or generated media.

## Engineering rules
- Type hints on public functions; `pathlib.Path`; validate paths/IDs before I/O; raise explicit exceptions with context.
- Module-level `logging.getLogger(__name__)` for diagnostics; no `print()` in library code; never log secrets or full payloads.
- Network calls need timeouts and `raise_for_status()`; MCP tools return JSON-serializable dicts.
- Pin dev dependencies in `requirements-dev.txt`; never upgrade Torch/TorchAudio/CUDA in `.venv-tts`.

## Testing
- Unit-test pure logic with mocks; integration tests skip when ComfyUI is unreachable; GPU/TTS smoke tests are manual.
- Run ruff, mypy, and targeted pytest before declaring work done. Tests must not write to `output/`.

## GPU memory
- 16 GB VRAM is shared by ComfyUI and TTS: one GPU job at a time; release models between stages; check occupancy before batching.

## Git
- Not yet a repository; no remote. Init, commits, and remotes only when requested. Conventional commits; small diffs; review `git status --short` and `git diff --stat` first.

## Incremental and token-efficient work
- Change one layer at a time; keep the three existing MCP tools working.
- Inspect only relevant files; do not re-read unchanged files; prefer targeted tests and concise output.
- Load skills on demand; keep `docs/PROJECT_STATUS.md` current; record notable decisions in `docs/adr/`.
