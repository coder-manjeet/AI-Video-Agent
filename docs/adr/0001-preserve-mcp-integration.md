# ADR 0001 — Preserve the working MCP integration and environment split

- **Status:** Accepted
- **Date:** 2026-11-17

## Context
`server.py` + `comfyui_client.py` already provide three working FastMCP tools registered through `opencode.json`. The LTX workflow has been validated against local ComfyUI. `.venv` holds the MCP stack; `.venv-tts` holds a fragile CUDA 13.0 / torch 2.10.0 stack that must not be downgraded.

## Decision
Keep the root scripts and `opencode.json` unchanged for now. Do not migrate to `src/`, add MCP servers, or change the tested workflow as part of developer-tooling work. Install dev tools (ruff, pytest, pytest-asyncio, mypy) only in `.venv` and pin them in `requirements-dev.txt`.

## Consequences
- Existing tool names and behaviour stay stable; new features are added as small, tested layers.
- `.venv-tts` remains reproducible for future Chatterbox work.
- The `src/` restructure is deferred until it solves a concrete problem and can be verified against all three tools.
