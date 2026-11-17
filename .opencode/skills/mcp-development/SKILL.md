---
name: mcp-development
description: Use when adding, changing, or debugging FastMCP tools in server.py, or when reviewing the OpenCode MCP integration.
---

# MCP Development

`server.py` (`local-video-agent`) and `tts_server.py` (`local-tts-agent`) define FastMCP tools over stdio; `opencode.json` registers both. The video server's three tools (`comfyui_status`, `create_video`, `check_video_status`) are working — extend, do not replace them. The TTS server exposes `generate_speech` and `get_audio_metadata`; generation is delegated to `tts_worker.py` under `.venv-tts` so Torch never loads into an MCP process.

## Tool design
- One tool = one action. Async functions, typed arguments, typed `dict` result, JSON-serializable values only.
- The docstring is the tool description: state what it does and what it returns (e.g. `prompt_id`, `status`).
- Validate inputs at the boundary before any network call; reject empty paths/IDs and unknown files early.
- Return explicit states (`submitted`, `pending`, `running`, `completed`, `error`) rather than bare `None`.
- Delegate business logic to `comfyui_client.py` or future services; keep decorators thin.

## Errors and transport
- Always set httpx timeouts and call `raise_for_status()`; wrap low-level errors with actionable messages.
- Never log tokens, credentials, or full request payloads. Avoid unbounded polling loops.
- Keep `COMFYUI_URL` overridable via env var; do not hardcode a second base URL in new tools.
- Do not edit `opencode.json` or rename tools without explicit approval — clients depend on those names.

## Testing
- Test the underlying async functions with `httpx.MockTransport` or monkeypatched clients (see the `testing-quality` skill).
- Verify a live tool only when ComfyUI responds to `/system_stats`; use cheap status calls, not a full video job.

## Example skeleton
```python
@mcp.tool()
async def new_tool(value: str) -> dict[str, object]:
    """Describe the action and the returned fields."""
    if not value.strip():
        raise ValueError("value must not be empty")
    return {"status": "ok", "value": value.strip()}
```
