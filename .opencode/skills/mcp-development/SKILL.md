---
name: mcp-development
description: Use when adding, changing, or debugging FastMCP tools in server.py, or when reviewing the OpenCode MCP integration.
---

# MCP Development

`server.py` (`local-video-agent`), `tts_server.py` (`local-tts-agent`) and `media_server.py` (`local-media-agent`) define FastMCP tools over stdio; `opencode.json` registers all three. The video server's three tools (`comfyui_status`, `create_video`, `check_video_status`) are working — extend, do not replace them. The TTS server exposes `generate_speech` and `get_audio_metadata`; generation is delegated to `tts_worker.py` under `.venv-tts` so Torch never loads into an MCP process. The media server exposes `probe_media`, `merge_narration` and `validate_media`; all FFmpeg logic lives in `media_service.py` under `.venv`.

## Tool design
- One tool = one action. Async functions, typed arguments, typed `dict` result, JSON-serializable values only.
- The docstring is the tool description: state what it does and what it returns (e.g. `prompt_id`, `status`).
- Validate inputs at the boundary before any network call; reject empty paths/IDs and unknown files early.
- Return explicit states (`submitted`, `pending`, `running`, `completed`, `error`) rather than bare `None`.
- Delegate business logic to `comfyui_client.py`, `tts_service.py` or `media_service.py`; keep decorators thin. Move blocking subprocess calls to `asyncio.to_thread`.
- Video/TTS tools raise on bad input; media tools instead return `{"status": "error", "error": "..."}` for expected failures. Follow the convention of the server you are editing.

## Errors and transport
- Always set httpx timeouts and call `raise_for_status()`; wrap low-level errors with actionable messages.
- Never log tokens, credentials, or full request payloads. Avoid unbounded polling loops.
- Keep `COMFYUI_URL` overridable via env var; do not hardcode a second base URL in new tools.
- Do not edit `opencode.json` or rename tools without explicit approval — clients depend on those names. Back up `opencode.json` before an approved config change, then verify with `opencode mcp list`.

## Testing
- Test the underlying async functions with `httpx.MockTransport` or monkeypatched clients (see the `testing-quality` skill). Media tool tests monkeypatch `media_service` functions; service tests use tiny generated fixtures instead of AI models.
- Verify a live tool only when ComfyUI responds to `/system_stats`; use cheap status calls, not a full video job. A small real FFmpeg render is acceptable for the media server because it needs no GPU or model.

## Example skeleton
```python
@mcp.tool()
async def new_tool(value: str) -> dict[str, object]:
    """Describe the action and the returned fields."""
    if not value.strip():
        raise ValueError("value must not be empty")
    return {"status": "ok", "value": value.strip()}
```
