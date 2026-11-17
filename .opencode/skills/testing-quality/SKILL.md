---
name: testing-quality
description: Use when adding or running pytest tests, mocking HTTP/ComfyUI calls, validating results, or debugging a failing check.
---

# Testing and Quality

Dev tools live in `.venv` and are pinned in `requirements-dev.txt`. The default test suite must not need a GPU or a running ComfyUI.

## Commands
```powershell
& .\.venv\Scripts\python.exe -m pytest -q tests/unit
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m mypy server.py comfyui_client.py
```

## Test layers
- **Unit** (`tests/unit/`): workflow patching, path validation, metadata mapping, FFmpeg command builders. Use `httpx.MockTransport` or monkeypatch clients; `tmp_path` for files.
- **Integration** (`tests/integration/`): real `/system_stats` or short status polls; skip automatically when ComfyUI is unreachable. Mark with `@pytest.mark.integration`.
- **Manual smoke**: short TTS sentence, tiny ComfyUI render, final MP4 validation. Never part of the default suite.

## Rules
- Name files `test_*.py` and tests `test_*`. Async tests need `@pytest.mark.asyncio` explicitly.
- Assert both success and failure paths (missing file, HTTP error, pending job, missing output).
- No test writes to `output/`, `projects/`, or model caches; no network beyond localhost.
- A regression fix must add a failing-then-passing test. Run only relevant tests while iterating; run the full checks before finishing.

## Example
```python
@pytest.mark.asyncio
async def test_missing_image(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await upload_image(str(tmp_path / "missing.png"))
```
