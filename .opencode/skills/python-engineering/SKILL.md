---
name: python-engineering
description: Use when writing or refactoring Python code, adding type hints, validating inputs, or running lint/type checks in this project.
---

# Python Engineering

Current layout is flat root scripts (`server.py`, `comfyui_client.py`) executed by `.venv`. Shared conventions live in `AGENTS.md`; this skill covers Python-specific practice.

## Workflow
1. Change one module or function; keep the public interface stable.
2. Check the changed file: `& .\.venv\Scripts\python.exe -m ruff check <file>`
3. Type-check: `& .\.venv\Scripts\python.exe -m mypy <file>`
4. Run the targeted tests, then report only the result.

## Rules
- Annotate all public functions and their returns; prefer `dict[str, object]` over untyped `dict`.
- Use `pathlib.Path`, `expanduser().resolve()`, and `is_file()` before I/O.
- Read config with `os.getenv(name, default)`; never hardcode URLs, tokens, or user paths in new code.
- Keep pure logic (workflow patching, path validation, metadata mapping) in functions that can be tested without ComfyUI or a GPU.
- Raise `FileNotFoundError` / `ValueError` / `httpx.HTTPStatusError` with context; do not swallow exceptions or return ambiguous `None`.
- Keep module imports side-effect free; `logging.getLogger(__name__)` instead of `print()`.
- New dependencies must be free/local where possible and pinned in `requirements-dev.txt` (dev) or a dedicated manifest (runtime).

## Example
```python
from pathlib import Path

def resolve_image(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    return resolved
```

Do not rename existing functions or change the tested workflow node IDs without checking callers.
