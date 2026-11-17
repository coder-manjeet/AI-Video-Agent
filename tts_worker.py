"""Subprocess entry point for TTS generation in the ``.venv-tts`` environment.

The FastMCP server runs under ``.venv`` (which owns the MCP SDK). Torch and
Chatterbox stay in ``.venv-tts``: this worker is started with that
interpreter, reads a JSON request file, calls ``TTSService.generate()`` and
writes a JSON response file. Using files instead of stdout keeps the result
unambiguous even when model libraries print to stdout.

Usage: ``python tts_worker.py REQUEST_JSON RESPONSE_JSON``
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tts_service import TTSService


def _load_request(path: Path) -> dict[str, Any]:
    """Read and validate the JSON request object."""
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("TTS worker request must be a JSON object")
    return payload


def _write_response(path: Path, payload: dict[str, Any]) -> None:
    """Write the JSON response with UTF-8 encoding."""
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one generation request; return a process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        sys.stderr.write("usage: tts_worker.py REQUEST_JSON RESPONSE_JSON\n")
        return 2

    request_path = Path(args[0])
    response_path = Path(args[1])
    try:
        payload = _load_request(request_path)
        metadata = TTSService().generate(
            text=payload["text"],
            language=payload["language"],
            out_path=payload["out_path"],
            reference_wav=payload.get("reference_wav"),
            expressiveness=payload.get("expressiveness"),
        )
    except Exception as exc:  # noqa: BLE001 - worker must report every failure as JSON
        _write_response(
            response_path,
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
        )
        return 1

    _write_response(response_path, {"ok": True, "metadata": metadata})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
