"""FastMCP server ``local-tts-agent`` for local Chatterbox TTS.

This process runs under ``.venv`` (which owns ``mcp==1.30.0``). Speech
generation is delegated to ``tts_worker.py`` under ``.venv-tts`` so Torch and
Chatterbox never load into the MCP server. Metadata reads call
``TTSService.get_metadata()`` directly: that path uses audio I/O only and
never constructs a provider or loads a model.

English (``en``) is the production target; ``hi`` and ``hinglish`` remain
experimental. Business logic lives in module-level helpers so the tool
decorators stay thin.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

import tts_service
from tts_service import DEFAULT_VOICE, TTSService

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
TTS_PYTHON = PROJECT_ROOT / ".venv-tts" / "Scripts" / "python.exe"
TTS_WORKER = PROJECT_ROOT / "tts_worker.py"
SUBPROCESS_TIMEOUT_S = float(os.getenv("TTS_SUBPROCESS_TIMEOUT_S", "300"))

mcp = FastMCP("local-tts-agent")


def _resolve_wav_path(value: str, label: str) -> Path:
    """Resolve a non-empty ``.wav`` path without touching the filesystem."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() != ".wav":
        raise ValueError(f"{label} must end in .wav: {path}")
    return path


def _validate_output_path(output_path: str, overwrite: bool) -> Path:
    """Validate the destination and refuse to replace audio without consent."""
    path = _resolve_wav_path(output_path, "output_path")
    if path.is_dir():
        raise ValueError(f"output_path must be a file, not a directory: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output WAV already exists: {path}. "
            "Pass overwrite=True to explicitly replace it."
        )
    return path


def _validate_voice(voice: str) -> None:
    """Accept only the voices the current service actually supports."""
    if not isinstance(voice, str) or not voice.strip():
        raise ValueError("voice must be a non-empty string")
    if voice.strip() != DEFAULT_VOICE:
        raise ValueError(
            f"Unsupported voice '{voice}'. Supported voices: {DEFAULT_VOICE}. "
            "For a licensed reference voice, pass reference_wav instead."
        )


def _validate_expressiveness(expressiveness: float | None) -> float | None:
    """Validate the Chatterbox ``exaggeration`` alias at the MCP boundary."""
    if expressiveness is None:
        return None
    value = float(expressiveness)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("expressiveness must be a finite, non-negative float")
    return value


def _log_tail(log_path: Path, limit: int = 1500) -> str:
    """Return the tail of a worker log for actionable error messages."""
    if not log_path.is_file():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace").strip()[-limit:]


def _run_tts_worker(payload: dict[str, Any]) -> dict[str, object]:
    """Run the TTS worker and return its metadata, or raise with context."""
    if not TTS_PYTHON.is_file():
        raise FileNotFoundError(f"TTS interpreter not found: {TTS_PYTHON}")
    if not TTS_WORKER.is_file():
        raise FileNotFoundError(f"TTS worker not found: {TTS_WORKER}")

    with tempfile.TemporaryDirectory(prefix="local-tts-agent-") as tmp:
        request = Path(tmp) / "request.json"
        response = Path(tmp) / "response.json"
        log = Path(tmp) / "worker.log"
        request.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        # Redirect child output to a file instead of pipes: model libraries
        # emit heavy progress output, and file handles cannot keep a parent
        # waiting for pipe EOF after the worker exits.
        with log.open("w", encoding="utf-8") as log_file:
            try:
                completed = subprocess.run(
                    [str(TTS_PYTHON), str(TTS_WORKER), str(request), str(response)],
                    cwd=str(PROJECT_ROOT),
                    # The MCP server owns stdin for JSON-RPC. A child Python
                    # that inherits that pipe hangs on Windows, so detach it.
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=SUBPROCESS_TIMEOUT_S,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                detail = _log_tail(log)
                LOGGER.error("TTS worker timed out after %ss", SUBPROCESS_TIMEOUT_S)
                raise RuntimeError(
                    f"TTS worker timed out after {SUBPROCESS_TIMEOUT_S:.0f}s; "
                    "check ComfyUI VRAM and the TTS environment"
                    + (f". Last output: {detail}" if detail else "")
                ) from exc
        if not response.is_file():
            detail = _log_tail(log)
            LOGGER.error(
                "TTS worker exited %s without a response file", completed.returncode
            )
            suffix = f". Last output: {detail}" if detail else ""
            raise RuntimeError(
                f"TTS worker produced no response (exit {completed.returncode}){suffix}"
            )
        try:
            result: object = json.loads(response.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("TTS worker returned invalid JSON") from exc

    if not isinstance(result, dict):
        raise TypeError("TTS worker response must be a JSON object")
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error") or "TTS generation failed"))
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("TTS worker response is missing metadata")
    return metadata


@mcp.tool()
async def generate_speech(
    text: str,
    output_path: str,
    language: str = "en",
    voice: str = DEFAULT_VOICE,
    reference_wav: str | None = None,
    expressiveness: float | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate narration as a 24 kHz WAV with local Chatterbox TTS.

    ``en`` is the production language; ``hi`` and ``hinglish`` are
    experimental. Supported voices are the default narrator and a licensed
    reference WAV supplied via ``reference_wav`` (voice cloning).
    ``expressiveness`` maps to Chatterbox ``exaggeration``. Existing WAVs are
    never replaced unless ``overwrite=True``. Returns a status plus the WAV
    metadata (path, duration_s, sample_rate, samples, peak, rms, language,
    voice, generation timings).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    destination = _validate_output_path(output_path, overwrite)
    _validate_voice(voice)
    effective_language = tts_service.resolve_language(language, text)
    expressive = _validate_expressiveness(expressiveness)

    reference: Path | None = None
    if reference_wav is not None:
        reference = _resolve_wav_path(reference_wav, "reference_wav")
        if not reference.is_file():
            raise FileNotFoundError(f"Reference WAV not found: {reference}")

    payload: dict[str, Any] = {
        "text": text,
        "language": effective_language,
        "out_path": str(destination),
        "reference_wav": str(reference) if reference is not None else None,
        "expressiveness": expressive,
    }
    metadata = await asyncio.to_thread(_run_tts_worker, payload)
    return {"status": "generated", **metadata}


@mcp.tool()
async def get_audio_metadata(wav_path: str) -> dict[str, object]:
    """Read WAV metadata without loading the TTS model or using the GPU.

    Returns the resolved path, duration_s, sample_rate, samples, peak and rms.
    """
    path = _resolve_wav_path(wav_path, "wav_path")
    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")
    service = TTSService()
    metadata = await asyncio.to_thread(service.get_metadata, path)
    return {"status": "ok", **metadata}


if __name__ == "__main__":
    mcp.run(transport="stdio")
