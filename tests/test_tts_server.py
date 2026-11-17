"""Unit tests for the two TTS MCP tools and their worker protocol.

No GPU, model weights, or real subprocess is used: the worker call is
monkeypatched so the tool boundary and payload can be asserted directly.
"""

from __future__ import annotations

import json
import math
import subprocess
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tts_server
import tts_service
import tts_worker
from tts_service import DEFAULT_VOICE, SAMPLE_RATE


def _write_sine_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(SAMPLE_RATE * 0.05)
    step = 2.0 * math.pi * 440.0 / SAMPLE_RATE
    pcm = array("h", (int(12000 * math.sin(step * i)) for i in range(frames)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def _fake_metadata(out_path: str, language: str) -> dict[str, object]:
    return {
        "path": out_path,
        "language": language,
        "duration_s": 3.5,
        "sample_rate": SAMPLE_RATE,
        "samples": 84_000,
        "peak": 0.9,
        "rms": 0.1,
        "voice": DEFAULT_VOICE,
        "reference_used": False,
        "model_load_s": 0.0,
        "generation_s": 1.0,
    }


def test_server_name_and_tools() -> None:
    assert tts_server.mcp.name == "local-tts-agent"


@pytest.mark.asyncio
async def test_registered_tool_names() -> None:
    tools = await tts_server.mcp.list_tools()
    assert {tool.name for tool in tools} == {
        "generate_speech",
        "get_audio_metadata",
    }


@pytest.mark.asyncio
async def test_generate_speech_passes_validated_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_worker(payload: dict[str, Any]) -> dict[str, object]:
        captured.update(payload)
        return _fake_metadata(str(payload["out_path"]), str(payload["language"]))

    monkeypatch.setattr(tts_server, "_run_tts_worker", fake_worker)
    out_path = tmp_path / "nested" / "speech.wav"

    result = await tts_server.generate_speech(
        "Hello from OpenCode.",
        str(out_path),
        language="en",
        expressiveness=0.7,
    )

    assert result["status"] == "generated"
    assert result["path"] == str(out_path.resolve())
    assert captured["text"] == "Hello from OpenCode."
    assert captured["language"] == "en"
    assert captured["out_path"] == str(out_path.resolve())
    assert captured["expressiveness"] == 0.7
    assert captured["reference_wav"] is None


@pytest.mark.asyncio
async def test_generate_speech_refuses_existing_wav(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_worker(_payload: dict[str, Any]) -> dict[str, object]:
        pytest.fail("worker must not run when the output already exists")

    monkeypatch.setattr(tts_server, "_run_tts_worker", unexpected_worker)
    existing = tmp_path / "approved.wav"
    existing.write_bytes(b"do not replace")

    with pytest.raises(FileExistsError, match="already exists"):
        await tts_server.generate_speech("Text.", str(existing))


@pytest.mark.asyncio
async def test_generate_speech_overwrite_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tts_server,
        "_run_tts_worker",
        lambda payload: _fake_metadata(str(payload["out_path"]), "en"),
    )
    existing = tmp_path / "approved.wav"
    existing.write_bytes(b"old")

    result = await tts_server.generate_speech("Text.", str(existing), overwrite=True)

    assert result["status"] == "generated"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        ({"text": "  "}, "non-empty"),
        ({"output_path": "speech.mp3"}, "must end in .wav"),
        ({"language": "de"}, "Unsupported language"),
        ({"voice": "other-voice"}, "Unsupported voice"),
        ({"expressiveness": -0.1}, "non-negative"),
        ({"expressiveness": math.nan}, "finite"),
        ({"reference_wav": "missing.wav"}, "Reference WAV not found"),
    ],
)
async def test_generate_speech_rejects_invalid_input(
    kwargs: dict[str, Any], expected_error: str, tmp_path: Path
) -> None:
    defaults: dict[str, Any] = {
        "text": "Hello.",
        "output_path": str(tmp_path / "speech.wav"),
    }
    defaults.update(kwargs)

    with pytest.raises((ValueError, FileNotFoundError), match=expected_error):
        await tts_server.generate_speech(**defaults)


@pytest.mark.asyncio
async def test_generate_speech_accepts_reference_voice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    reference = tmp_path / "licensed.wav"
    _write_sine_wav(reference)

    def fake_worker(payload: dict[str, Any]) -> dict[str, object]:
        captured.update(payload)
        return _fake_metadata(str(payload["out_path"]), "en")

    monkeypatch.setattr(tts_server, "_run_tts_worker", fake_worker)

    await tts_server.generate_speech(
        "Cloned voice.",
        str(tmp_path / "cloned.wav"),
        reference_wav=str(reference),
    )

    assert captured["reference_wav"] == str(reference.resolve())


@pytest.mark.asyncio
async def test_get_audio_metadata_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wav_path = tmp_path / "speech.wav"
    _write_sine_wav(wav_path)

    def forbidden_load() -> None:
        pytest.fail("get_audio_metadata must not load the TTS model")

    monkeypatch.setattr(tts_service, "_MODEL", None)
    monkeypatch.setattr(tts_service, "_load_chatterbox_model", forbidden_load)

    result = await tts_server.get_audio_metadata(str(wav_path))

    assert result["status"] == "ok"
    assert result["sample_rate"] == SAMPLE_RATE
    assert result["path"] == str(wav_path.resolve())
    assert tts_service._MODEL is None


@pytest.mark.asyncio
async def test_get_audio_metadata_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="WAV file not found"):
        await tts_server.get_audio_metadata(str(tmp_path / "missing.wav"))


def test_worker_writes_success_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    class FakeService:
        def generate(self, **kwargs: Any) -> dict[str, object]:
            captured.update(kwargs)
            return {"path": "out.wav", "samples": 10}

    monkeypatch.setattr(tts_worker, "TTSService", FakeService)
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "text": "Worker test.",
                "language": "en",
                "out_path": "out.wav",
                "reference_wav": None,
                "expressiveness": 0.4,
            }
        ),
        encoding="utf-8",
    )

    assert tts_worker.main([str(request), str(response)]) == 0
    assert json.loads(response.read_text(encoding="utf-8")) == {
        "ok": True,
        "metadata": {"path": "out.wav", "samples": 10},
    }
    assert captured["language"] == "en"
    assert captured["expressiveness"] == 0.4


def test_worker_writes_error_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingService:
        def generate(self, **_kwargs: Any) -> dict[str, object]:
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(tts_worker, "TTSService", FailingService)
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "text": "Failure test.",
                "language": "en",
                "out_path": "out.wav",
                "reference_wav": None,
                "expressiveness": None,
            }
        ),
        encoding="utf-8",
    )

    assert tts_worker.main([str(request), str(response)]) == 1
    payload = json.loads(response.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert "model unavailable" in payload["error"]


def test_run_worker_detaches_stdin_and_reports_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_kwargs: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        seen_kwargs.update(kwargs)
        Path(command[-1]).write_text(
            json.dumps({"ok": False, "error": "ValueError: bad text"}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(tts_server.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="bad text"):
        tts_server._run_tts_worker({"text": "x"})

    assert seen_kwargs["stdin"] == subprocess.DEVNULL


def test_run_worker_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(cmd=["tts_worker.py"], timeout=300)

    monkeypatch.setattr(tts_server.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out after 300s"):
        tts_server._run_tts_worker({"text": "x"})
