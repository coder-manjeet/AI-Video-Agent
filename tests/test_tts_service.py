"""TTS service unit tests with the model, filesystem side effects and HTTP
layer fully mocked. No GPU, network or weight download is involved.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

import tts_service
from tts_service import MIN_FREE_VRAM_GB, SAMPLE_RATE, TTSService

SAMPLE_COUNT = 2400  # 0.1 s at 24 kHz
EXPECTED_KEYS = {
    "path",
    "language",
    "duration_s",
    "sample_rate",
    "samples",
    "peak",
    "rms",
    "voice",
    "reference_used",
    "model_load_s",
    "generation_s",
}


class FakeProvider:
    """Provider double that records calls and returns plain PCM samples."""

    def __init__(self) -> None:
        self.name = "fake-voice"
        self.calls: list[dict[str, Any]] = []

    def synthesize(
        self,
        text: str,
        language: str,
        reference_wav: Path | None,
        expressiveness: float | None,
    ) -> list[float]:
        self.calls.append(
            {
                "text": text,
                "language": language,
                "reference_wav": reference_wav,
                "expressiveness": expressiveness,
            }
        )
        return [
            0.25 * math.sin(index / 10.0) for index in range(SAMPLE_COUNT)
        ]


class FakeModel:
    """Minimal Chatterbox-shaped model for the singleton loader test."""

    def generate(self, **_kwargs: object) -> list[float]:
        return [0.0] * SAMPLE_COUNT


class FakeResponse:
    """Minimal httpx.Response double."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload if payload is not None else {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _device(free_gb: float) -> dict[str, Any]:
    return {
        "name": "fake cuda:0",
        "type": "cuda",
        "vram_free": free_gb * 1024**3,
    }


@pytest.fixture()
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture()
def service(
    provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> TTSService:
    monkeypatch.setattr(
        TTSService,
        "_ensure_vram_available",
        lambda _self, min_free_gb=MIN_FREE_VRAM_GB: {
            "free_gb": 12.0,
            "freed": False,
        },
    )
    return TTSService(provider=provider)


def test_generate_returns_full_metadata_and_creates_parents(
    service: TTSService, provider: FakeProvider, tmp_path: Path
) -> None:
    out_path = tmp_path / "nested" / "deeper" / "speech.wav"

    metadata = service.generate("Hello there.", "en", out_path)

    assert set(metadata) == EXPECTED_KEYS
    assert out_path.parent.is_dir()
    assert metadata["path"] == str(out_path)
    assert metadata["language"] == "en"
    assert metadata["sample_rate"] == SAMPLE_RATE
    assert metadata["samples"] == SAMPLE_COUNT
    assert metadata["duration_s"] == pytest.approx(0.1, abs=1e-6)
    assert isinstance(metadata["peak"], float)
    assert isinstance(metadata["rms"], float)
    assert metadata["voice"] == "fake-voice"
    assert metadata["reference_used"] is False
    assert isinstance(metadata["model_load_s"], float)
    assert isinstance(metadata["generation_s"], float)
    assert provider.calls[0]["text"] == "Hello there."


def test_model_loaded_at_most_once_across_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader_calls: list[int] = []

    def fake_loader() -> tuple[object, float]:
        loader_calls.append(1)
        return FakeModel(), 0.01

    monkeypatch.setattr(tts_service, "_MODEL", None)
    monkeypatch.setattr(tts_service, "_MODEL_LOAD_S", 0.0)
    monkeypatch.setattr(tts_service, "_load_chatterbox_model", fake_loader)
    monkeypatch.setattr(
        TTSService,
        "_ensure_vram_available",
        lambda _self, min_free_gb=MIN_FREE_VRAM_GB: {"free_gb": 12.0},
    )

    service = TTSService()
    first = service.generate("First line.", "en", tmp_path / "first.wav")
    second = service.generate("दूसरी पंक्ति।", "hi", tmp_path / "second.wav")

    assert len(loader_calls) == 1
    assert first["model_load_s"] == pytest.approx(0.01)
    assert second["model_load_s"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("language", "text", "expected"),
    [
        ("en", "Hello world.", "en"),
        ("hi", "नमस्ते दुनिया।", "hi"),
        # Hinglish has no official code: Devanagari-heavy text routes to "hi",
        # Latin-only text routes to "en"; neither may crash.
        ("hinglish", "Namaste नमस्ते दोस्तों", "hi"),
        ("hinglish", "Hello dost, how are you?", "en"),
    ],
)
def test_language_routing(
    service: TTSService,
    provider: FakeProvider,
    tmp_path: Path,
    language: str,
    text: str,
    expected: str,
) -> None:
    metadata = service.generate(text, language, tmp_path / "speech.wav")

    assert provider.calls[-1]["language"] == expected
    assert metadata["language"] == language


def test_reference_wav_is_passed_to_provider(
    service: TTSService, provider: FakeProvider, tmp_path: Path
) -> None:
    reference = tmp_path / "authorized_reference.wav"
    reference.write_bytes(b"placeholder reference audio")

    metadata = service.generate(
        "Voice cloned narration.",
        "en",
        tmp_path / "cloned.wav",
        reference_wav=reference,
        expressiveness=0.7,
    )

    assert provider.calls[-1]["reference_wav"] == reference
    assert provider.calls[-1]["expressiveness"] == 0.7
    assert metadata["reference_used"] is True
    assert metadata["voice"] == "reference:authorized_reference.wav"


def test_ensure_vram_available_when_sufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(url: str, timeout: float) -> FakeResponse:
        assert url.endswith("/system_stats")
        return FakeResponse({"devices": [_device(12.0)]})

    def unexpected_post(*_args: object, **_kwargs: object) -> FakeResponse:
        pytest.fail("/free must not be called when VRAM is sufficient")

    monkeypatch.setattr(tts_service.httpx, "get", fake_get)
    monkeypatch.setattr(tts_service.httpx, "post", unexpected_post)

    result = TTSService()._ensure_vram_available()

    assert result["freed"] is False
    assert result["required_gb"] == MIN_FREE_VRAM_GB
    assert result["source"] == "comfyui_system_stats"
    assert float(result["free_gb"]) == pytest.approx(12.0, abs=0.01)


def test_ensure_vram_available_frees_when_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter([2.0, 9.0])
    posted: list[dict[str, Any]] = []

    def fake_get(url: str, timeout: float) -> FakeResponse:
        assert url.endswith("/system_stats")
        return FakeResponse({"devices": [_device(next(readings))]})

    def fake_post(
        url: str, json: dict[str, Any], timeout: float
    ) -> FakeResponse:
        assert url.endswith("/free")
        posted.append(json)
        return FakeResponse()

    monkeypatch.setattr(tts_service.httpx, "get", fake_get)
    monkeypatch.setattr(tts_service.httpx, "post", fake_post)

    result = TTSService()._ensure_vram_available()

    assert posted == [tts_service.FREE_PAYLOAD]
    assert result["freed"] is True
    assert float(result["free_before_gb"]) == pytest.approx(2.0, abs=0.01)
    assert float(result["free_gb"]) == pytest.approx(9.0, abs=0.01)


def test_ensure_vram_available_raises_when_still_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tts_service.httpx,
        "get",
        lambda url, timeout: FakeResponse({"devices": [_device(1.0)]}),
    )
    monkeypatch.setattr(
        tts_service.httpx,
        "post",
        lambda url, json, timeout: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="Insufficient free VRAM"):
        TTSService()._ensure_vram_available(min_free_gb=8.0)
