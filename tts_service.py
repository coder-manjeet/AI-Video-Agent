"""Local, provider-independent TTS service layer.

The default provider is Chatterbox Multilingual TTS, which lives in the
separate ``.venv-tts`` environment. This module is deliberately importable
from ``.venv`` (MCP/dev tooling): ``torch``, ``chatterbox`` and ``soundfile``
are imported only on the code paths that actually need them. When
``soundfile`` is unavailable, WAV I/O uses the Python standard-library
``wave`` module instead, so metadata tests need no GPU, network or extra
dependency.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
import wave
from array import array
from collections.abc import Iterable, Sequence
from itertools import chain
from pathlib import Path
from typing import Any, Protocol, cast

import httpx

LOGGER = logging.getLogger(__name__)

# --- Configuration -------------------------------------------------------
SAMPLE_RATE = 24_000
DEFAULT_VOICE = "chatterbox-default-narrator"
DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = frozenset({"en", "hi", "hinglish"})
DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.5
DEFAULT_TEMPERATURE = 0.8
MIN_FREE_VRAM_GB = 8.0
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8000")
HTTP_TIMEOUT_S = 15.0
FREE_PAYLOAD: dict[str, bool] = {"unload_models": True, "free_memory": True}

# Hinglish has no official Chatterbox language code. See resolve_language().
_DEVANAGARI_START = 0x0900
_DEVANAGARI_END = 0x097F
HINGLISH_DEVANAGARI_SHARE = 0.20
_BYTES_PER_GIB = 1024**3


class TTSProvider(Protocol):
    """Minimal backend contract; Kokoro or another engine can implement it."""

    name: str

    def synthesize(
        self,
        text: str,
        language: str,
        reference_wav: Path | None,
        expressiveness: float | None,
    ) -> Sequence[float]:
        """Return mono PCM samples in the range ``[-1.0, 1.0]``."""
        ...


def _as_samples(audio: object) -> list[float]:
    """Normalise a provider result (torch tensor or plain sequence) to floats."""
    value: object = audio
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        value = reshape(-1)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes)):
        raise TypeError("TTS provider returned non-audio data")
    try:
        return [float(sample) for sample in cast(Iterable[Any], value)]
    except (TypeError, ValueError) as exc:
        raise TypeError("TTS provider returned an unsupported audio type") from exc


class ChatterboxProvider:
    """Adapter around ``ChatterboxMultilingualTTS`` (imported lazily)."""

    def __init__(self, model: Any, voice: str = DEFAULT_VOICE) -> None:
        self._model = model
        self.name = voice

    def synthesize(
        self,
        text: str,
        language: str,
        reference_wav: Path | None,
        expressiveness: float | None,
    ) -> Sequence[float]:
        """Call the official Chatterbox ``generate`` API."""
        # Chatterbox calls the expressiveness knob ``exaggeration``; the other
        # documented knobs keep their pinned defaults unless extended later.
        exaggeration = (
            DEFAULT_EXAGGERATION if expressiveness is None else expressiveness
        )
        LOGGER.debug(
            "Chatterbox generate language=%s reference_used=%s",
            language,
            reference_wav is not None,
        )
        wav = self._model.generate(
            text=text,
            language_id=language,
            audio_prompt_path=str(reference_wav) if reference_wav else None,
            exaggeration=exaggeration,
            cfg_weight=DEFAULT_CFG_WEIGHT,
            temperature=DEFAULT_TEMPERATURE,
        )
        return _as_samples(wav)


# Model is process-wide: Chatterbox weights are loaded exactly once.
_MODEL: Any = None
_MODEL_LOAD_S = 0.0
_MODEL_LOCK = threading.Lock()


def _load_chatterbox_model() -> tuple[Any, float]:
    """Import and load Chatterbox in the TTS environment (no side effects at import)."""
    import torch  # type: ignore[import-not-found, import-untyped]
    from chatterbox.mtl_tts import (  # type: ignore[import-not-found, import-untyped]
        ChatterboxMultilingualTTS,
    )

    started = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_pretrained(torch.device("cuda"))
    return model, time.perf_counter() - started


def _get_chatterbox_model() -> tuple[Any, float]:
    """Return the process-wide model instance and its one-time load duration."""
    global _MODEL, _MODEL_LOAD_S
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                LOGGER.info("Loading Chatterbox multilingual model")
                _MODEL, _MODEL_LOAD_S = _load_chatterbox_model()
    return _MODEL, _MODEL_LOAD_S


def resolve_language(language: str, text: str) -> str:
    """Map the public language code to a Chatterbox ``language_id``.

    ``en`` and ``hi`` pass through. ``hinglish`` has no official Chatterbox
    code, so we use a script heuristic: among Latin and Devanagari letters,
    a Devanagari share of at least ``HINGLISH_DEVANAGARI_SHARE`` routes to
    ``hi``; otherwise the request routes to ``en``. The multilingual
    tokenizer can still emit tokens from either script, so this only selects
    the pronunciation prior. Real Hinglish quality still needs empirical
    evaluation; the service never crashes on mixed-script input.
    """
    normalized = language.strip().lower()
    if normalized not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
        )
    if normalized != "hinglish":
        return normalized

    devanagari = sum(
        1 for char in text if _DEVANAGARI_START <= ord(char) <= _DEVANAGARI_END
    )
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    total = devanagari + latin
    if total == 0:
        return DEFAULT_LANGUAGE
    return "hi" if devanagari / total >= HINGLISH_DEVANAGARI_SHARE else "en"


def _write_wav_stdlib(path: Path, samples: Sequence[float]) -> None:
    """Write mono 16-bit PCM WAV using only the standard library."""
    pcm = array(
        "h",
        (
            int(max(-1.0, min(1.0, float(sample))) * 32767.0)
            for sample in samples
        ),
    )
    if sys.byteorder == "big":
        pcm.byteswap()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def _write_wav(path: Path, samples: Sequence[float]) -> None:
    """Write a 24 kHz WAV, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf  # type: ignore[import-not-found, import-untyped]
    except ImportError:
        LOGGER.debug("soundfile unavailable; using stdlib wave writer")
        _write_wav_stdlib(path, samples)
        return
    sf.write(str(path), list(samples), SAMPLE_RATE, subtype="PCM_16")


def _read_via_soundfile(path: Path) -> tuple[int, int, Iterable[float]]:
    """Read sample rate, frame count and flattened samples with soundfile."""
    import soundfile as sf  # type: ignore[import-not-found, import-untyped]

    info = sf.info(str(path))
    data, _sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
    return (
        int(info.samplerate),
        int(info.frames),
        chain.from_iterable(data.tolist()),
    )


def _read_wav_stdlib(path: Path) -> tuple[int, int, Iterable[float]]:
    """Read sample rate, frame count and normalised samples with stdlib wave."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(
                f"Unsupported WAV sample width: {wav.getsampwidth()} bytes"
            )
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        raw = wav.readframes(frames)

    pcm = array("h")
    pcm.frombytes(raw)
    if sys.byteorder == "big":
        pcm.byteswap()
    scale = 1.0 / 32768.0
    return sample_rate, frames, (sample * scale for sample in pcm)


def _peak_and_rms(samples: Iterable[float]) -> tuple[float, float]:
    """Compute absolute peak and RMS without requiring numpy."""
    peak = 0.0
    energy = 0.0
    count = 0
    for value in samples:
        sample = abs(float(value))
        peak = max(peak, sample)
        energy += sample * sample
        count += 1
    rms = math.sqrt(energy / count) if count else 0.0
    return peak, rms


class TTSService:
    """Public TTS surface; backend details stay behind :class:`TTSProvider`."""

    def __init__(self, provider: TTSProvider | None = None) -> None:
        self._provider = provider
        self._model_load_s = 0.0

    def _get_provider(self) -> TTSProvider:
        if self._provider is None:
            model, load_s = _get_chatterbox_model()
            self._model_load_s = load_s
            self._provider = ChatterboxProvider(model)
        return self._provider

    @staticmethod
    def _validate_reference(reference_wav: str | Path | None) -> Path | None:
        if reference_wav is None:
            return None
        path = Path(reference_wav).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Reference WAV not found: {path}")
        return path

    @staticmethod
    def _validate_expressiveness(expressiveness: float | None) -> None:
        if expressiveness is None:
            return
        value = float(expressiveness)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("expressiveness must be a finite, non-negative float")

    def _fetch_free_vram_gb(self) -> float:
        """Read free CUDA VRAM reported by ComfyUI, in GiB."""
        response = httpx.get(f"{COMFYUI_URL}/system_stats", timeout=HTTP_TIMEOUT_S)
        response.raise_for_status()
        devices = response.json().get("devices", [])
        cuda_devices = [
            device
            for device in devices
            if str(device.get("type", "")).lower() == "cuda"
        ]
        if not cuda_devices:
            raise RuntimeError("ComfyUI /system_stats reported no CUDA device")
        device = max(cuda_devices, key=lambda item: float(item.get("vram_free", 0)))
        return float(device.get("vram_free", 0)) / _BYTES_PER_GIB

    def _ensure_vram_available(
        self, min_free_gb: float = MIN_FREE_VRAM_GB
    ) -> dict[str, object]:
        """Free ComfyUI VRAM if needed and verify enough space for TTS.

        ComfyUI is never closed: ``POST /free`` only unloads its models, and
        they reload automatically on the next render.
        """
        try:
            free_before = self._fetch_free_vram_gb()
        except (httpx.HTTPError, TypeError, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"Could not read ComfyUI VRAM stats from {COMFYUI_URL}: {exc}"
            ) from exc

        freed = False
        free_after = free_before
        if free_before < min_free_gb:
            LOGGER.info(
                "Free VRAM %.2f GiB is below %.2f GiB; unloading ComfyUI models",
                free_before,
                min_free_gb,
            )
            try:
                response = httpx.post(
                    f"{COMFYUI_URL}/free",
                    json=FREE_PAYLOAD,
                    timeout=HTTP_TIMEOUT_S,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"Could not ask ComfyUI to free VRAM: {exc}"
                ) from exc
            freed = True
            try:
                free_after = self._fetch_free_vram_gb()
            except (httpx.HTTPError, TypeError, ValueError, KeyError) as exc:
                raise RuntimeError(
                    f"Could not re-check ComfyUI VRAM after /free: {exc}"
                ) from exc

        if free_after < min_free_gb:
            raise RuntimeError(
                f"Insufficient free VRAM for TTS: {free_after:.2f} GiB available, "
                f"{min_free_gb:.2f} GiB required. ComfyUI was left running; "
                "free GPU memory manually before retrying."
            )

        return {
            "free_gb": round(free_after, 3),
            "free_before_gb": round(free_before, 3),
            "required_gb": float(min_free_gb),
            "freed": freed,
            "source": "comfyui_system_stats",
        }

    def get_metadata(self, wav_path: str | Path) -> dict[str, object]:
        """Return WAV metadata using audio I/O only (never loads a model)."""
        path = Path(wav_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"WAV file not found: {path}")
        try:
            sample_rate, frames, samples = _read_via_soundfile(path)
        except ImportError:
            sample_rate, frames, samples = _read_wav_stdlib(path)
        peak, rms = _peak_and_rms(samples)
        return {
            "path": str(path),
            "duration_s": frames / sample_rate if sample_rate else 0.0,
            "sample_rate": sample_rate,
            "samples": frames,
            "peak": peak,
            "rms": rms,
        }

    def generate(
        self,
        text: str,
        language: str,
        out_path: str | Path,
        reference_wav: str | Path | None = None,
        expressiveness: float | None = None,
    ) -> dict[str, object]:
        """Synthesise speech to ``out_path`` and return WAV metadata.

        ``expressiveness`` maps to Chatterbox's ``exaggeration`` parameter;
        ``None`` uses the documented default (0.5). A supplied
        ``reference_wav`` is used for authorised voice cloning via
        ``audio_prompt_path``; otherwise the default narrator voice is used.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        effective_language = resolve_language(language, text)
        reference = self._validate_reference(reference_wav)
        self._validate_expressiveness(expressiveness)
        destination = Path(out_path).expanduser().resolve()

        self._ensure_vram_available()
        provider = self._get_provider()
        started = time.perf_counter()
        samples = provider.synthesize(
            text, effective_language, reference, expressiveness
        )
        generation_s = time.perf_counter() - started
        _write_wav(destination, samples)

        metadata = self.get_metadata(destination)
        metadata.update(
            {
                "language": language.strip().lower(),
                "voice": (
                    f"reference:{reference.name}"
                    if reference is not None
                    else provider.name
                ),
                "reference_used": reference is not None,
                "model_load_s": self._model_load_s,
                "generation_s": generation_s,
            }
        )
        LOGGER.info(
            "Generated TTS WAV language=%s effective_language=%s samples=%s",
            language,
            effective_language,
            metadata["samples"],
        )
        return metadata
