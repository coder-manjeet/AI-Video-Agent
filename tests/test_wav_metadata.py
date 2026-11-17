"""WAV metadata tests: no model load, no GPU, no network.

The dev environment (``.venv``) has no ``soundfile`` or ``numpy``, so
``TTSService`` uses its standard-library ``wave`` fallback there. In
``.venv-tts`` the same public method uses ``soundfile``. Both paths are
covered through the public API without changing either environment.
"""

from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

import pytest

from tts_service import SAMPLE_RATE, TTSService

DURATION_S = 0.25
AMPLITUDE = 0.5
FREQUENCY_HZ = 440.0


def _write_sine_wav(path: Path) -> list[int]:
    frame_count = int(SAMPLE_RATE * DURATION_S)
    phase_step = 2.0 * math.pi * FREQUENCY_HZ / SAMPLE_RATE
    pcm = [
        int(AMPLITUDE * 32767.0 * math.sin(phase_step * index))
        for index in range(frame_count)
    ]
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(array("h", pcm).tobytes())
    return pcm


def test_get_metadata_reads_sine_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "sine.wav"
    pcm = _write_sine_wav(wav_path)

    metadata = TTSService().get_metadata(wav_path)

    expected_peak = max(abs(sample) / 32768.0 for sample in pcm)
    expected_rms = math.sqrt(
        sum((sample / 32768.0) ** 2 for sample in pcm) / len(pcm)
    )
    assert metadata["sample_rate"] == SAMPLE_RATE
    assert metadata["samples"] == len(pcm)
    assert metadata["duration_s"] == pytest.approx(DURATION_S, abs=1e-6)
    assert metadata["peak"] == pytest.approx(expected_peak, abs=1e-6)
    assert metadata["rms"] == pytest.approx(expected_rms, abs=1e-6)
    assert metadata["path"] == str(wav_path.resolve())


def test_get_metadata_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="WAV file not found"):
        TTSService().get_metadata(tmp_path / "missing.wav")
