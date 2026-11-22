"""Deterministic tests for scene narration generation and registration."""

from __future__ import annotations

import wave
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from project_service import ProjectService
from scene_audio_service import SceneAudioService
from scene_service import SceneService


def _write_wav(path: Path, *, frames: int = 2400, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def _fixture(tmp_path: Path, *, narration: str = "Hello there.") -> tuple[
    ProjectService, SceneService, Path
]:
    projects = ProjectService(tmp_path)
    projects.create_project("demo", "Demo")
    scenes = SceneService(projects)
    scenes.add_scene(
        "demo",
        title="Scene One",
        scene_id="one",
        duration=5.0,
        narration=narration,
    )
    return projects, scenes, projects.project_dir("demo")


def _worker(
    calls: list[dict[str, Any]],
    *,
    frames: int = 2400,
    sample_rate: int = 24000,
    response: dict[str, object] | None = None,
) -> Any:
    def run(payload: dict[str, Any], timeout_s: float) -> dict[str, object]:
        calls.append({"payload": dict(payload), "timeout_s": timeout_s})
        output = Path(payload["out_path"])
        _write_wav(output, frames=frames, sample_rate=sample_rate)
        if response is not None:
            return response
        return {
            "duration_s": frames / sample_rate,
            "peak": 0.25,
            "rms": 0.1,
            "voice": "chatterbox-default-narrator",
            "model_load_s": 0.1,
            "generation_s": 0.2,
        }

    return run


def test_generate_creates_versioned_wav_and_registers_manifest(tmp_path: Path) -> None:
    projects, scenes, root = _fixture(tmp_path)
    calls: list[dict[str, Any]] = []
    service = SceneAudioService(projects, scenes, worker_runner=_worker(calls))

    result = service.generate_scene_narration("demo", "one", {"language": "en"})

    assert result.project_id == "demo"
    assert result.scene_id == "one"
    assert result.version == 1
    assert result.wav_path.startswith("assets/audio/narration/one-narration-v0001-en-")
    assert result.wav_path.endswith(".wav")
    assert result.duration_s == Decimal("0.1")
    assert result.sample_rate == 24000
    assert result.channels == 1
    assert result.sample_width_bytes == 2
    assert result.samples == 2400
    assert result.effective_language == "en"
    assert result.reference_used is False
    assert (root / result.wav_path).is_file()
    assert calls[0]["payload"]["text"] == "Hello there."
    assert calls[0]["payload"]["reference_wav"] is None

    scene = scenes.get_scene("demo", "one")
    assert scene.narration_audio_path == result.wav_path
    assert scene.duration == 5.0
    metadata = scene.extra["narration_generation"]
    assert isinstance(metadata, dict)
    assert metadata["version"] == 1
    assert metadata["wav_path"] == result.wav_path


def test_second_generation_uses_new_version_without_overwrite(tmp_path: Path) -> None:
    projects, scenes, _root = _fixture(tmp_path)
    calls: list[dict[str, Any]] = []
    service = SceneAudioService(projects, scenes, worker_runner=_worker(calls))

    first = service.generate_scene_narration("demo", "one")
    second = service.generate_scene_narration("demo", "one")

    assert first.version == 1
    assert second.version == 2
    assert first.wav_path != second.wav_path
    assert len({call["payload"]["out_path"] for call in calls}) == 2
    assert scenes.get_scene("demo", "one").narration_audio_path == second.wav_path


def test_generate_status_wraps_structured_errors(tmp_path: Path) -> None:
    projects, scenes, _root = _fixture(tmp_path)
    service = SceneAudioService(projects, scenes, worker_runner=_worker([]))

    payload = service.generate("demo", "one", {"voice": "other-voice"})

    assert payload["status"] == "error"
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "audio.narration.voice_unsupported"
    assert error["stage"] == "config"


@pytest.mark.parametrize(
    ("config", "code"),
    [
        ({"speed": 1.2}, "audio.narration.config_invalid"),
        ({"reference_wav": "../outside.wav"}, "audio.narration.unsafe_reference_path"),
    ],
)
def test_generate_rejects_unsupported_config(
    tmp_path: Path, config: dict[str, object], code: str
) -> None:
    projects, scenes, _root = _fixture(tmp_path)
    service = SceneAudioService(projects, scenes, worker_runner=_worker([]))

    with pytest.raises(Exception) as raised:
        service.generate_scene_narration("demo", "one", config)

    assert getattr(raised.value, "code", None) == code


def test_generate_rejects_scene_without_narration_text(tmp_path: Path) -> None:
    projects, scenes, _root = _fixture(tmp_path, narration="")
    service = SceneAudioService(projects, scenes, worker_runner=_worker([]))

    with pytest.raises(Exception) as raised:
        service.generate_scene_narration("demo", "one")

    assert getattr(raised.value, "code", None) == "audio.narration.missing_text"


def test_generate_rejects_wav_with_wrong_profile(tmp_path: Path) -> None:
    projects, scenes, _root = _fixture(tmp_path)
    service = SceneAudioService(
        projects, scenes, worker_runner=_worker([], sample_rate=48000)
    )

    with pytest.raises(Exception) as raised:
        service.generate_scene_narration("demo", "one")

    assert getattr(raised.value, "code", None) == "audio.narration.invalid_wav"
    assert scenes.get_scene("demo", "one").narration_audio_path is None


def test_generate_rejects_duration_mismatch_and_invalid_response(tmp_path: Path) -> None:
    projects, scenes, _root = _fixture(tmp_path)
    service = SceneAudioService(
        projects,
        scenes,
        worker_runner=_worker([], response={"duration_s": 9.0}),
    )

    with pytest.raises(Exception) as raised:
        service.generate_scene_narration("demo", "one")

    assert getattr(raised.value, "code", None) == "audio.narration.invalid_wav"

    service = SceneAudioService(
        projects,
        scenes,
        worker_runner=_worker([], response=[]),  # type: ignore[arg-type]
    )
    with pytest.raises(Exception) as raised:
        service.generate_scene_narration("demo", "one")

    assert getattr(raised.value, "code", None) == "audio.narration.worker_invalid_response"


def test_generate_accepts_reference_wav(tmp_path: Path) -> None:
    projects, scenes, root = _fixture(tmp_path)
    reference = root / "assets" / "voice" / "sample.wav"
    _write_wav(reference)
    calls: list[dict[str, Any]] = []
    service = SceneAudioService(projects, scenes, worker_runner=_worker(calls))

    result = service.generate_scene_narration(
        "demo", "one", {"reference_wav": "assets/voice/sample.wav"}
    )

    assert result.reference_used is True
    assert result.voice == "reference:sample.wav"
    assert calls[0]["payload"]["reference_wav"] == str(reference)
