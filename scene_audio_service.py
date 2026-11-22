"""MCP-independent scene narration generation and registration.

Speech synthesis stays in the existing ``.venv-tts`` Chatterbox worker: this
service only builds the request, runs the same isolated subprocess protocol used
by ``tts_server.py``, verifies the resulting WAV with the standard library and
registers it through :class:`~scene_service.SceneService`. No worker internals,
model behaviour, ComfyUI workflow or scene-generation service is modified.

Generated files use unique versioned names; existing narration is never
replaced. All manifest paths are project-relative.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import tempfile
import uuid
import wave
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from audio_models import (
    AudioPipelineError,
    NarrationConfig,
    NarrationResult,
    decimal_text,
    update_scene_extra,
)
from project_paths import (
    PathValidationError,
    resolve_within_project,
    to_project_relative,
    validate_identifier,
)
from project_service import ProjectService
from scene_service import SceneNotFoundError, SceneService, SceneServiceError
from tts_service import DEFAULT_VOICE, resolve_language

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
TTS_PYTHON = PROJECT_ROOT / ".venv-tts" / "Scripts" / "python.exe"
TTS_WORKER = PROJECT_ROOT / "tts_worker.py"
WAV_SUFFIX = ".wav"
EXPECTED_SAMPLE_RATE = 24_000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH_BYTES = 2
DURATION_CHECK_TOLERANCE_S = Decimal("0.05")
_VERSION_RE = re.compile(r"-narration-v(\d+)-")

WorkerRunner = Callable[[dict[str, Any], float], dict[str, object]]


def _fail(
    code: str,
    message: str,
    location: str = "",
    *,
    stage: str = "narration",
) -> AudioPipelineError:
    return AudioPipelineError(code, message, location, stage=stage)


def _log_tail(path: Path, limit: int = 1500) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()[-limit:]


def run_tts_worker(payload: dict[str, Any], timeout_s: float) -> dict[str, object]:
    """Run the existing TTS worker with a JSON file protocol and a timeout."""
    if not TTS_PYTHON.is_file():
        raise _fail(
            "audio.narration.tts_unavailable",
            f"TTS interpreter not found: {TTS_PYTHON}",
            stage="tts",
        )
    if not TTS_WORKER.is_file():
        raise _fail(
            "audio.narration.tts_unavailable",
            f"TTS worker not found: {TTS_WORKER}",
            stage="tts",
        )
    with tempfile.TemporaryDirectory(prefix="scene-audio-") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        log = Path(directory) / "worker.log"
        request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with log.open("w", encoding="utf-8") as log_file:
            try:
                completed = subprocess.run(
                    [str(TTS_PYTHON), str(TTS_WORKER), str(request), str(response)],
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                detail = _log_tail(log)
                raise _fail(
                    "audio.narration.tts_timeout",
                    f"TTS worker timed out after {timeout_s:.0f}s"
                    + (f". Last output: {detail}" if detail else ""),
                    stage="tts",
                ) from exc
        if not response.is_file():
            detail = _log_tail(log)
            raise _fail(
                "audio.narration.worker_failed",
                f"TTS worker produced no response (exit {completed.returncode})"
                + (f". Last output: {detail}" if detail else ""),
                stage="tts",
            )
        try:
            decoded: object = json.loads(response.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise _fail(
                "audio.narration.worker_invalid_response",
                "TTS worker returned invalid JSON",
                str(response),
                stage="tts",
            ) from exc
    if not isinstance(decoded, dict):
        raise _fail(
            "audio.narration.worker_invalid_response",
            "TTS worker response must be a JSON object",
            stage="tts",
        )
    if not decoded.get("ok"):
        detail = str(decoded.get("error") or "TTS generation failed")
        raise _fail(
            "audio.narration.generation_failed", detail, stage="tts"
        )
    metadata = decoded.get("metadata")
    if not isinstance(metadata, dict):
        raise _fail(
            "audio.narration.worker_invalid_response",
            "TTS worker response is missing metadata",
            stage="tts",
        )
    return metadata


@dataclass(frozen=True)
class _WavInfo:
    sample_rate: int
    channels: int
    sample_width_bytes: int
    samples: int
    duration_s: Decimal
    file_size_bytes: int


def _inspect_wav(path: Path) -> _WavInfo:
    """Verify the strict Chatterbox WAV profile without loading audio libraries."""
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise _fail(
            "audio.narration.invalid_wav",
            f"Could not read narration WAV {path}: {exc}",
            str(path),
            stage="verify",
        ) from exc
    if channels != EXPECTED_CHANNELS:
        raise _fail(
            "audio.narration.invalid_wav",
            f"Expected {EXPECTED_CHANNELS} channel(s), got {channels}",
            str(path),
            stage="verify",
        )
    if sample_width != EXPECTED_SAMPLE_WIDTH_BYTES:
        raise _fail(
            "audio.narration.invalid_wav",
            f"Expected {EXPECTED_SAMPLE_WIDTH_BYTES}-byte PCM, got {sample_width}",
            str(path),
            stage="verify",
        )
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise _fail(
            "audio.narration.invalid_wav",
            f"Expected {EXPECTED_SAMPLE_RATE} Hz, got {sample_rate}",
            str(path),
            stage="verify",
        )
    if frames <= 0:
        raise _fail(
            "audio.narration.invalid_wav",
            "Narration WAV contains no audio frames",
            str(path),
            stage="verify",
        )
    size = path.stat().st_size
    if size <= 0:
        raise _fail(
            "audio.narration.invalid_wav",
            "Narration WAV is empty",
            str(path),
            stage="verify",
        )
    return _WavInfo(
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        samples=frames,
        duration_s=Decimal(frames) / Decimal(sample_rate),
        file_size_bytes=size,
    )


def _metadata_number(metadata: Mapping[str, object], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class SceneAudioService:
    """Generate narration for one scene and register it in the scene manifest."""

    def __init__(
        self,
        projects: ProjectService | None = None,
        scenes: SceneService | None = None,
        *,
        worker_runner: WorkerRunner | None = None,
    ) -> None:
        self.projects = projects if projects is not None else ProjectService()
        self.scenes = scenes if scenes is not None else SceneService(self.projects)
        self.worker_runner: WorkerRunner = (
            worker_runner if worker_runner is not None else run_tts_worker
        )

    def generate(
        self,
        project_id: str,
        scene_id: str,
        config: NarrationConfig | Mapping[str, Any] | None = None,
        *,
        text: str | None = None,
    ) -> dict[str, object]:
        """Return a JSON-serializable status, result or structured error."""
        try:
            result = self.generate_scene_narration(
                project_id, scene_id, config, text=text
            )
        except AudioPipelineError as exc:
            return {"status": "error", "error": exc.to_dict()}
        return {"status": "ok", "result": result.to_dict()}

    def generate_scene_narration(
        self,
        project_id: str,
        scene_id: str,
        config: NarrationConfig | Mapping[str, Any] | None = None,
        *,
        text: str | None = None,
    ) -> NarrationResult:
        """Synthesize, verify and register narration for one scene."""
        for label, value in (("project_id", project_id), ("scene_id", scene_id)):
            try:
                validate_identifier(value, label)
            except PathValidationError as exc:
                raise _fail("audio.identifier.invalid", str(exc), label) from exc
        if not self.projects.exists(project_id):
            raise _fail(
                "audio.project.not_found",
                f"Project not found: {project_id}",
                "project_id",
                stage="project",
            )
        try:
            settings = NarrationConfig.from_dict(config)
        except (TypeError, ValueError) as exc:
            raise _fail(
                "audio.narration.config_invalid", str(exc), stage="config"
            ) from exc

        try:
            scene = self.scenes.get_scene(project_id, scene_id)
        except SceneNotFoundError as exc:
            raise _fail(
                "audio.scene.not_found",
                str(exc),
                "scene_id",
                stage="scene",
            ) from exc
        except SceneServiceError as exc:
            raise _fail(
                "audio.scene.unreadable", str(exc), "scene_id", stage="scene"
            ) from exc

        narration_text = scene.narration if text is None else text
        if not isinstance(narration_text, str) or not narration_text.strip():
            raise _fail(
                "audio.narration.missing_text",
                f"Scene {scene_id} has no narration text; no speech is fabricated",
                f"scene:{scene_id}.narration",
                stage="scene",
            )

        if settings.voice != DEFAULT_VOICE:
            raise _fail(
                "audio.narration.voice_unsupported",
                f"Unsupported voice {settings.voice!r}. Only {DEFAULT_VOICE} is "
                "supported; a licensed reference WAV can be supplied via "
                "reference_wav.",
                "voice",
                stage="config",
            )
        reference: Path | None = None
        if settings.reference_wav is not None:
            root = self.projects.project_dir(project_id)
            try:
                reference = resolve_within_project(
                    root, settings.reference_wav, "reference_wav"
                )
            except PathValidationError as exc:
                raise _fail(
                    "audio.narration.unsafe_reference_path",
                    str(exc),
                    "reference_wav",
                    stage="config",
                ) from exc
            if reference.suffix.lower() != WAV_SUFFIX or not reference.is_file():
                raise _fail(
                    "audio.narration.reference_missing",
                    f"Reference WAV not found: {reference}",
                    "reference_wav",
                    stage="config",
                )

        root = self.projects.project_dir(project_id)
        try:
            output_dir = resolve_within_project(
                root, settings.output_root, "output_root"
            )
        except PathValidationError as exc:
            raise _fail(
                "audio.narration.unsafe_output_path",
                str(exc),
                "output_root",
                stage="config",
            ) from exc
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _fail(
                "audio.narration.output_dir_failed",
                f"Could not create narration directory {output_dir}: {exc}",
                str(output_dir),
                stage="output",
            ) from exc
        destination = self._versioned_path(output_dir, scene_id, settings.language)
        relative = to_project_relative(destination, root)

        effective_language = resolve_language(settings.language, narration_text)
        payload: dict[str, Any] = {
            "text": narration_text,
            "language": effective_language,
            "out_path": str(destination),
            "reference_wav": str(reference) if reference is not None else None,
            "expressiveness": settings.expressiveness,
        }
        logger.info(
            "Generating narration for scene %s in project %s (language=%s)",
            scene_id,
            project_id,
            settings.language,
        )
        try:
            metadata = self.worker_runner(payload, settings.timeout_s)
        except AudioPipelineError:
            raise
        except Exception as exc:
            raise _fail(
                "audio.narration.worker_failed",
                f"{type(exc).__name__}: {exc}",
                stage="tts",
            ) from exc
        if not isinstance(metadata, Mapping):
            raise _fail(
                "audio.narration.worker_invalid_response",
                "TTS worker metadata must be a mapping",
                stage="tts",
            )

        info = _inspect_wav(destination)
        duration = info.duration_s
        reported = _metadata_number(metadata, "duration_s")
        if reported is not None and abs(duration - Decimal(str(reported))) > DURATION_CHECK_TOLERANCE_S:
            raise _fail(
                "audio.narration.invalid_wav",
                "Narration WAV duration does not match TTS worker metadata",
                str(destination),
                stage="verify",
            )

        generated_at = datetime.now(UTC).isoformat()
        peak = _metadata_number(metadata, "peak") or 0.0
        rms = _metadata_number(metadata, "rms") or 0.0
        result = NarrationResult(
            project_id=project_id,
            scene_id=scene_id,
            wav_path=relative,
            version=self._version_of(destination.name, scene_id),
            duration_s=duration,
            sample_rate=info.sample_rate,
            channels=info.channels,
            sample_width_bytes=info.sample_width_bytes,
            samples=info.samples,
            file_size_bytes=info.file_size_bytes,
            language=settings.language,
            effective_language=effective_language,
            voice=(
                f"reference:{reference.name}"
                if reference is not None
                else str(metadata.get("voice") or DEFAULT_VOICE)
            ),
            expressiveness=settings.expressiveness,
            reference_used=reference is not None,
            text_sha256=hashlib.sha256(narration_text.encode("utf-8")).hexdigest(),
            generated_at=generated_at,
            peak=peak,
            rms=rms,
            model_load_s=_metadata_number(metadata, "model_load_s") or 0.0,
            generation_s=_metadata_number(metadata, "generation_s") or 0.0,
        )

        try:
            self.scenes.register_asset(
                project_id, scene_id, "narration_audio", relative
            )
        except (SceneServiceError, ValueError, PathValidationError) as exc:
            raise _fail(
                "audio.narration.registration_failed",
                f"Generated WAV could not be registered with the scene: {exc}",
                f"scene:{scene_id}.narration_audio_path",
                stage="register",
            ) from exc
        metadata = result.to_manifest_metadata()

        def store_narration_extra(extra: dict[str, object]) -> None:
            extra["narration_generation"] = metadata

        try:
            update_scene_extra(
                self.projects, project_id, scene_id, store_narration_extra
            )
        except (OSError, ValueError, TypeError, KeyError, PathValidationError) as exc:
            raise _fail(
                "audio.narration.metadata_failed",
                f"Narration metadata could not be stored: {exc}",
                f"scene:{scene_id}.extra.narration_generation",
                stage="register",
            ) from exc
        logger.info(
            "Registered narration %s (version %d, %ss) for scene %s",
            relative,
            result.version,
            decimal_text(result.duration_s),
            scene_id,
        )
        return result

    @staticmethod
    def _version_of(filename: str, scene_id: str) -> int:
        match = _VERSION_RE.search(filename)
        return int(match.group(1)) if match and scene_id in filename else 0

    @staticmethod
    def _versioned_path(output_dir: Path, scene_id: str, language: str) -> Path:
        versions = [
            int(match.group(1))
            for path in output_dir.glob(f"{scene_id}-narration-v*{WAV_SUFFIX}")
            if (match := _VERSION_RE.search(path.name))
        ]
        version = max(versions, default=0) + 1
        language_tag = re.sub(r"[^a-z0-9]+", "", language.lower()) or "en"
        for _ in range(20):
            candidate = output_dir / (
                f"{scene_id}-narration-v{version:04d}-"
                f"{language_tag}-{uuid.uuid4().hex[:8]}{WAV_SUFFIX}"
            )
            if not candidate.exists():
                return candidate
        raise _fail(
            "audio.narration.output_collision",
            f"Could not allocate a unique narration filename in {output_dir}",
            str(output_dir),
            stage="output",
        )
