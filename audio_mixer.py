"""MCP-independent staged FFmpeg audio mixing for the final WAV master.

Each bus (narration, ducked music, SFX) is rendered to an intermediate WAV in a
unique render directory before the final mix. Explicit argument lists are used
throughout; no shell string is ever executed. Source assets are never modified,
existing masters are never overwritten, and failed render directories are kept
for diagnosis.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import uuid
import wave
from array import array
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from sys import byteorder
from typing import Any

from audio_arrangement import AudioArrangement
from audio_models import (
    AudioCommand,
    AudioMixConfig,
    AudioMixResult,
    AudioPipelineError,
    MusicSettings,
    ResolvedAudioAsset,
    SfxSettings,
    ValidationIssue,
    decimal_text,
)
from audio_timeline import AudioTimeline
from project_paths import (
    PathValidationError,
    resolve_within_project,
    validate_identifier,
)
from project_service import ProjectService

logger = logging.getLogger(__name__)

CommandRunner = Callable[[Sequence[str], float], None]
ProbeCallable = Callable[..., Mapping[str, object]]
_STDERR_TAIL_CHARS = 1000


def _fail(
    code: str,
    message: str,
    location: str = "",
    *,
    stage: str = "mix",
) -> AudioPipelineError:
    return AudioPipelineError(code, message, location, stage=stage)


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _dec(value: Decimal, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _ms(value: Decimal) -> str:
    return str(int((value * 1000).to_integral_value(rounding="ROUND_HALF_UP")))


def run_ffmpeg(command: Sequence[str], timeout_s: float) -> None:
    """Execute one FFmpeg argument list without a shell."""
    values = [str(part) for part in command]
    if not values:
        raise ValueError("command must not be empty")
    try:
        completed = subprocess.run(
            values,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise _fail(
            "audio.ffmpeg.not_installed",
            f"FFmpeg executable not found: {values[0]}",
            stage="ffmpeg",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise _fail(
            "audio.ffmpeg.timeout",
            f"FFmpeg timed out after {timeout_s:.0f}s",
            stage="ffmpeg",
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if len(detail) > _STDERR_TAIL_CHARS:
            detail = detail[-_STDERR_TAIL_CHARS:]
        raise _fail(
            "audio.ffmpeg.failed",
            f"FFmpeg failed (exit {completed.returncode})"
            + (f": {detail}" if detail else ""),
            stage="ffmpeg",
        )


def probe_audio(path: str | Path, *, timeout_s: float) -> Mapping[str, object]:
    """Default master prober backed by the existing media service."""
    import media_service

    return media_service.probe_media(path, timeout_s=timeout_s)


class AudioMixer:
    """Render narration, music and SFX buses into one validated WAV master."""

    _last_peak: float = 0.0
    _last_clipped: int = 0
    _pending_master: Path | None = None

    def __init__(
        self,
        projects: ProjectService | None = None,
        *,
        command_runner: CommandRunner | None = None,
        probe: ProbeCallable | None = None,
        check_executables: bool | None = None,
    ) -> None:
        self.projects = projects if projects is not None else ProjectService()
        self.command_runner: CommandRunner = (
            command_runner if command_runner is not None else run_ffmpeg
        )
        self.probe: ProbeCallable = probe if probe is not None else probe_audio
        self.check_executables = (
            check_executables
            if check_executables is not None
            else command_runner is None and probe is None
        )

    def render(
        self,
        project_id: str,
        timeline: AudioTimeline,
        arrangement: AudioArrangement,
        config: AudioMixConfig | Mapping[str, Any] | None = None,
    ) -> dict[str, object]:
        """Return a JSON-serializable status, result or structured error."""
        try:
            result = self.mix(project_id, timeline, arrangement, config)
        except AudioPipelineError as exc:
            return {"status": "error", "error": exc.to_dict()}
        return {"status": "ok", "result": result.to_dict()}

    def mix(
        self,
        project_id: str,
        timeline: AudioTimeline,
        arrangement: AudioArrangement,
        config: AudioMixConfig | Mapping[str, Any] | None = None,
    ) -> AudioMixResult:
        """Render and verify the synchronized master WAV."""
        if not isinstance(timeline, AudioTimeline):
            raise TypeError("timeline must be an AudioTimeline")
        if not isinstance(arrangement, AudioArrangement):
            raise TypeError("arrangement must be an AudioArrangement")
        if arrangement.project_id != project_id:
            raise _fail(
                "audio.mix.project_mismatch",
                "Arrangement belongs to a different project",
                "project_id",
            )
        try:
            settings = AudioMixConfig.from_dict(config)
        except (TypeError, ValueError) as exc:
            raise _fail("audio.mix.config_invalid", str(exc), stage="config") from exc
        try:
            validate_identifier(project_id, "project_id")
        except PathValidationError as exc:
            raise _fail("audio.identifier.invalid", str(exc), "project_id") from exc
        if not self.projects.exists(project_id):
            raise _fail(
                "audio.project.not_found",
                f"Project not found: {project_id}",
                "project_id",
                stage="project",
            )
        self._check_executables(settings)

        root = self.projects.project_dir(project_id)
        render_dir, render_id = self._create_render_dir(root, settings)
        warnings = list(arrangement.warnings)
        commands: list[AudioCommand] = []
        buses: dict[str, Path] = {}

        narration_scenes = [
            scene
            for scene in timeline.scenes
            if scene.narration_present and scene.narration_path
        ]
        if narration_scenes:
            command = self._build_narration_command(
                root, render_dir, timeline, narration_scenes, settings
            )
            self._execute(command, settings, commands, "narration")
            buses["narration"] = render_dir / "narration_bus.wav"
            self._require_output(buses["narration"], "narration bus")
        else:
            warnings.append(
                ValidationIssue(
                    "audio.mix.no_narration",
                    "No narration assets are present; no speech track is fabricated",
                    "timeline.scenes",
                )
            )

        if arrangement.music is not None:
            command = self._build_music_command(
                arrangement.music,
                render_dir,
                timeline.total_duration_s,
                buses.get("narration"),
                settings,
            )
            self._execute(command, settings, commands, "music")
            buses["music"] = render_dir / "music_bus.wav"
            self._require_output(buses["music"], "music bus")

        if arrangement.sfx:
            command = self._build_sfx_command(
                arrangement.sfx, render_dir, timeline.total_duration_s, settings
            )
            self._execute(command, settings, commands, "sfx")
            buses["sfx"] = render_dir / "sfx_bus.wav"
            self._require_output(buses["sfx"], "SFX bus")

        final_command = self._build_final_command(
            buses, render_dir, timeline.total_duration_s, settings
        )
        self._execute(final_command, settings, commands, "final")
        # The final command writes to its last argument; keep it distinct from
        # every bus and promote it only after verification.
        temp_master = self._pending_master
        if temp_master is None:
            raise _fail(
                "audio.mix.output_missing",
                "Final mixing command did not declare a temporary master path",
                stage="output",
            )
        self._require_output(temp_master, "temporary master")

        final_path = render_dir / settings.master_filename
        if final_path.exists():
            raise _fail(
                "audio.mix.output_collision",
                f"Master WAV appeared during mixing: {final_path}. No file was replaced.",
                str(final_path),
                stage="output",
            )
        duration = self._validate_master(
            temp_master, timeline.total_duration_s, settings, warnings
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temp_master.replace(final_path)
        self._require_output(final_path, "master WAV")

        ducking_applied = bool(
            arrangement.music is not None
            and isinstance(arrangement.music.settings, MusicSettings)
            and arrangement.music.settings.ducking is not None
            and "narration" in buses
        )
        result = AudioMixResult(
            project_id=project_id,
            render_id=render_id,
            render_dir=str(render_dir),
            master_path=str(final_path),
            master_duration_s=duration,
            sample_rate=settings.sample_rate,
            channels=settings.channels,
            sample_width_bytes=settings.sample_width_bytes,
            narration_inputs=len(narration_scenes),
            music_inputs=1 if arrangement.music is not None else 0,
            sfx_inputs=len(arrangement.sfx),
            ducking_applied=ducking_applied,
            peak=self._last_peak,
            clipped_samples=self._last_clipped,
            commands=tuple(commands),
            warnings=tuple(warnings),
        )
        logger.info(
            "Rendered audio master %s (%ss) for project %s",
            final_path,
            decimal_text(duration),
            project_id,
        )
        return result

    def _check_executables(self, config: AudioMixConfig) -> None:
        if not self.check_executables:
            return
        if shutil.which(config.ffmpeg_bin) is None:
            raise _fail(
                "audio.ffmpeg.not_installed",
                f"FFmpeg executable not found: {config.ffmpeg_bin}",
                stage="ffmpeg",
            )
        if shutil.which(config.ffprobe_bin) is None:
            raise _fail(
                "audio.ffprobe.not_installed",
                f"FFprobe executable not found: {config.ffprobe_bin}",
                stage="ffprobe",
            )

    def _create_render_dir(
        self, root: Path, config: AudioMixConfig
    ) -> tuple[Path, str]:
        if config.render_dir is not None:
            try:
                target = resolve_within_project(
                    root, Path(config.render_dir), "audio render directory"
                )
                render_id = validate_identifier(target.name, "render_id")
            except PathValidationError as exc:
                raise _fail("audio.render.unsafe_path", str(exc), stage="render") from exc
            if target.exists() and not target.is_dir():
                raise _fail(
                    "audio.render.collision",
                    f"Audio render path exists and is not a directory: {target}",
                    str(target),
                    stage="render",
                )
            try:
                target.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise _fail(
                    "audio.render.create_failed",
                    f"Could not create audio render directory {target}: {exc}",
                    str(target),
                    stage="render",
                ) from exc
            return target, render_id

        suffix = config.render_id or uuid.uuid4().hex
        suffix = suffix.removeprefix("mix-")
        try:
            identifier = validate_identifier(suffix, "render_id")
            render_id = validate_identifier(f"mix-{identifier}", "render_id")
        except PathValidationError as exc:
            raise _fail("audio.render.unsafe_path", str(exc), stage="render") from exc
        try:
            target = resolve_within_project(
                root, Path(config.render_root) / render_id, "audio render directory"
            )
        except PathValidationError as exc:
            raise _fail("audio.render.unsafe_path", str(exc), stage="render") from exc
        if target.exists():
            raise _fail(
                "audio.render.collision",
                f"Audio render directory already exists: {target}",
                str(target),
                stage="render",
            )
        try:
            target.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise _fail(
                "audio.render.create_failed",
                f"Could not create audio render directory {target}: {exc}",
                str(target),
                stage="render",
            ) from exc
        return target, render_id

    def _execute(
        self,
        command: list[str],
        config: AudioMixConfig,
        commands: list[AudioCommand],
        label: str,
    ) -> None:
        logger.debug("Running %s stage with %d arguments", label, len(command))
        self.command_runner(command, config.ffmpeg_timeout_s)
        commands.append(AudioCommand(tuple(command), 0))

    @staticmethod
    def _require_output(path: Path, label: str) -> None:
        if not path.is_file():
            raise _fail(
                "audio.output.missing",
                f"{label} does not exist: {path}",
                str(path),
                stage="output",
            )
        if path.stat().st_size <= 0:
            raise _fail(
                "audio.output.empty",
                f"{label} is empty: {path}",
                str(path),
                stage="output",
            )

    def _resolve_input(self, root: Path, relative: str, label: str) -> Path:
        try:
            path = resolve_within_project(root, relative, label)
        except PathValidationError as exc:
            raise _fail(
                "audio.asset.unsafe_path",
                str(exc),
                label,
                stage="input",
            ) from exc
        self._require_output(path, label)
        return path

    def _build_narration_command(
        self,
        root: Path,
        render_dir: Path,
        timeline: AudioTimeline,
        scenes: Sequence[Any],
        config: AudioMixConfig,
    ) -> list[str]:
        timing_by_id = {scene.scene_id: scene for scene in timeline.scenes}
        command = [config.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-n"]
        filters: list[str] = []
        labels: list[str] = []
        for index, scene in enumerate(scenes):
            path = self._resolve_input(
                root, str(scene.narration_path), f"scene:{scene.scene_id}.narration_audio_path"
            )
            command.extend(["-i", str(path)])
            timing = timing_by_id[scene.scene_id]
            filters.append(
                f"[{index}:a]aresample={config.sample_rate},"
                "aformat=channel_layouts=stereo,"
                f"adelay={_ms(timing.video_start_s)}:all=1,"
                f"volume=1.000000[n{index}]"
            )
            labels.append(f"[n{index}]")
        if len(labels) == 1:
            filters.append(f"{labels[0]}anull[narration]")
        else:
            filters.append(
                "".join(labels)
                + f"amix=inputs={len(labels)}:normalize=0:duration=longest[narration]"
            )
        command.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[narration]",
                "-ar",
                str(config.sample_rate),
                "-ac",
                str(config.channels),
                "-c:a",
                "pcm_s16le",
                "-sample_fmt",
                "s16",
                str(render_dir / "narration_bus.wav"),
            ]
        )
        return command

    def _build_music_command(
        self,
        music: ResolvedAudioAsset,
        render_dir: Path,
        total: Decimal,
        narration_bus: Path | None,
        config: AudioMixConfig,
    ) -> list[str]:
        settings = music.settings
        if not isinstance(settings, MusicSettings):
            raise TypeError("music settings must be MusicSettings")
        command = [config.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-n"]
        if settings.loop:
            command.extend(["-stream_loop", "-1"])
        command.extend(["-i", music.absolute_path])
        if narration_bus is not None and settings.ducking is not None:
            command.extend(["-i", str(narration_bus)])

        trim_end = settings.trim_start_s + music.trimmed_duration_s
        chain = [
            f"atrim=start={_dec(settings.trim_start_s)}:end={_dec(trim_end)}",
            "asetpts=PTS-STARTPTS",
            f"volume={_fmt(settings.volume)}",
        ]
        if settings.fade_in_s > 0:
            chain.append(
                f"afade=t=in:st={_dec(settings.fade_in_s, 3)}:d={_dec(settings.fade_in_s)}"
            )
        if settings.fade_out_s > 0:
            # Fades are applied before padding/looping, so they complete at the
            # end of the usable asset rather than being truncated by apad/atrim.
            fade_start = max(Decimal(0), music.trimmed_duration_s - settings.fade_out_s)
            chain.append(
                f"afade=t=out:st={_dec(fade_start, 3)}:d={_dec(settings.fade_out_s)}"
            )
        if music.start_s > 0:
            chain.append(f"adelay={_ms(music.start_s)}:all=1")
        chain.extend(
            [
                f"apad=whole_dur={_dec(total)}",
                f"atrim=end={_dec(total)}",
                "asetpts=PTS-STARTPTS",
                f"aresample={config.sample_rate}",
            ]
        )
        filters = ["[0:a]" + ",".join(chain) + "[m]"]
        if narration_bus is not None and settings.ducking is not None:
            ducking = settings.ducking
            ratio = min(20.0, max(1.0, ducking.ratio))
            attack = min(2000, ducking.attack_ms)
            release = min(9000, ducking.release_ms)
            sidechain = (
                f"[1:a]aresample={config.sample_rate},"
                f"apad=whole_dur={_dec(total)},"
                f"atrim=end={_dec(total)},"
                "asetpts=PTS-STARTPTS[sc]"
            )
            filters.append(sidechain)
            filters.append(
                "[m][sc]sidechaincompress="
                f"threshold={_fmt(ducking.threshold, 4)}:"
                f"ratio={_fmt(ratio, 3)}:"
                f"attack={_fmt(float(attack), 2)}:"
                f"release={_fmt(float(release), 2)}:"
                "mix=1[music]"
            )
        else:
            filters.append("[m]anull[music]")
        command.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[music]",
                "-ar",
                str(config.sample_rate),
                "-ac",
                str(config.channels),
                "-c:a",
                "pcm_s16le",
                "-sample_fmt",
                "s16",
                str(render_dir / "music_bus.wav"),
            ]
        )
        return command

    def _build_sfx_command(
        self,
        assets: Sequence[ResolvedAudioAsset],
        render_dir: Path,
        total: Decimal,
        config: AudioMixConfig,
    ) -> list[str]:
        command = [config.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-n"]
        filters: list[str] = []
        labels: list[str] = []
        for index, asset in enumerate(assets):
            settings = asset.settings
            if not isinstance(settings, SfxSettings):
                raise TypeError("SFX settings must be SfxSettings")
            command.extend(["-i", asset.absolute_path])
            trim_start = settings.trim_start_s
            trim_end = trim_start + asset.trimmed_duration_s
            chain = [
                f"atrim=start={_dec(trim_start)}:end={_dec(trim_end)}",
                "asetpts=PTS-STARTPTS",
                f"volume={_fmt(settings.volume)}",
            ]
            if settings.fade_in_s > 0:
                chain.append(
                    f"afade=t=in:st={_dec(settings.fade_in_s, 3)}:"
                    f"d={_dec(settings.fade_in_s)}"
                )
            if settings.fade_out_s > 0:
                fade_start = asset.trimmed_duration_s - settings.fade_out_s
                chain.append(
                    f"afade=t=out:st={_dec(fade_start, 3)}:"
                    f"d={_dec(settings.fade_out_s)}"
                )
            chain.extend(
                [
                    f"adelay={_ms(asset.start_s)}:all=1",
                    f"aresample={config.sample_rate}",
                    "aformat=channel_layouts=stereo",
                ]
            )
            filters.append(f"[{index}:a]" + ",".join(chain) + f"[s{index}]")
            labels.append(f"[s{index}]")
        if len(labels) == 1:
            filters.append(f"{labels[0]}anull[sfx]")
        else:
            filters.append(
                "".join(labels)
                + f"amix=inputs={len(labels)}:normalize=0:duration=longest[sfx]"
            )
        command.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[sfx]",
                "-ar",
                str(config.sample_rate),
                "-ac",
                str(config.channels),
                "-c:a",
                "pcm_s16le",
                "-sample_fmt",
                "s16",
                str(render_dir / "sfx_bus.wav"),
            ]
        )
        return command

    def _build_final_command(
        self,
        buses: Mapping[str, Path],
        render_dir: Path,
        total: Decimal,
        config: AudioMixConfig,
    ) -> list[str]:
        command = [config.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-n"]
        labels: list[str] = []
        filters: list[str] = []
        if buses:
            for index, name in enumerate(("narration", "music", "sfx")):
                path = buses.get(name)
                if path is None:
                    continue
                command.extend(["-i", str(path)])
                filters.append(
                    f"[{index}:a]aresample={config.sample_rate},"
                    f"aformat=channel_layouts=stereo[b{index}]"
                )
                labels.append(f"[b{index}]")
        else:
            command.extend(
                [
                    "-f",
                    "lavfi",
                    "-t",
                    _dec(total),
                    "-i",
                    f"anullsrc=r={config.sample_rate}:cl=stereo",
                ]
            )
            filters.append("[0:a]anull[b0]")
            labels.append("[b0]")
        if len(labels) == 1:
            filters.append(f"{labels[0]}anull[mixed]")
        else:
            filters.append(
                "".join(labels)
                + f"amix=inputs={len(labels)}:normalize=0:duration=longest[mixed]"
            )
        filters.append(
            "[mixed]alimiter="
            f"limit={_fmt(config.limiter_limit, 3)}:level=false,"
            f"apad=whole_dur={_dec(total)},"
            f"atrim=end={_dec(total)},"
            "asetpts=PTS-STARTPTS[out]"
        )
        temp_master = render_dir / f".audio_master.{uuid.uuid4().hex[:8]}.tmp.wav"
        command.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[out]",
                "-ar",
                str(config.sample_rate),
                "-ac",
                str(config.channels),
                "-c:a",
                "pcm_s16le",
                "-sample_fmt",
                "s16",
                str(temp_master),
            ]
        )
        # Remember this exact path: the caller verifies and promotes it.
        self._pending_master = temp_master
        return command

    def _validate_master(
        self,
        path: Path,
        expected_duration: Decimal,
        config: AudioMixConfig,
        warnings: list[ValidationIssue],
    ) -> Decimal:
        try:
            info = self.probe(str(path), timeout_s=config.ffprobe_timeout_s)
        except Exception as exc:
            raise _fail(
                "audio.mix.probe_failed",
                f"Could not probe master WAV {path}: {exc}",
                str(path),
                stage="validate",
            ) from exc
        if not isinstance(info, Mapping):
            raise _fail(
                "audio.mix.probe_failed",
                "Master probe returned a non-mapping result",
                str(path),
                stage="validate",
            )
        if info.get("audio_codec") != "pcm_s16le":
            raise _fail(
                "audio.mix.invalid_output",
                f"Master codec must be pcm_s16le, got {info.get('audio_codec')!r}",
                str(path),
                stage="validate",
            )
        if info.get("sample_rate") != config.sample_rate:
            raise _fail(
                "audio.mix.invalid_output",
                f"Master sample rate must be {config.sample_rate}, "
                f"got {info.get('sample_rate')!r}",
                str(path),
                stage="validate",
            )
        if info.get("channels") != config.channels:
            raise _fail(
                "audio.mix.invalid_output",
                f"Master channel count must be {config.channels}, "
                f"got {info.get('channels')!r}",
                str(path),
                stage="validate",
            )
        raw_duration = info.get("duration_s")
        if isinstance(raw_duration, bool) or not isinstance(raw_duration, (int, float)):
            raise _fail(
                "audio.mix.invalid_output",
                "Master WAV has no measurable duration",
                str(path),
                stage="validate",
            )
        duration = Decimal(str(raw_duration))
        if abs(duration - expected_duration) > config.duration_tolerance_s:
            raise _fail(
                "audio.mix.duration_mismatch",
                f"Master duration {decimal_text(duration)}s differs from expected "
                f"{decimal_text(expected_duration)}s",
                str(path),
                stage="validate",
            )
        peak, clipped = self._wav_levels(path, config)
        self._last_peak = peak
        self._last_clipped = clipped
        if not math.isfinite(peak) or peak > 1.0:
            raise _fail(
                "audio.mix.clipping",
                f"Master peak {peak:.6f} exceeds full scale",
                str(path),
                stage="validate",
            )
        if clipped:
            warnings.append(
                ValidationIssue(
                    "audio.mix.clipped_samples",
                    f"{clipped} master sample(s) reached the PCM limit; "
                    "consider lowering asset volumes",
                    str(path),
                )
            )
        return duration

    @staticmethod
    def _wav_levels(path: Path, config: AudioMixConfig) -> tuple[float, int]:
        """Return normalized peak and the count of full-scale samples."""
        try:
            with wave.open(str(path), "rb") as wav:
                if wav.getnchannels() != config.channels:
                    raise ValueError("unexpected channel count")
                if wav.getsampwidth() != config.sample_width_bytes:
                    raise ValueError("unexpected sample width")
                frames = wav.getnframes()
                samples = array("h")
                samples.frombytes(wav.readframes(frames))
        except (OSError, EOFError, wave.Error, ValueError) as exc:
            raise _fail(
                "audio.mix.invalid_output",
                f"Could not read master WAV samples: {exc}",
                str(path),
                stage="validate",
            ) from exc
        if byteorder == "big":
            samples.byteswap()
        peak = max((abs(int(sample)) for sample in samples), default=0) / 32768.0
        clipped = sum(1 for sample in samples if abs(int(sample)) >= 32767)
        return peak, clipped
