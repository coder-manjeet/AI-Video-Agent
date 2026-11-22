"""Isolated end-to-end render integration using tiny generated fixtures.

This test exercises the real planner, audio production coordinator, subtitle
service, final renderer, QA validator, orchestrator and job store without
ComfyUI, TTS models, production assets or any network access. All files are
created under ``tmp_path``.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

import media_service
from project_service import ProjectService
from render_job_store import RenderJobStore
from render_orchestrator import RenderOrchestrator
from scene_service import SceneService

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe are not on PATH"
)


def _make_tiny_clip(path: Path, *, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=10",
        "-t",
        f"{seconds:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=60)


def _write_wav(path: Path, *, seconds: float, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(sample_rate * seconds))


@requires_ffmpeg
def test_complete_render_pipeline_produces_validated_outputs(tmp_path: Path) -> None:
    projects = ProjectService(tmp_path / "projects")
    projects.create_project(
        "pipeline",
        "Pipeline",
        settings={"width": 64, "height": 48, "fps": 10},
    )
    scenes = SceneService(projects)
    root = projects.project_dir("pipeline")
    originals: dict[str, bytes] = {}
    for scene_id, video_s, narration_s, text in (
        ("opening", 0.5, 0.8, "Hello from the opening scene."),
        ("closing", 0.8, 0.4, "And this is the closing narration."),
    ):
        scenes.add_scene(
            "pipeline", title=scene_id, scene_id=scene_id, duration=video_s, narration=text
        )
        clip = root / "video" / "scenes" / f"{scene_id}.mp4"
        narration = root / "assets" / "audio" / "narration" / f"{scene_id}.wav"
        _make_tiny_clip(clip, seconds=video_s)
        _write_wav(narration, seconds=narration_s)
        scenes.update_scene(
            "pipeline",
            scene_id,
            {
                "generated_video_path": f"video/scenes/{clip.name}",
                "narration_audio_path": f"assets/audio/narration/{narration.name}",
            },
        )
        originals[f"video/scenes/{clip.name}"] = clip.read_bytes()

    manifests_before = {
        "project.json": (root / "project.json").read_bytes(),
        "script/scenes.json": (root / "script/scenes.json").read_bytes(),
    }
    store = RenderJobStore(root=tmp_path / "job-store", recover_on_init=False)
    orchestrator = RenderOrchestrator(projects, scenes, store=store)
    job = orchestrator.create_job(
        "pipeline",
        {
            "preset": "custom",
            "width": 64,
            "height": 48,
            "fps": 10,
            "preset_x264": "ultrafast",
            "crf": 28,
            "render_root": "video/renders",
            "ffmpeg_timeout_s": 60,
            "ffprobe_timeout_s": 30,
            "subtitle_language": "en",
            "subtitle_granularity": "sentence",
        },
        render_id="integration-final",
    )
    assert job.status == "queued"

    result = orchestrator.execute_job(job.job_id)
    assert result["status"] == "completed", result

    completed = store.get_job(job.job_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.validation_summary["valid"] is True
    assert completed.output_paths["video"] == (
        "video/renders/audio-integration-final/final.mp4"
    )

    final_path = root / completed.output_paths["video"]
    assert final_path.is_file() and final_path.stat().st_size > 0
    info = media_service.probe_media(final_path)
    assert info["status"] == "ok"
    assert info["width"] == 64
    assert info["height"] == 48
    assert info["fps"] == pytest.approx(10.0, abs=0.05)
    assert info["video_codec"] == "h264"
    assert info["audio_codec"] == "aac"
    assert info["sample_rate"] == 48000
    assert isinstance(info["duration_s"], float)
    assert info["duration_s"] == pytest.approx(1.2, abs=0.5)

    for key in ("subtitle_srt", "subtitle_vtt"):
        subtitle = root / completed.output_paths[key]
        assert subtitle.is_file() and subtitle.stat().st_size > 0
        text = subtitle.read_text(encoding="utf-8-sig")
        assert "Hello from the opening scene." in text
        assert "closing narration" in text
    srt_text = (root / completed.output_paths["subtitle_srt"]).read_text(
        encoding="utf-8-sig"
    )
    vtt_text = (root / completed.output_paths["subtitle_vtt"]).read_text(
        encoding="utf-8-sig"
    )
    assert "-->" in srt_text and "," in srt_text
    assert vtt_text.startswith("WEBVTT") and "-->" in vtt_text

    report_json = root / completed.output_paths["report_json"]
    report_md = root / completed.output_paths["report_md"]
    assert report_json.is_file() and report_md.is_file()
    import json

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["project"]["project_id"] == "pipeline"
    assert payload["render"]["render_id"] == "integration-final"
    assert payload["scenes"][0]["scene_id"] == "opening"
    assert "C:\\" not in json.dumps(payload)

    assert (root / "project.json").read_bytes() == manifests_before["project.json"]
    assert (root / "script/scenes.json").read_bytes() == manifests_before["script/scenes.json"]
    for relative, original in originals.items():
        assert (root / relative).read_bytes() == original

    # A restart loads the same terminal job from disk instead of forgetting it.
    restarted = RenderJobStore(root=tmp_path / "job-store", recover_on_init=False)
    restored = restarted.get_job(job.job_id)
    assert restored is not None and restored.status == "completed"
