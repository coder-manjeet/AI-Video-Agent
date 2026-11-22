"""Fully mocked tests for independent, per-scene animation generation."""

from __future__ import annotations

import copy
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import httpx
import pytest

from project_service import ProjectService
from scene_service import SceneService
from scene_video_service import SceneVideoService


def _png(width: int, height: int) -> bytes:
    data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = (
        struct.pack(">I", len(data))
        + b"IHDR"
        + data
        + struct.pack(">I", zlib.crc32(b"IHDR" + data) & 0xFFFFFFFF)
    )
    return b"\x89PNG\r\n\x1a\n" + ihdr


def _object_info(*, model: str = "ltxv-2b-0.9.6-distilled-04-25.safetensors") -> dict[str, Any]:
    return {
        "CLIPLoader": {
            "input": {
                "required": {
                    "clip_name": [["t5xxl_fp8_e4m3fn.safetensors"]],
                    "type": [["ltxv"]],
                }
            }
        },
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [[model]]}}
        },
    }


def _probe(
    *,
    duration: float = 97 / 24,
    width: int = 768,
    height: int = 512,
    fps: float = 24.0,
    codec: str | None = "h264",
    format_name: str = "mov,mp4,m4a,3gp,3g2,mj2",
) -> Any:
    def probe(path: str | Path) -> dict[str, Any]:
        file = Path(path)
        return {
            "status": "ok",
            "path": str(file),
            "duration_s": duration,
            "size_bytes": file.stat().st_size,
            "width": width,
            "height": height,
            "fps": fps,
            "video_codec": codec,
            "format_name": format_name,
        }

    return probe


class FakeComfyClient:
    """Records ComfyUI calls without a server or GPU."""

    def __init__(
        self,
        *,
        object_info: dict[str, Any] | None = None,
        payload_override: dict[str, Any] | None = None,
    ) -> None:
        self.object_info = (
            _object_info() if object_info is None else object_info
        )
        self.payload_override = payload_override
        self.history_mode = "success"
        self.history_errors_remaining = 0
        self.submit_failures_remaining = 0
        self.prompt_counter = 0
        self.submissions: list[dict[str, Any]] = []
        self.uploads: list[str] = []
        self.downloads: list[str] = []
        self.cancellations: list[str] = []
        self.inflight: set[str] = set()
        self.max_inflight = 0

    def load_workflow_file(self, path: str | Path) -> dict[str, Any]:
        payload = (
            self.payload_override
            if self.payload_override is not None
            else json.loads(Path(path).read_text(encoding="utf-8"))
        )
        assert isinstance(payload, dict)
        return copy.deepcopy(payload)

    async def upload_image(self, image_path: str) -> str:
        self.uploads.append(image_path)
        return Path(image_path).name

    async def submit_workflow(self, workflow: Any) -> str:
        if self.inflight:
            raise RuntimeError("overlapping ComfyUI submissions detected")
        if self.submit_failures_remaining > 0:
            self.submit_failures_remaining -= 1
            raise httpx.ConnectError("simulated submit failure")
        self.prompt_counter += 1
        prompt_id = f"video-prompt-{self.prompt_counter}"
        self.submissions.append(copy.deepcopy(workflow))
        self.inflight.add(prompt_id)
        self.max_inflight = max(self.max_inflight, len(self.inflight))
        return prompt_id

    async def get_prompt_history(self, prompt_id: str) -> dict[str, Any]:
        if self.history_mode == "pending":
            return {}
        if self.history_mode == "error" or self.history_errors_remaining > 0:
            if self.history_errors_remaining > 0:
                self.history_errors_remaining -= 1
            self.inflight.discard(prompt_id)
            return {
                prompt_id: {
                    "status": {
                        "status_str": "error",
                        "completed": False,
                        "messages": ["simulated LTX failure"],
                    },
                    "outputs": {},
                }
            }
        self.inflight.discard(prompt_id)
        return {
            prompt_id: {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "103": {
                        "gifs": [
                            {
                                "filename": f"{prompt_id}.mp4",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                },
            }
        }

    async def download_output_file(
        self,
        output: Any,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        path = Path(destination)
        if path.exists() and not overwrite:
            raise FileExistsError(f"exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-fake-mp4" + b"\x00" * 32)
        self.downloads.append(path.name)
        return path

    async def get_object_info(self) -> dict[str, Any]:
        return self.object_info

    async def cancel_prompt(self, prompt_id: str) -> bool:
        self.cancellations.append(prompt_id)
        self.inflight.discard(prompt_id)
        return True


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        *,
        probe: Any | None = None,
        client: FakeComfyClient | None = None,
        allow_default_prompt: bool = False,
        max_retries: int = 1,
    ) -> None:
        self.projects = ProjectService(tmp_path)
        self.projects.create_project("demo", "Demo")
        self.scenes = SceneService(self.projects)
        self.client = client if client is not None else FakeComfyClient()
        self.probe = probe if probe is not None else _probe()
        self.service = SceneVideoService(
            self.projects,
            self.scenes,
            client=self.client,
            probe_media=self.probe,
            poll_interval_s=0.001,
            timeout_s=0.1,
            max_retries=max_retries,
            allow_default_prompt=allow_default_prompt,
        )

    def add_scene(
        self,
        scene_id: str,
        title: str,
        *,
        animation_prompt: str | None = "Slow, gentle motion",
        image: bool = True,
        image_bytes: bytes | None = None,
    ) -> Any:
        self.scenes.add_scene(
            "demo",
            title=title,
            scene_id=scene_id,
            animation_prompt=animation_prompt if animation_prompt is not None else "",
            narration=f"Narration for {title}",
        )
        if image:
            image_dir = self.projects.project_dir("demo") / "assets" / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            target = image_dir / f"{scene_id}.png"
            target.write_bytes(image_bytes if image_bytes is not None else _png(768, 512))
            self.scenes.update_scene(
                "demo", scene_id, {"image_path": f"assets/images/{scene_id}.png"}
            )
        return self.scenes.get_scene("demo", scene_id)

    def manifest(self) -> dict[str, Any]:
        path = self.projects.project_dir("demo") / "script" / "scenes.json"
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return payload

    def scene_data(self, scene_id: str) -> dict[str, Any]:
        for item in self.manifest()["scenes"]:
            if item["scene_id"] == scene_id:
                return item
        raise AssertionError(f"scene {scene_id} missing from manifest")


@pytest.mark.asyncio
async def test_successful_generation_registers_path_metadata_and_patches_workflow(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    other = harness.add_scene("s2", "Closing")

    result = await harness.service.generate_scene_video(
        "demo", "s1", negative_prompt="No flicker and no camera movement"
    )

    assert result["status"] == "completed"
    assert result["output_path"] == "video/scenes/scene-001-opening.mp4"
    assert result["duration_s"] == pytest.approx(97 / 24)
    assert result["prompt_id"] == "video-prompt-1"
    assert result["error"] is None
    asset = harness.projects.project_dir("demo") / str(result["output_path"])
    assert asset.is_file() and asset.stat().st_size > 0
    scene = harness.scene_data("s1")
    assert scene["generated_video_path"] == "video/scenes/scene-001-opening.mp4"
    assert scene["status"] == "video_generated"
    generation = scene["provider_options"]["video_generation"]
    assert generation["actual_duration_s"] == pytest.approx(97 / 24)
    assert generation["output_resolution"] == "768x512"
    assert generation["frames"] == 97 and generation["fps"] == 24
    assert scene["image_path"] == "assets/images/s1.png"
    untouched = harness.scene_data("s2")
    assert untouched["generated_video_path"] == other.generated_video_path
    assert untouched["narration"] == other.narration
    submission = harness.client.submissions[0]
    assert submission["6"]["inputs"]["text"] == "Slow, gentle motion"
    assert submission["7"]["inputs"]["text"] == "No flicker and no camera movement"
    assert submission["78"]["inputs"]["image"] == "s1.png"
    assert submission["95"]["inputs"] == {
        **submission["95"]["inputs"],
        "width": 768,
        "height": 512,
        "length": 97,
    }
    assert submission["69"]["inputs"]["frame_rate"] == 24
    assert submission["103"]["inputs"]["frame_rate"] == 24
    assert submission["103"]["inputs"]["filename_prefix"] == "scene-001-opening"
    original = json.loads(
        (Path(__file__).resolve().parents[1] / "ltx_image_to_video_api.json").read_text(
            encoding="utf-8"
        )
    )
    assert submission["6"]["inputs"]["text"] != original["6"]["inputs"]["text"]


@pytest.mark.asyncio
async def test_eligibility_reports_actionable_errors(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening", image=False, animation_prompt=None)
    eligibility = harness.service.validate_scene_eligibility("demo", "s1")
    assert not eligibility["valid"]
    assert any("image_path" in error for error in eligibility["errors"])
    assert any("animation prompt" in error for error in eligibility["errors"])

    missing = harness.service.validate_scene_eligibility("nope", "s1")
    assert any("Project not found" in error for error in missing["errors"])
    unknown = harness.service.validate_scene_eligibility("demo", "unknown")
    assert any("Could not load scene" in error for error in unknown["errors"])

    image_dir = harness.projects.project_dir("demo") / "assets" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "s1.png").write_bytes(b"not a png")
    harness.scenes.update_scene("demo", "s1", {"image_path": "assets/images/s1.png"})
    invalid = harness.service.validate_scene_eligibility("demo", "s1")
    assert any("not a readable PNG/JPEG" in error for error in invalid["errors"])

    allowed = harness.service.validate_scene_eligibility(
        "demo", "s1", allow_default_prompt=True
    )
    assert allowed["animation_prompt"] is not None


@pytest.mark.asyncio
async def test_missing_image_does_not_submit_or_change_manifest(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening", image=False)
    before = harness.scene_data("s1")

    result = await harness.service.generate_scene_video("demo", "s1")

    assert result["status"] == "failed"
    assert "image_path" in str(result["error"])
    assert harness.client.submissions == []
    assert harness.client.uploads == []
    assert harness.scene_data("s1") == before
    assert harness.service.get_job(str(result["job_id"]))["status"] == "failed"


@pytest.mark.asyncio
async def test_default_prompt_requires_explicit_opt_in(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening", animation_prompt=None)

    denied = await harness.service.generate_scene_video("demo", "s1")
    assert denied["status"] == "failed"
    assert "animation prompt" in str(denied["error"])

    allowed = await harness.service.generate_scene_video(
        "demo", "s1", allow_default_prompt=True
    )
    assert allowed["status"] == "completed"
    assert harness.client.submissions[0]["6"]["inputs"]["text"]


@pytest.mark.asyncio
async def test_workflow_with_changed_node_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(
        (Path(__file__).resolve().parents[1] / "ltx_image_to_video_api.json").read_text(
            encoding="utf-8"
        )
    )
    payload["103"]["class_type"] = "SaveImage"
    harness = Harness(tmp_path, client=FakeComfyClient(payload_override=payload))
    harness.add_scene("s1", "Opening")

    result = await harness.service.generate_scene_video("demo", "s1")

    assert result["status"] == "failed"
    assert "incompatible" in str(result["error"])
    assert harness.client.submissions == []


@pytest.mark.asyncio
async def test_missing_model_is_rejected(tmp_path: Path) -> None:
    client = FakeComfyClient(
        object_info=_object_info(model="other-checkpoint.safetensors")
    )
    harness = Harness(tmp_path, client=client)
    harness.add_scene("s1", "Opening")

    result = await harness.service.generate_scene_video("demo", "s1")

    assert result["status"] == "failed"
    assert "not available" in str(result["error"])
    assert client.submissions == []
    verdict = await harness.service.verify_models()
    assert verdict["valid"] is False


@pytest.mark.asyncio
async def test_submission_retry_uses_second_attempt(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.client.submit_failures_remaining = 1
    harness.add_scene("s1", "Opening")

    result = await harness.service.generate_scene_video("demo", "s1")

    assert result["status"] == "completed"
    assert result["attempt"] == 2
    assert harness.client.prompt_counter == 1


@pytest.mark.asyncio
async def test_timeout_creates_no_output_and_no_registration(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.client.history_mode = "pending"

    result = await harness.service.generate_scene_video("demo", "s1", timeout_s=0.005)

    assert result["status"] == "timeout"
    assert "Timed out" in str(result["error"])
    assert harness.client.downloads == []
    assert harness.scene_data("s1")["generated_video_path"] is None
    assert list(
        (harness.projects.project_dir("demo") / "video" / "scenes").glob("*.mp4")
    ) == []


@pytest.mark.asyncio
async def test_comfyui_job_error_is_reported(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.client.history_mode = "error"

    result = await harness.service.generate_scene_video("demo", "s1")

    assert result["status"] == "failed"
    assert "failed" in str(result["error"])
    assert harness.client.downloads == []


@pytest.mark.parametrize(
    "probe",
    [
        _probe(codec=None),
        _probe(width=512, height=512),
        _probe(format_name="matroska,webm"),
        _probe(duration=0.0),
    ],
)
@pytest.mark.asyncio
async def test_invalid_output_is_discarded_and_not_registered(
    tmp_path: Path, probe: Any
) -> None:
    harness = Harness(tmp_path, probe=probe)
    harness.add_scene("s1", "Opening")

    result = await harness.service.generate_scene_video("demo", "s1")

    assert result["status"] == "failed"
    assert harness.scene_data("s1")["generated_video_path"] is None
    assert list((harness.projects.project_dir("demo") / "video" / "scenes").glob("*.mp4")) == []


@pytest.mark.asyncio
async def test_existing_video_is_protected_and_regenerate_creates_v2(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")

    first = await harness.service.generate_scene_video("demo", "s1")
    skipped = await harness.service.generate_scene_video("demo", "s1")
    second = await harness.service.generate_scene_video("demo", "s1", regenerate=True)

    assert first["output_path"] == "video/scenes/scene-001-opening.mp4"
    assert skipped["status"] == "skipped" and skipped["error"] == "video_exists"
    assert second["output_path"] == "video/scenes/scene-001-opening-v2.mp4"
    videos = harness.projects.project_dir("demo") / "video" / "scenes"
    assert (videos / "scene-001-opening.mp4").is_file()
    assert (videos / "scene-001-opening-v2.mp4").is_file()
    assert harness.scene_data("s1")["generated_video_path"] == second["output_path"]


@pytest.mark.asyncio
async def test_failed_regeneration_preserves_previous_video(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    first = await harness.service.generate_scene_video("demo", "s1")

    harness.client.history_mode = "error"
    failed = await harness.service.generate_scene_video("demo", "s1", regenerate=True)
    assert failed["status"] == "failed"
    assert harness.scene_data("s1")["generated_video_path"] == first["output_path"]
    assert harness.scene_data("s1")["status"] == "video_generated"

    harness.client.history_mode = "success"
    retried = await harness.service.retry_scene_video("demo", "s1")
    assert retried["status"] == "completed"
    assert retried["output_path"] == "video/scenes/scene-001-opening-v2.mp4"


@pytest.mark.asyncio
async def test_filename_allocation_skips_existing_and_registered_paths(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    videos = harness.projects.project_dir("demo") / "video" / "scenes"
    videos.mkdir(parents=True, exist_ok=True)
    (videos / "scene-001-opening.mp4").write_bytes(b"keep")

    result = await harness.service.generate_scene_video("demo", "s1")

    assert result["output_path"] == "video/scenes/scene-001-opening-v2.mp4"
    assert (videos / "scene-001-opening.mp4").read_bytes() == b"keep"


@pytest.mark.asyncio
async def test_retry_job_reuses_saved_options_and_increments_attempt(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.client.history_errors_remaining = 1

    failed = await harness.service.generate_scene_video(
        "demo", "s1", animation_prompt="Custom motion", seed=17
    )
    assert failed["status"] == "failed"

    retried = await harness.service.retry_job(str(failed["job_id"]))

    assert retried["status"] == "completed"
    assert retried["attempt"] == 2
    assert retried["job_id"] == failed["job_id"]
    assert harness.client.submissions[0]["6"]["inputs"]["text"] == "Custom motion"
    assert harness.client.submissions[0]["102"]["inputs"]["noise_seed"] == 17


@pytest.mark.asyncio
async def test_sequential_generation_has_no_overlap(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.add_scene("s2", "Closing")

    results = await harness.service.generate_scene_videos("demo", scene_ids=["s1", "s2"])

    assert [item["status"] for item in results] == ["completed", "completed"]
    assert harness.client.max_inflight == 1
    assert harness.scene_data("s2")["generated_video_path"] == (
        "video/scenes/scene-002-closing.mp4"
    )


@pytest.mark.asyncio
async def test_other_scenes_keep_all_their_data(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.add_scene("s2", "Closing")
    before = harness.scene_data("s2")

    await harness.service.generate_scene_video("demo", "s1")

    assert harness.scene_data("s2") == before


@pytest.mark.asyncio
async def test_cancel_job_is_tracked_and_allows_retry(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.client.history_mode = "pending"
    result = await harness.service.generate_scene_video("demo", "s1", timeout_s=0.005)
    job_id = str(result["job_id"])
    assert result["status"] == "timeout"

    cancelled = await harness.service.cancel_job(job_id)
    assert cancelled["supported"] is True
    assert harness.service.get_job(job_id)["status"] == "cancelled"
    assert harness.client.cancellations == ["video-prompt-1"]

    harness.client.history_mode = "success"
    retry = await harness.service.retry_job(job_id)
    assert retry["status"] == "completed"


@pytest.mark.asyncio
async def test_job_registry_tracks_ids_timestamps_and_errors(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.add_scene("s2", "Closing")

    await harness.service.generate_scene_video("demo", "s1")
    harness.client.history_mode = "error"
    failed = await harness.service.generate_scene_video("demo", "s2")

    jobs = harness.service.list_jobs(project_id="demo", scene_id="s1")
    assert len(jobs) == 1
    assert jobs[0]["prompt_id"] == "video-prompt-1"
    assert jobs[0]["submitted_at"] and jobs[0]["completed_at"]
    assert jobs[0]["output_path"] == "video/scenes/scene-001-opening.mp4"
    failed_job = harness.service.get_job(str(failed["job_id"]))
    assert failed_job is not None and failed_job["error"]
    assert harness.service.get_job("missing") is None


@pytest.mark.asyncio
async def test_frames_and_fps_are_validated_before_submission(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")

    bad_frames = await harness.service.generate_scene_video("demo", "s1", frames=0)
    bad_fps = await harness.service.generate_scene_video("demo", "s1", fps=200)
    assert bad_frames["status"] == "failed" and "frames" in str(bad_frames["error"])
    assert bad_fps["status"] == "failed" and "fps" in str(bad_fps["error"])
    assert harness.client.submissions == []


@pytest.mark.asyncio
async def test_duration_estimate_only_produces_a_warning(tmp_path: Path) -> None:
    harness = Harness(tmp_path, probe=_probe(duration=8.0))
    harness.add_scene("s1", "Opening")

    result = await harness.service.generate_scene_video("demo", "s1")

    assert result["status"] == "completed"
    assert result["duration_s"] == pytest.approx(8.0)
    assert any("duration" in warning for warning in result["warnings"])
    assert harness.scene_data("s1")["provider_options"]["video_generation"][
        "actual_duration_s"
    ] == pytest.approx(8.0)


def test_validate_workflow_reports_the_verified_default(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    verdict = harness.service.validate_workflow()

    assert verdict["valid"] is True
    assert verdict["workflow_file"] == "ltx_image_to_video_api.json"
    assert [item["name"] for item in harness.service.list_workflows()] == [
        "ltx_image_to_video"
    ]
