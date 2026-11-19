"""Tests for sequential, mock-backed scene image generation."""

from __future__ import annotations

import copy
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import httpx
import pytest

from character_models import CharacterSceneOverride, VisualIdentity
from character_service import CharacterService
from project_service import ProjectService
from scene_image_service import (
    JobNotFoundError,
    SceneImageService,
    SceneImageServiceError,
)
from scene_service import Scene, SceneService


def _png(width: int, height: int) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = (
        struct.pack(">I", len(data))
        + b"IHDR"
        + data
        + struct.pack(">I", zlib.crc32(b"IHDR" + data) & 0xFFFFFFFF)
    )
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))
    return signature + ihdr + iend


class FakeComfyClient:
    """In-memory ComfyUI double that records submissions and downloads."""

    def __init__(self, *, png: bytes | None = None) -> None:
        self.png = png if png is not None else _png(768, 512)
        self.history_mode = "success"
        self.history_errors_remaining = 0
        self.submit_failures_remaining = 0
        self.prompt_counter = 0
        self.submissions: list[dict[str, Any]] = []
        self.downloads: list[str] = []
        self.inflight: set[str] = set()
        self.max_inflight = 0

    def load_workflow_file(self, path: str | Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        return payload

    async def submit_workflow(self, workflow: Any) -> str:
        if self.inflight:
            raise RuntimeError("overlapping ComfyUI submissions detected")
        if self.submit_failures_remaining > 0:
            self.submit_failures_remaining -= 1
            raise httpx.ConnectError("simulated submit failure")
        self.prompt_counter += 1
        prompt_id = f"prompt-{self.prompt_counter}"
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
                        "messages": ["simulated ComfyUI failure"],
                    },
                    "outputs": {},
                }
            }
        self.inflight.discard(prompt_id)
        return {
            prompt_id: {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "10": {
                        "images": [
                            {
                                "filename": f"{prompt_id}.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                },
            }
        }

    async def download_output_image(
        self,
        image: Any,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        path = Path(destination)
        if path.exists() and not overwrite:
            raise FileExistsError(f"exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.png)
        self.downloads.append(path.name)
        return path


class Harness:
    def __init__(self, tmp_path: Path, *, png: bytes | None = None) -> None:
        self.projects = ProjectService(tmp_path)
        self.projects.create_project("demo", "Demo")
        self.scenes = SceneService(self.projects)
        self.client = FakeComfyClient(png=png)
        self.service = SceneImageService(
            self.projects,
            self.scenes,
            client=self.client,
            poll_interval_s=0.001,
            timeout_s=0.1,
            max_retries=1,
        )

    def add_scene(self, scene_id: str, title: str, **kwargs: Any) -> Scene:
        return self.scenes.add_scene(
            "demo",
            title=title,
            scene_id=scene_id,
            image_prompt=kwargs.pop("image_prompt", f"Prompt for {title}"),
            **kwargs,
        )

    def manifest(self) -> dict[str, Any]:
        path = self.projects.project_dir("demo") / "script" / "scenes.json"
        return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_successful_generation_registers_project_relative_path(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path, png=_png(512, 512))
    harness.add_scene("s1", "Opening")
    unchanged = harness.add_scene("s2", "Closing")

    result = await harness.service.generate_scene_image("demo", "s1", width=512, height=512)

    assert result["status"] == "completed"
    assert result["image_path"] == "assets/images/scene-001-opening.png"
    assert result["filename"] == "scene-001-opening.png"
    assert result["width"] == 512
    assert result["height"] == 512
    assert result["prompt_id"] == "prompt-1"
    assert result["warnings"] == []
    asset = harness.projects.project_dir("demo") / result["image_path"]
    assert asset.is_file()
    assert harness.scenes.get_scene("demo", "s1").status == "image_generated"
    assert harness.scenes.get_scene("demo", "s2").image_path == unchanged.image_path
    assert harness.scenes.get_scene("demo", "s2").status == unchanged.status
    assert harness.client.submissions[0]["4"]["inputs"]["text"] == (
        "Prompt for Opening"
    )


@pytest.mark.asyncio
async def test_existing_image_is_protected_and_regenerate_creates_version(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.add_scene("s2", "Closing")

    first = await harness.service.generate_scene_image("demo", "s1")
    original = (
        harness.projects.project_dir("demo") / first["image_path"]
    ).read_bytes()
    harness.client.png = _png(400, 300)

    skipped = await harness.service.generate_scene_image("demo", "s1")
    assert skipped["status"] == "skipped"
    assert skipped["error"] == "image_exists"
    assert len(harness.client.downloads) == 1

    regenerated = await harness.service.generate_scene_image(
        "demo", "s1", regenerate=True, width=400, height=300
    )
    assert regenerated["status"] == "completed"
    assert regenerated["filename"] == "scene-001-opening-v2.png"
    assert (
        harness.projects.project_dir("demo") / first["image_path"]
    ).read_bytes() == original
    assert harness.scenes.get_scene("demo", "s1").image_path == (
        regenerated["image_path"]
    )
    assert harness.scenes.get_scene("demo", "s2").image_path is None
    assert len(harness.client.submissions) == 2


@pytest.mark.asyncio
async def test_failed_generation_does_not_modify_scene_manifest(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.client.history_mode = "error"
    before = harness.manifest()

    result = await harness.service.generate_scene_image("demo", "s1")

    assert result["status"] == "failed"
    assert "simulated ComfyUI failure" in result["error"]
    assert result["prompt_id"] == "prompt-1"
    assert harness.manifest() == before
    job = harness.service.get_job(result["job_id"])
    assert job is not None
    assert job["status"] == "failed"
    assert harness.client.downloads == []


@pytest.mark.asyncio
async def test_timeout_does_not_register_an_image(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.client.history_mode = "pending"
    before = harness.manifest()

    result = await harness.service.generate_scene_image("demo", "s1", timeout_s=0.03)

    assert result["status"] == "timeout"
    assert "Timed out" in result["error"]
    assert harness.manifest() == before
    assert harness.client.downloads == []


@pytest.mark.asyncio
async def test_retry_after_failed_generation_succeeds(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.client.history_errors_remaining = 1

    failed = await harness.service.generate_scene_image("demo", "s1")
    assert failed["status"] == "failed"
    assert harness.scenes.get_scene("demo", "s1").image_path is None

    retried = await harness.service.retry_scene_image("demo", "s1")

    assert retried["status"] == "completed"
    assert retried["prompt_id"] == "prompt-2"
    assert retried["image_path"] == "assets/images/scene-001-opening.png"
    assert harness.scenes.get_scene("demo", "s1").status == "image_generated"
    assert len(harness.client.submissions) == 2


@pytest.mark.asyncio
async def test_retry_job_rejects_unknown_and_completed_jobs(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    completed = await harness.service.generate_scene_image("demo", "s1")

    with pytest.raises(JobNotFoundError):
        await harness.service.retry_job("missing-job")
    with pytest.raises(SceneImageServiceError, match="cannot be retried"):
        await harness.service.retry_job(completed["job_id"])


@pytest.mark.asyncio
async def test_submission_failure_is_retried(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.client.submit_failures_remaining = 1

    result = await harness.service.generate_scene_image("demo", "s1")

    assert result["status"] == "completed"
    assert result["attempt"] == 2
    assert len(harness.client.submissions) == 1


@pytest.mark.asyncio
async def test_missing_scene_images_are_detected_and_generated(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    harness.add_scene("s2", "Middle")
    harness.add_scene("s3", "Closing")
    project_root = harness.projects.project_dir("demo")
    image_dir = project_root / "assets" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "existing.png").write_bytes(_png(512, 512))
    harness.scenes.register_asset("demo", "s2", "image", image_dir / "existing.png")
    (image_dir / "stale.png").write_bytes(_png(512, 512))
    harness.scenes.register_asset("demo", "s3", "image", image_dir / "stale.png")
    (image_dir / "stale.png").unlink()

    assert harness.service.list_missing_images("demo") == ["s1", "s3"]

    results = await harness.service.generate_scene_images("demo")

    assert [item["scene_id"] for item in results] == ["s1", "s3"]
    assert all(item["status"] == "completed" for item in results)
    assert harness.scenes.get_scene("demo", "s2").image_path == (
        "assets/images/existing.png"
    )
    assert harness.client.max_inflight == 1


@pytest.mark.asyncio
async def test_multi_scene_generation_is_sequential_and_isolated(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    for index, title in enumerate(("One", "Two", "Three"), start=1):
        harness.add_scene(f"s{index}", title)
    before = {
        scene.scene_id: scene.to_dict()
        for scene in harness.scenes.load_scenes("demo")
    }

    results = await harness.service.generate_scene_images(
        "demo", scene_ids=["s1", "s2", "s3"]
    )

    assert [item["status"] for item in results] == ["completed"] * 3
    assert [item["filename"] for item in results] == [
        "scene-001-one.png",
        "scene-002-two.png",
        "scene-003-three.png",
    ]
    assert harness.client.max_inflight == 1
    assert len(harness.client.submissions) == 3
    assert [
        item["4"]["inputs"]["text"] for item in harness.client.submissions
    ] == ["Prompt for One", "Prompt for Two", "Prompt for Three"]
    for scene in harness.scenes.load_scenes("demo"):
        if scene.scene_id == "s1":
            continue
        original = before[scene.scene_id]
        changed = {"image_path", "status"}
        current = {
            key: value
            for key, value in scene.to_dict().items()
            if key not in changed
        }
        previous = {
            key: value for key, value in original.items() if key not in changed
        }
        assert current == previous
        assert scene.image_path == (
            f"assets/images/scene-{scene.scene_number:03d}-{scene.title.lower()}.png"
        )
        assert scene.status == "image_generated"


@pytest.mark.asyncio
async def test_negative_prompt_and_consistency_metadata_are_reported(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening", negative_prompt="blurry text")

    result = await harness.service.generate_scene_image(
        "demo",
        "s1",
        character_references=["assets/characters/hero.png"],
        visual_style="cinematic",
        color_palette="warm amber",
        character_appearance="tall explorer",
    )

    assert result["status"] == "completed"
    warnings = " ".join(result["warnings"])
    assert "negative prompt" in warnings.lower()
    assert "character consistency" in warnings.lower()
    assert "visual style" in warnings.lower()
    assert "color palette" in warnings.lower()
    assert "character appearance" in warnings.lower()
    patched = harness.client.submissions[0]
    assert patched["4"]["inputs"]["text"] == "Prompt for Opening"
    assert patched["5"]["class_type"] == "ConditioningZeroOut"


@pytest.mark.asyncio
async def test_wrong_dimensions_from_comfyui_are_rejected(tmp_path: Path) -> None:
    harness = Harness(tmp_path, png=_png(300, 200))
    harness.add_scene("s1", "Opening")
    before = harness.manifest()

    result = await harness.service.generate_scene_image(
        "demo", "s1", width=512, height=512
    )

    assert result["status"] == "failed"
    assert "300x200" in result["error"]
    assert harness.manifest() == before


@pytest.mark.asyncio
async def test_composed_prompt_is_used_only_when_opted_in(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.projects.update_visual_identity(
        "demo", identity=VisualIdentity(illustration_style="cinematic")
    )
    CharacterService(harness.projects).create_character(
        "demo", name="Aria", character_id="aria", description="a cartographer"
    )
    harness.add_scene("s1", "Opening", description="A rainy street")
    harness.scenes.update_scene(
        "demo",
        "s1",
        {
            "character_ids": ["aria"],
            "character_overrides": {
                "aria": CharacterSceneOverride(expression="determined").to_dict()
            },
        },
    )

    direct = await harness.service.generate_scene_image("demo", "s1", regenerate=True)
    composed = await harness.service.generate_scene_image(
        "demo", "s1", regenerate=True, compose_prompt=True
    )

    assert direct["status"] == "completed"
    assert composed["status"] == "completed"
    direct_text = harness.client.submissions[-2]["4"]["inputs"]["text"]
    composed_text = harness.client.submissions[-1]["4"]["inputs"]["text"]
    assert direct_text == "Prompt for Opening"
    assert "style: cinematic" in composed_text
    assert "a cartographer" in composed_text
    assert "expression: determined" in composed_text
    assert "A rainy street" in composed_text
    assert "Prompt: Prompt for Opening" in composed_text

    direct_job = harness.service.get_job(direct["job_id"])
    composed_job = harness.service.get_job(composed["job_id"])
    assert direct_job is not None and composed_job is not None
    assert direct_job["metadata"]["compose_prompt"] is False
    assert direct_job["metadata"]["prompt_composition"] is None
    assert composed_job["metadata"]["compose_prompt"] is True
    assert composed_job["metadata"]["prompt_composition"]["original_positive_prompt"] == (
        "Prompt for Opening"
    )


@pytest.mark.asyncio
async def test_retry_reuses_composed_prompt_and_consistency_metadata(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.projects.update_visual_identity(
        "demo", identity={"color_palette": "warm amber"}
    )
    CharacterService(harness.projects).create_character("demo", name="Aria", character_id="aria")
    harness.add_scene("s1", "Opening")
    harness.scenes.update_scene("demo", "s1", {"character_ids": ["aria"]})
    harness.client.history_errors_remaining = 1

    failed = await harness.service.generate_scene_image(
        "demo",
        "s1",
        compose_prompt=True,
        character_references=["assets/characters/aria/ref.png"],
    )
    assert failed["status"] == "failed"
    first_text = harness.client.submissions[0]["4"]["inputs"]["text"]

    retried = await harness.service.retry_scene_image("demo", "s1")

    assert retried["status"] == "completed"
    assert harness.client.submissions[1]["4"]["inputs"]["text"] == first_text
    assert "character consistency" in " ".join(retried["warnings"]).lower()
    job = harness.service.get_job(retried["job_id"])
    assert job is not None
    assert job["options"]["compose_prompt"] is False
    assert job["options"]["character_references"] == ["assets/characters/aria/ref.png"]


def test_validate_workflow_reports_verified_workflow(tmp_path: Path) -> None:
    harness = Harness(tmp_path)

    report = harness.service.validate_workflow()

    assert report["valid"] is True
    assert report["errors"] == []


@pytest.mark.asyncio
async def test_job_registry_tracks_history(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.add_scene("s1", "Opening")
    result = await harness.service.generate_scene_image("demo", "s1")

    jobs = harness.service.list_jobs(project_id="demo", scene_id="s1")
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == result["job_id"]
    assert jobs[0]["status"] == "completed"
    assert harness.service.get_job("unknown") is None
