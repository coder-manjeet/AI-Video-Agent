
import json
import logging
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

COMFYUI_URL = "http://127.0.0.1:8000"

PROJECT_DIR = Path(__file__).resolve().parent
WORKFLOW_PATH = PROJECT_DIR / "ltx_image_to_video_api.json"
logger = logging.getLogger(__name__)


def load_workflow() -> dict:
    """Load the saved LTX image-to-video API workflow."""
    with WORKFLOW_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


async def upload_image(image_path: str) -> str:
    """Upload a local image to ComfyUI and return its filename."""

    path = Path(image_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    async with httpx.AsyncClient(timeout=60) as client:
        with path.open("rb") as image_file:
            response = await client.post(
                f"{COMFYUI_URL}/upload/image",
                files={
                    "image": (
                        path.name,
                        image_file,
                        "application/octet-stream",
                    )
                },
                data={"overwrite": "true"},
            )

        response.raise_for_status()
        result = response.json()

    return result["name"]


async def generate_video(image_path: str, prompt: str) -> str:
    """Submit the LTX image-to-video workflow to ComfyUI."""

    image_name = await upload_image(image_path)

    workflow = load_workflow()

    # Set the uploaded image
    workflow["78"]["inputs"]["image"] = image_name

    # Set the positive prompt
    workflow["6"]["inputs"]["text"] = prompt

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow},
        )

        response.raise_for_status()
        result = response.json()

    return result["prompt_id"]


async def get_video_result(prompt_id: str) -> dict:
    """Get the generated video details from ComfyUI history."""

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{COMFYUI_URL}/history/{prompt_id}"
        )

        response.raise_for_status()
        history = response.json()

    job = history.get(prompt_id)

    if not job:
        return {
            "status": "pending",
            "video": None,
        }

    status = job.get("status", {})

    if not status.get("completed"):
        return {
            "status": status.get("status_str", "running"),
            "video": None,
        }

    for node_output in job.get("outputs", {}).values():
        for video in node_output.get("gifs", []):
            return {
                "status": "completed",
                "filename": video["filename"],
                "path": video.get("fullpath"),
                "duration": video.get("metadata", {}).get("duration"),
                "width": video.get("metadata", {}).get("width"),
                "height": video.get("metadata", {}).get("height"),
            }

    return {
        "status": "completed",
        "video": None,
        "message": "No video output found.",
    }


def load_workflow_file(path: str | Path) -> dict[str, Any]:
    """Load any API-format workflow file without modifying it."""
    workflow_path = Path(path).expanduser().resolve()
    if not workflow_path.is_file():
        raise FileNotFoundError(f"Workflow not found: {workflow_path}")
    with workflow_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Workflow must be a JSON object: {workflow_path}")
    return payload


async def submit_workflow(workflow: Mapping[str, Any]) -> str:
    """Submit an API-format workflow and return its ComfyUI prompt ID."""
    if not workflow:
        raise ValueError("workflow must not be empty")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": dict(workflow)},
        )
        response.raise_for_status()
        return response.json()["prompt_id"]


async def get_prompt_history(prompt_id: str) -> dict[str, Any]:
    """Read the ComfyUI history entry for a prompt ID."""
    if not prompt_id or not isinstance(prompt_id, str):
        raise ValueError("prompt_id must be a non-empty string")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
        response.raise_for_status()
        payload = response.json()
    return payload if isinstance(payload, dict) else {}


async def get_object_info() -> dict[str, Any]:
    """Read ComfyUI node definitions, including available model file lists."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{COMFYUI_URL}/object_info")
        response.raise_for_status()
        payload = response.json()
    return payload if isinstance(payload, dict) else {}


async def cancel_prompt(prompt_id: str) -> bool:
    """Remove a queued ComfyUI prompt. Returns True when the request succeeded.

    ComfyUI may report a still-running job; deletion only affects jobs that
    have not started, so callers must still verify job state after cancelling.
    """
    if not prompt_id or not isinstance(prompt_id, str):
        raise ValueError("prompt_id must be a non-empty string")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{COMFYUI_URL}/queue", json={"delete": prompt_id}
        )
        response.raise_for_status()
    logger.info("Requested cancellation of ComfyUI prompt %s", prompt_id)
    return True


async def download_output_file(
    output: Mapping[str, Any],
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Download any ComfyUI output (image or video) without overwriting."""
    filename = output.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("output entry has no filename")
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {target}")
    params = {
        "filename": filename,
        "subfolder": output.get("subfolder", ""),
        "type": output.get("type", "output"),
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(f"{COMFYUI_URL}/view", params=params)
        response.raise_for_status()
        content = response.content

    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temp.write_bytes(content)
        temp.replace(target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    logger.info("Downloaded ComfyUI output to %s", target)
    return target


async def download_output_image(
    image: Mapping[str, Any],
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Download a ComfyUI output image to *destination* without overwriting."""
    return await download_output_file(image, destination, overwrite=overwrite)
