
import json
from pathlib import Path

import httpx

COMFYUI_URL = "http://127.0.0.1:8000"

PROJECT_DIR = Path(__file__).resolve().parent
WORKFLOW_PATH = PROJECT_DIR / "ltx_image_to_video_api.json"


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