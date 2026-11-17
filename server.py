
import os

import httpx
from mcp.server.fastmcp import FastMCP

from comfyui_client import generate_video, get_video_result

COMFYUI_URL = os.getenv(
    "COMFYUI_URL",
    "http://127.0.0.1:8000"
)

mcp = FastMCP("Local Video Agent")


@mcp.tool()
async def comfyui_status() -> dict:
    """
    Check whether the local ComfyUI server is running.
    """

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{COMFYUI_URL}/system_stats"
        )

        response.raise_for_status()

        data = response.json()

        return {
            "status": "connected",
            "comfyui_version": data["system"].get("comfyui_version"),
            "python_version": data["system"].get("python_version"),
            "devices": [
                device.get("name")
                for device in data.get("devices", [])
            ]
        }
    
@mcp.tool()
async def create_video(image_path: str, prompt: str) -> dict:
    """Generate a video from an image using the local LTX workflow."""

    prompt_id = await generate_video(image_path, prompt)

    return {
        "status": "submitted",
        "prompt_id": prompt_id,
        "message": "Video generation submitted to ComfyUI."
    }


@mcp.tool()
async def check_video_status(prompt_id: str) -> dict:
    """Check video generation status and retrieve output details."""

    return await get_video_result(prompt_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
