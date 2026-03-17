"""Async ComfyUI API client with WebSocket progress tracking."""

from __future__ import annotations

import asyncio
import json
import uuid
import logging
from pathlib import Path
from typing import Optional

import aiohttp

from .config import ComfyUIConfig

logger = logging.getLogger(__name__)


class ComfyUIError(Exception):
    """Error from ComfyUI API."""
    pass


class ComfyUIClient:
    """Async client for ComfyUI HTTP + WebSocket API."""

    def __init__(self, config: ComfyUIConfig):
        self.config = config
        self.client_id = str(uuid.uuid4())
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    async def __aenter__(self) -> ComfyUIClient:
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        ws_url = f"{self.config.ws_url}?clientId={self.client_id}"
        self._ws = await self._session.ws_connect(ws_url)
        logger.info(f"Connected to ComfyUI at {self.config.http_url}")

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    async def queue_prompt(self, workflow: dict, prompt_id: Optional[str] = None) -> str:
        """Submit a workflow to ComfyUI. Returns prompt_id."""
        if prompt_id is None:
            prompt_id = str(uuid.uuid4())

        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
            "prompt_id": prompt_id,
        }

        async with self._session.post(
            f"{self.config.http_url}/prompt",
            json=payload,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise ComfyUIError(f"Failed to queue prompt: {resp.status} {error_text}")
            result = await resp.json()

        if result.get("node_errors"):
            raise ComfyUIError(f"Workflow validation errors: {result['node_errors']}")

        logger.info(f"Queued prompt {prompt_id}")
        return prompt_id

    async def wait_for_completion(self, prompt_id: str, timeout: int = 600) -> dict:
        """Wait for a specific prompt to finish via WebSocket. Returns history."""
        try:
            async with asyncio.timeout(timeout):
                while True:
                    msg = await self._ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("type") == "executing":
                            exec_data = data["data"]
                            if exec_data.get("prompt_id") == prompt_id:
                                if exec_data.get("node") is None:
                                    break  # Execution done
                                else:
                                    logger.debug(f"Executing node: {exec_data['node']}")
                        elif data.get("type") == "execution_error":
                            if data["data"].get("prompt_id") == prompt_id:
                                raise ComfyUIError(
                                    f"Execution error: {data['data'].get('exception_message', 'unknown')}"
                                )
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        raise ComfyUIError("WebSocket connection lost")
        except TimeoutError:
            raise ComfyUIError(f"Prompt {prompt_id} timed out after {timeout}s")

        return await self.get_history(prompt_id)

    async def get_history(self, prompt_id: str) -> dict:
        """Get execution history for a prompt."""
        async with self._session.get(
            f"{self.config.http_url}/history/{prompt_id}"
        ) as resp:
            data = await resp.json()
        return data.get(prompt_id, {})

    async def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download an output image."""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        async with self._session.get(
            f"{self.config.http_url}/view",
            params=params,
        ) as resp:
            if resp.status != 200:
                raise ComfyUIError(f"Failed to get image {filename}: {resp.status}")
            return await resp.read()

    async def download_outputs(self, history: dict, dest_dir: Path) -> list[Path]:
        """Download all output images from a completed prompt to dest_dir."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []

        outputs = history.get("outputs", {})
        for node_id, node_output in outputs.items():
            if "images" not in node_output:
                continue
            for image_info in node_output["images"]:
                image_data = await self.get_image(
                    image_info["filename"],
                    image_info.get("subfolder", ""),
                    image_info.get("type", "output"),
                )
                dest_path = dest_dir / image_info["filename"]
                dest_path.write_bytes(image_data)
                downloaded.append(dest_path)
                logger.debug(f"Downloaded {dest_path}")

        return downloaded

    async def generate_and_download(
        self, workflow: dict, dest_dir: Path, timeout: int = 600
    ) -> list[Path]:
        """Queue a workflow, wait for completion, download outputs. All-in-one."""
        prompt_id = await self.queue_prompt(workflow)
        history = await self.wait_for_completion(prompt_id, timeout=timeout)

        status = history.get("status", {})
        if status.get("status_str") == "error":
            raise ComfyUIError(f"Generation failed: {status.get('messages', [])}")

        return await self.download_outputs(history, dest_dir)
