from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp
import docker

from llm_bench.config.schema import EngineConfig
from llm_bench.engines.base import GenerateRequest, GenerateResponse, ServingEngine

logger = logging.getLogger(__name__)
VLLM_IMAGE = "vllm/vllm-openai:latest"
VLLM_PORT = 8000


class VLLMEngine(ServingEngine):
    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        self._container: Any = None
        self._port = VLLM_PORT

    def _build_launch_cmd(self) -> list[str]:
        cmd = [
            "--model",
            self.config.model,
            "--tensor-parallel-size",
            str(self.config.tp_size),
            "--port",
            str(self._port),
        ]
        param_map = {
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "max_num_seqs": "--max-num-seqs",
            "enable_prefix_caching": "--enable-prefix-caching",
            "speculative_model": "--speculative-model",
        }
        for key, flag in param_map.items():
            value = self.config.engine_params.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])
        return cmd

    def _build_container_config(self) -> dict[str, Any]:
        cmd = self._build_launch_cmd()
        gpus = ",".join(str(i) for i in range(self.config.tp_size))
        return {
            "image": VLLM_IMAGE,
            "command": cmd,
            "runtime": "nvidia",
            "environment": {"NVIDIA_VISIBLE_DEVICES": gpus},
            "ports": {f"{self._port}/tcp": self._port},
            "detach": True,
            "auto_remove": True,
            "shm_size": "4g",
        }

    @property
    def _health_url(self) -> str:
        return f"{self._base_url}/health"

    @property
    def _metrics_url(self) -> str:
        return f"{self._base_url}/metrics"

    async def start(self) -> None:
        logger.info(
            "Starting vLLM container for %s (TP=%d)",
            self.config.model,
            self.config.tp_size,
        )
        client = docker.from_env()
        container_cfg = self._build_container_config()
        self._container = client.containers.run(**container_cfg)
        self._base_url = f"http://localhost:{self._port}"
        for attempt in range(120):
            if await self.health_check():
                logger.info("vLLM engine ready after %d seconds", attempt)
                return
            await asyncio.sleep(1)
        raise TimeoutError("vLLM engine failed to start within 120 seconds")

    async def stop(self) -> None:
        if self._container:
            logger.info("Stopping vLLM container")
            self._container.stop(timeout=10)
            self._container = None

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._health_url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        payload = {
            "model": self.config.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        token_times: list[float] = []
        full_text = ""
        async with aiohttp.ClientSession() as session:
            if request.stream:
                async with session.post(
                    f"{self._base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    async for line in resp.content:
                        decoded = line.decode("utf-8").strip()
                        if not decoded.startswith("data: "):
                            continue
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        token_times.append(time.perf_counter())
                        full_text += data["choices"][0].get("text", "")
            else:
                async with session.post(
                    f"{self._base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    data = await resp.json()
                    token_times.append(time.perf_counter())
                    full_text = data["choices"][0]["text"]
        return GenerateResponse(
            text=full_text,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(token_times),
            token_times=token_times,
        )

    def get_engine_metrics(self) -> dict[str, Any]:
        import urllib.request

        try:
            with urllib.request.urlopen(self._metrics_url, timeout=5) as resp:
                text = resp.read().decode()
            metrics: dict[str, Any] = {}
            for line in text.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    metrics[parts[0]] = float(parts[1])
            return metrics
        except Exception as e:
            logger.warning("Failed to scrape vLLM metrics: %s", e)
            return {}
