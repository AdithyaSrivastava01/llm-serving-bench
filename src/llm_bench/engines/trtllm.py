from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp

import docker
from llm_bench.config.schema import EngineConfig, LaunchMode
from llm_bench.engines.base import GenerateRequest, GenerateResponse, ServingEngine

logger = logging.getLogger(__name__)
TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3"
TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
ENGINE_CACHE_BASE = Path.home() / ".cache" / "llm-bench" / "trtllm-engines"


class TRTLLMEngine(ServingEngine):
    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        self._container: Any = None
        self._process: asyncio.subprocess.Process | None = None
        self._port = TRITON_HTTP_PORT

    @property
    def _engine_cache_dir(self) -> Path:
        model_name = self.config.model.split("/")[-1]
        return ENGINE_CACHE_BASE / f"{model_name}-tp{self.config.tp_size}"

    def _build_engine_cmd(self) -> list[str]:
        cmd = [
            "trtllm-build",
            "--model_dir",
            f"/models/{self.config.model.split('/')[-1]}",
            "--tp_size",
            str(self.config.tp_size),
            "--max_batch_size",
            str(self.config.engine_params.get("max_batch_size", 64)),
            "--max_input_len",
            "2048",
            "--max_seq_len",
            "4096",
        ]
        kv_frac = self.config.engine_params.get("kv_cache_free_gpu_mem_fraction")
        if kv_frac is not None:
            cmd.extend(["--kv_cache_free_gpu_mem_fraction", str(kv_frac)])
        decoding = self.config.engine_params.get("decoding_mode", "auto")
        if decoding == "medusa":
            cmd.extend(["--speculative_decoding_mode", "medusa"])
        elif self.config.engine_params.get("enable_chunked_context"):
            cmd.append("--enable_chunked_context")
        return cmd

    def _build_container_config(self) -> dict[str, Any]:
        gpus = ",".join(str(i) for i in range(self.config.tp_size))
        return {
            "image": TRITON_IMAGE,
            "command": [
                "tritonserver",
                "--model-repository=/model_repo",
                "--http-port",
                str(self._port),
                "--grpc-port",
                str(TRITON_GRPC_PORT),
            ],
            "runtime": "nvidia",
            "environment": {"NVIDIA_VISIBLE_DEVICES": gpus},
            "ports": {
                f"{self._port}/tcp": self._port,
                f"{TRITON_GRPC_PORT}/tcp": TRITON_GRPC_PORT,
            },
            "volumes": {str(self._engine_cache_dir): {"bind": "/engine", "mode": "ro"}},
            "detach": True,
            "auto_remove": True,
            "shm_size": "4g",
        }

    @property
    def _health_url(self) -> str:
        return f"{self._base_url}/v2/health/ready"

    async def start(self) -> None:
        logger.info(
            "Starting TRT-LLM/Triton (%s) for %s (TP=%d)",
            self.config.launch_mode.value,
            self.config.model,
            self.config.tp_size,
        )
        if not self._engine_cache_dir.exists():
            logger.warning("No pre-built TRT engine at %s", self._engine_cache_dir)
        self._base_url = f"http://localhost:{self._port}"

        if self.config.launch_mode == LaunchMode.DOCKER:
            client = docker.from_env()
            self._container = client.containers.run(**self._build_container_config())
        else:
            cmd = [
                "tritonserver",
                f"--model-repository={self._engine_cache_dir}",
                f"--http-port={self._port}",
                f"--grpc-port={TRITON_GRPC_PORT}",
            ]
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        timeout = 180
        for attempt in range(timeout):
            if await self.health_check():
                logger.info("TRT-LLM ready after %d seconds", attempt)
                return
            await asyncio.sleep(1)
        raise TimeoutError(f"TRT-LLM failed to start within {timeout} seconds")

    async def stop(self) -> None:
        if self._container:
            self._container.stop(timeout=10)
            self._container = None
        if self._process:
            self._process.terminate()
            await self._process.wait()
            self._process = None

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
            "text_input": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        token_times: list[float] = []
        full_text = ""
        async with aiohttp.ClientSession() as session:
            if request.stream:
                async with session.post(
                    f"{self._base_url}/v2/models/ensemble/generate_stream",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    async for line in resp.content:
                        decoded = line.decode("utf-8").strip()
                        if not decoded.startswith("data:"):
                            continue
                        data_str = decoded[5:].strip()
                        if not data_str:
                            continue
                        data = json.loads(data_str)
                        token_times.append(time.perf_counter())
                        full_text += data.get("text_output", "")
            else:
                async with session.post(
                    f"{self._base_url}/v2/models/ensemble/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    data = await resp.json()
                    token_times.append(time.perf_counter())
                    full_text = data.get("text_output", "")
        return GenerateResponse(
            text=full_text,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(token_times),
            token_times=token_times,
        )

    def get_engine_metrics(self) -> dict[str, Any]:
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self._base_url}/metrics", timeout=5) as resp:
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
            logger.warning("Failed to scrape TRT-LLM metrics: %s", e)
            return {}
