from __future__ import annotations

from typing import Any

from llm_bench.config.schema import EngineConfig
from llm_bench.engines.base import GenerateRequest, GenerateResponse, ServingEngine


class VLLMEngine(ServingEngine):
    """Engine adapter for vLLM serving backend."""

    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        self._base_url = f"http://localhost:8000"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def health_check(self) -> bool:
        return False

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        raise NotImplementedError

    def get_engine_metrics(self) -> dict[str, Any]:
        return {}
