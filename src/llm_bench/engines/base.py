from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from llm_bench.config.schema import EngineConfig


@dataclass
class GenerateRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    stream: bool = True


@dataclass
class GenerateResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    token_times: list[float]
    finish_reason: str = "stop"


class ServingEngine(ABC):
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._base_url: str = ""

    @abstractmethod
    async def start(self) -> None: ...
    @abstractmethod
    async def stop(self) -> None: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...
    @abstractmethod
    def get_engine_metrics(self) -> dict[str, Any]: ...

    @property
    def base_url(self) -> str:
        return self._base_url
