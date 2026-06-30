from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GenerateRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class GenerateResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    token_times: list[float] = field(default_factory=list)


class ServingEngine(ABC):
    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...
