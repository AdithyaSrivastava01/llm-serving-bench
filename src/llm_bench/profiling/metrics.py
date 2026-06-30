from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    start_time: float
    first_token_time: float
    token_times: list[float]
    end_time: float
    error: str | None = None

    @property
    def latency(self) -> float:
        return self.end_time - self.start_time

    @property
    def ttft(self) -> float:
        """Time to first token."""
        return self.first_token_time - self.start_time

    @property
    def throughput(self) -> float:
        """Completion tokens per second."""
        duration = self.latency
        if duration <= 0:
            return 0.0
        return self.completion_tokens / duration
