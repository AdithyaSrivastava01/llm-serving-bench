from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


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
    def ttft(self) -> float:
        return self.first_token_time - self.start_time

    @property
    def itl_values(self) -> list[float]:
        if len(self.token_times) < 2:
            return []
        return [
            self.token_times[i] - self.token_times[i - 1] for i in range(1, len(self.token_times))
        ]

    @property
    def e2e_latency(self) -> float:
        return self.end_time - self.start_time


@dataclass
class BenchmarkResult:
    config_hash: str
    engine: str
    model: str
    workload: str
    requests: list[RequestMetrics] = field(default_factory=list)
    engine_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def total_requests(self) -> int:
        return len(self.requests)

    @property
    def successful_requests(self) -> list[RequestMetrics]:
        return [r for r in self.requests if r.error is None]

    @property
    def mean_ttft(self) -> float:
        ttfts = [r.ttft for r in self.successful_requests]
        return sum(ttfts) / len(ttfts) if ttfts else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.requests:
            rows.append(
                {
                    "request_id": r.request_id,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "ttft": r.ttft,
                    "e2e_latency": r.e2e_latency,
                    "itl_mean": (sum(r.itl_values) / len(r.itl_values) if r.itl_values else 0.0),
                    "itl_values": r.itl_values,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "error": r.error,
                }
            )
        return pd.DataFrame(rows)
