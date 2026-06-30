from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EngineType(str, Enum):
    VLLM = "vllm"
    TRTLLM = "trtllm"
    SGLANG = "sglang"


class WorkloadType(str, Enum):
    STANDARD = "standard"
    PREFIX_HEAVY = "prefix_heavy"
    SPECULATIVE = "speculative"


class EngineConfig(BaseModel):
    engine: EngineType
    model: str
    tp_size: int = 1
    engine_params: dict[str, Any] = Field(default_factory=dict)


class WorkloadConfig(BaseModel):
    workload: WorkloadType
    num_requests: int = 100
    concurrency: int = 1
    warmup_requests: int = 10
    max_tokens: int = 256
    workload_params: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    num_repetitions: int = 3
    warmup_requests: int = 10
    enable_profiling: bool = False
    request_rate: float | None = None  # None = closed-loop, float = Poisson QPS


class RunConfig(BaseModel):
    engine: EngineConfig
    workload: WorkloadConfig
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @property
    def config_hash(self) -> str:
        data = self.model_dump(mode="json")
        raw = json.dumps(data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
