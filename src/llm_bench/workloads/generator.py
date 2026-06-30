from __future__ import annotations

from llm_bench.config.schema import WorkloadConfig, WorkloadType
from llm_bench.engines.base import GenerateRequest
from llm_bench.workloads.patterns import (
    PrefixHeavyWorkload,
    SpeculativeWorkload,
    StandardWorkload,
)


class WorkloadGenerator:
    def __init__(self, config: WorkloadConfig, seed: int | None = None) -> None:
        self.config = config
        self.seed = seed

    def generate(self) -> list[GenerateRequest]:
        match self.config.workload:
            case WorkloadType.STANDARD:
                wl = StandardWorkload(
                    num_requests=self.config.num_requests,
                    max_tokens=self.config.max_tokens,
                    seed=self.seed,
                )
            case WorkloadType.PREFIX_HEAVY:
                params = self.config.workload_params
                wl = PrefixHeavyWorkload(
                    num_requests=self.config.num_requests,
                    max_tokens=self.config.max_tokens,
                    prefix_length=params.get("prefix_length", 1000),
                    prefix_reuse_ratio=params.get("prefix_reuse_ratio", 1.0),
                    seed=self.seed,
                )
            case WorkloadType.SPECULATIVE:
                wl = SpeculativeWorkload(
                    num_requests=self.config.num_requests,
                    max_tokens=self.config.max_tokens,
                    seed=self.seed,
                )
        return wl.generate()
