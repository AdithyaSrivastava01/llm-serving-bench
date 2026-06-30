from __future__ import annotations

import random
from abc import ABC, abstractmethod

from llm_bench.engines.base import GenerateRequest
from llm_bench.workloads.datasets import SyntheticDataset


class Workload(ABC):
    @abstractmethod
    def generate(self) -> list[GenerateRequest]: ...


class StandardWorkload(Workload):
    def __init__(
        self, num_requests: int = 100, max_tokens: int = 256, seed: int | None = None
    ) -> None:
        self.num_requests = num_requests
        self.max_tokens = max_tokens
        self.seed = seed

    def generate(self) -> list[GenerateRequest]:
        rng = random.Random(self.seed)
        requests = []
        for i in range(self.num_requests):
            prompt_tokens = rng.choice([50, 200, 500, 1024, 2048])
            ds = SyntheticDataset(
                num_samples=1, prompt_tokens=prompt_tokens, seed=(self.seed or 0) + i
            )
            entry = ds.load()[0]
            requests.append(
                GenerateRequest(prompt=entry.prompt, max_tokens=self.max_tokens, stream=True)
            )
        return requests


class PrefixHeavyWorkload(Workload):
    def __init__(
        self,
        num_requests: int = 100,
        max_tokens: int = 256,
        prefix_length: int = 1000,
        prefix_reuse_ratio: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.num_requests = num_requests
        self.max_tokens = max_tokens
        self.prefix_length = prefix_length
        self.prefix_reuse_ratio = prefix_reuse_ratio
        self.seed = seed

    def generate(self) -> list[GenerateRequest]:
        rng = random.Random(self.seed)
        prefix_ds = SyntheticDataset(
            num_samples=1, prompt_tokens=self.prefix_length, seed=self.seed
        )
        shared_prefix = prefix_ds.load()[0].prompt
        alt_prefix_ds = SyntheticDataset(
            num_samples=1,
            prompt_tokens=self.prefix_length,
            seed=(self.seed or 0) + 999999,
        )
        alt_prefix = alt_prefix_ds.load()[0].prompt
        requests = []
        for i in range(self.num_requests):
            use_shared = rng.random() < self.prefix_reuse_ratio
            prefix = shared_prefix if use_shared else alt_prefix
            suffix_ds = SyntheticDataset(
                num_samples=1, prompt_tokens=50, seed=(self.seed or 0) + i + 1000
            )
            suffix = suffix_ds.load()[0].prompt
            requests.append(
                GenerateRequest(
                    prompt=f"{prefix}\n\nUser query: {suffix}",
                    max_tokens=self.max_tokens,
                    stream=True,
                )
            )
        return requests


class SpeculativeWorkload(Workload):
    def __init__(
        self, num_requests: int = 100, max_tokens: int = 256, seed: int | None = None
    ) -> None:
        self.num_requests = num_requests
        self.max_tokens = max_tokens
        self.seed = seed

    def generate(self) -> list[GenerateRequest]:
        requests = []
        for i in range(self.num_requests):
            ds = SyntheticDataset(num_samples=1, prompt_tokens=200, seed=(self.seed or 0) + i)
            entry = ds.load()[0]
            requests.append(
                GenerateRequest(prompt=entry.prompt, max_tokens=self.max_tokens, stream=True)
            )
        return requests
