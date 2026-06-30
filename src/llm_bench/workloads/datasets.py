from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetEntry:
    prompt: str
    expected_output_tokens: int = 256


class Dataset(ABC):
    @abstractmethod
    def load(self) -> list[DatasetEntry]: ...


class SyntheticDataset(Dataset):
    WORD_POOL = [
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "it",
        "for",
        "that",
        "was",
        "on",
        "are",
        "with",
        "as",
        "at",
        "be",
        "this",
        "have",
        "from",
        "or",
        "an",
        "by",
        "not",
        "but",
        "what",
        "all",
        "were",
        "when",
        "we",
        "there",
        "can",
        "said",
        "each",
        "which",
        "do",
        "how",
        "if",
        "will",
        "up",
        "about",
        "data",
        "model",
        "system",
        "code",
        "function",
        "process",
        "network",
        "compute",
        "memory",
        "kernel",
        "batch",
        "layer",
        "token",
        "query",
        "server",
        "request",
        "cache",
        "latency",
        "throughput",
        "inference",
    ]

    def __init__(
        self,
        num_samples: int = 100,
        prompt_tokens: int = 100,
        output_tokens: int = 256,
        seed: int | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.seed = seed

    def load(self) -> list[DatasetEntry]:
        rng = random.Random(self.seed)
        entries = []
        words_needed = int(self.prompt_tokens / 1.3)
        for _ in range(self.num_samples):
            words = [rng.choice(self.WORD_POOL) for _ in range(words_needed)]
            prompt = " ".join(words)
            entries.append(
                DatasetEntry(prompt=prompt, expected_output_tokens=self.output_tokens)
            )
        return entries


class ShareGPTDataset(Dataset):
    def __init__(self, path: Path, num_samples: int | None = None) -> None:
        self.path = path
        self.num_samples = num_samples

    def load(self) -> list[DatasetEntry]:
        with open(self.path) as f:
            data = json.load(f)
        entries = []
        for item in data:
            convs = item.get("conversations", [])
            human_turns = [c["value"] for c in convs if c["from"] == "human"]
            gpt_turns = [c["value"] for c in convs if c["from"] == "gpt"]
            if not human_turns:
                continue
            prompt = human_turns[0]
            output_tokens = len(gpt_turns[0].split()) if gpt_turns else 256
            entries.append(
                DatasetEntry(prompt=prompt, expected_output_tokens=output_tokens)
            )
        if self.num_samples is not None:
            entries = entries[: self.num_samples]
        return entries
