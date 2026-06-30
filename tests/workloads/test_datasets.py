import json
from pathlib import Path

from llm_bench.workloads.datasets import DatasetEntry, ShareGPTDataset, SyntheticDataset


def test_dataset_entry_structure() -> None:
    entry = DatasetEntry(prompt="Hello", expected_output_tokens=50)
    assert entry.prompt == "Hello"
    assert entry.expected_output_tokens == 50


def test_synthetic_dataset_generates_correct_count() -> None:
    ds = SyntheticDataset(num_samples=20, prompt_tokens=100, seed=42)
    entries = ds.load()
    assert len(entries) == 20


def test_synthetic_dataset_deterministic() -> None:
    ds1 = SyntheticDataset(num_samples=5, prompt_tokens=50, seed=42)
    ds2 = SyntheticDataset(num_samples=5, prompt_tokens=50, seed=42)
    entries1 = ds1.load()
    entries2 = ds2.load()
    for e1, e2 in zip(entries1, entries2):
        assert e1.prompt == e2.prompt


def test_synthetic_dataset_different_seeds_differ() -> None:
    ds1 = SyntheticDataset(num_samples=5, prompt_tokens=50, seed=1)
    ds2 = SyntheticDataset(num_samples=5, prompt_tokens=50, seed=2)
    entries1 = ds1.load()
    entries2 = ds2.load()
    prompts1 = [e.prompt for e in entries1]
    prompts2 = [e.prompt for e in entries2]
    assert prompts1 != prompts2


def test_sharegpt_dataset_loads_from_file(tmp_path: Path) -> None:
    data = [
        {
            "conversations": [
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt", "value": "Python is a programming language."},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Explain ML."},
                {"from": "gpt", "value": "Machine learning is a subset of AI."},
            ]
        },
    ]
    f = tmp_path / "sharegpt.json"
    f.write_text(json.dumps(data))
    ds = ShareGPTDataset(path=f, num_samples=2)
    entries = ds.load()
    assert len(entries) == 2
    assert "Python" in entries[0].prompt


def test_sharegpt_dataset_truncates_to_num_samples(tmp_path: Path) -> None:
    data = [
        {
            "conversations": [
                {"from": "human", "value": f"Question {i}"},
                {"from": "gpt", "value": f"Answer {i}"},
            ]
        }
        for i in range(100)
    ]
    f = tmp_path / "sharegpt.json"
    f.write_text(json.dumps(data))
    ds = ShareGPTDataset(path=f, num_samples=10)
    entries = ds.load()
    assert len(entries) == 10
