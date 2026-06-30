import json
from pathlib import Path

import pandas as pd
import pytest

from llm_bench.analysis.export import ResultExporter


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "engine": ["vllm", "vllm", "sglang"],
            "workload": ["standard", "standard", "standard"],
            "concurrency": [1, 4, 1],
            "ttft": [0.05, 0.08, 0.04],
            "e2e_latency": [0.2, 0.35, 0.18],
            "throughput_tps": [500, 450, 520],
        }
    )


def test_export_csv(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    exporter = ResultExporter(output_dir=tmp_path)
    path = exporter.to_csv(sample_df, "results")
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 3


def test_export_json(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    exporter = ResultExporter(output_dir=tmp_path)
    path = exporter.to_json(sample_df, "results")
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert len(data) == 3


def test_export_parquet(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    exporter = ResultExporter(output_dir=tmp_path)
    path = exporter.to_parquet(sample_df, "results")
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == 3


def test_export_summary_table(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    exporter = ResultExporter(output_dir=tmp_path)
    path = exporter.summary_table(sample_df, "summary")
    assert path.exists()
    loaded = pd.read_csv(path)
    assert "engine" in loaded.columns
