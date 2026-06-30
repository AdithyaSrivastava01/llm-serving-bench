from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_bench.analysis.plots import BenchmarkPlotter


@pytest.fixture
def sample_results_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for engine in ["vllm", "trtllm", "sglang"]:
        for conc in [1, 4, 16, 32]:
            for _ in range(20):
                rows.append(
                    {
                        "engine": engine,
                        "concurrency": conc,
                        "ttft": rng.exponential(0.05) + 0.01 * conc,
                        "itl_mean": rng.exponential(0.01) + 0.001 * conc,
                        "e2e_latency": rng.exponential(0.2) + 0.05 * conc,
                        "throughput_tps": rng.normal(500 - 5 * conc, 50),
                        "workload": "standard",
                    }
                )
    return pd.DataFrame(rows)


def test_plotter_creates_output_dir(tmp_path: Path, sample_results_df: pd.DataFrame) -> None:
    plotter = BenchmarkPlotter(output_dir=tmp_path / "plots")
    plotter.plot_ttft_vs_concurrency(sample_results_df)
    assert (tmp_path / "plots").exists()


def test_plot_ttft_vs_concurrency(tmp_path: Path, sample_results_df: pd.DataFrame) -> None:
    plotter = BenchmarkPlotter(output_dir=tmp_path / "plots")
    path = plotter.plot_ttft_vs_concurrency(sample_results_df)
    assert path.exists()
    assert path.suffix == ".png"


def test_plot_itl_distribution(tmp_path: Path, sample_results_df: pd.DataFrame) -> None:
    plotter = BenchmarkPlotter(output_dir=tmp_path / "plots")
    path = plotter.plot_itl_distribution(sample_results_df)
    assert path.exists()


def test_plot_throughput_bars(tmp_path: Path, sample_results_df: pd.DataFrame) -> None:
    plotter = BenchmarkPlotter(output_dir=tmp_path / "plots")
    path = plotter.plot_throughput_vs_concurrency(sample_results_df)
    assert path.exists()


def test_plot_latency_heatmap(tmp_path: Path, sample_results_df: pd.DataFrame) -> None:
    plotter = BenchmarkPlotter(output_dir=tmp_path / "plots")
    path = plotter.plot_latency_heatmap(sample_results_df)
    assert path.exists()
