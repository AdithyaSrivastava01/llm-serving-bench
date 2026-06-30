from pathlib import Path

import pytest

from llm_bench.config.schema import (
    BenchmarkConfig,
    EngineConfig,
    EngineType,
    RunConfig,
    WorkloadConfig,
    WorkloadType,
)
from llm_bench.runner.orchestrator import BenchmarkOrchestrator


@pytest.fixture
def run_config() -> RunConfig:
    return RunConfig(
        engine=EngineConfig(engine=EngineType.VLLM, model="test-model"),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD, num_requests=5),
        benchmark=BenchmarkConfig(num_repetitions=1, enable_profiling=False),
    )


@pytest.fixture
def orchestrator(tmp_path: Path) -> BenchmarkOrchestrator:
    return BenchmarkOrchestrator(results_dir=tmp_path / "results")


def test_orchestrator_creates_results_dir(tmp_path: Path) -> None:
    orch = BenchmarkOrchestrator(results_dir=tmp_path / "results")
    assert orch.results_dir == tmp_path / "results"


def test_orchestrator_run_dir_structure(
    orchestrator: BenchmarkOrchestrator, run_config: RunConfig
) -> None:
    run_dir = orchestrator._run_dir(run_config)
    assert "vllm" in str(run_dir)
    assert run_config.config_hash in str(run_dir)


def test_orchestrator_checks_existing_results(
    orchestrator: BenchmarkOrchestrator, run_config: RunConfig
) -> None:
    assert orchestrator._has_completed_results(run_config) is False
    run_dir = orchestrator._run_dir(run_config)
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.parquet").touch()
    assert orchestrator._has_completed_results(run_config) is True


def test_orchestrator_saves_config(
    orchestrator: BenchmarkOrchestrator, run_config: RunConfig
) -> None:
    run_dir = orchestrator._run_dir(run_config)
    run_dir.mkdir(parents=True)
    orchestrator._save_run_config(run_config, run_dir)
    assert (run_dir / "config.json").exists()
