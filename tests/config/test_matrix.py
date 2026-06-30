from pathlib import Path

from llm_bench.config.matrix import MatrixExpander, MatrixFilter
from llm_bench.config.schema import EngineType, RunConfig, WorkloadType


def test_load_matrix_from_yaml(tmp_path: Path) -> None:
    yaml_content = """\
matrix:
  engines: [vllm]
  models: [meta-llama/Llama-3.1-8B-Instruct]
  workloads: [standard]
  tp_size: [1]
  concurrency: [1, 4]
  engine_params:
    vllm: {}
"""
    f = tmp_path / "matrix.yaml"
    f.write_text(yaml_content)
    expander = MatrixExpander.from_yaml(f)
    configs = expander.expand()
    assert len(configs) == 2
    assert all(isinstance(c, RunConfig) for c in configs)


def test_cartesian_product_expansion(tmp_path: Path) -> None:
    yaml_content = """\
matrix:
  engines: [vllm, sglang]
  models: [meta-llama/Llama-3.1-8B-Instruct]
  workloads: [standard]
  tp_size: [1]
  concurrency: [1, 4]
  engine_params:
    vllm: {}
    sglang: {}
"""
    f = tmp_path / "matrix.yaml"
    f.write_text(yaml_content)
    expander = MatrixExpander.from_yaml(f)
    configs = expander.expand()
    assert len(configs) == 4


def test_filter_skips_speculative_on_sglang(tmp_path: Path) -> None:
    yaml_content = """\
matrix:
  engines: [vllm, sglang]
  models: [meta-llama/Llama-3.1-8B-Instruct]
  workloads: [speculative]
  tp_size: [1]
  concurrency: [1]
  engine_params:
    vllm: {}
    sglang: {}
"""
    f = tmp_path / "matrix.yaml"
    f.write_text(yaml_content)
    expander = MatrixExpander.from_yaml(f)
    configs = expander.expand()
    engines = {c.engine.engine for c in configs}
    assert EngineType.SGLANG not in engines
    assert EngineType.VLLM in engines


def test_filter_skips_tp2_when_max_gpus_1() -> None:
    filt = MatrixFilter(num_gpus=1)
    from llm_bench.config.schema import BenchmarkConfig, EngineConfig, WorkloadConfig

    run = RunConfig(
        engine=EngineConfig(
            engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct", tp_size=2
        ),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD),
        benchmark=BenchmarkConfig(),
    )
    assert filt.should_skip(run) is not None


def test_filter_allows_tp1_when_max_gpus_1() -> None:
    filt = MatrixFilter(num_gpus=1)
    from llm_bench.config.schema import BenchmarkConfig, EngineConfig, WorkloadConfig

    run = RunConfig(
        engine=EngineConfig(
            engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct", tp_size=1
        ),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD),
        benchmark=BenchmarkConfig(),
    )
    assert filt.should_skip(run) is None


def test_engine_params_applied(tmp_path: Path) -> None:
    yaml_content = """\
matrix:
  engines: [vllm]
  models: [meta-llama/Llama-3.1-8B-Instruct]
  workloads: [standard]
  tp_size: [1]
  concurrency: [1]
  engine_params:
    vllm:
      enable_prefix_caching: [true, false]
"""
    f = tmp_path / "matrix.yaml"
    f.write_text(yaml_content)
    expander = MatrixExpander.from_yaml(f)
    configs = expander.expand()
    assert len(configs) == 2
    caching_values = {c.engine.engine_params.get("enable_prefix_caching") for c in configs}
    assert caching_values == {True, False}


def test_skipped_configs_logged(tmp_path: Path) -> None:
    yaml_content = """\
matrix:
  engines: [sglang]
  models: [meta-llama/Llama-3.1-8B-Instruct]
  workloads: [speculative]
  tp_size: [1]
  concurrency: [1]
  engine_params:
    sglang: {}
"""
    f = tmp_path / "matrix.yaml"
    f.write_text(yaml_content)
    expander = MatrixExpander.from_yaml(f)
    configs = expander.expand()
    assert len(configs) == 0
    assert len(expander.skipped) == 1
    assert "speculative" in expander.skipped[0].lower()
