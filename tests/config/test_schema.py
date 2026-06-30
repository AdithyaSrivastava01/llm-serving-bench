import pytest
from llm_bench.config.schema import (
    EngineType,
    WorkloadType,
    EngineConfig,
    WorkloadConfig,
    BenchmarkConfig,
    RunConfig,
)


def test_engine_type_enum() -> None:
    assert EngineType.VLLM.value == "vllm"
    assert EngineType.TRTLLM.value == "trtllm"
    assert EngineType.SGLANG.value == "sglang"


def test_workload_type_enum() -> None:
    assert WorkloadType.STANDARD.value == "standard"
    assert WorkloadType.PREFIX_HEAVY.value == "prefix_heavy"
    assert WorkloadType.SPECULATIVE.value == "speculative"


def test_engine_config_defaults() -> None:
    cfg = EngineConfig(engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct")
    assert cfg.tp_size == 1
    assert cfg.engine_params == {}


def test_engine_config_with_params() -> None:
    cfg = EngineConfig(
        engine=EngineType.VLLM,
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_size=2,
        engine_params={"enable_prefix_caching": True, "gpu_memory_utilization": 0.9},
    )
    assert cfg.tp_size == 2
    assert cfg.engine_params["enable_prefix_caching"] is True


def test_workload_config_defaults() -> None:
    cfg = WorkloadConfig(workload=WorkloadType.STANDARD)
    assert cfg.num_requests == 100
    assert cfg.concurrency == 1
    assert cfg.warmup_requests == 10


def test_workload_config_prefix_heavy() -> None:
    cfg = WorkloadConfig(
        workload=WorkloadType.PREFIX_HEAVY,
        concurrency=16,
        num_requests=200,
        workload_params={"prefix_length": 1000, "prefix_reuse_ratio": 0.8},
    )
    assert cfg.workload_params["prefix_length"] == 1000


def test_benchmark_config_validates() -> None:
    cfg = BenchmarkConfig(
        num_repetitions=3,
        warmup_requests=10,
        enable_profiling=False,
    )
    assert cfg.num_repetitions == 3


def test_run_config_composes() -> None:
    engine_cfg = EngineConfig(
        engine=EngineType.SGLANG, model="meta-llama/Llama-3.1-8B-Instruct"
    )
    workload_cfg = WorkloadConfig(workload=WorkloadType.STANDARD, concurrency=4)
    bench_cfg = BenchmarkConfig()
    run = RunConfig(engine=engine_cfg, workload=workload_cfg, benchmark=bench_cfg)
    assert run.engine.engine == EngineType.SGLANG
    assert run.workload.concurrency == 4


def test_run_config_hash_deterministic() -> None:
    run1 = RunConfig(
        engine=EngineConfig(
            engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct"
        ),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD),
        benchmark=BenchmarkConfig(),
    )
    run2 = RunConfig(
        engine=EngineConfig(
            engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct"
        ),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD),
        benchmark=BenchmarkConfig(),
    )
    assert run1.config_hash == run2.config_hash


def test_run_config_hash_differs_on_change() -> None:
    run1 = RunConfig(
        engine=EngineConfig(
            engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct"
        ),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD, concurrency=1),
        benchmark=BenchmarkConfig(),
    )
    run2 = RunConfig(
        engine=EngineConfig(
            engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct"
        ),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD, concurrency=16),
        benchmark=BenchmarkConfig(),
    )
    assert run1.config_hash != run2.config_hash
