from llm_bench.config.schema import WorkloadConfig, WorkloadType
from llm_bench.engines.base import GenerateRequest
from llm_bench.workloads.generator import WorkloadGenerator
from llm_bench.workloads.patterns import (
    PrefixHeavyWorkload,
    SpeculativeWorkload,
    StandardWorkload,
)


def test_standard_workload_generates_requests() -> None:
    wl = StandardWorkload(num_requests=10, max_tokens=128, seed=42)
    requests = wl.generate()
    assert len(requests) == 10
    assert all(isinstance(r, GenerateRequest) for r in requests)
    assert all(r.max_tokens == 128 for r in requests)
    assert all(r.stream is True for r in requests)


def test_prefix_heavy_workload_shares_prefix() -> None:
    wl = PrefixHeavyWorkload(
        num_requests=5,
        max_tokens=128,
        prefix_length=500,
        prefix_reuse_ratio=1.0,
        seed=42,
    )
    requests = wl.generate()
    assert len(requests) == 5
    prefix = requests[0].prompt[:200]
    for r in requests:
        assert r.prompt.startswith(prefix)


def test_prefix_heavy_workload_partial_reuse() -> None:
    wl = PrefixHeavyWorkload(
        num_requests=10,
        max_tokens=128,
        prefix_length=500,
        prefix_reuse_ratio=0.5,
        seed=42,
    )
    requests = wl.generate()
    assert len(requests) == 10
    prefix = requests[0].prompt[:200]
    shared = sum(1 for r in requests if r.prompt.startswith(prefix))
    assert 3 <= shared <= 8


def test_speculative_workload_generates_requests() -> None:
    wl = SpeculativeWorkload(num_requests=10, max_tokens=256, seed=42)
    requests = wl.generate()
    assert len(requests) == 10
    assert all(r.max_tokens == 256 for r in requests)


def test_generator_creates_correct_workload() -> None:
    cfg = WorkloadConfig(workload=WorkloadType.STANDARD, num_requests=5, max_tokens=64)
    gen = WorkloadGenerator(cfg, seed=42)
    requests = gen.generate()
    assert len(requests) == 5


def test_generator_prefix_heavy_passes_params() -> None:
    cfg = WorkloadConfig(
        workload=WorkloadType.PREFIX_HEAVY,
        num_requests=5,
        max_tokens=64,
        workload_params={"prefix_length": 1000, "prefix_reuse_ratio": 0.8},
    )
    gen = WorkloadGenerator(cfg, seed=42)
    requests = gen.generate()
    assert len(requests) == 5
