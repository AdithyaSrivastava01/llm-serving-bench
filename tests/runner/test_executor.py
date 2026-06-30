import pytest

from llm_bench.runner.executor import RequestExecutor, poisson_arrival_times


def test_executor_init() -> None:
    executor = RequestExecutor(concurrency=2)
    assert executor.concurrency == 2


@pytest.mark.asyncio
async def test_poisson_arrival_spacing() -> None:
    times = poisson_arrival_times(rate=10.0, n=100, seed=42)
    assert len(times) == 100
    assert all(t >= 0 for t in times)
    mean_gap = sum(times) / len(times)
    assert 0.05 < mean_gap < 0.2
