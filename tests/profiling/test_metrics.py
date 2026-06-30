import pytest
from llm_bench.profiling.metrics import RequestMetrics, BenchmarkResult


def test_request_metrics_itl_computation() -> None:
    m = RequestMetrics(
        request_id="r1",
        prompt_tokens=10,
        completion_tokens=3,
        start_time=0.0,
        first_token_time=0.1,
        token_times=[0.1, 0.15, 0.2],
        end_time=0.2,
    )
    assert m.ttft == pytest.approx(0.1)
    assert m.itl_values == pytest.approx([0.05, 0.05])
    assert m.e2e_latency == pytest.approx(0.2)


def test_request_metrics_single_token() -> None:
    m = RequestMetrics(
        request_id="r2",
        prompt_tokens=10,
        completion_tokens=1,
        start_time=0.0,
        first_token_time=0.05,
        token_times=[0.05],
        end_time=0.05,
    )
    assert m.ttft == pytest.approx(0.05)
    assert m.itl_values == []
    assert m.e2e_latency == pytest.approx(0.05)


def test_benchmark_result_aggregation() -> None:
    metrics = [
        RequestMetrics(
            request_id=f"r{i}",
            prompt_tokens=10,
            completion_tokens=5,
            start_time=float(i),
            first_token_time=float(i) + 0.1,
            token_times=[float(i) + 0.1 + 0.02 * j for j in range(5)],
            end_time=float(i) + 0.18,
        )
        for i in range(10)
    ]
    result = BenchmarkResult(
        config_hash="abc123",
        engine="vllm",
        model="test-model",
        workload="standard",
        requests=metrics,
    )
    assert result.total_requests == 10
    assert result.mean_ttft == pytest.approx(0.1)


def test_benchmark_result_to_dataframe() -> None:
    metrics = [
        RequestMetrics(
            request_id="r0",
            prompt_tokens=10,
            completion_tokens=3,
            start_time=0.0,
            first_token_time=0.1,
            token_times=[0.1, 0.15, 0.2],
            end_time=0.2,
        )
    ]
    result = BenchmarkResult(
        config_hash="abc123",
        engine="vllm",
        model="test-model",
        workload="standard",
        requests=metrics,
    )
    df = result.to_dataframe()
    assert len(df) == 1
    assert "ttft" in df.columns
    assert "e2e_latency" in df.columns
    assert "request_id" in df.columns
