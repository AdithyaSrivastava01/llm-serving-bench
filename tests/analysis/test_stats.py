import numpy as np
import pytest

from llm_bench.analysis.stats import (
    compare_engines,
    compute_confidence_interval,
    compute_percentiles,
    compute_summary_stats,
)


def test_compute_percentiles() -> None:
    data = list(range(1, 101))
    p = compute_percentiles(data, [50, 95, 99])
    assert p[50] == pytest.approx(50.5, abs=1)
    assert p[95] == pytest.approx(95.05, abs=1)
    assert p[99] == pytest.approx(99.01, abs=1)


def test_compute_confidence_interval() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=10.0, scale=1.0, size=1000).tolist()
    lower, upper = compute_confidence_interval(data, confidence=0.95)
    assert lower < 10.0 < upper
    assert upper - lower < 0.2


def test_compute_summary_stats() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = compute_summary_stats(data)
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["median"] == pytest.approx(3.0)
    assert stats["std"] == pytest.approx(np.std(data, ddof=1), rel=0.01)
    assert "p50" in stats
    assert "p95" in stats
    assert "p99" in stats
    assert "cv" in stats


def test_compute_summary_stats_empty() -> None:
    stats = compute_summary_stats([])
    assert stats["mean"] == 0.0
    assert stats["count"] == 0


def test_compare_engines_speedup() -> None:
    baseline = [1.0, 1.1, 0.9, 1.05, 0.95]
    candidate = [0.5, 0.55, 0.45, 0.52, 0.48]
    comparison = compare_engines(baseline, candidate)
    assert comparison["speedup"] == pytest.approx(2.0, rel=0.1)
    assert comparison["is_significant"] is True
