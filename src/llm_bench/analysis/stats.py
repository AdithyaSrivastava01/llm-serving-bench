from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def compute_percentiles(
    data: list[float], percentiles: list[int] | None = None
) -> dict[int, float]:
    if not data:
        return {}
    if percentiles is None:
        percentiles = [50, 95, 99]
    arr = np.array(data)
    return {p: float(np.percentile(arr, p)) for p in percentiles}


def compute_confidence_interval(
    data: list[float], confidence: float = 0.95, n_bootstrap: int = 10000
) -> tuple[float, float]:
    if len(data) < 2:
        return (data[0], data[0]) if data else (0.0, 0.0)
    rng = np.random.default_rng(42)
    arr = np.array(data)
    boot_means = np.array(
        [
            rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_bootstrap)
        ]
    )
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return lower, upper


def compute_summary_stats(data: list[float]) -> dict[str, float]:
    if not data:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "cv": 0.0,
            "count": 0,
        }
    arr = np.array(data)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "median": float(np.median(arr)),
        "std": std,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "cv": std / mean if mean != 0 else 0.0,
        "count": len(data),
    }


def compare_engines(
    baseline: list[float], candidate: list[float], alpha: float = 0.05
) -> dict[str, float | bool]:
    baseline_mean = np.mean(baseline)
    candidate_mean = np.mean(candidate)
    speedup = float(baseline_mean / candidate_mean) if candidate_mean > 0 else 0.0
    if len(baseline) >= 2 and len(candidate) >= 2:
        t_stat, p_value = scipy_stats.ttest_ind(baseline, candidate, equal_var=False)
        is_significant = bool(p_value < alpha)
    else:
        p_value = 1.0
        is_significant = False
    return {
        "speedup": speedup,
        "baseline_mean": float(baseline_mean),
        "candidate_mean": float(candidate_mean),
        "p_value": float(p_value),
        "is_significant": is_significant,
    }
