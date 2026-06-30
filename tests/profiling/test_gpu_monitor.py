import pytest

from llm_bench.profiling.gpu_monitor import GPUMonitor, GPUSample


def test_gpu_sample_structure() -> None:
    sample = GPUSample(
        timestamp=1.0,
        gpu_id=0,
        memory_used_mb=16000,
        memory_total_mb=81920,
        gpu_utilization_pct=85.0,
    )
    assert sample.memory_used_mb == 16000
    assert sample.memory_utilization_pct == pytest.approx(16000 / 81920 * 100, rel=0.01)


def test_gpu_monitor_init() -> None:
    monitor = GPUMonitor(poll_interval=0.5, gpu_ids=[0, 1])
    assert monitor.poll_interval == 0.5
    assert monitor.gpu_ids == [0, 1]


def test_gpu_monitor_peak_memory() -> None:
    monitor = GPUMonitor(gpu_ids=[0])
    monitor._samples = [
        GPUSample(
            timestamp=1.0,
            gpu_id=0,
            memory_used_mb=10000,
            memory_total_mb=81920,
            gpu_utilization_pct=50.0,
        ),
        GPUSample(
            timestamp=2.0,
            gpu_id=0,
            memory_used_mb=20000,
            memory_total_mb=81920,
            gpu_utilization_pct=80.0,
        ),
        GPUSample(
            timestamp=3.0,
            gpu_id=0,
            memory_used_mb=15000,
            memory_total_mb=81920,
            gpu_utilization_pct=60.0,
        ),
    ]
    assert monitor.peak_memory_mb(gpu_id=0) == 20000


def test_gpu_monitor_to_dataframe() -> None:
    monitor = GPUMonitor(gpu_ids=[0])
    monitor._samples = [
        GPUSample(
            timestamp=1.0,
            gpu_id=0,
            memory_used_mb=10000,
            memory_total_mb=81920,
            gpu_utilization_pct=50.0,
        ),
        GPUSample(
            timestamp=2.0,
            gpu_id=0,
            memory_used_mb=20000,
            memory_total_mb=81920,
            gpu_utilization_pct=80.0,
        ),
    ]
    df = monitor.to_dataframe()
    assert len(df) == 2
    assert "memory_used_mb" in df.columns
    assert "timestamp" in df.columns
