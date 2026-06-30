import pytest
from pathlib import Path
from llm_bench.profiling.nsight import NsightProfiler


def test_nsight_profiler_builds_cmd() -> None:
    profiler = NsightProfiler(output_dir=Path("/tmp/traces"))
    cmd = profiler.build_profile_cmd(target_pid=1234, trace_name="vllm_run1")
    assert "nsys" in cmd[0]
    assert "profile" in cmd
    assert "1234" in " ".join(cmd) or "-p" in cmd


def test_nsight_profiler_output_path() -> None:
    profiler = NsightProfiler(output_dir=Path("/tmp/traces"))
    path = profiler.trace_path("vllm_run1")
    assert path == Path("/tmp/traces/vllm_run1.nsys-rep")


def test_nsight_profiler_wraps_command() -> None:
    profiler = NsightProfiler(output_dir=Path("/tmp/traces"))
    wrapped = profiler.wrap_command(
        cmd=["python", "-m", "vllm.entrypoints.openai.api_server"],
        trace_name="test_trace",
    )
    assert wrapped[0] == "nsys"
    assert "profile" in wrapped
    assert "python" in wrapped
    assert "test_trace" in " ".join(wrapped)


def test_nsight_profiler_stats_export_cmd() -> None:
    profiler = NsightProfiler(output_dir=Path("/tmp/traces"))
    cmd = profiler.build_stats_cmd(
        trace_path=Path("/tmp/traces/test.nsys-rep"), report="gpukernsum"
    )
    assert "nsys" in cmd[0]
    assert "stats" in cmd
    assert "--format" in cmd
    assert "csv" in cmd
    assert "gpukernsum" in cmd
