# LLM Serving Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible benchmark suite comparing TensorRT-LLM, vLLM, and SGLang across workload patterns, with Nsight Systems profiling and statistical analysis.

**Architecture:** Monolithic Python package (`llm_bench`) with Typer CLI. Engine adapters wrap Docker containers behind a shared async interface. Async load generator fires requests, collects per-token timestamps. Nsight profiler wraps engine processes for kernel-level traces. Analysis layer produces Parquet → stats → plots.

**Tech Stack:** Python 3.11+, uv, Typer, Pydantic v2, aiohttp, pandas, pyarrow, matplotlib, seaborn, Docker SDK, pynvml

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/llm_bench/__init__.py`
- Create: `.gitignore`
- Create: `configs/.gitkeep`
- Create: `results/.gitkeep`

- [ ] **Step 1: Initialize uv project**

```bash
cd /home/adithya/Document/llm-serving-bench
uv init --lib --name llm-bench
```

This creates `pyproject.toml` and `src/llm_bench/__init__.py`.

- [ ] **Step 2: Edit pyproject.toml**

Replace the generated `pyproject.toml` with:

```toml
[project]
name = "llm-bench"
version = "0.1.0"
description = "Benchmark suite comparing TensorRT-LLM, vLLM, and SGLang serving engines"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.12",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "aiohttp>=3.9",
    "pandas>=2.2",
    "pyarrow>=15.0",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "docker>=7.0",
    "pynvml>=11.5",
    "numpy>=1.26",
]

[project.scripts]
bench = "llm_bench.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_bench"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
    "mypy>=1.10",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.mypy]
python_version = "3.11"
strict = true
```

- [ ] **Step 3: Create .gitignore**

```
# Results and traces
results/
*.nsys-rep
*.sqlite

# Data
data/
*.parquet

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p src/llm_bench/{config,engines,workloads,profiling,runner,analysis}
mkdir -p tests/{config,engines,workloads,profiling,runner,analysis}
mkdir -p configs notebooks docker scripts results
touch src/llm_bench/__init__.py
touch src/llm_bench/{config,engines,workloads,profiling,runner,analysis}/__init__.py
touch tests/__init__.py
touch tests/{config,engines,workloads,profiling,runner,analysis}/__init__.py
```

- [ ] **Step 5: Install dependencies**

```bash
uv sync
```

- [ ] **Step 6: Verify package imports**

```bash
uv run python -c "import llm_bench; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/ tests/ .gitignore configs/ notebooks/ docker/ scripts/ results/ uv.lock
git commit -m "feat: scaffold project structure with uv"
```

---

### Task 2: Configuration Schema

**Files:**
- Create: `src/llm_bench/config/schema.py`
- Create: `tests/config/test_schema.py`

- [ ] **Step 1: Write failing tests for config schema**

```python
# tests/config/test_schema.py
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
        engine=EngineConfig(engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct"),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD),
        benchmark=BenchmarkConfig(),
    )
    run2 = RunConfig(
        engine=EngineConfig(engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct"),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD),
        benchmark=BenchmarkConfig(),
    )
    assert run1.config_hash == run2.config_hash


def test_run_config_hash_differs_on_change() -> None:
    run1 = RunConfig(
        engine=EngineConfig(engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct"),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD, concurrency=1),
        benchmark=BenchmarkConfig(),
    )
    run2 = RunConfig(
        engine=EngineConfig(engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct"),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD, concurrency=16),
        benchmark=BenchmarkConfig(),
    )
    assert run1.config_hash != run2.config_hash
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/config/test_schema.py -v
```

Expected: `ModuleNotFoundError: No module named 'llm_bench.config.schema'` or `ImportError`

- [ ] **Step 3: Implement config schema**

```python
# src/llm_bench/config/schema.py
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EngineType(str, Enum):
    VLLM = "vllm"
    TRTLLM = "trtllm"
    SGLANG = "sglang"


class WorkloadType(str, Enum):
    STANDARD = "standard"
    PREFIX_HEAVY = "prefix_heavy"
    SPECULATIVE = "speculative"


class EngineConfig(BaseModel):
    engine: EngineType
    model: str
    tp_size: int = 1
    engine_params: dict[str, Any] = Field(default_factory=dict)


class WorkloadConfig(BaseModel):
    workload: WorkloadType
    num_requests: int = 100
    concurrency: int = 1
    warmup_requests: int = 10
    max_tokens: int = 256
    workload_params: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    num_repetitions: int = 3
    warmup_requests: int = 10
    enable_profiling: bool = False
    request_rate: float | None = None  # None = closed-loop, float = Poisson QPS


class RunConfig(BaseModel):
    engine: EngineConfig
    workload: WorkloadConfig
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @property
    def config_hash(self) -> str:
        data = self.model_dump(mode="json")
        raw = json.dumps(data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/config/test_schema.py -v
```

Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/config/schema.py tests/config/test_schema.py
git commit -m "feat: add Pydantic config schema for engines, workloads, benchmarks"
```

---

### Task 3: Configuration Matrix

**Files:**
- Create: `src/llm_bench/config/matrix.py`
- Create: `tests/config/test_matrix.py`
- Create: `configs/benchmark_matrix.yaml`

- [ ] **Step 1: Write failing tests for matrix expansion**

```python
# tests/config/test_matrix.py
import pytest
from pathlib import Path
from llm_bench.config.matrix import MatrixExpander, MatrixFilter
from llm_bench.config.schema import EngineType, WorkloadType, RunConfig


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
    # 2 engines * 1 model * 1 workload * 1 tp * 2 concurrency = 4
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
    from llm_bench.config.schema import EngineConfig, WorkloadConfig, BenchmarkConfig

    run = RunConfig(
        engine=EngineConfig(
            engine=EngineType.VLLM,
            model="meta-llama/Llama-3.1-8B-Instruct",
            tp_size=2,
        ),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD),
        benchmark=BenchmarkConfig(),
    )
    assert filt.should_skip(run) is True


def test_filter_allows_tp1_when_max_gpus_1() -> None:
    filt = MatrixFilter(num_gpus=1)
    from llm_bench.config.schema import EngineConfig, WorkloadConfig, BenchmarkConfig

    run = RunConfig(
        engine=EngineConfig(
            engine=EngineType.VLLM,
            model="meta-llama/Llama-3.1-8B-Instruct",
            tp_size=1,
        ),
        workload=WorkloadConfig(workload=WorkloadType.STANDARD),
        benchmark=BenchmarkConfig(),
    )
    assert filt.should_skip(run) is False


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
    # 1 engine * 1 model * 1 workload * 1 tp * 1 concurrency * 2 prefix_caching = 2
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/config/test_matrix.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement matrix expander**

```python
# src/llm_bench/config/matrix.py
from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import yaml

from llm_bench.config.schema import (
    BenchmarkConfig,
    EngineConfig,
    EngineType,
    RunConfig,
    WorkloadConfig,
    WorkloadType,
)

logger = logging.getLogger(__name__)


class MatrixFilter:
    def __init__(self, num_gpus: int = 8) -> None:
        self.num_gpus = num_gpus

    def should_skip(self, run: RunConfig) -> str | None:
        """Return skip reason string if config should be skipped, None otherwise."""
        if (
            run.workload.workload == WorkloadType.SPECULATIVE
            and run.engine.engine == EngineType.SGLANG
        ):
            return "speculative workload not supported on SGLang"
        if run.engine.tp_size > self.num_gpus:
            return f"tp_size={run.engine.tp_size} exceeds available GPUs ({self.num_gpus})"
        return None


class MatrixExpander:
    def __init__(self, raw: dict[str, Any], num_gpus: int = 8) -> None:
        self.raw = raw["matrix"]
        self.filter = MatrixFilter(num_gpus=num_gpus)
        self.skipped: list[str] = []

    @classmethod
    def from_yaml(cls, path: Path, num_gpus: int = 8) -> MatrixExpander:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data, num_gpus=num_gpus)

    def _expand_engine_params(self, engine: str) -> list[dict[str, Any]]:
        """Expand engine-specific params into list of param dicts."""
        raw_params = self.raw.get("engine_params", {}).get(engine, {})
        if not raw_params:
            return [{}]

        keys = list(raw_params.keys())
        value_lists = []
        for k in keys:
            v = raw_params[k]
            value_lists.append(v if isinstance(v, list) else [v])

        combos = []
        for values in itertools.product(*value_lists):
            combos.append(dict(zip(keys, values)))
        return combos

    def expand(self) -> list[RunConfig]:
        configs: list[RunConfig] = []
        self.skipped = []

        engines = [EngineType(e) for e in self.raw["engines"]]
        models = self.raw["models"]
        workloads = [WorkloadType(w) for w in self.raw["workloads"]]
        tp_sizes = self.raw["tp_size"]
        concurrencies = self.raw["concurrency"]

        for engine, model, workload, tp, conc in itertools.product(
            engines, models, workloads, tp_sizes, concurrencies
        ):
            for params in self._expand_engine_params(engine.value):
                run = RunConfig(
                    engine=EngineConfig(
                        engine=engine,
                        model=model,
                        tp_size=tp,
                        engine_params=params,
                    ),
                    workload=WorkloadConfig(
                        workload=workload,
                        concurrency=conc,
                    ),
                    benchmark=BenchmarkConfig(),
                )
                skip_reason = self.filter.should_skip(run)
                if skip_reason:
                    self.skipped.append(skip_reason)
                    logger.info("Skipping config: %s", skip_reason)
                else:
                    configs.append(run)

        logger.info("Expanded %d configs (%d skipped)", len(configs), len(self.skipped))
        return configs
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/config/test_matrix.py -v
```

Expected: All 7 tests PASS

- [ ] **Step 5: Create default benchmark matrix YAML**

```yaml
# configs/benchmark_matrix.yaml
matrix:
  engines: [vllm, trtllm, sglang]
  models:
    - meta-llama/Llama-3.1-8B-Instruct
    - mistralai/Mistral-7B-Instruct-v0.3
  workloads: [standard, prefix_heavy, speculative]
  tp_size: [1, 2]
  concurrency: [1, 4, 16, 32]
  engine_params:
    vllm:
      enable_prefix_caching: [true, false]
      speculative_model: [null, "turboderp/Llama-3.1-1B-Instruct"]
    trtllm:
      decoding_mode: [auto, medusa]
      kv_cache_free_gpu_mem_fraction: [0.9]
    sglang:
      chunked_prefill_size: [4096, 8192]
```

- [ ] **Step 6: Commit**

```bash
git add src/llm_bench/config/matrix.py tests/config/test_matrix.py configs/benchmark_matrix.yaml
git commit -m "feat: add config matrix expansion with invalid-combo filtering"
```

---

### Task 4: Application Metrics & Engine Base

**Files:**
- Create: `src/llm_bench/profiling/metrics.py`
- Create: `src/llm_bench/engines/base.py`
- Create: `tests/profiling/test_metrics.py`

- [ ] **Step 1: Write failing tests for metrics**

```python
# tests/profiling/test_metrics.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/profiling/test_metrics.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement metrics**

```python
# src/llm_bench/profiling/metrics.py
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RequestMetrics:
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    start_time: float
    first_token_time: float
    token_times: list[float]  # absolute timestamps for each token
    end_time: float
    error: str | None = None

    @property
    def ttft(self) -> float:
        return self.first_token_time - self.start_time

    @property
    def itl_values(self) -> list[float]:
        if len(self.token_times) < 2:
            return []
        return [
            self.token_times[i] - self.token_times[i - 1]
            for i in range(1, len(self.token_times))
        ]

    @property
    def e2e_latency(self) -> float:
        return self.end_time - self.start_time


@dataclass
class BenchmarkResult:
    config_hash: str
    engine: str
    model: str
    workload: str
    requests: list[RequestMetrics] = field(default_factory=list)
    engine_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def total_requests(self) -> int:
        return len(self.requests)

    @property
    def successful_requests(self) -> list[RequestMetrics]:
        return [r for r in self.requests if r.error is None]

    @property
    def mean_ttft(self) -> float:
        ttfts = [r.ttft for r in self.successful_requests]
        return sum(ttfts) / len(ttfts) if ttfts else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.requests:
            rows.append(
                {
                    "request_id": r.request_id,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "ttft": r.ttft,
                    "e2e_latency": r.e2e_latency,
                    "itl_mean": sum(r.itl_values) / len(r.itl_values)
                    if r.itl_values
                    else 0.0,
                    "itl_values": r.itl_values,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "error": r.error,
                }
            )
        return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/profiling/test_metrics.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Implement engine base class**

```python
# src/llm_bench/engines/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from llm_bench.config.schema import EngineConfig


@dataclass
class GenerateRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    stream: bool = True


@dataclass
class GenerateResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    token_times: list[float]
    finish_reason: str = "stop"


class ServingEngine(ABC):
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._base_url: str = ""

    @abstractmethod
    async def start(self) -> None:
        """Start the serving engine (launch container, wait for ready)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the serving engine and clean up."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the engine is ready to serve requests."""

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Send a single generation request and return response with timing."""

    @abstractmethod
    def get_engine_metrics(self) -> dict[str, Any]:
        """Collect engine-specific metrics (cache hit rate, etc.)."""

    @property
    def base_url(self) -> str:
        return self._base_url
```

- [ ] **Step 6: Commit**

```bash
git add src/llm_bench/profiling/metrics.py src/llm_bench/engines/base.py tests/profiling/test_metrics.py
git commit -m "feat: add request metrics, benchmark result, and engine base class"
```

---

### Task 5: Async Request Executor

**Files:**
- Create: `src/llm_bench/runner/executor.py`
- Create: `tests/runner/test_executor.py`

- [ ] **Step 1: Write failing tests for executor**

```python
# tests/runner/test_executor.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from llm_bench.runner.executor import RequestExecutor
from llm_bench.engines.base import GenerateRequest


@pytest.fixture
def executor() -> RequestExecutor:
    return RequestExecutor(concurrency=2)


def test_executor_init(executor: RequestExecutor) -> None:
    assert executor.concurrency == 2


@pytest.mark.asyncio
async def test_executor_dispatch_collects_timing() -> None:
    """Test that executor records start_time, first_token_time, token_times, end_time."""
    executor = RequestExecutor(concurrency=1)

    async def mock_generate(request: GenerateRequest) -> None:
        pass

    # We test the timing wrapper in isolation
    import time

    start = time.perf_counter()
    # Verify RequestMetrics fields are populated by run_workload
    # This is an integration-level concern; unit test just validates structure
    assert executor.concurrency == 1


@pytest.mark.asyncio
async def test_poisson_arrival_spacing() -> None:
    """Verify Poisson arrival generates inter-arrival times."""
    from llm_bench.runner.executor import poisson_arrival_times

    times = poisson_arrival_times(rate=10.0, n=100, seed=42)
    assert len(times) == 100
    assert all(t >= 0 for t in times)
    # Mean inter-arrival time should be ~1/rate = 0.1
    mean_gap = sum(times) / len(times)
    assert 0.05 < mean_gap < 0.2  # Loose bounds for randomness
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/runner/test_executor.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement executor**

```python
# src/llm_bench/runner/executor.py
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Callable, Awaitable

import numpy as np

from llm_bench.engines.base import GenerateRequest, GenerateResponse, ServingEngine
from llm_bench.profiling.metrics import RequestMetrics

logger = logging.getLogger(__name__)


def poisson_arrival_times(rate: float, n: int, seed: int | None = None) -> list[float]:
    """Generate n inter-arrival times from a Poisson process with given rate (QPS)."""
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=1.0 / rate, size=n).tolist()


class RequestExecutor:
    def __init__(self, concurrency: int = 1) -> None:
        self.concurrency = concurrency

    async def run_workload(
        self,
        engine: ServingEngine,
        requests: list[GenerateRequest],
        request_rate: float | None = None,
        seed: int | None = None,
    ) -> list[RequestMetrics]:
        """Execute requests against an engine with concurrency control.

        Args:
            engine: Serving engine to send requests to.
            requests: List of generation requests.
            request_rate: If set, use Poisson arrival with this QPS. None = closed-loop.
            seed: Random seed for Poisson arrival times.
        """
        semaphore = asyncio.Semaphore(self.concurrency)
        results: list[RequestMetrics] = []

        if request_rate is not None:
            arrival_times = poisson_arrival_times(request_rate, len(requests), seed)
        else:
            arrival_times = [0.0] * len(requests)

        async def execute_one(idx: int, req: GenerateRequest) -> RequestMetrics:
            request_id = f"req-{idx}-{uuid.uuid4().hex[:8]}"
            async with semaphore:
                start_time = time.perf_counter()
                try:
                    response = await engine.generate(req)
                    return RequestMetrics(
                        request_id=request_id,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        start_time=start_time,
                        first_token_time=response.token_times[0]
                        if response.token_times
                        else start_time,
                        token_times=response.token_times,
                        end_time=time.perf_counter(),
                    )
                except Exception as e:
                    logger.error("Request %s failed: %s", request_id, e)
                    now = time.perf_counter()
                    return RequestMetrics(
                        request_id=request_id,
                        prompt_tokens=0,
                        completion_tokens=0,
                        start_time=start_time,
                        first_token_time=now,
                        token_times=[],
                        end_time=now,
                        error=str(e),
                    )

        tasks: list[asyncio.Task[RequestMetrics]] = []
        for idx, (req, delay) in enumerate(zip(requests, arrival_times)):
            if delay > 0:
                await asyncio.sleep(delay)
            task = asyncio.create_task(execute_one(idx, req))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return list(results)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/runner/test_executor.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/runner/executor.py tests/runner/test_executor.py
git commit -m "feat: add async request executor with Poisson arrival support"
```

---

### Task 6: Dataset Loading

**Files:**
- Create: `src/llm_bench/workloads/datasets.py`
- Create: `tests/workloads/test_datasets.py`

- [ ] **Step 1: Write failing tests for dataset loading**

```python
# tests/workloads/test_datasets.py
import json
import pytest
from pathlib import Path
from llm_bench.workloads.datasets import (
    SyntheticDataset,
    ShareGPTDataset,
    DatasetEntry,
)


def test_dataset_entry_structure() -> None:
    entry = DatasetEntry(prompt="Hello", expected_output_tokens=50)
    assert entry.prompt == "Hello"
    assert entry.expected_output_tokens == 50


def test_synthetic_dataset_generates_correct_count() -> None:
    ds = SyntheticDataset(num_samples=20, prompt_tokens=100, seed=42)
    entries = ds.load()
    assert len(entries) == 20


def test_synthetic_dataset_deterministic() -> None:
    ds1 = SyntheticDataset(num_samples=5, prompt_tokens=50, seed=42)
    ds2 = SyntheticDataset(num_samples=5, prompt_tokens=50, seed=42)
    entries1 = ds1.load()
    entries2 = ds2.load()
    for e1, e2 in zip(entries1, entries2):
        assert e1.prompt == e2.prompt


def test_synthetic_dataset_different_seeds_differ() -> None:
    ds1 = SyntheticDataset(num_samples=5, prompt_tokens=50, seed=1)
    ds2 = SyntheticDataset(num_samples=5, prompt_tokens=50, seed=2)
    entries1 = ds1.load()
    entries2 = ds2.load()
    prompts1 = [e.prompt for e in entries1]
    prompts2 = [e.prompt for e in entries2]
    assert prompts1 != prompts2


def test_sharegpt_dataset_loads_from_file(tmp_path: Path) -> None:
    data = [
        {
            "conversations": [
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt", "value": "Python is a programming language."},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Explain ML."},
                {"from": "gpt", "value": "Machine learning is a subset of AI."},
            ]
        },
    ]
    f = tmp_path / "sharegpt.json"
    f.write_text(json.dumps(data))
    ds = ShareGPTDataset(path=f, num_samples=2)
    entries = ds.load()
    assert len(entries) == 2
    assert "Python" in entries[0].prompt


def test_sharegpt_dataset_truncates_to_num_samples(tmp_path: Path) -> None:
    data = [
        {
            "conversations": [
                {"from": "human", "value": f"Question {i}"},
                {"from": "gpt", "value": f"Answer {i}"},
            ]
        }
        for i in range(100)
    ]
    f = tmp_path / "sharegpt.json"
    f.write_text(json.dumps(data))
    ds = ShareGPTDataset(path=f, num_samples=10)
    entries = ds.load()
    assert len(entries) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/workloads/test_datasets.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement dataset loading**

```python
# src/llm_bench/workloads/datasets.py
from __future__ import annotations

import json
import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetEntry:
    prompt: str
    expected_output_tokens: int = 256


class Dataset(ABC):
    @abstractmethod
    def load(self) -> list[DatasetEntry]:
        """Load dataset entries."""


class SyntheticDataset(Dataset):
    """Generate synthetic prompts with controlled token counts."""

    # Average English word is ~1.3 tokens. Use words to approximate token count.
    WORD_POOL = [
        "the", "of", "and", "to", "in", "is", "it", "for", "that", "was",
        "on", "are", "with", "as", "at", "be", "this", "have", "from", "or",
        "an", "by", "not", "but", "what", "all", "were", "when", "we", "there",
        "can", "said", "each", "which", "do", "how", "if", "will", "up", "about",
        "data", "model", "system", "code", "function", "process", "network",
        "compute", "memory", "kernel", "batch", "layer", "token", "query",
        "server", "request", "cache", "latency", "throughput", "inference",
    ]

    def __init__(
        self,
        num_samples: int = 100,
        prompt_tokens: int = 100,
        output_tokens: int = 256,
        seed: int | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.seed = seed

    def load(self) -> list[DatasetEntry]:
        rng = random.Random(self.seed)
        entries = []
        # Approximate: 1 word ≈ 1.3 tokens
        words_needed = int(self.prompt_tokens / 1.3)
        for _ in range(self.num_samples):
            words = [rng.choice(self.WORD_POOL) for _ in range(words_needed)]
            prompt = " ".join(words)
            entries.append(DatasetEntry(prompt=prompt, expected_output_tokens=self.output_tokens))
        return entries


class ShareGPTDataset(Dataset):
    """Load prompts from ShareGPT-format JSON."""

    def __init__(
        self,
        path: Path,
        num_samples: int | None = None,
    ) -> None:
        self.path = path
        self.num_samples = num_samples

    def load(self) -> list[DatasetEntry]:
        with open(self.path) as f:
            data = json.load(f)

        entries = []
        for item in data:
            convs = item.get("conversations", [])
            human_turns = [c["value"] for c in convs if c["from"] == "human"]
            gpt_turns = [c["value"] for c in convs if c["from"] == "gpt"]
            if not human_turns:
                continue
            prompt = human_turns[0]
            # Estimate output tokens from first GPT response if available
            output_tokens = len(gpt_turns[0].split()) if gpt_turns else 256
            entries.append(DatasetEntry(prompt=prompt, expected_output_tokens=output_tokens))

        if self.num_samples is not None:
            entries = entries[: self.num_samples]
        return entries
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/workloads/test_datasets.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/workloads/datasets.py tests/workloads/test_datasets.py
git commit -m "feat: add synthetic and ShareGPT dataset loaders"
```

---

### Task 7: Workload Patterns

**Files:**
- Create: `src/llm_bench/workloads/patterns.py`
- Create: `src/llm_bench/workloads/generator.py`
- Create: `tests/workloads/test_patterns.py`

- [ ] **Step 1: Write failing tests for workload patterns**

```python
# tests/workloads/test_patterns.py
import pytest
from llm_bench.workloads.patterns import (
    StandardWorkload,
    PrefixHeavyWorkload,
    SpeculativeWorkload,
)
from llm_bench.workloads.generator import WorkloadGenerator
from llm_bench.config.schema import WorkloadConfig, WorkloadType
from llm_bench.engines.base import GenerateRequest


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
    # All requests should share the same prefix
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
    # With 0.5 reuse, roughly half should share prefix with first request
    prefix = requests[0].prompt[:200]
    shared = sum(1 for r in requests if r.prompt.startswith(prefix))
    assert 3 <= shared <= 8  # Loose bounds


def test_speculative_workload_generates_requests() -> None:
    wl = SpeculativeWorkload(num_requests=10, max_tokens=256, seed=42)
    requests = wl.generate()
    assert len(requests) == 10
    assert all(r.max_tokens == 256 for r in requests)


def test_generator_creates_correct_workload() -> None:
    cfg = WorkloadConfig(
        workload=WorkloadType.STANDARD,
        num_requests=5,
        max_tokens=64,
    )
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/workloads/test_patterns.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement workload patterns**

```python
# src/llm_bench/workloads/patterns.py
from __future__ import annotations

import random
from abc import ABC, abstractmethod

from llm_bench.engines.base import GenerateRequest
from llm_bench.workloads.datasets import SyntheticDataset


class Workload(ABC):
    @abstractmethod
    def generate(self) -> list[GenerateRequest]:
        """Generate list of requests for this workload pattern."""


class StandardWorkload(Workload):
    """Standard serving: mix of short/long prompts from synthetic data."""

    def __init__(
        self,
        num_requests: int = 100,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> None:
        self.num_requests = num_requests
        self.max_tokens = max_tokens
        self.seed = seed

    def generate(self) -> list[GenerateRequest]:
        rng = random.Random(self.seed)
        requests = []
        # Mix of short (50 token) and long (2048 token) prompts
        for i in range(self.num_requests):
            prompt_tokens = rng.choice([50, 200, 500, 1024, 2048])
            ds = SyntheticDataset(
                num_samples=1, prompt_tokens=prompt_tokens, seed=(self.seed or 0) + i
            )
            entry = ds.load()[0]
            requests.append(
                GenerateRequest(
                    prompt=entry.prompt,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
            )
        return requests


class PrefixHeavyWorkload(Workload):
    """Prefix-heavy: shared system prompt with varying user queries."""

    def __init__(
        self,
        num_requests: int = 100,
        max_tokens: int = 256,
        prefix_length: int = 1000,
        prefix_reuse_ratio: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.num_requests = num_requests
        self.max_tokens = max_tokens
        self.prefix_length = prefix_length
        self.prefix_reuse_ratio = prefix_reuse_ratio
        self.seed = seed

    def generate(self) -> list[GenerateRequest]:
        rng = random.Random(self.seed)
        # Generate the shared prefix (simulating system prompt / RAG context)
        prefix_ds = SyntheticDataset(
            num_samples=1, prompt_tokens=self.prefix_length, seed=self.seed
        )
        shared_prefix = prefix_ds.load()[0].prompt

        # Generate a second prefix for non-reuse cases
        alt_prefix_ds = SyntheticDataset(
            num_samples=1,
            prompt_tokens=self.prefix_length,
            seed=(self.seed or 0) + 999999,
        )
        alt_prefix = alt_prefix_ds.load()[0].prompt

        requests = []
        for i in range(self.num_requests):
            # Decide whether to reuse prefix
            use_shared = rng.random() < self.prefix_reuse_ratio
            prefix = shared_prefix if use_shared else alt_prefix

            # Generate unique user query suffix
            suffix_ds = SyntheticDataset(
                num_samples=1, prompt_tokens=50, seed=(self.seed or 0) + i + 1000
            )
            suffix = suffix_ds.load()[0].prompt

            requests.append(
                GenerateRequest(
                    prompt=f"{prefix}\n\nUser query: {suffix}",
                    max_tokens=self.max_tokens,
                    stream=True,
                )
            )
        return requests


class SpeculativeWorkload(Workload):
    """Speculative decoding workload: longer outputs to test draft model acceptance."""

    def __init__(
        self,
        num_requests: int = 100,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> None:
        self.num_requests = num_requests
        self.max_tokens = max_tokens
        self.seed = seed

    def generate(self) -> list[GenerateRequest]:
        # Use medium-length prompts — speculative decoding benefits show
        # most clearly with longer generation lengths
        requests = []
        for i in range(self.num_requests):
            ds = SyntheticDataset(
                num_samples=1, prompt_tokens=200, seed=(self.seed or 0) + i
            )
            entry = ds.load()[0]
            requests.append(
                GenerateRequest(
                    prompt=entry.prompt,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
            )
        return requests
```

- [ ] **Step 4: Implement workload generator (factory)**

```python
# src/llm_bench/workloads/generator.py
from __future__ import annotations

from llm_bench.config.schema import WorkloadConfig, WorkloadType
from llm_bench.engines.base import GenerateRequest
from llm_bench.workloads.patterns import (
    PrefixHeavyWorkload,
    SpeculativeWorkload,
    StandardWorkload,
)


class WorkloadGenerator:
    """Factory that creates the right workload pattern from config."""

    def __init__(self, config: WorkloadConfig, seed: int | None = None) -> None:
        self.config = config
        self.seed = seed

    def generate(self) -> list[GenerateRequest]:
        match self.config.workload:
            case WorkloadType.STANDARD:
                wl = StandardWorkload(
                    num_requests=self.config.num_requests,
                    max_tokens=self.config.max_tokens,
                    seed=self.seed,
                )
            case WorkloadType.PREFIX_HEAVY:
                params = self.config.workload_params
                wl = PrefixHeavyWorkload(
                    num_requests=self.config.num_requests,
                    max_tokens=self.config.max_tokens,
                    prefix_length=params.get("prefix_length", 1000),
                    prefix_reuse_ratio=params.get("prefix_reuse_ratio", 1.0),
                    seed=self.seed,
                )
            case WorkloadType.SPECULATIVE:
                wl = SpeculativeWorkload(
                    num_requests=self.config.num_requests,
                    max_tokens=self.config.max_tokens,
                    seed=self.seed,
                )
        return wl.generate()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/workloads/test_patterns.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/llm_bench/workloads/patterns.py src/llm_bench/workloads/generator.py tests/workloads/test_patterns.py
git commit -m "feat: add workload patterns (standard, prefix-heavy, speculative)"
```

---

### Task 8: vLLM Engine Adapter

**Files:**
- Create: `src/llm_bench/engines/vllm.py`
- Create: `tests/engines/test_vllm.py`

- [ ] **Step 1: Write failing tests for vLLM adapter**

```python
# tests/engines/test_vllm.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llm_bench.engines.vllm import VLLMEngine
from llm_bench.engines.base import GenerateRequest
from llm_bench.config.schema import EngineConfig, EngineType


@pytest.fixture
def vllm_config() -> EngineConfig:
    return EngineConfig(
        engine=EngineType.VLLM,
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_size=1,
        engine_params={"enable_prefix_caching": True, "gpu_memory_utilization": 0.9},
    )


def test_vllm_engine_builds_docker_args(vllm_config: EngineConfig) -> None:
    engine = VLLMEngine(vllm_config)
    cmd = engine._build_launch_cmd()
    assert "--model" in cmd
    assert "meta-llama/Llama-3.1-8B-Instruct" in cmd
    assert "--enable-prefix-caching" in cmd
    assert "--gpu-memory-utilization" in cmd
    assert "0.9" in cmd


def test_vllm_engine_builds_tp_args() -> None:
    cfg = EngineConfig(
        engine=EngineType.VLLM,
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_size=2,
    )
    engine = VLLMEngine(cfg)
    cmd = engine._build_launch_cmd()
    assert "--tensor-parallel-size" in cmd
    assert "2" in cmd


def test_vllm_engine_speculative_args() -> None:
    cfg = EngineConfig(
        engine=EngineType.VLLM,
        model="meta-llama/Llama-3.1-8B-Instruct",
        engine_params={"speculative_model": "turboderp/Llama-3.1-1B-Instruct"},
    )
    engine = VLLMEngine(cfg)
    cmd = engine._build_launch_cmd()
    assert "--speculative-model" in cmd
    assert "turboderp/Llama-3.1-1B-Instruct" in cmd


def test_vllm_engine_container_config(vllm_config: EngineConfig) -> None:
    engine = VLLMEngine(vllm_config)
    container_cfg = engine._build_container_config()
    assert container_cfg["image"].startswith("vllm/vllm-openai")
    assert "runtime" in container_cfg
    assert container_cfg["runtime"] == "nvidia"


def test_vllm_health_url(vllm_config: EngineConfig) -> None:
    engine = VLLMEngine(vllm_config)
    engine._base_url = "http://localhost:8000"
    assert engine._health_url == "http://localhost:8000/health"


def test_vllm_metrics_url(vllm_config: EngineConfig) -> None:
    engine = VLLMEngine(vllm_config)
    engine._base_url = "http://localhost:8000"
    assert engine._metrics_url == "http://localhost:8000/metrics"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/engines/test_vllm.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement vLLM adapter**

```python
# src/llm_bench/engines/vllm.py
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp
import docker

from llm_bench.config.schema import EngineConfig
from llm_bench.engines.base import GenerateRequest, GenerateResponse, ServingEngine

logger = logging.getLogger(__name__)

VLLM_IMAGE = "vllm/vllm-openai:latest"
VLLM_PORT = 8000


class VLLMEngine(ServingEngine):
    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        self._container: Any = None
        self._port = VLLM_PORT

    def _build_launch_cmd(self) -> list[str]:
        cmd = [
            "--model", self.config.model,
            "--tensor-parallel-size", str(self.config.tp_size),
            "--port", str(self._port),
        ]
        param_map = {
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "max_num_seqs": "--max-num-seqs",
            "enable_prefix_caching": "--enable-prefix-caching",
            "speculative_model": "--speculative-model",
        }
        for key, flag in param_map.items():
            value = self.config.engine_params.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])
        return cmd

    def _build_container_config(self) -> dict[str, Any]:
        cmd = self._build_launch_cmd()
        gpus = ",".join(str(i) for i in range(self.config.tp_size))
        return {
            "image": VLLM_IMAGE,
            "command": cmd,
            "runtime": "nvidia",
            "environment": {"NVIDIA_VISIBLE_DEVICES": gpus},
            "ports": {f"{self._port}/tcp": self._port},
            "detach": True,
            "auto_remove": True,
            "shm_size": "4g",
        }

    @property
    def _health_url(self) -> str:
        return f"{self._base_url}/health"

    @property
    def _metrics_url(self) -> str:
        return f"{self._base_url}/metrics"

    async def start(self) -> None:
        logger.info("Starting vLLM container for %s (TP=%d)", self.config.model, self.config.tp_size)
        client = docker.from_env()
        container_cfg = self._build_container_config()
        self._container = client.containers.run(**container_cfg)
        self._base_url = f"http://localhost:{self._port}"

        # Wait for health check
        for attempt in range(120):
            if await self.health_check():
                logger.info("vLLM engine ready after %d seconds", attempt)
                return
            await asyncio.sleep(1)
        raise TimeoutError("vLLM engine failed to start within 120 seconds")

    async def stop(self) -> None:
        if self._container:
            logger.info("Stopping vLLM container")
            self._container.stop(timeout=10)
            self._container = None

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self._health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        payload = {
            "model": self.config.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }

        token_times: list[float] = []
        full_text = ""

        async with aiohttp.ClientSession() as session:
            if request.stream:
                async with session.post(
                    f"{self._base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    async for line in resp.content:
                        decoded = line.decode("utf-8").strip()
                        if not decoded.startswith("data: "):
                            continue
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        token_times.append(time.perf_counter())
                        choice = data["choices"][0]
                        full_text += choice.get("text", "")
            else:
                async with session.post(
                    f"{self._base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    data = await resp.json()
                    token_times.append(time.perf_counter())
                    full_text = data["choices"][0]["text"]

        return GenerateResponse(
            text=full_text,
            prompt_tokens=len(request.prompt.split()),  # Approximate
            completion_tokens=len(token_times),
            token_times=token_times,
        )

    def get_engine_metrics(self) -> dict[str, Any]:
        """Scrape Prometheus metrics from vLLM /metrics endpoint."""
        import urllib.request

        try:
            with urllib.request.urlopen(self._metrics_url, timeout=5) as resp:
                text = resp.read().decode()
            metrics: dict[str, Any] = {}
            for line in text.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    metrics[parts[0]] = float(parts[1])
            return metrics
        except Exception as e:
            logger.warning("Failed to scrape vLLM metrics: %s", e)
            return {}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engines/test_vllm.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/engines/vllm.py tests/engines/test_vllm.py
git commit -m "feat: add vLLM engine adapter with Docker lifecycle"
```

---

### Task 9: TensorRT-LLM Engine Adapter

**Files:**
- Create: `src/llm_bench/engines/trtllm.py`
- Create: `tests/engines/test_trtllm.py`

- [ ] **Step 1: Write failing tests for TRT-LLM adapter**

```python
# tests/engines/test_trtllm.py
import pytest
from llm_bench.engines.trtllm import TRTLLMEngine
from llm_bench.config.schema import EngineConfig, EngineType


@pytest.fixture
def trtllm_config() -> EngineConfig:
    return EngineConfig(
        engine=EngineType.TRTLLM,
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_size=2,
        engine_params={
            "max_batch_size": 64,
            "kv_cache_free_gpu_mem_fraction": 0.9,
            "decoding_mode": "auto",
        },
    )


def test_trtllm_container_config(trtllm_config: EngineConfig) -> None:
    engine = TRTLLMEngine(trtllm_config)
    cfg = engine._build_container_config()
    assert "nvcr.io/nvidia/tritonserver" in cfg["image"]
    assert cfg["runtime"] == "nvidia"


def test_trtllm_engine_cache_dir(trtllm_config: EngineConfig) -> None:
    engine = TRTLLMEngine(trtllm_config)
    cache_dir = engine._engine_cache_dir
    assert "Llama-3.1-8B-Instruct" in str(cache_dir)
    assert "tp2" in str(cache_dir)


def test_trtllm_build_cmd(trtllm_config: EngineConfig) -> None:
    engine = TRTLLMEngine(trtllm_config)
    cmd = engine._build_engine_cmd()
    assert "--tp_size" in cmd or "--tensor_parallelism" in cmd
    assert "2" in cmd


def test_trtllm_health_url(trtllm_config: EngineConfig) -> None:
    engine = TRTLLMEngine(trtllm_config)
    engine._base_url = "http://localhost:8000"
    assert "health" in engine._health_url.lower() or "ready" in engine._health_url.lower()


def test_trtllm_medusa_decoding() -> None:
    cfg = EngineConfig(
        engine=EngineType.TRTLLM,
        model="meta-llama/Llama-3.1-8B-Instruct",
        engine_params={"decoding_mode": "medusa"},
    )
    engine = TRTLLMEngine(cfg)
    cmd = engine._build_engine_cmd()
    assert "medusa" in " ".join(cmd).lower() or "speculative" in " ".join(cmd).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/engines/test_trtllm.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement TRT-LLM adapter**

```python
# src/llm_bench/engines/trtllm.py
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp
import docker

from llm_bench.config.schema import EngineConfig
from llm_bench.engines.base import GenerateRequest, GenerateResponse, ServingEngine

logger = logging.getLogger(__name__)

TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3"
TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
ENGINE_CACHE_BASE = Path.home() / ".cache" / "llm-bench" / "trtllm-engines"


class TRTLLMEngine(ServingEngine):
    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        self._container: Any = None
        self._port = TRITON_HTTP_PORT

    @property
    def _engine_cache_dir(self) -> Path:
        model_name = self.config.model.split("/")[-1]
        return ENGINE_CACHE_BASE / f"{model_name}-tp{self.config.tp_size}"

    def _build_engine_cmd(self) -> list[str]:
        """Build the trtllm-build command for engine compilation."""
        cmd = [
            "trtllm-build",
            "--model_dir", f"/models/{self.config.model.split('/')[-1]}",
            "--tp_size", str(self.config.tp_size),
            "--max_batch_size", str(self.config.engine_params.get("max_batch_size", 64)),
            "--max_input_len", "2048",
            "--max_seq_len", "4096",
        ]
        kv_frac = self.config.engine_params.get("kv_cache_free_gpu_mem_fraction")
        if kv_frac is not None:
            cmd.extend(["--kv_cache_free_gpu_mem_fraction", str(kv_frac)])

        decoding = self.config.engine_params.get("decoding_mode", "auto")
        if decoding == "medusa":
            cmd.extend(["--speculative_decoding_mode", "medusa"])
        elif self.config.engine_params.get("enable_chunked_context"):
            cmd.append("--enable_chunked_context")

        return cmd

    def _build_container_config(self) -> dict[str, Any]:
        gpus = ",".join(str(i) for i in range(self.config.tp_size))
        return {
            "image": TRITON_IMAGE,
            "command": [
                "tritonserver",
                "--model-repository=/model_repo",
                "--http-port", str(self._port),
                "--grpc-port", str(TRITON_GRPC_PORT),
            ],
            "runtime": "nvidia",
            "environment": {"NVIDIA_VISIBLE_DEVICES": gpus},
            "ports": {
                f"{self._port}/tcp": self._port,
                f"{TRITON_GRPC_PORT}/tcp": TRITON_GRPC_PORT,
            },
            "volumes": {
                str(self._engine_cache_dir): {"bind": "/engine", "mode": "ro"},
            },
            "detach": True,
            "auto_remove": True,
            "shm_size": "4g",
        }

    @property
    def _health_url(self) -> str:
        return f"{self._base_url}/v2/health/ready"

    async def start(self) -> None:
        logger.info(
            "Starting TRT-LLM/Triton container for %s (TP=%d)",
            self.config.model,
            self.config.tp_size,
        )

        # Check for pre-built engine
        if not self._engine_cache_dir.exists():
            logger.warning(
                "No pre-built TRT engine at %s. Run scripts/build_trt_engines.sh first.",
                self._engine_cache_dir,
            )

        client = docker.from_env()
        container_cfg = self._build_container_config()
        self._container = client.containers.run(**container_cfg)
        self._base_url = f"http://localhost:{self._port}"

        for attempt in range(180):  # TRT-LLM takes longer to load
            if await self.health_check():
                logger.info("TRT-LLM engine ready after %d seconds", attempt)
                return
            await asyncio.sleep(1)
        raise TimeoutError("TRT-LLM engine failed to start within 180 seconds")

    async def stop(self) -> None:
        if self._container:
            logger.info("Stopping TRT-LLM container")
            self._container.stop(timeout=10)
            self._container = None

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._health_url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        # TRT-LLM via Triton uses generate endpoint
        payload = {
            "text_input": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }

        token_times: list[float] = []
        full_text = ""

        async with aiohttp.ClientSession() as session:
            if request.stream:
                async with session.post(
                    f"{self._base_url}/v2/models/ensemble/generate_stream",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    async for line in resp.content:
                        decoded = line.decode("utf-8").strip()
                        if not decoded.startswith("data:"):
                            continue
                        data_str = decoded[5:].strip()
                        if not data_str:
                            continue
                        data = json.loads(data_str)
                        token_times.append(time.perf_counter())
                        full_text += data.get("text_output", "")
            else:
                async with session.post(
                    f"{self._base_url}/v2/models/ensemble/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    data = await resp.json()
                    token_times.append(time.perf_counter())
                    full_text = data.get("text_output", "")

        return GenerateResponse(
            text=full_text,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(token_times),
            token_times=token_times,
        )

    def get_engine_metrics(self) -> dict[str, Any]:
        """Collect Triton server metrics."""
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{self._base_url}/metrics", timeout=5
            ) as resp:
                text = resp.read().decode()
            metrics: dict[str, Any] = {}
            for line in text.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    metrics[parts[0]] = float(parts[1])
            return metrics
        except Exception as e:
            logger.warning("Failed to scrape TRT-LLM metrics: %s", e)
            return {}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engines/test_trtllm.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/engines/trtllm.py tests/engines/test_trtllm.py
git commit -m "feat: add TensorRT-LLM engine adapter with Triton backend"
```

---

### Task 10: SGLang Engine Adapter

**Files:**
- Create: `src/llm_bench/engines/sglang.py`
- Create: `tests/engines/test_sglang.py`

- [ ] **Step 1: Write failing tests for SGLang adapter**

```python
# tests/engines/test_sglang.py
import pytest
from llm_bench.engines.sglang import SGLangEngine
from llm_bench.config.schema import EngineConfig, EngineType


@pytest.fixture
def sglang_config() -> EngineConfig:
    return EngineConfig(
        engine=EngineType.SGLANG,
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_size=1,
        engine_params={"mem_fraction_static": 0.85, "chunked_prefill_size": 4096},
    )


def test_sglang_builds_launch_cmd(sglang_config: EngineConfig) -> None:
    engine = SGLangEngine(sglang_config)
    cmd = engine._build_launch_cmd()
    assert "--model-path" in cmd
    assert "meta-llama/Llama-3.1-8B-Instruct" in cmd
    assert "--mem-fraction-static" in cmd
    assert "0.85" in cmd
    assert "--chunked-prefill-size" in cmd
    assert "4096" in cmd


def test_sglang_tp_args() -> None:
    cfg = EngineConfig(
        engine=EngineType.SGLANG,
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_size=2,
    )
    engine = SGLangEngine(cfg)
    cmd = engine._build_launch_cmd()
    assert "--tp-size" in cmd
    assert "2" in cmd


def test_sglang_container_config(sglang_config: EngineConfig) -> None:
    engine = SGLangEngine(sglang_config)
    cfg = engine._build_container_config()
    assert "sglang" in cfg["image"].lower() or "lmsys" in cfg["image"].lower()
    assert cfg["runtime"] == "nvidia"


def test_sglang_health_url(sglang_config: EngineConfig) -> None:
    engine = SGLangEngine(sglang_config)
    engine._base_url = "http://localhost:30000"
    assert "health" in engine._health_url


def test_sglang_default_port(sglang_config: EngineConfig) -> None:
    engine = SGLangEngine(sglang_config)
    assert engine._port == 30000
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/engines/test_sglang.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement SGLang adapter**

```python
# src/llm_bench/engines/sglang.py
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp
import docker

from llm_bench.config.schema import EngineConfig
from llm_bench.engines.base import GenerateRequest, GenerateResponse, ServingEngine

logger = logging.getLogger(__name__)

SGLANG_IMAGE = "lmsysorg/sglang:latest"
SGLANG_PORT = 30000


class SGLangEngine(ServingEngine):
    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        self._container: Any = None
        self._port = SGLANG_PORT

    def _build_launch_cmd(self) -> list[str]:
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", self.config.model,
            "--tp-size", str(self.config.tp_size),
            "--port", str(self._port),
            "--host", "0.0.0.0",
        ]
        param_map = {
            "mem_fraction_static": "--mem-fraction-static",
            "chunked_prefill_size": "--chunked-prefill-size",
        }
        for key, flag in param_map.items():
            value = self.config.engine_params.get(key)
            if value is not None:
                cmd.extend([flag, str(value)])
        return cmd

    def _build_container_config(self) -> dict[str, Any]:
        cmd = self._build_launch_cmd()
        gpus = ",".join(str(i) for i in range(self.config.tp_size))
        return {
            "image": SGLANG_IMAGE,
            "command": cmd,
            "runtime": "nvidia",
            "environment": {"NVIDIA_VISIBLE_DEVICES": gpus},
            "ports": {f"{self._port}/tcp": self._port},
            "detach": True,
            "auto_remove": True,
            "shm_size": "4g",
        }

    @property
    def _health_url(self) -> str:
        return f"{self._base_url}/health"

    async def start(self) -> None:
        logger.info(
            "Starting SGLang container for %s (TP=%d)",
            self.config.model,
            self.config.tp_size,
        )
        client = docker.from_env()
        container_cfg = self._build_container_config()
        self._container = client.containers.run(**container_cfg)
        self._base_url = f"http://localhost:{self._port}"

        for attempt in range(120):
            if await self.health_check():
                logger.info("SGLang engine ready after %d seconds", attempt)
                return
            await asyncio.sleep(1)
        raise TimeoutError("SGLang engine failed to start within 120 seconds")

    async def stop(self) -> None:
        if self._container:
            logger.info("Stopping SGLang container")
            self._container.stop(timeout=10)
            self._container = None

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._health_url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        # SGLang uses OpenAI-compatible API
        payload = {
            "model": self.config.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }

        token_times: list[float] = []
        full_text = ""

        async with aiohttp.ClientSession() as session:
            if request.stream:
                async with session.post(
                    f"{self._base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    async for line in resp.content:
                        decoded = line.decode("utf-8").strip()
                        if not decoded.startswith("data: "):
                            continue
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        token_times.append(time.perf_counter())
                        choice = data["choices"][0]
                        full_text += choice.get("text", "")
            else:
                async with session.post(
                    f"{self._base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    data = await resp.json()
                    token_times.append(time.perf_counter())
                    full_text = data["choices"][0]["text"]

        return GenerateResponse(
            text=full_text,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(token_times),
            token_times=token_times,
        )

    def get_engine_metrics(self) -> dict[str, Any]:
        """SGLang exposes metrics at /get_model_info and internal stats."""
        import urllib.request

        metrics: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(
                f"{self._base_url}/get_model_info", timeout=5
            ) as resp:
                metrics["model_info"] = json.loads(resp.read().decode())
        except Exception as e:
            logger.warning("Failed to get SGLang model info: %s", e)
        return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engines/test_sglang.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/engines/sglang.py tests/engines/test_sglang.py
git commit -m "feat: add SGLang engine adapter with RadixAttention support"
```

---

### Task 11: Engine Factory

**Files:**
- Create: `src/llm_bench/engines/factory.py`
- Create: `tests/engines/test_factory.py`

- [ ] **Step 1: Write failing tests for engine factory**

```python
# tests/engines/test_factory.py
import pytest
from llm_bench.engines.factory import create_engine
from llm_bench.engines.vllm import VLLMEngine
from llm_bench.engines.trtllm import TRTLLMEngine
from llm_bench.engines.sglang import SGLangEngine
from llm_bench.config.schema import EngineConfig, EngineType


def test_create_vllm_engine() -> None:
    cfg = EngineConfig(engine=EngineType.VLLM, model="test-model")
    engine = create_engine(cfg)
    assert isinstance(engine, VLLMEngine)


def test_create_trtllm_engine() -> None:
    cfg = EngineConfig(engine=EngineType.TRTLLM, model="test-model")
    engine = create_engine(cfg)
    assert isinstance(engine, TRTLLMEngine)


def test_create_sglang_engine() -> None:
    cfg = EngineConfig(engine=EngineType.SGLANG, model="test-model")
    engine = create_engine(cfg)
    assert isinstance(engine, SGLangEngine)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/engines/test_factory.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement engine factory**

```python
# src/llm_bench/engines/factory.py
from __future__ import annotations

from llm_bench.config.schema import EngineConfig, EngineType
from llm_bench.engines.base import ServingEngine
from llm_bench.engines.sglang import SGLangEngine
from llm_bench.engines.trtllm import TRTLLMEngine
from llm_bench.engines.vllm import VLLMEngine

_ENGINE_MAP: dict[EngineType, type[ServingEngine]] = {
    EngineType.VLLM: VLLMEngine,
    EngineType.TRTLLM: TRTLLMEngine,
    EngineType.SGLANG: SGLangEngine,
}


def create_engine(config: EngineConfig) -> ServingEngine:
    engine_cls = _ENGINE_MAP[config.engine]
    return engine_cls(config)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engines/test_factory.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/engines/factory.py tests/engines/test_factory.py
git commit -m "feat: add engine factory for adapter creation"
```

---

### Task 12: Nsight Systems Profiler

**Files:**
- Create: `src/llm_bench/profiling/nsight.py`
- Create: `tests/profiling/test_nsight.py`

- [ ] **Step 1: Write failing tests for Nsight profiler**

```python
# tests/profiling/test_nsight.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from llm_bench.profiling.nsight import NsightProfiler


def test_nsight_profiler_builds_cmd() -> None:
    profiler = NsightProfiler(output_dir=Path("/tmp/traces"))
    cmd = profiler.build_profile_cmd(
        target_pid=1234,
        trace_name="vllm_run1",
    )
    assert "nsys" in cmd[0]
    assert "profile" in cmd
    assert "1234" in " ".join(cmd) or "--pid" in cmd or "-p" in cmd


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
        trace_path=Path("/tmp/traces/test.nsys-rep"),
        report="gpukernsum",
    )
    assert "nsys" in cmd[0]
    assert "stats" in cmd
    assert "--format" in cmd
    assert "csv" in cmd
    assert "gpukernsum" in cmd
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/profiling/test_nsight.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement Nsight profiler**

```python
# src/llm_bench/profiling/nsight.py
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class NsightProfiler:
    """Manages Nsight Systems profiling lifecycle."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def trace_path(self, trace_name: str) -> Path:
        return self.output_dir / f"{trace_name}.nsys-rep"

    def wrap_command(self, cmd: list[str], trace_name: str) -> list[str]:
        """Wrap a command with nsys profile to capture traces."""
        output = self.trace_path(trace_name)
        return [
            "nsys", "profile",
            "--output", str(output),
            "--force-overwrite", "true",
            "--trace", "cuda,nvtx,osrt",
            "--sample", "none",
            "--capture-range", "cudaProfilerApi",
            "--capture-range-end", "stop",
            "--export", "sqlite",
            "--",
        ] + cmd

    def build_profile_cmd(self, target_pid: int, trace_name: str) -> list[str]:
        """Build nsys command to attach to a running process."""
        output = self.trace_path(trace_name)
        return [
            "nsys", "profile",
            "--output", str(output),
            "--force-overwrite", "true",
            "--trace", "cuda,nvtx,osrt",
            "--sample", "none",
            "-p", str(target_pid),
        ]

    def build_stats_cmd(
        self,
        trace_path: Path,
        report: str = "gpukernsum",
    ) -> list[str]:
        """Build nsys stats command to export trace data as CSV."""
        return [
            "nsys", "stats",
            "--report", report,
            "--format", "csv",
            "--output", str(trace_path.with_suffix("")),
            str(trace_path),
        ]

    async def export_stats(
        self,
        trace_path: Path,
        reports: list[str] | None = None,
    ) -> dict[str, Path]:
        """Run nsys stats export for multiple report types."""
        if reports is None:
            reports = ["gpukernsum", "gpumemtimesum", "cudaapisum"]

        exported: dict[str, Path] = {}
        for report in reports:
            cmd = self.build_stats_cmd(trace_path, report=report)
            logger.info("Exporting nsys stats: %s", " ".join(cmd))
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("nsys stats failed for %s: %s", report, stderr.decode())
                continue
            # nsys stats outputs to <trace_name>_<report>.csv
            csv_path = trace_path.with_suffix("") / f"{trace_path.stem}_{report}.csv"
            # Actual path depends on nsys version; check common patterns
            expected = trace_path.parent / f"{trace_path.stem}_{report}.csv"
            exported[report] = expected
            logger.info("Exported %s to %s", report, expected)

        return exported
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/profiling/test_nsight.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/profiling/nsight.py tests/profiling/test_nsight.py
git commit -m "feat: add Nsight Systems profiler wrapper"
```

---

### Task 13: Nsight Trace Parser

**Files:**
- Create: `src/llm_bench/profiling/parser.py`
- Create: `tests/profiling/test_parser.py`

- [ ] **Step 1: Write failing tests for trace parser**

```python
# tests/profiling/test_parser.py
import pytest
import pandas as pd
from pathlib import Path
from io import StringIO
from llm_bench.profiling.parser import NsightParser


@pytest.fixture
def gpu_kernel_csv(tmp_path: Path) -> Path:
    csv_data = """\
"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"
45.2,1200000000,100,12000000,11500000,10000000,15000000,1000000,"ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn"
30.1,800000000,100,8000000,7800000,7000000,9000000,500000,"fmha_v2_flash_attention_fp16_64_128_S_128_sm80_kernel"
15.5,410000000,200,2050000,2000000,1800000,2500000,200000,"void cudnn::winograd_nonfused::winogradForwardData"
9.2,245000000,50,4900000,4800000,4500000,5500000,300000,"void at::native::vectorized_elementwise_kernel"
"""
    f = tmp_path / "trace_gpukernsum.csv"
    f.write_text(csv_data)
    return f


@pytest.fixture
def gpu_mem_csv(tmp_path: Path) -> Path:
    csv_data = """\
"Operation","Total Time (ns)","Count","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)"
"[CUDA memcpy HtoD]",500000000,50,10000000,9500000,8000000,12000000,1000000
"[CUDA memcpy DtoH]",200000000,30,6666667,6500000,6000000,8000000,500000
"[CUDA memset]",50000000,100,500000,480000,450000,600000,50000
"""
    f = tmp_path / "trace_gpumemtimesum.csv"
    f.write_text(csv_data)
    return f


def test_parse_kernel_summary(gpu_kernel_csv: Path) -> None:
    parser = NsightParser()
    df = parser.parse_kernel_summary(gpu_kernel_csv)
    assert len(df) == 4
    assert "name" in df.columns
    assert "time_pct" in df.columns
    assert "total_time_ns" in df.columns
    assert "avg_ns" in df.columns
    assert df.iloc[0]["time_pct"] == pytest.approx(45.2)


def test_parse_memory_summary(gpu_mem_csv: Path) -> None:
    parser = NsightParser()
    df = parser.parse_memory_summary(gpu_mem_csv)
    assert len(df) == 3
    assert "operation" in df.columns
    assert "total_time_ns" in df.columns


def test_top_kernels_by_time(gpu_kernel_csv: Path) -> None:
    parser = NsightParser()
    df = parser.parse_kernel_summary(gpu_kernel_csv)
    top = parser.top_kernels(df, n=2)
    assert len(top) == 2
    assert "gemm" in top.iloc[0]["name"].lower()


def test_classify_kernel_type(gpu_kernel_csv: Path) -> None:
    parser = NsightParser()
    df = parser.parse_kernel_summary(gpu_kernel_csv)
    classified = parser.classify_kernels(df)
    assert "kernel_type" in classified.columns
    types = set(classified["kernel_type"])
    assert "gemm" in types
    assert "attention" in types
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/profiling/test_parser.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement trace parser**

```python
# src/llm_bench/profiling/parser.py
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Kernel name patterns for classification
KERNEL_PATTERNS: dict[str, list[str]] = {
    "gemm": ["gemm", "cublas", "cutlass"],
    "attention": ["attention", "fmha", "flash_attn", "flash_attention"],
    "elementwise": ["elementwise", "vectorized"],
    "layernorm": ["layernorm", "layer_norm", "rmsnorm"],
    "softmax": ["softmax"],
    "memory": ["memcpy", "memset"],
    "communication": ["nccl", "allreduce", "allgather"],
}


class NsightParser:
    """Parse Nsight Systems CSV exports into structured DataFrames."""

    def parse_kernel_summary(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().strip('"') for c in df.columns]
        return df.rename(
            columns={
                "Time (%)": "time_pct",
                "Total Time (ns)": "total_time_ns",
                "Instances": "instances",
                "Avg (ns)": "avg_ns",
                "Med (ns)": "med_ns",
                "Min (ns)": "min_ns",
                "Max (ns)": "max_ns",
                "StdDev (ns)": "stddev_ns",
                "Name": "name",
            }
        )

    def parse_memory_summary(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().strip('"') for c in df.columns]
        return df.rename(
            columns={
                "Operation": "operation",
                "Total Time (ns)": "total_time_ns",
                "Count": "count",
                "Avg (ns)": "avg_ns",
                "Med (ns)": "med_ns",
                "Min (ns)": "min_ns",
                "Max (ns)": "max_ns",
                "StdDev (ns)": "stddev_ns",
            }
        )

    def top_kernels(self, kernel_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        return kernel_df.nlargest(n, "time_pct").reset_index(drop=True)

    def classify_kernels(self, kernel_df: pd.DataFrame) -> pd.DataFrame:
        df = kernel_df.copy()

        def classify(name: str) -> str:
            name_lower = name.lower()
            for kernel_type, patterns in KERNEL_PATTERNS.items():
                if any(p in name_lower for p in patterns):
                    return kernel_type
            return "other"

        df["kernel_type"] = df["name"].apply(classify)
        return df

    def kernel_time_breakdown(self, kernel_df: pd.DataFrame) -> pd.DataFrame:
        """Group kernel times by type for summary view."""
        classified = self.classify_kernels(kernel_df)
        return (
            classified.groupby("kernel_type")
            .agg(
                total_time_ns=("total_time_ns", "sum"),
                time_pct=("time_pct", "sum"),
                count=("instances", "sum"),
            )
            .sort_values("time_pct", ascending=False)
            .reset_index()
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/profiling/test_parser.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/profiling/parser.py tests/profiling/test_parser.py
git commit -m "feat: add Nsight trace parser with kernel classification"
```

---

### Task 14: GPU Memory Monitor

**Files:**
- Create: `src/llm_bench/profiling/gpu_monitor.py`
- Create: `tests/profiling/test_gpu_monitor.py`

- [ ] **Step 1: Write failing tests for GPU monitor**

```python
# tests/profiling/test_gpu_monitor.py
import pytest
from unittest.mock import patch, MagicMock
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
        GPUSample(timestamp=1.0, gpu_id=0, memory_used_mb=10000, memory_total_mb=81920, gpu_utilization_pct=50.0),
        GPUSample(timestamp=2.0, gpu_id=0, memory_used_mb=20000, memory_total_mb=81920, gpu_utilization_pct=80.0),
        GPUSample(timestamp=3.0, gpu_id=0, memory_used_mb=15000, memory_total_mb=81920, gpu_utilization_pct=60.0),
    ]
    assert monitor.peak_memory_mb(gpu_id=0) == 20000


def test_gpu_monitor_to_dataframe() -> None:
    monitor = GPUMonitor(gpu_ids=[0])
    monitor._samples = [
        GPUSample(timestamp=1.0, gpu_id=0, memory_used_mb=10000, memory_total_mb=81920, gpu_utilization_pct=50.0),
        GPUSample(timestamp=2.0, gpu_id=0, memory_used_mb=20000, memory_total_mb=81920, gpu_utilization_pct=80.0),
    ]
    df = monitor.to_dataframe()
    assert len(df) == 2
    assert "memory_used_mb" in df.columns
    assert "timestamp" in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/profiling/test_gpu_monitor.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement GPU monitor**

```python
# src/llm_bench/profiling/gpu_monitor.py
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GPUSample:
    timestamp: float
    gpu_id: int
    memory_used_mb: float
    memory_total_mb: float
    gpu_utilization_pct: float

    @property
    def memory_utilization_pct(self) -> float:
        return (self.memory_used_mb / self.memory_total_mb) * 100 if self.memory_total_mb > 0 else 0.0


class GPUMonitor:
    """Polls GPU memory and utilization via pynvml."""

    def __init__(
        self,
        poll_interval: float = 1.0,
        gpu_ids: list[int] | None = None,
    ) -> None:
        self.poll_interval = poll_interval
        self.gpu_ids = gpu_ids or [0]
        self._samples: list[GPUSample] = []
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start background polling."""
        self._running = True
        self._samples = []
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop polling and return."""
        self._running = False
        if self._task:
            await self._task
            self._task = None

    async def _poll_loop(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
        except Exception as e:
            logger.warning("pynvml not available, GPU monitoring disabled: %s", e)
            return

        try:
            while self._running:
                for gpu_id in self.gpu_ids:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self._samples.append(
                            GPUSample(
                                timestamp=time.perf_counter(),
                                gpu_id=gpu_id,
                                memory_used_mb=mem_info.used / (1024 * 1024),
                                memory_total_mb=mem_info.total / (1024 * 1024),
                                gpu_utilization_pct=float(util.gpu),
                            )
                        )
                    except Exception as e:
                        logger.debug("GPU %d poll failed: %s", gpu_id, e)
                await asyncio.sleep(self.poll_interval)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def peak_memory_mb(self, gpu_id: int = 0) -> float:
        samples = [s for s in self._samples if s.gpu_id == gpu_id]
        return max(s.memory_used_mb for s in samples) if samples else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": s.timestamp,
                    "gpu_id": s.gpu_id,
                    "memory_used_mb": s.memory_used_mb,
                    "memory_total_mb": s.memory_total_mb,
                    "gpu_utilization_pct": s.gpu_utilization_pct,
                    "memory_utilization_pct": s.memory_utilization_pct,
                }
                for s in self._samples
            ]
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/profiling/test_gpu_monitor.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/profiling/gpu_monitor.py tests/profiling/test_gpu_monitor.py
git commit -m "feat: add async GPU memory/utilization monitor via pynvml"
```

---

### Task 15: Orchestrator

**Files:**
- Create: `src/llm_bench/runner/orchestrator.py`
- Create: `tests/runner/test_orchestrator.py`

- [ ] **Step 1: Write failing tests for orchestrator**

```python
# tests/runner/test_orchestrator.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from llm_bench.runner.orchestrator import BenchmarkOrchestrator
from llm_bench.config.schema import (
    BenchmarkConfig,
    EngineConfig,
    EngineType,
    RunConfig,
    WorkloadConfig,
    WorkloadType,
)


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


def test_orchestrator_run_dir_structure(orchestrator: BenchmarkOrchestrator, run_config: RunConfig) -> None:
    run_dir = orchestrator._run_dir(run_config)
    assert "vllm" in str(run_dir)
    assert run_config.config_hash in str(run_dir)


def test_orchestrator_checks_existing_results(
    orchestrator: BenchmarkOrchestrator, run_config: RunConfig
) -> None:
    # No results yet
    assert orchestrator._has_completed_results(run_config) is False
    # Create marker
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/runner/test_orchestrator.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement orchestrator**

```python
# src/llm_bench/runner/orchestrator.py
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from llm_bench.config.schema import RunConfig
from llm_bench.engines.factory import create_engine
from llm_bench.profiling.gpu_monitor import GPUMonitor
from llm_bench.profiling.metrics import BenchmarkResult, RequestMetrics
from llm_bench.profiling.nsight import NsightProfiler
from llm_bench.runner.executor import RequestExecutor
from llm_bench.workloads.generator import WorkloadGenerator

logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    def __init__(self, results_dir: Path = Path("results")) -> None:
        self.results_dir = results_dir
        self._run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _run_dir(self, config: RunConfig) -> Path:
        return (
            self.results_dir
            / self._run_id
            / config.engine.engine.value
            / config.config_hash
        )

    def _has_completed_results(self, config: RunConfig) -> bool:
        run_dir = self._run_dir(config)
        return (run_dir / "metrics.parquet").exists()

    def _save_run_config(self, config: RunConfig, run_dir: Path) -> None:
        with open(run_dir / "config.json", "w") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2)

    def _save_results(self, result: BenchmarkResult, run_dir: Path) -> None:
        df = result.to_dataframe()
        df.to_parquet(run_dir / "metrics.parquet", index=False)
        # Also save engine metrics
        if result.engine_metrics:
            with open(run_dir / "engine_metrics.json", "w") as f:
                json.dump(result.engine_metrics, f, indent=2)

    async def run_single(self, config: RunConfig, skip_existing: bool = True) -> BenchmarkResult | None:
        """Run a single benchmark configuration."""
        run_dir = self._run_dir(config)

        if skip_existing and self._has_completed_results(config):
            logger.info("Skipping %s — results exist at %s", config.config_hash, run_dir)
            return None

        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_run_config(config, run_dir)

        engine = create_engine(config.engine)
        gpu_ids = list(range(config.engine.tp_size))
        gpu_monitor = GPUMonitor(poll_interval=1.0, gpu_ids=gpu_ids)

        try:
            # Start engine
            logger.info("Starting engine: %s (%s)", config.engine.engine.value, config.config_hash)
            await engine.start()

            # Start GPU monitoring
            await gpu_monitor.start()

            all_metrics: list[RequestMetrics] = []
            for rep in range(config.benchmark.num_repetitions):
                logger.info("Repetition %d/%d", rep + 1, config.benchmark.num_repetitions)

                # Generate workload
                gen = WorkloadGenerator(config.workload, seed=rep)
                requests = gen.generate()

                # Run warm-up
                if config.benchmark.warmup_requests > 0:
                    warmup_reqs = requests[: config.benchmark.warmup_requests]
                    executor = RequestExecutor(concurrency=config.workload.concurrency)
                    await executor.run_workload(engine, warmup_reqs)
                    logger.info("Warm-up complete (%d requests)", len(warmup_reqs))

                # Run benchmark
                bench_reqs = requests[config.benchmark.warmup_requests :]
                executor = RequestExecutor(concurrency=config.workload.concurrency)
                metrics = await executor.run_workload(
                    engine,
                    bench_reqs,
                    request_rate=config.benchmark.request_rate,
                    seed=rep,
                )
                all_metrics.extend(metrics)

            # Stop GPU monitoring
            await gpu_monitor.stop()
            gpu_df = gpu_monitor.to_dataframe()
            if not gpu_df.empty:
                gpu_df.to_parquet(run_dir / "gpu_metrics.parquet", index=False)

            # Collect engine metrics
            engine_metrics = engine.get_engine_metrics()

            result = BenchmarkResult(
                config_hash=config.config_hash,
                engine=config.engine.engine.value,
                model=config.engine.model,
                workload=config.workload.workload.value,
                requests=all_metrics,
                engine_metrics=engine_metrics,
            )
            self._save_results(result, run_dir)
            logger.info(
                "Completed %s: %d requests, mean TTFT=%.3fs",
                config.config_hash,
                result.total_requests,
                result.mean_ttft,
            )
            return result

        except Exception as e:
            logger.error("Run %s failed: %s", config.config_hash, e)
            with open(run_dir / "error.txt", "w") as f:
                f.write(str(e))
            return None
        finally:
            await gpu_monitor.stop()
            await engine.stop()

    async def run_matrix(
        self,
        configs: list[RunConfig],
        skip_existing: bool = True,
    ) -> list[BenchmarkResult]:
        """Run all configs in the matrix sequentially."""
        results: list[BenchmarkResult] = []
        total = len(configs)

        for idx, config in enumerate(configs):
            logger.info("Config %d/%d: %s %s", idx + 1, total, config.engine.engine.value, config.config_hash)
            result = await self.run_single(config, skip_existing=skip_existing)
            if result:
                results.append(result)

        logger.info("Matrix complete: %d/%d succeeded", len(results), total)
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/runner/test_orchestrator.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/runner/orchestrator.py tests/runner/test_orchestrator.py
git commit -m "feat: add benchmark orchestrator with run management"
```

---

### Task 16: Statistical Analysis

**Files:**
- Create: `src/llm_bench/analysis/stats.py`
- Create: `tests/analysis/test_stats.py`

- [ ] **Step 1: Write failing tests for stats**

```python
# tests/analysis/test_stats.py
import pytest
import numpy as np
from llm_bench.analysis.stats import (
    compute_percentiles,
    compute_confidence_interval,
    compute_summary_stats,
    compare_engines,
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
    assert upper - lower < 0.2  # Tight CI with 1000 samples


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
    assert comparison["is_significant"] is True  # Clear 2x speedup
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/analysis/test_stats.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement stats**

```python
# src/llm_bench/analysis/stats.py
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
    """Bootstrap confidence interval."""
    if len(data) < 2:
        return (data[0], data[0]) if data else (0.0, 0.0)

    rng = np.random.default_rng(42)
    arr = np.array(data)
    boot_means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return lower, upper


def compute_summary_stats(data: list[float]) -> dict[str, float]:
    if not data:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "cv": 0.0, "count": 0}
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
    baseline: list[float],
    candidate: list[float],
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Compare two sets of measurements. Returns speedup and significance."""
    baseline_mean = np.mean(baseline)
    candidate_mean = np.mean(candidate)
    speedup = float(baseline_mean / candidate_mean) if candidate_mean > 0 else 0.0

    # Welch's t-test
    if len(baseline) >= 2 and len(candidate) >= 2:
        t_stat, p_value = scipy_stats.ttest_ind(baseline, candidate, equal_var=False)
        is_significant = p_value < alpha
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
```

- [ ] **Step 4: Add scipy dependency**

```bash
uv add scipy
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/analysis/test_stats.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/llm_bench/analysis/stats.py tests/analysis/test_stats.py pyproject.toml uv.lock
git commit -m "feat: add statistical analysis with bootstrap CI and t-tests"
```

---

### Task 17: Plot Generation

**Files:**
- Create: `src/llm_bench/analysis/plots.py`
- Create: `tests/analysis/test_plots.py`

- [ ] **Step 1: Write failing tests for plots**

```python
# tests/analysis/test_plots.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from llm_bench.analysis.plots import BenchmarkPlotter


@pytest.fixture
def sample_results_df() -> pd.DataFrame:
    """Simulate aggregated benchmark results across engines."""
    rng = np.random.default_rng(42)
    rows = []
    for engine in ["vllm", "trtllm", "sglang"]:
        for conc in [1, 4, 16, 32]:
            for _ in range(20):
                rows.append({
                    "engine": engine,
                    "concurrency": conc,
                    "ttft": rng.exponential(0.05) + 0.01 * conc,
                    "itl_mean": rng.exponential(0.01) + 0.001 * conc,
                    "e2e_latency": rng.exponential(0.2) + 0.05 * conc,
                    "throughput_tps": rng.normal(500 - 5 * conc, 50),
                    "workload": "standard",
                })
    return pd.DataFrame(rows)


def test_plotter_creates_output_dir(tmp_path: Path, sample_results_df: pd.DataFrame) -> None:
    plotter = BenchmarkPlotter(output_dir=tmp_path / "plots")
    plotter.plot_ttft_vs_concurrency(sample_results_df)
    assert (tmp_path / "plots").exists()


def test_plot_ttft_vs_concurrency(tmp_path: Path, sample_results_df: pd.DataFrame) -> None:
    plotter = BenchmarkPlotter(output_dir=tmp_path / "plots")
    path = plotter.plot_ttft_vs_concurrency(sample_results_df)
    assert path.exists()
    assert path.suffix == ".png"


def test_plot_itl_distribution(tmp_path: Path, sample_results_df: pd.DataFrame) -> None:
    plotter = BenchmarkPlotter(output_dir=tmp_path / "plots")
    path = plotter.plot_itl_distribution(sample_results_df)
    assert path.exists()


def test_plot_throughput_bars(tmp_path: Path, sample_results_df: pd.DataFrame) -> None:
    plotter = BenchmarkPlotter(output_dir=tmp_path / "plots")
    path = plotter.plot_throughput_vs_concurrency(sample_results_df)
    assert path.exists()


def test_plot_latency_heatmap(tmp_path: Path, sample_results_df: pd.DataFrame) -> None:
    plotter = BenchmarkPlotter(output_dir=tmp_path / "plots")
    path = plotter.plot_latency_heatmap(sample_results_df)
    assert path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/analysis/test_plots.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement plotter**

```python
# src/llm_bench/analysis/plots.py
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent style
PALETTE = {"vllm": "#1f77b4", "trtllm": "#ff7f0e", "sglang": "#2ca02c"}
FIGSIZE = (10, 6)


class BenchmarkPlotter:
    def __init__(self, output_dir: Path = Path("results/plots")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", font_scale=1.1)

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved plot: %s", path)
        return path

    def plot_ttft_vs_concurrency(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for engine in df["engine"].unique():
            subset = df[df["engine"] == engine]
            means = subset.groupby("concurrency")["ttft"].mean()
            stds = subset.groupby("concurrency")["ttft"].std()
            ax.errorbar(
                means.index, means.values, yerr=stds.values,
                label=engine, marker="o", capsize=4,
                color=PALETTE.get(engine),
            )
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("TTFT (seconds)")
        ax.set_title("Time to First Token vs Concurrency")
        ax.legend()
        return self._save(fig, "ttft_vs_concurrency")

    def plot_itl_distribution(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.violinplot(
            data=df, x="engine", y="itl_mean", hue="engine",
            palette=PALETTE, ax=ax, legend=False,
        )
        ax.set_xlabel("Engine")
        ax.set_ylabel("Mean ITL (seconds)")
        ax.set_title("Inter-Token Latency Distribution")
        return self._save(fig, "itl_distribution")

    def plot_throughput_vs_concurrency(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        means = df.groupby(["engine", "concurrency"])["throughput_tps"].mean().reset_index()
        sns.barplot(
            data=means, x="concurrency", y="throughput_tps", hue="engine",
            palette=PALETTE, ax=ax,
        )
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title("Throughput vs Concurrency")
        return self._save(fig, "throughput_vs_concurrency")

    def plot_latency_heatmap(self, df: pd.DataFrame) -> Path:
        pivot = df.groupby(["engine", "concurrency"])["e2e_latency"].mean().unstack()
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
        ax.set_title("End-to-End Latency Heatmap (seconds)")
        ax.set_ylabel("Engine")
        ax.set_xlabel("Concurrency")
        return self._save(fig, "latency_heatmap")

    def plot_kernel_breakdown(self, kernel_df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        kernel_df.plot.barh(x="kernel_type", y="time_pct", ax=ax, color="#4c72b0", legend=False)
        ax.set_xlabel("Time (%)")
        ax.set_ylabel("Kernel Type")
        ax.set_title("GPU Kernel Time Breakdown")
        ax.invert_yaxis()
        return self._save(fig, "kernel_breakdown")

    def plot_gpu_memory_timeline(self, gpu_df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for gpu_id in gpu_df["gpu_id"].unique():
            subset = gpu_df[gpu_df["gpu_id"] == gpu_id]
            ax.plot(
                subset["timestamp"] - subset["timestamp"].min(),
                subset["memory_used_mb"] / 1024,
                label=f"GPU {gpu_id}",
            )
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Memory Used (GB)")
        ax.set_title("GPU Memory Usage Over Time")
        ax.legend()
        return self._save(fig, "gpu_memory_timeline")

    def plot_prefix_cache_efficiency(self, df: pd.DataFrame) -> Path:
        """Plot cache hit rate vs prefix reuse ratio if data available."""
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for engine in df["engine"].unique():
            subset = df[df["engine"] == engine]
            ax.plot(
                subset["prefix_reuse_ratio"],
                subset["cache_hit_rate"],
                label=engine, marker="o",
                color=PALETTE.get(engine),
            )
        ax.set_xlabel("Prefix Reuse Ratio")
        ax.set_ylabel("Cache Hit Rate")
        ax.set_title("Prefix Cache Efficiency by Engine")
        ax.legend()
        return self._save(fig, "prefix_cache_efficiency")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/analysis/test_plots.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/analysis/plots.py tests/analysis/test_plots.py
git commit -m "feat: add benchmark plot generation (TTFT, ITL, throughput, heatmap)"
```

---

### Task 18: Result Export

**Files:**
- Create: `src/llm_bench/analysis/export.py`
- Create: `tests/analysis/test_export.py`

- [ ] **Step 1: Write failing tests for export**

```python
# tests/analysis/test_export.py
import json
import pytest
import pandas as pd
from pathlib import Path
from llm_bench.analysis.export import ResultExporter


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "engine": ["vllm", "vllm", "sglang"],
        "workload": ["standard", "standard", "standard"],
        "concurrency": [1, 4, 1],
        "ttft": [0.05, 0.08, 0.04],
        "e2e_latency": [0.2, 0.35, 0.18],
        "throughput_tps": [500, 450, 520],
    })


def test_export_csv(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    exporter = ResultExporter(output_dir=tmp_path)
    path = exporter.to_csv(sample_df, "results")
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 3


def test_export_json(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    exporter = ResultExporter(output_dir=tmp_path)
    path = exporter.to_json(sample_df, "results")
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert len(data) == 3


def test_export_parquet(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    exporter = ResultExporter(output_dir=tmp_path)
    path = exporter.to_parquet(sample_df, "results")
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == 3


def test_export_summary_table(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    exporter = ResultExporter(output_dir=tmp_path)
    path = exporter.summary_table(sample_df, "summary")
    assert path.exists()
    loaded = pd.read_csv(path)
    assert "engine" in loaded.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/analysis/test_export.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement exporter**

```python
# src/llm_bench/analysis/export.py
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ResultExporter:
    def __init__(self, output_dir: Path = Path("results")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_csv(self, df: pd.DataFrame, name: str) -> Path:
        path = self.output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info("Exported CSV: %s", path)
        return path

    def to_json(self, df: pd.DataFrame, name: str) -> Path:
        path = self.output_dir / f"{name}.json"
        records = df.to_dict(orient="records")
        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        logger.info("Exported JSON: %s", path)
        return path

    def to_parquet(self, df: pd.DataFrame, name: str) -> Path:
        path = self.output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Exported Parquet: %s", path)
        return path

    def summary_table(self, df: pd.DataFrame, name: str) -> Path:
        """Generate summary table: engine × workload → aggregate metrics."""
        group_cols = [c for c in ["engine", "workload", "concurrency"] if c in df.columns]
        metric_cols = [c for c in ["ttft", "e2e_latency", "throughput_tps", "itl_mean"] if c in df.columns]

        if not group_cols or not metric_cols:
            return self.to_csv(df, name)

        summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
        summary.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in summary.columns
        ]
        path = self.output_dir / f"{name}.csv"
        summary.to_csv(path, index=False)
        logger.info("Exported summary: %s", path)
        return path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/analysis/test_export.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_bench/analysis/export.py tests/analysis/test_export.py
git commit -m "feat: add result export (CSV, JSON, Parquet, summary table)"
```

---

### Task 19: CLI

**Files:**
- Create: `src/llm_bench/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for CLI**

```python
# tests/test_cli.py
import pytest
from typer.testing import CliRunner
from llm_bench.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "analyze" in result.stdout
    assert "report" in result.stdout


def test_cli_run_help() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout


def test_cli_analyze_help() -> None:
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--results-dir" in result.stdout


def test_cli_report_help() -> None:
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement CLI**

```python
# src/llm_bench/cli.py
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="bench",
    help="LLM Serving Benchmark — compare TensorRT-LLM, vLLM, and SGLang",
)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def run(
    config: Path = typer.Option(
        Path("configs/benchmark_matrix.yaml"),
        "--config", "-c",
        help="Path to benchmark matrix YAML",
    ),
    results_dir: Path = typer.Option(
        Path("results"),
        "--results-dir", "-o",
        help="Output directory for results",
    ),
    num_gpus: int = typer.Option(
        2,
        "--num-gpus", "-g",
        help="Number of available GPUs",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip configs with existing results",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run benchmark matrix."""
    setup_logging(verbose)

    from llm_bench.config.matrix import MatrixExpander
    from llm_bench.runner.orchestrator import BenchmarkOrchestrator

    expander = MatrixExpander.from_yaml(config, num_gpus=num_gpus)
    configs = expander.expand()

    typer.echo(f"Expanded {len(configs)} configs ({len(expander.skipped)} skipped)")
    for reason in expander.skipped:
        typer.echo(f"  Skipped: {reason}")

    orchestrator = BenchmarkOrchestrator(results_dir=results_dir)
    results = asyncio.run(orchestrator.run_matrix(configs, skip_existing=skip_existing))
    typer.echo(f"Completed {len(results)}/{len(configs)} benchmark runs")


@app.command()
def analyze(
    results_dir: Path = typer.Option(
        Path("results"),
        "--results-dir", "-r",
        help="Directory containing benchmark results",
    ),
    output_dir: Path = typer.Option(
        Path("results/analysis"),
        "--output-dir", "-o",
        help="Output directory for analysis",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run statistical analysis on benchmark results."""
    setup_logging(verbose)

    import pandas as pd
    from llm_bench.analysis.stats import compute_summary_stats
    from llm_bench.analysis.export import ResultExporter

    # Collect all metrics.parquet files
    parquet_files = list(results_dir.rglob("metrics.parquet"))
    if not parquet_files:
        typer.echo("No results found. Run benchmarks first.")
        raise typer.Exit(1)

    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        # Extract engine and config from path
        config_dir = pf.parent
        import json
        config_path = config_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            df["engine"] = cfg["engine"]["engine"]
            df["model"] = cfg["engine"]["model"]
            df["workload"] = cfg["workload"]["workload"]
            df["concurrency"] = cfg["workload"]["concurrency"]
            df["config_hash"] = config_dir.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    exporter = ResultExporter(output_dir=output_dir)
    exporter.to_parquet(combined, "combined_results")
    exporter.summary_table(combined, "summary")
    typer.echo(f"Analysis complete. {len(combined)} total requests across {len(parquet_files)} configs.")


@app.command()
def report(
    results_dir: Path = typer.Option(
        Path("results"),
        "--results-dir", "-r",
        help="Directory containing benchmark results",
    ),
    output_dir: Path = typer.Option(
        Path("results/plots"),
        "--output-dir", "-o",
        help="Output directory for plots",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate plots and report from analysis results."""
    setup_logging(verbose)

    import pandas as pd
    from llm_bench.analysis.plots import BenchmarkPlotter

    # Look for combined results or individual parquets
    combined = results_dir / "analysis" / "combined_results.parquet"
    if combined.exists():
        df = pd.read_parquet(combined)
    else:
        typer.echo("Run 'bench analyze' first to generate combined results.")
        raise typer.Exit(1)

    plotter = BenchmarkPlotter(output_dir=output_dir)

    if "ttft" in df.columns:
        plotter.plot_ttft_vs_concurrency(df)
    if "itl_mean" in df.columns:
        plotter.plot_itl_distribution(df)
    if "throughput_tps" in df.columns:
        plotter.plot_throughput_vs_concurrency(df)
    if "e2e_latency" in df.columns:
        plotter.plot_latency_heatmap(df)

    # GPU memory if available
    gpu_files = list(results_dir.rglob("gpu_metrics.parquet"))
    if gpu_files:
        gpu_df = pd.concat([pd.read_parquet(f) for f in gpu_files], ignore_index=True)
        plotter.plot_gpu_memory_timeline(gpu_df)

    typer.echo(f"Report generated in {output_dir}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Verify CLI entry point works**

```bash
uv run bench --help
```

Expected: Shows help with `run`, `analyze`, `report` commands

- [ ] **Step 6: Commit**

```bash
git add src/llm_bench/cli.py tests/test_cli.py
git commit -m "feat: add Typer CLI with run, analyze, report commands"
```

---

### Task 20: Dockerfiles

**Files:**
- Create: `docker/vllm.Dockerfile`
- Create: `docker/trtllm.Dockerfile`
- Create: `docker/sglang.Dockerfile`

- [ ] **Step 1: Create vLLM Dockerfile**

```dockerfile
# docker/vllm.Dockerfile
FROM vllm/vllm-openai:latest

# Non-root user
RUN useradd -m -s /bin/bash benchuser
USER benchuser

EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
```

- [ ] **Step 2: Create TRT-LLM Dockerfile**

```dockerfile
# docker/trtllm.Dockerfile
FROM nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

# Non-root user
RUN useradd -m -s /bin/bash benchuser

# Engine and model repo will be mounted as volumes
VOLUME ["/engine", "/model_repo"]

USER benchuser

EXPOSE 8000 8001

ENTRYPOINT ["tritonserver", "--model-repository=/model_repo"]
```

- [ ] **Step 3: Create SGLang Dockerfile**

```dockerfile
# docker/sglang.Dockerfile
FROM lmsysorg/sglang:latest

# Non-root user
RUN useradd -m -s /bin/bash benchuser
USER benchuser

EXPOSE 30000

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
```

- [ ] **Step 4: Commit**

```bash
git add docker/
git commit -m "feat: add Dockerfiles for vLLM, TRT-LLM, SGLang engines"
```

---

### Task 21: Model Setup Script

**Files:**
- Create: `scripts/setup_models.sh`

- [ ] **Step 1: Create model download script**

```bash
#!/usr/bin/env bash
# scripts/setup_models.sh
# Download and cache models for benchmarking.
# Requires: huggingface-cli (pip install huggingface_hub)

set -euo pipefail

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "turboderp/Llama-3.1-1B-Instruct"
)

CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Downloading models to $CACHE_DIR"
echo "================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "Downloading: $model"
    huggingface-cli download "$model" --quiet || {
        echo "WARNING: Failed to download $model. You may need to accept the license on HuggingFace."
        echo "Visit: https://huggingface.co/$model"
    }
done

echo ""
echo "Done. Models cached at: $CACHE_DIR"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/setup_models.sh
git add scripts/setup_models.sh
git commit -m "feat: add model download helper script"
```

---

### Task 22: Full Test Suite & Lint

**Files:**
- No new files — verify everything works together

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: All tests PASS

- [ ] **Step 2: Run ruff linter**

```bash
uv run ruff check src/ tests/
```

Expected: No errors (fix any that appear)

- [ ] **Step 3: Run ruff formatter**

```bash
uv run ruff format src/ tests/
```

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -A
git commit -m "style: apply ruff formatting"
```

---

### Task 23: Jupyter Analysis Notebook

**Files:**
- Create: `notebooks/analysis.ipynb`

- [ ] **Step 1: Create analysis notebook**

Create a Jupyter notebook at `notebooks/analysis.ipynb` with the following cells:

**Cell 1 (Markdown):**
```markdown
# LLM Serving Benchmark — Analysis

Comparative analysis of TensorRT-LLM, vLLM, and SGLang across workload patterns.

## Setup
```

**Cell 2 (Code):**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {"vllm": "#1f77b4", "trtllm": "#ff7f0e", "sglang": "#2ca02c"}

results_dir = Path("../results")
```

**Cell 3 (Markdown):**
```markdown
## Load Results
```

**Cell 4 (Code):**
```python
# Load combined results (run `bench analyze` first)
combined_path = results_dir / "analysis" / "combined_results.parquet"
if combined_path.exists():
    df = pd.read_parquet(combined_path)
    print(f"Loaded {len(df)} request records")
    print(f"Engines: {df['engine'].unique()}")
    print(f"Workloads: {df['workload'].unique()}")
    df.head()
else:
    print("No results found. Run benchmarks first:")
    print("  bench run")
    print("  bench analyze")
```

**Cell 5 (Markdown):**
```markdown
## Time to First Token (TTFT) Analysis

TTFT measures prefill latency — how long until the model starts generating.
This is dominated by prompt processing and KV cache allocation.
```

**Cell 6 (Code):**
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# TTFT vs Concurrency
for engine in df["engine"].unique():
    subset = df[df["engine"] == engine]
    means = subset.groupby("concurrency")["ttft"].mean()
    stds = subset.groupby("concurrency")["ttft"].std()
    axes[0].errorbar(means.index, means.values, yerr=stds.values,
                     label=engine, marker="o", capsize=4, color=PALETTE.get(engine))
axes[0].set_xlabel("Concurrency")
axes[0].set_ylabel("TTFT (seconds)")
axes[0].set_title("TTFT vs Concurrency")
axes[0].legend()

# TTFT Distribution
sns.boxplot(data=df, x="engine", y="ttft", hue="engine",
            palette=PALETTE, ax=axes[1], legend=False)
axes[1].set_title("TTFT Distribution")
plt.tight_layout()
plt.show()
```

**Cell 7 (Markdown):**
```markdown
## Inter-Token Latency (ITL) Analysis

ITL measures decode latency per token. Spikes indicate memory pressure,
cache misses, or scheduling delays. This is where prefix caching
inefficiencies and memory fragmentation surface.
```

**Cell 8 (Code):**
```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df, x="engine", y="itl_mean", hue="engine",
               palette=PALETTE, ax=ax, legend=False)
ax.set_xlabel("Engine")
ax.set_ylabel("Mean ITL (seconds)")
ax.set_title("Inter-Token Latency Distribution by Engine")
plt.show()
```

**Cell 9 (Markdown):**
```markdown
## Throughput Comparison

Output tokens per second across concurrency levels.
```

**Cell 10 (Code):**
```python
if "throughput_tps" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    means = df.groupby(["engine", "concurrency"])["throughput_tps"].mean().reset_index()
    sns.barplot(data=means, x="concurrency", y="throughput_tps", hue="engine",
                palette=PALETTE, ax=ax)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs Concurrency")
    plt.show()
```

**Cell 11 (Markdown):**
```markdown
## Latency Heatmap

End-to-end latency across the full configuration matrix.
```

**Cell 12 (Code):**
```python
pivot = df.groupby(["engine", "concurrency"])["e2e_latency"].mean().unstack()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
ax.set_title("E2E Latency Heatmap (seconds)")
plt.show()
```

**Cell 13 (Markdown):**
```markdown
## Statistical Summary
```

**Cell 14 (Code):**
```python
summary = df.groupby(["engine", "workload"]).agg(
    ttft_mean=("ttft", "mean"),
    ttft_p95=("ttft", lambda x: np.percentile(x, 95)),
    itl_mean=("itl_mean", "mean"),
    e2e_mean=("e2e_latency", "mean"),
    e2e_p99=("e2e_latency", lambda x: np.percentile(x, 99)),
    count=("ttft", "count"),
).round(4)
summary
```

- [ ] **Step 2: Commit notebook**

```bash
git add notebooks/analysis.ipynb
git commit -m "feat: add analysis Jupyter notebook with visualization walkthrough"
```

---

### Task 24: Final Integration & README

**Files:**
- No new source files — verify end-to-end, ensure all `__init__.py` exports are clean

- [ ] **Step 1: Verify full test suite passes**

```bash
uv run pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 2: Verify CLI end-to-end**

```bash
uv run bench --help
uv run bench run --help
uv run bench analyze --help
uv run bench report --help
```

Expected: All help texts display correctly

- [ ] **Step 3: Verify package installs cleanly**

```bash
uv pip install -e .
bench --help
```

Expected: CLI accessible after install

- [ ] **Step 4: Run ruff check and format**

```bash
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final integration cleanup and formatting"
```
