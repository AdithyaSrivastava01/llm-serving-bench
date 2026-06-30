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
                        engine=engine, model=model, tp_size=tp, engine_params=params
                    ),
                    workload=WorkloadConfig(workload=workload, concurrency=conc),
                    benchmark=BenchmarkConfig(),
                )
                skip_reason = self.filter.should_skip(run)
                if skip_reason:
                    msg = skip_reason
                    logger.debug(
                        "Skipping %s/%s/%s: %s",
                        engine.value,
                        model,
                        workload.value,
                        msg,
                    )
                    self.skipped.append(msg)
                else:
                    configs.append(run)
        return configs
