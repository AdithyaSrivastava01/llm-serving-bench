from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from llm_bench.config.schema import RunConfig
from llm_bench.engines.factory import create_engine
from llm_bench.profiling.gpu_monitor import GPUMonitor
from llm_bench.profiling.metrics import BenchmarkResult, RequestMetrics
from llm_bench.runner.executor import RequestExecutor
from llm_bench.workloads.generator import WorkloadGenerator

logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    def __init__(self, results_dir: Path = Path("results")) -> None:
        self.results_dir = results_dir
        self._run_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

    def _run_dir(self, config: RunConfig) -> Path:
        return (
            self.results_dir
            / self._run_id
            / config.engine.engine.value
            / config.config_hash
        )

    def _has_completed_results(self, config: RunConfig) -> bool:
        return (self._run_dir(config) / "metrics.parquet").exists()

    def _save_run_config(self, config: RunConfig, run_dir: Path) -> None:
        with open(run_dir / "config.json", "w") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2)

    def _save_results(self, result: BenchmarkResult, run_dir: Path) -> None:
        df = result.to_dataframe()
        df.to_parquet(run_dir / "metrics.parquet", index=False)
        if result.engine_metrics:
            with open(run_dir / "engine_metrics.json", "w") as f:
                json.dump(result.engine_metrics, f, indent=2)

    async def run_single(
        self, config: RunConfig, skip_existing: bool = True
    ) -> BenchmarkResult | None:
        run_dir = self._run_dir(config)
        if skip_existing and self._has_completed_results(config):
            logger.info("Skipping %s — results exist", config.config_hash)
            return None
        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_run_config(config, run_dir)
        engine = create_engine(config.engine)
        gpu_ids = list(range(config.engine.tp_size))
        gpu_monitor = GPUMonitor(poll_interval=1.0, gpu_ids=gpu_ids)
        try:
            logger.info(
                "Starting engine: %s (%s)",
                config.engine.engine.value,
                config.config_hash,
            )
            await engine.start()
            await gpu_monitor.start()
            all_metrics: list[RequestMetrics] = []
            for rep in range(config.benchmark.num_repetitions):
                logger.info(
                    "Repetition %d/%d", rep + 1, config.benchmark.num_repetitions
                )
                gen = WorkloadGenerator(config.workload, seed=rep)
                requests = gen.generate()
                if config.benchmark.warmup_requests > 0:
                    warmup_reqs = requests[: config.benchmark.warmup_requests]
                    executor = RequestExecutor(concurrency=config.workload.concurrency)
                    await executor.run_workload(engine, warmup_reqs)
                bench_reqs = requests[config.benchmark.warmup_requests :]
                executor = RequestExecutor(concurrency=config.workload.concurrency)
                metrics = await executor.run_workload(
                    engine,
                    bench_reqs,
                    request_rate=config.benchmark.request_rate,
                    seed=rep,
                )
                all_metrics.extend(metrics)
            await gpu_monitor.stop()
            gpu_df = gpu_monitor.to_dataframe()
            if not gpu_df.empty:
                gpu_df.to_parquet(run_dir / "gpu_metrics.parquet", index=False)
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
        self, configs: list[RunConfig], skip_existing: bool = True
    ) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for idx, config in enumerate(configs):
            logger.info(
                "Config %d/%d: %s %s",
                idx + 1,
                len(configs),
                config.engine.engine.value,
                config.config_hash,
            )
            result = await self.run_single(config, skip_existing=skip_existing)
            if result:
                results.append(result)
        logger.info("Matrix complete: %d/%d succeeded", len(results), len(configs))
        return results
