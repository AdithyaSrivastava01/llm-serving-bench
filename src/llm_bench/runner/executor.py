from __future__ import annotations

import asyncio
import logging
import time
import uuid

import numpy as np

from llm_bench.engines.base import GenerateRequest, GenerateResponse, ServingEngine
from llm_bench.profiling.metrics import RequestMetrics

logger = logging.getLogger(__name__)


def poisson_arrival_times(rate: float, n: int, seed: int | None = None) -> list[float]:
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
        semaphore = asyncio.Semaphore(self.concurrency)

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
                        first_token_time=(
                            response.token_times[0]
                            if response.token_times
                            else start_time
                        ),
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
