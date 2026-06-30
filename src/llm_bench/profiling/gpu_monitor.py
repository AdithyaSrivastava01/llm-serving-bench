from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

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
        return (
            (self.memory_used_mb / self.memory_total_mb) * 100 if self.memory_total_mb > 0 else 0.0
        )


class GPUMonitor:
    def __init__(self, poll_interval: float = 1.0, gpu_ids: list[int] | None = None) -> None:
        self.poll_interval = poll_interval
        self.gpu_ids = gpu_ids or [0]
        self._samples: list[GPUSample] = []
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._running = True
        self._samples = []
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
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
