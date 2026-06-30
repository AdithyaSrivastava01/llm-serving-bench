from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class NsightProfiler:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def trace_path(self, trace_name: str) -> Path:
        return self.output_dir / f"{trace_name}.nsys-rep"

    def wrap_command(self, cmd: list[str], trace_name: str) -> list[str]:
        output = self.trace_path(trace_name)
        return [
            "nsys",
            "profile",
            "--output",
            str(output),
            "--force-overwrite",
            "true",
            "--trace",
            "cuda,nvtx,osrt",
            "--sample",
            "none",
            "--capture-range",
            "cudaProfilerApi",
            "--capture-range-end",
            "stop",
            "--export",
            "sqlite",
            "--",
        ] + cmd

    def build_profile_cmd(self, target_pid: int, trace_name: str) -> list[str]:
        output = self.trace_path(trace_name)
        return [
            "nsys",
            "profile",
            "--output",
            str(output),
            "--force-overwrite",
            "true",
            "--trace",
            "cuda,nvtx,osrt",
            "--sample",
            "none",
            "-p",
            str(target_pid),
        ]

    def build_stats_cmd(
        self, trace_path: Path, report: str = "gpukernsum"
    ) -> list[str]:
        return [
            "nsys",
            "stats",
            "--report",
            report,
            "--format",
            "csv",
            "--output",
            str(trace_path.with_suffix("")),
            str(trace_path),
        ]

    async def export_stats(
        self, trace_path: Path, reports: list[str] | None = None
    ) -> dict[str, Path]:
        if reports is None:
            reports = ["gpukernsum", "gpumemtimesum", "cudaapisum"]
        exported: dict[str, Path] = {}
        for report in reports:
            cmd = self.build_stats_cmd(trace_path, report=report)
            logger.info("Exporting nsys stats: %s", " ".join(cmd))
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("nsys stats failed for %s: %s", report, stderr.decode())
                continue
            expected = trace_path.parent / f"{trace_path.stem}_{report}.csv"
            exported[report] = expected
        return exported
