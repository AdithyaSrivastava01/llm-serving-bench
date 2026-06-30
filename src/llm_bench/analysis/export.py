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
        return path

    def to_json(self, df: pd.DataFrame, name: str) -> Path:
        path = self.output_dir / f"{name}.json"
        records = df.to_dict(orient="records")
        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        return path

    def to_parquet(self, df: pd.DataFrame, name: str) -> Path:
        path = self.output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    def summary_table(self, df: pd.DataFrame, name: str) -> Path:
        group_cols = [
            c for c in ["engine", "workload", "concurrency"] if c in df.columns
        ]
        metric_cols = [
            c
            for c in ["ttft", "e2e_latency", "throughput_tps", "itl_mean"]
            if c in df.columns
        ]
        if not group_cols or not metric_cols:
            return self.to_csv(df, name)
        summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
        summary.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns
        ]
        path = self.output_dir / f"{name}.csv"
        summary.to_csv(path, index=False)
        return path
