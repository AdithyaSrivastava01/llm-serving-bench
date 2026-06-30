from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

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
