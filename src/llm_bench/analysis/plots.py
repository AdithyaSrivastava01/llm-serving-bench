from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
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
        return path

    def plot_ttft_vs_concurrency(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for engine in df["engine"].unique():
            subset = df[df["engine"] == engine]
            means = subset.groupby("concurrency")["ttft"].mean()
            stds = subset.groupby("concurrency")["ttft"].std()
            ax.errorbar(
                means.index,
                means.values,
                yerr=stds.values,
                label=engine,
                marker="o",
                capsize=4,
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
            data=df,
            x="engine",
            y="itl_mean",
            hue="engine",
            palette=PALETTE,
            ax=ax,
            legend=False,
        )
        ax.set_xlabel("Engine")
        ax.set_ylabel("Mean ITL (seconds)")
        ax.set_title("Inter-Token Latency Distribution")
        return self._save(fig, "itl_distribution")

    def plot_throughput_vs_concurrency(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        means = df.groupby(["engine", "concurrency"])["throughput_tps"].mean().reset_index()
        sns.barplot(
            data=means,
            x="concurrency",
            y="throughput_tps",
            hue="engine",
            palette=PALETTE,
            ax=ax,
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
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for engine in df["engine"].unique():
            subset = df[df["engine"] == engine]
            ax.plot(
                subset["prefix_reuse_ratio"],
                subset["cache_hit_rate"],
                label=engine,
                marker="o",
                color=PALETTE.get(engine),
            )
        ax.set_xlabel("Prefix Reuse Ratio")
        ax.set_ylabel("Cache Hit Rate")
        ax.set_title("Prefix Cache Efficiency by Engine")
        ax.legend()
        return self._save(fig, "prefix_cache_efficiency")
