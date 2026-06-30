from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

app = typer.Typer(
    name="bench", help="LLM Serving Benchmark — compare TensorRT-LLM, vLLM, and SGLang"
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
        "--config",
        "-c",
        help="Path to benchmark matrix YAML",
    ),
    results_dir: Path = typer.Option(
        Path("results"), "--results-dir", "-o", help="Output directory for results"
    ),
    num_gpus: int = typer.Option(
        2, "--num-gpus", "-g", help="Number of available GPUs"
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip configs with existing results",
    ),
    mode: str = typer.Option(
        "process",
        "--mode",
        "-m",
        help="Launch mode: 'process' (direct), 'docker' (container), or 'connect' (existing server)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run benchmark matrix."""
    setup_logging(verbose)
    from llm_bench.config.matrix import MatrixExpander
    from llm_bench.config.schema import LaunchMode
    from llm_bench.runner.orchestrator import BenchmarkOrchestrator

    launch_mode = LaunchMode(mode)
    expander = MatrixExpander.from_yaml(config, num_gpus=num_gpus)
    configs = expander.expand()
    for cfg in configs:
        cfg.engine.launch_mode = launch_mode
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
        "--results-dir",
        "-r",
        help="Directory containing benchmark results",
    ),
    output_dir: Path = typer.Option(
        Path("results/analysis"),
        "--output-dir",
        "-o",
        help="Output directory for analysis",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run statistical analysis on benchmark results."""
    setup_logging(verbose)
    import json

    import pandas as pd

    from llm_bench.analysis.export import ResultExporter

    parquet_files = list(results_dir.rglob("metrics.parquet"))
    if not parquet_files:
        typer.echo("No results found. Run benchmarks first.")
        raise typer.Exit(1)
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        config_path = pf.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            df["engine"] = cfg["engine"]["engine"]
            df["model"] = cfg["engine"]["model"]
            df["workload"] = cfg["workload"]["workload"]
            df["concurrency"] = cfg["workload"]["concurrency"]
            df["config_hash"] = pf.parent.name
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    exporter = ResultExporter(output_dir=output_dir)
    exporter.to_parquet(combined, "combined_results")
    exporter.summary_table(combined, "summary")
    typer.echo(
        f"Analysis complete. {len(combined)} total requests across {len(parquet_files)} configs."
    )


@app.command()
def report(
    results_dir: Path = typer.Option(
        Path("results"),
        "--results-dir",
        "-r",
        help="Directory containing benchmark results",
    ),
    output_dir: Path = typer.Option(
        Path("results/plots"), "--output-dir", "-o", help="Output directory for plots"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate plots and report from analysis results."""
    setup_logging(verbose)
    import pandas as pd

    from llm_bench.analysis.plots import BenchmarkPlotter

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
    gpu_files = list(results_dir.rglob("gpu_metrics.parquet"))
    if gpu_files:
        gpu_df = pd.concat([pd.read_parquet(f) for f in gpu_files], ignore_index=True)
        plotter.plot_gpu_memory_timeline(gpu_df)
    typer.echo(f"Report generated in {output_dir}")
