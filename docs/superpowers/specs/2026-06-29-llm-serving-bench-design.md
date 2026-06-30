# LLM Serving Benchmark — Design Spec

## Overview

Monolithic Python framework for stress-testing and comparing three LLM serving engines: TensorRT-LLM, vLLM, and SGLang. Targets portfolio demonstration of deep systems-level understanding of GPU inference serving.

**Goal:** Reproducible benchmark suite that isolates prefix caching inefficiencies, memory fragmentation, and speculative decoding tradeoffs across engines, backed by application-level metrics and Nsight Systems kernel-level profiling.

**Hardware:** 2+ GPUs (tensor-parallel configs are real, not simulated).

**Models:** Small models — Llama 3.1 8B Instruct, Mistral 7B Instruct v0.3.

## Project Structure

```
llm-serving-bench/
├── pyproject.toml              # uv-managed, CLI entry point
├── src/
│   └── llm_bench/
│       ├── __init__.py
│       ├── cli.py              # Typer CLI — bench run, bench analyze, bench report
│       ├── config/
│       │   ├── schema.py       # Pydantic models: BenchmarkConfig, EngineConfig, WorkloadConfig
│       │   └── matrix.py       # Config matrix generation — cartesian product with filters
│       ├── engines/
│       │   ├── base.py         # Abstract ServingEngine interface
│       │   ├── vllm.py         # vLLM adapter
│       │   ├── trtllm.py       # TensorRT-LLM adapter (Triton backend)
│       │   └── sglang.py       # SGLang adapter
│       ├── workloads/
│       │   ├── generator.py    # Request generator — prompt construction, arrival patterns
│       │   ├── patterns.py     # Workload patterns: standard, prefix_heavy, speculative
│       │   └── datasets.py     # Dataset loading (ShareGPT, synthetic)
│       ├── profiling/
│       │   ├── nsight.py       # Nsight Systems launch, trace capture, .nsys-rep management
│       │   ├── parser.py       # nsys stats CSV export → structured metrics
│       │   └── metrics.py      # Application-level metrics: TTFT, ITL, throughput, memory
│       ├── runner/
│       │   ├── orchestrator.py # Main loop: config → start → workload → collect → stop
│       │   └── executor.py     # Async HTTP client, request dispatch, timing
│       └── analysis/
│           ├── stats.py        # Statistical analysis — aggregates, CI, significance
│           ├── plots.py        # matplotlib/seaborn chart generation
│           └── export.py       # CSV/JSON/Parquet result export
├── configs/
│   └── benchmark_matrix.yaml   # Default benchmark matrix
├── notebooks/
│   └── analysis.ipynb          # Interactive analysis walkthrough
├── docker/
│   ├── vllm.Dockerfile
│   ├── trtllm.Dockerfile
│   └── sglang.Dockerfile
├── results/                    # Git-ignored, structured output
└── scripts/
    └── setup_models.sh         # Model download helper
```

## Engine Adapters

### Interface

```python
class ServingEngine(ABC):
    async def start(self, config: EngineConfig) -> None
    async def stop(self) -> None
    async def health_check(self) -> bool
    async def generate(self, request: Request) -> Response
    def get_engine_metrics(self) -> dict
```

### vLLM

- Launch: `python -m vllm.entrypoints.openai.api_server`
- OpenAI-compatible HTTP API
- Prefix caching stats via `/metrics` Prometheus endpoint
- Config knobs: `gpu_memory_utilization`, `max_num_seqs`, `enable_prefix_caching`, `speculative_model`

### TensorRT-LLM

- Runs via Triton Inference Server with TRT-LLM backend
- Requires model engine build step — adapter checks for pre-built engines in cache directory before compiling
- gRPC or HTTP endpoint
- Config knobs: `max_batch_size`, `kv_cache_free_gpu_mem_fraction`, `enable_chunked_context`, `decoding_mode` (speculative)
- Pre-build script provided for all needed engine configs

### SGLang

- Launch: `python -m sglang.launch_server`
- OpenAI-compatible HTTP API
- RadixAttention prefix caching always-on — key differentiator to measure
- Config knobs: `mem_fraction_static`, `tp_size`, `chunked_prefill_size`

### Lifecycle

Docker containers for each engine. Adapter starts container, polls health endpoint, runs workload, stops container. Clean isolation between runs.

## Workload Patterns

### Standard Serving

- ShareGPT dataset for realistic prompt/completion length distributions
- Poisson arrival process with configurable QPS
- Concurrency levels: 1, 4, 16, 32 simultaneous requests
- Mix of short (50 token) and long (2048 token) prompts

### Prefix-Heavy

Targets prefix caching behavior differences:

- Shared system prompt (500-2000 tokens) across all requests in a batch
- Simulates RAG pattern: fixed retrieval context + varying user query
- Warm-up phase first, then steady-state measurement to isolate caching effects
- Varies prefix length and prefix reuse ratio to map efficiency curve
- Measures cache hit rate differences: SGLang RadixAttention vs vLLM prefix caching vs TRT-LLM

### Speculative Decoding

- Draft model configured (e.g., Llama 3.1 8B target + Llama 3.1 1B draft)
- Measures acceptance rate, draft overhead, net speedup
- Compares vLLM and TRT-LLM (both support speculative decoding)
- SGLang included as non-speculative baseline

### Load Generator

- Async `aiohttp` client with semaphore-based concurrency control
- Streaming SSE support for inter-token latency measurement
- Per-request timing: request sent → first token → each subsequent token → last token
- Records raw timestamps (not just aggregates) for offline statistical analysis
- Configurable warm-up period excluded from measurements
- Poisson arrival process for realistic traffic simulation

### Datasets

- ShareGPT — downloaded and cached locally
- Synthetic — generated prompts with controlled token counts for reproducibility
- Referenced in configs, not checked into git

## Profiling

### Application-Level Metrics (always collected)

- **TTFT** — time to first token (prefill latency)
- **ITL** — inter-token latency (full per-token distribution)
- **End-to-end latency** — request start to completion
- **Throughput** — tokens/second (input + output), requests/second
- **GPU memory** — peak allocation, fragmentation via `nvidia-smi` polling
- **Cache hit rate** — from engine-specific metrics endpoints where available

All stored as raw per-request records in Parquet format. Aggregation at analysis time.

### Nsight Systems Trace Profiling

- `nsys profile` launched programmatically wrapping the engine server process
- Captures CUDA kernels, memory ops, CUDA stream activity, NVTX ranges
- Controlled by config flag — off by default (adds ~5-10% overhead)
- Produces `.nsys-rep` files in `results/<run_id>/traces/`

### Trace Parsing Pipeline

- `nsys stats --format csv` exports kernel-level data from `.nsys-rep` files
- Parser extracts:
  - Top kernels by time (attention, GEMM, memory copy)
  - Memory allocation/deallocation timeline
  - CUDA stream utilization and overlap
  - Kernel launch gaps (idle GPU time)
- Output as DataFrames, joined with application metrics by timestamp
- Pinned to known nsys version in Docker images (version-sensitive CLI)

### Key Insight

Correlate application-level stalls (high ITL spikes) with kernel-level events (memory fragmentation, cache misses, kernel launch delays). This is the core differentiator — isolating prefix caching inefficiencies and memory fragmentation through trace evidence.

Trace files are large (100MB+) — git-ignored, not committed.

## Configuration Matrix

### YAML Definition

```yaml
matrix:
  engines: [vllm, trtllm, sglang]
  models: [meta-llama/Llama-3.1-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3]
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

### Matrix Expansion

Cartesian product with filters — skip invalid combos:
- Speculative workload on SGLang (no speculative decoding support)
- TP=2 when only 1 GPU available

`matrix.py` generates valid config list, logs skipped combos with reason.

**Estimated size:** 30-40 valid configurations.

### Orchestrator Flow

1. Parse matrix → generate valid config list
2. For each config:
   - Build/pull Docker container for engine
   - Start engine with config params
   - Wait for health check (timeout + retry)
   - Run warm-up phase (results discarded)
   - Run benchmark phase (results recorded)
   - Optionally run Nsight-profiled phase
   - Stop engine, wait for clean shutdown
   - Save results to `results/<run_id>/<engine>/<config_hash>/`
3. After all runs: generate summary CSV, trigger analysis

### Run Management

- Each run gets UUID + human-readable timestamp ID
- Results structure: `results/<run_id>/metrics.parquet`, `traces/`, `config.json`
- Failed runs logged with error — don't abort entire matrix, continue to next config
- Resume support — check for existing results, skip completed configs on re-run

## Analysis & Results

### Statistical Analysis

- Per-config: mean, median, p50/p95/p99, std dev, coefficient of variation
- Cross-config: relative speedup/regression vs baseline (vLLM default config)
- 95% confidence intervals via bootstrapping (3+ repetitions per config)
- Outlier detection — flag anomalous runs
- Statistical significance tests between engines on same workload

### Static Plots (committed to `results/plots/`)

- TTFT vs concurrency — line chart, one line per engine
- ITL distribution — box/violin plots, engine comparison
- Throughput vs concurrency — bar charts
- Prefix cache hit rate vs prefix reuse ratio — line chart per engine
- Speculative decoding acceptance rate breakdown
- GPU memory timeline — line chart from nvidia-smi polling
- Nsight kernel time breakdown — stacked bar chart
- Latency heatmap across full config matrix

All generated programmatically via `bench report` CLI command from Parquet data.

### Jupyter Notebook (`notebooks/analysis.ipynb`)

- Narrative walkthrough: "what did we find?"
- Loads result Parquet files, runs analysis inline
- Commentary explaining why certain engines win on certain workloads
- Prefix caching deep-dive: RadixAttention vs vLLM with Nsight trace evidence
- Speculative decoding tradeoff analysis
- Publishable quality — readable without running benchmarks

### README Integration

- Key plots embedded as PNGs
- Summary table: engine × workload → winner with metric
- Links to notebook for full analysis

### Export Formats

- Parquet — primary raw storage
- CSV — aggregate results
- JSON — machine-readable config + results pairs

## Tech Stack

- **Language:** Python 3.11+
- **Package management:** uv
- **CLI:** Typer
- **Config:** Pydantic v2 + YAML
- **HTTP client:** aiohttp (async)
- **Data:** pandas, pyarrow (Parquet)
- **Plots:** matplotlib, seaborn
- **Profiling:** Nsight Systems CLI (`nsys`)
- **Containers:** Docker SDK for Python
- **GPU monitoring:** pynvml / nvidia-smi
- **Notebook:** Jupyter

## Non-Goals

- Web dashboard or Streamlit app (out of scope — notebooks + static plots suffice for portfolio)
- Support for engines beyond vLLM, TRT-LLM, SGLang
- Real-time monitoring during benchmark runs
- Cloud deployment automation
- Multi-node distributed inference
