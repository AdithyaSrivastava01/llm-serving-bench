import pytest

from llm_bench.config.schema import EngineConfig, EngineType
from llm_bench.engines.vllm import VLLMEngine


@pytest.fixture
def vllm_config() -> EngineConfig:
    return EngineConfig(
        engine=EngineType.VLLM,
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_size=1,
        engine_params={"enable_prefix_caching": True, "gpu_memory_utilization": 0.9},
    )


def test_vllm_engine_builds_docker_args(vllm_config: EngineConfig) -> None:
    engine = VLLMEngine(vllm_config)
    cmd = engine._build_launch_cmd()
    assert "--model" in cmd
    assert "meta-llama/Llama-3.1-8B-Instruct" in cmd
    assert "--enable-prefix-caching" in cmd
    assert "--gpu-memory-utilization" in cmd
    assert "0.9" in cmd


def test_vllm_engine_builds_tp_args() -> None:
    cfg = EngineConfig(engine=EngineType.VLLM, model="meta-llama/Llama-3.1-8B-Instruct", tp_size=2)
    engine = VLLMEngine(cfg)
    cmd = engine._build_launch_cmd()
    assert "--tensor-parallel-size" in cmd
    assert "2" in cmd


def test_vllm_engine_speculative_args() -> None:
    cfg = EngineConfig(
        engine=EngineType.VLLM,
        model="meta-llama/Llama-3.1-8B-Instruct",
        engine_params={"speculative_model": "turboderp/Llama-3.1-1B-Instruct"},
    )
    engine = VLLMEngine(cfg)
    cmd = engine._build_launch_cmd()
    assert "--speculative-model" in cmd
    assert "turboderp/Llama-3.1-1B-Instruct" in cmd


def test_vllm_engine_container_config(vllm_config: EngineConfig) -> None:
    engine = VLLMEngine(vllm_config)
    container_cfg = engine._build_container_config()
    assert container_cfg["image"].startswith("vllm/vllm-openai")
    assert container_cfg["runtime"] == "nvidia"


def test_vllm_health_url(vllm_config: EngineConfig) -> None:
    engine = VLLMEngine(vllm_config)
    engine._base_url = "http://localhost:8000"
    assert engine._health_url == "http://localhost:8000/health"


def test_vllm_metrics_url(vllm_config: EngineConfig) -> None:
    engine = VLLMEngine(vllm_config)
    engine._base_url = "http://localhost:8000"
    assert engine._metrics_url == "http://localhost:8000/metrics"
