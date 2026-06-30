import pytest

from llm_bench.config.schema import EngineConfig, EngineType
from llm_bench.engines.trtllm import TRTLLMEngine


@pytest.fixture
def trtllm_config() -> EngineConfig:
    return EngineConfig(
        engine=EngineType.TRTLLM,
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_size=2,
        engine_params={
            "max_batch_size": 64,
            "kv_cache_free_gpu_mem_fraction": 0.9,
            "decoding_mode": "auto",
        },
    )


def test_trtllm_container_config(trtllm_config: EngineConfig) -> None:
    engine = TRTLLMEngine(trtllm_config)
    cfg = engine._build_container_config()
    assert "nvcr.io/nvidia/tritonserver" in cfg["image"]
    assert cfg["runtime"] == "nvidia"


def test_trtllm_engine_cache_dir(trtllm_config: EngineConfig) -> None:
    engine = TRTLLMEngine(trtllm_config)
    cache_dir = engine._engine_cache_dir
    assert "Llama-3.1-8B-Instruct" in str(cache_dir)
    assert "tp2" in str(cache_dir)


def test_trtllm_build_cmd(trtllm_config: EngineConfig) -> None:
    engine = TRTLLMEngine(trtllm_config)
    cmd = engine._build_engine_cmd()
    assert "--tp_size" in cmd
    assert "2" in cmd


def test_trtllm_health_url(trtllm_config: EngineConfig) -> None:
    engine = TRTLLMEngine(trtllm_config)
    engine._base_url = "http://localhost:8000"
    assert "health" in engine._health_url.lower() or "ready" in engine._health_url.lower()


def test_trtllm_medusa_decoding() -> None:
    cfg = EngineConfig(
        engine=EngineType.TRTLLM,
        model="meta-llama/Llama-3.1-8B-Instruct",
        engine_params={"decoding_mode": "medusa"},
    )
    engine = TRTLLMEngine(cfg)
    cmd = engine._build_engine_cmd()
    assert "medusa" in " ".join(cmd).lower() or "speculative" in " ".join(cmd).lower()
