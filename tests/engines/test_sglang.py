import pytest
from llm_bench.engines.sglang import SGLangEngine
from llm_bench.config.schema import EngineConfig, EngineType


@pytest.fixture
def sglang_config() -> EngineConfig:
    return EngineConfig(
        engine=EngineType.SGLANG,
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_size=1,
        engine_params={"mem_fraction_static": 0.85, "chunked_prefill_size": 4096},
    )


def test_sglang_builds_launch_cmd(sglang_config: EngineConfig) -> None:
    engine = SGLangEngine(sglang_config)
    cmd = engine._build_launch_cmd()
    assert "--model-path" in cmd
    assert "meta-llama/Llama-3.1-8B-Instruct" in cmd
    assert "--mem-fraction-static" in cmd
    assert "0.85" in cmd
    assert "--chunked-prefill-size" in cmd
    assert "4096" in cmd


def test_sglang_tp_args() -> None:
    cfg = EngineConfig(
        engine=EngineType.SGLANG, model="meta-llama/Llama-3.1-8B-Instruct", tp_size=2
    )
    engine = SGLangEngine(cfg)
    cmd = engine._build_launch_cmd()
    assert "--tp-size" in cmd
    assert "2" in cmd


def test_sglang_container_config(sglang_config: EngineConfig) -> None:
    engine = SGLangEngine(sglang_config)
    cfg = engine._build_container_config()
    assert "sglang" in cfg["image"].lower() or "lmsys" in cfg["image"].lower()
    assert cfg["runtime"] == "nvidia"


def test_sglang_health_url(sglang_config: EngineConfig) -> None:
    engine = SGLangEngine(sglang_config)
    engine._base_url = "http://localhost:30000"
    assert "health" in engine._health_url


def test_sglang_default_port(sglang_config: EngineConfig) -> None:
    engine = SGLangEngine(sglang_config)
    assert engine._port == 30000
