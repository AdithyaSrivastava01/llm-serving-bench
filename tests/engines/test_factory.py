from llm_bench.config.schema import EngineConfig, EngineType
from llm_bench.engines.factory import create_engine
from llm_bench.engines.sglang import SGLangEngine
from llm_bench.engines.trtllm import TRTLLMEngine
from llm_bench.engines.vllm import VLLMEngine


def test_create_vllm_engine() -> None:
    cfg = EngineConfig(engine=EngineType.VLLM, model="test-model")
    engine = create_engine(cfg)
    assert isinstance(engine, VLLMEngine)


def test_create_trtllm_engine() -> None:
    cfg = EngineConfig(engine=EngineType.TRTLLM, model="test-model")
    engine = create_engine(cfg)
    assert isinstance(engine, TRTLLMEngine)


def test_create_sglang_engine() -> None:
    cfg = EngineConfig(engine=EngineType.SGLANG, model="test-model")
    engine = create_engine(cfg)
    assert isinstance(engine, SGLangEngine)
