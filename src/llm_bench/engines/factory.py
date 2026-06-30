from __future__ import annotations

from llm_bench.config.schema import EngineConfig, EngineType
from llm_bench.engines.base import ServingEngine
from llm_bench.engines.sglang import SGLangEngine
from llm_bench.engines.trtllm import TRTLLMEngine
from llm_bench.engines.vllm import VLLMEngine

_ENGINE_MAP: dict[EngineType, type[ServingEngine]] = {
    EngineType.VLLM: VLLMEngine,
    EngineType.TRTLLM: TRTLLMEngine,
    EngineType.SGLANG: SGLangEngine,
}


def create_engine(config: EngineConfig) -> ServingEngine:
    engine_cls = _ENGINE_MAP[config.engine]
    return engine_cls(config)
