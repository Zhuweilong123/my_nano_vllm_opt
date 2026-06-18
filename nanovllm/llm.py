"""公开 API 入口：LLM 类是 LLMEngine 的便捷别名。"""

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """便捷别名，所有逻辑在 LLMEngine 中实现。"""
    pass
