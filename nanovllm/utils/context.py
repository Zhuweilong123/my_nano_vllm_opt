"""全局 Context 单例：携带每一步推理所需的元数据。

Attention 层等组件通过 get_context() 获取 slot_mapping、block_tables、序列长度等信息，
无需将张量逐层传递。每个 step 结束时调用 reset_context() 清空。
"""

from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    # 当前是 prefill 阶段还是 decode 阶段
    is_prefill: bool = False
    # FlashAttention varlen API 所需的累积序列长度（Q 侧）
    cu_seqlens_q: torch.Tensor | None = None
    # FlashAttention varlen API 所需的累积序列长度（KV 侧，含缓存前缀）
    cu_seqlens_k: torch.Tensor | None = None
    # 批次中最长序列的 Q 长度
    max_seqlen_q: int = 0
    # 批次中最长序列的 KV 长度
    max_seqlen_k: int = 0
    # 每个 token 映射到的物理 KV cache 槽位（-1 表示 padding）
    slot_mapping: torch.Tensor | None = None
    # decode 阶段每条序列的当前上下文长度
    context_lens: torch.Tensor | None = None
    # 分页 KV cache 的 block_table（用于 prefix cache 时的 cache 读取）
    block_tables: torch.Tensor | None = None

# 全局单例，每次 step 通过 set_context 替换整个对象（引用替换而非原地修改，保证线程安全）
_CONTEXT = Context()

def get_context():
    """返回当前步的 Context 单例"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """设置当前步的 Context（替换整个对象，避免原地修改的并发问题）"""
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """将 Context 重置为默认值（每个 step 结束时调用）"""
    global _CONTEXT
    _CONTEXT = Context()
