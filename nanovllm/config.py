"""中央配置：控制 nano-vllm 推理引擎的全局参数。

关键约定：
- kvcache_block_size 必须是 256 的整数倍（CUDA 内存对齐要求）
- num_kvcache_blocks == -1 表示运行时根据 GPU 可用内存自动计算
"""

import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    # 模型路径，指向 HuggingFace 格式的模型目录
    model: str
    # 单次 prefill 批次中所有序列的 token 总数上限
    max_num_batched_tokens: int = 16384
    # 单步调度的最大序列数
    max_num_seqs: int = 512
    # 最大模型长度（会被 hf_config.max_position_embeddings 钳制）
    max_model_len: int = 4096
    # 可用于 KV cache 的 GPU 内存比例（0~1）
    gpu_memory_utilization: float = 0.8
    # 张量并行 GPU 数量（1=单卡，2/4/8=多卡）
    tensor_parallel_size: int = 1
    # True 时禁用 CUDA graph 优化，强制使用 eager 模式
    enforce_eager: bool = False
    # 从 HuggingFace 加载的模型配置（__post_init__ 自动填充）
    hf_config: AutoConfig | None = None
    # 结束符 token ID（-1 表示未设置，由 LLMEngine.__init__ 赋值）
    eos: int = -1
    # KV cache 块大小，必须是 256 的整数倍（CUDA 内存对齐要求）
    kvcache_block_size: int = 256
    # KV cache 块数量（-1 表示由 allocate_kv_cache 根据 GPU 内存自动计算）
    num_kvcache_blocks: int = -1
    # GPU 显存监控间隔（秒），0 或负数表示禁用后台监控
    memory_monitor_interval: float = 0.2

    def __post_init__(self):
        assert os.path.isdir(self.model)  # 确保模型路径存在
        assert self.kvcache_block_size % 256 == 0  # CUDA 对齐约束
        assert 1 <= self.tensor_parallel_size <= 8  # 张量并行度的合理范围
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 取用户设定值和模型最大位置编码的较小值
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
