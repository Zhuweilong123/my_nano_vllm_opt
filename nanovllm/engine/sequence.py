"""序列对象：表示一条完整的生成请求（prompt + 生成的 token）。

每个 Sequence 由调度器管理，在 WAITING → RUNNING → FINISHED 状态间流转。
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()   # 等待预填充
    RUNNING = auto()   # 正在生成（decode 阶段）
    FINISHED = auto()  # 已完成（EOS 或达到 max_tokens）


class Sequence:
    # KV cache 块大小，由 LLMEngine.__init__ 从 Config.kvcache_block_size 同步
    block_size = 256
    # 全局计数器，为每个序列分配唯一 ID
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)   # 深拷贝，避免外部修改
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)  # 原始 prompt 长度（不含生成 token）
        # 已缓存的 token 数量（前缀缓存命中 + 已完成的 chunked prefill）
        self.num_cached_tokens = 0
        # 本轮调度要处理的 token 数量
        self.num_scheduled_tokens = 0
        # 当前是否为 prefill 阶段（首次或重新 prefill 后为 True）
        self.is_prefill = True
        # 分页 KV cache 的 block_table：逻辑块索引 → 物理块 ID 的映射
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """原始输入 token（不含生成部分）"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """生成的 token（不含原始 prompt）"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """容纳所有 token 所需的块数（向上取整）"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个块中的 token 数（可能 < block_size）"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """返回第 i 个逻辑块的 token ID 切片"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # IPC 序列化优化（共享内存传输）：
        # - prefill 阶段：序列化完整 token_ids（前缀缓存哈希计算所需）
        # - decode 阶段：只序列化 last_token（单个 int），最小化跨进程数据传输
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            # prefill 路径：收到完整 token_ids
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode 路径：只收到 last_token，重建空列表
            self.token_ids = []
            self.last_token = last_state
