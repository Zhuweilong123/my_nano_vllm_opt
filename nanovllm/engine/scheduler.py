"""双阶段调度器：prefill-then-decode，支持 chunked prefill、前缀缓存和抢占。

调度策略：
- Phase 1（prefill）：从 waiting 队列取出序列进行预填充，支持分段（chunked）
  以避免单个长序列独占批次。
- Phase 2（decode）：从 running 队列取序列，每步推进 1 个 token。
- 当 decode 阶段空闲块不足时，抢占（preempt）机制将序列退回 waiting 队列。
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """双阶段调度器：管理 waiting/running 队列，协调块分配和抢占。"""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """两个队列都为空时，所有序列处理完毕"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """将新序列加入等待队列"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """主调度算法：返回 (scheduled_seqs, is_prefill)。

        双阶段调度：
        - Phase 1（prefill）：处理 waiting 队列，支持 chunked prefill
        - Phase 2（decode）：处理 running 队列，每步 1 个 token
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # ============================================================
        # Phase 1: Prefill — 从 waiting 队列取出序列进行预填充
        # ============================================================
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 首次分配：检测前缀缓存命中，决定可预填充的 token 数
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break  # 空闲块不足，等待下次调度
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 分段预填充续传：继续处理上次未完成的 token
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # chunked prefill 限制：只有批处理中第一个序列允许部分预填充
            # 防止 KV cache 碎片化——后续序列必须完整地一次预填充
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 所有 token 都已处理完毕 → 转为 RUNNING，准备 decode
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True  # is_prefill=True

        # ============================================================
        # Phase 2: Decode — 从 running 队列取序列，每步 1 个 token
        # ============================================================
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # 空闲块不足时的抢占策略：优先抢占最近加入的序列（LIFO）
                # 如果 running 队列中有其他序列，抢占它们；否则抢占自身
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # can_append 通过：调度 decode 步，必要时分配新块
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs  # 确保至少有一个序列被调度
        # 将调度的序列放回 running 队列头部（保持调度顺序）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False  # is_prefill=False

    def preempt(self, seq: Sequence):
        """抢占序列：退回 waiting 队列头部，释放其 KV cache 块，标记需重新 prefill。"""
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)  # 释放块（可能被其他序列复用）
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """每步调度后的后处理：哈希注册、token 追加、终止检测。

        分段预填充场景下：
        - cached_tokens < total_tokens → 仍有未处理的 token，跳过 append_token
        - cached_tokens == total_tokens → 预填充完成，追加生成的 token
        """
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)  # 注册新填满的块到前缀缓存
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # 分段预填充：还有更多 token 待处理，暂不追加生成 token
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # 终止条件：命中 EOS 或达到 max_tokens
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 释放 KV cache 块
                self.running.remove(seq)
