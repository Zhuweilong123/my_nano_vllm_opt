"""块管理器：实现分页 KV cache 的分配/释放，以及基于滚动哈希的前缀缓存检测。

核心概念：
- 每个 Block 对应 GPU KV cache 中的一段连续槽位 [block_size, num_kv_heads, head_dim]
- 前缀缓存：通过滚动 xxhash 识别相同前缀的序列，复用已写入的 KV cache 块
- ref_count：引用计数，支持多个序列共享同一前缀块
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """KV cache 块的元数据。

    hash: 块内容的滚动 xxhash 摘要（用于前缀缓存查找）
    ref_count: 引用此块的序列数量（共享前缀计数）
    token_ids: 块内容缓存（用于哈希碰撞验证）
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1       # -1 表示未初始化
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """写入新内容后更新哈希和 token 列表"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块为单次引用、空状态（分配时调用）"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """KV cache 块分配器，负责管理空闲/已用块和前缀缓存哈希映射。"""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希 → 块 ID 的映射，用于前缀缓存命中检测
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块 ID 队列（FIFO）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 当前被序列占用的块 ID 集合
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """滚动 xxhash 链：如果 prefix != -1，将前一个块的哈希混合进去，
        使得块 N 的哈希依赖于整个前缀历史（类似 Merkle 链式摘要）。"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """从空闲池弹出一个块，清除失效哈希映射，重置状态，标记为已使用。"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 如果该块之前有哈希记录（已释放但哈希映射残留），删除映射
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """ref_count 归零时，将块归还空闲池（哈希映射保留，供前缀缓存复用）。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """检测前缀缓存命中情况，返回可复用的缓存块数量。

        逐块计算滚动 xxhash 并查询 hash_to_block_id：
        - 块命中 + 已占用（used_block_ids）：复用，不消耗空闲块
        - 块命中 + 空闲：可回收，但需要分配新块
        - 首个缓存未命中即停止（前缀局部性假设，后续块不可能命中）
        - 空闲块不足时返回 -1 表示 OOM
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 哈希未命中 或 token 碰撞验证失败 → 此后不可能再有缓存命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1  # 已被引用的块，无需消耗空闲块
        if len(self.free_block_ids) < num_new_blocks:
            return -1  # 空闲块不足，拒绝分配
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """为序列分配 block_table。

        三类块的处理：
        - 缓存命中 + 已被占用 → ref_count += 1（共享）
        - 缓存命中 + 已释放 → ref_count = 1，从空闲池回收
        - 无命中 → 从空闲池全新分配
        """
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1  # 与其他序列共享
            else:
                block.ref_count = 1   # 从空闲池回收
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())  # 全新分配
        # 标记前缀区域为已缓存，prepare_prefill 将跳过这些 token 的 KV 计算
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放序列占用的块，ref_count 归零的块归还空闲池。

        哈希映射保留不删——释放的块仍可被前缀缓存命中回收。
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查是否可以在序列末尾追加一个 token。

        只有序列长度跨越块边界时才需要新块：len(seq) % block_size == 1
        （即在 block_size、2*block_size 等位置）。bool 转 int 得到 1 或 0。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """如果需要（序列长度刚好跨越块边界），分配新块加入 block_table。"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """计算已完成块的滚动哈希并注册到 hash_to_block_id，供未来前缀缓存命中。

        只处理本轮调度中新填满的块（start ~ end）。
        哈希链从前一个块的哈希继承（start == 0 时从 -1 开始新的链）。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return  # 没有新填满的块
        # 从前一个块的哈希继承，保证链式依赖跨越整个 token 序列
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
