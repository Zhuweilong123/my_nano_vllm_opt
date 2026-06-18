"""词表并行嵌入层 + 并行 LM Head。

采用 TP 感知的词表分片：每个 rank 持有 num_embeddings // tp_size 行。
前向时掩码过滤仅处理属于本分片的 token，然后 all_reduce 求和。
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """词表并行嵌入层：将词表按 TP rank 均分。

    前向传播：
    1. 将 token ID 偏移到本分片范围
    2. 仅本分片内的 token 参与 embedding 计算
    3. all_reduce 汇总所有分片的结果
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 构建 bool mask：标记属于当前分片词表范围的 token
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)  # 偏移到分片内索引
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y  # 非本分片的 token 清零
            dist.all_reduce(y)         # all_reduce 汇总各分片的 embedding
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """并行 LM Head（输出投影层）。

    优化：prefill 阶段只对每个序列的最后一个 token 计算 logits，
    避免为所有预填充位置生成 logits（因为它们不需要采样）。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # prefill 优化：只取每个序列最后一个 token 的隐藏状态
            # cu_seqlens_q[1:] 是每个序列的 token 结束位置，-1 得到最后一个索引
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 从所有 tp rank 收集 logit 分片，在 rank 0 拼接为完整 logits
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
