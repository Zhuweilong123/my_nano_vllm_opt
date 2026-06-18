"""RoPE（旋转位置编码）实现，用于 LLaMA/Qwen 等架构。

在每个 attention head 上对 Q 和 K 的每一对相邻维度应用 2D 旋转。
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """标准 RoPE 2D 旋转：将输入按最后一维配对旋转。

    输入 x 的最后一维必须为偶数，被分成两半 (x1, x2)：
    y1 = x1*cos - x2*sin
    y2 = x2*cos + x1*sin
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """预计算 cos/sin 缓存的 RoPE 模块。

    频率公式（标准 RoPE）：inv_freq[i] = base^(-2i / rotary_dim)
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # RoPE 频率公式：inv_freq[i] = base^(-2i/rotary_dim)，i ∈ [0, rotary_dim/2)
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # unsqueeze(1) 增加一个广播维度（head 维），
        # 使得形状 (max_pos, 1, 2*head_dim) 的缓存可被所有 head 共享
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按给定位置对 Q 和 K 应用 RoPE"""
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """RoPE 工厂函数，LRU 缓存确保只创建一次（简化单模型使用场景）"""
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
