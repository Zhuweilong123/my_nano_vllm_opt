"""RMSNorm（Root Mean Square 层归一化），支持 fused residual-add 路径。"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS Layer Normalization：x = x * weight / sqrt(mean(x^2) + eps)

    提供两种前向模式：
    - 普通模式（residual=None）：标准 RMSNorm
    - fused 模式（residual 非 None）：先计算 x + residual，再做归一化，
      避免额外的内存分配，用于 Transformer residual stream 模式。
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """标准 RMSNorm：x * rsqrt(mean(x^2) + eps) * weight"""
        orig_dtype = x.dtype
        x = x.float()  # 提升到 fp32 保证数值稳定性
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused 模式：x + residual → RMSNorm → (normalized, new_residual)

        将残差加法和归一化合并为单个 kernel，减少内存带宽开销。
        返回值包含更新后的残差（normalization 之前的 x + residual），
        供 Transformer 层后续使用。
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())  # fused: x + residual
        residual = x.to(orig_dtype)            # 保存更新后的残差
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """双模式分发：无 residual 走普通 RMSNorm，有则走 fused 路径"""
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
