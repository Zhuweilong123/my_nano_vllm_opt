"""SwiGLU 门控激活函数。"""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SwiGLU 激活：SiLU-gate * value。

    将输入沿最后一维对半分成 gate 和 value 两部分：
    output = silu(gate) * value
    等价于标准 SwiGLU 操作，假设输入维度为偶数（在 Qwen3 MLP 中满足）。
    """

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)  # 沿最后一维均分：x=gate, y=value
        return F.silu(x) * y
