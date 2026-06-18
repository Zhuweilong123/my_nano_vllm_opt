"""Gumbel-max 分类采样器。

使用 Gumbel-max trick 从 logits 分布中采样，比 torch.multinomial 更高效。
"""

import torch
from torch import nn


class Sampler(nn.Module):
    """基于 Gumbel-max trick 的 token 采样器。

    算法等价于：
        tokens ~ Categorical(softmax(logits / temperature))
    实现为：
        probs = softmax(logits / temperature)
        tokens = argmax(probs / Exp(1))

    原理：probs / Exp(1) 等价于 argmax(logits + Gumbel(0,1))，
    但数值上更稳定（当概率接近零时避免 log 操作）。
    Exp(1) 即 Gumbel(0,1) 的 -log 变换（Gumbel 最大值技巧）。
    """

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))  # 温度缩放
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-max trick：probs / Exp(1) 后 argmax = 从分类分布采样
        # clamp_min 防止噪声零值导致除零
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
