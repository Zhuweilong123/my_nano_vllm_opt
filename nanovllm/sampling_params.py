"""采样参数配置，每个请求可独立设置。"""

from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    # 采样温度，必须 > 1e-10（禁止 greedy 采样）
    # 原因：Gumbel-max 采样器要求温度 > 0，否则 logits/temperature 会除以零
    temperature: float = 1.0
    # 最大生成 token 数
    max_tokens: int = 64
    # 是否忽略 EOS，继续生成直至 max_tokens
    ignore_eos: bool = False

    def __post_init__(self):
        # 温度接近 0 会近似 greedy，但禁止真正为 0 以兼容 Gumbel-max 采样器
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
