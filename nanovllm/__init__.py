"""nano-vllm：轻量级 LLM 推理引擎。

快速开始：
    from nanovllm import LLM, SamplingParams

    llm = LLM(model="path/to/model")
    outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.8, max_tokens=128))
"""

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
