"""顶层推理引擎：编排调度、模型执行和多进程工作进程管理。

核心循环：schedule → model_runner.call("run") → postprocess
"""

import atexit
from dataclasses import fields
from time import perf_counter
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """顶层推理引擎：串起调度、执行、后处理三步循环。"""

    def __init__(self, model, **kwargs):
        # 从 kwargs 中筛选 Config 字段，构造配置
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size  # 同步块大小到 Sequence 类
        # 张量并行：为 rank 1..N 派生工作进程（spawn 上下文，CUDA 要求）
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # Rank 0 驱动进程
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id  # 从 tokenizer 获取 EOS token ID
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)  # 注册退出清理

    def exit(self):
        """清理：通知所有进程退出，销毁 NCCL 组。"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """添加推理请求：字符串自动分词 → Sequence → 入队等待调度。"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """单步调度-执行-后处理。

        返回值：
        - outputs: 本步完成的序列列表 [(seq_id, completion_token_ids), ...]
        - num_tokens 符号约定：
            - 正数 → prefill 处理的 token 总数（预填充吞吐量）
            - 负数 → decode 的序列数（decode 吞吐量，取负以示区分）
        """
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 显存监控刷新（主线程中调用，与 tqdm 共用 stdout）
        self.model_runner.memory_monitor.refresh()
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        """所有序列是否处理完毕"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        total_prompts = len(prompts)
        finished_count = 0

        # 显存监控启动（输出初始帧，接管终端进度显示）
        monitor = self.model_runner.memory_monitor
        monitor.start()
        monitor.set_progress(finished_count, total_prompts, 0, 0)

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                finished_count += 1
            # 更新显存监控框中的进度信息
            monitor.set_progress(
                finished_count, total_prompts,
                int(prefill_throughput), int(decode_throughput),
            )

        monitor.stop()

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
