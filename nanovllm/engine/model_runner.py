"""模型推理运行器：负责模型前向传播、采样，以及多进程 IPC 和 CUDA graph 加速。

张量并行（TP）架构：
- Rank 0：驱动进程，负责调度、准备输入、CUDA graph 重放、采样
- Rank 1..N：工作进程，在 __init__ 中进入 loop() 事件循环
- 进程间通过共享内存（SharedMemory）+ pickle 通信
"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """模型推理运行器。

    负责：
    - 模型加载与权重加载
    - KV cache 内存分配（依据 GPU 可用内存自动计算块数）
    - 预填充/解码输入准备（slot_mapping 构建）
    - CUDA graph 捕获与重放（decode 阶段加速）
    - 共享内存 IPC（多 GPU TP 通信）
    - 采样
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 初始化 NCCL 分布式通信组
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # 构建模型并加载权重
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 预热：触发 CUDA JIT 编译，获取峰值内存统计
        self.warmup_model()
        # 根据 GPU 可用内存分配 KV cache
        self.allocate_kv_cache()
        # 如果未禁用 CUDA graph，提前捕获 decode 图
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多进程 TP 设置
        if self.world_size > 1:
            if rank == 0:
                # Rank 0：创建共享内存（1 MiB），barrier 等待所有进程就绪
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # 工作进程：barrier 后打开共享内存，进入事件循环
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # 工作进程进入无限事件循环，从此不返回

    def exit(self):
        """清理：关闭共享内存、删除 CUDA graph、销毁 NCCL 进程组。"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()  # 删除共享内存段
        if not self.enforce_eager:
            del self.graphs, self.graph_pool  # 释放 CUDA graph 内存
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """工作进程事件循环（rank > 0 进入后不返回）。

        不断从共享内存读取方法名和参数 → 本地调用 → 等待下一次调用。
        "exit" 命令结束循环并退出进程。
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """共享内存 IPC 读取协议（工作进程端）。

        协议格式（小端）：
          bytes[0:4]   = payload 长度 N（4 字节无符号 int，小端）
          bytes[4:N+4] = pickle 编码的 [method_name, *args]
        Event 用于同步：rank 0 写入后 set，工作进程 wait 后读取并 clear。
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 等待 rank 0 写入完成
        n = int.from_bytes(self.shm.buf[0:4], "little")  # 读取 payload 长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化
        self.event.clear()  # 重置事件，等待下次调用
        return method_name, args

    def write_shm(self, method_name, *args):
        """共享内存 IPC 写入协议（驱动进程端）。

        将 pickled [method_name, *args] 连同 4 字节长度头写入共享内存，
        然后设置所有工作进程的 Event 通知它们读取。
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # 长度头
        self.shm.buf[4:n+4] = data                    # pickle payload
        for event in self.event:
            event.set()  # 广播通知所有工作进程

    def call(self, method_name, *args):
        """跨进程方法调用路由。

        - Rank 0：先通过 write_shm 广播到所有工作进程，再本地调用
        - 工作进程：直接本地调用（由 loop() 驱动）
        返回值仅 rank 0 使用。
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """使用虚拟序列运行一次模型，触发 CUDA JIT 编译并捕获峰值内存。

        预热后的峰值内存统计将用于 allocate_kv_cache 的容量计算。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)  # 一次完整 prefill → 触发所有 kernel 编译
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据 GPU 可用内存计算 KV cache 容量并分配 5D 张量。

        内存预算公式：
          available = total * gpu_memory_utilization - used - peak + current

        解释：
        - total * gpu_memory_utilization：允许 KV cache 使用的最大内存
        - used：当前已分配（含模型权重）
        - peak：预热期间的峰值分配（分配后大部被释放）
        - + current：恢复峰值释放后实际占用的内存量
          避免 (peak > current) 时的双重扣除

        KV cache 形状：
          [2, num_hidden_layers, num_blocks, block_size, num_kv_heads, head_dim]
          其中 dim=0 的 2 对应 key 和 value
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 每块占用的字节数：2(k/v) * 层数 * 块大小 * kv头数 * 头维度 * dtype大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 分配单一大块连续 KV cache 张量，然后每层引用对应切片
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # key cache 切片
                module.v_cache = self.kv_cache[1, layer_id]  # value cache 切片
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """构建 block_tables 张量供 flash_attn 使用。

        每行对应一个序列的 block_table，长度对齐到 max_len，不足部分填 -1。
        flash_attn 使用此张量进行分页 KV cache 寻址。
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """准备 prefill 阶段的输入张量。

        核心：构建 slot_mapping，将每个逻辑 token 位置映射到物理 KV cache 槽位。
        slot_mapping 是连接逻辑 token 顺序和物理分页存储的桥梁。

        算法（每条序列）：
          1. start = num_cached_tokens（跳过已缓存的前缀）
          2. end = start + num_scheduled_tokens
          3. 遍历 [start_block, end_block)：
             - 每个块的基址 = block_table[i] * block_size
             - 起始块需加块内偏移 (start % block_size)
             - 中间块覆盖整个 [base, base+block_size)
             - 最后一个块只覆盖 [base, base+end-i*block_size)
          4. 如果 cu_seqlens_k > cu_seqlens_q（有前缀缓存），构建 block_tables
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]  # 累积 Q 序列长度（仅新 token）
        cu_seqlens_k = [0]  # 累积 KV 序列长度（含缓存前缀）
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # 计算本序列要处理的 token 范围
            start = seq.num_cached_tokens   # 跳过已缓存的前缀
            seqlen_q = seq.num_scheduled_tokens  # 本轮要处理的 token 数
            end = start + seqlen_q
            seqlen_k = end  # KV 侧：包含缓存前缀的总长度
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup 阶段无 block_table，跳过 slot_mapping
                continue
            # 计算 token 范围涉及的块索引
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size  # 向上取整
            for i in range(start_block, end_block):
                # 块的物理起始槽位 = block_id * block_size
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    # 起始块：可能不是整块，加上块内偏移
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    # 完整块：从 slot_start 到块末尾
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # 最后一个块：可能不是整块，只到 end 位置
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # 前缀缓存检测：如果 KV 总长度 > Q 长度，说明有缓存前缀命中
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # 异步 pinned memory → GPU 传输
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """准备 decode 阶段的输入张量。

        每条序列只处理 1 个 token（自回归生成）。
        slot_mapping 将新 token 写入 KV cache 的最后一个块的最后一个槽位：
          block_table[-1] * block_size + last_block_num_tokens - 1
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 物理槽位 = 最后 block 的基址 + 块内偏移 - 1（为即将写入的 token 腾位置）
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """运行模型前向传播。

        选择执行路径：
        - Prefill / enforce_eager / bs > 512 → eager 模式
        - Decode + bs ≤ 512 → CUDA graph 重放（避免 kernel launch overhead）

        CUDA graph 重放机制：
        - 图在 capture 阶段以最大维度预分配了缓冲区
        - 重放时：先复制实际输入到缓冲区，再用 sentinel 值填充未使用槽位
          - slot_mapping 填 -1：padding token，store_kvcache kernel 跳过
          - context_lens 填 0：防止 flash_attn 读取垃圾数据
        - 输出从 buffer 切片 [:bs] 获取实际结果
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            # 选择第一个 ≥ bs 的图大小（例如 bs=3 → 选 bs=4 的图）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 填充实际输入
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # 未使用的槽位用 sentinel 值填充
            graph_vars["slot_mapping"].fill_(-1)   # -1 → store_kvcache 跳过
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()     # 0 → 安全的空 context
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()  # 重放捕获的 CUDA graph
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """单步推理：准备输入 → 模型前向 → 采样 → 返回 token ID。

        只有 rank 0 负责采样（工作进程不需要 logits→tokens 转换）。
        推理结束后 reset_context 清理当前步的元数据。
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()  # 清理 Context，准备下一步
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """捕获 CUDA graph 用于 decode 加速。

        为渐进批次大小 [1, 2, 4, 8, 16, 32, ..., max_bs] 各捕获一个图。
        从最大 bs 开始倒序捕获以便共享 graph pool（减少 CUDA 内存开销）。

        所有图共享同一组预分配的 graph_vars 张量（最大维度），
        重放时用实际输入覆盖前 [:bs] 部分。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 预分配最大维度的缓冲区（所有图共享）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 渐进批次大小表
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从最大 bs 开始倒序捕获（第一个图创建 pool，后续共享）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup（触发 CUDA 编译）
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()  # 后续图复用此 pool
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
