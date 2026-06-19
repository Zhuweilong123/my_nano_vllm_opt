# nano-vllm 系统架构文档

## 1. 项目概述

nano-vllm 是一个轻量级 LLM 推理引擎，实现了：
- **分页 KV Cache（PagedAttention）**：块级内存管理，支持前缀缓存共享
- **张量并行（Tensor Parallelism）**：多 GPU 模型分片推理
- **CUDA Graph 加速**：decode 阶段消除 kernel launch overhead
- **连续批处理（Continuous Batching）**：双阶段调度器（prefill + decode）
- **Chunked Prefill**：长序列分段预填充，避免阻塞 decode

目前支持 **Qwen3** 模型，架构设计可扩展到其他 LLaMA 风格模型。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         LLMEngine                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────────┐  │
│  │ Scheduler │───►│ Model    │───►│ Scheduler.postprocess()  │  │
│  │.schedule()│    │ Runner   │    │                          │  │
│  └──────────┘    │.call(run)│    └──────────────────────────┘  │
│       ▲          └────┬─────┘              │                    │
│       │               │                    │                    │
│  ┌────┴─────────┐    │ IPC (共享内存)      ▼                    │
│  │ BlockManager │    │               ┌─────────────┐           │
│  │ (前缀缓存)    │◄───┘               │ 已完成序列    │           │
│  └──────────────┘                    └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 模块依赖关系

```
nanovllm/
├── __init__.py          # 公开 API: LLM, SamplingParams
├── llm.py               # LLM(LLMEngine) 便捷别名
├── config.py            # 全局配置 Config
├── sampling_params.py   # 采样参数 SamplingParams
│
├── engine/              # 核心推理引擎
│   ├── llm_engine.py    # 顶层编排：schedule → run → postprocess
│   ├── scheduler.py     # 双阶段调度器
│   ├── block_manager.py # 分页 KV Cache 块管理器 + 前缀缓存
│   ├── model_runner.py  # 模型执行器（IPC + CUDA Graph + 采样）
│   └── sequence.py      # 序列对象（状态机 + IPC 序列化）
│
├── layers/              # 模型组件
│   ├── attention.py     # FlashAttention + paged KV cache
│   ├── linear.py        # TP 感知的线性层层次结构
│   ├── embed_head.py    # 词表并行嵌入 + LM Head
│   ├── rotary_embedding.py # RoPE 位置编码
│   ├── layernorm.py     # RMSNorm (fused residual 模式)
│   ├── activation.py    # SwiGLU (SiluAndMul)
│   └── sampler.py       # Gumbel-max 采样器
│
├── models/              # 模型定义
│   └── qwen3.py         # Qwen3 完整模型组装
│
└── utils/               # 工具
    ├── context.py       # 全局 Context 单例（元数据传递）
    └── loader.py        # Safetensor 权重加载器
```

---

## 3. 核心数据流

### 3.1 主循环：generate() → step()

```
LLMEngine.generate(prompts)
  │
  ├── for each prompt:
  │     tokenizer.encode(prompt) → Sequence(token_ids, sampling_params)
  │     Scheduler.add(seq)  → waiting 队列
  │
  └── while not finished:
        │
        LLMEngine.step()
          │
          ├── (1) Scheduler.schedule()
          │       返回 (seqs, is_prefill)
          │
          ├── (2) ModelRunner.call("run", seqs, is_prefill)
          │       │
          │       ├── rank 0: write_shm() 广播方法名+参数到工作进程
          │       ├── all ranks: call("run")
          │       │     ├── prepare_prefill(seqs) 或 prepare_decode(seqs)
          │       │     │     └── 构建 input_ids, positions, slot_mapping
          │       │     │         → set_context(...)
          │       │     ├── run_model()  [eager 或 CUDA graph]
          │       │     │     └── model.forward() → logits
          │       │     ├── Sampler(logits, temperatures) → token_ids
          │       │     └── reset_context()
          │       └── rank 0 返回 token_ids
          │
          └── (3) Scheduler.postprocess(seqs, token_ids, is_prefill)
                 ├── BlockManager.hash_blocks() 注册前缀缓存
                 ├── 更新 seq.num_cached_tokens
                 ├── seq.append_token(token_id)
                 └── 检测 EOS / max_tokens → FINISHED, deallocate
```

### 3.2 张量并行 IPC 协议

```
Rank 0 (驱动)                      Rank 1..N (工作进程)
─────────────                      ─────────────────
write_shm("run", seqs, is_prefill)
  │
  ├── pickle.dumps(["run", seqs, is_prefill])
  ├── shm.buf[0:4] = len(payload)  [小端]
  ├── shm.buf[4:N+4] = payload
  └── event.set() ───────────────► event.wait()
                                    read_shm()
                                      ├── int.from_bytes(shm.buf[0:4])
                                      ├── pickle.loads(shm.buf[4:N+4])
                                      └── event.clear()
call("run", seqs, is_prefill)       call("run", seqs, is_prefill)
  └── local run()                     └── local run()

共享内存大小: 1 MiB (2^20 bytes)
同步原语: multiprocessing.Event
序列化: pickle (配合 Sequence.__getstate__ 优化)
```

---

## 4. 调度器 (Scheduler)

### 4.1 双阶段调度

```
┌──────────────────────────────────────────────────────┐
│                   Scheduler.schedule()                │
│                                                      │
│  Phase 1: Prefill                                    │
│  ┌────────────────────────────────────────────────┐  │
│  │ while waiting 非空 AND bs < max_num_seqs:      │  │
│  │   seq = waiting[0]                             │  │
│  │   BlockManager.can_allocate(seq) → cached_blocks│  │
│  │   BlockManager.allocate(seq, cached)            │  │
│  │   如果 num_scheduled == total → waiting→running │  │
│  │   chunked prefill: 仅第一个序列允许分段          │  │
│  └────────────────────────────────────────────────┘  │
│                      │                               │
│                      ▼ (有 prefill 结果则返回)        │
│                                                      │
│  Phase 2: Decode                                     │
│  ┌────────────────────────────────────────────────┐  │
│  │ while running 非空 AND bs < max_num_seqs:      │  │
│  │   BlockManager.can_append(seq)                 │  │
│  │   if 空闲块不足: preempt(其他序列或自身)         │  │
│  │   BlockManager.may_append(seq)                 │  │
│  │   num_scheduled_tokens = 1                     │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 4.2 Chunked Prefill 限制

```python
# 只有批处理中第一个序列允许分段预填充
# 原因：防止 KV cache 碎片化，后续序列必须完整一次性预填充
if remaining < num_tokens and scheduled_seqs:
    break  # 停止调度更多序列
```

### 4.3 抢占策略 (Preemption)

当 decode 阶段空闲块不足时：
1. 优先抢占 running 队列末尾的序列（LIFO，最近加入的）
2. 如果 running 队列为空，抢占当前序列自身
3. 抢占操作：释放 KV cache 块 → 退回 waiting 队列头部 → 标记需重新 prefill

### 4.4 序列状态机

```
WAITING ──[prefill 完成]──► RUNNING ──[EOS/max_tokens]──► FINISHED
   ▲                            │
   └───── [preempt 抢占] ───────┘
```

---

## 5. 分页 KV Cache 与前缀缓存

### 5.1 Block 数据结构

```
Block (元数据)                     GPU KV Cache (物理)
────────────                       ──────────────────
block_id: int                      shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
ref_count: int                            │      │          │
hash: int (xxhash)                       key/value  层索引    块索引
token_ids: list[int] (碰撞验证)                               │
                                                   ┌──────────┴──────────┐
                                               block_table[i]          block_size 个 token
```

### 5.2 前缀缓存算法

```
Sequence token_ids:
[The, cat, sat, on, the, mat, ...]
 └─── block 0 ──┘└─── block 1 ──┘

缓存检测流程 (can_allocate):
1. 逐块计算滚动 xxhash:
   h0 = xxhash(block_0_tokens)
   h1 = xxhash(h0 + block_1_tokens)    ← 链式哈希（类似 Merkle）

2. 在 hash_to_block_id 中查找:
   hash_to_block_id[h0] → block_id_5   ← 命中!
   hash_to_block_id[h1] → -1           ← 未命中，停止

3. 返回 num_cached_blocks = 1

4. allocate 时:
   缓存块: ref_count += 1 (与已有序列共享)
   新块: 从空闲池分配
```

### 5.3 引用计数与生命周期

```
allocate:     ref_count = 1 (首次) 或 ref_count += 1 (共享)
deallocate:   ref_count -= 1
              当 ref_count == 0 → 归还空闲池 (哈希映射保留)
```

---

## 6. Slot Mapping

slot_mapping 是将**逻辑 token 位置**映射到**物理 KV cache 槽位**的桥梁。

### 6.1 Prefill Slot Mapping

```
序列 token 范围: start=4, end=10 (6 tokens, block_size=6)
block_table = [3, 7]

逻辑 token:     [t4, t5, t6, t7, t8, t9]
                  │   │   │──────────│
                  │   │   block 7 (完整)
                  │   │
                  │   block 3 (部分: offset=4%6=4, 末尾)
                  │
块索引:          start_block=0    end_block=1

slot 计算:
  block 0 (i=0): slot_start = 3*6 + 4 = 22
                  slot_end   = 3*6 + 6 = 24    ← 完整块边界
                  → [22, 23]

  block 1 (i=1): slot_start = 7*6 + 0 = 42
                  slot_end   = 7*6 + (10 - 1*4) = 46   ← end - i*block_size
                  → [42, 43, 44, 45]

slot_mapping = [22, 23, 42, 43, 44, 45]
```

### 6.2 Decode Slot Mapping

```
每条序列追加 1 个 token:
slot = block_table[-1] * block_size + last_block_num_tokens - 1
     = 最后一个块的起始槽位 + 块内已有 token 数 - 1 (0-based)
```

---

## 7. CUDA Graph 加速

### 7.1 捕获策略

```
渐进批次大小: [1, 2, 4, 8, 16, 32, 48, ..., max_bs]
从最大 bs 开始倒序捕获 → 共享 graph pool 减少内存

每个图:
  1. warmup: model(zero_input) [触发 CUDA JIT]
  2. capture: torch.cuda.graph(graph, pool):
       outputs = model(zero_input)
```

### 7.2 重放机制

```
decode 时 (bs ≤ 512, enforce_eager=False):
  1. 选择第一个 ≥ bs 的图大小
  2. 将实际输入复制到图的预分配缓冲区 [:bs]
  3. 未使用槽位填充 sentinel:
     - slot_mapping = -1  → store_kvcache kernel 跳过
     - context_lens = 0   → flash_attn 读取安全空值
  4. graph.replay()
  5. 从 outputs[:bs] 切片取实际结果
```

### 7.3 执行路径选择

```
run_model(input_ids, positions, is_prefill):
  if is_prefill or enforce_eager or bs > 512:
    → eager 模式 (model.forward + compute_logits)
  else:
    → CUDA Graph 重放
```

---

## 8. 模型结构 (Qwen3)

### 8.1 层级结构

```
Qwen3ForCausalLM
├── Qwen3Model
│   ├── VocabParallelEmbedding     ← TP 分片词表嵌入
│   ├── Qwen3DecoderLayer × N      ← Transformer 层堆叠
│   │   ├── RMSNorm (pre-attn)     ← fused residual: x+res → norm
│   │   ├── Qwen3Attention
│   │   │   ├── QKVParallelLinear  ← 融合 Q/K/V 投影 (TP 列并行)
│   │   │   ├── RMSNorm (q_norm)   ← QK-normalization (Qwen3 特有)
│   │   │   ├── RMSNorm (k_norm)
│   │   │   ├── RotaryEmbedding    ← RoPE
│   │   │   └── Attention          ← flash_attn + paged KV cache
│   │   ├── RMSNorm (post-attn)    ← fused residual
│   │   └── Qwen3MLP
│   │       ├── MergedColumnParallelLinear (gate+up, TP 列并行)
│   │       ├── SiluAndMul         ← SwiGLU 激活
│   │       └── RowParallelLinear  (down, TP 行并行 + all_reduce)
│   └── RMSNorm (final)            ← 最终归一化
└── ParallelLMHead                  ← 输出 logits (TP gather)
```

### 8.2 Fused Residual Stream

```
每个 DecoderLayer 的数据流:

  hidden_states, residual
        │
        ▼
  RMSNorm(hidden_states, residual)  ← fused: x+res → norm → (normalized, new_residual)
        │
        ▼
  SelfAttention(positions, hidden_states)
        │
        ▼
  RMSNorm(hidden_states, residual)  ← fused: attn_output+res → norm
        │
        ▼
  MLP(hidden_states)
        │
        ▼
  return hidden_states, residual

第一层: residual=None → 跳过残差加法
```

### 8.3 张量并行分片策略

```
类型                    维度     前向行为
─────────────────────────────────────────────────
ColumnParallelLinear    dim=0   各 rank 独立计算，输出沿 dim=-1 自然拼接
RowParallelLinear       dim=1   各 rank 独立计算 → all_reduce 求和
QKVParallelLinear       dim=0   融合 Q+K+V，列并行
MergedColumnParallel    dim=0   gate+up 两列合并，列并行
VocabParallelEmbedding  dim=0   词表分片 → mask + all_reduce
ParallelLMHead          dim=0   词表分片 → gather + cat
```

### 8.4 权重加载：打包模块映射

```python
# Qwen3ForCausalLM.packed_modules_mapping
{
    "q_proj":     ("qkv_proj",     "q"),   # HF 权重名 → (目标参数, 分片ID)
    "k_proj":     ("qkv_proj",     "k"),
    "v_proj":     ("qkv_proj",     "v"),
    "gate_proj":  ("gate_up_proj",  0),    # 0 = gate 部分
    "up_proj":    ("gate_up_proj",  1),    # 1 = up 部分
}
```

---

## 9. 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | (必填) | HuggingFace 模型路径 |
| `max_num_batched_tokens` | 16384 | 单次 prefill 批次总 token 上限 |
| `max_num_seqs` | 512 | 单步最大序列数 |
| `max_model_len` | 4096 | 最大上下文长度 |
| `gpu_memory_utilization` | 0.9 | KV cache 可用的 GPU 内存比例 |
| `tensor_parallel_size` | 1 | 张量并行 GPU 数 |
| `enforce_eager` | False | 禁用 CUDA Graph（调试用） |
| `kvcache_block_size` | 256 | KV cache 块大小（必须是 256 的倍数）|
| `num_kvcache_blocks` | -1 | 块数量（-1=自动计算）|

---

## 10. 调用示例

```python
from nanovllm import LLM, SamplingParams

# 初始化引擎
llm = LLM(
    model="/path/to/Qwen3-0.6B",
    enforce_eager=True,        # 调试时禁用 CUDA graph
    tensor_parallel_size=1,    # 单卡推理
)

# 批量生成
outputs = llm.generate(
    prompts=["你好，请介绍一下自己", "1+1等于几？"],
    sampling_params=SamplingParams(
        temperature=0.8,
        max_tokens=128,
    ),
)

for out in outputs:
    print(out["text"])
```

---

## 11. 进程模型

```
单卡 (tp_size=1):
  LLMEngine (主进程)
    ├── Scheduler
    └── ModelRunner (rank=0)

多卡 (tp_size=N):
  LLMEngine (主进程)
    ├── Scheduler
    ├── ModelRunner rank 0 (驱动进程)
    │     └── write_shm → SharedMemory → read_shm → ModelRunner rank 1..N-1
    └── mp.Process × (N-1) (工作进程, 在 loop() 中等待 IPC 指令)

NCCL 通信: tcp://localhost:2333
共享内存名: "nanovllm"
进程上下文: spawn (CUDA 要求)
```
