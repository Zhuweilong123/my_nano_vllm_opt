# nano-vllm 环境搭建与显存监控模块开发记录

## 1. GitHub 网络连接问题

### 现象
```bash
git clone https://github.com/GeeeekExplorer/nano-vllm.git
# error: RPC 失败。curl 28 Failed to connect to github.com port 443 after 136127 ms
```

### 根因
中国网络环境下 GitHub HTTPS 443 端口被限流/阻断，但 **SSH 22 端口正常**。

### 解决
改用 SSH 协议连接 GitHub：
```bash
# 生成 SSH key
ssh-keygen -t ed25519 -C "your@email" -f ~/.ssh/id_ed25519

# 添加公钥到 GitHub Settings → SSH Keys
# 推送时使用 SSH URL
git remote set-url origin git@github.com:Zhuweilong123/my_nano_vllm_opt.git
```

---

## 2. PyTorch CUDA 驱动兼容性问题

### 现象
```python
import torch
torch.cuda.is_available()  # False
# UserWarning: NVIDIA driver is too old (found version 12020)
# torch.__version__ = 2.12.1+cu130
```

### 根因
| 组件 | 版本 | 说明 |
|------|------|------|
| NVIDIA 驱动 | 535.309.01 | 支持到 CUDA 12.2 |
| conda CUDA Toolkit | 13.0 | 远超驱动支持范围 |
| PyTorch | 2.12.1+cu130 | 需要 CUDA 13.0 运行时 |

PyTorch +cu130 版本编译时链接 CUDA 13.0，在 CUDA 12.2 驱动上无法初始化。

### 解决
降级 PyTorch 到与驱动兼容的 CUDA 12.1 版本：
```bash
pip uninstall torch triton -y
pip install torch==2.5.1 -i https://mirrors.ustc.edu.cn/pypi/web/simple/
```
降级后 `torch.cuda.is_available()` → `True`。

---

## 3. flash-attn 编译失败

### 现象
```
ModuleNotFoundError: No module named 'flash_attn'
OSError: CUDA_HOME environment variable is not set
```

### 根因
`flash-attn` 只有源码分发包（sdist），需要 `nvcc` 编译器从源码编译。conda 环境中的 `nvidia-cuda-nvcc-cu12` 包只包含 ptxas（PTX 汇编器），不含 nvcc（CUDA C++ 编译器）。

### 解决（需用户手动完成）
```bash
# 方案A：通过 apt 安装完整 CUDA Toolkit
sudo apt-get install -y nvidia-cuda-toolkit

# 方案B：通过 conda 安装（较慢）
conda install -c nvidia cuda-nvcc -y

# 然后安装 flash-attn
pip install flash-attn --no-build-isolation
```

---

## 4. 显存监控模块开发

### 4.1 需求

1. **独立模块** `nanovllm/utils/memory_monitor.py`
2. **四大显存类别**：固有占用 / KV Cache 池 / 模型权重 / 激活值
3. **终端 ASCII 条形图可视化**，标注模型名和显存使用率
4. **监控周期可配置**，默认 1 秒
5. **原地刷新**，不影响其他终端输出

### 4.2 架构设计

#### 测量锚点

在 `ModelRunner.__init__()` 中按顺序插入 3 个锚点：

| 锚点 | 时机 | 类别 | 计算方法 |
|------|------|------|---------|
| `calibrate()` | NCCL 初始化后、模型加载前 | Category 0（基线） | `torch.cuda.mem_get_info()` |
| `record_weights()` | `load_model()` 后 | Category 2（权重） | `当前总量 - 基线` |
| `record_kv_cache()` | `allocate_kv_cache()` 后 | Category 1（KV Cache） | `kv_cache.numel() * element_size()` |

Category 3（激活值）动态计算：`当前总量 - (基线 + 权重 + KV Cache)`。

#### 显存分类彩色 ASCII 可视化

```
┌────────────────────────────────────────────────────────────────────┐
│  Qwen2.5-1.5B  |  GPU: 7.03 / 11.76 GiB (59.8%)                   │
│  Generating [████████░░░░░░░░░░░░] 3/10 (30%) | Prefill: 150 tok/s │
├────────────────────────────────────────────────────────────────────┤
│  固有占用     │ ██▏                             0.99 GiB  (  8.5%) │
│  KV Cache     │ █████████▎                      4.22 GiB  ( 35.9%) │
│  模型权重     │ ███████▊                        3.55 GiB  ( 30.2%) │
│  激活值       │ █████▏                          2.35 GiB  ( 20.0%) │
└────────────────────────────────────────────────────────────────────┘
```

| 类别 | 颜色 | 说明 |
|------|------|------|
| 固有占用 | 灰色 `\033[90m` | CUDA 驱动/桌面环境等 |
| KV Cache | 绿色 `\033[32m` | 分页注意力缓存池 |
| 模型权重 | 蓝色 `\033[34m` | nn.Parameter |
| 激活值 | 黄色 `\033[33m` | 中间张量 + CUDA graph buffer |

---

### 4.3 问题迭代

#### 问题 1：ANSI 颜色码导致表格撕裂

**现象**：
```
┌────────────────┐
│  模型名  ...   │         ← 右边框没有对齐到 70 列
```

**根因**：`len(title)` 把 `\033[1m`、`\033[0m` 等不可见 ANSI 控制码计入字符长度，导致 `pad` 计算错误，行宽溢出换行。

**解决**：新增 `_strip_ansi()` 和 `_display_len()` 函数，计算宽度时先剔除 ANSI 控制序列。

---

#### 问题 2：CJK 中文字符对齐

**现象**：
```
  固有占用 │ ████████
  KV Cache  │ ████████         ← 条形图起点不一致
  模型权重   │ ████████
  激活值     │ ████████
```

**根因**：Python 的 `str.ljust()` 按字符数填充而非终端显示宽度。中文字符占 2 列，`"激活值"(3汉字=6列)` 被 `ljust(10)` 当成 3 字符加 7 空格(=7列)，共占 13 列；`"KV Cache"(8字符=8列)` 加 2 空格占 10 列。

**解决**：新增 `_cjk_ljust()` 函数，通过 `unicodedata.east_asian_width()` 判断每个字符的终端列宽（CJK/全角=2，ASCII=1），补齐到统一显示宽度。

---

#### 问题 3：后台守护线程与 tqdm 终端控制权竞态

**现象**：
```
Generating: 0%|...| 0/1 [00:00<?]┌──────────────────┐
│ ...                              │
└──────────────────┘0,  1.23it/s]
```

监控框、进度条、文本碎片混在一起。

**根因**：

```
后台守护线程 (monitor)          主线程 (tqdm)
     │                              │
     │ write("\033[10A")            │
     │                              │ write("\r" + bar_text)  ← \r 回到行首
     │ write(frame_line_1)          │
     │ ...                          │ write("\r" + new_bar)   ← tqdm 内部守护线程周期性刷新
     │                              │
     ▼                              ▼
           stdout 交错写入，终端不可控
```

tqdm 内部有一个**守护线程**周期性调用 `refresh()`，向 stdout 写 `\r` + 进度条文本。监控器的 `\033[{N}A`（上移光标）+ 多行覆写与 tqdm 的 `\r` 时序完全不可控，必然产生撕裂。

**尝试过的方案**：

| 尝试 | 效果 | 失败原因 |
|------|------|---------|
| `tqdm.write()` | 进度条消失 | `write()` 每次都输出新块，不原地刷新 |
| `external_write_mode` | 仍然撕裂 | 多行输出与 tqdm 内部守护线程依然竞态 |
| 主线程 `refresh()` 钩子 | 首帧粘连进度条 | `start()` 在 tqdm 之后调用，首次 `\n` + tqdm `\r` 冲突 |
| `start()` 移到 tqdm 之前 | 完全混乱 | tqdm 守护线程周期性 `\r` 刷新仍然破坏 `\033[{N}A` |

**最终方案：完全移除 tqdm，进度嵌入监控框**。

将生成进度 `Generating [████░░░] N/M (X%)` 和吞吐量 `Prefill: X tok/s  Decode: Y tok/s` 嵌入监控框的第二标题行。监控框通过 `\033[{N}A` + 逐行覆写实现原地刷新。单线程独占 stdout，无竞态。

关键代码变更：

```python
# llm_engine.py — generate() 中
monitor.set_progress(finished_count, total_prompts,
                     int(prefill_speed), int(decode_speed))

# 替代了原来的:
# pbar = tqdm(total=len(prompts))
# pbar.set_postfix({...})
# pbar.update(1)
```

---

### 4.4 最终文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `nanovllm/utils/memory_monitor.py` | **新建** | 监控器核心：校准/记录/刷新/渲染 |
| `nanovllm/config.py` | +2 行 | 新增 `memory_monitor_interval: float = 1.0` |
| `nanovllm/engine/model_runner.py` | +21 行 | 插入 calibrate/record_weights/record_kv_cache 锚点 |
| `nanovllm/engine/llm_engine.py` | 修改 | 移除 tqdm，改用 monitor.set_progress() |

---

### 4.5 使用方式

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/model",
    memory_monitor_interval=1.0,  # 监控周期（秒），0=禁用
)
outputs = llm.generate(prompts, SamplingParams())
```

运行时终端实时显示：
```
┌────────────────────────────────────────────────────────────────────┐
│  Qwen2.5-1.5B  |  GPU: 7.03 / 11.76 GiB (59.8%)                   │
│  Generating [████████████████░░░░] 2/2 (100%) | Prefill: 6 Decode:│
├────────────────────────────────────────────────────────────────────┤
│  固有占用     │ ██▏                             0.99 GiB  (  8.5%) │
│  KV Cache     │ █████████▎                      4.22 GiB  ( 35.9%) │
│  模型权重     │ ███████▊                        3.55 GiB  ( 30.2%) │
│  激活值       │ █████▏                          2.35 GiB  ( 20.0%) │
└────────────────────────────────────────────────────────────────────┘
```
