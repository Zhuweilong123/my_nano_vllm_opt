"""GPU 显存监控器：在主推理循环中采集 + 终端 ASCII 条形图原地刷新。

显存分类（四大类别）：
  Category 0 — 固有占用：驱动、CUDA 上下文、桌面环境等推理前已存在的显存基线
  Category 1 — KV Cache 池：allocate_kv_cache 分配的 5D 张量
  Category 2 — 模型权重：load_model 后模型参数占用的显存
  Category 3 — 激活值：推理过程中的中间张量、CUDA graph buffer 等（动态计算）

使用方式：
  1. 在模型加载前调用 calibrate() 记录基线
  2. 在权重加载后调用 record_weights() 记录模型显存
  3. 在 KV cache 分配后调用 record_kv_cache() 记录池大小
  4. 在推理循环中调用 refresh() 按间隔原地刷新（主线程，与 tqdm 共存）

注意：refresh() 必须在主线程中调用，不在独立线程运行，
      避免与 tqdm 的终端控制冲突。
"""

import re
import sys
import time
import unicodedata
import torch


def _strip_ansi(text: str) -> str:
    """去除 ANSI 转义序列，用于计算纯文本显示宽度。"""
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _char_width(ch: str) -> int:
    """单字符终端显示宽度：CJK/全角=2，其他=1。"""
    w = unicodedata.east_asian_width(ch)
    return 2 if w in ("W", "F") else 1


def _display_len(text: str) -> int:
    """计算去除 ANSI 码后的终端显示宽度（CJK 字符计为 2）。"""
    plain = _strip_ansi(text)
    return sum(_char_width(c) for c in plain)


def _cjk_ljust(text: str, width: int) -> str:
    """CJK 感知的左对齐填充，确保终端显示宽度为 width。"""
    current = _display_len(text)
    if current >= width:
        return text
    return text + " " * (width - current)


def _bytes_to_gib(num_bytes: int) -> str:
    """将字节数格式化为人类可读的 GiB 字符串。"""
    return f"{num_bytes / (1024**3):.2f} GiB"


def _bar_char(ratio: float) -> str:
    """根据占比返回对应的 1/8 精度条形图字符。"""
    steps = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    idx = min(int(ratio * 8), 7)
    return steps[idx]


class GPUMemoryMonitor:
    """GPU 显存监控器 — 在主线程中按间隔原地刷新 ANSI 彩色条形图。"""

    CATEGORY_NAMES = ["固有占用", "KV Cache", "模型权重", "激活值"]

    COLORS = {
        "固有占用": "\033[90m",
        "KV Cache":  "\033[32m",
        "模型权重":  "\033[34m",
        "激活值":    "\033[33m",
        "reset":     "\033[0m",
        "bold":      "\033[1m",
        "dim":       "\033[2m",
    }

    def __init__(
        self,
        model_name: str = "Unknown",
        interval: float = 1.0,
    ):
        self.model_name = model_name
        self.interval = interval

        self.baseline_memory = 0
        self.kv_cache_memory = 0
        self.weights_memory = 0
        self.total_memory = 0

        self._started = False
        self._first_frame = True
        self._num_lines = 0
        self._last_time = 0.0

        # 进度信息（替代 tqdm）
        self._progress_done = 0
        self._progress_total = 0
        self._prefill_speed = 0
        self._decode_speed = 0

    # ── 校准与记录 API ────────────────────────

    def calibrate(self):
        """[测量点 0] 模型加载前调用，记录基线显存。"""
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        self.baseline_memory = total - free
        self.total_memory = total

    def record_weights(self):
        """[测量点 2] load_model 后调用，记录模型权重显存。"""
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        current_used = total - free
        self.weights_memory = current_used - self.baseline_memory

    def record_kv_cache(self, kv_cache_bytes: int):
        """[测量点 1] 记录 KV cache 池的显存占用。"""
        self.kv_cache_memory = kv_cache_bytes

    # ── 主循环 API ────────────────────────────

    def set_progress(self, done: int, total: int, prefill_speed: int, decode_speed: int):
        """更新生成进度（替代 tqdm 的进度条）。"""
        self._progress_done = done
        self._progress_total = total
        self._prefill_speed = prefill_speed
        self._decode_speed = decode_speed

    def start(self):
        """输出初始帧并标记监控开始。必须在主线程调用。"""
        self._started = True
        self._last_time = time.time()
        self._snapshot()

    def refresh(self):
        """按间隔刷新监控框。在推理主循环中每次 step() 后调用。

        只有距上次刷新 >= interval 秒时才实际输出。
        """
        if not self._started or self.interval <= 0:
            return
        now = time.time()
        if now - self._last_time < self.interval:
            return
        self._last_time = now
        self._snapshot()

    def stop(self):
        """停止监控，输出最终快照后换行。"""
        if not self._started:
            return
        self._started = False
        self._snapshot()
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _snapshot(self):
        free, total = torch.cuda.mem_get_info()
        current_used = total - free

        activations = max(
            0,
            current_used - self.baseline_memory
            - self.weights_memory - self.kv_cache_memory,
        )

        categories = [
            self.baseline_memory,
            self.kv_cache_memory,
            self.weights_memory,
            activations,
        ]

        self._render(total, current_used, categories)

    def _render(self, total, used, categories):
        """渲染 + 原地刷新终端 ASCII 条形图。

        刷新策略：
        - 首次：直接打印，记录行数
        - 后续：光标上移 N 行，逐行覆写（不触碰下方的 tqdm 进度条）
        """
        width = 70
        bar_area = 30
        label_width = 12          # 标签列宽（终端显示列数，CJK 已处理）
        total_bar_chars = 26      # 条形图有效字符数

        used_rate = used / total * 100 if total > 0 else 0
        C = self.COLORS

        # ── 构建行 ──
        lines = []

        # 标题行 1：模型名 + 显存使用率
        title1 = (
            f"{C['bold']}  {self.model_name}  |  "
            f"GPU: {_bytes_to_gib(used)} / {_bytes_to_gib(total)} ({used_rate:.1f}%){C['reset']}"
        )
        pad1 = max(0, width - 2 - _display_len(title1))
        lines.append(f"┌{'─' * (width - 2)}┐")
        lines.append(f"│{title1}{' ' * pad1}│")

        # 标题行 2：生成进度 + 吞吐量（替代 tqdm）
        progress_bar_width = 20
        if self._progress_total > 0:
            ratio_done = self._progress_done / self._progress_total
            filled = int(ratio_done * progress_bar_width)
            pbar = "█" * filled + "░" * (progress_bar_width - filled)
            pct = ratio_done * 100
        else:
            pbar = "░" * progress_bar_width
            pct = 0.0
        title2 = (
            f"  Generating [{pbar}] {self._progress_done}/{self._progress_total} ({pct:.0f}%)"
            f"  |  Prefill: {self._prefill_speed} tok/s  Decode: {self._decode_speed} tok/s"
        )
        pad2 = max(0, width - 2 - _display_len(title2))
        lines.append(f"│{C['dim']}{title2}{' ' * pad2}{C['reset']}│")
        lines.append(f"├{'─' * (width - 2)}┤")

        # 各类别行
        for name, mem_bytes in zip(self.CATEGORY_NAMES, categories):
            ratio = mem_bytes / total if total > 0 else 0
            pct = ratio * 100
            color = C.get(name, "")

            bar_chars = int(ratio * total_bar_chars)
            bar = "█" * bar_chars
            frac = ratio * total_bar_chars - bar_chars
            if frac > 0:
                bar += _bar_char(frac)
            bar = bar.ljust(bar_area, " ")

            mem_str = _bytes_to_gib(mem_bytes)
            pct_str = f"({pct:5.1f}%)"
            # CJK 感知的标签对齐
            name_padded = _cjk_ljust(name, label_width)

            line = (
                f"│  {color}{name_padded}{C['reset']} "
                f"{C['dim']}│{C['reset']} "
                f"{color}{bar}{C['reset']} "
                f"{mem_str:>9s}  {pct_str:>8s} │"
            )
            lines.append(line)

        # 底栏
        lines.append(f"└{'─' * (width - 2)}┘")

        # ── 输出 ──
        if self._first_frame:
            sys.stdout.write("\n" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            self._num_lines = len(lines) + 1
            self._first_frame = False
        else:
            # 原地刷新：上移 N 行，逐行覆写
            # 不使用 \\033[J（会清除下方的 tqdm 进度条）
            # 框架行数固定，每次覆写正好覆盖上一帧
            sys.stdout.write(f"\033[{self._num_lines}A")
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            self._num_lines = len(lines)  # 不含 leading \n

    def print_summary(self):
        """输出一次性显存摘要。"""
        self._first_frame = True
        self._snapshot()
