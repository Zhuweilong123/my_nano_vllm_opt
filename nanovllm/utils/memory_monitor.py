"""GPU 显存监控器：终端条形图原地刷新 + matplotlib 饼图输出。

显存分类：
  Category 0 — 固有占用：驱动、CUDA 上下文等推理前已存在的显存基线
  Category 1 — KV Cache 池：allocate_kv_cache 分配的 5D 张量
  Category 2 — 模型权重：load_model 后模型参数占用的显存
  Category 3 — 激活值：推理过程中的中间张量、CUDA graph buffer 等（动态计算）

配置开关 show_memory_pie:
  True  → 每个采样周期保存一张 matplotlib 饼图到文件
  False → 仅终端条形图，不生成图片（默认）
"""

import os
import re
import sys
import time
import unicodedata
import torch


# ── 工具函数 ──────────────────────────────────────────

def _strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _char_width(ch: str) -> int:
    w = unicodedata.east_asian_width(ch)
    return 2 if w in ("W", "F") else 1


def _display_len(text: str) -> int:
    plain = _strip_ansi(text)
    return sum(_char_width(c) for c in plain)


def _cjk_ljust(text: str, width: int) -> str:
    current = _display_len(text)
    if current >= width:
        return text
    return text + " " * (width - current)


def _bytes_to_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.3f} GiB"


def _bar_char(ratio: float) -> str:
    steps = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    idx = min(int(ratio * 8), 7)
    return steps[idx]


def _percent(mem_bytes: int, total: int) -> float:
    """计算百分比，total=0 时返回 0。"""
    return mem_bytes / total * 100 if total > 0 else 0.0


# ── matplotlib 饼图 ───────────────────────────────────

# matplotlib 颜色（与终端 ANSI 色对应）
PIE_COLORS = ["#888888", "#44bb44", "#4488ff", "#ddbb44"]
PIE_LABELS = ["固有占用", "KV Cache", "模型权重", "激活值"]


def _render_matplotlib_pie(
    categories: list[int],
    total: int,
    model_name: str,
    output_path: str,
    used_rate: float,
):
    """使用 matplotlib 渲染饼图并保存到文件。

    Args:
        categories: 四个类别的字节数 [baseline, kv_cache, weights, activations]
        total: 总字节数
        model_name: 模型名（显示在标题）
        output_path: 输出 PNG 文件路径
        used_rate: 显存使用率百分比
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 注册 CJK 字体（Noto Sans CJK SC，系统自带）
    _cjk_font = None
    for fpath in fm.findSystemFonts():
        if "NotoSansCJK" in fpath and "Regular" in fpath:
            _cjk_font = fpath
            break
    if _cjk_font:
        _cjk_prop = fm.FontProperties(fname=_cjk_font)
    else:
        _cjk_prop = None

    # 过滤掉为 0 的类别（不显示在饼图中）
    sizes = []
    labels = []
    colors = []
    for mem, label, color in zip(categories, PIE_LABELS, PIE_COLORS):
        if mem > 0:
            sizes.append(mem)
            labels.append(label)
            colors.append(color)

    if not sizes:
        return

    fig, (ax_pie, ax_table) = plt.subplots(
        1, 2, figsize=(12, 6),
        gridspec_kw={"width_ratios": [1, 0.55]},
    )

    # ── 饼图 ──
    wedges, texts, autotexts = ax_pie.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.6,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(11)

    # 中心文字：总显存
    centre_text = f"{_bytes_to_gib(total)}\n({used_rate:.2f}%)"
    ax_pie.text(
        0, 0, centre_text,
        ha="center", va="center",
        fontsize=13, fontweight="bold",
    )
    title = f"GPU Memory — {model_name}"
    ax_pie.set_title(title, fontsize=14, fontweight="bold", pad=18, fontproperties=_cjk_prop)

    # ── 图例表格 ──
    ax_table.axis("off")
    ax_table.set_xlim(0, 10)
    ax_table.set_ylim(0, len(sizes) + 1.5)

    table_data = []
    for label, color, mem in zip(labels, colors, sizes):
        pct = _percent(mem, total)
        table_data.append([
            f"{label}",
            f"{_bytes_to_gib(mem)}",
            f"{pct:.1f}%",
        ])

    # 表头
    col_labels = ["类别", "显存", "占比"]
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        colWidths=[0.28, 0.25, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # 给表格行着色 + 设置 CJK 字体
    for i, color in enumerate(colors):
        for j in range(3):
            cell = table[i + 1, j]
            cell.set_facecolor(color + "30")
            cell.set_text_props(weight="bold", fontproperties=_cjk_prop)

    # 标题行样式
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold", fontsize=11, fontproperties=_cjk_prop)
        cell.set_edgecolor("#cccccc")

    plt.tight_layout(pad=2)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── 监控器类 ──────────────────────────────────────────

class GPUMemoryMonitor:
    """GPU 显存监控器 — 终端条形图 + matplotlib 饼图。"""

    CATEGORY_NAMES = PIE_LABELS

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
        show_pie: bool = False,
        pie_output_dir: str = "",
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_name = model_name
        self.interval = interval
        self.show_pie = show_pie
        self.pie_output_dir = pie_output_dir or os.getcwd()
        self.gpu_memory_utilization = gpu_memory_utilization

        self.baseline_memory = 0
        self.kv_cache_memory = 0
        self.weights_memory = 0
        self.total_memory = 0

        self._started = False
        self._first_frame = True
        self._num_lines = 0
        self._last_time = 0.0

        self._progress_done = 0
        self._progress_total = 0
        self._prefill_speed = 0
        self._decode_speed = 0
        self._pie_start_saved = False   # 是否已保存开始饼图

    # ── 校准与记录 ──────────────────────────────

    def calibrate(self):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        self.baseline_memory = total - free
        self.total_memory = total

    def record_weights(self):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        current_used = total - free
        self.weights_memory = current_used - self.baseline_memory

    def record_kv_cache(self, kv_cache_bytes: int):
        self.kv_cache_memory = kv_cache_bytes

    # ── 主循环 API ──────────────────────────────

    def set_progress(self, done: int, total: int, prefill_speed: int, decode_speed: int):
        self._progress_done = done
        self._progress_total = total
        self._prefill_speed = prefill_speed
        self._decode_speed = decode_speed
        # 首次设置进度 → 保存推理开始时的饼图
        if self.show_pie and not self._pie_start_saved and total > 0:
            self._save_pie()
            self._pie_start_saved = True

    def start(self):
        self._started = True
        self._last_time = time.time()
        self._snapshot()
        self._pie_start_saved = False  # 每次 generate 重新计时

    def refresh(self):
        if not self._started or self.interval <= 0:
            return
        now = time.time()
        if now - self._last_time < self.interval:
            return
        self._last_time = now
        self._snapshot()

    def stop(self):
        if not self._started:
            return
        self._started = False
        self._snapshot()          # 最后一次条形图刷新
        if self.show_pie:
            self._save_pie(final=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    # ── 内部 ────────────────────────────────────

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

    def _save_pie(
        self,
        categories: list[int] | None = None,
        total: int = 0,
        used: int = 0,
        final: bool = False,
    ):
        """保存 matplotlib 饼图到文件。

        目录结构: {pie_output_dir}/{model_name}/
        文件名: {gpu_memory_utilization}_{num_prompts}[_final].png
        """
        if categories is None:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            categories = [
                self.baseline_memory,
                self.kv_cache_memory,
                self.weights_memory,
                max(0, used - self.baseline_memory - self.weights_memory - self.kv_cache_memory),
            ]
            total = total

        # 子目录: 按模型名划分
        out_dir = os.path.join(self.pie_output_dir, self.model_name)
        os.makedirs(out_dir, exist_ok=True)

        # 文件名: gpu_util_numPrompts[_final].png
        base = f"{self.gpu_memory_utilization}_{self._progress_total}"
        filename = f"{base}_final.png" if final else f"{base}.png"
        path = os.path.join(out_dir, filename)

        used_rate = used / total * 100 if total > 0 else 0
        _render_matplotlib_pie(categories, total, self.model_name, path, used_rate)

    # ── 终端条形图渲染 ──────────────────────────

    def _render(self, total, used, categories):
        width = 70
        bar_area = 30
        label_width = 12
        total_bar_chars = 26
        C = self.COLORS
        used_rate = used / total * 100 if total > 0 else 0

        lines = []

        # 标题行 1
        title1 = (
            f"{C['bold']}  {self.model_name}  |  "
            f"GPU: {_bytes_to_gib(used)} / {_bytes_to_gib(total)} ({used_rate:.1f}%){C['reset']}"
        )
        pad1 = max(0, width - 2 - _display_len(title1))
        lines.append(f"┌{'─' * (width - 2)}┐")
        lines.append(f"│{title1}{' ' * pad1}│")

        # 标题行 2：生成进度
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

        # 各类别条形图
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
            name_padded = _cjk_ljust(name, label_width)

            line = (
                f"│  {color}{name_padded}{C['reset']} "
                f"{C['dim']}│{C['reset']} "
                f"{color}{bar}{C['reset']} "
                f"{mem_str:>9s}  {pct_str:>8s} │"
            )
            lines.append(line)

        lines.append(f"└{'─' * (width - 2)}┘")

        # 输出
        out = "\n".join(lines)
        if self._first_frame:
            sys.stdout.write("\n" + out + "\n")
            sys.stdout.flush()
            self._num_lines = len(lines) + 1
            self._first_frame = False
        else:
            sys.stdout.write(f"\033[{self._num_lines}A")
            sys.stdout.write(out + "\n")
            sys.stdout.flush()
            self._num_lines = len(lines)

    def print_summary(self):
        self._first_frame = True
        self._snapshot()
