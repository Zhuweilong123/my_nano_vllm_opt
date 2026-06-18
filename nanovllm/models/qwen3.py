"""Qwen3 模型定义：通过组合各层组件构建完整的 Causal LM。

架构（标准 pre-norm Transformer）：
  Qwen3ForCausalLM
    ├── Qwen3Model
    │     ├── VocabParallelEmbedding       (TP-sharded input embedding)
    │     ├── Qwen3DecoderLayer × N        (Transformer stack)
    │     │     ├── RMSNorm                (pre-attn norm, fused residual)
    │     │     ├── Qwen3Attention
    │     │     │     ├── QKVParallelLinear (fused Q/K/V projection)
    │     │     │     ├── RMSNorm (q_norm/k_norm)  (QK-normalization)
    │     │     │     ├── RoPE
    │     │     │     └── Attention (flash_attn + paged KV cache)
    │     │     ├── RMSNorm                (post-attn norm, fused residual)
    │     │     └── Qwen3MLP
    │     │           ├── MergedColumnParallelLinear (gate+up)
    │     │           ├── SiluAndMul       (SwiGLU)
    │     │           └── RowParallelLinear (down projection)
    │     └── RMSNorm                      (final norm)
    └── ParallelLMHead                      (output logits)
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3 的自注意力模块。

    关键特性：
    - 融合 QKV 投影（单次 linear 替代三次）
    - QK-normalization：在 RoPE 之前对 Q 和 K 分别做 RMSNorm（Qwen3 特有优化）
    - GQA 支持（K/V head 数可少于 Q head 数）
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            # Qwen3 特有的 QK-normalization：在 RoPE 之前对 Q 和 K 分别做 RMSNorm
            # 当 qkv_bias=False 时启用，稳定训练并改善长文本性能
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 1. 融合 QKV 投影（一次 linear 替代三次独立投影）
        qkv = self.qkv_proj(hidden_states)
        # 2. 分离 Q/K/V（K 和 V 大小相同：num_kv_heads * head_dim）
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 3. QK-normalization（RoPE 之前，Qwen3 特有）
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 4. RoPE 位置编码
        q, k = self.rotary_emb(positions, q, k)
        # 5. flash_attn（带分页 KV cache）
        o = self.attn(q, k, v)
        # 6. 输出投影（行并行，隐含 all_reduce）
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused residual stream 模式。

        每个子层（attention、MLP）在归一化后的副本上操作，
        残差通过 add_rms_forward 原地更新，避免额外的内存分配。

        residual=None 表示第一层，无需残差加法。
        """
        if residual is None:
            # 第一层：无残差输入，直接归一化
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # fused: hidden_states + residual → RMSNorm，同时返回新的 residual
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        # fused post-attn: attention_output + residual → RMSNorm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    # HuggingFace checkpoint 权重名 → 融合参数名 + 子分片 ID 的映射
    # 用于 loader.load_model 将独立的分片权重路由到融合参数中的正确位置
    # 例如：checkpoint "q_proj.weight" → qkv_proj 参数, shard "q"
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),   # 0 = 合并列的前半部分
        "up_proj": ("gate_up_proj", 1),      # 1 = 后半部分
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
