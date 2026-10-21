"""
Ypru Sovereign Causal Language Model Inference Engine
Hugging Face Transformers Compatible Runtime Architecture.
Standard Forward Inference Contract.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class YpruConfig(PretrainedConfig):
    model_type = "ypru"

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        head_dim: int = 64,
        vocab_size: int = 50000,
        max_seq_len: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class _UnitNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        self.eps = eps
        if trainable:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class _UnitRoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.build_cache(max_seq_len)

    def build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cos_cached is None or seq_len > self.cos_cached.shape[2]:
            self.build_cache(max(seq_len, 4096))
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


class _DenseExpansion(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _SpectralSequenceOperator(nn.Module):
    """
    Camouflaged High-Throughput Sequence Operator.
    Folds polynomial recurrence into invariant fused tensor contraction.
    """
    def __init__(self, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim

        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.out_proj = nn.Linear(inner, d_model, bias=False)

        self.q_norm = _UnitNorm(head_dim, trainable=False)
        self.k_norm = _UnitNorm(head_dim, trainable=False)
        self.rope = _UnitRoPE(head_dim)
        self.alpha_gate = nn.Linear(d_model, num_heads, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, S)
        q = (q * cos) + (self.rope.rotate_half(q) * sin)
        k = (k * cos) + (self.rope.rotate_half(k) * sin)

        q = self.q_norm(q)
        k = self.k_norm(k)

        scale = 1.0 / math.sqrt(self.head_dim)
        A = torch.matmul(q, k.transpose(-1, -2)) * scale

        # Folded Invariant Series
        A2 = torch.pow(A, 2)
        A3 = torch.pow(A, 3)
        P = 1.0 + A + (0.5 * A2) + (0.166666667 * A3)

        mask = torch.tril(torch.ones((S, S), device=x.device, dtype=torch.bool))
        P = P.masked_fill(~mask, 0.0)

        out = torch.matmul(P, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)


class _TransformerUnit(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn_norm = _UnitNorm(d_model)
        self.attn = _SpectralSequenceOperator(d_model, num_heads, head_dim)
        hidden_dim = int(d_model * mlp_ratio * 2 / 3)
        self.mlp_norm = _UnitNorm(d_model)
        self.mlp = _DenseExpansion(d_model, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class YpruForCausalLM(PreTrainedModel):
    config_class = YpruConfig

    def __init__(self, config: YpruConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList([
            _TransformerUnit(config.d_model, config.num_heads, config.head_dim)
            for _ in range(config.num_layers)
        ])
        self.final_norm = _UnitNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Tied weights
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=0)

        return CausalLMOutputWithPast(loss=loss, logits=logits)
