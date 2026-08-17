from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from cmf.config import TrainConfig


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 8 / 3):
        super().__init__()
        hidden = int(dim * mult)
        hidden = (hidden + 7) // 8 * 8
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w3(nn.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    """RoPE multi-head attention via mlx.core.fast SDPA. ctx=None is self-attn."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def _shape(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        return x.reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, ctx: mx.array | None = None) -> mx.array:
        kv = x if ctx is None else ctx
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(kv))
        v = self._shape(self.v_proj(kv))
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=10_000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=10_000.0, scale=1.0, offset=0)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = y.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.o_proj(y)


class DualStreamBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.n_fast = nn.RMSNorm(dim)
        self.n_slow = nn.RMSNorm(dim)
        self.self_fast = Attention(dim, heads)
        self.self_slow = Attention(dim, heads)
        self.n_fast_x = nn.RMSNorm(dim)
        self.n_slow_x = nn.RMSNorm(dim)
        self.cross_fast = Attention(dim, heads)  # fast queries slow
        self.cross_slow = Attention(dim, heads)  # slow queries fast — the lag edge
        self.n_fast_ff = nn.RMSNorm(dim)
        self.n_slow_ff = nn.RMSNorm(dim)
        self.ff_fast = SwiGLU(dim)
        self.ff_slow = SwiGLU(dim)
        self.drop = nn.Dropout(dropout)

    def __call__(self, fast: mx.array, slow: mx.array) -> tuple[mx.array, mx.array]:
        fast = fast + self.drop(self.self_fast(self.n_fast(fast)))
        slow = slow + self.drop(self.self_slow(self.n_slow(slow)))
        fast = fast + self.drop(self.cross_fast(self.n_fast_x(fast), self.n_slow_x(slow)))
        slow = slow + self.drop(self.cross_slow(self.n_slow_x(slow), self.n_fast_x(fast)))
        fast = fast + self.drop(self.ff_fast(self.n_fast_ff(fast)))
        slow = slow + self.drop(self.ff_slow(self.n_slow_ff(slow)))
        return fast, slow


class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        w = mx.softmax(self.score(x).squeeze(-1), axis=-1)
        return mx.sum(x * mx.expand_dims(w, -1), axis=1)


@dataclass
class ModelOut:
    logits: mx.array
    value: mx.array
    p_up: mx.array
    next_ret: mx.array
    lag: mx.array
    z: mx.array


class FusionModel(nn.Module):
    """
    Dual-stream fusion transformer.

    Fast stream  = Binance microstructure (T x 24)
    Slow stream  = Polymarket CLOB (T x 16)
    Cross-attn   = each venue reads the other; slow-attends-fast is the lag channel
    Context      = position + Hayashi-Yoshida lag features, gated into the pooled state
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.fast_in = nn.Linear(cfg.fast_dim, d)
        self.slow_in = nn.Linear(cfg.slow_dim, d)
        self.blocks = [DualStreamBlock(d, cfg.heads, cfg.dropout) for _ in range(cfg.layers)]
        self.pool_fast = AttnPool(d)
        self.pool_slow = AttnPool(d)
        self.ctx = nn.Linear(cfg.pos_dim + cfg.lag_dim, d)
        self.gate = nn.Linear(3 * d, d)
        self.norm = nn.RMSNorm(d)
        self.policy = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, cfg.n_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.p_up = nn.Linear(d, 1)
        self.next_ret = nn.Linear(d, 1)
        self.lag_head = nn.Linear(d, 1)

    def encode(self, fast: mx.array, slow: mx.array, pos: mx.array, lag: mx.array) -> mx.array:
        h_f = self.fast_in(fast)
        h_s = self.slow_in(slow)
        for block in self.blocks:
            h_f, h_s = block(h_f, h_s)
        z_f = self.pool_fast(h_f)
        z_s = self.pool_slow(h_s)
        z_c = self.ctx(mx.concatenate([pos, lag], axis=-1))
        g = mx.sigmoid(self.gate(mx.concatenate([z_f, z_s, z_c], axis=-1)))
        return self.norm(g * z_f + (1.0 - g) * z_s + z_c)

    def __call__(self, fast: mx.array, slow: mx.array, pos: mx.array, lag: mx.array) -> ModelOut:
        z = self.encode(fast, slow, pos, lag)
        return ModelOut(
            logits=self.policy(z),
            value=self.value(z).squeeze(-1),
            p_up=self.p_up(z).squeeze(-1),
            next_ret=self.next_ret(z).squeeze(-1),
            lag=self.lag_head(z).squeeze(-1),
            z=z,
        )

    def count_params(self) -> int:
        from mlx.utils import tree_flatten

        return int(sum(v.size for _, v in tree_flatten(self.parameters())))
