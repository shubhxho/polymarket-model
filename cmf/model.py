from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from cmf.config import TrainConfig


def _rms(x: mx.array) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def _valid_tokens(x: mx.array) -> mx.array:
    """True where a history row is not left-padding."""
    return mx.sum(mx.abs(x), axis=-1) > 0


def _alibi(heads: int, tq: int, tk: int) -> mx.array:
    """[H, Tq, Tk] local-attention bias (Press et al., ALiBi)."""
    i = mx.arange(tq).astype(mx.float32)[:, None]
    j = mx.arange(tk).astype(mx.float32)[None, :]
    dist = mx.abs(i - j)
    slopes = 1.0 / mx.power(2.0, mx.linspace(1.0, 8.0, heads))
    return -slopes[:, None, None] * dist[None, :, :]


def _pad_mask(valid_k: mx.array, tq: int, heads: int, alibi: mx.array | None) -> mx.array:
    """Additive SDPA mask [B, H, Tq, Tk]."""
    pad = mx.where(valid_k[:, None, None, :], 0.0, -1.0e9)
    pad = mx.broadcast_to(pad, (valid_k.shape[0], heads, tq, valid_k.shape[1]))
    if alibi is None:
        return pad
    return pad + alibi[None, :, :, :]


def _pool4(x: mx.array) -> mx.array:
    b, t, d = x.shape
    t4 = t // 4
    if t4 == 0:
        return x
    return mx.mean(x[:, : t4 * 4, :].reshape(b, t4, 4, d), axis=2)


def _unpool4(x: mx.array, t: int) -> mx.array:
    y = mx.repeat(x, 4, axis=1)
    if y.shape[1] < t:
        pad = mx.repeat(y[:, -1:, :], t - y.shape[1], axis=1)
        y = mx.concatenate([y, pad], axis=1)
    return y[:, :t, :]


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


class FeatureGate(nn.Module):
    """TFT-style per-tick gate so dead microstructure channels drop out."""

    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.gate = nn.Linear(in_dim, in_dim)
        self.proj = nn.Linear(in_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x * mx.sigmoid(self.gate(x)))


class CondRMS(nn.Module):
    """RMSNorm modulated by a shared FiLM from position + lag."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

    def __call__(self, x: mx.array, scale: mx.array, shift: mx.array) -> mx.array:
        return self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]


class Attention(nn.Module):
    """RoPE + QK-norm multi-head attention via mlx.core.fast SDPA."""

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

    def __call__(
        self,
        x: mx.array,
        ctx: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        kv = x if ctx is None else ctx
        q = _rms(self._shape(self.q_proj(x)))
        k = _rms(self._shape(self.k_proj(kv)))
        v = self._shape(self.v_proj(kv))
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=10_000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=10_000.0, scale=1.0, offset=0)
        if mask is None:
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.o_proj(y)


class DualStreamBlock(nn.Module):
    def __init__(self, dim: int, heads: int, cond_dim: int, dropout: float, ls_init: float):
        super().__init__()
        self.heads = heads
        self.mod_fast = nn.Linear(cond_dim, 2 * dim)
        self.mod_slow = nn.Linear(cond_dim, 2 * dim)
        self.n_fast = CondRMS(dim)
        self.n_slow = CondRMS(dim)
        self.self_fast = Attention(dim, heads)
        self.self_slow = Attention(dim, heads)
        self.n_fast_x = CondRMS(dim)
        self.n_slow_x = CondRMS(dim)
        self.cross_fast = Attention(dim, heads)
        self.cross_slow = Attention(dim, heads)
        self.n_fast_ff = CondRMS(dim)
        self.n_slow_ff = CondRMS(dim)
        self.ff_fast = SwiGLU(dim)
        self.ff_slow = SwiGLU(dim)
        self.drop = nn.Dropout(dropout)
        self.ls_sf = mx.ones((dim,)) * ls_init
        self.ls_ss = mx.ones((dim,)) * ls_init
        self.ls_cf = mx.ones((dim,)) * ls_init
        self.ls_cs = mx.ones((dim,)) * ls_init
        self.ls_ff = mx.ones((dim,)) * ls_init
        self.ls_fs = mx.ones((dim,)) * ls_init

    def __call__(
        self,
        fast: mx.array,
        slow: mx.array,
        cond: mx.array,
        valid_f: mx.array,
        valid_s: mx.array,
    ) -> tuple[mx.array, mx.array]:
        sf, sh = mx.split(self.mod_fast(cond), 2, axis=-1)
        of, oh = mx.split(self.mod_slow(cond), 2, axis=-1)
        tf, ts = fast.shape[1], slow.shape[1]
        alibi_f = _alibi(self.heads, tf, tf)
        alibi_s = _alibi(self.heads, ts, ts)
        alibi_fs = _alibi(self.heads, tf, ts)
        alibi_sf = _alibi(self.heads, ts, tf)
        m_ff = _pad_mask(valid_f, tf, self.heads, alibi_f)
        m_ss = _pad_mask(valid_s, ts, self.heads, alibi_s)
        m_fs = _pad_mask(valid_s, tf, self.heads, alibi_fs)
        m_sf = _pad_mask(valid_f, ts, self.heads, alibi_sf)

        nf, ns = self.n_fast(fast, sf, sh), self.n_slow(slow, of, oh)
        fast = fast + self.drop(self.ls_sf * self.self_fast(nf, mask=m_ff))
        slow = slow + self.drop(self.ls_ss * self.self_slow(ns, mask=m_ss))
        nfx, nsx = self.n_fast_x(fast, sf, sh), self.n_slow_x(slow, of, oh)
        fast = fast + self.drop(self.ls_cf * self.cross_fast(nfx, nsx, mask=m_fs))
        # re-norm after the fast update so the lag edge sees the new lead state
        nfx, nsx = self.n_fast_x(fast, sf, sh), self.n_slow_x(slow, of, oh)
        slow = slow + self.drop(self.ls_cs * self.cross_slow(nsx, nfx, mask=m_sf))
        fast = fast + self.drop(self.ls_ff * self.ff_fast(self.n_fast_ff(fast, sf, sh)))
        slow = slow + self.drop(self.ls_fs * self.ff_slow(self.n_slow_ff(slow, of, oh)))
        fast = fast * valid_f[:, :, None]
        slow = slow * valid_s[:, :, None]
        return fast, slow


class QueryPool(nn.Module):
    """Learned query over the stream, mixed with the last valid tick."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query = mx.random.normal((1, 1, dim)) * 0.02
        self.attn = Attention(dim, heads)
        self.mix = nn.Linear(2 * dim, dim)

    def __call__(self, x: mx.array, valid: mx.array) -> mx.array:
        b, t, _ = x.shape
        q = mx.broadcast_to(self.query, (b, 1, x.shape[-1]))
        mask = _pad_mask(valid, 1, self.heads, None)
        pooled = self.attn(q, x, mask=mask).squeeze(1)
        idx = mx.maximum(mx.sum(valid.astype(mx.int32), axis=-1) - 1, 0)
        last = x[mx.arange(b), idx]
        return self.mix(mx.concatenate([pooled, last], axis=-1))


@dataclass
class ModelOut:
    logits: mx.array
    value: mx.array
    p_up: mx.array
    next_ret: mx.array
    lag: mx.array
    z: mx.array
    uncertainty: mx.array
    log_var: mx.array


class FusionModel(nn.Module):
    """
    CMF-2 dual-stream fusion transformer.

    Fast stream  = Binance microstructure (T x 24)
    Slow stream  = Polymarket CLOB (T x 16)
    Cross-attn   = each venue reads the other; slow-attends-fast is the lag channel
    Context      = FiLM from position + Hayashi-Yoshida lag into every block
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.heads = cfg.heads
        cond = d
        self.fast_in = FeatureGate(cfg.fast_dim, d)
        self.slow_in = FeatureGate(cfg.slow_dim, d)
        self.cond_in = nn.Sequential(
            nn.Linear(cfg.pos_dim + cfg.lag_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.blocks = [
            DualStreamBlock(d, cfg.heads, cond, cfg.dropout, cfg.layer_scale) for _ in range(cfg.layers)
        ]
        self.coarse = [
            DualStreamBlock(d, cfg.heads, cond, cfg.dropout, cfg.layer_scale) for _ in range(cfg.coarse_layers)
        ]
        self.pool_fast = QueryPool(d, cfg.heads)
        self.pool_slow = QueryPool(d, cfg.heads)
        self.gate = nn.Linear(3 * d, d)
        self.norm = nn.RMSNorm(d)
        self.policy = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, cfg.n_actions))
        self.value = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.p_up = nn.Linear(d, 1)
        self.next_ret = nn.Linear(d, 1)
        self.lag_head = nn.Linear(d, 1)
        self.unc = nn.Linear(d, 1)
        self.log_temp = mx.zeros((1,))

    def encode(self, fast: mx.array, slow: mx.array, pos: mx.array, lag: mx.array) -> mx.array:
        valid_f = _valid_tokens(fast)
        valid_s = _valid_tokens(slow)
        cond = self.cond_in(mx.concatenate([pos, lag], axis=-1))
        h_f = self.fast_in(fast) * valid_f[:, :, None]
        h_s = self.slow_in(slow) * valid_s[:, :, None]
        for block in self.blocks:
            h_f, h_s = block(h_f, h_s, cond, valid_f, valid_s)
        if self.coarse:
            c_f, c_s = _pool4(h_f), _pool4(h_s)
            vf = _valid_tokens(c_f)
            vs = _valid_tokens(c_s)
            for block in self.coarse:
                c_f, c_s = block(c_f, c_s, cond, vf, vs)
            h_f = h_f + _unpool4(c_f, h_f.shape[1])
            h_s = h_s + _unpool4(c_s, h_s.shape[1])
        z_f = self.pool_fast(h_f, valid_f)
        z_s = self.pool_slow(h_s, valid_s)
        g = mx.sigmoid(self.gate(mx.concatenate([z_f, z_s, cond], axis=-1)))
        return self.norm(g * z_f + (1.0 - g) * z_s + cond)

    def __call__(self, fast: mx.array, slow: mx.array, pos: mx.array, lag: mx.array) -> ModelOut:
        z = self.encode(fast, slow, pos, lag)
        temp = mx.clip(nn.softplus(self.log_temp) + 0.35, 0.25, 3.0)
        log_var = mx.clip(self.unc(z).squeeze(-1), -4.0, 4.0)
        return ModelOut(
            logits=self.policy(z),
            value=self.value(z).squeeze(-1),
            p_up=self.p_up(z).squeeze(-1) / temp,
            next_ret=self.next_ret(z).squeeze(-1),
            lag=self.lag_head(z).squeeze(-1),
            z=z,
            uncertainty=mx.sigmoid(log_var),
            log_var=log_var,
        )

    def count_params(self) -> int:
        from mlx.utils import tree_flatten

        return int(sum(v.size for _, v in tree_flatten(self.parameters())))
