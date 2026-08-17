from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from cmf.config import TrainConfig


@dataclass
class LacunaOut:
    logits: mx.array
    value: mx.array


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int = 18, history_len: int = 5, output_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * history_len, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.tanh(self.ln1(self.fc1(x)))
        return mx.tanh(self.ln2(self.fc2(h)))


class LacunaModel(nn.Module):
    """Faithful reimplementation of LACUNA's Phase-5 actor-critic (38k tanh MLP)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.input_dim = 18
        self.history_len = cfg.lacuna_history
        temporal_dim = 32
        combined = self.input_dim + temporal_dim
        self.actor_temporal = TemporalEncoder(self.input_dim, self.history_len, temporal_dim)
        self.critic_temporal = TemporalEncoder(self.input_dim, self.history_len, temporal_dim)
        h = cfg.lacuna_hidden
        c = cfg.lacuna_critic_hidden
        self.a1 = nn.Linear(combined, h)
        self.aln1 = nn.LayerNorm(h)
        self.a2 = nn.Linear(h, h)
        self.aln2 = nn.LayerNorm(h)
        self.a3 = nn.Linear(h, cfg.n_actions)
        self.c1 = nn.Linear(combined, c)
        self.cln1 = nn.LayerNorm(c)
        self.c2 = nn.Linear(c, c)
        self.cln2 = nn.LayerNorm(c)
        self.c3 = nn.Linear(c, 1)

    def __call__(self, current: mx.array, temporal: mx.array) -> LacunaOut:
        ta = self.actor_temporal(temporal)
        tc = self.critic_temporal(temporal)
        ha = mx.concatenate([current, ta], axis=-1)
        hc = mx.concatenate([current, tc], axis=-1)
        ha = mx.tanh(self.aln1(self.a1(ha)))
        ha = mx.tanh(self.aln2(self.a2(ha)))
        hc = mx.tanh(self.cln1(self.c1(hc)))
        hc = mx.tanh(self.cln2(self.c2(hc)))
        return LacunaOut(logits=self.a3(ha), value=self.c3(hc).squeeze(-1))

    def count_params(self) -> int:
        from mlx.utils import tree_flatten

        return int(sum(v.size for _, v in tree_flatten(self.parameters())))
