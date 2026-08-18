from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    history: int = 64
    fast_dim: int = 24
    slow_dim: int = 16
    pos_dim: int = 8
    lag_dim: int = 8
    n_actions: int = 3

    # CMF-1 Large: ~28M params, ~110 MB F32. Old 96/3 card is `small`.
    dim: int = 384
    heads: int = 6
    layers: int = 6
    dropout: float = 0.05
    lacuna_hidden: int = 64
    lacuna_critic_hidden: int = 96
    lacuna_history: int = 5

    episode_ticks: int = 240
    dt: float = 1.0
    trade_size: float = 50.0
    max_spread: float = 0.04
    lag_min: float = 4.0
    lag_max: float = 14.0

    pretrain_steps: int = 500
    pretrain_batch: int = 24
    pretrain_lr: float = 2e-4

    ppo_updates: int = 24
    rollout_envs: int = 8
    rollout_steps: int = 48
    ppo_epochs: int = 3
    ppo_batch: int = 32
    ppo_lr: float = 2.5e-4
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.03
    value_coef: float = 0.5
    aux_coef: float = 0.35
    max_grad_norm: float = 1.0
    target_kl: float = 0.03
    weight_decay: float = 0.02

    eval_episodes: int = 48
    seed: int = 7
    checkpoint_dir: Path = Field(default=Path("checkpoints"))

    @property
    def buffer_size(self) -> int:
        return self.rollout_envs * self.rollout_steps
