from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from cmf.baseline import LacunaModel
from cmf.config import TrainConfig
from cmf.model import FusionModel, ModelOut


def _as_mx(x: np.ndarray) -> mx.array:
    return mx.array(np.ascontiguousarray(x))


def categorical_log_prob(logits: mx.array, actions: mx.array) -> mx.array:
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    idx = mx.arange(actions.shape[0])
    return logp[idx, actions]


def entropy_from_logits(logits: mx.array) -> mx.array:
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    p = mx.exp(logp)
    return -mx.sum(p * logp, axis=-1)


def gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    ret = np.zeros(n, dtype=np.float32)
    gae_acc = 0.0
    for t in range(n - 1, -1, -1):
        next_v = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * (1.0 - dones[t]) - values[t]
        gae_acc = delta + gamma * lam * (1.0 - dones[t]) * gae_acc
        adv[t] = gae_acc
        ret[t] = adv[t] + values[t]
    return adv, ret


@dataclass
class Rollout:
    fast: np.ndarray
    slow: np.ndarray
    pos: np.ndarray
    lag: np.ndarray
    lacuna: np.ndarray
    lacuna_hist: np.ndarray
    actions: np.ndarray
    logp: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    p_up: np.ndarray
    next_ret: np.ndarray
    lag_y: np.ndarray
    oracle: np.ndarray


class FusionTrainer:
    def __init__(self, cfg: TrainConfig, model: FusionModel):
        self.cfg = cfg
        self.model = model
        mx.eval(self.model.parameters())
        decay = optim.cosine_decay(cfg.ppo_lr, cfg.ppo_updates * cfg.ppo_epochs * 4, end=cfg.ppo_lr * 0.1)
        self.opt = optim.AdamW(learning_rate=decay, weight_decay=cfg.weight_decay)
        self._loss_and_grad = nn.value_and_grad(self.model, self._loss)

    def forward(self, batch: dict[str, np.ndarray]) -> ModelOut:
        return self.model(
            _as_mx(batch["fast"]),
            _as_mx(batch["slow"]),
            _as_mx(batch["pos"]),
            _as_mx(batch["lag"]),
        )

    def act(self, batch: dict[str, np.ndarray], greedy: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self.forward(batch)
        mx.eval(out.logits, out.value)
        logits = np.array(out.logits)
        values = np.array(out.value)
        if greedy:
            actions = np.argmax(logits, axis=-1).astype(np.int32)
        else:
            # softmax in numpy for sampling
            z = logits - logits.max(axis=-1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=-1, keepdims=True)
            actions = np.array([np.random.choice(p.shape[1], p=row) for row in p], dtype=np.int32)
        logp = np.array(categorical_log_prob(out.logits, mx.array(actions)))
        return actions, logp.astype(np.float32), values.astype(np.float32)

    def _loss(self, model: FusionModel, batch: dict) -> mx.array:
        out = model(batch["fast"], batch["slow"], batch["pos"], batch["lag"])
        logp = categorical_log_prob(out.logits, batch["actions"])
        ratio = mx.exp(logp - batch["old_logp"])
        unclipped = ratio * batch["adv"]
        clipped = mx.clip(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * batch["adv"]
        policy = -mx.mean(mx.minimum(unclipped, clipped))
        v_clip = batch["old_value"] + mx.clip(
            out.value - batch["old_value"], -self.cfg.clip_eps, self.cfg.clip_eps
        )
        v_loss = 0.5 * mx.mean(mx.maximum((out.value - batch["ret"]) ** 2, (v_clip - batch["ret"]) ** 2))
        ent = mx.mean(entropy_from_logits(out.logits))
        aux = (
            mx.mean(nn.losses.binary_cross_entropy(out.p_up, batch["p_up"], with_logits=True))
            + mx.mean((out.next_ret - batch["next_ret"]) ** 2)
            + mx.mean((out.lag - batch["lag_y"]) ** 2)
        )
        oracle = batch["oracle"]
        bc = -mx.mean(categorical_log_prob(out.logits, oracle))
        return (
            policy
            + self.cfg.value_coef * v_loss
            - self.cfg.entropy_coef * ent
            + self.cfg.aux_coef * aux
            + 0.45 * bc
        )

    def update(self, roll: Rollout) -> dict[str, float]:
        adv, ret = gae(
            roll.rewards,
            roll.values,
            roll.dones,
            float(roll.values[-1]),
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(roll.actions)
        metrics: dict[str, list[float]] = {"loss": [], "kl": []}
        for _ in range(self.cfg.ppo_epochs):
            perm = np.random.permutation(n)
            epoch_kl = 0.0
            batches = 0
            for start in range(0, n, self.cfg.ppo_batch):
                idx = perm[start : start + self.cfg.ppo_batch]
                batch = {
                    "fast": _as_mx(roll.fast[idx]),
                    "slow": _as_mx(roll.slow[idx]),
                    "pos": _as_mx(roll.pos[idx]),
                    "lag": _as_mx(roll.lag[idx]),
                    "actions": _as_mx(roll.actions[idx].astype(np.int32)),
                    "old_logp": _as_mx(roll.logp[idx]),
                    "old_value": _as_mx(roll.values[idx]),
                    "adv": _as_mx(adv[idx]),
                    "ret": _as_mx(ret[idx]),
                    "p_up": _as_mx(roll.p_up[idx]),
                    "next_ret": _as_mx(roll.next_ret[idx]),
                    "lag_y": _as_mx(roll.lag_y[idx]),
                    "oracle": _as_mx(roll.oracle[idx].astype(np.int32)),
                }
                loss, grads = self._loss_and_grad(self.model, batch)
                grads, _ = optim.clip_grad_norm(grads, self.cfg.max_grad_norm)
                self.opt.update(self.model, grads)
                mx.eval(self.model.parameters(), self.opt.state, loss)
                with mx.stream(mx.default_stream(mx.default_device())):
                    out = self.model(batch["fast"], batch["slow"], batch["pos"], batch["lag"])
                    new_logp = categorical_log_prob(out.logits, batch["actions"])
                    kl = float(np.array(mx.mean(batch["old_logp"] - new_logp)))
                metrics["loss"].append(float(np.array(loss)))
                metrics["kl"].append(kl)
                epoch_kl += kl
                batches += 1
            if batches and epoch_kl / batches > self.cfg.target_kl:
                break
        return {"loss": float(np.mean(metrics["loss"])), "kl": float(np.mean(metrics["kl"]))}


class Pretrainer:
    def __init__(self, cfg: TrainConfig, model: FusionModel):
        self.cfg = cfg
        self.model = model
        mx.eval(self.model.parameters())
        decay = optim.cosine_decay(cfg.pretrain_lr, max(cfg.pretrain_steps, 1), end=cfg.pretrain_lr * 0.15)
        self.opt = optim.AdamW(learning_rate=decay, weight_decay=cfg.weight_decay)
        self._step = nn.value_and_grad(self.model, self._loss)

    def _loss(self, model: FusionModel, batch: dict) -> mx.array:
        out = model(batch["fast"], batch["slow"], batch["pos"], batch["lag"])
        bce = mx.mean(nn.losses.binary_cross_entropy(out.p_up, batch["p_up"], with_logits=True))
        ret = mx.mean((out.next_ret - batch["next_ret"]) ** 2)
        lag = mx.mean((out.lag - batch["lag_y"]) ** 2)
        logp = categorical_log_prob(out.logits, batch["oracle"])
        bc = -mx.mean(logp)
        return bce + ret + lag + 1.15 * bc

    def step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        mx_batch = {k: _as_mx(v) if k != "oracle" else mx.array(v.astype(np.int32)) for k, v in batch.items()}
        loss, grads = self._step(self.model, mx_batch)
        grads, _ = optim.clip_grad_norm(grads, self.cfg.max_grad_norm)
        self.opt.update(self.model, grads)
        mx.eval(self.model.parameters(), self.opt.state, loss)
        return {"pretrain_loss": float(np.array(loss))}


class LacunaTrainer:
    def __init__(self, cfg: TrainConfig, model: LacunaModel):
        self.cfg = cfg
        self.model = model
        mx.eval(self.model.parameters())
        self.opt = optim.AdamW(learning_rate=cfg.ppo_lr, weight_decay=0.0)
        self._loss_and_grad = nn.value_and_grad(self.model, self._loss)

    def act(self, current: np.ndarray, temporal: np.ndarray, greedy: bool = False):
        out = self.model(_as_mx(current), _as_mx(temporal))
        mx.eval(out.logits, out.value)
        logits = np.array(out.logits)
        values = np.array(out.value)
        if greedy:
            actions = np.argmax(logits, axis=-1).astype(np.int32)
        else:
            z = logits - logits.max(axis=-1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=-1, keepdims=True)
            actions = np.array([np.random.choice(p.shape[1], p=row) for row in p], dtype=np.int32)
        logp = np.array(categorical_log_prob(out.logits, mx.array(actions)))
        return actions, logp.astype(np.float32), values.astype(np.float32)

    def _loss(self, model: LacunaModel, batch: dict) -> mx.array:
        out = model(batch["lacuna"], batch["lacuna_hist"])
        logp = categorical_log_prob(out.logits, batch["actions"])
        ratio = mx.exp(logp - batch["old_logp"])
        unclipped = ratio * batch["adv"]
        clipped = mx.clip(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * batch["adv"]
        policy = -mx.mean(mx.minimum(unclipped, clipped))
        v_loss = 0.5 * mx.mean((out.value - batch["ret"]) ** 2)
        ent = mx.mean(entropy_from_logits(out.logits))
        return policy + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * ent

    def update(self, roll: Rollout) -> dict[str, float]:
        adv, ret = gae(
            roll.rewards,
            roll.values,
            roll.dones,
            float(roll.values[-1]),
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(roll.actions)
        last = 0.0
        for _ in range(self.cfg.ppo_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.cfg.ppo_batch):
                idx = perm[start : start + self.cfg.ppo_batch]
                batch = {
                    "lacuna": _as_mx(roll.lacuna[idx]),
                    "lacuna_hist": _as_mx(roll.lacuna_hist[idx]),
                    "actions": _as_mx(roll.actions[idx].astype(np.int32)),
                    "old_logp": _as_mx(roll.logp[idx]),
                    "adv": _as_mx(adv[idx]),
                    "ret": _as_mx(ret[idx]),
                }
                loss, grads = self._loss_and_grad(self.model, batch)
                grads, _ = optim.clip_grad_norm(grads, self.cfg.max_grad_norm)
                self.opt.update(self.model, grads)
                mx.eval(self.model.parameters(), self.opt.state, loss)
                last = float(np.array(loss))
        return {"loss": last}
