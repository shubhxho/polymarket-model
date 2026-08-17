from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.table import Table

from cmf.baseline import LacunaModel
from cmf.config import TrainConfig
from cmf.features import has_native
from cmf.model import FusionModel
from cmf.policy import decide_from_logit
from cmf.ppo import FusionTrainer, LacunaTrainer, Pretrainer, Rollout, categorical_log_prob
from cmf.dataset import bank_stats, load_windows
from cmf.simulator import LagMarket, stack_obs

console = Console()


def _seed_everything(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    mx.random.seed(seed)
    return np.random.default_rng(seed)


def _make_envs(cfg: TrainConfig, n: int, rng: np.random.Generator) -> list[LagMarket]:
    return [LagMarket(cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1)))) for _ in range(n)]


def _collect_pretrain_batch(envs: list[LagMarket], batch: int) -> dict[str, np.ndarray]:
    """Take several labeled frames from each reset instead of one reset per row."""
    obs, p_up, nxt, lag, oracle = [], [], [], [], []
    env_i = 0
    while len(obs) < batch:
        env = envs[env_i % len(envs)]
        env_i += 1
        o = env.reset()
        picks = np.unique(
            np.random.randint(env.history, env.ticks - 2, size=max(8, batch // max(len(envs), 1)))
        )
        for target in sorted(int(x) for x in picks):
            while env.t < target and not env.done:
                o, _, done, _ = env.step(env.oracle_action())
                if done:
                    break
            if env.done:
                break
            info = env.labels()
            action = int(info["oracle"])
            # keep every trade label; downsample HOLDs so BC does not collapse
            if action == 0 and np.random.random() < 0.55:
                continue
            obs.append(o)
            p_up.append(info["p_up"])
            nxt.append(info["next_ret"])
            lag.append(info["lag"])
            oracle.append(action)
            if len(obs) >= batch:
                break
    packed = stack_obs(obs[:batch])
    packed["p_up"] = np.asarray(p_up[:batch], dtype=np.float32)
    packed["next_ret"] = np.asarray(nxt[:batch], dtype=np.float32)
    packed["lag_y"] = np.asarray(lag[:batch], dtype=np.float32)
    packed["oracle"] = np.asarray(oracle[:batch], dtype=np.int32)
    return packed


def _rollout_fusion(cfg: TrainConfig, trainer: FusionTrainer, envs: list[LagMarket]) -> tuple[Rollout, float]:
    buf = {k: [] for k in (
        "fast", "slow", "pos", "lag", "lacuna", "lacuna_hist",
        "actions", "logp", "values", "rewards", "dones", "p_up", "next_ret", "lag_y", "oracle",
    )}
    states = [e.reset() for e in envs]
    ep_pnls: list[float] = []
    running = [0.0] * len(envs)
    for _ in range(cfg.rollout_steps):
        packed = stack_obs(states)
        out = trainer.forward(packed)
        mx.eval(out.logits, out.value, out.p_up)
        logits = np.array(out.p_up)
        values = np.array(out.value).astype(np.float32)
        actions = np.zeros(len(envs), dtype=np.int32)
        for i, env in enumerate(envs):
            actions[i] = decide_from_logit(float(logits[i]), states[i].ask, states[i].bid, env.side)
        logp = np.array(categorical_log_prob(out.logits, mx.array(actions))).astype(np.float32)
        nxt_states = []
        for i, env in enumerate(envs):
            oracle_a = env.oracle_action()
            nxt, reward, done, info = env.step(int(actions[i]))
            running[i] += reward
            buf["fast"].append(packed["fast"][i])
            buf["slow"].append(packed["slow"][i])
            buf["pos"].append(packed["pos"][i])
            buf["lag"].append(packed["lag"][i])
            buf["lacuna"].append(packed["lacuna"][i])
            buf["lacuna_hist"].append(packed["lacuna_hist"][i])
            buf["actions"].append(actions[i])
            buf["logp"].append(logp[i])
            buf["values"].append(values[i])
            buf["rewards"].append(reward)
            buf["dones"].append(float(done))
            buf["p_up"].append(info["p_up"])
            buf["next_ret"].append(info["next_ret"])
            buf["lag_y"].append(info["lag"])
            buf["oracle"].append(oracle_a)
            if done:
                ep_pnls.append(running[i])
                running[i] = 0.0
                nxt = env.reset()
            nxt_states.append(nxt)
        states = nxt_states
    roll = Rollout(
        fast=np.stack(buf["fast"]),
        slow=np.stack(buf["slow"]),
        pos=np.stack(buf["pos"]),
        lag=np.stack(buf["lag"]),
        lacuna=np.stack(buf["lacuna"]),
        lacuna_hist=np.stack(buf["lacuna_hist"]),
        actions=np.asarray(buf["actions"], dtype=np.int32),
        logp=np.asarray(buf["logp"], dtype=np.float32),
        values=np.asarray(buf["values"], dtype=np.float32),
        rewards=np.asarray(buf["rewards"], dtype=np.float32),
        dones=np.asarray(buf["dones"], dtype=np.float32),
        p_up=np.asarray(buf["p_up"], dtype=np.float32),
        next_ret=np.asarray(buf["next_ret"], dtype=np.float32),
        lag_y=np.asarray(buf["lag_y"], dtype=np.float32),
        oracle=np.asarray(buf["oracle"], dtype=np.int32),
    )
    mean_pnl = float(np.mean(ep_pnls)) if ep_pnls else float(np.mean(roll.rewards) * cfg.episode_ticks)
    return roll, mean_pnl


def _rollout_lacuna(cfg: TrainConfig, trainer: LacunaTrainer, envs: list[LagMarket]) -> tuple[Rollout, float]:
    buf = {k: [] for k in (
        "fast", "slow", "pos", "lag", "lacuna", "lacuna_hist",
        "actions", "logp", "values", "rewards", "dones", "p_up", "next_ret", "lag_y", "oracle",
    )}
    states = [e.reset() for e in envs]
    ep_pnls: list[float] = []
    running = [0.0] * len(envs)
    for _ in range(cfg.rollout_steps):
        packed = stack_obs(states)
        actions, logp, values = trainer.act(packed["lacuna"], packed["lacuna_hist"], greedy=False)
        nxt_states = []
        for i, env in enumerate(envs):
            oracle_a = env.oracle_action()
            nxt, reward, done, info = env.step(int(actions[i]))
            running[i] += reward
            buf["fast"].append(packed["fast"][i])
            buf["slow"].append(packed["slow"][i])
            buf["pos"].append(packed["pos"][i])
            buf["lag"].append(packed["lag"][i])
            buf["lacuna"].append(packed["lacuna"][i])
            buf["lacuna_hist"].append(packed["lacuna_hist"][i])
            buf["actions"].append(actions[i])
            buf["logp"].append(logp[i])
            buf["values"].append(values[i])
            buf["rewards"].append(reward)
            buf["dones"].append(float(done))
            buf["p_up"].append(info["p_up"])
            buf["next_ret"].append(info["next_ret"])
            buf["lag_y"].append(info["lag"])
            buf["oracle"].append(oracle_a)
            if done:
                ep_pnls.append(running[i])
                running[i] = 0.0
                nxt = env.reset()
            nxt_states.append(nxt)
        states = nxt_states
    roll = Rollout(
        fast=np.stack(buf["fast"]),
        slow=np.stack(buf["slow"]),
        pos=np.stack(buf["pos"]),
        lag=np.stack(buf["lag"]),
        lacuna=np.stack(buf["lacuna"]),
        lacuna_hist=np.stack(buf["lacuna_hist"]),
        actions=np.asarray(buf["actions"], dtype=np.int32),
        logp=np.asarray(buf["logp"], dtype=np.float32),
        values=np.asarray(buf["values"], dtype=np.float32),
        rewards=np.asarray(buf["rewards"], dtype=np.float32),
        dones=np.asarray(buf["dones"], dtype=np.float32),
        p_up=np.asarray(buf["p_up"], dtype=np.float32),
        next_ret=np.asarray(buf["next_ret"], dtype=np.float32),
        lag_y=np.asarray(buf["lag_y"], dtype=np.float32),
        oracle=np.asarray(buf["oracle"], dtype=np.int32),
    )
    mean_pnl = float(np.mean(ep_pnls)) if ep_pnls else float(np.mean(roll.rewards) * cfg.episode_ticks)
    return roll, mean_pnl


def evaluate(cfg: TrainConfig, kind: str, model, n: int, rng: np.random.Generator) -> dict[str, float]:
    if model is not None:
        model.eval()
    pnls = []
    wins = 0
    trades = 0
    p_up_hit = []
    for i in range(n):
        env = LagMarket(cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + i))
        obs = env.reset()
        pnl = 0.0
        while True:
            packed = stack_obs([obs])
            if kind == "fusion":
                out = model(mx.array(packed["fast"]), mx.array(packed["slow"]),
                            mx.array(packed["pos"]), mx.array(packed["lag"]))
                mx.eval(out.logits, out.p_up)
                logit = float(np.array(out.p_up)[0])
                action = decide_from_logit(logit, obs.ask, obs.bid, env.side)
                p_up_hit.append(float((logit > 0) == (env.resolved_up > 0.5)))
            elif kind == "lacuna":
                out = model(mx.array(packed["lacuna"]), mx.array(packed["lacuna_hist"]))
                mx.eval(out.logits)
                action = int(np.argmax(np.array(out.logits)[0]))
            elif kind == "oracle":
                action = env.oracle_action()
            else:
                action = int(rng.integers(0, 3))
            prev_side = env.side
            obs, reward, done, _ = env.step(action)
            if env.side != prev_side:
                trades += 1
            pnl += reward
            if done:
                break
        pnls.append(pnl)
        if pnl > 0:
            wins += 1
    if model is not None:
        model.train()
    arr = np.asarray(pnls, dtype=np.float64)
    sharpe = float(arr.mean() / (arr.std() + 1e-8) * np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "pnl": float(arr.mean()),
        "pnl_std": float(arr.std()),
        "win_rate": wins / max(n, 1),
        "trades_per_ep": trades / max(n, 1),
        "sharpe": sharpe,
        "p_up_acc": float(np.mean(p_up_hit)) if p_up_hit else float("nan"),
    }


def save_model(model: FusionModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path))


def train(cfg: TrainConfig) -> dict:
    rng = _seed_everything(cfg.seed)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    windows = load_windows()
    LagMarket.path_bank = windows or None
    console.print(f"real data: {bank_stats(windows)}")
    fusion = FusionModel(cfg)
    lacuna = LacunaModel(cfg)
    mx.eval(fusion.parameters(), lacuna.parameters())

    console.rule("[bold]cross-market fusion")
    console.print(
        f"native C++ engine: {has_native()}  |  fusion params: {fusion.count_params():,}  |  "
        f"lacuna params: {lacuna.count_params():,}  |  device: {mx.default_device()}"
    )

    pre_envs = _make_envs(cfg, max(8, cfg.pretrain_batch // 8), rng)
    pre = Pretrainer(cfg, fusion)
    t0 = time.time()
    pre_losses = []
    for step in range(cfg.pretrain_steps):
        batch = _collect_pretrain_batch(pre_envs, cfg.pretrain_batch)
        stats = pre.step(batch)
        pre_losses.append(stats["pretrain_loss"])
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == cfg.pretrain_steps:
            console.print(f"pretrain {step + 1:>4}/{cfg.pretrain_steps}  loss={stats['pretrain_loss']:.4f}")
    console.print(f"pretrain done in {time.time() - t0:.1f}s  last_loss={pre_losses[-1]:.4f}")

    f_trainer = FusionTrainer(cfg, fusion)
    l_trainer = LacunaTrainer(cfg, lacuna)
    f_envs = _make_envs(cfg, cfg.rollout_envs, rng)
    l_envs = _make_envs(cfg, cfg.rollout_envs, rng)

    history = []
    for upd in range(cfg.ppo_updates):
        f_roll, f_pnl = _rollout_fusion(cfg, f_trainer, f_envs)
        l_roll, l_pnl = _rollout_lacuna(cfg, l_trainer, l_envs)
        f_stats = f_trainer.update(f_roll)
        l_stats = l_trainer.update(l_roll)
        row = {
            "update": upd + 1,
            "fusion_loss": f_stats["loss"],
            "fusion_kl": f_stats["kl"],
            "fusion_pnl": f_pnl,
            "lacuna_loss": l_stats["loss"],
            "lacuna_pnl": l_pnl,
        }
        history.append(row)
        console.print(
            f"ppo {upd + 1:>3}/{cfg.ppo_updates}  "
            f"fusion loss={f_stats['loss']:.3f} kl={f_stats['kl']:.4f} pnl={f_pnl:+.2f}  "
            f"lacuna loss={l_stats['loss']:.3f} pnl={l_pnl:+.2f}"
        )

    eval_rng = np.random.default_rng(cfg.seed + 999)
    results = {
        "fusion": evaluate(cfg, "fusion", fusion, cfg.eval_episodes, eval_rng),
        "lacuna": evaluate(cfg, "lacuna", lacuna, cfg.eval_episodes, eval_rng),
        "oracle": evaluate(cfg, "oracle", None, cfg.eval_episodes, eval_rng),
        "random": evaluate(cfg, "random", None, cfg.eval_episodes, eval_rng),
    }

    table = Table(title=f"held-out eval ({cfg.eval_episodes} episodes)")
    table.add_column("policy")
    table.add_column("mean PnL", justify="right")
    table.add_column("std", justify="right")
    table.add_column("win %", justify="right")
    table.add_column("trades/ep", justify="right")
    table.add_column("sharpe", justify="right")
    table.add_column("P(up) acc", justify="right")
    for name, m in results.items():
        acc = "—" if np.isnan(m["p_up_acc"]) else f"{100 * m['p_up_acc']:.1f}%"
        table.add_row(
            name,
            f"{m['pnl']:+.2f}",
            f"{m['pnl_std']:.2f}",
            f"{100 * m['win_rate']:.1f}%",
            f"{m['trades_per_ep']:.2f}",
            f"{m['sharpe']:.2f}",
            acc,
        )
    console.print(table)

    ckpt = cfg.checkpoint_dir / "fusion.safetensors"
    save_model(fusion, ckpt)
    lacuna.save_weights(str(cfg.checkpoint_dir / "lacuna.safetensors"))
    summary = {
        "native": has_native(),
        "real_windows": len(windows),
        "fusion_params": fusion.count_params(),
        "lacuna_params": lacuna.count_params(),
        "pretrain_loss": pre_losses[-1] if pre_losses else None,
        "results": results,
        "history": history,
        "config": json.loads(cfg.model_dump_json()),
    }
    (cfg.checkpoint_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    console.print(f"wrote {ckpt} and {cfg.checkpoint_dir / 'metrics.json'}")
    return summary
