from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cmf.config import TrainConfig
from cmf.features import (
    FAST_DIM,
    LAG_DIM,
    POS_DIM,
    RAW_FAST_DIM,
    RAW_SLOW_DIM,
    SLOW_DIM,
    featurize_episode,
    lacuna_vector,
    window_at,
)

HOLD, BUY, SELL = 0, 1, 2


@dataclass
class StepObs:
    fast_win: np.ndarray
    slow_win: np.ndarray
    pos: np.ndarray
    lag: np.ndarray
    lacuna: np.ndarray
    lacuna_hist: np.ndarray
    mid: float = 0.5
    bid: float = 0.5
    ask: float = 0.5


class LagMarket:
    path_bank: list | None = None

    """
    15-minute binary market whose CLOB lags a faster futures tape.

    Binance is GBM + jumps + Hawkes-signed flow.
    Polymarket mid is a delayed logistic of the futures move from the window open,
    plus inventory noise and a wide spread. Resolution is sign(S_end - S_open).
    """

    def __init__(self, cfg: TrainConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.bank: list | None = None
        self.real_mix: float = 0.8
        self.ticks = cfg.episode_ticks
        self.history = cfg.history
        self.full_ticks = 900
        self.reset()

    def reset(self) -> StepObs:
        self.t = self.history - 1
        self.done = False
        self.side = 0
        self.entry = 0.0
        self.shares = 0.0
        self.size = self.cfg.trade_size
        self.hold_steps = 0
        self.true_lag = int(self.rng.integers(int(self.cfg.lag_min), int(self.cfg.lag_max) + 1))
        bank = self.bank if getattr(self, "bank", None) is not None else getattr(LagMarket, "path_bank", None)
        mix = float(getattr(self, "real_mix", 0.8))
        if bank and self.rng.random() < mix:
            self._finish_from_price(np.asarray(bank[int(self.rng.integers(0, len(bank)))], dtype=np.float64))
        else:
            self._simulate_paths()
        self.fast_feat, self.slow_feat, self.lag_feat = featurize_episode(self.fast_raw, self.slow_raw)
        self.resolved_up = float(self.bn_path[-1] > self.bn_open)
        return self._obs()

    def _simulate_paths(self) -> None:
        rng = self.rng
        n = self.full_ticks
        dt = self.cfg.dt
        s0 = float(rng.uniform(80.0, 120_000.0))
        mu = float(rng.normal(0.0, 1.2e-5))
        sigma = float(rng.uniform(4e-4, 1.6e-3))
        jump_p = float(rng.uniform(0.01, 0.04))
        s = np.empty(n, dtype=np.float64)
        s[0] = s0
        hawkes_buy = 0.15
        hawkes_sell = 0.15
        trades_qty = np.zeros(n, dtype=np.float32)
        trades_sign = np.zeros(n, dtype=np.float32)
        jumps = np.zeros(n, dtype=np.float32)
        funding = float(rng.normal(0.0, 5e-5))
        oi = float(rng.uniform(-0.04, 0.04))
        basis = float(rng.normal(0.0, 4e-4))
        for i in range(1, n):
            hawkes_buy *= np.exp(-1.4 * dt)
            hawkes_sell *= np.exp(-1.4 * dt)
            z = rng.normal()
            js = 0.0
            if rng.random() < jump_p:
                js = float(rng.normal(0.0, 3.5) * sigma)
                jumps[i] = 1.0
                if js > 0:
                    hawkes_buy += 0.7
                else:
                    hawkes_sell += 0.7
            s[i] = s[i - 1] * np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z + js)
            intensity = hawkes_buy + hawkes_sell + 0.2
            n_tr = rng.poisson(intensity)
            if n_tr > 0:
                p_buy = hawkes_buy / (hawkes_buy + hawkes_sell + 1e-6)
                sign = 1.0 if rng.random() < p_buy else -1.0
                trades_sign[i] = sign
                trades_qty[i] = float(rng.lognormal(2.2, 0.8) * n_tr)
                s[i] *= np.exp(sign * 0.08 * sigma)

        self._finish_from_price(s, trades_qty, trades_sign, jumps)

    def _finish_from_price(
        self,
        s: np.ndarray,
        trades_qty: np.ndarray | None = None,
        trades_sign: np.ndarray | None = None,
        jumps: np.ndarray | None = None,
    ) -> None:
        rng = self.rng
        n = int(s.shape[0])
        if n != self.full_ticks:
            x = np.linspace(0.0, 1.0, n)
            s = np.interp(np.linspace(0.0, 1.0, self.full_ticks), x, s)
            n = self.full_ticks
        if trades_qty is None:
            trades_qty = np.zeros(n, dtype=np.float32)
            trades_sign = np.zeros(n, dtype=np.float32)
            jumps = np.zeros(n, dtype=np.float32)
            for i in range(n):
                if rng.random() < 0.2:
                    trades_sign[i] = float(rng.choice([-1.0, 1.0]))
                    trades_qty[i] = float(rng.lognormal(2.0, 0.7))
        funding = float(rng.normal(0.0, 5e-5))
        oi = float(rng.uniform(-0.04, 0.04))
        basis = float(rng.normal(0.0, 4e-4))
        start = n - self.ticks
        self.bn_open = float(s[0])
        self.bn_path = s[start:].astype(np.float32)
        self.bn_full = s
        rv = float(np.std(np.diff(s) / s[:-1]) * np.sqrt(900) + 1e-6)
        move = (s - s[0]) / (s[0] * rv)
        fair = 1.0 / (1.0 + np.exp(-3.4 * move))
        fair = np.clip(fair, 0.06, 0.94)
        self._rv = rv
        lagged = np.empty_like(fair)
        lagged[: self.true_lag] = 0.5
        lagged[self.true_lag :] = fair[: -self.true_lag]
        inv = rng.normal(0.0, 0.02, size=n)
        for i in range(1, n):
            inv[i] = 0.92 * inv[i - 1] + 0.08 * inv[i]
        stale = np.zeros(n, dtype=np.float32)
        last = lagged[0]
        last_i = 0
        quoted = np.empty(n, dtype=np.float64)
        for i in range(n):
            # CLOB only reprints when the delayed fair moved enough or a Poisson clock fires.
            if abs(lagged[i] - last) > 0.012 or rng.random() < 0.08:
                last = lagged[i] + inv[i]
                last_i = i
            quoted[i] = np.clip(last, 0.08, 0.92)
            stale[i] = float(i - last_i)

        q = quoted[start:]
        spr = 0.014 + 6.0 * np.abs(np.diff(q, prepend=q[0])) + rng.uniform(0.004, 0.01)
        spr = np.clip(spr, 0.008, self.cfg.max_spread)
        mid = q.astype(np.float32)
        bid = np.clip(mid - 0.5 * spr, 0.01, 0.98).astype(np.float32)
        ask = np.clip(mid + 0.5 * spr, 0.02, 0.99).astype(np.float32)

        bn = self.bn_path
        ret = np.zeros_like(bn)
        ret[1:] = (bn[1:] - bn[:-1]) / bn[:-1]
        vol = np.maximum(np.abs(bn) * 0.0004, 1e-6)
        fast = np.zeros((self.ticks, RAW_FAST_DIM), dtype=np.float32)
        slow = np.zeros((self.ticks, RAW_SLOW_DIM), dtype=np.float32)
        fq = trades_qty[start:]
        fs = trades_sign[start:]
        fj = jumps[start:]
        st = stale[start:]
        tte = np.linspace(self.ticks / 900.0, 1.0 / 900.0, self.ticks, dtype=np.float32)
        for i in range(self.ticks):
            fast[i, 0] = float(i)
            fast[i, 1] = bn[i]
            fast[i, 2] = bn[i] - vol[i]
            fast[i, 3] = bn[i] + vol[i]
            fast[i, 4] = float(80.0 + 40.0 * (fs[i] > 0))
            fast[i, 5] = float(80.0 + 40.0 * (fs[i] < 0))
            fast[i, 6] = fq[i]
            fast[i, 7] = fs[i]
            fast[i, 8] = fq[i] * bn[i]
            fast[i, 9] = float(fq[i] > 0)
            fast[i, 10] = funding
            fast[i, 11] = oi
            fast[i, 12] = float(max(-ret[i], 0.0) * 1e5)
            fast[i, 13] = float(max(ret[i], 0.0) * 1e5)
            fast[i, 14] = basis
            fast[i, 15] = fj[i]
            slow[i, 0] = float(i)
            slow[i, 1] = mid[i]
            slow[i, 2] = bid[i]
            slow[i, 3] = ask[i]
            slow[i, 4] = float(120.0 + 400.0 * (0.5 - mid[i] if mid[i] < 0.5 else 0.0))
            slow[i, 5] = float(120.0 + 400.0 * (mid[i] - 0.5 if mid[i] > 0.5 else 0.0))
            slow[i, 6] = float(rng.lognormal(1.5, 0.6) if rng.random() < 0.25 else 0.0)
            slow[i, 7] = float(rng.choice([-1.0, 1.0])) if slow[i, 6] > 0 else 0.0
            slow[i, 8] = slow[i, 4] * 3.0
            slow[i, 9] = slow[i, 5] * 3.0
            slow[i, 10] = tte[i]
            slow[i, 11] = st[i]

        self.fast_raw = fast
        self.slow_raw = slow
        self.mid = mid
        self.bid = bid
        self.ask = ask
        self.fair = fair[start:].astype(np.float32)
        self.next_ret = np.zeros(self.ticks, dtype=np.float32)
        horizon = 5
        self.next_ret[:-horizon] = (bn[horizon:] - bn[:-horizon]) / bn[:-horizon]
        self.true_lag_norm = np.full(self.ticks, self.true_lag / 16.0, dtype=np.float32)

    def _pos_vec(self) -> np.ndarray:
        mid = float(self.mid[self.t])
        pnl = 0.0
        if self.side != 0:
            mark = mid if self.side > 0 else 1.0 - mid
            pnl = (mark - self.entry) * self.shares
        tte = float(self.slow_raw[self.t, 10])
        return np.array(
            [
                1.0 if self.side != 0 else 0.0,
                float(self.side),
                float(np.tanh(pnl / 25.0)),
                float(np.clip(2 * self.entry - 1.0, -1, 1)),
                float(self.hold_steps / max(self.ticks, 1)),
                tte,
                float(np.tanh(self.shares / 20.0)),
                float(np.tanh(abs(self.shares) * 0.05)),
            ],
            dtype=np.float32,
        )

    def _obs(self) -> StepObs:
        t = self.t
        pos = self._pos_vec()
        fast_win = window_at(self.fast_feat, t, self.history)
        slow_win = window_at(self.slow_feat, t, self.history)
        lag = self.lag_feat[t]
        lacuna = lacuna_vector(self.fast_feat[t], self.slow_feat[t], pos)
        hist = []
        for k in range(self.cfg.lacuna_history):
            idx = max(0, t - (self.cfg.lacuna_history - 1 - k))
            hist.append(lacuna_vector(self.fast_feat[idx], self.slow_feat[idx], pos))
        return StepObs(
            fast_win=fast_win.astype(np.float32),
            slow_win=slow_win.astype(np.float32),
            pos=pos,
            lag=lag.astype(np.float32),
            lacuna=lacuna,
            lacuna_hist=np.concatenate(hist).astype(np.float32),
            mid=float(self.mid[t]),
            bid=float(self.bid[t]),
            ask=float(self.ask[t]),
        )

    def _open_price(self, side: int) -> float:
        """Pay the offer for the token being bought."""
        if side > 0:
            return float(self.ask[self.t])
        return float(max(1.0 - self.bid[self.t], 0.02))

    def _close_price(self, side: int) -> float:
        """Receive the bid for the token being sold."""
        if side > 0:
            return float(self.bid[self.t])
        return float(max(1.0 - self.ask[self.t], 0.01))

    def _close(self, settle: float | None = None) -> float:
        if self.side == 0:
            return 0.0
        if settle is None:
            exit_px = self._close_price(self.side)
        else:
            exit_px = settle if self.side > 0 else 1.0 - settle
        pnl = (exit_px - self.entry) * self.shares
        self.side = 0
        self.shares = 0.0
        self.entry = 0.0
        self.hold_steps = 0
        return float(pnl / self.size)

    def oracle_action(self) -> int:
        """Enter when unlagged fair clears the offer; hold to expiry unless the edge flips hard."""
        t = self.t
        fair = float(self.fair[t])
        ask = float(self.ask[t])
        bid = float(self.bid[t])
        up_edge = fair - ask
        down_edge = (1.0 - fair) - (1.0 - bid)
        if self.side == 0:
            if up_edge > 0.045:
                return BUY
            if down_edge > 0.045:
                return SELL
            return HOLD
        if self.side > 0 and down_edge > 0.10:
            return SELL
        if self.side < 0 and up_edge > 0.10:
            return BUY
        return HOLD

    def labels(self) -> dict[str, float]:
        return {
            "p_up": self.resolved_up,
            "next_ret": float(np.clip(self.next_ret[self.t] * 80.0, -1, 1)),
            "lag": float(self.true_lag_norm[self.t]),
            "oracle": float(self.oracle_action()),
            "ask": float(self.ask[self.t]),
            "bid": float(self.bid[self.t]),
        }

    def step(self, action: int) -> tuple[StepObs, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode already finished")
        reward = 0.0
        mid = float(self.mid[self.t])
        if action == BUY:
            if self.side < 0:
                reward += self._close()
            if self.side == 0:
                px = self._open_price(1)
                self.side = 1
                self.entry = px
                self.shares = self.size / max(px, 0.08)
                self.hold_steps = 0
        elif action == SELL:
            if self.side > 0:
                reward += self._close()
            if self.side == 0:
                px = self._open_price(-1)
                self.side = -1
                self.entry = px
                self.shares = self.size / max(px, 0.08)
                self.hold_steps = 0
        else:
            if self.side != 0:
                self.hold_steps += 1

        self.t += 1
        done = self.t >= self.ticks - 1
        if done:
            settle = 1.0 if self.resolved_up > 0.5 else 0.0
            reward += self._close(settle=settle)
            self.done = True
        info = {
            "mid": mid,
            "resolved_up": self.resolved_up,
            "true_lag": float(self.true_lag),
            **self.labels(),
        }
        return self._obs(), float(reward), done, info


def stack_obs(batch: list[StepObs]) -> dict[str, np.ndarray]:
    return {
        "fast": np.stack([o.fast_win for o in batch]).astype(np.float32),
        "slow": np.stack([o.slow_win for o in batch]).astype(np.float32),
        "pos": np.stack([o.pos for o in batch]).astype(np.float32),
        "lag": np.stack([o.lag for o in batch]).astype(np.float32),
        "lacuna": np.stack([o.lacuna for o in batch]).astype(np.float32),
        "lacuna_hist": np.stack([o.lacuna_hist for o in batch]).astype(np.float32),
    }


# keep unused imports honest for type checkers
_ = (FAST_DIM, SLOW_DIM, POS_DIM, LAG_DIM)
