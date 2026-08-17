import numpy as np

from cmf.policy import BUY, HOLD, SELL, decide_from_prob
from cmf.simulator import LagMarket
from cmf.config import TrainConfig


def test_enter_up_when_prob_clears_ask():
    assert decide_from_prob(0.80, ask=0.62, bid=0.60) == BUY


def test_enter_down_when_prob_clears_bid():
    assert decide_from_prob(0.20, ask=0.40, bid=0.38) == SELL


def test_hold_inside_spread():
    assert decide_from_prob(0.50, ask=0.52, bid=0.48) == HOLD


def test_round_trip_pays_spread():
    env = LagMarket(TrainConfig(episode_ticks=120), np.random.default_rng(1))
    env.reset()
    _, r1, _, _ = env.step(BUY)
    _, r2, _, _ = env.step(SELL)
    assert r1 + r2 < 0.0


def test_random_loses_and_oracle_is_better():
    cfg = TrainConfig(episode_ticks=160)
    rnd, ora = [], []
    for i in range(16):
        e = LagMarket(cfg, np.random.default_rng(10 + i))
        e.reset()
        p = 0.0
        while True:
            _, r, d, _ = e.step(int(e.rng.integers(0, 3)))
            p += r
            if d:
                break
        rnd.append(p)
        e = LagMarket(cfg, np.random.default_rng(10 + i))
        e.reset()
        p = 0.0
        while True:
            _, r, d, _ = e.step(e.oracle_action())
            p += r
            if d:
                break
        ora.append(p)
    assert float(np.mean(rnd)) < 0.0
    assert float(np.mean(ora)) > float(np.mean(rnd))
