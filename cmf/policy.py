from __future__ import annotations

import numpy as np

HOLD, BUY, SELL = 0, 1, 2


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + np.exp(-x))


def decide_from_prob(
    p_up: float,
    ask: float,
    bid: float,
    side: int = 0,
    enter: float = 0.045,
    flip: float = 0.10,
) -> int:
    """
    Trade a calibrated P(resolve UP) against the CLOB.

    Buy UP when the model probability clears the offer by `enter`.
    Buy DOWN when 1 - P(up) clears the DOWN offer. Hold through small noise.
    """
    up_edge = p_up - ask
    down_edge = (1.0 - p_up) - (1.0 - bid)
    if side == 0:
        if up_edge > enter:
            return BUY
        if down_edge > enter:
            return SELL
        return HOLD
    if side > 0 and down_edge > flip:
        return SELL
    if side < 0 and up_edge > flip:
        return BUY
    return HOLD


def decide_from_logit(logit: float, ask: float, bid: float, side: int = 0) -> int:
    return decide_from_prob(sigmoid(logit), ask, bid, side)
