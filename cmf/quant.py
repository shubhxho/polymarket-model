"""Closed-form and ensemble signals used on 15-minute binaries.

The digital-option price is the textbook model for 'will S_T beat S_0'.
The fusion transformer is the learned lag reader. Votes are blended;
complement-arb is a hard override when the book is crossed against $1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cmf.policy import BUY, HOLD, SELL, decide_from_prob


def _norm_cdf(x: float) -> float:
    return float(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))


def digital_up_prob(spot: float, strike: float, tau_sec: float, vol: float) -> float:
    """
    Black–Scholes cash-or-nothing / binary call under GBM.

    P(S_T > K | S_t) = Φ(d2),  d2 = (ln(S/K) − ½σ²τ) / (σ√τ)
    τ in years (seconds / 31536000 is wrong for a 15m option — use τ = seconds/year
    of *this* window: we treat the 15m contract as expiring in tau_sec,
    so σ is per-sqrt-second if passed as per-second vol).

    Here `vol` is per-sqrt-second realized vol (std of 1s log-returns).
    """
    if spot <= 0 or strike <= 0:
        return 0.5
    tau = max(float(tau_sec), 1e-6)
    sig = max(float(vol), 1e-8)
    d2 = (np.log(spot / strike) - 0.5 * sig * sig * tau) / (sig * np.sqrt(tau))
    return float(np.clip(_norm_cdf(float(d2)), 0.02, 0.98))


def lag_adjusted(digital: float, ret_lead: float, stale_sec: float) -> float:
    """If Binance already moved and the CLOB is stale, tilt the digital price."""
    tilt = 6.0 * ret_lead * min(stale_sec / 4.0, 2.0)
    return float(np.clip(digital + tilt, 0.02, 0.98))


def complement_sum(ask_up: float, bid_up: float) -> float:
    """Cost of buying UP at ask and DOWN at (1-bid)."""
    ask_down = max(1.0 - bid_up, 0.01)
    return float(ask_up + ask_down)


@dataclass
class Ensemble:
    digital: float
    fusion: float
    lag: float
    ensemble: float
    complement: float
    action: int
    reason: str
    votes: dict[str, int]


def blend(digital: float, fusion: float, lag: float) -> float:
    # Digital is the structural prior; fusion can abstain (0.5) if the net is cold.
    return float(np.clip(0.45 * digital + 0.35 * fusion + 0.20 * lag, 0.02, 0.98))


def ensemble_signal(
    *,
    spot: float,
    strike: float,
    tau_sec: float,
    vol: float,
    ret_lead: float,
    stale_sec: float,
    fusion_p: float,
    ask: float,
    bid: float,
    enter: float = 0.04,
) -> Ensemble:
    dig = digital_up_prob(spot, strike, tau_sec, vol)
    lag = lag_adjusted(dig, ret_lead, stale_sec)
    fus = float(np.clip(fusion_p, 0.02, 0.98))
    p = blend(dig, fus, lag)
    comp = complement_sum(ask, bid)
    votes = {
        "digital": decide_from_prob(dig, ask, bid, enter=enter),
        "fusion": decide_from_prob(fus, ask, bid, enter=enter),
        "lag": decide_from_prob(lag, ask, bid, enter=enter),
        "ensemble": decide_from_prob(p, ask, bid, enter=enter),
    }
    if comp < 0.992:
        return Ensemble(dig, fus, lag, p, comp, BUY, "complement-arb: ask_up+ask_down<1", votes)
    # require at least two of digital/fusion/lag to agree with the blend
    action = votes["ensemble"]
    agree = sum(1 for k in ("digital", "fusion", "lag") if votes[k] == action and action != HOLD)
    if action != HOLD and agree < 2:
        action = HOLD
        reason = "blend wanted a trade; heads disagreed — hold"
    elif action == HOLD:
        reason = "no edge after spread"
    else:
        reason = f"ensemble {action} with {agree} agreeing heads"
    return Ensemble(dig, fus, lag, p, comp, action, reason, votes)
