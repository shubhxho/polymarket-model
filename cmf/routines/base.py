from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Ctx:
    asset: str
    p_ens: float
    p_digital: float
    p_fusion: float
    ask: float
    bid: float
    tte_sec: float
    utc_hour: float
    utc_minute: float
    weekday: int  # 0=Mon


@dataclass
class Vote:
    """delta applied to ensemble p, or veto the trade."""
    name: str
    delta: float = 0.0
    veto: bool = False
    note: str = ""


class Routine(Protocol):
    name: str
    enabled: bool

    def vote(self, ctx: Ctx) -> Vote: ...
