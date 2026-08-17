from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

SideName = Literal["UP", "DOWN"]
Mode = Literal["paper", "live"]


@dataclass
class Fill:
    ts: float
    asset: str
    side: SideName
    usd: float
    price: float
    shares: float
    venue: str
    order_id: str
    note: str = ""


@dataclass
class BrokerStatus:
    mode: Mode
    live_ready: bool
    live_armed: bool
    address: str
    collateral: str
    paper_pnl: float
    spent_today: float
    max_usd: float
    max_daily_loss: float
    sdk: str
    reason: str = ""


class PaperBroker:
    """Instant fills at the displayed bid/ask. No chain, no keys."""

    def __init__(self, start_cash: float = 5.0):
        self.cash = float(start_cash)
        self.pnl = 0.0
        self.fills: list[Fill] = []
        self.positions: dict[str, dict[str, float]] = {}

    def buy(self, asset: str, side: SideName, usd: float, price: float) -> Fill:
        px = max(price, 0.02)
        spend = min(usd, self.cash)
        shares = spend / px
        self.cash -= spend
        pos = self.positions.setdefault(asset, {"side": 0.0, "shares": 0.0, "entry": 0.0})
        pos["side"] = 1.0 if side == "UP" else -1.0
        pos["shares"] = shares
        pos["entry"] = px
        fill = Fill(time.time(), asset, side, spend, px, shares, "paper", f"paper-{len(self.fills)+1}")
        self.fills.append(fill)
        return fill

    def mark(self, asset: str, mid: float) -> float:
        pos = self.positions.get(asset)
        if not pos or pos["shares"] <= 0:
            return 0.0
        mark = mid if pos["side"] > 0 else 1.0 - mid
        return (mark - pos["entry"]) * pos["shares"]


class LiveBroker:
    """
    Polymarket CLOB V2. Collateral is pUSD (USDC-backed), not USDT.

    Requires POLY_PRIVATE_KEY. Optional POLY_FUNDER, POLY_SIGNATURE_TYPE.
    Will not send unless arm() was called and CMF_LIVE=1.
    """

    def __init__(self, max_usd: float, max_daily_loss: float):
        self.max_usd = max_usd
        self.max_daily_loss = max_daily_loss
        self.armed = False
        self.spent_today = 0.0
        self.realized = 0.0
        self.fills: list[Fill] = []
        self.address = ""
        self.sdk_name = "missing"
        self._client: Any = None
        self._err = "set POLY_PRIVATE_KEY and uv sync --group trade"
        self._connect()

    def _connect(self) -> None:
        key = os.environ.get("POLY_PRIVATE_KEY", "").strip()
        if not key:
            return
        try:
            from py_clob_client_v2 import ClobClient  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._err = f"install py-clob-client-v2 ({exc})"
            return
        host = os.environ.get("POLY_CLOB_HOST", "https://clob.polymarket.com")
        funder = os.environ.get("POLY_FUNDER") or None
        sig = int(os.environ.get("POLY_SIGNATURE_TYPE", "0"))
        try:
            self._client = ClobClient(
                {
                    "host": host,
                    "chain": 137,
                    "key": key,
                    "signature_type": sig,
                    "funder": funder,
                }
            )
            if hasattr(self._client, "create_or_derive_api_creds"):
                creds = self._client.create_or_derive_api_creds()
                self._client.set_api_creds(creds)
            self.sdk_name = "py-clob-client-v2"
            self.address = funder or getattr(self._client, "address", "") or "derived"
            self._err = ""
        except TypeError:
            # some builds still take kwargs
            try:
                self._client = ClobClient(
                    host=host,
                    chain=137,
                    key=key,
                    signature_type=sig,
                    funder=funder,
                )
                self.sdk_name = "py-clob-client-v2"
                self.address = funder or "derived"
                self._err = ""
            except Exception as exc:  # noqa: BLE001
                self._err = f"clob client init failed: {exc}"
                self._client = None
        except Exception as exc:  # noqa: BLE001
            self._err = f"clob client init failed: {exc}"
            self._client = None

    @property
    def ready(self) -> bool:
        return self._client is not None and not self._err

    def arm(self, confirm: str) -> None:
        if os.environ.get("CMF_LIVE") != "1":
            raise RuntimeError("set CMF_LIVE=1 in the environment first")
        if confirm != "I_UNDERSTAND_REAL_ORDERS":
            raise RuntimeError("pass confirm=I_UNDERSTAND_REAL_ORDERS")
        if not self.ready:
            raise RuntimeError(self._err or "live client not ready")
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def buy(self, asset: str, token_id: str, side: SideName, usd: float, price: float) -> Fill:
        if not self.armed:
            raise RuntimeError("live broker is not armed")
        usd = min(usd, self.max_usd)
        if usd <= 0:
            raise RuntimeError("size is zero")
        if self.spent_today + usd > self.max_daily_loss + self.max_usd * 8:
            raise RuntimeError("daily notional cap")
        from py_clob_client_v2 import MarketOrderArgs, OrderType, Side  # type: ignore

        args = MarketOrderArgs(
            token_id=token_id,
            amount=float(usd),
            side=Side.BUY,
            order_type=OrderType.FAK,
        )
        resp = self._client.create_and_post_market_order(order_args=args, order_type=OrderType.FAK)
        oid = str((resp or {}).get("orderID") or (resp or {}).get("id") or "unknown")
        self.spent_today += usd
        fill = Fill(time.time(), asset, side, usd, price, usd / max(price, 0.02), "clob-v2", oid, note=str(resp)[:240])
        self.fills.append(fill)
        return fill


@dataclass
class ExecutionHub:
    paper: PaperBroker
    live: LiveBroker
    mode: Mode = "paper"
    fills: list[Fill] = field(default_factory=list)

    def status(self) -> BrokerStatus:
        return BrokerStatus(
            mode=self.mode,
            live_ready=self.live.ready,
            live_armed=self.live.armed,
            address=self.live.address,
            collateral="pUSD (USDC-backed on Polygon). Not USDT.",
            paper_pnl=self.paper.pnl + sum(self.paper.mark(a, 0.5) for a in self.paper.positions),
            spent_today=self.live.spent_today,
            max_usd=self.live.max_usd,
            max_daily_loss=self.live.max_daily_loss,
            sdk=self.live.sdk_name,
            reason=self.live._err,
        )

    def submit(self, *, asset: str, token_id: str, side: SideName, usd: float, price: float) -> Fill:
        if self.mode == "paper":
            fill = self.paper.buy(asset, side, usd, price)
        else:
            fill = self.live.buy(asset, token_id, side, usd, price)
        self.fills.append(fill)
        return fill
