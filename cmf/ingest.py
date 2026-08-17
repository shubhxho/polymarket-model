"""Fan-out fetch: Gamma, Binance mark, CLOB midpoint — all at once."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import aiohttp

from cmf.live import GAMMA_API, fetch_15m_markets

BINANCE = "https://fapi.binance.com"


async def _json(session: aiohttp.ClientSession, url: str) -> Any:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
        resp.raise_for_status()
        return await resp.json()


async def binance_marks(session: aiohttp.ClientSession, assets: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    symbols = {a: f"{a}USDT" for a in assets}

    async def one(asset: str, symbol: str) -> None:
        try:
            prem = await _json(session, f"{BINANCE}/fapi/v1/premiumIndex?symbol={symbol}")
            kl = await _json(session, f"{BINANCE}/fapi/v1/klines?symbol={symbol}&interval=1s&limit=60")
            closes = [float(k[4]) for k in kl]
            rets = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
                if closes[i - 1] > 0
            ]
            vol = float(__import__("numpy").std(rets)) if rets else 1e-4
            lead = (closes[-1] - closes[max(0, len(closes) - 8)]) / closes[max(0, len(closes) - 8)] if closes else 0.0
            out[asset] = {
                "mark": float(prem["markPrice"]),
                "index": float(prem["indexPrice"]),
                "vol_1s": vol,
                "ret_8s": float(lead),
            }
        except Exception:
            out[asset] = {"mark": 0.0, "index": 0.0, "vol_1s": 1e-4, "ret_8s": 0.0}

    await asyncio.gather(*(one(a, s) for a, s in symbols.items()))
    return out


async def clob_mid(session: aiohttp.ClientSession, token_id: str) -> dict[str, float]:
    url = f"https://clob.polymarket.com/midpoint?token_id={token_id}"
    try:
        data = await _json(session, url)
        mid = float(data.get("mid") or data.get("midpoint") or 0.5)
        return {"mid": mid}
    except Exception:
        return {"mid": 0.5}


async def snapshot(assets: list[str]) -> dict[str, Any]:
    """One coordinated pull of markets + futures marks."""
    async with aiohttp.ClientSession() as session:
        markets, marks = await asyncio.gather(
            fetch_15m_markets(assets),
            binance_marks(session, assets),
        )
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "markets": markets,
        "marks": marks,
    }
