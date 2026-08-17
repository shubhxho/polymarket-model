"""Download real Binance futures 1m history and cut it into 15-minute windows."""

from __future__ import annotations

import time
from pathlib import Path

import json
import urllib.parse
import urllib.request

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
ASSETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}
KLINE = "https://fapi.binance.com/fapi/v1/klines"


def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    out: list = []
    cur = start_ms
    while cur < end_ms:
        q = urllib.parse.urlencode(
            {"symbol": symbol, "interval": "1m", "startTime": cur, "limit": 1500}
        )
        with urllib.request.urlopen(f"{KLINE}?{q}", timeout=20) as resp:
            batch = json.loads(resp.read().decode())
        if not batch:
            break
        out.extend(batch)
        nxt = int(batch[-1][0]) + 60_000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.08)
    return out


def download(days: int = 45, assets: list[str] | None = None) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    want = assets or list(ASSETS)
    for asset in want:
        symbol = ASSETS[asset]
        print(f"fetch {symbol} {days}d …")
        rows = _fetch_klines(symbol, start, end)
        closes = np.array([float(x[4]) for x in rows], dtype=np.float64)
        path = DATA / f"{asset}_1m.npy"
        np.save(path, closes)
        print(f"  {len(closes)} minutes → {path}")
    return DATA


def upsample_15m(closes_1m: np.ndarray) -> np.ndarray:
    """15 one-minute closes → 900 one-second points (piecewise linear)."""
    if len(closes_1m) < 2:
        return np.repeat(closes_1m, 900)[:900]
    x = np.linspace(0.0, 1.0, len(closes_1m))
    xi = np.linspace(0.0, 1.0, 900)
    return np.interp(xi, x, closes_1m).astype(np.float64)


def load_windows(min_bars: int = 15) -> list[np.ndarray]:
    """Every non-overlapping 15-minute slice across all cached assets."""
    windows: list[np.ndarray] = []
    if not DATA.exists():
        return windows
    for path in sorted(DATA.glob("*_1m.npy")):
        closes = np.load(path)
        n = (len(closes) // min_bars) * min_bars
        for i in range(0, n - min_bars + 1, min_bars):
            sl = closes[i : i + min_bars]
            if sl.min() <= 0 or not np.isfinite(sl).all():
                continue
            windows.append(upsample_15m(sl))
    return windows


def bank_stats(windows: list[np.ndarray]) -> str:
    if not windows:
        return "0 windows"
    rets = [float(w[-1] / w[0] - 1.0) for w in windows]
    return (
        f"{len(windows)} real 15m windows | "
        f"mean ret {np.mean(rets):+.4f} | "
        f"up {100 * np.mean(np.array(rets) > 0):.1f}%"
    )
