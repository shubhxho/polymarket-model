from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque

import aiohttp
import numpy as np
import websockets

from cmf.config import TrainConfig
from cmf.features import RAW_FAST_DIM, RAW_SLOW_DIM, featurize_episode, window_at
from cmf.model import FusionModel
from cmf.policy import decide_from_prob

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_WSS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
FUTURES_WSS = "wss://fstream.binance.com"
FUTURES = {"BTC": "btcusdt", "ETH": "ethusdt", "SOL": "solusdt", "XRP": "xrpusdt"}


@dataclass
class LiveBook:
    bid: float = 0.0
    ask: float = 0.0
    bid_sz: float = 0.0
    ask_sz: float = 0.0
    mid: float = 0.5


class FuturesTape:
    def __init__(self, assets: list[str]):
        self.assets = assets
        self.mid: dict[str, float] = {a: 0.0 for a in assets}
        self.qty: dict[str, float] = {a: 0.0 for a in assets}
        self.sign: dict[str, float] = {a: 0.0 for a in assets}
        self.cvd: dict[str, float] = {a: 0.0 for a in assets}

    async def stream(self, stop: asyncio.Event) -> None:
        streams = "/".join(f"{FUTURES[a]}@aggTrade" for a in self.assets if a in FUTURES)
        url = f"{FUTURES_WSS}/stream?streams={streams}"
        while not stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    while not stop.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=8)
                        msg = json.loads(raw)
                        d = msg.get("data") or msg
                        if "s" not in d:
                            continue
                        symbol = d["s"].lower()
                        asset = next((a for a, s in FUTURES.items() if s == symbol), None)
                        if asset is None:
                            continue
                        px = float(d["p"])
                        qty = float(d["q"])
                        sign = -1.0 if d.get("m") else 1.0
                        self.mid[asset] = px
                        self.qty[asset] = qty
                        self.sign[asset] = sign
                        self.cvd[asset] += sign * qty * px
            except Exception:
                await asyncio.sleep(1.0)


class PolymarketBooks:
    def __init__(self) -> None:
        self.books: dict[str, LiveBook] = {}
        self.token_to_cid: dict[str, str] = {}

    def subscribe(self, cid: str, token_up: str) -> None:
        self.books.setdefault(cid, LiveBook())
        self.token_to_cid[token_up] = cid

    async def stream(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            tokens = list(self.token_to_cid)
            if not tokens:
                await asyncio.sleep(1.0)
                continue
            try:
                async with websockets.connect(CLOB_WSS, ping_interval=20) as ws:
                    await ws.send(json.dumps({"assets_ids": tokens, "type": "market"}))
                    while not stop.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=8)
                        msg = json.loads(raw)
                        events = msg if isinstance(msg, list) else [msg]
                        for ev in events:
                            aid = ev.get("asset_id") or ev.get("market")
                            cid = self.token_to_cid.get(str(aid))
                            if cid is None:
                                continue
                            bids = ev.get("bids") or []
                            asks = ev.get("asks") or []
                            if not bids or not asks:
                                continue
                            bid_px, bid_sz = float(bids[0]["price"]), float(bids[0]["size"])
                            ask_px, ask_sz = float(asks[0]["price"]), float(asks[0]["size"])
                            book = self.books[cid]
                            book.bid, book.ask = bid_px, ask_px
                            book.bid_sz, book.ask_sz = bid_sz, ask_sz
                            book.mid = 0.5 * (bid_px + ask_px)
            except Exception:
                await asyncio.sleep(1.5)


async def fetch_15m_markets(assets: list[str]) -> list[dict]:
    url = f"{GAMMA_API}/markets?closed=false&limit=200"
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url, timeout=aiohttp.ClientTimeout(total=12)) as resp:
            data = await resp.json()
    out = []
    now = datetime.now(timezone.utc)
    for m in data:
        q = (m.get("question") or "") + " " + (m.get("slug") or "")
        asset = next((a for a in assets if a.lower() in q.lower()), None)
        if asset is None:
            continue
        end_s = m.get("endDate") or m.get("end_date_iso")
        if not end_s:
            continue
        end = datetime.fromisoformat(end_s.replace("Z", "+00:00"))
        left = (end - now).total_seconds()
        if left < 45 or left > 16 * 60:
            continue
        tokens = m.get("clobTokenIds") or m.get("clob_token_ids") or []
        if isinstance(tokens, str):
            tokens = json.loads(tokens)
        if len(tokens) < 2:
            continue
        out.append(
            {
                "asset": asset,
                "cid": m.get("conditionId") or m.get("condition_id"),
                "token_up": tokens[0],
                "end": end,
            }
        )
    # one market per asset
    best: dict[str, dict] = {}
    for row in out:
        prev = best.get(row["asset"])
        if prev is None or row["end"] < prev["end"]:
            best[row["asset"]] = row
    return list(best.values())


def _raw_ticks(asset: str, tape: FuturesTape, book: LiveBook, tte: float, t: int) -> tuple[np.ndarray, np.ndarray]:
    mid_f = tape.mid[asset] or 1.0
    fast = np.zeros(RAW_FAST_DIM, dtype=np.float32)
    slow = np.zeros(RAW_SLOW_DIM, dtype=np.float32)
    fast[0] = t
    fast[1] = mid_f
    fast[2] = mid_f * 0.9997
    fast[3] = mid_f * 1.0003
    fast[4] = 50.0
    fast[5] = 50.0
    fast[6] = tape.qty[asset]
    fast[7] = tape.sign[asset]
    fast[8] = tape.qty[asset] * mid_f
    fast[9] = 1.0 if tape.qty[asset] > 0 else 0.0
    mid_s = book.mid or 0.5
    slow[0] = t
    slow[1] = mid_s
    slow[2] = book.bid or max(mid_s - 0.01, 0.01)
    slow[3] = book.ask or min(mid_s + 0.01, 0.99)
    slow[4] = book.bid_sz or 10.0
    slow[5] = book.ask_sz or 10.0
    slow[8] = (book.bid_sz or 10.0) * 3
    slow[9] = (book.ask_sz or 10.0) * 3
    slow[10] = tte
    return fast, slow


async def _loop(load: Path, size: float, assets: list[str]) -> None:
    import mlx.core as mx

    cfg = TrainConfig()
    model = FusionModel(cfg)
    model.load_weights(str(load))
    model.eval()
    mx.eval(model.parameters())

    tape = FuturesTape(assets)
    books = PolymarketBooks()
    stop = asyncio.Event()
    history: dict[str, Deque[tuple[np.ndarray, np.ndarray]]] = {}

    async def markets_refresh() -> list[dict]:
        markets = await fetch_15m_markets(assets)
        for m in markets:
            books.subscribe(m["cid"], m["token_up"])
            history.setdefault(m["cid"], deque(maxlen=cfg.history))
        return markets

    markets = await markets_refresh()
    print(f"live paper | {len(markets)} markets | load={load} | size=${size:.0f}")
    tasks = [
        asyncio.create_task(tape.stream(stop)),
        asyncio.create_task(books.stream(stop)),
    ]
    try:
        tick = 0
        while True:
            await asyncio.sleep(1.0)
            tick += 1
            now = datetime.now(timezone.utc)
            if tick % 30 == 0:
                markets = await markets_refresh()
            for m in markets:
                left = (m["end"] - now).total_seconds()
                if left <= 5:
                    continue
                book = books.books.get(m["cid"], LiveBook())
                if book.mid <= 0:
                    continue
                fast, slow = _raw_ticks(m["asset"], tape, book, left / 900.0, tick)
                history[m["cid"]].append((fast, slow))
                buf = list(history[m["cid"]])
                if len(buf) < 8:
                    continue
                fr = np.stack([x[0] for x in buf])
                sr = np.stack([x[1] for x in buf])
                ff, sf, lg = featurize_episode(fr, sr)
                t = ff.shape[0] - 1
                fast_w = window_at(ff, t, cfg.history)[None, ...]
                slow_w = window_at(sf, t, cfg.history)[None, ...]
                pos = np.zeros((1, cfg.pos_dim), dtype=np.float32)
                pos[0, 5] = left / 900.0
                lag = lg[t][None, ...]
                out = model(mx.array(fast_w), mx.array(slow_w), mx.array(pos), mx.array(lag))
                mx.eval(out.logits, out.p_up)
                p_up = float(1.0 / (1.0 + np.exp(-float(np.array(out.p_up)[0]))))
                action = decide_from_prob(p_up, float(book.ask or book.mid), float(book.bid or book.mid))
                names = {0: "HOLD", 1: "BUY UP", 2: "BUY DOWN"}
                if action != 0:
                    print(
                        f"{now.strftime('%H:%M:%S')} {m['asset']:4} {names[action]:9} "
                        f"mid={book.mid:.3f} p_up={p_up:.3f} tte={left/60:.1f}m size=${size:.0f}"
                    )
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for t in tasks:
            t.cancel()


def run_live(load: Path, size: float, assets: list[str]) -> None:
    asyncio.run(_loop(load, size, assets))
