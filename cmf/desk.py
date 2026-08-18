from __future__ import annotations

import asyncio
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from cmf.broker import ExecutionHub, LiveBroker, PaperBroker
from cmf.config import TrainConfig
from cmf.features import featurize_episode, window_at
from cmf.live import FuturesTape, LiveBook, PolymarketBooks, _raw_ticks, fetch_15m_markets
from cmf.ingest import binance_marks
from cmf.model import FusionModel
from cmf.quant import ensemble_signal
from cmf.routines import RoutineBank
from cmf.routines.base import Ctx

DOCS = Path(__file__).resolve().parents[1] / "docs"


class DeskState:
    def __init__(self, load: Path, size: float, assets: list[str], cash: float):
        self.size = size
        self.assets = assets
        self.hub = ExecutionHub(
            paper=PaperBroker(start_cash=cash),
            live=LiveBroker(
                max_usd=float(os.environ.get("CMF_MAX_USD", str(size))),
                max_daily_loss=float(os.environ.get("CMF_MAX_DAILY_LOSS", "25")),
            ),
        )
        self.tape = FuturesTape(assets)
        self.books = PolymarketBooks()
        self.markets: list[dict] = []
        self.history: dict[str, deque] = {}
        self.cards: dict[str, dict[str, Any]] = {}
        self.opens: dict[str, float] = {}
        self.marks: dict[str, dict[str, float]] = {}
        self.log: list[dict[str, Any]] = []
        self.tick = 0
        self.model_ok = False
        self.load = load
        self._model: FusionModel | None = None
        self._mx = None
        self.routines = RoutineBank()

    def load_model(self) -> None:
        import mlx.core as mx

        from cmf.io import load_bundle

        if self.load.exists() or (self.load.parent / "model.json").exists():
            model, _cfg = load_bundle(self.load if self.load.suffix else self.load.parent)
        else:
            model = FusionModel(TrainConfig())
        model.eval()
        mx.eval(model.parameters())
        self._model = model
        self._mx = mx
        self.model_ok = True

    def snapshot(self) -> dict[str, Any]:
        st = self.hub.status()
        return {
            "mode": st.mode,
            "live_ready": st.live_ready,
            "live_armed": st.live_armed,
            "address": st.address,
            "collateral": st.collateral,
            "paper_cash": self.hub.paper.cash,
            "paper_pnl": self.hub.paper.pnl,
            "spent_today": st.spent_today,
            "max_usd": st.max_usd,
            "sdk": st.sdk,
            "reason": st.reason,
            "model_ok": self.model_ok,
            "tick": self.tick,
            "cards": list(self.cards.values()),
            "fills": [
                {
                    "ts": f.ts,
                    "asset": f.asset,
                    "side": f.side,
                    "usd": f.usd,
                    "price": f.price,
                    "venue": f.venue,
                    "order_id": f.order_id,
                }
                for f in self.hub.fills[-40:]
            ],
            "log": self.log[-40:],
            "routines": self.routines.list(),
        }

    def note(self, msg: str) -> None:
        self.log.append({"ts": datetime.now(timezone.utc).strftime("%H:%M:%S"), "msg": msg})


async def engine_loop(state: DeskState, stop: asyncio.Event) -> None:
    cfg = TrainConfig()
    while not stop.is_set():
        await asyncio.sleep(1.0)
        state.tick += 1
        now = datetime.now(timezone.utc)
        if state.tick == 1 or state.tick % 30 == 0:
            try:
                state.markets = await fetch_15m_markets(state.assets)
                for m in state.markets:
                    state.books.subscribe(m["cid"], m["token_up"])
                    state.history.setdefault(m["cid"], deque(maxlen=cfg.history))
            except Exception as exc:  # noqa: BLE001
                state.note(f"market refresh failed: {exc}")
        if state.tick == 1 or state.tick % 10 == 0:
            try:
                async with aiohttp.ClientSession() as session:
                    state.marks = await binance_marks(session, state.assets)
            except Exception as exc:  # noqa: BLE001
                state.note(f"mark refresh failed: {exc}")
        if state._model is None:
            continue
        mx = state._mx
        import numpy as np

        names = {0: "HOLD", 1: "BUY UP", 2: "BUY DOWN"}
        for m in state.markets:
            left = (m["end"] - now).total_seconds()
            book = state.books.books.get(m["cid"], LiveBook())
            if book.mid <= 0 or left <= 8:
                continue
            fast, slow = _raw_ticks(m["asset"], state.tape, book, left / 900.0, state.tick)
            state.history[m["cid"]].append((fast, slow))
            buf = list(state.history[m["cid"]])
            if len(buf) < 8:
                continue
            fr = np.stack([x[0] for x in buf])
            sr = np.stack([x[1] for x in buf])
            ff, sf, lg = featurize_episode(fr, sr)
            t = ff.shape[0] - 1
            pos = np.zeros((1, cfg.pos_dim), dtype="float32")
            pos[0, 5] = left / 900.0
            out = state._model(
                mx.array(window_at(ff, t, cfg.history)[None, ...]),
                mx.array(window_at(sf, t, cfg.history)[None, ...]),
                mx.array(pos),
                mx.array(lg[t][None, ...]),
            )
            mx.eval(out.p_up, out.uncertainty)
            fusion_p = float(1.0 / (1.0 + np.exp(-float(np.array(out.p_up)[0]))))
            fusion_unc = float(np.array(out.uncertainty)[0])
            ask = float(book.ask or book.mid)
            bid = float(book.bid or book.mid)
            spot = float(state.tape.mid.get(m["asset"]) or state.marks.get(m["asset"], {}).get("mark") or 0.0)
            if m["cid"] not in state.opens and spot > 0:
                state.opens[m["cid"]] = spot
            strike = state.opens.get(m["cid"], spot)
            mk = state.marks.get(m["asset"], {})
            sig = ensemble_signal(
                spot=spot or 1.0,
                strike=strike or 1.0,
                tau_sec=left,
                vol=float(mk.get("vol_1s") or 1e-4),
                ret_lead=float(mk.get("ret_8s") or 0.0),
                stale_sec=4.0,
                fusion_p=fusion_p,
                ask=ask,
                bid=bid,
            )
            now_utc = datetime.now(timezone.utc)
            p_adj, rvotes = self.routines.apply(
                Ctx(
                    asset=m["asset"],
                    p_ens=sig.ensemble,
                    p_digital=sig.digital,
                    p_fusion=sig.fusion,
                    ask=ask,
                    bid=bid,
                    tte_sec=left,
                    utc_hour=now_utc.hour,
                    utc_minute=now_utc.minute,
                    weekday=now_utc.weekday(),
                ),
                sig.ensemble,
            )
            if any(v.veto for v in rvotes):
                action = 0
            else:
                from cmf.policy import decide_from_prob

                action = decide_from_prob(
                    p_adj, ask, bid, uncertainty=fusion_unc, tte=left / 900.0
                )
                if sig.action == 0 and action != 0 and sig.complement >= 0.992:
                    # still require two model heads unless complement-arb
                    action = sig.action
            card = {
                "asset": m["asset"],
                "cid": m["cid"],
                "token_up": m["token_up"],
                "mid": book.mid,
                "bid": bid,
                "ask": ask,
                "bn": spot,
                "p_up": p_adj,
                "p_raw": sig.ensemble,
                "fusion": sig.fusion,
                "digital": sig.digital,
                "lag": sig.lag,
                "complement": sig.complement,
                "up_edge": sig.ensemble - ask,
                "down_edge": (1.0 - sig.ensemble) - (1.0 - bid),
                "action": names[action],
                "reason": sig.reason,
                "routines": [{"name": v.name, "delta": v.delta, "note": v.note} for v in rvotes],
                "tte_min": left / 60.0,
            }
            state.cards[m["asset"]] = card
            if action == 0:
                continue
            side = "UP" if action == 1 else "DOWN"
            token = m["token_up"] if side == "UP" else m.get("token_down", "")
            px = ask if side == "UP" else (1.0 - bid)
            try:
                fill = state.hub.submit(
                    asset=m["asset"],
                    token_id=token,
                    side=side,
                    usd=state.size,
                    price=px,
                )
                state.note(f"{fill.venue} {side} {m['asset']} ${fill.usd:.2f} @ {px:.3f} · {sig.reason}")
            except Exception as exc:  # noqa: BLE001
                state.note(f"order skipped: {exc}")


def make_app(state: DeskState) -> web.Application:
    app = web.Application()

    async def index(_req: web.Request) -> web.FileResponse:
        return web.FileResponse(DOCS / "desk.html")

    async def api_state(_req: web.Request) -> web.Response:
        return web.json_response(state.snapshot())

    async def api_arm(req: web.Request) -> web.Response:
        body = await req.json()
        try:
            state.hub.live.arm(str(body.get("confirm", "")))
            state.hub.mode = "live"
            state.note("LIVE ARMED — real CLOB orders")
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **state.snapshot()})

    async def api_routines(req: web.Request) -> web.Response:
        body = await req.json()
        state.routines.set(str(body.get("name", "")), bool(body.get("enabled", True)))
        return web.json_response(state.snapshot())

    async def api_paper(_req: web.Request) -> web.Response:
        state.hub.live.disarm()
        state.hub.mode = "paper"
        state.note("back to paper")
        return web.json_response(state.snapshot())

    app.router.add_get("/", index)
    app.router.add_get("/desk.html", index)
    app.router.add_static("/static", str(DOCS), show_index=False)
    app.router.add_get("/styles.css", lambda r: web.FileResponse(DOCS / "desk.css"))
    app.router.add_get("/desk.css", lambda r: web.FileResponse(DOCS / "desk.css"))
    app.router.add_get("/desk.js", lambda r: web.FileResponse(DOCS / "desk.js"))
    app.router.add_get("/api/state", api_state)
    app.router.add_post("/api/arm", api_arm)
    app.router.add_post("/api/paper", api_paper)
    app.router.add_post("/api/routines", api_routines)
    return app


async def run_desk(load: Path, size: float, assets: list[str], cash: float, port: int) -> None:
    state = DeskState(load, size, assets, cash)
    state.load_model()
    state.note("desk up — paper mode. Collateral on Polymarket is pUSD, not USDT.")
    stop = asyncio.Event()
    app = make_app(state)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    print(f"desk → http://127.0.0.1:{port}/")
    tasks = [
        asyncio.create_task(state.tape.stream(stop)),
        asyncio.create_task(state.books.stream(stop)),
        asyncio.create_task(engine_loop(state, stop)),
    ]
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await runner.cleanup()
