from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from cmf.config import TrainConfig

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def train(
    pretrain_steps: int = typer.Option(500, help="Supervised pretrain steps on lag/expiry/oracle."),
    ppo_updates: int = typer.Option(24, help="PPO updates after pretrain."),
    eval_episodes: int = typer.Option(64, help="Held-out episodes per policy."),
    seed: int = 7,
    checkpoint_dir: Path = Path("checkpoints"),
    dim: int = 384,
    layers: int = 6,
    heads: int = 6,
) -> None:
    """Pretrain the fusion transformer, then PPO-train it against a LACUNA clone."""
    from cmf.train import train as run

    cfg = TrainConfig(
        pretrain_steps=pretrain_steps,
        ppo_updates=ppo_updates,
        eval_episodes=eval_episodes,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        dim=dim,
        layers=layers,
        heads=heads,
    )
    run(cfg)


@app.command()
def eval(
    load: Path = Path("checkpoints/fusion.safetensors"),
    episodes: int = 64,
    seed: int = 123,
) -> None:
    """Evaluate a saved fusion checkpoint against LACUNA/oracle/random."""
    import mlx.core as mx

    from cmf.model import FusionModel
    from cmf.train import evaluate

    cfg = TrainConfig(eval_episodes=episodes, seed=seed)
    from cmf.io import load_bundle

    model, loaded = load_bundle(load)
    cfg = loaded
    cfg.eval_episodes = episodes
    cfg.seed = seed
    mx.eval(model.parameters())
    import numpy as np

    metrics = evaluate(cfg, "fusion", model, episodes, np.random.default_rng(seed))
    console.print(metrics)


@app.command("live")
def live(
    load: Path = Path("checkpoints/fusion.safetensors"),
    size: float = 5.0,
    assets: str = "BTC,ETH,SOL,XRP",
    real: bool = typer.Option(False, "--live", help="Send real CLOB V2 orders. Requires CMF_LIVE=1 and a key."),
    confirm: str = typer.Option("", help="Must be I_UNDERSTAND_REAL_ORDERS with --live."),
) -> None:
    """Run the fusion policy on live 15-minute books. Paper unless --live."""
    from cmf.live import run_live

    run_live(
        load=load,
        size=size,
        assets=[a.strip().upper() for a in assets.split(",") if a.strip()],
        live=real,
        confirm=confirm,
    )


@app.command()
def desk(
    load: Path = Path("checkpoints/fusion.safetensors"),
    size: float = 5.0,
    cash: float = 5.0,
    port: int = 4174,
    assets: str = "BTC,ETH,SOL,XRP",
) -> None:
    """Open the execution desk (paper by default)."""
    import asyncio

    from cmf.desk import run_desk

    asyncio.run(
        run_desk(
            load=load,
            size=size,
            assets=[a.strip().upper() for a in assets.split(",") if a.strip()],
            cash=cash,
            port=port,
        )
    )


@app.command("fetch-data")
def fetch_data(days: int = 45) -> None:
    """Download Binance USDT-M 1m history and cut 15-minute windows."""
    from cmf.dataset import bank_stats, download, load_windows

    download(days=days)
    console.print(bank_stats(load_windows()))


@app.command()
def signal(assets: str = "BTC,ETH,SOL,XRP") -> None:
    """One-shot: fetch Gamma + Binance + books, print every head and the ensemble."""
    import asyncio

    from cmf.ingest import snapshot
    from cmf.quant import ensemble_signal

    snap = asyncio.run(snapshot([a.strip().upper() for a in assets.split(",") if a.strip()]))
    for m in snap["markets"]:
        mk = snap["marks"].get(m["asset"], {})
        spot = float(mk.get("mark") or 0.0)
        sig = ensemble_signal(
            spot=spot or 1.0,
            strike=spot or 1.0,
            tau_sec=max((m["end"] - __import__("datetime").datetime.now(__import__("datetime").timezone.utc)).total_seconds(), 1.0),
            vol=float(mk.get("vol_1s") or 1e-4),
            ret_lead=float(mk.get("ret_8s") or 0.0),
            stale_sec=4.0,
            fusion_p=0.5,
            ask=0.52,
            bid=0.48,
        )
        console.print(
            f"{m['asset']:4} digital={sig.digital:.3f} lag={sig.lag:.3f} "
            f"ens={sig.ensemble:.3f} {sig.reason}"
        )
    if not snap["markets"]:
        console.print("no active 15m markets")


@app.command()
def docs(port: int = 4173) -> None:
    """Serve the notation book at docs/ on localhost."""
    import http.server
    import socketserver
    from functools import partial

    root = Path(__file__).resolve().parents[1] / "docs"
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        console.print(f"notation book → http://127.0.0.1:{port}/")
        httpd.serve_forever()


if __name__ == "__main__":
    app()
