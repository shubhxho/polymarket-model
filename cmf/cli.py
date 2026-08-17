from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from cmf.config import TrainConfig

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def train(
    pretrain_steps: int = typer.Option(400, help="Supervised pretrain steps on lag/expiry/oracle."),
    ppo_updates: int = typer.Option(40, help="PPO updates after pretrain."),
    eval_episodes: int = typer.Option(48, help="Held-out episodes per policy."),
    seed: int = 7,
    checkpoint_dir: Path = Path("checkpoints"),
    dim: int = 96,
    layers: int = 3,
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
    model = FusionModel(cfg)
    model.load_weights(str(load))
    mx.eval(model.parameters())
    import numpy as np

    metrics = evaluate(cfg, "fusion", model, episodes, np.random.default_rng(seed))
    console.print(metrics)


@app.command("live")
def live(
    load: Path = Path("checkpoints/fusion.safetensors"),
    size: float = 50.0,
    assets: str = "BTC,ETH,SOL,XRP",
) -> None:
    """Paper-trade live 15-minute markets with a saved fusion policy."""
    from cmf.live import run_live

    run_live(load=load, size=size, assets=[a.strip().upper() for a in assets.split(",") if a.strip()])


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
