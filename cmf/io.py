"""Load/save CMF-2 as SafeTensors + model.json only."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from cmf.config import TrainConfig
from cmf.model import FusionModel


def save_bundle(model: FusionModel, cfg: TrainConfig, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "fusion.safetensors"
    spec = directory / "model.json"
    model.save_weights(str(weights))
    payload = json.loads(cfg.model_dump_json())
    payload["format"] = "safetensors"
    payload["params"] = model.count_params()
    payload["bytes_f32"] = model.count_params() * 4
    spec.write_text(json.dumps(payload, indent=2))
    return weights


def load_bundle(directory: Path | str, filename: str = "fusion.safetensors") -> tuple[FusionModel, TrainConfig]:
    directory = Path(directory)
    if directory.is_file():
        weights = directory
        directory = directory.parent
    else:
        weights = directory / filename
    spec = directory / "model.json"
    if spec.exists():
        cfg = TrainConfig.model_validate_json(spec.read_text())
    else:
        cfg = TrainConfig()
    if not weights.exists():
        raise FileNotFoundError(weights)
    model = FusionModel(cfg)
    model.load_weights(str(weights))
    mx.eval(model.parameters())
    if model.count_params() * 4 != weights.stat().st_size and abs(model.count_params() * 4 - weights.stat().st_size) > 4096:
        # header overhead is fine; explode only on huge mismatch (wrong architecture)
        expected = model.count_params() * 4
        actual = weights.stat().st_size
        if actual < expected * 0.5 or actual > expected * 2:
            raise RuntimeError(f"weight file {actual} B does not match model.json ({expected} B of F32)")
    return model, cfg
