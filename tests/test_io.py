from pathlib import Path

from cmf.config import TrainConfig
from cmf.io import load_bundle, save_bundle
from cmf.model import FusionModel
import mlx.core as mx


def test_safetensors_roundtrip(tmp_path: Path):
    cfg = TrainConfig(dim=32, layers=1, heads=4, history=8)
    model = FusionModel(cfg)
    mx.eval(model.parameters())
    save_bundle(model, cfg, tmp_path)
    assert (tmp_path / "fusion.safetensors").exists()
    spec = (tmp_path / "model.json").read_text()
    assert "safetensors" in spec
    loaded, cfg2 = load_bundle(tmp_path)
    assert cfg2.dim == 32
    assert loaded.count_params() == model.count_params()
