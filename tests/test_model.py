import mlx.core as mx
import numpy as np

from cmf.config import TrainConfig
from cmf.model import FusionModel
from cmf.ppo import Pretrainer, expiry_aux


def _batch(cfg: TrainConfig, n: int = 2) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "fast": rng.normal(size=(n, cfg.history, cfg.fast_dim)).astype(np.float32),
        "slow": rng.normal(size=(n, cfg.history, cfg.slow_dim)).astype(np.float32),
        "pos": rng.normal(size=(n, cfg.pos_dim)).astype(np.float32),
        "lag": rng.normal(size=(n, cfg.lag_dim)).astype(np.float32),
        "p_up": rng.integers(0, 2, size=n).astype(np.float32),
        "next_ret": rng.normal(size=n).astype(np.float32) * 0.1,
        "lag_y": rng.random(size=n).astype(np.float32),
        "oracle": rng.integers(0, 3, size=n).astype(np.int32),
        "ask": np.full(n, 0.56, dtype=np.float32),
        "bid": np.full(n, 0.52, dtype=np.float32),
    }


def test_forward_shapes_and_finite():
    cfg = TrainConfig(dim=32, layers=1, heads=4, history=8, coarse_layers=1)
    model = FusionModel(cfg)
    mx.eval(model.parameters())
    b = _batch(cfg)
    out = model(mx.array(b["fast"]), mx.array(b["slow"]), mx.array(b["pos"]), mx.array(b["lag"]))
    mx.eval(out.p_up, out.uncertainty, out.logits, out.value, out.log_var)
    assert tuple(out.p_up.shape) == (2,)
    assert tuple(out.uncertainty.shape) == (2,)
    assert tuple(out.logits.shape) == (2, 3)
    assert np.isfinite(np.array(out.p_up)).all()
    assert np.all((np.array(out.uncertainty) >= 0) & (np.array(out.uncertainty) <= 1))


def test_left_padding_is_masked():
    cfg = TrainConfig(dim=32, layers=1, heads=4, history=8, coarse_layers=0)
    model = FusionModel(cfg)
    model.eval()
    mx.eval(model.parameters())
    b = _batch(cfg, n=1)
    b["fast"][0, :3] = 0
    b["slow"][0, :3] = 0
    out = model(mx.array(b["fast"]), mx.array(b["slow"]), mx.array(b["pos"]), mx.array(b["lag"]))
    mx.eval(out.p_up)
    assert np.isfinite(float(np.array(out.p_up)[0]))


def test_pretrain_step_runs():
    cfg = TrainConfig(dim=32, layers=1, heads=4, history=8, coarse_layers=0, pretrain_lr=1e-3)
    model = FusionModel(cfg)
    pre = Pretrainer(cfg, model)
    stats = pre.step(_batch(cfg))
    assert np.isfinite(stats["pretrain_loss"])


def test_expiry_aux_uses_book():
    cfg = TrainConfig(dim=32, layers=1, heads=4, history=8, coarse_layers=0)
    model = FusionModel(cfg)
    mx.eval(model.parameters())
    b = _batch(cfg)
    mxb = {k: mx.array(v) for k, v in b.items()}
    out = model(mxb["fast"], mxb["slow"], mxb["pos"], mxb["lag"])
    loss = expiry_aux(out, mxb, cfg)
    mx.eval(loss)
    assert np.isfinite(float(np.array(loss)))
