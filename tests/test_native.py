import numpy as np

from cmf.features import FAST_DIM, LAG_DIM, SLOW_DIM, featurize_episode, has_native


def test_native_engine_is_built():
    assert has_native()


def test_featurize_shapes_and_finite():
    rng = np.random.default_rng(0)
    t = 48
    fast = np.zeros((t, 16), np.float32)
    slow = np.zeros((t, 12), np.float32)
    p, m = 100.0, 0.5
    for i in range(t):
        p *= 1.0 + float(rng.normal(0, 0.001))
        m = float(np.clip(m + rng.normal(0, 0.01), 0.08, 0.92))
        fast[i] = [i, p, p - 0.05, p + 0.05, 80, 70, 2.0, 1.0, p, 1, 0, 0, 0, 0, 0, 0]
        slow[i] = [i, m, m - 0.01, m + 0.01, 100, 90, 1, 1, 300, 270, 1 - i / t, 2]
    ff, sf, lg = featurize_episode(fast, slow)
    assert ff.shape == (t, FAST_DIM)
    assert sf.shape == (t, SLOW_DIM)
    assert lg.shape == (t, LAG_DIM)
    assert np.isfinite(ff).all()
    assert np.isfinite(sf).all()
    assert np.isfinite(lg).all()
