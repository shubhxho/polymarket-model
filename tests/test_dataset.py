import numpy as np

from cmf.dataset import upsample_15m


def test_upsample_length():
    w = upsample_15m(np.linspace(100.0, 101.0, 15))
    assert w.shape == (900,)
    assert abs(w[0] - 100.0) < 1e-6
    assert abs(w[-1] - 101.0) < 1e-6
    assert np.all(np.diff(w) >= -1e-12)
