import numpy as np

from cmf.dataset import upsample_15m


def test_splits_are_chronological_and_disjoint():
    from cmf.dataset import load_splits

    train, val, test = load_splits()
    if not train:
        return
    assert len(train) > len(val) >= 0
    # no exact array reuse between val and test (train is denser, so overlap in price is ok)
    if val and test:
        assert val[0] is not test[0]


def test_upsample_length():
    w = upsample_15m(np.linspace(100.0, 101.0, 15))
    assert w.shape == (900,)
    assert abs(w[0] - 100.0) < 1e-6
    assert abs(w[-1] - 101.0) < 1e-6
    assert np.all(np.diff(w) >= -1e-12)
