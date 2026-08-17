from cmf.policy import BUY, HOLD
from cmf.quant import blend, complement_sum, digital_up_prob, ensemble_signal


def test_digital_itm_goes_to_one():
    p = digital_up_prob(spot=110.0, strike=100.0, tau_sec=30.0, vol=1e-4)
    assert p > 0.9


def test_digital_atm_near_half():
    p = digital_up_prob(spot=100.0, strike=100.0, tau_sec=400.0, vol=5e-4)
    assert 0.35 < p < 0.65


def test_ensemble_needs_agreement():
    sig = ensemble_signal(
        spot=100.0,
        strike=100.0,
        tau_sec=400.0,
        vol=5e-4,
        ret_lead=0.0,
        stale_sec=4.0,
        fusion_p=0.50,
        ask=0.51,
        bid=0.49,
    )
    assert sig.action == HOLD


def test_complement_arb():
    assert complement_sum(0.40, 0.48) < 1.0
    assert blend(0.6, 0.6, 0.6) == 0.6
    _ = BUY
