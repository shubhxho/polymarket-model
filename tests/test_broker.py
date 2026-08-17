from cmf.broker import PaperBroker


def test_paper_buy_reduces_cash():
    b = PaperBroker(start_cash=5.0)
    fill = b.buy("BTC", "UP", 2.0, 0.40)
    assert fill.venue == "paper"
    assert abs(b.cash - 3.0) < 1e-9
    assert fill.shares == 5.0
