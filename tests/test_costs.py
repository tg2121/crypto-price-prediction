import numpy as np
import pandas as pd
import pytest

from src.costs import per_side_cost_bps, round_trip_cost_bps, trade_costs


def test_round_trip_formula_phase1_costs():
    """Round-trip cost identity from spec: 2 * (spread/2 + slippage) + commission."""
    spread, slippage, commission = 1.0, 5.0, 0.0
    rt = round_trip_cost_bps(spread, slippage, commission)
    expected = 2 * (spread / 2 + slippage) + commission
    assert rt == pytest.approx(expected)
    assert rt == pytest.approx(11.0)


def test_round_trip_formula_with_commission():
    spread, slippage, commission = 2.0, 4.0, 3.0
    rt = round_trip_cost_bps(spread, slippage, commission)
    assert rt == pytest.approx(2 * (spread / 2 + slippage) + commission)


def test_per_side_half_round_trip():
    spread, slippage, commission = 1.0, 5.0, 2.0
    assert 2 * per_side_cost_bps(spread, slippage, commission) == pytest.approx(
        round_trip_cost_bps(spread, slippage, commission)
    )


def test_trade_costs_round_trip_sum_equals_round_trip_formula():
    """Sum of per-bar costs over a full round trip == round_trip_cost_bps / 1e4."""
    pos = pd.Series(
        [0, 1, 1, 1, 0, 0],
        index=pd.date_range("2020-01-01", periods=6, freq="B"),
        dtype=float,
    )
    spread, slippage, commission = 1.0, 5.0, 0.0
    costs = trade_costs(pos, spread, slippage, commission)
    total_bps = float(costs.sum() * 10000)
    assert total_bps == pytest.approx(round_trip_cost_bps(spread, slippage, commission))


def test_trade_costs_no_change_no_cost_after_entry():
    pos = pd.Series([1, 1, 1, 1], index=pd.date_range("2020-01-01", periods=4, freq="B"), dtype=float)
    costs = trade_costs(pos, 1.0, 5.0, 0.0)
    assert costs.iloc[0] > 0
    assert (costs.iloc[1:] == 0).all()


def test_trade_costs_short_entry_costs_same_as_long():
    pos_long = pd.Series([1.0, 1.0], index=pd.date_range("2020-01-01", periods=2, freq="B"))
    pos_short = pd.Series([-1.0, -1.0], index=pd.date_range("2020-01-01", periods=2, freq="B"))
    c_long = trade_costs(pos_long, 1.0, 5.0, 0.0).iloc[0]
    c_short = trade_costs(pos_short, 1.0, 5.0, 0.0).iloc[0]
    assert c_long == pytest.approx(c_short)


def test_trade_costs_zero_when_no_trades():
    pos = pd.Series(np.zeros(10), index=pd.date_range("2020-01-01", periods=10, freq="B"))
    costs = trade_costs(pos, 1.0, 5.0, 0.0)
    assert (costs == 0).all()
