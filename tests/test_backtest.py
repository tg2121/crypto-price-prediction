"""Critical correctness tests: no look-ahead bias and correct cost application.

Look-ahead is verified two ways:
  (1) recompute positions on a truncated price series; positions on the
      shared dates must be byte-identical to the full-sample positions.
  (2) perturb the LAST close; positions for ALL prior dates must be unchanged
      (because position[t] is supposed to depend only on close[<t]).
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.costs import round_trip_cost_bps
from src.strategies import buy_and_hold, momentum_12_1, sma_50_200


def make_synthetic_prices(n=600, seed=0, drift=0.0005, vol=0.01):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close, "Volume": 1e6},
        index=idx,
    )


# ---------- no look-ahead ----------

@pytest.mark.parametrize("strategy_fn", [sma_50_200, momentum_12_1])
def test_no_lookahead_truncation_invariance(strategy_fn):
    prices = make_synthetic_prices(600)
    full = strategy_fn(prices)
    cut = 400
    truncated = strategy_fn(prices.iloc[:cut])
    common = full.iloc[:cut].index
    pd.testing.assert_series_equal(
        full.loc[common], truncated.loc[common], check_names=False
    )


@pytest.mark.parametrize("strategy_fn", [sma_50_200, momentum_12_1])
def test_no_lookahead_future_perturbation(strategy_fn):
    prices = make_synthetic_prices(400)
    pos = strategy_fn(prices)
    perturbed = prices.copy()
    perturbed.iloc[-1, perturbed.columns.get_loc("Close")] *= 1.5
    pos_perturbed = strategy_fn(perturbed)
    pd.testing.assert_series_equal(
        pos.iloc[:-1], pos_perturbed.iloc[:-1], check_names=False
    )


def test_no_lookahead_position_uses_prior_close_only():
    """A monotone-up price series should produce sma_50_200 = 1 from bar 201 onward,
    where bar 200 is the first bar with both SMAs defined; due to .shift(1)
    the position turns on at bar 201."""
    n = 400
    idx = pd.bdate_range("2010-01-01", periods=n)
    close = pd.Series(np.linspace(100, 200, n), index=idx)
    prices = pd.DataFrame({"Close": close})
    pos = sma_50_200(prices)
    # SMA50 vs SMA200 only meaningful from bar 199 onward; shifted: bar 200+
    assert pos.iloc[200] == 1
    # all bars before SMAs are defined must be 0
    assert (pos.iloc[:200] == 0).all()


# ---------- costs applied correctly ----------

def test_costs_only_on_position_change():
    prices = make_synthetic_prices(50)
    pos = pd.Series(1, index=prices.index, dtype=int)
    no_cost = run_backtest(prices, pos, spread_bps=0, slippage_bps=0, commission_bps=0)
    with_cost = run_backtest(prices, pos, spread_bps=1, slippage_bps=5, commission_bps=0)
    diff = no_cost["returns"] - with_cost["returns"]
    expected_entry = (1 / 2 + 5) / 10000  # per-side 5.5 bps
    assert diff.iloc[0] == pytest.approx(expected_entry)
    assert (diff.iloc[1:].abs() < 1e-12).all()


def test_round_trip_cost_in_backtest_matches_formula():
    prices = make_synthetic_prices(20)
    pos = pd.Series(0, index=prices.index, dtype=int)
    pos.iloc[5:10] = 1  # enter at bar 5, exit at bar 10
    res = run_backtest(prices, pos, spread_bps=1, slippage_bps=5, commission_bps=0)
    total_bps = float(res["costs"].sum() * 10000)
    assert total_bps == pytest.approx(round_trip_cost_bps(1, 5, 0))


def test_gross_minus_cost_equals_net():
    prices = make_synthetic_prices(100)
    pos = sma_50_200(prices)
    res = run_backtest(prices, pos, spread_bps=1, slippage_bps=5)
    np.testing.assert_allclose(
        (res["gross_returns"] - res["costs"]).values, res["returns"].values, atol=1e-15
    )


def test_buy_and_hold_matches_close_pct_change_minus_entry_cost():
    prices = make_synthetic_prices(30)
    pos = buy_and_hold(prices)
    res = run_backtest(prices, pos, spread_bps=1, slippage_bps=5)
    expected_gross = prices["Close"].pct_change().fillna(0.0)
    np.testing.assert_allclose(res["gross_returns"].values, expected_gross.values, atol=1e-15)
    # Only first bar pays entry cost
    nonzero_costs = res["costs"][res["costs"] > 0]
    assert len(nonzero_costs) == 1
    assert res["costs"].iloc[0] == pytest.approx(5.5e-4)


def test_trade_log_matches_position_changes():
    prices = make_synthetic_prices(20)
    pos = pd.Series(0, index=prices.index, dtype=int)
    pos.iloc[3:7] = 1
    pos.iloc[10:13] = 1
    res = run_backtest(prices, pos)
    # entries at 3, 10; exits at 7, 13 -> 4 trades
    assert len(res["trades"]) == 4
    assert list(res["trades"]["delta"].values) == [1, -1, 1, -1]
