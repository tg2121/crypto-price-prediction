import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualized_return,
    annualized_vol,
    cagr,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
    summary,
    time_underwater,
)


def test_sharpe_zero_for_constant_zero_returns():
    r = pd.Series(np.zeros(252))
    assert sharpe(r) == 0.0


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.005, 252 * 10))
    assert sharpe(r) > 0


def test_sharpe_annualization_factor():
    """Constant daily return d, zero vol → undefined; perturb slightly to test scale."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 252 * 10))
    # mean ≈ 0, vol ≈ 0.01 daily → annualized vol ≈ 0.01 * sqrt(252)
    assert annualized_vol(r) == pytest.approx(0.01 * np.sqrt(252), rel=0.1)


def test_cagr_doubles_in_one_year():
    eq = pd.Series(np.linspace(1.0, 2.0, 252))
    assert cagr(eq) == pytest.approx(1.0, rel=0.05)


def test_cagr_handles_short_series():
    assert cagr(pd.Series([1.0])) == 0.0
    assert cagr(pd.Series(dtype=float)) == 0.0


def test_max_drawdown_known_path():
    eq = pd.Series([1.0, 1.5, 0.75, 1.0, 1.2])
    # peak after bar 1 is 1.5; trough 0.75 → -50%
    assert max_drawdown(eq) == pytest.approx(-0.5)


def test_max_drawdown_no_drawdown_is_zero():
    eq = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown(eq) == pytest.approx(0.0)


def test_time_underwater():
    eq = pd.Series([1.0, 1.5, 1.0, 1.2, 1.5, 1.5])
    # peak path:    [1, 1.5, 1.5, 1.5, 1.5, 1.5]
    # underwater:   [F, F,   T,   T,   F,   F  ] → 2/6
    assert time_underwater(eq) == pytest.approx(2 / 6)


def test_hit_rate_ignores_zeros():
    r = pd.Series([0.01, -0.01, 0.02, 0.0, -0.005])
    # nonzero: 4, positive: 2 → 0.5
    assert hit_rate(r) == pytest.approx(0.5)


def test_profit_factor_known():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    # gains 0.05, losses 0.03 → 5/3
    assert profit_factor(r) == pytest.approx(5 / 3)


def test_profit_factor_no_losses_is_inf():
    r = pd.Series([0.01, 0.02, 0.0])
    assert profit_factor(r) == float("inf")


def test_annualized_return_scales():
    r = pd.Series([0.001] * 252)
    assert annualized_return(r) == pytest.approx(0.001 * 252)


def test_sortino_positive_when_drift_positive():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert sortino(r) > 0


def test_summary_returns_expected_keys():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    s = summary(r)
    expected_keys = {
        "cagr", "ann_return", "ann_vol", "sharpe", "sortino",
        "max_drawdown", "time_underwater", "hit_rate", "profit_factor",
    }
    assert expected_keys == set(s.keys())
