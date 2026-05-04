"""Baseline strategies as functions: prices -> position series in {-1, 0, 1}.

Convention enforced here: position[t] is the position HELD during day t and is
computable from data available at close of day t-1. PnL on day t equals
position[t] * (close[t]/close[t-1] - 1) - cost[t]. This is enforced by
.shift(1) on every signal so the no-look-ahead test passes.
"""
from __future__ import annotations

import pandas as pd


def momentum_12_1(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Time-series 12-1 momentum.

    Long when the return over (t-1-lookback, t-1-skip] is positive, else flat.
    Default lookback=252 trading days (~12mo), skip=21 (~1mo) drops the
    short-term reversal window.
    """
    close = prices["Close"]
    momentum = close.shift(skip) / close.shift(lookback) - 1.0
    raw_signal = (momentum > 0).astype(int)
    position = raw_signal.shift(1).fillna(0).astype(int)
    return position.rename("position")


def sma_50_200(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """Long when fast SMA > slow SMA (using prior close), else flat."""
    close = prices["Close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    raw_signal = (fast_ma > slow_ma).astype(int)
    position = raw_signal.shift(1).fillna(0).astype(int)
    return position.rename("position")


def buy_and_hold(prices: pd.DataFrame) -> pd.Series:
    return pd.Series(1, index=prices.index, name="position", dtype=int)


STRATEGIES = {
    "momentum_12_1": momentum_12_1,
    "sma_50_200": sma_50_200,
    "buy_and_hold": buy_and_hold,
}
