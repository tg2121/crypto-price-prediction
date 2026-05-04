"""Performance metrics for return/equity series.

All metrics assume daily-frequency returns. Sharpe/Sortino are annualized with
sqrt(252). CAGR uses calendar bars / 252.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _to_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def annualized_return(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _to_array(returns)
    if len(r) == 0:
        return 0.0
    return float(r.mean() * periods_per_year)


def annualized_vol(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _to_array(returns)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if len(equity) < 2:
        return 0.0
    eq = equity.dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0 or eq.iloc[-1] <= 0:
        return 0.0
    n_years = len(eq) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / n_years) - 1.0)


def sharpe(returns, periods_per_year: int = TRADING_DAYS, rf: float = 0.0) -> float:
    r = _to_array(returns)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    excess = r - rf / periods_per_year
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns, periods_per_year: int = TRADING_DAYS, rf: float = 0.0) -> float:
    r = _to_array(returns)
    if len(r) < 2:
        return 0.0
    excess = r - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    dsd = downside.std(ddof=1)
    if dsd == 0:
        return 0.0
    return float(excess.mean() / dsd * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    eq = equity.dropna()
    if len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def time_underwater(equity: pd.Series) -> float:
    """Fraction of bars spent strictly below the prior peak."""
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    return float((equity < peak).mean())


def hit_rate(returns) -> float:
    r = _to_array(returns)
    r = r[r != 0]
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def profit_factor(returns) -> float:
    r = _to_array(returns)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summary(returns: pd.Series, equity: pd.Series | None = None) -> dict:
    if equity is None:
        equity = (1.0 + pd.Series(returns).fillna(0)).cumprod()
    return {
        "cagr": cagr(equity),
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_vol(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(equity),
        "time_underwater": time_underwater(equity),
        "hit_rate": hit_rate(returns),
        "profit_factor": profit_factor(returns),
    }
