"""Vectorized backtest engine.

Inputs:
    prices    -- DataFrame with at least a 'Close' column.
    positions -- Series in {-1, 0, 1} aligned to prices.index. Position at t MUST
                 be set from data <= t-1 close (strategies in strategies.py do
                 this via .shift(1)).

Outputs (dict):
    returns        -- daily NET returns (after costs)
    gross_returns  -- daily GROSS returns (before costs)
    costs          -- daily cost (fraction of NAV)
    equity         -- (1 + net).cumprod()
    positions      -- aligned positions (float)
    trades         -- DataFrame of position changes with prices and per-trade cost

Daily PnL convention:
    return_t = close_t / close_{t-1} - 1
    gross_t  = position_t * return_t
    net_t    = gross_t - cost_t,  cost_t = |position_t - position_{t-1}| * per_side
"""
from __future__ import annotations

import pandas as pd

from .costs import trade_costs


def run_backtest(
    prices: pd.DataFrame,
    positions: pd.Series,
    spread_bps: float = 1.0,
    slippage_bps: float = 5.0,
    commission_bps: float = 0.0,
) -> dict:
    close = prices["Close"]
    rets = close.pct_change().fillna(0.0)

    pos = positions.reindex(close.index).fillna(0).astype(float)

    gross = pos * rets
    cost = trade_costs(pos, spread_bps, slippage_bps, commission_bps)
    net = gross - cost
    equity = (1.0 + net).cumprod()

    delta = pos.diff()
    if len(pos) > 0:
        delta.iloc[0] = pos.iloc[0]
    mask = delta != 0
    trades = pd.DataFrame(
        {
            "date": pos.index[mask],
            "from_position": (pos.shift(1).fillna(0))[mask].values,
            "to_position": pos[mask].values,
            "delta": delta[mask].values,
            "price": close[mask].values,
            "cost": cost[mask].values,
        }
    ).reset_index(drop=True)

    return {
        "returns": net,
        "gross_returns": gross,
        "costs": cost,
        "equity": equity,
        "positions": pos,
        "trades": trades,
    }
