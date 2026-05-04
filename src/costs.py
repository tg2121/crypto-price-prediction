"""Transaction cost model.

Per side cost (in bps of notional traded):
    spread_bps / 2   -- pay half the bid/ask spread on each side
  + slippage_bps     -- price impact / non-zero adverse fill
  + commission_bps/2 -- split commission per side so round-trip totals commission

Round-trip cost identity (used as a regression test):
    round_trip_bps == 2 * (spread_bps/2 + slippage_bps) + commission_bps

`trade_costs` returns a per-bar cost as a fraction of NAV: |Δposition_t| * per_side.
"""
from __future__ import annotations

import pandas as pd


def per_side_cost_bps(
    spread_bps: float,
    slippage_bps: float,
    commission_bps: float = 0.0,
) -> float:
    return spread_bps / 2.0 + slippage_bps + commission_bps / 2.0


def round_trip_cost_bps(
    spread_bps: float,
    slippage_bps: float,
    commission_bps: float = 0.0,
) -> float:
    return 2.0 * (spread_bps / 2.0 + slippage_bps) + commission_bps


def trade_costs(
    positions: pd.Series,
    spread_bps: float,
    slippage_bps: float,
    commission_bps: float = 0.0,
) -> pd.Series:
    """Per-bar transaction cost as a fraction of NAV.

    Treats the first bar as an entry from flat (delta_0 = position_0).
    Going 0 → 1 → 0 over three bars sums to round_trip_cost_bps/10000.
    """
    if len(positions) == 0:
        return positions.astype(float)
    delta = positions.astype(float).diff()
    delta.iloc[0] = positions.iloc[0]
    per_side = per_side_cost_bps(spread_bps, slippage_bps, commission_bps) / 10000.0
    return delta.abs() * per_side
