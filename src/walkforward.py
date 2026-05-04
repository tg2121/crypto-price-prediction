"""Expanding-window walk-forward splits.

Train period grows; test period rolls forward in fixed-year increments. Strategies
in this codebase have no fitted parameters, so the train portion serves only as
indicator warmup and as the IS reporting window — the OOS period (concatenation
of all test slices) is the one used for the decision rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp     # exclusive: returns < train_end are IS
    test_start: pd.Timestamp    # inclusive: returns >= test_start are OOS
    test_end: pd.Timestamp      # exclusive
    label: str


def expanding_yearly_splits(
    index: pd.DatetimeIndex,
    train_years: int = 4,
    test_years: int = 1,
) -> list[Split]:
    """Build expanding-window splits over `index`.

    First split: train = [index[0], index[0] + train_years), test = [+train_years, +train_years+test_years).
    Subsequent splits expand training and roll the test window forward by `test_years`.
    """
    if len(index) == 0:
        return []
    start = index[0]
    end = index[-1]
    last = end + pd.Timedelta(days=1)

    splits: list[Split] = []
    train_end = start + pd.DateOffset(years=train_years)
    while train_end < last:
        test_end = min(train_end + pd.DateOffset(years=test_years), last)
        if test_end <= train_end:
            break
        splits.append(
            Split(
                train_start=start,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
                label=f"train→{train_end.year - 1}_test_{train_end.year}",
            )
        )
        train_end = test_end
    return splits


def slice_returns(returns: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Half-open slice [start, end) on a date-indexed series."""
    mask = (returns.index >= start) & (returns.index < end)
    return returns[mask]


def concat_oos(splits: list[Split], returns: pd.Series) -> pd.Series:
    pieces = [slice_returns(returns, s.test_start, s.test_end) for s in splits]
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()
