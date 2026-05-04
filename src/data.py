"""yfinance loader with parquet caching.

Returns clean OHLCV dataframes indexed by trading date. Cached pulls live in data/.
auto_adjust=True so Close is split- and dividend-adjusted (correct for total-return
backtests of buy-and-hold ETFs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[[c for c in OHLCV if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df.sort_index()
    df = df.dropna(subset=["Close"])
    return df


def load(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load OHLCV for `ticker`. Uses parquet cache unless refresh=True."""
    DATA_DIR.mkdir(exist_ok=True)
    p = cache_path(ticker)

    if p.exists() and not refresh:
        df = pd.read_parquet(p)
    else:
        import yfinance as yf
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        df = _normalize(raw)
        df.to_parquet(p)

    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def load_universe(
    tickers: list[str],
    start: str,
    end: Optional[str] = None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    return {t: load(t, start, end, refresh) for t in tickers}


def common_index(panels: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Intersection of date indices across all assets."""
    idx: Optional[pd.DatetimeIndex] = None
    for df in panels.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    return idx if idx is not None else pd.DatetimeIndex([])
