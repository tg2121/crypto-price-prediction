#!/usr/bin/env python3
"""
SOXL market-timing signal backtest.

Backtests a discretionary LONG/CASH signal history for SOXL against:
  (1) SOXL buy & hold
  (2) the signals with ZERO cost  (marketing-style)
  (3) the signals with REAL cost   (0.15%/switch) + a tax-drag variant
  (4) a vanilla 200-day SMA rule    (0.15%/switch)

EXECUTION / NO-LOOK-AHEAD CONVENTION
------------------------------------
A signal is dated on its DECISION day D (it uses information through D's close).
You cannot fill same-day, so the trade executes at the CLOSE of the next trading
day E (= first trading session strictly after D). The position is established at
E's close, therefore it only earns market returns starting the FOLLOWING session
(E+1). Implemented as:  pos_earning_today = state_as_of_yesterday_close.
The 200-day SMA rule uses the identical lag: signal computed from close[t],
filled at close[t+1], returns begin t+2 -- so both strategies are penalized the
same way and the comparison is fair.

DATA
----
Adjusted close (splits + dividends). Sources tried in order:
  0. a local CSV (path via argv[1] or $SOXL_CSV; needs Date + Adj Close/Close)
  1. yfinance  SOXL  auto_adjust=True  from 2010-03-01
  2. stooq     SOXL.US (direct CSV)            "
  3. stooq via pandas-datareader               "
The actual pulled date range is printed so you can confirm it is real.
"""

import io
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- params
START          = "2010-03-01"
RF             = 0.04        # risk-free for Sharpe
COST_PER_SWITCH= 0.0015      # 0.15% per LONG<->CASH transition (slippage+spread, 3x ETF)
TAX_RATE       = 0.35        # short-term cap-gains on each realized LONG-episode GAIN
SMA_WINDOW     = 200
ANN            = 252         # trading days / yr for vol & Sharpe annualization
OUTPNG         = "soxl_backtest.png"

# ----------------------------------------------------------------------------- signals
# (decision date, state effective at next session's close), chronological.
SIGNALS_TXT = """
2010-03-12 CASH, 2010-03-26 LONG, 2010-04-30 LONG, 2010-05-28 CASH, 2010-06-04 LONG,
2011-03-04 CASH, 2011-03-11 LONG, 2011-03-25 CASH, 2011-04-15 LONG, 2011-04-22 CASH,
2011-05-20 LONG, 2011-05-27 CASH, 2011-06-03 LONG, 2011-07-01 CASH, 2011-07-15 LONG,
2011-07-22 CASH, 2011-07-29 LONG, 2011-08-12 CASH, 2011-08-19 LONG, 2011-08-26 CASH,
2011-09-02 LONG, 2011-09-09 CASH, 2011-09-30 LONG, 2011-10-07 CASH, 2011-11-18 LONG,
2012-04-27 CASH, 2012-05-04 LONG, 2012-06-08 CASH, 2012-07-13 LONG, 2012-07-20 CASH,
2012-09-21 LONG, 2012-11-02 CASH, 2012-11-09 LONG, 2012-11-30 CASH, 2012-12-28 LONG,
2013-04-12 CASH, 2013-04-19 LONG, 2013-04-26 CASH, 2013-06-21 LONG, 2013-06-28 CASH,
2013-07-26 LONG, 2013-08-02 CASH, 2013-08-09 LONG, 2013-09-06 CASH, 2013-12-13 LONG,
2013-12-20 CASH, 2014-01-24 LONG, 2014-02-07 CASH, 2014-04-11 LONG, 2014-04-18 CASH,
2014-04-25 LONG, 2014-05-09 CASH, 2014-05-16 LONG, 2014-05-23 CASH, 2014-07-25 LONG,
2014-08-15 CASH, 2014-10-03 LONG, 2014-10-24 CASH, 2015-01-16 LONG, 2015-01-23 CASH,
2015-01-30 LONG, 2015-02-06 CASH, 2015-03-27 LONG, 2015-04-10 CASH, 2015-04-17 LONG,
2015-05-01 CASH, 2015-06-12 LONG, 2015-06-19 CASH, 2015-06-26 LONG, 2015-08-28 CASH,
2015-09-04 LONG, 2015-09-11 CASH, 2015-09-25 LONG, 2015-10-02 CASH, 2015-11-13 LONG,
2016-01-22 CASH, 2016-02-05 LONG, 2016-07-01 CASH, 2016-09-09 LONG, 2016-09-16 CASH,
2016-10-14 LONG, 2016-10-21 CASH, 2016-11-04 LONG, 2016-11-11 CASH, 2017-04-14 LONG,
2017-04-21 CASH, 2017-06-16 LONG, 2017-06-23 CASH, 2017-06-30 LONG, 2017-07-07 CASH,
2017-08-04 LONG, 2017-09-01 CASH, 2017-12-01 LONG, 2017-12-15 CASH, 2017-12-29 LONG,
2018-01-05 CASH, 2018-02-02 LONG, 2018-02-16 CASH, 2018-03-23 LONG, 2018-04-13 CASH,
2018-04-20 LONG, 2018-05-04 CASH, 2018-06-22 LONG, 2018-07-06 CASH, 2018-07-13 LONG,
2018-07-20 CASH, 2018-08-10 LONG, 2018-08-24 CASH, 2018-09-07 LONG, 2018-09-14 CASH,
2018-09-28 LONG, 2018-11-02 CASH, 2018-11-23 LONG, 2018-11-30 CASH, 2018-12-07 LONG,
2018-12-28 CASH, 2019-05-10 LONG, 2019-06-07 CASH, 2019-06-14 LONG, 2019-08-16 CASH,
2019-08-23 LONG, 2019-10-04 CASH, 2020-01-31 LONG, 2020-02-07 CASH, 2020-02-21 LONG,
2020-03-13 CASH, 2020-03-20 LONG, 2020-03-27 CASH, 2020-09-11 LONG, 2020-09-25 CASH,
2020-10-30 LONG, 2021-02-12 CASH, 2021-03-05 LONG, 2021-03-12 CASH, 2021-04-30 LONG,
2021-05-21 CASH, 2021-07-16 LONG, 2021-07-23 CASH, 2021-08-20 LONG, 2021-08-27 CASH,
2021-10-01 LONG, 2021-10-15 CASH, 2022-01-07 LONG, 2022-01-14 CASH, 2022-01-21 LONG,
2022-02-25 CASH, 2022-03-04 LONG, 2022-03-18 CASH, 2022-04-08 LONG, 2022-05-13 CASH,
2022-05-20 LONG, 2022-05-27 CASH, 2022-06-10 LONG, 2022-06-24 CASH, 2022-07-01 LONG,
2022-07-08 CASH, 2022-08-26 LONG, 2022-09-09 CASH, 2022-09-16 LONG, 2022-10-21 CASH,
2022-12-23 LONG, 2023-05-05 CASH, 2023-05-12 LONG, 2023-06-30 CASH, 2023-08-11 LONG,
2023-09-01 CASH, 2023-09-08 LONG, 2023-10-06 CASH, 2023-10-13 LONG, 2023-11-03 CASH,
2024-04-12 LONG, 2024-04-26 CASH, 2024-07-19 LONG, 2024-08-16 CASH, 2024-09-06 LONG,
2024-09-13 CASH, 2024-11-01 LONG, 2024-11-08 CASH, 2024-11-15 LONG, 2024-12-06 CASH,
2024-12-20 LONG, 2025-01-03 CASH, 2025-01-10 LONG, 2025-02-21 CASH, 2025-02-28 LONG,
2025-03-14 CASH, 2025-03-21 LONG, 2025-04-25 CASH, 2025-07-04 LONG, 2025-07-11 CASH,
2025-11-21 LONG, 2025-11-28 CASH, 2026-03-06 LONG, 2026-03-27 CASH, 2026-04-03 LONG,
2026-04-10 CASH
"""


def parse_signals(txt):
    out = []
    for tok in txt.replace("\n", " ").split(","):
        tok = tok.strip()
        if not tok:
            continue
        d, st = tok.split()
        st = st.upper()
        assert st in ("LONG", "CASH"), tok
        out.append((pd.Timestamp(d), st))
    out.sort(key=lambda x: x[0])
    return out


# ----------------------------------------------------------------------------- data
def _series_from_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    datecol = next(c for c in df.columns if c.lower() in ("date", "datetime"))
    lc = {c.lower(): c for c in df.columns}
    pcol = lc.get("adj close") or lc.get("adjclose") or lc.get("adj_close") or lc.get("close")
    if pcol is None:
        raise SystemExit(f"CSV has no Adj Close/Close column; found {list(df.columns)}")
    s = pd.Series(df[pcol].astype(float).values,
                  index=pd.to_datetime(df[datecol])).sort_index()
    return s[s.index >= pd.Timestamp(START)].dropna()


def fetch_prices():
    # 0) local CSV override -----------------------------------------------------
    csv = os.environ.get("SOXL_CSV") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if csv and os.path.exists(csv):
        return _series_from_csv(csv), f"local CSV: {csv}"

    # 1) yfinance ----------------------------------------------------------------
    try:
        import yfinance as yf
        df = yf.download("SOXL", start=START, auto_adjust=True, progress=False)
        if len(df):
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            s = close.dropna()
            s.index = pd.to_datetime(s.index)
            return s.sort_index(), "yfinance SOXL (auto_adjust=True)"
        print("yfinance returned 0 rows.", file=sys.stderr)
    except Exception as e:
        print(f"yfinance failed: {e}", file=sys.stderr)

    # 2) stooq direct CSV --------------------------------------------------------
    try:
        import urllib.request
        url = "https://stooq.com/q/d/l/?s=soxl.us&i=d"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=30).read().decode()
        df = pd.read_csv(io.StringIO(raw))
        df = df[df["Date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]
        s = pd.Series(df["Close"].astype(float).values,
                      index=pd.to_datetime(df["Date"])).sort_index()
        s = s[s.index >= pd.Timestamp(START)].dropna()
        if len(s):
            return s, "stooq SOXL.US (split/div-adjusted, direct CSV)"
    except Exception as e:
        print(f"stooq direct failed: {e}", file=sys.stderr)

    # 3) stooq via pandas-datareader --------------------------------------------
    try:
        from pandas_datareader import data as web
        df = web.DataReader("SOXL.US", "stooq")
        s = df["Close"].sort_index()
        s = s[s.index >= pd.Timestamp(START)].dropna()
        if len(s):
            return s, "stooq via pandas-datareader"
    except Exception as e:
        print(f"pandas-datareader stooq failed: {e}", file=sys.stderr)

    raise SystemExit(
        "ERROR: could not fetch SOXL prices from any source.\n"
        "       In a sandbox with a network allowlist, supply a CSV instead:\n"
        "       SOXL_CSV=/path/to/soxl.csv python3 soxl_backtest.py\n"
        "       (CSV needs a Date column and an 'Adj Close' or 'Close' column.)"
    )


# ----------------------------------------------------------------------------- engine
def build_signal_state(index, signals):
    """state as-of each trading day's CLOSE (the position established by then)."""
    state = pd.Series(np.nan, index=index, dtype=object)
    skipped = []
    for d, st in signals:
        loc = index.searchsorted(d, side="right")   # first session strictly after d
        if loc >= len(index):
            skipped.append((d, st))
            continue
        state.iloc[loc] = st                          # filled at that session's close
    state = state.ffill().fillna("CASH")              # flat before the first signal
    return state, skipped


def build_sma_state(price):
    sma = price.rolling(SMA_WINDOW).mean()
    sig = (price > sma).astype(float)
    sig[sma.isna()] = 0.0
    # decision at close[t-1] -> filled at close[t]; mirror the discretionary lag
    state = sig.shift(1).fillna(0.0).map({1.0: "LONG", 0.0: "CASH"})
    return state


def run_strategy(price, state, cost=0.0, tax=0.0):
    """Daily-compounded equity for a LONG/CASH state series.

    pos earning day t's return = state as of t-1's close (no look-ahead).
    A 'switch' (cost) is charged at the close where the state changes.
    Tax is levied on each realized LONG-episode positive gain at exit.
    """
    r = price.pct_change().fillna(0.0).values
    st = state.values
    st_prev = np.empty(len(st), dtype=object)
    st_prev[0] = "CASH"
    st_prev[1:] = st[:-1]

    eq = np.empty(len(price))
    e = 1.0
    entry_eq = None
    round_trips = 0
    switches = 0
    tax_paid = 0.0

    for i in range(len(price)):
        pos = 1.0 if st_prev[i] == "LONG" else 0.0
        e *= (1.0 + pos * r[i])                       # market move on yesterday's position
        if st[i] != st_prev[i]:                       # transacted at this close
            switches += 1
            if cost:
                e *= (1.0 - cost)
            if st[i] == "LONG":
                entry_eq = e                          # basis = post-entry-cost equity
            elif entry_eq is not None:                # closed a LONG episode
                gain = e - entry_eq
                if tax and gain > 0:
                    t = tax * gain
                    e -= t
                    tax_paid += t
                round_trips += 1
                entry_eq = None
        eq[i] = e

    return pd.Series(eq, index=price.index), dict(
        round_trips=round_trips, switches=switches, tax_paid=tax_paid)


# ----------------------------------------------------------------------------- metrics
def metrics(eq):
    r = eq.pct_change().iloc[1:]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0
    maxdd = (eq / eq.cummax() - 1.0).min()
    vol = r.std(ddof=1) * np.sqrt(ANN)
    sharpe = (r.mean() * ANN - RF) / vol if vol > 0 else float("nan")
    return dict(CAGR=cagr, MaxDD=maxdd, Vol=vol, Sharpe=sharpe, Final=eq.iloc[-1])


def annual_return(eq):
    return eq.groupby(eq.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1.0)


def invested_pct(state):
    """% of sessions actually holding SOXL (pos earns return = prior close state)."""
    pos = (state.shift(1).fillna("CASH") == "LONG").astype(float)
    return pos.groupby(pos.index.year).mean() * 100.0


def pct(x):
    return f"{x*100:6.1f}%"


def money(x):
    return f"${x:,.2f}"


# ----------------------------------------------------------------------------- main
def main():
    price, source = fetch_prices()
    if len(price) < SMA_WINDOW + 5:
        raise SystemExit(f"Not enough data ({len(price)} rows) from {source}.")

    signals = parse_signals(SIGNALS_TXT)

    print("=" * 78)
    print("DATA SOURCE :", source)
    print("DATE RANGE  :", price.index[0].date(), "->", price.index[-1].date(),
          f"({len(price)} trading days)")
    print("FIRST CLOSE :", money(price.iloc[0]), " LAST CLOSE:", money(price.iloc[-1]))
    print("SIGNALS     :", len(signals), "flips;",
          f"last actionable signal date {signals[-1][0].date()}")
    print("CONVENTION  : fill at NEXT session close; returns begin the session after.")
    print("COSTS       :", f"{COST_PER_SWITCH*100:.2f}% per switch; tax {TAX_RATE*100:.0f}% "
          "on realized LONG-episode gains (taxable variant).")
    print("=" * 78)

    sig_state, skipped = build_signal_state(price.index, signals)
    sma_state = build_sma_state(price)
    if skipped:
        print(f"NOTE: {len(skipped)} signal(s) dated after the data ends were ignored:",
              ", ".join(f"{d.date()} {s}" for d, s in skipped))

    # equity curves
    bh = (1.0 + price.pct_change().fillna(0.0)).cumprod()
    sig0, i0 = run_strategy(price, sig_state, cost=0.0, tax=0.0)
    sigc, ic = run_strategy(price, sig_state, cost=COST_PER_SWITCH, tax=0.0)
    sigt, it = run_strategy(price, sig_state, cost=COST_PER_SWITCH, tax=TAX_RATE)
    sma, ism = run_strategy(price, sma_state, cost=COST_PER_SWITCH, tax=0.0)

    curves = {
        "SOXL Buy&Hold":            (bh, dict(round_trips=0)),
        "Signals (0 cost)":         (sig0, i0),
        "Signals (real cost)":      (sigc, ic),
        "Signals (cost+35% tax)":   (sigt, it),
        "200d SMA (real cost)":     (sma, ism),
    }

    # ---- summary table
    print("\nSUMMARY")
    rows = []
    for name, (eq, info) in curves.items():
        m = metrics(eq)
        rows.append([name, pct(m["CAGR"]), pct(m["MaxDD"]), pct(m["Vol"]),
                     f"{m['Sharpe']:5.2f}", money(m["Final"]),
                     info.get("round_trips", 0)])
    summ = pd.DataFrame(rows, columns=["Strategy", "CAGR", "MaxDD", "AnnVol",
                                       "Sharpe", "Final$", "RoundTrips"])
    print(summ.to_string(index=False))

    # ---- per-year exposure vs SOXL annual return
    print("\nPER-YEAR  (SOXL B&H return | % time invested: Signals / SMA)")
    soxl_yr = annual_return(bh)
    sig_inv = invested_pct(sig_state)
    sma_inv = invested_pct(sma_state)
    yr = pd.DataFrame({
        "SOXL_BH_ret": (soxl_yr * 100).round(1),
        "Signals_inv%": sig_inv.round(1),
        "SMA_inv%": sma_inv.round(1),
        "Signals_ret": (annual_return(sigc) * 100).round(1),
        "SMA_ret": (annual_return(sma) * 100).round(1),
    })
    print(yr.to_string())

    # ---- crash / up-year isolation
    print("\nCRASH vs UP YEARS  (did signals dodge crashes / catch rallies?)")
    focus = [2013, 2017, 2018, 2022, 2023]
    tag = {2013: "UP", 2017: "UP", 2023: "UP", 2018: "CRASH", 2022: "CRASH"}
    fr = []
    for y in focus:
        if y in yr.index:
            row = yr.loc[y]
            fr.append([y, tag[y], f"{row['SOXL_BH_ret']:.1f}%",
                       f"{row['Signals_inv%']:.0f}%", f"{row['Signals_ret']:.1f}%",
                       f"{row['SMA_inv%']:.0f}%", f"{row['SMA_ret']:.1f}%"])
    print(pd.DataFrame(fr, columns=["Year", "Type", "SOXL", "Sig.inv",
                                    "Sig.ret", "SMA.inv", "SMA.ret"]).to_string(index=False))

    # ---- chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 7))
        styles = {"SOXL Buy&Hold": ("black", "-"),
                  "Signals (0 cost)": ("tab:green", "-"),
                  "Signals (real cost)": ("tab:blue", "-"),
                  "Signals (cost+35% tax)": ("tab:blue", "--"),
                  "200d SMA (real cost)": ("tab:orange", "-")}
        for name, (eq, _) in curves.items():
            c, ls = styles[name]
            ax.plot(eq.index, eq.values, label=name, color=c, linestyle=ls, lw=1.6)
        ax.set_yscale("log")
        ax.set_title(f"SOXL strategies, $1 -> growth (log scale)\n{source} | "
                     f"{price.index[0].date()} to {price.index[-1].date()}")
        ax.set_ylabel("Growth of $1 (log)")
        ax.legend(loc="upper left")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPNG, dpi=130)
        print(f"\nChart saved -> {OUTPNG}")
    except Exception as e:
        print(f"\nChart skipped: {e}", file=sys.stderr)

    # ---- verdict
    bh_f, s0_f, sc_f, st_f, sm_f = (bh.iloc[-1], sig0.iloc[-1], sigc.iloc[-1],
                                    sigt.iloc[-1], sma.iloc[-1])
    print("\nVERDICT (final value of $1)")
    print(f"  SOXL Buy & Hold .............. {money(bh_f)}")
    print(f"  Signals, zero cost .......... {money(s0_f)}")
    print(f"  Signals, real cost .......... {money(sc_f)}")
    print(f"  Signals, real cost + tax .... {money(st_f)}")
    print(f"  200-day SMA, real cost ...... {money(sm_f)}")
    def verb(a, b):
        return f"BEAT by {money(a-b)}" if a > b else f"LOST by {money(b-a)}"
    print("\n  After real cost, the Signals:")
    print(f"    vs Buy & Hold : {verb(sc_f, bh_f)}")
    print(f"    vs 200d SMA   : {verb(sc_f, sm_f)}")
    print("  After real cost + 35% tax, the Signals:")
    print(f"    vs Buy & Hold : {verb(st_f, bh_f)}")
    print(f"    vs 200d SMA   : {verb(st_f, sm_f)}")


if __name__ == "__main__":
    main()
