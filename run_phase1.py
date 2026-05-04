"""Phase 1 entry point.

Loads the YAML config, fetches OHLCV via yfinance (cached), runs each strategy
on each asset using an expanding-window walk-forward, computes IS/OOS metrics,
bootstrap CIs for excess return over SPY and for Sharpe, deflated Sharpe across
all (asset × strategy) trials, and prints a verdict per pair.

Decision rule (per the spec):
    BEATS SPY  ⇔  OOS deflated-Sharpe haircut > 0
                  AND  bootstrap 95% CI for OOS annualized excess return excludes 0

Outputs (results/):
    summary.csv / summary.parquet  -- one row per (asset, strategy)
    equity_curves.parquet          -- daily equity curve per run
    trades.parquet                 -- combined trade log
    meta.json                      -- config, splits, IS/OOS windows
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import strategies as strat_mod
from src.backtest import run_backtest
from src.data import common_index, load_universe
from src.metrics import cagr, max_drawdown
from src.stats import (
    bonferroni_alpha,
    bootstrap_ci,
    deflated_sharpe,
    sharpe_skew_kurt,
)
from src.walkforward import expanding_yearly_splits, slice_returns

RESULTS_DIR = ROOT / "results"
TRADING_DAYS = 252


def annualized_sharpe(rets) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def annualized_excess_mean(paired: np.ndarray) -> float:
    return float((paired[:, 0] - paired[:, 1]).mean() * TRADING_DAYS)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str) -> int:
    cfg = load_config(Path(config_path))
    universe = cfg["universe"]
    benchmark = cfg["benchmark"]
    start = cfg["start_date"]
    end = cfg.get("end_date") or None

    spread_bps = float(cfg["costs"]["spread_bps"])
    slippage_bps = float(cfg["costs"]["slippage_bps"])
    commission_bps = float(cfg["costs"].get("commission_bps", 0.0))

    train_years = int(cfg["walkforward"]["train_years"])
    test_years = int(cfg["walkforward"]["test_years"])

    n_boot = int(cfg["bootstrap"]["n_iter"])
    confidence = float(cfg["bootstrap"]["confidence"])
    alpha = 1.0 - confidence

    print(f"Loading {universe} from {start} to {end or 'today'}...")
    panels = load_universe(universe, start, end)
    idx = common_index(panels)
    if len(idx) == 0:
        print("ERROR: empty common index across universe.")
        return 1
    for t in panels:
        panels[t] = panels[t].loc[idx]

    spy_close = panels[benchmark]["Close"]
    spy_rets = spy_close.pct_change().fillna(0.0)

    splits = expanding_yearly_splits(idx, train_years=train_years, test_years=test_years)
    if not splits:
        print("ERROR: not enough data for walk-forward splits.")
        return 1
    is_end = splits[0].train_end
    oos_start = splits[0].test_start
    oos_end = splits[-1].test_end

    strategy_specs = cfg["strategies"]

    # ---------- pass 1: backtests ----------
    runs: dict[tuple[str, str], dict] = {}
    for asset in universe:
        prices = panels[asset]
        for sname, sparams in strategy_specs.items():
            fn = strat_mod.STRATEGIES[sname]
            positions = fn(prices, **(sparams or {}))
            res = run_backtest(prices, positions, spread_bps, slippage_bps, commission_bps)
            runs[(asset, sname)] = res

    # ---------- DSR cross-trial Sharpe variance (per-period, OOS) ----------
    per_period_sharpes = []
    for (asset, sname), res in runs.items():
        oos_rets = slice_returns(res["returns"], oos_start, oos_end)
        sr_d, _, _, _ = sharpe_skew_kurt(oos_rets.values)
        per_period_sharpes.append(sr_d)
    per_period_sharpes_arr = np.array(per_period_sharpes)
    n_trials = len(per_period_sharpes_arr)
    bonf_alpha = bonferroni_alpha(alpha, n_trials)

    # ---------- pass 2: per-trial summary ----------
    rng_master = np.random.default_rng(20260504)
    rows = []
    equity_curves: dict[str, pd.Series] = {}
    trade_logs: dict[str, pd.DataFrame] = {}

    for asset in universe:
        for sname in strategy_specs.keys():
            res = runs[(asset, sname)]
            net = res["returns"]

            is_rets = slice_returns(net, idx[0], is_end)
            oos_rets = slice_returns(net, oos_start, oos_end)
            spy_oos = slice_returns(spy_rets, oos_start, oos_end).reindex(oos_rets.index).fillna(0.0)

            is_sr_ann = annualized_sharpe(is_rets.values)
            oos_sr_ann = annualized_sharpe(oos_rets.values)

            oos_eq = (1.0 + oos_rets).cumprod()
            oos_cagr_v = cagr(oos_eq)
            oos_mdd = max_drawdown(oos_eq)

            dsr = deflated_sharpe(oos_rets.values, per_period_sharpes_arr)
            dsr_ann_haircut = dsr["dsr_haircut_period"] * np.sqrt(TRADING_DAYS)
            sr_star_ann = dsr["sr_star_period"] * np.sqrt(TRADING_DAYS)

            paired = np.column_stack([oos_rets.values, spy_oos.values])
            rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            point_excess, lo_excess, hi_excess, _ = bootstrap_ci(
                paired, statistic=annualized_excess_mean, n_iter=n_boot, confidence=confidence, rng=rng,
            )

            rng2 = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            sr_point, sr_lo, sr_hi, _ = bootstrap_ci(
                oos_rets.values, statistic=annualized_sharpe, n_iter=n_boot, confidence=confidence, rng=rng2,
            )

            beats = bool((dsr["dsr_haircut_period"] > 0) and (lo_excess > 0))
            verdict = "BEATS SPY" if beats else "DOES NOT BEAT SPY"

            n_trades = int((res["trades"]["delta"] != 0).sum())

            rows.append({
                "asset": asset,
                "strategy": sname,
                "n_trades": n_trades,
                "is_sharpe_ann": is_sr_ann,
                "oos_sharpe_ann": oos_sr_ann,
                "oos_sharpe_lo": sr_lo,
                "oos_sharpe_hi": sr_hi,
                "sr_star_ann": sr_star_ann,
                "deflated_sharpe_ann": dsr_ann_haircut,
                "psr_deflated": dsr["psr_deflated"],
                "oos_cagr": oos_cagr_v,
                "oos_max_dd": oos_mdd,
                "excess_ann_vs_spy": point_excess,
                "excess_ci_lo": lo_excess,
                "excess_ci_hi": hi_excess,
                "verdict": verdict,
            })

            run_key = f"{asset}__{sname}"
            equity_curves[run_key] = res["equity"]
            trade_logs[run_key] = res["trades"]

    # ---------- benchmark row: SPY buy-and-hold over the same OOS window ----------
    spy_oos_full = slice_returns(spy_rets, oos_start, oos_end)
    spy_is = slice_returns(spy_rets, idx[0], is_end)
    spy_eq_oos = (1.0 + spy_oos_full).cumprod()
    rows.append({
        "asset": benchmark,
        "strategy": "buy_and_hold (benchmark)",
        "n_trades": 1,
        "is_sharpe_ann": annualized_sharpe(spy_is.values),
        "oos_sharpe_ann": annualized_sharpe(spy_oos_full.values),
        "oos_sharpe_lo": np.nan,
        "oos_sharpe_hi": np.nan,
        "sr_star_ann": np.nan,
        "deflated_sharpe_ann": np.nan,
        "psr_deflated": np.nan,
        "oos_cagr": cagr(spy_eq_oos),
        "oos_max_dd": max_drawdown(spy_eq_oos),
        "excess_ann_vs_spy": 0.0,
        "excess_ci_lo": 0.0,
        "excess_ci_hi": 0.0,
        "verdict": "(benchmark)",
    })
    equity_curves["SPY__buy_and_hold_benchmark"] = (1.0 + spy_rets).cumprod()

    summary_df = pd.DataFrame(rows)

    # ---------- save ----------
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_df.to_parquet(RESULTS_DIR / "summary.parquet")
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    eq_df = pd.DataFrame(equity_curves)
    eq_df.to_parquet(RESULTS_DIR / "equity_curves.parquet")

    if trade_logs:
        all_trades = []
        for run_key, tdf in trade_logs.items():
            t2 = tdf.copy()
            t2["run"] = run_key
            all_trades.append(t2)
        pd.concat(all_trades, ignore_index=True).to_parquet(RESULTS_DIR / "trades.parquet")

    meta = {
        "config": cfg,
        "n_trials_for_dsr": int(n_trials),
        "bonferroni_alpha": bonf_alpha,
        "confidence": confidence,
        "is_period": [str(idx[0].date()), str(is_end.date())],
        "oos_period": [str(oos_start.date()), str(oos_end.date())],
        "splits": [
            {
                "label": s.label,
                "train": [str(s.train_start.date()), str(s.train_end.date())],
                "test": [str(s.test_start.date()), str(s.test_end.date())],
            }
            for s in splits
        ],
    }
    with open(RESULTS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # ---------- print ----------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print()
    print("=" * 110)
    print("PHASE 1 RESULTS")
    print("=" * 110)
    print(f"Universe:    {universe}")
    print(f"Benchmark:   {benchmark}")
    print(f"IS period:   {idx[0].date()} → {is_end.date()}")
    print(f"OOS period:  {oos_start.date()} → {oos_end.date()}")
    print(
        f"Walk-forward splits: {len(splits)} | "
        f"trials for DSR: {n_trials} | "
        f"Bonferroni α: {bonf_alpha:.4f}"
    )
    print()

    cols = [
        "asset", "strategy", "is_sharpe_ann", "oos_sharpe_ann",
        "deflated_sharpe_ann", "oos_cagr", "oos_max_dd",
        "excess_ann_vs_spy", "excess_ci_lo", "excess_ci_hi", "verdict",
    ]
    print(summary_df[cols].to_string(index=False))
    print()

    print('Decision: BEATS SPY  ⇔  deflated_sharpe_ann > 0  AND  excess CI excludes 0')
    print("-" * 110)
    for r in rows:
        if r["strategy"].endswith("(benchmark)"):
            continue
        print(
            f"  {r['asset']:<5} {r['strategy']:<18} → {r['verdict']:<18} "
            f"DSR_ann={r['deflated_sharpe_ann']:+.3f}  "
            f"excess_ann={r['excess_ann_vs_spy']:+.4f}  "
            f"CI=[{r['excess_ci_lo']:+.4f}, {r['excess_ci_hi']:+.4f}]  "
            f"PSR_deflated={r['psr_deflated']:.3f}"
        )
    print()
    print(f"Saved: {RESULTS_DIR}/summary.csv, summary.parquet, equity_curves.parquet, trades.parquet, meta.json")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()
    sys.exit(main(args.config))
