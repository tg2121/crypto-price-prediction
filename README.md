# Phase 1 — Pattern recognition on price/volume vs SPY buy-and-hold

A rigorous test of whether two simple price/volume strategies beat SPY buy-and-hold,
after costs, on liquid US ETFs, with strict walk-forward out-of-sample reporting and
multiple-testing correction.

## Goal

Test whether pattern recognition on price/volume data alone (no fundamentals) can
produce risk-adjusted returns meaningfully above buy-and-hold SPY. The realistic
ceiling for a retail-built daily-bars system is 5–12% annualized after costs. Most
apparent edges are overfitting. The goal here is **truth**, not validation.

## Phase 1 scope

- **Universe**: SPY, QQQ, IWM, TLT, GLD
- **Timeframe**: daily bars, 2010-01-01 → today (yfinance, auto-adjusted)
- **Two baseline strategies only**:
  - `momentum_12_1` — long if return over (t-252, t-21] is positive, else flat
  - `sma_50_200` — long if SMA50 > SMA200, else flat
- **Costs**: 1 bp spread, 5 bp slippage per side, $0 commission, applied on every
  position change
- **Walk-forward**: expanding window, 4-year train / 1-year test, rolled forward
- **Benchmark**: SPY buy-and-hold

## Methodology — non-negotiable

1. **No look-ahead bias.** Position at time `t` is computable from data at `t-1`
   close. Tested in `tests/test_backtest.py` two ways: truncation invariance, and
   future-perturbation invariance.
2. **Costs on every position change.** Round-trip cost identity
   `2 × (spread/2 + slippage) + commission` is regression-tested in
   `tests/test_costs.py`.
3. **Walk-forward only.** No fitting on the full sample and reporting in-sample.
4. **Distributions, not point estimates.** Bootstrap (1000x, iid) CIs for
   annualized excess return over SPY and for OOS Sharpe.
5. **Deflated Sharpe ratio** (Bailey & López de Prado 2014) computed across all
   trials (5 assets × 2 strategies = 10). Bonferroni correction reported.
6. **Decision rule for "this works"**: out-of-sample **deflated Sharpe haircut
   > 0** AND bootstrap **95% CI for annualized excess return over SPY excludes 0**.

## Repo layout

```
src/
  data.py          # yfinance loader, parquet cache in data/
  costs.py         # transaction cost model
  strategies.py    # baseline strategies → position series in {-1, 0, 1}
  backtest.py      # vectorized engine: prices + positions → equity, returns, trades
  metrics.py       # CAGR, Sharpe, Sortino, max drawdown, time underwater, hit rate, profit factor
  stats.py         # PSR, deflated Sharpe, bootstrap CIs, Bonferroni
  walkforward.py   # expanding-window splitter
configs/
  phase1.yaml      # universe, dates, strategy params, costs, walk-forward, bootstrap
notebooks/
  phase1_results.ipynb   # equity curves, drawdowns, summary table
results/           # parquet + json outputs
tests/             # pytest: costs, look-ahead, metrics, stats
data/              # cached yfinance pulls
run_phase1.py      # top-level entry point
requirements.txt
```

## Run

```bash
pip install -r requirements.txt
pytest -q
python run_phase1.py --config configs/phase1.yaml
jupyter notebook notebooks/phase1_results.ipynb
```

## Outputs

`run_phase1.py` writes to `results/`:
- `summary.csv` / `summary.parquet` — one row per `(asset, strategy)` with IS Sharpe,
  OOS Sharpe (with bootstrap CI), deflated Sharpe (annualized haircut + PSR), CAGR,
  max drawdown, annualized excess return vs SPY (with bootstrap 95% CI), and the
  verdict.
- `equity_curves.parquet` — daily equity per run plus SPY benchmark
- `trades.parquet` — combined trade log
- `meta.json` — config snapshot, walk-forward split definitions, IS/OOS windows

## Caveats

The decision rule is conservative on purpose. With 10 trials, even by luck one
or two will show in-sample edges, so the deflated Sharpe haircuts the observed
Sharpe by the expected maximum under the null and the bootstrap CI on excess
return must exclude zero. If a strategy is reported as **DOES NOT BEAT SPY**,
believe it.
