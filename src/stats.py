"""Statistical inference for backtests.

Implements:
  - Probabilistic Sharpe Ratio (Bailey & López de Prado 2012)
  - Deflated Sharpe Ratio (Bailey & López de Prado 2014)
        DSR = PSR(SR*) where SR* is the expected max Sharpe over N trials with
        cross-trial variance V[SR]:
            SR* = sqrt(V[SR]) * ((1-γ) Φ⁻¹(1-1/N) + γ Φ⁻¹(1-1/(Ne)))
        γ ≈ 0.5772 (Euler-Mascheroni). All SRs are PER-PERIOD (daily) inside
        the formula; we annualize for display.
  - Bootstrap confidence intervals (iid resampling).
  - Bonferroni adjustment helpers.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import stats as sst

EULER = 0.5772156649015329


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[~np.isnan(r)]


def sharpe_skew_kurt(returns) -> tuple[float, float, float, int]:
    """Per-period Sharpe (mean/std), skewness, non-excess kurtosis, n."""
    r = _clean(returns)
    n = len(r)
    if n < 3:
        return 0.0, 0.0, 3.0, n
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0, 0.0, 3.0, n
    sr = float(r.mean() / sd)
    skew = float(sst.skew(r, bias=False))
    kurt = float(sst.kurtosis(r, fisher=False, bias=False))
    return sr, skew, kurt, n


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """P(true SR > sr_benchmark) given observed sample. sr_benchmark in per-period units."""
    sr, skew, kurt, n = sharpe_skew_kurt(returns)
    if n < 3:
        return 0.5
    denom = np.sqrt(max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(sst.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum Sharpe across N trials under SR=0 null with cross-trial variance var_sr."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    a = sst.norm.ppf(1.0 - 1.0 / n_trials)
    b = sst.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1.0 - EULER) * a + EULER * b))


def deflated_sharpe(returns, trial_sharpes_per_period) -> dict:
    """Deflated Sharpe diagnostics for one strategy in a multi-trial study.

    Returns dict with both the haircut (observed_sr - sr_star) and the PSR(SR*)
    probability. The decision rule "deflated Sharpe > 0" maps to dsr_haircut > 0
    (equivalently psr_deflated > 0.5).
    """
    sr_obs, skew, kurt, n = sharpe_skew_kurt(returns)
    sr_all = np.asarray(trial_sharpes_per_period, dtype=float)
    sr_all = sr_all[~np.isnan(sr_all)]
    n_trials = len(sr_all)
    var_sr = float(sr_all.var(ddof=1)) if n_trials > 1 else 0.0
    sr_star = expected_max_sharpe(n_trials, var_sr)

    if n < 3:
        return {
            "observed_sr_period": sr_obs,
            "sr_star_period": sr_star,
            "dsr_haircut_period": sr_obs - sr_star,
            "psr_deflated": 0.5,
            "z": 0.0,
            "n_obs": n,
            "n_trials": n_trials,
            "var_sr_across_trials": var_sr,
            "skew": skew,
            "kurt": kurt,
        }

    denom = np.sqrt(max(1.0 - skew * sr_obs + ((kurt - 1.0) / 4.0) * sr_obs ** 2, 1e-12))
    z = (sr_obs - sr_star) * np.sqrt(n - 1) / denom
    psr = float(sst.norm.cdf(z))
    return {
        "observed_sr_period": sr_obs,
        "sr_star_period": sr_star,
        "dsr_haircut_period": sr_obs - sr_star,
        "psr_deflated": psr,
        "z": float(z),
        "n_obs": int(n),
        "n_trials": int(n_trials),
        "var_sr_across_trials": var_sr,
        "skew": skew,
        "kurt": kurt,
    }


def bootstrap_ci(
    data,
    statistic: Callable[[np.ndarray], float],
    n_iter: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float, np.ndarray]:
    """IID bootstrap CI. data may be 1D or 2D (rows = obs); statistic gets resampled rows.

    Returns (point_estimate, lo, hi, samples).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[~np.isnan(arr)]
    n = arr.shape[0]
    if n < 2:
        point = float(statistic(arr)) if n > 0 else 0.0
        return point, point, point, np.array([])
    samples = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        samples[i] = statistic(arr[idx])
    lo_q = (1.0 - confidence) / 2.0
    hi_q = 1.0 - lo_q
    lo = float(np.quantile(samples, lo_q))
    hi = float(np.quantile(samples, hi_q))
    point = float(statistic(arr))
    return point, lo, hi, samples


def bonferroni_alpha(alpha: float, n_trials: int) -> float:
    return float(alpha / max(n_trials, 1))


def bonferroni_p(p_value: float, n_trials: int) -> float:
    return float(min(1.0, p_value * max(n_trials, 1)))
