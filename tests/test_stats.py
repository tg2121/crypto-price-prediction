"""Tests for stats: PSR, DSR, bootstrap CI, Bonferroni."""
import numpy as np
import pytest

from src.stats import (
    bonferroni_alpha,
    bonferroni_p,
    bootstrap_ci,
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


def test_psr_equals_half_at_observed_sr():
    """PSR(SR_benchmark = observed SR) = 0.5 by construction."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 1000)
    sr_obs = float(r.mean() / r.std(ddof=1))
    assert probabilistic_sharpe_ratio(r, sr_benchmark=sr_obs) == pytest.approx(0.5, abs=1e-9)


def test_psr_low_when_observed_sr_below_benchmark():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, 2000)
    p = probabilistic_sharpe_ratio(r, sr_benchmark=0.5)  # benchmark much higher than realized
    assert p < 0.05


def test_psr_high_when_strong_signal():
    rng = np.random.default_rng(0)
    r = rng.normal(0.002, 0.01, 2000)
    p = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    assert p > 0.95


def test_expected_max_sharpe_grows_with_n_trials():
    a = expected_max_sharpe(2, 1.0)
    b = expected_max_sharpe(20, 1.0)
    assert b > a > 0


def test_dsr_haircut_negative_under_pure_noise():
    """Random returns: SR_obs ~ 0, expected_max under N=10 trials > 0 → haircut < 0."""
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 1000)
    trial_srs = rng.normal(0, 0.05, 10)  # plausible cross-trial spread
    out = deflated_sharpe(r, trial_srs)
    assert out["sr_star_period"] >= 0
    assert out["psr_deflated"] <= 0.6


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1000)
    point, lo, hi, _ = bootstrap_ci(r, statistic=lambda x: float(x.mean()), n_iter=500)
    assert lo <= point <= hi


def test_bootstrap_ci_paired_2d():
    rng = np.random.default_rng(0)
    a = rng.normal(0.002, 0.01, 500)
    b = rng.normal(0.001, 0.01, 500)
    paired = np.column_stack([a, b])
    point, lo, hi, _ = bootstrap_ci(
        paired, statistic=lambda x: float((x[:, 0] - x[:, 1]).mean()), n_iter=500
    )
    assert lo <= point <= hi


def test_bonferroni_alpha():
    assert bonferroni_alpha(0.05, 10) == pytest.approx(0.005)


def test_bonferroni_p_clipped():
    assert bonferroni_p(0.5, 10) == 1.0
    assert bonferroni_p(0.01, 5) == pytest.approx(0.05)
