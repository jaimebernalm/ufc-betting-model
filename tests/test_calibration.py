"""Unit tests for calibration methods."""

from __future__ import annotations

import numpy as np

from ufc_pred.calibration.methods import (
    IsotonicCalibration,
    PlattScaling,
    TemperatureScaling,
    build,
)


def _make_overconfident_data(n: int = 1000, seed: int = 0):
    """Overconfident model: claims p but reality is squeezed toward 0.5.

    Generates p_raw uniformly in [0.05, 0.95], then samples y with
    p_true = 0.5 + 0.5 * (p_raw - 0.5), i.e. real-world prob is half the
    distance from 0.5 that the model claims. A perfect calibrator would
    learn this exact shrinkage.
    """
    rng = np.random.default_rng(seed)
    p_raw = rng.uniform(0.05, 0.95, size=n)
    p_true = 0.5 + 0.5 * (p_raw - 0.5)
    y = (rng.uniform(size=n) < p_true).astype(int)
    return p_raw, y


def test_temperature_identity_when_perfect():
    # If raw == true probability, T should converge near 1.
    rng = np.random.default_rng(0)
    p_raw = rng.uniform(0.05, 0.95, size=2000)
    y = (rng.uniform(size=2000) < p_raw).astype(int)
    cal = TemperatureScaling().fit(p_raw, y)
    assert 0.7 < cal.T < 1.4, f"expected T ≈ 1, got {cal.T}"


def test_temperature_squeezes_overconfident():
    p_raw, y = _make_overconfident_data(n=2000)
    cal = TemperatureScaling().fit(p_raw, y)
    # Overconfident → T > 1 (squeezes toward 0.5)
    assert cal.T > 1.2, f"expected T > 1.2 on overconfident data, got {cal.T}"
    # Calibrated probs should be closer to 0.5 than raw on extremes
    p_cal = cal.transform(p_raw)
    extremes = (p_raw > 0.8) | (p_raw < 0.2)
    raw_dist = np.abs(p_raw[extremes] - 0.5)
    cal_dist = np.abs(p_cal[extremes] - 0.5)
    assert cal_dist.mean() < raw_dist.mean(), "calibration should shrink extreme probs toward 0.5"


def test_temperature_preserves_ordering():
    # Calibrated probabilities should be monotonic in raw (T > 0 preserves order).
    p_raw, y = _make_overconfident_data(n=500)
    cal = TemperatureScaling().fit(p_raw, y)
    p_cal = cal.transform(p_raw)
    order = np.argsort(p_raw)
    assert np.all(np.diff(p_cal[order]) >= -1e-9), "calibration must preserve order"


def test_platt_two_params():
    p_raw, y = _make_overconfident_data(n=2000)
    cal = PlattScaling().fit(p_raw, y)
    # On overconfident data, a should be < 1 (squeezing).
    assert cal.a < 1.0, f"expected a < 1 on overconfident data, got {cal.a}"


def test_isotonic_monotonic():
    p_raw, y = _make_overconfident_data(n=2000)
    cal = IsotonicCalibration().fit(p_raw, y)
    grid = np.linspace(0.05, 0.95, 50)
    p_cal = cal.transform(grid)
    assert np.all(np.diff(p_cal) >= -1e-9), "isotonic output must be monotonic"


def test_build_factory():
    for name in ("temperature", "platt", "isotonic"):
        cal = build(name)
        assert hasattr(cal, "fit") and hasattr(cal, "transform")


def test_all_methods_improve_log_loss_on_overconfident_data():
    """Sanity: on data designed to be overconfident, all 3 methods should
    improve held-out log loss vs the raw predictions."""
    from sklearn.metrics import log_loss as sk_log_loss

    p_fit, y_fit = _make_overconfident_data(n=2000, seed=0)
    p_test, y_test = _make_overconfident_data(n=2000, seed=1)

    raw_ll = sk_log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6))
    for name in ("temperature", "platt", "isotonic"):
        cal = build(name).fit(p_fit, y_fit)
        p_cal = cal.transform(p_test)
        cal_ll = sk_log_loss(y_test, np.clip(p_cal, 1e-6, 1 - 1e-6))
        assert cal_ll < raw_ll, f"{name} should improve log loss; raw={raw_ll:.4f}  cal={cal_ll:.4f}"


def test_outputs_stay_in_unit_interval():
    p_raw, y = _make_overconfident_data(n=500)
    for name in ("temperature", "platt", "isotonic"):
        cal = build(name).fit(p_raw, y)
        # Test on extreme values that aren't in training
        edge = np.array([1e-5, 0.001, 0.5, 0.999, 1 - 1e-5])
        out = cal.transform(edge)
        assert np.all(out > 0) and np.all(out < 1), f"{name}: outputs outside (0,1): {out}"
