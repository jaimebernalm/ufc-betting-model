"""Model functions for the walk-forward harness.

stats_*  : the classic "predict the fight from features" CatBoost (control family A).
resid_*  : market-anchored residual models, logit p = logit q + f(x), CatBoost with
           baseline = logit q so trees only learn the correction to the closing line.

All models are corner-symmetric: training rows are augmented with the swapped
copy (label flipped, signed diffs negated, q -> 1-q) and predictions average the
two orientations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ufc_pred.features.static_v1 import _swap_red_blue, prepare
from ufc_pred.research.harness import logit, sigmoid
from ufc_pred.utils.time_splits import recency_weights

STATS_V2 = dict(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3)
RESID_V0 = dict(iterations=300, learning_rate=0.03, depth=3, l2_leaf_reg=20, rsm=0.5)

# Columns the harness adds that must never reach the feature matrix.
_HARNESS_COLS = ["q", "has_odds", "y", "card", "year", "debut", "odds_source"]


def _matrix(frame: pd.DataFrame, augment: bool, extra_drop: list[str]):
    frame = frame.drop(columns=[c for c in _HARNESS_COLS + extra_drop if c in frame])
    return prepare(frame, augment_symmetry=augment, one_hot=False)


def _swap(frame: pd.DataFrame, signed_extra: list[str]) -> pd.DataFrame:
    out = _swap_red_blue(frame)
    if "q" in out:
        out["q"] = 1 - out["q"]
    for c in signed_extra:
        if c in out:
            out[c] = -out[c]
    return out


def make_stats_fn(params: dict = STATS_V2, half_life: float = 4.0, threads: int = 8, drop: list[str] | None = None):
    drop = drop or []

    def fn(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
        X, y, d, cat = _matrix(train, True, drop)
        ref = test.date.min() - pd.Timedelta(days=1)
        w = recency_weights(d, reference_date=ref, half_life_years=half_life)
        m = CatBoostClassifier(**params, loss_function="Logloss", random_seed=seed, verbose=False,
                               allow_writing_files=False, thread_count=threads)
        m.fit(Pool(X, y, cat_features=cat, weight=w))
        both = pd.concat([test, _swap_red_blue(test)], ignore_index=True)
        Xt, _, _, _ = _matrix(both, False, drop)
        Xt = Xt.reindex(columns=m.feature_names_)
        p = m.predict_proba(Pool(Xt, cat_features=cat))[:, 1]
        n = len(test)
        return (p[:n] + 1 - p[n:]) / 2

    return fn


def fit_flb_slope(train: pd.DataFrame, half_life: float = 4.0, ref_date=None) -> float:
    """Recency-weighted, symmetric, intercept-free logistic slope of y on logit q (train only)."""
    from sklearn.linear_model import LogisticRegression

    t = train[train.has_odds]
    x = logit(t.q.to_numpy())
    ref = ref_date if ref_date is not None else t.date.max()
    w = recency_weights(t.date, reference_date=ref, half_life_years=half_life)
    X = np.concatenate([x, -x])[:, None]
    y = np.concatenate([t.y.to_numpy(), 1 - t.y.to_numpy()])
    lr = LogisticRegression(C=1e4, fit_intercept=False).fit(X, y, sample_weight=np.concatenate([w, w]))
    return float(lr.coef_[0, 0])


def make_flb_fn(half_life: float = 4.0):
    """M0 as a model function: p = sigma(a * logit q), a fitted on the training years."""

    def fn(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
        a = fit_flb_slope(train, half_life, test.date.min() - pd.Timedelta(days=1))
        return np.where(test.has_odds, sigmoid(a * logit(test.q.fillna(0.5))), np.nan)

    return fn


GLM_DIFFS = [
    "skill_diff_mean", "age_dif", "reach_dif", "height_dif", "win_dif", "loss_dif", "sig_str_dif",
    "avg_td_dif", "avg_sub_att_dif", "total_round_dif", "win_streak_dif", "lose_streak_dif",
    "ko_dif", "sub_dif", "total_title_bout_dif", "longest_win_streak_dif",
]


def make_glm_resid_fn(cols: list[str] = GLM_DIFFS, C: float = 0.05, half_life: float = 4.0, flb: bool = True):
    """Logistic regression with offset (a*logit q): only antisymmetric diff features, no intercept.

    Corner symmetry is exact: every feature is Blue-Red (or R-B) and flips sign on a swap,
    so p(swap) = 1 - p. Features are winsorised and standardised on the training rows;
    missing values become 0 (= no information).
    """
    from scipy.optimize import minimize

    def design(frame: pd.DataFrame, stats=None):
        X = frame[cols].astype(float).to_numpy()
        miss = np.isnan(X)
        if stats is None:
            # symmetric winsorisation keeps the model exactly antisymmetric under a corner swap
            cap = np.nanpercentile(np.abs(X), 99, axis=0)
            cap[~np.isfinite(cap) | (cap == 0)] = 1.0
            sd = np.nanstd(np.clip(X, -cap, cap), axis=0)
            sd[~np.isfinite(sd) | (sd == 0)] = 1.0
            stats = (cap, sd)
        cap, sd = stats
        Xc = np.where(miss, 0.0, np.clip(X, -cap, cap) / sd)
        return Xc, stats

    def fn(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
        tr = train[train.has_odds]
        ref = test.date.min() - pd.Timedelta(days=1)
        a = fit_flb_slope(tr, half_life, ref) if flb else 1.0
        X, stats = design(tr)
        off = a * logit(tr.q.to_numpy())
        y = tr.y.to_numpy().astype(float)
        w = recency_weights(tr.date, reference_date=ref, half_life_years=half_life)
        w = w / w.mean()
        lam = 1.0 / (C * len(y))

        def loss(beta):
            z = off + X @ beta
            p = sigmoid(z)
            ll = (w * (np.logaddexp(0.0, z) - y * z)).mean()
            g = -(X * (w * (y - p))[:, None]).mean(0)
            return ll + lam * beta @ beta, g + 2 * lam * beta

        res = minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
        if not res.success or not np.all(np.isfinite(res.x)) or np.abs(res.jac).max() > 1e-3:
            raise RuntimeError(f"GLM offset fit did not converge: {res.message}, max|grad|={np.abs(res.jac).max():.2e}")
        beta = res.x
        fn.fits.append(dict(test_start=str(ref.date()), n=len(y), a=a, nit=int(res.nit), max_grad=float(np.abs(res.jac).max())))
        out = np.full(len(test), np.nan)
        ok = test.has_odds.to_numpy()
        Xt, _ = design(test[ok], stats)
        out[ok] = sigmoid(a * logit(test.q[ok].to_numpy()) + Xt @ beta)
        return out

    fn.fits = []
    return fn


def make_resid_fn(
    params: dict = RESID_V0,
    half_life: float = 4.0,
    threads: int = 8,
    logit_q_feature: bool = False,
    keep: list[str] | None = None,
    drop: list[str] | None = None,
    signed_extra: list[str] | None = None,
    flb_baseline: bool = False,
):
    """Residual model over the closing line.

    keep: if given, restrict features to these columns (+ categoricals they include).
    signed_extra: extra columns (beyond *_dif / skill_diff_mean) that flip sign on corner swap.
    """
    drop = list(drop or [])
    signed_extra = list(signed_extra or [])

    def build(frame: pd.DataFrame, augment: bool, a: float):
        f = frame.copy()
        if logit_q_feature:
            f["mkt_logit_q"] = logit(f.q)
        sx = signed_extra + (["mkt_logit_q"] if logit_q_feature else [])
        if augment:
            sw = _swap(f, sx)
            sw["Winner"] = sw["Winner"].map({"Red": "Blue", "Blue": "Red"})
            f = pd.concat([f, sw], ignore_index=True)
        base = a * logit(f.q)
        X, y, d, cat = _matrix(f, False, drop)
        if keep is not None:
            cols = [c for c in X if c in keep or c == "mkt_logit_q"]
            X = X[cols]
            cat = [c for c in cat if c in cols]
        return X, y, d, cat, base

    def fn(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
        train = train[train.has_odds]
        ref = test.date.min() - pd.Timedelta(days=1)
        a = fit_flb_slope(train, half_life, ref) if flb_baseline else 1.0
        X, y, d, cat, base = build(train, True, a)
        w = recency_weights(d, reference_date=ref, half_life_years=half_life)
        m = CatBoostClassifier(**params, loss_function="Logloss", random_seed=seed, verbose=False,
                               allow_writing_files=False, thread_count=threads)
        m.fit(Pool(X, y, cat_features=cat, weight=w, baseline=base))
        fn.feature_names = list(m.feature_names_)
        out = np.full(len(test), np.nan)
        ok = test.has_odds.to_numpy()
        if not ok.any():
            return out
        t = test[ok].reset_index(drop=True)
        sx = signed_extra + (["mkt_logit_q"] if logit_q_feature else [])
        tt = t.copy()
        if logit_q_feature:
            tt["mkt_logit_q"] = logit(tt.q)
        both = pd.concat([tt, _swap(tt, sx)], ignore_index=True)
        Xt, _, _, _ = _matrix(both, False, drop)
        Xt = Xt.reindex(columns=m.feature_names_)
        raw = m.predict(Pool(Xt, cat_features=cat), prediction_type="RawFormulaVal")
        n = len(t)
        lq = a * logit(t.q.to_numpy())
        p_red = sigmoid(lq + raw[:n])
        p_red_from_swap = 1 - sigmoid(-lq + raw[n:])
        out[ok] = (p_red + p_red_from_swap) / 2
        return out

    return fn
