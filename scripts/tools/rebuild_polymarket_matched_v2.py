"""Rebuild data/interim/polymarket_matched_to_kaggle_v2.parquet.

Same algorithm as notebook 05's matcher: deep-normalize names, ±2-day window,
exact->last->substring->fuzzy passes, plus ±14-day wide fallback for
rescheduled fights. Writes the same columns the downstream backtest scripts
expect.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]

from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET

APOSTROPHES = "'’ʼ`‘"


def deep_norm(s):
    if pd.isna(s) or s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    for ch in APOSTROPHES + "-.,":
        s = s.replace(ch, "")
    return re.sub(r"\s+", " ", s).strip().lower()


def last_token(s):
    s = deep_norm(s)
    return s.split()[-1] if s else ""


# Polymarket
poly = pd.read_parquet(ROOT / "data/raw/polymarket/historical_2024-04-13_to_2026-05-24.parquet")
poly = poly.dropna(subset=["closing_price_a", "closing_price_b", "winner"]).copy()
poly["fd_n"] = pd.to_datetime(poly["fight_date"]).dt.tz_localize(None).dt.normalize()
poly["fd_m1"] = poly["fd_n"] - pd.Timedelta(days=1)
poly["a_dn"] = poly["fighter_a"].map(deep_norm)
poly["b_dn"] = poly["fighter_b"].map(deep_norm)
poly["a_last"] = poly["fighter_a"].map(last_token)
poly["b_last"] = poly["fighter_b"].map(last_token)

# Kaggle fights + skill features
fights = pd.read_parquet(HISTORY_PARQUET)
fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
fights["date"] = pd.to_datetime(fights["date"])
fights["R_dn"] = fights["R_fighter"].map(deep_norm)
fights["B_dn"] = fights["B_fighter"].map(deep_norm)
fights["R_last"] = fights["R_fighter"].map(last_token)
fights["B_last"] = fights["B_fighter"].map(last_token)

sk = pd.read_parquet(SKILL_V3_PARQUET)
sk["date"] = pd.to_datetime(sk["date"])
fights = fights.merge(
    sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
    on=["date", "R_fighter", "B_fighter"],
    how="left",
    validate="many_to_one",
)

print(f"Polymarket rows: {len(poly)}   Kaggle rows: {len(fights)}")
print(f"PM date range: {poly.fd_n.min().date()} -> {poly.fd_n.max().date()}")
print(f"Kaggle date range: {fights.date.min().date()} -> {fights.date.max().date()}")

fights_by_date = {d: g for d, g in fights.groupby("date")}


def candidates_local(row):
    dates = set()
    for d in (row["fd_n"], row["fd_m1"]):
        for off in (-1, 0, 1):
            dates.add(d + pd.Timedelta(days=off))
    pools = [fights_by_date[d] for d in dates if d in fights_by_date]
    return pd.concat(pools) if pools else pd.DataFrame()


def match_one(row):
    pool = candidates_local(row)
    if len(pool) > 0:
        a, b = row["a_dn"], row["b_dn"]
        a_l, b_l = row["a_last"], row["b_last"]
        h = pool[((pool["R_dn"] == a) & (pool["B_dn"] == b)) | ((pool["R_dn"] == b) & (pool["B_dn"] == a))]
        if len(h) == 1:
            k = h.iloc[0]
            return k, k["R_dn"] == a, "exact_full"
        h = pool[
            ((pool["R_last"] == a_l) & (pool["B_last"] == b_l))
            | ((pool["R_last"] == b_l) & (pool["B_last"] == a_l))
        ]
        if len(h) == 1:
            k = h.iloc[0]
            return k, k["R_last"] == a_l, "exact_last"
        h = pool[
            ((pool["R_dn"].str.contains(a_l, regex=False)) & (pool["B_dn"].str.contains(b_l, regex=False)))
            | ((pool["R_dn"].str.contains(b_l, regex=False)) & (pool["B_dn"].str.contains(a_l, regex=False)))
        ]
        if len(h) == 1:
            k = h.iloc[0]
            return k, a_l in k["R_dn"], "substr"

        def fp(k):
            return (
                max(fuzz.ratio(a_l, k["R_last"]), fuzz.ratio(a_l, k["B_last"]))
                + max(fuzz.ratio(b_l, k["R_last"]), fuzz.ratio(b_l, k["B_last"]))
            ) / 2

        pool = pool.copy()
        pool["fuz"] = pool.apply(fp, axis=1)
        best = pool.sort_values("fuz", ascending=False).head(2)
        if (
            len(best) > 0
            and best.iloc[0]["fuz"] >= 88
            and (len(best) == 1 or best.iloc[0]["fuz"] - best.iloc[1]["fuz"] >= 5)
        ):
            k = best.iloc[0]
            return k, fuzz.ratio(a_l, k["R_last"]) >= fuzz.ratio(a_l, k["B_last"]), "fuzzy"
    d = row["fd_n"]
    w = fights[(fights["date"] >= d - pd.Timedelta(days=14)) & (fights["date"] <= d + pd.Timedelta(days=14))]
    h = w[
        ((w["R_last"] == row["a_last"]) & (w["B_last"] == row["b_last"]))
        | ((w["R_last"] == row["b_last"]) & (w["B_last"] == row["a_last"]))
    ]
    if len(h) == 1:
        k = h.iloc[0]
        return k, k["R_last"] == row["a_last"], "wide_14d"
    return None, None, "no"


matches = []
matched_kag_rows = []
for i, row in poly.iterrows():
    k, a_is_red, reason = match_one(row)
    if k is None:
        continue
    if k.name in matched_kag_rows:  # dedupe
        continue
    matches.append(
        {
            "date": k["date"],
            "a_is_red": a_is_red,
            "poly_a": row["fighter_a"],
            "poly_b": row["fighter_b"],
            "closing_price_a": row["closing_price_a"],
            "closing_price_b": row["closing_price_b"],
            "market_id": str(row["market_id"]),
            "market_slug": row["market_slug"],
            "token_id_a": str(row["token_id_a"]),
            "token_id_b": str(row["token_id_b"]),
            "kag_R": k["R_fighter"],
            "kag_B": k["B_fighter"],
            "kag_winner": k["Winner"],
            "match_reason": reason,
        }
    )
    matched_kag_rows.append(k.name)

m_df = pd.DataFrame(matches)
matched_fights = fights.loc[matched_kag_rows].reset_index(drop=True)
print(f"\nMatched: {len(m_df)}/{len(poly)} ({len(m_df) / len(poly) * 100:.1f}%)")
print(f"Match date range: {m_df.date.min().date()} -> {m_df.date.max().date()}")
print("By pass:")
for r, n in m_df["match_reason"].value_counts().items():
    print(f"  {r:12s}: {n}")

# Build the output schema downstream scripts expect:
# columns: date, R_fighter, B_fighter, Winner, R_odds, B_odds,
#          polymarket_p_red, polymarket_p_blue, model_p_red, match_reason
out_rows = []
for i, m in enumerate(matches):
    kg = matched_fights.iloc[i]
    if m["a_is_red"]:
        p_red = m["closing_price_a"]
        p_blue = m["closing_price_b"]
    else:
        p_red = m["closing_price_b"]
        p_blue = m["closing_price_a"]

    # Approx American odds from the no-vig probabilities for legacy compat.
    def to_am(p):
        dec = 1.0 / float(p)
        return (dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)

    out_rows.append(
        {
            "date": kg["date"],
            "R_fighter": kg["R_fighter"],
            "B_fighter": kg["B_fighter"],
            "Winner": kg["Winner"],
            "R_odds": to_am(p_red),
            "B_odds": to_am(p_blue),
            "polymarket_p_red": float(p_red),
            "polymarket_p_blue": float(p_blue),
            "match_reason": m["match_reason"],
            "market_id": m["market_id"],
            "market_slug": m["market_slug"],
            "token_id_red": m["token_id_a"] if m["a_is_red"] else m["token_id_b"],
            "token_id_blue": m["token_id_b"] if m["a_is_red"] else m["token_id_a"],
        }
    )

# Add model_p_red from the trainval real model (best-effort; for legacy notebook compat).
out_df = pd.DataFrame(out_rows)
try:
    payload = joblib.load(ROOT / "artifacts/models/v3_catboost_full2000_trainval.joblib")
    X, _, _, _ = prepare(matched_fights, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=payload["columns"], fill_value=None)
    for c in payload.get("cat_features", []):
        X[c] = X[c].fillna("__missing__").astype(str)
    out_df["model_p_red"] = payload["model"].predict_proba(X)[:, 1]
except Exception as e:
    print(f"(skip model_p_red: {e})")
    out_df["model_p_red"] = np.nan

out_df = out_df.sort_values("date").reset_index(drop=True)
out_path = ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet"
out_df.to_parquet(out_path, index=False)
print(
    f"\nWrote {out_path}  rows={len(out_df)}  range={out_df.date.min().date()} -> {out_df.date.max().date()}"
)
