"""Match Kalshi historical per-fight markets to Kaggle fights.

Extracted from notebook 08's matching cells so scripts (anchor strategy,
model-on-top ablation, segmentation) share one implementation. Prices in
`data/raw/kalshi/historical.parquet` are the canonical T-90min pre-fight
capture (finding #35/#36); the nb08 sanity filter is applied here.

Output: one row per matched fight, Kalshi prices oriented to the Kaggle
Red/Blue corners, alongside the Kaggle sportsbook moneylines.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from ufc_pred.paths import ROOT

APOSTROPHES = "'’ʼ`‘"


def _deep_norm(s) -> str:
    if pd.isna(s) or s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    for ch in APOSTROPHES + "-.,":
        s = s.replace(ch, "")
    return re.sub(r"\s+", " ", s).strip().lower()


def _last_token(s) -> str:
    s = _deep_norm(s)
    return s.split()[-1] if s else ""


def load_kalshi(path: Path | None = None) -> pd.DataFrame:
    """Kalshi historical rows passing the nb08 sanity filter."""
    kal = pd.read_parquet(path or ROOT / "data/raw/kalshi/historical.parquet")
    kal = kal.dropna(subset=["close_yes_price_a", "close_yes_price_b", "winner"]).copy()
    ab_sum = kal["close_yes_price_a"] + kal["close_yes_price_b"]
    keep = (
        (kal["close_yes_price_a"] >= 0.02)
        & (kal["close_yes_price_b"] >= 0.02)
        & (ab_sum >= 0.80)
        & (ab_sum <= 1.30)
        & ((kal["volume_a"] + kal["volume_b"]) >= 100)
    )
    kal = kal[keep].reset_index(drop=True)
    kal["fd_n"] = pd.to_datetime(kal["fight_date"]).dt.tz_localize(None).dt.normalize()
    kal["fd_m1"] = kal["fd_n"] - pd.Timedelta(days=1)
    kal["a_dn"] = kal["fighter_a"].map(_deep_norm)
    kal["b_dn"] = kal["fighter_b"].map(_deep_norm)
    kal["a_last"] = kal["fighter_a"].map(_last_token)
    kal["b_last"] = kal["fighter_b"].map(_last_token)
    return kal


def _match_one(row, fights: pd.DataFrame, fights_by_date: dict):
    dates = set()
    for d in (row["fd_n"], row["fd_m1"]):
        for off in (-1, 0, 1):
            dates.add(d + pd.Timedelta(days=off))
    pools = [fights_by_date[d] for d in dates if d in fights_by_date]
    if pools:
        pool = pd.concat(pools)
        a, b = row["a_dn"], row["b_dn"]
        a_l, b_l = row["a_last"], row["b_last"]
        h = pool[((pool["R_dn"] == a) & (pool["B_dn"] == b)) | ((pool["R_dn"] == b) & (pool["B_dn"] == a))]
        if len(h) == 1:
            k = h.iloc[0]
            return k, bool(k["R_dn"] == a), "exact_full"
        h = pool[
            ((pool["R_last"] == a_l) & (pool["B_last"] == b_l))
            | ((pool["R_last"] == b_l) & (pool["B_last"] == a_l))
        ]
        if len(h) == 1:
            k = h.iloc[0]
            return k, bool(k["R_last"] == a_l), "exact_last"
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
        return k, bool(k["R_last"] == row["a_last"]), "wide_14d"
    return None, None, "no"


def match_kalshi_to_fights(
    fights: pd.DataFrame, kal: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (matched_meta, matched_fights) aligned row-for-row.

    matched_meta columns: date, kalshi Red/Blue prices+volumes, match_reason,
    winner agreement. Rows where Kalshi and Kaggle disagree on the winner are
    dropped (settlement ambiguity).
    """
    if kal is None:
        kal = load_kalshi()
    fights = fights.copy()
    fights["date"] = pd.to_datetime(fights["date"])
    fights["R_dn"] = fights["R_fighter"].map(_deep_norm)
    fights["B_dn"] = fights["B_fighter"].map(_deep_norm)
    fights["R_last"] = fights["R_fighter"].map(_last_token)
    fights["B_last"] = fights["B_fighter"].map(_last_token)
    fights_by_date = {d: g for d, g in fights.groupby("date")}

    rows, kag_idx = [], []
    for _, r in kal.iterrows():
        k, a_is_red, reason = _match_one(r, fights, fights_by_date)
        if k is None:
            continue
        pa, pb = r["close_yes_price_a"], r["close_yes_price_b"]
        rows.append(
            {
                "date": k["date"],
                "a_is_red": bool(a_is_red),
                "kal_p_red": pa if a_is_red else pb,
                "kal_p_blue": pb if a_is_red else pa,
                "kal_winner": r["winner"],
                "volume": float(r["volume_a"] + r["volume_b"]),
                "match_reason": reason,
                "event_ticker": r.get("event_ticker", ""),
                "close_time": r.get("close_time", pd.NaT),
            }
        )
        kag_idx.append(k.name)

    meta = pd.DataFrame(rows)
    matched = fights.loc[kag_idx].reset_index(drop=True)
    # Winner agreement check (nb08): drop disagreements.
    kag_ab = np.where(
        (meta["a_is_red"] & (matched["Winner"].to_numpy() == "Red"))
        | (~meta["a_is_red"] & (matched["Winner"].to_numpy() == "Blue")),
        "A",
        "B",
    )
    ok = kag_ab == meta["kal_winner"].to_numpy()
    return meta[ok].reset_index(drop=True), matched[ok.tolist()].reset_index(drop=True)
