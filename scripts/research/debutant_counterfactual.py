"""What would the account be worth if the debutant fights had been bettable?

Scores the skipped fights the way the BACKTEST does (the fight's own row in
fights.parquet, with pre-fight 0-0 stats for the debutant), prices them at the
Kalshi ask actually captured live, and settles against Kalshi results.
"""

import glob
import json

import joblib
import numpy as np
import pandas as pd

from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import MODELS, PROCESSED

SHARPEN_T = 1.25
FEE = 0.07
EDGE_THR = 0.03

fights = pd.read_parquet(HISTORY_PARQUET)
fights = fights[fights["Winner"].isin(["Red", "Blue"])].reset_index(drop=True)
fights["date"] = pd.to_datetime(fights["date"])
sk = pd.read_parquet(SKILL_V3_PARQUET)
sk["date"] = pd.to_datetime(sk["date"])
fights = fights.merge(
    sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
    on=["date", "R_fighter", "B_fighter"],
    how="left",
    validate="many_to_one",
)

payload = joblib.load(MODELS / "v3_catboost_full2000_trainval.joblib")
cols, cat = payload["columns"], payload.get("cat_features", [])
X, _, _, _ = prepare(fights, augment_symmetry=False, one_hot=False)
X = X.reindex(columns=cols, fill_value=None)
for c in cat:
    X[c] = X[c].fillna("__missing__").astype(str)
fights["p_red"] = payload["model"].predict_proba(X)[:, 1]


def sharpen(p, T=SHARPEN_T):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-T * np.log(p / (1 - p))))


settle = json.load(open("/tmp/settle_all.json"))

# ---- gather the skipped ("No fighter match") fights with their captured prices ----
skipped = []
for p_ in sorted(glob.glob(str(PROCESSED / "bet_notifications" / "*.json"))):
    d = json.load(open(p_))
    rec = d.get("recommendation", {})
    if rec.get("status") != "error":
        continue
    f = d["fight"]
    skipped.append(
        {
            "date": pd.Timestamp(d["fight_date_utc"]).tz_convert("UTC"),
            "fa": f["fighter_a_ufc"],
            "fb": f["fighter_b_ufc"],
            "ta": f["kalshi_ticker_a"],
            "tb": f["kalshi_ticker_b"],
            "ask_a": d["kalshi_snapshot"]["a_ask"],
            "ask_b": d["kalshi_snapshot"]["b_ask"],
            "bank": d["bankrolls_at_capture"],
        }
    )

rows = []
for s in skipped:
    m = fights[
        ((fights.R_fighter == s["fa"]) & (fights.B_fighter == s["fb"]))
        | ((fights.R_fighter == s["fb"]) & (fights.B_fighter == s["fa"]))
    ]
    if m.empty:
        rows.append({**s, "scoreable": False})
        continue
    r = m.sort_values("date").iloc[-1]
    p_a = r["p_red"] if r["R_fighter"] == s["fa"] else 1 - r["p_red"]
    p_a = sharpen(p_a)
    # pick the side with the better fee-correct edge
    best = None
    for side, price, p, tk in (
        ("A", s["ask_a"], p_a, s["ta"]),
        ("B", s["ask_b"], 1 - p_a, s["tb"]),
    ):
        if not price or price <= 0:
            continue
        fee = FEE * price * (1 - price)
        ev = p * (1 - price - fee) - (1 - p) * (price + fee)
        edge = ev / price
        if best is None or edge > best["edge"]:
            best = {"side": side, "price": price, "p": p, "edge": edge, "ticker": tk}
    won = settle.get(best["ticker"], [None, None])[1] == "yes"
    rows.append(
        {
            **s,
            "scoreable": True,
            "p_a": p_a,
            **best,
            "won": won,
            "bet_name": s["fa"] if best["side"] == "A" else s["fb"],
        }
    )

df = pd.DataFrame(rows).sort_values("date")
print("=" * 100)
print("THE 11 FIGHTS THE LIVE MODEL SKIPPED — scored the way the backtest would have")
print("=" * 100)
for _, r in df.iterrows():
    if not r["scoreable"]:
        print(f"{str(r['date'])[:10]}  {r['fa']} vs {r['fb']:<22s}  NOT SCOREABLE (card not ingested)")
        continue
    act = "BET" if r["edge"] > EDGE_THR else "no bet"
    print(
        f"{str(r['date'])[:10]}  {r['fa']} vs {r['fb']:<22s}  "
        f"p_a={r['p_a']:.3f}  {act:6s} {r['bet_name'][:20]:<21s} @{r['price'] * 100:3.0f}c "
        f"edge {r['edge'] * 100:+5.1f}c  -> {'WIN' if r['won'] else 'LOSS'}"
    )

df.to_csv("/tmp/debutant_cf.csv", index=False)
