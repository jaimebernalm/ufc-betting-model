"""Train/serve parity test.

BACKTEST path : the fight's own row in fights.parquet (stats as recorded).
LIVE path     : build_upcoming_row -> _last_fighter_stats reconstructs from the
                fighter's PREVIOUS fight row + manual W/L/streak increments.

If those disagree, every backtest ever run is measuring a different model from
the one that trades. Skill features are copied from the recorded row in both
arms so this isolates the reconstruction, not the skill posterior.
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd

from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL
from ufc_pred.features.static_v1 import prepare
from ufc_pred.inference.upcoming_builder import build_upcoming_row
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.ingest.rankings_attach import load_rankings
from ufc_pred.paths import MODELS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 250

f = pd.read_parquet(HISTORY_PARQUET)
f = f[f["Winner"].isin(["Red", "Blue"])].reset_index(drop=True)
f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None)
sk = pd.read_parquet(SKILL)
sk["date"] = pd.to_datetime(sk["date"]).dt.tz_localize(None)
f = f.merge(
    sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
    on=["date", "R_fighter", "B_fighter"],
    how="left",
    validate="many_to_one",
)

seen = {}
rp = []
bp = []
for _, r in f.sort_values("date").iterrows():
    rp.append(seen.get(r["R_fighter"], 0))
    bp.append(seen.get(r["B_fighter"], 0))
    seen[r["R_fighter"]] = seen.get(r["R_fighter"], 0) + 1
    seen[r["B_fighter"]] = seen.get(r["B_fighter"], 0) + 1
i = f.sort_values("date").index
f["r_prior"] = pd.Series(rp, index=i).reindex(f.index)
f["b_prior"] = pd.Series(bp, index=i).reindex(f.index)
f["is_debut"] = (f["r_prior"] == 0) | (f["b_prior"] == 0)

poly = pd.read_parquet(ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet")
poly["date"] = pd.to_datetime(poly["date"]).dt.tz_localize(None)
P = poly[["date", "R_fighter", "B_fighter", "polymarket_p_red", "polymarket_p_blue"]].merge(
    f, on=["date", "R_fighter", "B_fighter"], how="inner"
)
P = P[~P["is_debut"]].reset_index(drop=True)
target = P.sample(n=min(N, len(P)), random_state=1).reset_index(drop=True)
print(f"Polymarket debut-excluded fights: {len(P)};  testing parity on {len(target)}")

rankings = load_rankings()
live_rows, ok_idx = [], []
for n, row in target.iterrows():
    try:
        lr = build_upcoming_row(
            fighter_a=row["R_fighter"],
            fighter_b=row["B_fighter"],
            fight_date=row["date"],
            weight_class=row["weight_class"],
            fights=f,
            gender=row.get("gender", "MALE"),
            title_bout=bool(row.get("title_bout", False)),
            no_of_rounds=int(row.get("no_of_rounds", 3) or 3),
            rankings=rankings,
        )
        # isolate the reconstruction: take skill from the recorded row
        lr["skill_diff_mean"] = row["skill_diff_mean"]
        lr["skill_diff_std"] = row["skill_diff_std"]
        live_rows.append(lr.iloc[0])
        ok_idx.append(n)
    except Exception as e:
        if n < 5:
            print("  build failed:", e)
print(f"built {len(live_rows)} live rows")

live = pd.DataFrame(live_rows).reset_index(drop=True)
back = target.loc[ok_idx].reset_index(drop=True)

# ------------------------------------------------------------- feature diffs
tv = joblib.load(MODELS / "v3_catboost_full2000_trainval.joblib")
cols = tv["columns"]
cat = tv.get("cat_features", [])


def featurize(frame):
    X, _, _, _ = prepare(frame, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=cols, fill_value=None)
    for c in cat:
        X[c] = X[c].fillna("__missing__").astype(str)
    return X


Xl, Xb = featurize(live), featurize(back)
num = [c for c in cols if c not in cat and pd.api.types.is_numeric_dtype(Xb[c])]
rows = []
for c in num:
    a = pd.to_numeric(Xl[c], errors="coerce").astype(float)
    b = pd.to_numeric(Xb[c], errors="coerce").astype(float)
    both = a.notna() & b.notna()
    if both.sum() == 0:
        continue
    d = a[both] - b[both]
    denom = b[both].abs().replace(0, np.nan)
    rows.append(
        dict(
            feature=c,
            n=int(both.sum()),
            pct_differing=100.0 * float((d.abs() > 1e-9).mean()),
            mean_diff=float(d.mean()),
            mean_abs_diff=float(d.abs().mean()),
            median_rel=float((d.abs() / denom).median() * 100),
        )
    )
D = pd.DataFrame(rows).sort_values("pct_differing", ascending=False)
print(f"\nnumeric features compared: {len(D)}")
print(f"features differing on >50% of rows: {(D['pct_differing'] > 50).sum()}")
print(f"features identical everywhere    : {(D['pct_differing'] == 0).sum()}")
print("\nTop 20 most-divergent features (live reconstruction vs recorded row):")
print(
    D.head(20).to_string(
        index=False,
        formatters={
            "pct_differing": "{:.1f}%".format,
            "mean_diff": "{:+.4f}".format,
            "mean_abs_diff": "{:.4f}".format,
            "median_rel": "{:.2f}%".format,
        },
    )
)

# ---------------------------------------------------------- prediction impact
p_live = tv["model"].predict_proba(Xl)[:, 1]
p_back = tv["model"].predict_proba(Xb)[:, 1]
print(
    f"\np(Red) live vs backtest:  mean |diff| = {np.abs(p_live - p_back).mean():.4f}   "
    f"max = {np.abs(p_live - p_back).max():.4f}   corr = {np.corrcoef(p_live, p_back)[0, 1]:.4f}"
)
print(f"  rows where |diff| > 0.05 : {(np.abs(p_live - p_back) > 0.05).sum()} / {len(p_live)}")
print(f"  rows where |diff| > 0.10 : {(np.abs(p_live - p_back) > 0.10).sum()} / {len(p_live)}")

out = back[["date", "R_fighter", "B_fighter", "Winner", "polymarket_p_red", "polymarket_p_blue"]].copy()
out["p_back"], out["p_live"] = p_back, p_live
out.to_parquet("/tmp/parity.parquet", index=False)
live.to_parquet("/tmp/parity_live_rows.parquet", index=False)
back.to_parquet("/tmp/parity_back_rows.parquet", index=False)
print("\nwrote /tmp/parity.parquet, /tmp/parity_live_rows.parquet, /tmp/parity_back_rows.parquet")
