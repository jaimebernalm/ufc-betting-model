"""Build the accredited Kalshi opportunity table (venue validation set).

Only prices captured live with a timestamp count as executable:
  * data/processed/bet_notifications/*.json            (live runner, Jun-Aug 2026)
  * data/processed/shadow_candidate_2026_09_14_v2/captures/*.json  (shadow, Sep-Oct 2026)
The historical "T-90" Kalshi parquet is NOT used here (timing not accredited).

Output: artifacts/r2026_10/opportunities_kalshi.parquet, oriented to (R_fighter, B_fighter)
of the outcome table, with ask/bid per side, capture time, outcome y (Red wins) and the
old system's model probability for reference.
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from ufc_pred.paths import ROOT

OUT = ROOT / "artifacts/r2026_10/opportunities_kalshi.parquet"
SOURCES = {
    "live_notification": ROOT / "data/processed/bet_notifications",
    "shadow_v2": ROOT / "data/processed/shadow_candidate_2026_09_14_v2/captures",
}
HISTORY = ROOT / "data/processed/shadow_candidate_2026_09_14_v2/history.parquet"


def norm(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").replace(".", "").replace("'", "").split())


def load_captures() -> pd.DataFrame:
    rows = []
    for src, folder in SOURCES.items():
        for f in sorted(glob.glob(str(folder / "*.json"))):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            ks = j.get("kalshi_snapshot") or {}
            fight = j.get("fight") or {}
            model = j.get("model") or {}
            if ks.get("a_ask") is None or ks.get("b_ask") is None:
                continue
            fd = pd.Timestamp(j["fight_date_utc"])
            rows.append(
                dict(
                    source=src,
                    file=f.split("/")[-1],
                    event=j.get("event"),
                    fight_date_utc=fd,
                    card_date=(fd.tz_convert("America/New_York")).tz_localize(None).normalize(),
                    captured_at_utc=pd.Timestamp(j.get("captured_at_utc")),
                    # snapshot asks and model probabilities follow KALSHI's ticker_a/ticker_b order;
                    # kalshi_swap_vs_ufc=True means ticker_a is the UFC fighter_b
                    a=fight.get("fighter_b_ufc")
                    if fight.get("kalshi_swap_vs_ufc")
                    else fight.get("fighter_a_ufc"),
                    b=fight.get("fighter_a_ufc")
                    if fight.get("kalshi_swap_vs_ufc")
                    else fight.get("fighter_b_ufc"),
                    swap_vs_ufc=fight.get("kalshi_swap_vs_ufc"),
                    a_ask=ks.get("a_ask"),
                    b_ask=ks.get("b_ask"),
                    a_bid=ks.get("a_bid"),
                    b_bid=ks.get("b_bid"),
                    volume=ks.get("volume"),
                    old_p_a=model.get("p_a_real"),
                    old_p_a_raw=model.get("p_a_real_raw"),
                )
            )
    d = pd.DataFrame(rows)
    # one row per fight: keep the earliest accredited capture (closest to the decision rule)
    d = d.sort_values("captured_at_utc").drop_duplicates(subset=["card_date", "a", "b"], keep="first")
    return d.reset_index(drop=True)


def attach_outcomes(d: pd.DataFrame) -> pd.DataFrame:
    h = pd.read_parquet(HISTORY)[["date", "R_fighter", "B_fighter", "Winner"]]
    h = h[h.Winner.isin(["Red", "Blue"])].copy()
    h["key"] = [frozenset((norm(r), norm(b))) for r, b in zip(h.R_fighter, h.B_fighter)]
    out = []
    for _, r in d.iterrows():
        k = frozenset((norm(r.a), norm(r.b)))
        m = h[(h.key == k) & ((h.date - r.card_date).abs() <= pd.Timedelta(days=1))]
        if len(m) != 1:
            out.append(dict(R_fighter=None, B_fighter=None, date=pd.NaT, y=np.nan, a_is_red=np.nan))
            continue
        m = m.iloc[0]
        a_is_red = norm(m.R_fighter) == norm(r.a)
        out.append(
            dict(
                R_fighter=m.R_fighter,
                B_fighter=m.B_fighter,
                date=m.date,
                y=int(m.Winner == "Red"),
                a_is_red=a_is_red,
            )
        )
    return pd.concat([d, pd.DataFrame(out)], axis=1)


def main():
    d = attach_outcomes(load_captures())
    print(f"captures with asks: {len(d)}; matched to a decisive result: {d.y.notna().sum()}")
    print(d[d.y.isna()][["card_date", "a", "b"]].to_string())
    d = d[d.y.notna()].copy()
    flip = ~d.a_is_red.astype(bool)
    # orient everything to Red/Blue of the outcome table
    d["R_ask"] = np.where(flip, d.b_ask, d.a_ask)
    d["B_ask"] = np.where(flip, d.a_ask, d.b_ask)
    d["R_bid"] = np.where(flip, d.b_bid, d.a_bid)
    d["B_bid"] = np.where(flip, d.a_bid, d.b_bid)
    d["old_p_R"] = np.where(flip, 1 - d.old_p_a, d.old_p_a)
    d["old_p_R_raw"] = np.where(flip, 1 - d.old_p_a_raw, d.old_p_a_raw)
    mid_r = (d.R_ask + d.R_bid) / 2
    mid_b = (d.B_ask + d.B_bid) / 2
    d["q_kalshi_mid"] = mid_r / (mid_r + mid_b)
    d["card"] = d.date.dt.strftime("%Y-%m-%d")
    # fight_date_utc is the scheduled segment/card start, not the bout's own start; a capture
    # within a few minutes after it is still pre-bout for later fights, but timing is NOT verified
    # per bout -> venue results are a diagnostic, not an executable T-90 backtest.
    d["minutes_after_sched_start"] = (d.captured_at_utc - d.fight_date_utc).dt.total_seconds() / 60
    cols = [
        "date",
        "R_fighter",
        "B_fighter",
        "card",
        "y",
        "source",
        "event",
        "captured_at_utc",
        "fight_date_utc",
        "R_ask",
        "B_ask",
        "R_bid",
        "B_bid",
        "q_kalshi_mid",
        "volume",
        "old_p_R",
        "old_p_R_raw",
        "minutes_after_sched_start",
    ]
    d[cols].to_parquet(OUT, index=False)
    print(d.groupby("source").size(), d.card.nunique(), "cards ->", OUT)


if __name__ == "__main__":
    main()
