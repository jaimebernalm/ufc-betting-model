"""Market-structure features from the closing method props (Kaggle/BFO), keyed like training_features.

mkt_props_dif   = logit(q_props_R) - logit(q_R): win prob implied by KO/SUB/DEC props vs the moneyline
                  (antisymmetric; the `_dif` suffix makes _swap_red_blue negate it)
mkt_finish_prob = implied P(fight ends inside the distance) from the props (corner-symmetric)
Both are closing prices, same timestamp class as q.
"""

import numpy as np

from ufc_pred.research.harness import KEYS, OUT, american_to_prob, load_base, logit

COLS = ["r_ko_odds", "r_sub_odds", "r_dec_odds", "b_ko_odds", "b_sub_odds", "b_dec_odds"]


def main():
    df = load_base()
    ok = df[COLS].notna().all(1) & df.has_odds & (df[COLS].abs() >= 100).all(1)
    P = {c: american_to_prob(df[c]) for c in COLS}
    r = P["r_ko_odds"] + P["r_sub_odds"] + P["r_dec_odds"]
    b = P["b_ko_odds"] + P["b_sub_odds"] + P["b_dec_odds"]
    fin = (P["r_ko_odds"] + P["r_sub_odds"] + P["b_ko_odds"] + P["b_sub_odds"]) / (r + b)
    out = df[KEYS].copy()
    out["mkt_props_dif"] = np.where(ok, logit(r / (r + b)) - logit(df.q.fillna(0.5)), np.nan)
    out["mkt_finish_prob"] = np.where(ok, fin, np.nan)
    path = OUT / "claude/mkt_extra.parquet"
    out.to_parquet(path, index=False)
    print(path, out.notna().mean().round(3).to_dict())


if __name__ == "__main__":
    main()
