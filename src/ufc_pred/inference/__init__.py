"""Live inference: score an upcoming fight and size a bet against a venue price.

The pipeline, in order:
  `upcoming_builder`  build the model's feature row for a scheduled fight
  `skill_for_upcoming` attach the Bayesian skill posterior as of the fight date
  `ensemble_predict`   average predict_proba across the deployed seeds
  `sizing`             fee-correct fractional-Kelly stake, capped
  `recommend`          tie the above together into one recommendation dict

`polymarket_card` / `kalshi_card` supply the venue-side market snapshot.

Read-only with respect to the exchange — no module here places an order.
"""

from ufc_pred.inference.recommend import predict_one_fight

__all__ = ["predict_one_fight"]
