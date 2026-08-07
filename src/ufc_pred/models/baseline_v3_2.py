"""v3.2: v3 (skill features) + v2 (derived no-scrape features).

Diagnostic combo. v2 missed the log-loss keep-or-kill bar on its own but gave a
large ECE improvement; v3 cleared the log-loss bar. This checks whether the two
signals are complementary or redundant.

Derived v2 columns are all R_/B_ prefixed and are handled by the standard
red/blue swap, so only `skill_diff_mean` needs an explicit sign flip.
"""

from __future__ import annotations

from ufc_pred.features.joins import join_skill_v3_and_derived
from ufc_pred.models._report import main
from ufc_pred.models._spec import ModelSpec, Recipe

VERSION = "v3_2_catboost_skill_derived"

SPEC = ModelSpec(
    version=VERSION,
    recipe=Recipe(
        name="v3.2 (scalar skill + derived no-scrape)",
        join=join_skill_v3_and_derived,
        join_before_winner_filter=True,
        flip_columns=("skill_diff_mean",),
    ),
    early_stopping=True,
)


if __name__ == "__main__":
    main(SPEC)
