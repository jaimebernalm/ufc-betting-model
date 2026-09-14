"""Versioned, conservative identity keys. Display names are not unique IDs.

The two Bruno Silva profiles were verified in the UFCStats export. Unknown
homonyms fail closed instead of choosing a profile by listing order.
"""

import json
import re
import unicodedata

from ufc_pred.paths import CONFIGS

ALIAS_PATH = CONFIGS / "fighter_aliases.json"
ALIASES = json.loads(ALIAS_PATH.read_text())["aliases"] if ALIAS_PATH.exists() else {}

VERSION = "2026-09-14.2"
PROFILE_URLS = {
    "Mike Davis": "http://ufcstats.com/fighter-details/fb3e61720be4690c",
    "Joey Gomez": "http://ufcstats.com/fighter-details/0778f94eb5d588a5",
    "Michael McDonald": "http://ufcstats.com/fighter-details/d0314416a7f26527",
    "Jean Silva": "http://ufcstats.com/fighter-details/52ef95b5860fb28c",
    "Victor Valenzuela": "http://ufcstats.com/fighter-details/078695e385ec2f57",
    "Bruno Silva [FLW]": "http://ufcstats.com/fighter-details/294aa73dbf37d281",
    "Bruno Silva [MW]": "http://ufcstats.com/fighter-details/12ebd7d157e91701",
}


def normalise(name):
    text = "".join(
        c for c in unicodedata.normalize("NFKD", str(name).casefold()) if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def canonical(name, weight_class=None):
    name = str(name).strip()
    seen = set()
    while name in ALIASES and name not in seen:
        seen.add(name)
        name = ALIASES[name]
    if name in PROFILE_URLS:
        return name
    if normalise(name) in {"su mudaerji", "sumudaerji"}:
        return "Sumudaerji"
    if normalise(name) == "bruno silva":
        wc = str(weight_class).lower()
        if "middleweight" in wc or wc == "185 lbs.":
            return "Bruno Silva [MW]"
        if "flyweight" in wc or "bantamweight" in wc or wc == "125 lbs.":
            return "Bruno Silva [FLW]"
        raise ValueError("Bruno Silva is ambiguous: supply weight class or stable profile identity")
    return str(name).strip()


def canonicalize_frame(fights):
    out = fights.copy()
    for side in ("R", "B"):
        out[f"{side}_fighter"] = [
            canonical(n, wc) for n, wc in zip(out[f"{side}_fighter"], out["weight_class"], strict=True)
        ]
    return out


def verified_envelope(fights):
    """Apply the audited outcome corrections without adding any future rows."""
    import pandas as pd

    out = canonicalize_frame(fights)
    path = CONFIGS / "fight_result_corrections.json"
    if not path.exists():
        return out
    registry = json.loads(path.read_text())
    corrections = {}
    excluded = set()
    for row in registry["changes"]:
        a, b = normalise(row["red"]), normalise(row["blue"])
        corrections[(row["date"], a, b)] = row["corrected"]
        corrections[(row["date"], b, a)] = {"Red": "Blue", "Blue": "Red"}.get(
            row["corrected"], row["corrected"]
        )
    for row in registry["excluded"]:
        excluded.add((row["date"], frozenset((normalise(row["red"]), normalise(row["blue"])))))
    keep = []
    for idx, row in out.iterrows():
        date = str(pd.Timestamp(row["date"]).date())
        a, b = normalise(row["R_fighter"]), normalise(row["B_fighter"])
        keep.append((date, frozenset((a, b))) not in excluded)
        if (date, a, b) in corrections:
            out.at[idx, "Winner"] = corrections[(date, a, b)]
    return out.loc[keep].copy()
