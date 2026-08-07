"""Match Kalshi fighter names to canonical UFCstats names from fights.parquet.

Kalshi uses display names (e.g. "Belal Muhammad", "Bryce Mitchell"). The
UFCstats source uses the same convention most of the time, but there are
edge cases: accented characters, hyphenated names, "Jr.", nickname variants.

We do:
  1. Fast exact + ASCII-folded exact lookup
  2. Fuzzy match scoped to fighters who were active near the fight date
     (±90 days), which dramatically reduces false matches vs global search
  3. Confidence score returned so callers can drop low-confidence rows
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz, process


@dataclass
class NameMatch:
    kalshi_name: str
    canonical: str | None
    score: float  # 0-100; 100 = exact
    method: str  # "exact" | "ascii_exact" | "fuzzy_local" | "fuzzy_global" | "none"


def _normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


def build_name_index(fights: pd.DataFrame) -> dict:
    """Build lookup structures from fights.parquet once, reuse for many matches."""
    all_names = pd.unique(pd.concat([fights["R_fighter"], fights["B_fighter"]]).dropna())
    norm_to_canon: dict[str, str] = {}
    for n in all_names:
        norm_to_canon.setdefault(_normalize(n), n)
    fights_local = fights[["date", "R_fighter", "B_fighter"]].copy()
    fights_local["date"] = pd.to_datetime(fights_local["date"])
    return {
        "all_names": list(all_names),
        "norm_to_canon": norm_to_canon,
        "fights_local": fights_local,
    }


def match_name(
    name: str,
    fight_date: pd.Timestamp | None,
    index: dict,
    *,
    local_window_days: int = 90,
    min_score_local: float = 88.0,
    min_score_global: float = 92.0,
) -> NameMatch:
    if not name:
        return NameMatch(name, None, 0.0, "none")
    # 1. exact
    if name in index["norm_to_canon"].values():
        return NameMatch(name, name, 100.0, "exact")
    # 2. ASCII-folded exact
    canon = index["norm_to_canon"].get(_normalize(name))
    if canon:
        return NameMatch(name, canon, 100.0, "ascii_exact")
    # 3. fuzzy, scoped to recently-active fighters
    if fight_date is not None:
        fd = pd.Timestamp(fight_date)
        if fd.tzinfo is not None:
            fd = fd.tz_convert("UTC").tz_localize(None)
        fl = index["fights_local"]
        window = fl[
            (fl["date"] >= fd - pd.Timedelta(days=local_window_days))
            & (fl["date"] <= fd + pd.Timedelta(days=local_window_days))
        ]
        local_pool = pd.unique(pd.concat([window["R_fighter"], window["B_fighter"]]).dropna()).tolist()
        if local_pool:
            best = process.extractOne(name, local_pool, scorer=fuzz.WRatio)
            if best and best[1] >= min_score_local and _surname_ok(name, best[0]):
                return NameMatch(name, best[0], float(best[1]), "fuzzy_local")
    # 4. fuzzy global (slow; only as fallback)
    best = process.extractOne(name, index["all_names"], scorer=fuzz.WRatio)
    if best and best[1] >= min_score_global and _surname_ok(name, best[0]):
        return NameMatch(name, best[0], float(best[1]), "fuzzy_global")
    return NameMatch(name, None, float(best[1]) if best else 0.0, "none")


def _surname_ok(a: str, b: str, min_surname_score: float = 90.0) -> bool:
    """Guard against 'Aline Pereira' -> 'Alice Pereira' style false positives.

    Require the LAST token of each name to be very similar, regardless of how
    similar the full strings are. Surname is the discriminating signal in MMA
    naming; mismatched first names with shared surname are a different person.
    """
    sa, sb = _normalize(a).split(), _normalize(b).split()
    if not sa or not sb:
        return True
    surname_score = fuzz.ratio(sa[-1], sb[-1])
    if surname_score < min_surname_score:
        return False
    # Also: first-name initial must agree if both are present.
    if len(sa) > 1 and len(sb) > 1 and sa[0][:1] != sb[0][:1]:
        if fuzz.ratio(sa[0], sb[0]) < 75:
            return False
    return True


def match_card(
    pairs: list[tuple[str, str, pd.Timestamp]],
    fights: pd.DataFrame,
) -> pd.DataFrame:
    """Convenience: match many (fighter_a, fighter_b, fight_date) triples."""
    idx = build_name_index(fights)
    rows = []
    for a, b, d in pairs:
        ma = match_name(a, d, idx)
        mb = match_name(b, d, idx)
        rows.append(
            {
                "fighter_a_kalshi": a,
                "fighter_a_canon": ma.canonical,
                "score_a": ma.score,
                "method_a": ma.method,
                "fighter_b_kalshi": b,
                "fighter_b_canon": mb.canonical,
                "score_b": mb.score,
                "method_b": mb.method,
                "fight_date": d,
            }
        )
    return pd.DataFrame(rows)
