"""One pre-fight feature constructor for historical training and shadow inference.

Immutable raw bout results/stats are accumulated by stable identity and day.
A bout contributes only starting the following day. Profile metadata is frozen
with its source hash; its historical publication timestamp is unavailable.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_pred.ingest.identity import PROFILE_URLS, canonical, canonicalize_frame, normalise
from ufc_pred.ingest.rankings_attach import attach_ranks

RATES = ["avg_SIG_STR_landed", "avg_SIG_STR_pct", "avg_TD_landed", "avg_TD_pct", "avg_SUB_ATT"]
COUNTERS = [
    "current_lose_streak",
    "current_win_streak",
    "draw",
    "longest_win_streak",
    "losses",
    "total_rounds_fought",
    "total_title_bouts",
    "win_by_Decision_Majority",
    "win_by_Decision_Split",
    "win_by_Decision_Unanimous",
    "win_by_KO/TKO",
    "win_by_Submission",
    "win_by_TKO_Doctor_Stoppage",
    "wins",
]
METHODS = dict(
    zip(
        [
            "Decision - Majority",
            "Decision - Split",
            "Decision - Unanimous",
            "KO/TKO",
            "Submission",
            "TKO - Doctor's Stoppage",
        ],
        COUNTERS[7:13],
        strict=True,
    )
)

METHODS.update(
    dict(
        zip(
            ["M-DEC", "S-DEC", "U-DEC", "SUB"],
            [COUNTERS[7], COUNTERS[8], COUNTERS[9], COUNTERS[11]],
            strict=True,
        )
    )
)


def _measurement(value, kind):
    text = str(value)
    if kind == "height":
        m = re.match(r"(\d+)'\s*(\d+)", text)
        return round((int(m[1]) * 12 + int(m[2])) * 2.54, 2) if m else np.nan
    m = re.search(r"\d+", text)
    return round(int(m[0]) * (2.54 if kind == "reach" else 1), 2) if m else np.nan


class RawHistoryStateSource:
    def __init__(self, source_dir: Path):
        self.source_dir = Path(source_dir)
        events = pd.read_csv(self.source_dir / "ufc_event_details.csv")
        events["EVENT"] = events.EVENT.str.strip()
        events["date"] = pd.to_datetime(events.DATE)
        results = pd.read_csv(self.source_dir / "ufc_fight_results.csv")
        results["EVENT"] = results.EVENT.str.strip()
        results["BOUT"] = results.BOUT.str.strip()
        results["METHOD"] = results.METHOD.str.strip()
        results = results.merge(events[["EVENT", "date"]], on="EVENT", validate="many_to_one")
        self.last_event = results.date.max()
        stats = pd.read_csv(self.source_dir / "ufc_fight_stats.csv")
        for col in ["EVENT", "BOUT", "FIGHTER"]:
            stats[col] = stats[col].str.strip()
        for col, prefix in [("SIG.STR.", "sig"), ("TD", "td")]:
            pairs = stats[col].str.extract(r"(\d+)\s+of\s+(\d+)").astype(float).fillna(0)
            stats[prefix + "_landed"], stats[prefix + "_attempted"] = pairs[0], pairs[1]
        stats["sub_attempted"] = pd.to_numeric(stats["SUB.ATT"], errors="coerce").fillna(0)
        totals = ["sig_landed", "sig_attempted", "td_landed", "td_attempted", "sub_attempted"]
        aggregate = stats.groupby(["EVENT", "BOUT", "FIGHTER"])[totals].sum()
        self.profiles = {}
        profile_rows = pd.read_csv(self.source_dir / "ufc_fighter_tott.csv")
        for r in profile_rows.itertuples(index=False):
            name = canonical(r.FIGHTER, r.WEIGHT)
            if name in PROFILE_URLS and PROFILE_URLS[name] != r.URL:
                continue
            key = normalise(name)
            profile = {
                "Stance": r.STANCE,
                "Height_cms": _measurement(r.HEIGHT, "height"),
                "Reach_cms": _measurement(r.REACH, "reach"),
                "Weight_lbs": _measurement(r.WEIGHT, "weight"),
                "DOB": pd.to_datetime(r.DOB, errors="coerce"),
                "URL": r.URL,
            }
            if pd.isna(profile["Reach_cms"]):
                profile["Reach_cms"] = profile["Height_cms"]
            if key in self.profiles and self.profiles[key]["URL"] != r.URL:
                profile = {"ambiguous": True, "URL": None}
            self.profiles[key] = profile
        supplement = self.source_dir.parent / "ufcstats_verified_profiles/profiles.json"
        if supplement.exists():
            for name, row in json.loads(supplement.read_text()).items():
                key = normalise(canonical(name))
                self.profiles[key] = {
                    "Stance": row["STANCE"],
                    "Height_cms": _measurement(row["HEIGHT"], "height"),
                    "Reach_cms": _measurement(row["REACH"], "reach"),
                    "Weight_lbs": _measurement(row["WEIGHT"], "weight"),
                    "DOB": pd.to_datetime(row["DOB"], errors="coerce"),
                    "URL": row["URL"],
                }
                if pd.isna(self.profiles[key]["Reach_cms"]):
                    self.profiles[key]["Reach_cms"] = self.profiles[key]["Height_cms"]
        entries = []
        for r in results.itertuples(index=False):
            names = re.split(r"\s+vs\.?\s+", r.BOUT, maxsplit=1)
            outcomes = str(r.OUTCOME).split("/")
            if len(names) != 2 or len(outcomes) != 2:
                continue
            try:
                mins, secs = map(int, str(r.TIME).split(":"))
                seconds = (int(r.ROUND) - 1) * 300 + mins * 60 + secs
            except (ValueError, TypeError):
                seconds = 0
            for name, outcome in zip(names, outcomes, strict=True):
                key = normalise(canonical(name, r.WEIGHTCLASS))
                vals = (
                    aggregate.loc[(r.EVENT, r.BOUT, name)].to_dict()
                    if (r.EVENT, r.BOUT, name) in aggregate.index
                    else dict.fromkeys(totals, 0.0)
                )
                has_stats = (r.EVENT, r.BOUT, name) in aggregate.index
                entries.append(
                    dict(
                        key=key,
                        date=r.date,
                        outcome=outcome.strip(),
                        rounds=r.ROUND,
                        method=r.METHOD,
                        title="title" in str(r.WEIGHTCLASS).lower(),
                        seconds=seconds if has_stats else 0,
                        **vals,
                    )
                )
        self.entries = entries
        self._build_states(entries)

    def _build_states(self, entries):
        """Shared chronological accumulation for frozen and newly observed bouts."""
        totals = ["sig_landed", "sig_attempted", "td_landed", "td_attempted", "sub_attempted"]
        self.states = {}
        self.first_date = {}
        if not entries:
            return
        for key, group in pd.DataFrame(entries).sort_values("date", kind="stable").groupby("key", sort=False):
            state = dict.fromkeys(COUNTERS, 0)
            cumulative = dict.fromkeys(totals + ["seconds"], 0.0)
            running_win = 0
            dates, states = [], []
            self.first_date[key] = group.date.min()
            for date, day in group.groupby("date", sort=True):
                for r in day.itertuples(index=False):
                    for col in cumulative:
                        cumulative[col] += getattr(r, col)
                    state["total_rounds_fought"] += int(r.rounds) if pd.notna(r.rounds) else 0
                    state["total_title_bouts"] += int(r.title)
                    if r.outcome == "W":
                        state["wins"] += 1
                        state["current_win_streak"] += 1
                        state["current_lose_streak"] = 0
                        running_win += 1
                        state["longest_win_streak"] = max(state["longest_win_streak"], running_win)
                        if r.method in METHODS:
                            state[METHODS[r.method]] += 1
                    elif r.outcome == "L":
                        state["losses"] += 1
                        state["current_lose_streak"] += 1
                        state["current_win_streak"] = 0
                        running_win = 0
                    elif r.outcome == "D":
                        state["draw"] += 1
                        running_win = 0
                sec = cumulative["seconds"]
                values = [
                    round(cumulative["sig_landed"] * 60 / sec, 2) if sec else 0,
                    round(cumulative["sig_landed"] / cumulative["sig_attempted"], 2)
                    if cumulative["sig_attempted"]
                    else 0,
                    round(cumulative["td_landed"] * 900 / sec, 2) if sec else 0,
                    round(cumulative["td_landed"] / cumulative["td_attempted"], 2)
                    if cumulative["td_attempted"]
                    else 0,
                    round(cumulative["sub_attempted"] * 900 / sec, 1) if sec else 0,
                ]
                dates.append(date + pd.Timedelta(days=1))
                states.append({**state, **dict(zip(RATES, values, strict=True))})
            self.states[key] = (pd.DatetimeIndex(dates), states)

    def __call__(self, fighter, fight_date):
        key = normalise(fighter)
        date = pd.Timestamp(fight_date).normalize()
        if key not in self.profiles:
            raise ValueError(f"Missing verified raw profile: {fighter}")
        profile = self.profiles[key]
        if profile.get("ambiguous"):
            raise ValueError(f"Unresolved profile homonym: {fighter}")
        state = dict.fromkeys(COUNTERS + RATES, 0.0)
        if key in self.states:
            dates, values = self.states[key]
            i = dates.searchsorted(date, side="right") - 1
            if i >= 0:
                state.update(values[i])
        for col in ["Stance", "Height_cms", "Reach_cms", "Weight_lbs"]:
            state[col] = profile[col]
        dob = profile["DOB"]
        state["age"] = (
            np.nan if pd.isna(dob) else date.year - dob.year - ((date.month, date.day) < (dob.month, dob.day))
        )
        return state


def apply_state_features(fights, state_source, rankings):
    """Used by both build_upcoming_row and the historical rebuild."""
    from ufc_pred.inference.upcoming_builder import _DIFF_PAIRS

    out = canonicalize_frame(fights).reset_index(drop=True)
    out["date"] = pd.to_datetime(out.date)
    for side in ["R", "B"]:
        states = pd.DataFrame(
            [state_source(n, d) for n, d in zip(out[f"{side}_fighter"], out.date, strict=True)]
        )
        for c in states:
            out[f"{side}_{c}"] = states[c].to_numpy()
    for col, base in _DIFF_PAIRS:
        out[col] = pd.to_numeric(out[f"B_{base}"], errors="coerce") - pd.to_numeric(
            out[f"R_{base}"], errors="coerce"
        )
    out = attach_ranks(out, rankings=rankings)
    out["date"] = pd.to_datetime(out.date)
    return out
