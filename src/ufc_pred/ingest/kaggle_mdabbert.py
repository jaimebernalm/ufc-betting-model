"""Loader for the mdabbert/ultimate-ufc-dataset Kaggle source."""

from __future__ import annotations

import pandas as pd

from ufc_pred.paths import PROCESSED, RAW

SOURCE_DIR = RAW / "kaggle_mdabbert_ultimate_ufc"
HISTORY_CSV = SOURCE_DIR / "ufc-master.csv"
UPCOMING_CSV = SOURCE_DIR / "upcoming.csv"

HISTORY_PARQUET = PROCESSED / "fights.parquet"
UPCOMING_PARQUET = PROCESSED / "upcoming.parquet"


def load_history() -> pd.DataFrame:
    df = pd.read_csv(HISTORY_CSV, parse_dates=["date"])
    return _clean(df)


def load_upcoming() -> pd.DataFrame:
    df = pd.read_csv(UPCOMING_CSV, parse_dates=["date"])
    return _clean(df, require_winner=False)


def _clean(df: pd.DataFrame, *, require_winner: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df = df.dropna(subset=["date", "R_fighter", "B_fighter"])

    if require_winner:
        df = df[df["Winner"].isin(["Red", "Blue", "Draw"])]

    for col in ("R_fighter", "B_fighter"):
        df[col] = df[col].str.strip()

    df = df.sort_values("date").reset_index(drop=True)
    return df


def build() -> dict[str, int]:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    history = load_history()
    upcoming = load_upcoming()

    history.to_parquet(HISTORY_PARQUET, index=False)
    upcoming.to_parquet(UPCOMING_PARQUET, index=False)

    return {
        "history_rows": len(history),
        "upcoming_rows": len(upcoming),
        "history_first_date": str(history["date"].min().date()),
        "history_last_date": str(history["date"].max().date()),
    }


if __name__ == "__main__":
    stats = build()
    for k, v in stats.items():
        print(f"{k}: {v}")
