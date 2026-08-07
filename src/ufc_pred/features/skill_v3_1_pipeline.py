"""Walk-forward pipeline for v3.1 time-varying skill features.

Mirrors skill_v3_pipeline but uses the random-walk model. Writes
data/processed/skill_features_v3_1.parquet.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ufc_pred.features.skill_v3 import build_index
from ufc_pred.features.skill_v3_1 import (
    encode_careers,
    fit_nuts,
    skill_diff_for_target_fights,
)
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import PROCESSED

OUTPUT = PROCESSED / "skill_features_v3_1.parquet"
LOG = PROCESSED / "skill_features_v3_1.log.csv"


def month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def build(
    output_path: Path = OUTPUT,
    log_path: Path = LOG,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    max_career_len_cap: int = 30,
    min_train_fights: int = 50,
    shard_id: int = 0,
    n_shards: int = 1,
    progress_bar: bool = True,
) -> dict:
    """Build skill features.

    For parallel runs, set `n_shards > 1` and call N times with shard_id = 0..N-1.
    Each shard processes a contiguous block of months and writes its own parquet
    (suffixed with `.shard{shard_id}`). Use `merge_shards()` to combine them.
    """
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    fights = fights.sort_values("date").reset_index(drop=True)

    index = build_index(fights)
    all_months = sorted({month_floor(d) for d in fights["date"]})

    # Sharding: split months into n_shards contiguous blocks; this shard takes
    # block `shard_id`. Contiguous (not interleaved) keeps locality of the
    # `fights` slice cheap.
    if n_shards > 1:
        import math

        block = math.ceil(len(all_months) / n_shards)
        months = all_months[shard_id * block : (shard_id + 1) * block]
        output_path = output_path.with_suffix(f".shard{shard_id}.parquet")
        log_path = log_path.with_suffix(f".shard{shard_id}.csv")
    else:
        months = all_months

    # Resume: keep any months already saved in this shard's parquet and skip
    # them in the loop below.
    out_frames: list[pd.DataFrame] = []
    fit_log: list[dict] = []
    if output_path.exists():
        prior_df = pd.read_parquet(output_path)
        prior_df["date"] = pd.to_datetime(prior_df["date"])
        done_months = {month_floor(d) for d in prior_df["date"]}
        if done_months:
            out_frames.append(prior_df)
            months = [m for m in months if m not in done_months]
            print(
                f"[shard {shard_id}] resuming: {len(done_months)} months already in parquet; "
                f"{len(months)} remaining",
                flush=True,
            )
            if log_path.exists():
                fit_log = pd.read_csv(log_path).to_dict("records")

    overall_t0 = time.time()
    iterator = tqdm(months, desc=f"v3.1 shard {shard_id}/{n_shards}") if progress_bar else months
    for m_start in iterator:
        prior_mask = fights["date"] < m_start
        target_mask = (fights["date"] >= m_start) & (fights["date"] < m_start + pd.offsets.MonthBegin(1))
        target = fights.loc[target_mask]
        if len(target) == 0:
            continue

        prior = fights.loc[prior_mask]
        n_prior = len(prior)
        if n_prior < min_train_fights:
            nan_df = pd.DataFrame(
                {
                    "date": target["date"].to_numpy(),
                    "R_fighter": target["R_fighter"].to_numpy(),
                    "B_fighter": target["B_fighter"].to_numpy(),
                    "skill_diff_mean": float("nan"),
                    "skill_diff_std": float("nan"),
                }
            )
            out_frames.append(nan_df)
            fit_log.append({"month": str(m_start.date()), "n_prior": n_prior, "fit": False, "secs": 0.0})
            continue

        t0 = time.time()
        enc = encode_careers(prior, index, max_career_len_cap=max_career_len_cap)
        samples = fit_nuts(
            enc,
            index,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False,
            seed=int(m_start.year) * 100 + int(m_start.month),
        )
        feats = skill_diff_for_target_fights(target, samples, enc, index)
        out_frames.append(feats)
        secs = time.time() - t0
        fit_log.append({"month": str(m_start.date()), "n_prior": n_prior, "fit": True, "secs": secs})

        # Checkpoint partial output every 12 months so an interrupted run
        # leaves recoverable progress on disk.
        if len(out_frames) % 12 == 0:
            pd.concat(out_frames, ignore_index=True).to_parquet(output_path, index=False)
            pd.DataFrame(fit_log).to_csv(log_path, index=False)

    result = pd.concat(out_frames, ignore_index=True)
    result.to_parquet(output_path, index=False)
    pd.DataFrame(fit_log).to_csv(log_path, index=False)

    total_secs = time.time() - overall_t0
    return {
        "rows": int(len(result)),
        "n_months_fit": int(sum(1 for r in fit_log if r["fit"])),
        "n_months_skipped": int(sum(1 for r in fit_log if not r["fit"])),
        "total_minutes": round(total_secs / 60, 1),
        "output": str(output_path),
        "log": str(log_path),
    }


def merge_shards(
    n_shards: int,
    output_path: Path = OUTPUT,
    log_path: Path = LOG,
    remove_shards: bool = True,
) -> dict:
    """Combine per-shard parquets into the canonical OUTPUT parquet."""
    frames, logs = [], []
    for shard_id in range(n_shards):
        sp = output_path.with_suffix(f".shard{shard_id}.parquet")
        lp = log_path.with_suffix(f".shard{shard_id}.csv")
        if sp.exists():
            frames.append(pd.read_parquet(sp))
        if lp.exists():
            logs.append(pd.read_csv(lp))
    if not frames:
        raise FileNotFoundError(f"No shard parquets found at {output_path.parent}")

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "R_fighter", "B_fighter"])
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_parquet(output_path, index=False)

    if logs:
        merged_log = pd.concat(logs, ignore_index=True).sort_values("month")
        merged_log.to_csv(log_path, index=False)

    if remove_shards:
        for shard_id in range(n_shards):
            output_path.with_suffix(f".shard{shard_id}.parquet").unlink(missing_ok=True)
            log_path.with_suffix(f".shard{shard_id}.csv").unlink(missing_ok=True)

    return {"rows": int(len(merged)), "output": str(output_path), "log": str(log_path)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all .shard*.parquet files into the canonical output and exit.",
    )
    args = parser.parse_args()

    if args.merge:
        stats = merge_shards(args.n_shards)
    else:
        stats = build(
            shard_id=args.shard_id,
            n_shards=args.n_shards,
            progress_bar=not args.no_progress,
        )
    for k, v in stats.items():
        print(f"{k}: {v}")
