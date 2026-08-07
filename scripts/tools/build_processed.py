"""Build processed Parquet files from raw Kaggle CSVs.

Usage:
    python scripts/build_processed.py
"""

from ufc_pred.ingest.kaggle_mdabbert import build

if __name__ == "__main__":
    stats = build()
    print("Wrote processed Parquet files:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
