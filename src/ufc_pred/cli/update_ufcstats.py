"""CLI: scrape any UFC events newer than the current ufc-master.csv cutoff."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from ufc_pred.ingest.ufcstats_update import update_master


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--since", help="Override cutoff date (YYYY-MM-DD). Default: max date in ufc-master.csv")
    p.add_argument("--dry-run", action="store_true", help="List events to scrape without modifying the CSV.")
    args = p.parse_args()

    since = datetime.strptime(args.since, "%Y-%m-%d") if args.since else None
    summary = update_master(since=since, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
