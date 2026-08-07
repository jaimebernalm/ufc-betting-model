"""Command-line entry points, installed as console scripts by `pip install -e .`.

    ufc-preview    read-only preview of the next Kalshi card
    ufc-predict    score + size a full card (Polymarket or Kalshi)
    ufc-runner     per-fight T-90min watchdog (notifies; never places orders)
    ufc-update     refresh the UFCStats fight history
    ufc-bankrolls  reconcile the account ledger against settled markets

Exploratory and maintenance scripts live outside the package, in `scripts/`.
"""
