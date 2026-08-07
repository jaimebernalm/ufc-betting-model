"""Event-triggered watchdog: discover tonight's UFC card, fetch live Kalshi
orderbooks, run model + Kelly sizing, send macOS notifications and persist
JSON records. Notify-only — does NOT place real orders.

Capture triggers (watchdog mode), per fight, evaluated each tick:
  first_fight    : the card opener has no predecessor — capture at T-60min.
  prev_resolved  : the PREVIOUS fight's Kalshi market has resolved — settled,
                   trading closed post-fight-start (current Kalshi regime:
                   close at fight end, settle ~5-8 min later), or price pinned
                   to ~0.99/0.01 and stable 90s+ (legacy regime). Capture
                   immediately: bankrolls were synced at tick start, so Kelly
                   sizes off a bankroll that includes every officially-resolved
                   fight (worst case the just-ended fight books ~6 min later).
  fallback_time  : the previous fight still isn't resolved by T-25min before
                   this fight's scheduled start (long fight / schedule drift)
                   — capture anyway so the bet isn't missed. Tagged [LATE].
                   SUPPRESSED while the previous fight's Kalshi market is still
                   openly trading (status 'active'): that means the fight hasn't
                   started and the card is simply running behind schedule (e.g.
                   weather delay), so firing off the stale scheduled clock would
                   place a blind bet against a frozen bankroll. In that case we
                   wait for prev_resolved instead.

prev_resolved is evaluated BEFORE fallback_time so a real resolution always
wins; the scheduled-clock fallback only fires when the prior market is no
longer active yet still can't be confirmed resolved.

A fight is never captured earlier than T-90min (price quality, nb09). The
"too late / stale" cut-off is delay-proof: a fight is skipped as stale only
once it has ITSELF resolved (we genuinely missed the window), not merely
because its scheduled start + grace elapsed — on a delayed card the real start
can be long after the schedule.

Usage:
    # End-to-end smoke test: predict every fight on tonight's card RIGHT NOW
    # regardless of how far away each fight is. Use this first.
    PYTHONPATH=src .conda/bin/python scripts/bet_runner.py --once --dry-run

    # Same, but writes to the idempotency log (so re-running suppresses dups)
    PYTHONPATH=src .conda/bin/python scripts/bet_runner.py --once

    # Watchdog mode (event-triggered, see above)
    PYTHONPATH=src .conda/bin/python scripts/bet_runner.py --watchdog

    # Time-shift NOW for watchdog timing tests
    PYTHONPATH=src .conda/bin/python scripts/bet_runner.py --watchdog --time-offset-min -90 --dry-run

The pipeline:
  1. Scrape today's UFC card via ufc_schedule (per-fight start times)
  2. Sync bankrolls from resolved fights (settled OR provisionally pinned)
  3. Pull live Kalshi orderbooks via kalshi_card.fetch_next_card
  4. Match each UFC fight to a Kalshi market by name (surname + accent-folded)
  5. For each fight whose trigger fired:
       - run predict_one_fight() and Kelly sizing off the synced bankroll
       - send macOS notification
       - write JSON record to data/processed/bet_notifications/<date>_<key>.json
       - log to idempotency file (skipped under --dry-run)
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ufc_pred.paths import ROOT

load_dotenv(ROOT / ".env")

from ufc_pred.inference import predict_one_fight
from ufc_pred.inference.kalshi_card import CardFight, fetch_next_card
from ufc_pred.ingest.kalshi_client import KalshiClient
from ufc_pred.ingest.kalshi_resolution import market_resolution
from ufc_pred.ingest.rankings_attach import load_rankings
from ufc_pred.ingest.ufc_schedule import (
    FightSchedule,
    fetch_card_results,
    fetch_card_schedule,
    fetch_today_event,
    write_audit_log,
)
from ufc_pred.ingest.ufcstats_state import UFCStatsStateSource
from ufc_pred.notify import notify
from ufc_pred.ops.bankroll import sync as sync_bankrolls
from ufc_pred.ops.fills import save_fills
from ufc_pred.paths import CONFIGS, PROCESSED

FIGHTS_PATH = PROCESSED / "fights.parquet"
BANKROLLS_PATH = CONFIGS / "bankrolls.json"
NOTIF_DIR = PROCESSED / "bet_notifications"
IDEMPOTENCY_PATH = PROCESSED / "bet_runner_idempotency.json"

EARLIEST_CAPTURE_MIN = 90  # never capture earlier than T-90min (nb09 price quality)
FIRST_FIGHT_CAPTURE_MIN = 60  # card opener fires at T-60 (no prior fight to wait for)
FALLBACK_BEFORE_START_MIN = 25  # capture by T-25min even if prev fight unresolved
STALE_AFTER_START_MIN = 15  # past scheduled start + this -> too late, skip
MAX_HISTORY_AGE_DAYS = 14  # refuse to bet if fights.parquet is older than this


# ---------------------------------------------------------------------------
# Name matching: UFC.com FightSchedule -> Kalshi CardFight
# ---------------------------------------------------------------------------


def _fold(s: str) -> str:
    """Lowercase + strip accents for loose matching."""
    if not s:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c)
    ).strip()


# Generational suffixes are not part of the surname and are inconsistently
# present across sources (UFC.com lists "Michael Aswell Jr."; Kalshi lists
# "Michael Aswell"). Drop them before taking the last token, else the surname
# resolves to "jr." and the fight fails to match its Kalshi market.
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

# Full-name overrides (folded/lowercased keys) for fighters whose Kalshi listing
# uses a long legal name whose LAST token is not the surname UFC.com uses, so
# last-token extraction picks the wrong word (e.g. "...Prestes De Matos" -> the
# matcher needs "oliveira", not "matos"). Suffix-stripping can't fix these.
_FULLNAME_ALIASES = {
    "vinicius de oliveira prestes de matos": "vinicius oliveira",
}


def _surname(s: str) -> str:
    folded = _FULLNAME_ALIASES.get(_fold(s), _fold(s))
    # Hyphenated surnames are inconsistently hyphenated across sources
    # (UFC.com "Benoît Saint Denis" vs Kalshi "Benoit Saint-Denis") — split
    # on hyphens too so both sides resolve to the same last token.
    folded = folded.replace("-", " ")
    parts = [p for p in folded.split() if p.rstrip(".") not in _NAME_SUFFIXES]
    return parts[-1] if parts else ""


def match_kalshi(
    fight: FightSchedule,
    kalshi_card: list[CardFight],
) -> tuple[CardFight, bool] | None:
    """Find the Kalshi CardFight that corresponds to `fight`.

    Returns (kalshi_fight, swap) where `swap` is True if Kalshi's fighter_a
    matches UFC's fighter_b (because Kalshi sometimes flips red/blue order).
    Callers must remap names if swap=True so notification text stays correct.
    """
    ufc_a_surname = _surname(fight.fighter_a)
    ufc_b_surname = _surname(fight.fighter_b)
    for kal in kalshi_card:
        kal_a_surname = _surname(kal.fighter_a)
        kal_b_surname = _surname(kal.fighter_b)
        if {ufc_a_surname, ufc_b_surname} == {kal_a_surname, kal_b_surname}:
            swap = kal_a_surname == ufc_b_surname
            return kal, swap
    return None


def make_fight_key(date: pd.Timestamp, fight: FightSchedule) -> str:
    """Stable idempotency key: sorted surnames so A/B order doesn't matter."""
    surnames = sorted([_surname(fight.fighter_a).title(), _surname(fight.fighter_b).title()])
    return f"{date.strftime('%Y-%m-%d')}_{surnames[0]}_{surnames[1]}"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def load_idempotency() -> dict:
    if not IDEMPOTENCY_PATH.exists():
        return {}
    try:
        return json.loads(IDEMPOTENCY_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def save_idempotency(log: dict) -> None:
    IDEMPOTENCY_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDEMPOTENCY_PATH.write_text(json.dumps(log, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# JSON record
# ---------------------------------------------------------------------------


def write_record(date: pd.Timestamp, key: str, payload: dict) -> Path:
    NOTIF_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTIF_DIR / f"{key}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# Notification formatting
# ---------------------------------------------------------------------------


def build_notification(
    fight: FightSchedule,
    kalshi: CardFight,
    result: dict,
    *,
    dry_run: bool,
    late: bool = False,
    cash_short: dict | None = None,
) -> tuple[str, str, str]:
    """Return (title, subtitle, message) tuple for the macOS notification.

    Title:    🥊 UFC Bet  (prefixed with [TEST] or [LATE] when applicable)
    Subtitle: <fighter_a> vs <fighter_b> — main event, T-90min
    Message:  $7.42 on Bonfim @ 50¢  (edge +37¢, p=0.87)
    """
    prefix = ""
    if dry_run:
        prefix += "[TEST] "
    if late:
        prefix += "[LATE] "
    title = f"{prefix}🥊 UFC Bet"

    label = {
        "main_card": "main card",
        "prelim": "prelim",
        "early_prelim": "early prelim",
    }.get(fight.card_position, "")
    local = fight.start_time_utc.tz_convert("US/Eastern")
    subtitle = f"{fight.fighter_a} vs {fight.fighter_b} — {label} at {local.strftime('%H:%M %Z')}"

    if result["status"] == "ok" and result["orders"]:
        order = next(iter(result["orders"].values()))
        msg = (
            f"BET ${order['total_stake_usd']:.2f} on {order['side_name']} "
            f"@ {order['limit_price'] * 100:.0f}¢ "
            f"(edge {order['edge_cents']:+.1f}¢)"
        )
        if cash_short:
            # Winnings still locked in unsettled positions: spendable cash is
            # below the recommended stake. Place what cash allows.
            msg += (
                f" ⚠️ CASH ${cash_short['cash']:.2f} < stake — "
                f"place up to cash, rest unfillable until settlement"
            )
    elif result["status"] == "skip":
        msg = f"NO BET — {result['error'] or 'no edge'}"
    else:
        msg = f"⚠️ ERROR — {result['error']}"

    return title, subtitle, msg


# ---------------------------------------------------------------------------
# Per-fight handler
# ---------------------------------------------------------------------------


def process_one(
    fight: FightSchedule,
    kalshi_card: list[CardFight],
    fights_df: pd.DataFrame,
    rankings: pd.DataFrame,
    bankrolls: dict[str, float],
    event: dict,
    *,
    dry_run: bool,
    now: pd.Timestamp,
    idempotency: dict,
    verbose: bool,
    late: bool = False,
    trigger: str = "",
    kalshi_client: KalshiClient | None = None,
    state_source: UFCStatsStateSource | None = None,
) -> dict:
    """Run one fight through the full pipeline. Returns a small status dict."""
    key = make_fight_key(event["date"], fight)

    if not dry_run and key in idempotency:
        if verbose:
            print(f"  [skip] {key}: already notified at {idempotency[key]}")
        return {"key": key, "result": "already_notified"}

    matched = match_kalshi(fight, kalshi_card)
    if matched is None:
        if verbose:
            print(f"  [no-market] {fight.fighter_a} vs {fight.fighter_b}")
        # Notify ONCE per fight, under a side key that does NOT block future
        # reprocessing: a market that lists late can still fire normally, but
        # a fight whose market already settled+delisted (schedule drift) won't
        # re-alert every tick.
        nm_key = f"{key}#no_market_notified"
        if not dry_run and nm_key not in idempotency:
            notify(
                "⚠️ UFC Bet — no market",
                f"{fight.fighter_a} vs {fight.fighter_b}",
                "Not on Kalshi (cancelled? already settled? not yet listed?)",
            )
            idempotency[nm_key] = now.isoformat()
            save_idempotency(idempotency)
        return {"key": key, "result": "no_market"}

    kalshi, swap = matched
    if verbose:
        local = fight.start_time_utc.tz_convert("US/Eastern")
        print(
            f"\n  ➤ {fight.fighter_a} vs {fight.fighter_b}  "
            f"({fight.card_position}, {local.strftime('%H:%M %Z')})"
        )
        print(f"    Kalshi: {kalshi.market_id}  (swap={swap})")

    result = predict_one_fight(
        kalshi,
        fights_df,
        rankings,
        bankrolls,
        weight_class_hint=fight.weight_class,
        title_bout=fight.title_bout,
        no_of_rounds=fight.no_of_rounds,
        state_source=state_source,
    )

    # Spendable-cash check: mid-card, winnings sit in unsettled positions, so
    # the ledger bankroll can exceed actual placeable cash. Warn, don't resize.
    cash_short = None
    if kalshi_client is not None and result["status"] == "ok" and result["orders"]:
        try:
            cash = kalshi_client.get_balance().get("balance", 0) / 100.0
            total_stake = sum(o["total_stake_usd"] for o in result["orders"].values())
            if total_stake > cash:
                cash_short = {"cash": round(cash, 2), "stake": round(total_stake, 2)}
                if verbose:
                    print(f"    ⚠️ cash ${cash:.2f} < recommended stake ${total_stake:.2f}")
        except Exception as e:
            print(f"  [cash-check] balance query failed: {e}", file=sys.stderr)

    payload = {
        "event": event["title"],
        "fight_date_utc": fight.start_time_utc.isoformat(),
        "captured_at_utc": now.isoformat(),
        "fight": {
            "fighter_a_ufc": fight.fighter_a,
            "fighter_b_ufc": fight.fighter_b,
            "card_position": fight.card_position,
            "weight_class": fight.weight_class,
            "kalshi_event": kalshi.market_id,
            "kalshi_ticker_a": kalshi.token_a,
            "kalshi_ticker_b": kalshi.token_b,
            "kalshi_swap_vs_ufc": swap,
        },
        "kalshi_snapshot": {
            "a_ask": result["market"]["best_ask_a"],
            "b_ask": result["market"]["best_ask_b"],
            "a_bid": result["market"]["best_bid_a"],
            "b_bid": result["market"]["best_bid_b"],
            "volume": result["market"]["volume"],
            "liquidity": result["market"]["liquidity"],
        },
        "model": result["model"],
        "recommendation": {
            "status": result["status"],
            "error": result["error"],
            "summary_line": result["summary_line"],
            "orders": result["orders"],
            "per_account_recommendations": result["recommendations"],
        },
        "bankrolls_at_capture": bankrolls,
        "dry_run": dry_run,
        "late_recovery": late,
        "capture_trigger": trigger,
        "cash_check": cash_short,
    }

    record_path = write_record(event["date"], key, payload)
    if verbose:
        print(f"    → {result['summary_line']}")
        print(f"    record: {record_path.relative_to(ROOT)}")

    title, subtitle, msg = build_notification(
        fight, kalshi, result, dry_run=dry_run, late=late, cash_short=cash_short
    )
    notify(title, subtitle, msg)

    if not dry_run:
        idempotency[key] = now.isoformat()
        save_idempotency(idempotency)

    return {"key": key, "result": "notified", "summary": result["summary_line"]}


# ---------------------------------------------------------------------------
# Trigger classification (pure logic — testable without network)
# ---------------------------------------------------------------------------


def fight_trigger(
    i: int,
    fight: FightSchedule,
    ordered: list[FightSchedule],
    now: pd.Timestamp,
    *,
    prev_resolution,  # callable(prev_fight) -> Optional[(kind, yes_won)]
    prev_market_live,  # callable(prev_fight) -> bool (True = prev still trading)
    this_resolution,  # callable(fight) -> Optional[(kind, yes_won)]; THIS fight
) -> str | None:
    """Decide whether fight `i` should be captured NOW. Returns the trigger
    name or None. Caller has already filtered idempotency/window bounds."""
    start = fight.start_time_utc
    earliest = start - pd.Timedelta(minutes=EARLIEST_CAPTURE_MIN)
    if now < earliest:
        return None
    # Too-late guard, delay-proof: a fight is "stale" only once it has ITSELF
    # resolved (we genuinely missed the betting window) — NOT merely because its
    # scheduled start + grace elapsed. On a delayed card the real start can be
    # long after the schedule, so the scheduled clock cannot decide staleness.
    if this_resolution(fight) is not None:
        return None
    if i == 0:
        # Card opener: no prior fight to react to — fire at T-60.
        if now >= start - pd.Timedelta(minutes=FIRST_FIGHT_CAPTURE_MIN):
            return "first_fight"
        return None
    # Primary path — delay-proof: fire the instant the prior fight resolves.
    # Checked BEFORE the scheduled-clock fallback so a real resolution always
    # wins and a stale schedule can never preempt it.
    if prev_resolution(ordered[i - 1]) is not None:
        return "prev_resolved"
    # Prior fight unresolved. If its Kalshi market is still openly trading, the
    # card simply hasn't reached this fight yet (e.g. weather delay) — do NOT
    # fire a blind, frozen-bankroll bet off the scheduled clock; keep waiting.
    if prev_market_live(ordered[i - 1]):
        return None
    # Prior fight unresolved AND no longer openly trading (genuine detection
    # gap): the scheduled-clock fallback is the last resort so the bet isn't
    # missed. Tagged [LATE].
    if now >= start - pd.Timedelta(minutes=FALLBACK_BEFORE_START_MIN):
        return "fallback_time"
    return None


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def load_bankrolls() -> dict[str, float]:
    if not BANKROLLS_PATH.exists():
        return {"A": 300.0, "B": 300.0, "C": 300.0}
    d = json.loads(BANKROLLS_PATH.read_text())
    return {"A": float(d["A"]), "B": float(d["B"]), "C": float(d["C"])}


def _heartbeat(now: pd.Timestamp, msg: str) -> None:
    """One-line stamped log entry for every watchdog tick, regardless of verbose
    mode. Used by launchd-driven runs so the log shows the runner is alive."""
    print(f"[{now.isoformat()}] {msg}", flush=True)


def run(args: argparse.Namespace) -> int:
    now = pd.Timestamp.now(tz="UTC")
    if args.time_offset_min:
        now = now + pd.Timedelta(minutes=args.time_offset_min)
    verbose = args.verbose or args.once or args.dry_run

    if verbose:
        print(
            f"═══ bet_runner — {now.isoformat()}  "
            f"(mode: {'once' if args.once else 'watchdog'}"
            f"{', DRY-RUN' if args.dry_run else ''}) ═══"
        )

    event = fetch_today_event(now_utc=now)
    if event is None:
        _heartbeat(now, "tick: no UFC event today")
        return 0
    if verbose:
        print(f"Event: {event['title']}  ({event['url']})")

    schedule = fetch_card_schedule(
        event["url"],
        main_card_ts=event.get("main_card_ts"),
        prelims_ts=event.get("prelims_ts"),
    )
    if not schedule:
        print(f"⚠ schedule scrape returned 0 fights for {event['url']}")
        return 1

    if not args.dry_run:
        # Audit log — persists every successful scrape for replay
        write_audit_log(event["date"], event, schedule)

    # Live UFC.com results — fastest "fight is over" signal (winner badge
    # appears ~1-2 min after the result; Kalshi can lag by an hour). Fetched
    # fresh each tick (cache bypass); failure degrades to Kalshi signals.
    ufc_results = None
    try:
        ufc_results = fetch_card_results(event["url"])
    except Exception as e:
        print(f"  [results] UFC.com results fetch failed: {e}", file=sys.stderr)
    ufc_winners: dict[frozenset, str] = {
        frozenset({_surname(r.fighter_red), _surname(r.fighter_blue)}): r.winner
        for r in (ufc_results or [])
        if r.winner
    }

    # Auto-roll card_state to tonight's card. Rolling was a manual runbook
    # step and got skipped for three straight cards (06-20 → 07-11), leaving
    # sync a silent no-op and every bet sized on a frozen ledger. On a new
    # card there are no notifications for the new date yet, so clearing the
    # exclusion lists cannot double-book anything.
    if not args.dry_run:
        event_date_str = pd.Timestamp(event["date"]).strftime("%Y-%m-%d")
        try:
            bk = json.loads(BANKROLLS_PATH.read_text())
            st = bk.get("card_state", {})
            if st.get("card_date") != event_date_str:
                bk["card_state"] = {
                    "card_date": event_date_str,
                    "baseline_set_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "fights_excluded": [],
                    "fights_provisional": {},
                }
                BANKROLLS_PATH.write_text(json.dumps(bk, indent=2))
                print(f"  [sync] rolled card_state {st.get('card_date')} -> {event_date_str}")
        except Exception as e:
            print(f"  [sync] warning: card_state roll failed: {e}", file=sys.stderr)

    # Sync per-account bankrolls from any newly-resolved fights BEFORE we
    # size new bets. Idempotent — already-applied fights are skipped.
    if not args.dry_run:
        try:
            sync_bankrolls(ufc_results=ufc_results, verbose=verbose)
        except Exception as e:
            print(f"  [sync] warning: bankroll sync failed: {e}", file=sys.stderr)
        # Snapshot actual fills every tick — Kalshi drops portfolio history
        # within days, and the fills store is the ground truth for what was
        # really bet (vs what the notification recommended).
        try:
            save_fills(verbose=verbose)
        except Exception as e:
            print(f"  [fills] warning: fills snapshot failed: {e}", file=sys.stderr)

    fights_df = pd.read_parquet(FIGHTS_PATH)
    # Freshness guard: predictions built on a stale fight history produce
    # fake edges (2026-07-11 postmortem: fights.parquet frozen at 05-16 →
    # model blind to 2 months of results, big-edge bets tracked the market,
    # −$561 over 5 cards). Refuse to fire rather than bet blind.
    # Two-stage so a legitimate >2-week UFC break doesn't false-positive:
    # only block if ufcstats.com confirms completed events are missing (or
    # the confirmation itself fails — no bet beats a blind bet).
    max_fight_date = pd.to_datetime(fights_df["date"]).max().tz_localize("UTC")
    staleness_days = (now - max_fight_date).days
    if staleness_days > MAX_HISTORY_AGE_DAYS:
        try:
            from ufc_pred.ingest.ufcstats_update import update_master

            missed = [
                e
                for e in update_master(dry_run=True).get("new_events", [])
                if pd.Timestamp(e["date"]).tz_localize("UTC") < now.normalize()
            ]
            detail = f"{len(missed)} completed event(s) missing"
        except Exception as e:
            missed = None
            detail = f"freshness check failed ({e})"
        if missed is None or missed:
            msg = (
                f"fights.parquet ends {max_fight_date.date()} "
                f"({staleness_days}d ago); {detail}. "
                f"Run update_ufcstats + build_processed, then restart."
            )
            print(f"✗ STALE DATA — {msg}", file=sys.stderr)
            # Notify once per day, not once per launchd tick.
            marker = PROCESSED / "stale_data_notified.txt"
            today = str(now.date())
            if not marker.exists() or marker.read_text().strip() != today:
                notify("⛔ bet_runner: STALE DATA — no bets", "", msg, sound="Basso")
                marker.write_text(today)
            _heartbeat(now, f"tick: BLOCKED stale data ({staleness_days}d)")
            return 1
    rankings = load_rankings()
    bankrolls = load_bankrolls()
    if verbose:
        print(f"Bankrolls: A=${bankrolls['A']:.0f} B=${bankrolls['B']:.0f} C=${bankrolls['C']:.0f}")

    kalshi_card = fetch_next_card(target_date=pd.Timestamp(event["date"]).tz_localize("UTC"))
    if not kalshi_card:
        # Fall back to soonest card (in case ticker date encoding differs)
        kalshi_card = fetch_next_card()
    if verbose:
        print(f"Kalshi: fetched {len(kalshi_card)} fights")

    idempotency = load_idempotency()

    try:
        api_client: KalshiClient | None = KalshiClient()
    except Exception as e:
        print(
            f"  [api] KalshiClient init failed (cash check / triggers degraded): {e}",
            file=sys.stderr,
        )
        api_client = None

    # Classify each scheduled fight (watchdog mode). A fight fires when it is
    # inside [T-90min, start+15min] AND one of its triggers is true:
    #   first_fight   : card opener (no predecessor) -> fire at T-90min
    #   prev_resolved : previous fight's market resolved (settled or pinned
    #                   stable >= 90s) -> fire NOW with a lag-0 bankroll
    #   fallback_time : previous fight unresolved but we're at T-25min ->
    #                   fire anyway so the bet isn't missed (tagged [LATE])
    # Idempotency makes firing once-only; `--once` bypasses all gating.
    eligible: list[tuple[FightSchedule, bool, str]] = []  # (fight, is_late, trigger)
    if args.once:
        eligible = [(f, False, "once") for f in schedule]
    else:
        ordered = sorted(schedule, key=lambda f: f.start_time_utc)
        resolution_client = api_client

        # Per-tick memo: each fight's resolution is probed both as "this fight"
        # (stale guard) and as the next fight's "prev" — cache to halve API load.
        _resolution_memo: dict[str, object] = {}

        def resolve_prev(prev: FightSchedule):
            memo_key = make_fight_key(event["date"], prev)
            if memo_key in _resolution_memo:
                return _resolution_memo[memo_key]
            res = _resolve_prev_uncached(prev)
            _resolution_memo[memo_key] = res
            return res

        def _resolve_prev_uncached(prev: FightSchedule):
            # 1) UFC.com winner badge — fastest signal.
            fkey = frozenset({_surname(prev.fighter_a), _surname(prev.fighter_b)})
            if fkey in ufc_winners:
                if verbose:
                    print(
                        f"  [trigger] {prev.fighter_a} vs {prev.fighter_b} "
                        f"resolved (ufc_result: {ufc_winners[fkey]})"
                    )
                return ("ufc_result", True)
            # 2) Kalshi market signals (settled / closed / legacy pin).
            prev_matched = match_kalshi(prev, kalshi_card)
            if prev_matched is None:
                return None  # can't poll; fallback_time covers it
            try:
                res = market_resolution(
                    resolution_client,
                    prev_matched[0].token_a,
                    fight_start_utc=prev.start_time_utc,
                    now_utc=now,
                )
            except Exception as e:
                print(
                    f"  [resolution] error polling {prev.fighter_a} vs {prev.fighter_b}: {e}",
                    file=sys.stderr,
                )
                return None
            if res is not None and verbose:
                print(f"  [trigger] {prev.fighter_a} vs {prev.fighter_b} resolved ({res[0]})")
            return res

        def prev_market_live(prev: FightSchedule) -> bool:
            """True if the previous fight's Kalshi market is still OPENLY
            trading (status 'active') — i.e. that fight has not started yet, so
            the card is running behind its schedule. Suppresses the scheduled-
            clock fallback during a delay. Conservative: returns False when the
            market can't be polled, so a genuine detection gap still lets the
            fallback fire."""
            if resolution_client is None:
                return False
            prev_matched = match_kalshi(prev, kalshi_card)
            if prev_matched is None:
                return False
            try:
                m = resolution_client.get_market(prev_matched[0].token_a)
            except Exception as e:
                print(
                    f"  [delay-gate] error polling {prev.fighter_a} vs {prev.fighter_b}: {e}",
                    file=sys.stderr,
                )
                return False
            m = m.get("market", m)
            live = m.get("status", "") == "active"
            if live and verbose:
                print(
                    f"  [delay-gate] {prev.fighter_a} vs {prev.fighter_b} "
                    f"still trading (active) — card delayed, holding fallback"
                )
            return live

        for i, f in enumerate(ordered):
            key = make_fight_key(event["date"], f)
            if key in idempotency:
                if verbose:
                    print(f"  [skip:notified] {f.fighter_a} vs {f.fighter_b}")
                continue
            trigger = fight_trigger(
                i,
                f,
                ordered,
                now,
                prev_resolution=resolve_prev,
                prev_market_live=prev_market_live,
                this_resolution=resolve_prev,
            )
            if trigger is not None:
                eligible.append((f, trigger == "fallback_time", trigger))
            elif verbose:
                start_l = f.start_time_utc.tz_convert("US/Eastern").strftime("%H:%M %Z")
                tag = (
                    "future"
                    if now < f.start_time_utc - pd.Timedelta(minutes=EARLIEST_CAPTURE_MIN)
                    else (
                        "stale"
                        if now > f.start_time_utc + pd.Timedelta(minutes=STALE_AFTER_START_MIN)
                        else "waiting:prev-fight"
                    )
                )
                print(f"  [{tag}] {f.fighter_a} vs {f.fighter_b} (start {start_l})")

    if verbose:
        print(f"Processing {len(eligible)}/{len(schedule)} fights")

    summary = []
    # Share one raw-state source across the card.  Its immutable fight-detail
    # cache avoids refetching common opponent histories for every prediction.
    state_source = UFCStatsStateSource()
    try:
        for f, is_late, trigger in eligible:
            try:
                r = process_one(
                    f,
                    kalshi_card,
                    fights_df,
                    rankings,
                    bankrolls,
                    event,
                    dry_run=args.dry_run,
                    now=now,
                    idempotency=idempotency,
                    verbose=verbose,
                    late=is_late,
                    trigger=trigger,
                    kalshi_client=api_client,
                    state_source=state_source,
                )
                summary.append(r)
            except Exception as e:
                print(
                    f"  ⚠ error processing {f.fighter_a} vs {f.fighter_b}: {e}",
                    file=sys.stderr,
                )
                import traceback

                traceback.print_exc()
    finally:
        state_source.close()

    if verbose:
        print(f"\n═══ done — {len(summary)} fights processed ═══")
        for r in summary:
            print(f"  {r['key']}: {r['result']}{' — ' + r.get('summary', '') if r.get('summary') else ''}")
    else:
        # launchd heartbeat: always emit one summary line per tick
        fired = [r for r in summary if r["result"] == "notified"]
        if fired:
            _heartbeat(
                now,
                f"tick: {event['title']} — fired {len(fired)} (keys: {', '.join(r['key'] for r in fired)})",
            )
        elif summary:
            _heartbeat(
                now,
                f"tick: {event['title']} — {len(eligible)} eligible, 0 fired (all deduped or errored)",
            )
        else:
            _heartbeat(now, f"tick: {event['title']} — no fight in window")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--once", action="store_true", help="Run for every fight on tonight's card, no time-gate."
    )
    ap.add_argument(
        "--watchdog",
        action="store_true",
        help="Event-triggered mode: fire when the previous fight resolves "
        "(or first fight at T-90min, or T-25min fallback).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Notification subjects prefixed [TEST]; idempotency log not written.",
    )
    ap.add_argument(
        "--time-offset-min",
        type=int,
        default=0,
        help="Pretend NOW is this many minutes from real time. Negative = earlier. Phase 3.",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-fight detail (default on for --once/--dry-run).",
    )
    args = ap.parse_args()
    if not args.once and not args.watchdog:
        args.watchdog = True  # default
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
