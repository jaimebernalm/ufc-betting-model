"""Scrape a single UFC event from ufcstats.com into rows that match the
mdabbert ``ufc-master.csv`` schema (118 columns).

Convention: R_fighter = Red corner (1st on fight-detail page),
B_fighter = Blue corner (2nd on fight-detail page).
Career averages (SLpM, Str.Acc, TD Avg, Sub Avg, ...) are taken as a
snapshot of the fighter's current page stats — this matches mdabbert's
pipeline. Win/loss/streak/title-bout counts are computed strictly from
fight-history rows dated *before* the event date.

Odds columns (R_odds, B_odds, R_ev, B_ev, r/b_dec_odds, r/b_sub_odds,
r/b_ko_odds) and ranking columns are left as NaN — they're populated
elsewhere (Polymarket eval) or by ``rankings_attach``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_date

from .ufcstats_client import UFCStatsClient

# --- column groups (kept here so the row builder can produce a stable schema) ---

ODDS_COLS = [
    "R_odds",
    "B_odds",
    "R_ev",
    "B_ev",
    "r_dec_odds",
    "b_dec_odds",
    "r_sub_odds",
    "b_sub_odds",
    "r_ko_odds",
    "b_ko_odds",
]

RANK_COLS = [
    "B_match_weightclass_rank",
    "R_match_weightclass_rank",
    *(
        f"R_{wc}_rank"
        for wc in [
            "Women's Flyweight",
            "Women's Featherweight",
            "Women's Strawweight",
            "Women's Bantamweight",
            "Heavyweight",
            "Light Heavyweight",
            "Middleweight",
            "Welterweight",
            "Lightweight",
            "Featherweight",
            "Bantamweight",
            "Flyweight",
            "Pound-for-Pound",
        ]
    ),
    *(
        f"B_{wc}_rank"
        for wc in [
            "Women's Flyweight",
            "Women's Featherweight",
            "Women's Strawweight",
            "Women's Bantamweight",
            "Heavyweight",
            "Light Heavyweight",
            "Middleweight",
            "Welterweight",
            "Lightweight",
            "Featherweight",
            "Bantamweight",
            "Flyweight",
            "Pound-for-Pound",
        ]
    ),
    "better_rank",
]


# ---------- event page ----------


@dataclass
class EventMeta:
    date: datetime
    location: str
    country: str
    fight_urls: list[str]


def parse_event_page(html: str) -> EventMeta:
    bs = BeautifulSoup(html, "html.parser")

    # Date and location appear as two consecutive list items
    date_str = None
    location = None
    for li in bs.find_all("li", {"class": "b-list__box-list-item"}):
        text = li.get_text(" ", strip=True)
        if text.startswith("Date:"):
            date_str = text[len("Date:") :].strip()
        elif text.startswith("Location:"):
            location = text[len("Location:") :].strip()
    if date_str is None or location is None:
        raise RuntimeError("Could not parse event date/location")

    event_date = datetime.strptime(date_str, "%B %d, %Y")
    country = location.split(",")[-1].strip()

    # Fight links — one green "win" flag <a> per completed fight; for upcoming
    # the row anchors point at the fight-details page too.
    fight_urls: list[str] = []
    seen = set()
    rows = bs.find_all("tr", {"class": "b-fight-details__table-row"})
    for row in rows:
        a = row.find("a", href=re.compile(r"fight-details/"))
        if a is None:
            continue
        href = a["href"]
        if href in seen:
            continue
        seen.add(href)
        fight_urls.append(href)

    return EventMeta(event_date, location, country, fight_urls)


# ---------- fight detail page ----------


@dataclass
class FightInfo:
    red_name: str
    blue_name: str
    red_fighter_url: str
    blue_fighter_url: str
    winner: str  # 'Red' | 'Blue' | 'Draw' | 'No Contest'
    weight_class: str
    title_bout: bool
    no_of_rounds: int
    finish: str | None
    finish_details: str | None
    finish_round: float | None
    finish_round_time: str | None


_FINISH_NORMALIZE = {
    "Submission": "SUB",
    "Decision - Unanimous": "U-DEC",
    "Decision - Split": "S-DEC",
    "Decision - Majority": "M-DEC",
    "TKO - Doctor's Stoppage": "TKO - Doctor's Stoppage",
    "KO/TKO": "KO/TKO",
    "DQ": "DQ",
    "Overturned": "Overturned",
    "Could Not Continue": "Could Not Continue",
    "Other": "Other",
}


def _normalize_finish(s: str | None) -> str | None:
    if not s:
        return s
    return _FINISH_NORMALIZE.get(s, s)


def _shorten_finish_details(s: str) -> str | None:
    """Reproduce mdabbert's terse style for finish_details.

    UFC stats writes things like "Punch to Head At Distance" or
    "Marlon Perry 27 - 30. Mike Bell 27 - 30. ..." for decisions.
    mdabbert stored just the leading strike name ("Punch", "Punches", "Knees")
    and left scorecards as NaN.
    """
    s = s.strip()
    if not s:
        return None
    # Decision scorecards start with a name followed by a number pattern
    if re.match(r"^[A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+\s+\d", s):
        return None
    # Take everything before " to ", " At ", or a punctuation/scorecard break
    head = re.split(r"\s+(?:to|At|from|From)\s+", s, maxsplit=1)[0]
    head = head.split(".")[0].strip()
    return head or None


_WEIGHT_CLASSES = [
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
    "Catch Weight",
    "Open Weight",
]


def _extract_weight_class(title_text: str) -> str:
    for wc in _WEIGHT_CLASSES:
        if wc in title_text:
            return wc
    return title_text.replace("Bout", "").replace("Title", "").strip() or "Unknown"


def parse_fight_detail(html: str) -> FightInfo:
    bs = BeautifulSoup(html, "html.parser")

    # Two person blocks: index 0 = Red corner, 1 = Blue corner
    persons = bs.find_all("div", {"class": "b-fight-details__person"})
    if len(persons) < 2:
        raise RuntimeError("fight detail page missing person blocks")

    def person_info(p):
        a = p.find("a", {"class": "b-link"})
        st = p.find("i", {"class": "b-fight-details__person-status"})
        return (
            a.get_text(strip=True) if a else "",
            a["href"] if a and a.has_attr("href") else "",
            st.get_text(strip=True) if st else "",
        )

    red_name, red_url, red_status = person_info(persons[0])
    blue_name, blue_url, blue_status = person_info(persons[1])

    if red_status == "W":
        winner = "Red"
    elif blue_status == "W":
        winner = "Blue"
    elif red_status == "D" and blue_status == "D":
        winner = "Draw"
    elif red_status == "NC" or blue_status == "NC":
        winner = "No Contest"
    else:
        winner = "No Contest"

    title_block = bs.find("i", {"class": "b-fight-details__fight-title"})
    title_text = title_block.get_text(" ", strip=True) if title_block else ""
    title_bout = bool(title_block and title_block.find("img", src=re.compile(r"belt")))
    weight_class = _extract_weight_class(title_text)

    # Method / Details
    finish = None
    finish_details = None
    for it in bs.find_all("i", {"class": "b-fight-details__text-item_first"}):
        text = it.get_text(" ", strip=True)
        if text.startswith("Method:"):
            finish = text[len("Method:") :].strip()

    # "Details:" can be a `<p>` of its own (with the value inline), not just
    # the text-item_first variant. Look across the content block.
    content = bs.find("div", {"class": "b-fight-details__content"})
    if content is not None:
        for p in content.find_all("p"):
            t = p.get_text(" ", strip=True)
            if t.startswith("Details:"):
                val = t[len("Details:") :].strip()
                finish_details = _shorten_finish_details(val)
                break

    # Round, time, time format
    finish_round = None
    finish_round_time = None
    no_of_rounds = 3
    for it in bs.find_all("i", {"class": "b-fight-details__text-item"}):
        text = it.get_text(" ", strip=True)
        if text.startswith("Round:"):
            try:
                finish_round = float(text[len("Round:") :].strip())
            except ValueError:
                pass
        elif text.startswith("Time:") and not text.startswith("Time format"):
            finish_round_time = text[len("Time:") :].strip()
        elif text.startswith("Time format:"):
            fmt = text[len("Time format:") :].strip()
            # Examples: "5 Rnd (5-5-5-5-5)", "3 Rnd (5-5-5)", "1 Rnd (5)"
            m = re.match(r"(\d+)\s*Rnd", fmt)
            if m:
                no_of_rounds = int(m.group(1))

    # For upcoming bouts there's no result yet
    if winner == "No Contest" and red_status == "" and blue_status == "":
        finish = None

    return FightInfo(
        red_name=red_name,
        blue_name=blue_name,
        red_fighter_url=red_url,
        blue_fighter_url=blue_url,
        winner=winner,
        weight_class=weight_class,
        title_bout=title_bout,
        no_of_rounds=no_of_rounds,
        finish=_normalize_finish(finish),
        finish_details=finish_details,
        finish_round=finish_round,
        finish_round_time=finish_round_time,
    )


# ---------- fighter detail page ----------


@dataclass
class FighterStats:
    # snapshot
    stance: str | None
    height_cms: float | None
    reach_cms: float | None
    weight_lbs: int | None
    age: int | None
    avg_sig_str_landed: float | None
    avg_sig_str_pct: float | None
    avg_td_landed: float | None
    avg_td_pct: float | None
    avg_sub_att: float | None
    # rolling, pre-event-date
    wins: int
    losses: int
    draws: int
    current_win_streak: int
    current_lose_streak: int
    longest_win_streak: int
    total_rounds_fought: int
    total_title_bouts: int
    win_by_dec_majority: int
    win_by_dec_split: int
    win_by_dec_unanimous: int
    win_by_ko: int
    win_by_sub: int
    win_by_tko_doctor: int = 0


def _height_to_cm(s: str) -> float | None:
    s = s.strip()
    if not s or s == "--":
        return None
    s = s.replace("'", "").replace('"', "")
    parts = s.split()
    if len(parts) != 2:
        return None
    try:
        return round((int(parts[0]) * 12 + int(parts[1])) * 2.54, 2)
    except ValueError:
        return None


def _reach_to_cm(s: str, fallback_height_cm: float | None) -> float | None:
    s = s.strip().replace('"', "")
    if not s or s == "--":
        return fallback_height_cm
    try:
        return round(int(s) * 2.54, 2)
    except ValueError:
        return None


def _percent_to_decimal(s: str) -> float | None:
    s = s.strip().rstrip("%")
    if not s or s == "--":
        return None
    try:
        return int(s) / 100.0
    except ValueError:
        try:
            return float(s) / 100.0
        except ValueError:
            return None


def _to_float(s: str) -> float | None:
    s = s.strip()
    if not s or s == "--":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _info_value(li_text: str) -> str:
    return li_text.split(":", 1)[1].strip() if ":" in li_text else ""


def _parse_landed_attempted(value: str) -> tuple[int, int]:
    """Parse UFCStats' ``"12 of 34"`` cells.

    Missing statistics occur on a small number of early bouts.  They carry no
    usable time-normalised statistics and are represented as zeroes here.
    """
    m = re.match(r"\s*(\d+)\s+of\s+(\d+)\s*$", value or "")
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))


def _parse_int(value: str) -> int:
    m = re.search(r"\d+", value or "")
    return int(m.group()) if m else 0


def _fight_seconds(round_text: str, time_text: str) -> int:
    """Return elapsed fight seconds from profile-history round/time cells."""
    round_ = _parse_int(round_text)
    try:
        minutes, seconds = (int(x) for x in time_text.strip().split(":"))
    except (AttributeError, TypeError, ValueError):
        return 0
    if round_ <= 0:
        return 0
    return 300 * (round_ - 1) + 60 * minutes + seconds


def _pre_event_career_rates(
    bs: BeautifulSoup,
    event_date: datetime,
    fighter_url: str,
    fight_html_getter: Callable[[str], str],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Rebuild UFCStats profile rates using bouts strictly before ``event_date``.

    UFCStats fighter pages expose only the *current* career snapshot.  Reading
    those five values while constructing an older event row leaks later bouts.
    The profile history supplies landed counts, submission attempts, round and
    time.  Fight-detail totals supply the two attempt denominators.

    Returns ``(SLpM, Str.Acc, TD Avg, TD Acc, Sub Avg)`` using UFCStats' display
    precision.  This is also the live-inference source, eliminating the former
    train/serve split between recorded rows and ``_last_fighter_stats``.
    """
    sig_landed = sig_attempted = 0
    td_landed = td_attempted = 0
    sub_attempted = 0
    fight_seconds = 0
    target = pd.Timestamp(event_date)
    fighter_key = fighter_url.rstrip("/")

    for row in bs.find_all("tr", {"class": "b-fight-details__table-row"}):
        cols = row.find_all("p", {"class": "b-fight-details__table-text"})
        if len(cols) < 17:
            continue
        try:
            bout_date = pd.Timestamp(parse_date(cols[12].get_text(strip=True)))
        except (ValueError, TypeError):
            continue
        if bout_date >= target:
            continue

        elapsed = _fight_seconds(cols[15].get_text(strip=True), cols[16].get_text(strip=True))
        detail_url = next(
            (a["href"] for a in row.find_all("a", href=True) if "fight-details/" in a["href"]),
            None,
        )
        if not detail_url or elapsed <= 0:
            continue

        detail = BeautifulSoup(fight_html_getter(detail_url), "html.parser")
        totals = detail.find("tbody", {"class": "b-fight-details__table-body"})
        totals_row = totals.find("tr") if totals is not None else None
        cells = totals_row.find_all("td", recursive=False) if totals_row is not None else []
        if len(cells) < 8:
            continue

        fighter_links = [a.get("href", "").rstrip("/") for a in cells[0].find_all("a", href=True)]
        try:
            side = fighter_links.index(fighter_key)
        except ValueError:
            continue

        sig_values = cells[2].find_all("p")
        td_values = cells[5].find_all("p")
        sub_values = cells[7].find_all("p")
        if side >= len(sig_values) or side >= len(td_values) or side >= len(sub_values):
            continue

        sl, sa = _parse_landed_attempted(sig_values[side].get_text(strip=True))
        tl, ta = _parse_landed_attempted(td_values[side].get_text(strip=True))
        sig_landed += sl
        sig_attempted += sa
        td_landed += tl
        td_attempted += ta
        sub_attempted += _parse_int(sub_values[side].get_text(strip=True))
        fight_seconds += elapsed

    if fight_seconds <= 0:
        return None, None, None, None, None

    slpm = round(sig_landed * 60.0 / fight_seconds, 2)
    str_acc = round(sig_landed / sig_attempted, 2) if sig_attempted else 0.0
    td_avg = round(td_landed * 900.0 / fight_seconds, 2)
    td_acc = round(td_landed / td_attempted, 2) if td_attempted else 0.0
    sub_avg = round(sub_attempted * 900.0 / fight_seconds, 1)
    return slpm, str_acc, td_avg, td_acc, sub_avg


def parse_fighter_page(
    html: str,
    event_date: datetime,
    *,
    fighter_url: str | None = None,
    fight_html_getter: Callable[[str], str] | None = None,
) -> FighterStats:
    bs = BeautifulSoup(html, "html.parser")

    # Info / career-stat list — typed lookups by label
    info = {}
    for li in bs.find_all("li", {"class": "b-list__box-list-item"}):
        text = li.get_text(" ", strip=True)
        if not text or ":" not in text:
            continue
        key = text.split(":", 1)[0].strip().lower()
        val = text.split(":", 1)[1].strip()
        info[key] = val

    height_cm = _height_to_cm(info.get("height", ""))
    reach_cm = _reach_to_cm(info.get("reach", ""), height_cm)
    weight_str = info.get("weight", "").replace("lbs.", "").strip()
    try:
        weight_lbs = int(weight_str) if weight_str and weight_str != "--" else None
    except ValueError:
        weight_lbs = None

    stance = info.get("stance") or None
    if stance == "--":
        stance = None

    age: int | None = None
    dob_str = info.get("dob", "")
    if dob_str and dob_str != "--":
        try:
            dob = datetime.strptime(dob_str, "%b %d, %Y")
            age = event_date.year - dob.year - ((event_date.month, event_date.day) < (dob.month, dob.day))
        except ValueError:
            age = None

    slpm = _to_float(info.get("slpm", ""))
    str_acc = _percent_to_decimal(info.get("str. acc.", ""))
    td_avg = _to_float(info.get("td avg.", ""))
    td_acc = _percent_to_decimal(info.get("td acc.", ""))
    sub_avg = _to_float(info.get("sub. avg.", ""))

    # The five profile values above are a current snapshot.  For historical
    # row construction and live pre-fight inference, rebuild them at the target
    # date from immutable bout totals.  Keeping the old behavior when no loader
    # is supplied preserves this parser's lightweight standalone API.
    if fighter_url is not None and fight_html_getter is not None:
        rebuilt = _pre_event_career_rates(bs, event_date, fighter_url, fight_html_getter)
        if rebuilt[0] is not None:
            slpm, str_acc, td_avg, td_acc, sub_avg = rebuilt
        else:
            # A debutant has no prior UFC time.  Never fall back to the current
            # profile snapshot here: after the event it already contains the
            # target bout and would leak its statistics into the event row.
            slpm = str_acc = td_avg = td_acc = sub_avg = 0.0

    # Fight history rows — compute pre-event stats
    wins = losses = draws = 0
    current_win = current_lose = 0
    longest_win = 0
    temp_streak = 0
    rounds_total = 0
    title_bouts_pre = 0
    by_maj = by_split = by_unan = by_ko = by_sub = by_doctor = 0
    streak_ended = False  # True once we leave the leading streak

    rows = bs.find_all("tr", {"class": "b-fight-details__table-row"})
    # Iteration order is most-recent first (matches mdabbert's logic).
    for row in rows:
        cols = row.find_all("p", {"class": "b-fight-details__table-text"})
        if len(cols) < 17:
            continue
        result = cols[0].get_text(strip=True).lower()  # 'win'/'loss'/'draw'/'nc'
        if not result:
            continue
        # Fight date
        date_str = cols[12].get_text(strip=True)
        try:
            fight_date = parse_date(date_str)
        except (ValueError, TypeError):
            continue
        if fight_date >= event_date:
            continue  # only past fights count

        # Title bout? belt icon in this row
        if row.find("img", src=re.compile(r"belt")):
            title_bouts_pre += 1

        # Rounds in the fight
        try:
            rounds_total += int(cols[15].get_text(strip=True))
        except ValueError:
            pass

        # Method (col 13): U-DEC, S-DEC, M-DEC, KO/TKO, SUB
        method = cols[13].get_text(strip=True)
        if result == "win":
            if method == "M-DEC":
                by_maj += 1
            elif method == "S-DEC":
                by_split += 1
            elif method == "U-DEC":
                by_unan += 1
            elif method == "KO/TKO":
                by_ko += 1
            elif method == "TKO - Doctor's Stoppage":
                by_doctor += 1
            elif method == "SUB":
                by_sub += 1

        # Running totals
        if result == "win":
            wins += 1
            temp_streak += 1
            if temp_streak > longest_win:
                longest_win = temp_streak
        elif result == "loss":
            losses += 1
            temp_streak = 0
        elif result == "draw":
            draws += 1
            temp_streak = 0  # breaks longest-streak tracker

        # Leading streak (from most-recent backwards until interrupted by an
        # opposite result). Draws/NCs are silently ignored — they do not
        # extend or break the current streak (matches mdabbert's convention).
        if not streak_ended:
            if result == "win":
                if current_lose > 0:
                    streak_ended = True
                else:
                    current_win += 1
            elif result == "loss":
                if current_win > 0:
                    streak_ended = True
                else:
                    current_lose += 1

    return FighterStats(
        stance=stance,
        height_cms=height_cm,
        reach_cms=reach_cm,
        weight_lbs=weight_lbs,
        age=age,
        avg_sig_str_landed=slpm,
        avg_sig_str_pct=str_acc,
        avg_td_landed=td_avg,
        avg_td_pct=td_acc,
        avg_sub_att=sub_avg,
        wins=wins,
        losses=losses,
        draws=draws,
        current_win_streak=current_win,
        current_lose_streak=current_lose,
        longest_win_streak=longest_win,
        total_rounds_fought=rounds_total,
        total_title_bouts=title_bouts_pre,
        win_by_dec_majority=by_maj,
        win_by_dec_split=by_split,
        win_by_dec_unanimous=by_unan,
        win_by_ko=by_ko,
        win_by_sub=by_sub,
        win_by_tko_doctor=by_doctor,
    )


# ---------- row assembly ----------


def _total_fight_time_secs(round_: float | None, time_str: str | None) -> float | None:
    if round_ is None or time_str is None:
        return None
    try:
        r = int(round_)
        mm, ss = time_str.split(":")
        return 300.0 * (r - 1) + 60 * int(mm) + int(ss)
    except (ValueError, AttributeError):
        return None


def _row(
    event: EventMeta,
    fight: FightInfo,
    red: FighterStats,
    blue: FighterStats,
) -> dict:
    is_completed = fight.winner in ("Red", "Blue", "Draw", "No Contest") and fight.finish is not None

    gender = "FEMALE" if fight.weight_class.startswith("Women's") else "MALE"

    row = {
        # Identifiers / odds (NaN)
        "R_fighter": fight.red_name,
        "B_fighter": fight.blue_name,
        "R_odds": None,
        "B_odds": None,
        "R_ev": None,
        "B_ev": None,
        "date": event.date.strftime("%Y-%m-%d"),
        "location": event.location,
        "country": event.country,
        "Winner": fight.winner,
        "title_bout": fight.title_bout,
        "weight_class": fight.weight_class,
        "gender": gender,
        "no_of_rounds": fight.no_of_rounds,
        # Blue rolling/career
        "B_current_lose_streak": blue.current_lose_streak,
        "B_current_win_streak": blue.current_win_streak,
        "B_draw": blue.draws,
        "B_avg_SIG_STR_landed": blue.avg_sig_str_landed,
        "B_avg_SIG_STR_pct": blue.avg_sig_str_pct,
        "B_avg_SUB_ATT": blue.avg_sub_att,
        "B_avg_TD_landed": blue.avg_td_landed,
        "B_avg_TD_pct": blue.avg_td_pct,
        "B_longest_win_streak": blue.longest_win_streak,
        "B_losses": blue.losses,
        "B_total_rounds_fought": blue.total_rounds_fought,
        "B_total_title_bouts": blue.total_title_bouts,
        "B_win_by_Decision_Majority": blue.win_by_dec_majority,
        "B_win_by_Decision_Split": blue.win_by_dec_split,
        "B_win_by_Decision_Unanimous": blue.win_by_dec_unanimous,
        "B_win_by_KO/TKO": blue.win_by_ko,
        "B_win_by_Submission": blue.win_by_sub,
        "B_win_by_TKO_Doctor_Stoppage": blue.win_by_tko_doctor,
        "B_wins": blue.wins,
        "B_Stance": blue.stance,
        "B_Height_cms": blue.height_cms,
        "B_Reach_cms": blue.reach_cms,
        "B_Weight_lbs": blue.weight_lbs,
        # Red rolling/career
        "R_current_lose_streak": red.current_lose_streak,
        "R_current_win_streak": red.current_win_streak,
        "R_draw": red.draws,
        "R_avg_SIG_STR_landed": red.avg_sig_str_landed,
        "R_avg_SIG_STR_pct": red.avg_sig_str_pct,
        "R_avg_SUB_ATT": red.avg_sub_att,
        "R_avg_TD_landed": red.avg_td_landed,
        "R_avg_TD_pct": red.avg_td_pct,
        "R_longest_win_streak": red.longest_win_streak,
        "R_losses": red.losses,
        "R_total_rounds_fought": red.total_rounds_fought,
        "R_total_title_bouts": red.total_title_bouts,
        "R_win_by_Decision_Majority": red.win_by_dec_majority,
        "R_win_by_Decision_Split": red.win_by_dec_split,
        "R_win_by_Decision_Unanimous": red.win_by_dec_unanimous,
        "R_win_by_KO/TKO": red.win_by_ko,
        "R_win_by_Submission": red.win_by_sub,
        "R_win_by_TKO_Doctor_Stoppage": red.win_by_tko_doctor,
        "R_wins": red.wins,
        "R_Stance": red.stance,
        "R_Height_cms": red.height_cms,
        "R_Reach_cms": red.reach_cms,
        "R_Weight_lbs": red.weight_lbs,
        "R_age": red.age,
        "B_age": blue.age,
    }

    # Diffs (B - R) — match mdabbert's convention
    def diff(a, b):
        if a is None or b is None:
            return None
        return a - b

    row.update(
        {
            "lose_streak_dif": diff(blue.current_lose_streak, red.current_lose_streak),
            "win_streak_dif": diff(blue.current_win_streak, red.current_win_streak),
            "longest_win_streak_dif": diff(blue.longest_win_streak, red.longest_win_streak),
            "win_dif": diff(blue.wins, red.wins),
            "loss_dif": diff(blue.losses, red.losses),
            "total_round_dif": diff(blue.total_rounds_fought, red.total_rounds_fought),
            "total_title_bout_dif": diff(row["B_total_title_bouts"], row["R_total_title_bouts"]),
            "ko_dif": diff(blue.win_by_ko, red.win_by_ko),
            "sub_dif": diff(blue.win_by_sub, red.win_by_sub),
            "height_dif": diff(blue.height_cms, red.height_cms),
            "reach_dif": diff(blue.reach_cms, red.reach_cms),
            "age_dif": diff(blue.age, red.age),
            "sig_str_dif": diff(blue.avg_sig_str_landed, red.avg_sig_str_landed),
            "avg_sub_att_dif": diff(blue.avg_sub_att, red.avg_sub_att),
            "avg_td_dif": diff(blue.avg_td_landed, red.avg_td_landed),
            "empty_arena": 0,  # post-COVID default
        }
    )

    # Ranks (filled later by rankings_attach)
    for c in RANK_COLS:
        row[c] = None

    # Finish details
    row.update(
        {
            "finish": fight.finish if is_completed else None,
            "finish_details": fight.finish_details if is_completed else None,
            "finish_round": fight.finish_round if is_completed else None,
            "finish_round_time": fight.finish_round_time if is_completed else None,
            "total_fight_time_secs": (
                _total_fight_time_secs(fight.finish_round, fight.finish_round_time) if is_completed else None
            ),
        }
    )

    # Exotic odds (left NaN)
    for c in ("r_dec_odds", "b_dec_odds", "r_sub_odds", "b_sub_odds", "r_ko_odds", "b_ko_odds"):
        row[c] = None

    return row


def scrape_event(event_url: str, client: UFCStatsClient | None = None) -> pd.DataFrame:
    """Scrape one UFC event into a DataFrame matching ufc-master.csv schema."""
    owned = False
    if client is None:
        client = UFCStatsClient()
        owned = True
    try:
        event_html = client.get(event_url)
        event = parse_event_page(event_html)

        # Fight pages are immutable.  Cache them across both fighters because
        # raw pre-event career-rate reconstruction reads every prior detail.
        fight_html_cache: dict[str, str] = {}

        def get_fight_html(url: str) -> str:
            if url not in fight_html_cache:
                fight_html_cache[url] = client.get(url)
            return fight_html_cache[url]

        # Cache fighter pages so we don't refetch when the same fighter appears
        fighter_cache: dict[str, FighterStats] = {}

        rows: list[dict] = []
        for fight_url in event.fight_urls:
            fight_html = get_fight_html(fight_url)
            fight = parse_fight_detail(fight_html)

            for url in (fight.red_fighter_url, fight.blue_fighter_url):
                if url and url not in fighter_cache:
                    fhtml = client.get(url)
                    fighter_cache[url] = parse_fighter_page(
                        fhtml,
                        event.date,
                        fighter_url=url,
                        fight_html_getter=get_fight_html,
                    )

            red_stats = fighter_cache[fight.red_fighter_url]
            blue_stats = fighter_cache[fight.blue_fighter_url]
            rows.append(_row(event, fight, red_stats, blue_stats))

        return pd.DataFrame(rows)
    finally:
        if owned:
            client.__exit__(None, None, None)
