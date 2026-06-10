"""Data fetchers: historical results, ESPN schedule/results, betting odds."""
import datetime as dt
import os
import statistics

import pandas as pd
import requests

from .teams import canon, venue_country, TEAM_GROUP, WC_TEAMS

RESULTS_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")
ESPN_SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
                   "fifa.world/scoreboard")
ODDS_BASE = "https://api.the-odds-api.com/v4/sports"

WC_START = dt.date(2026, 6, 11)
WC_END = dt.date(2026, 7, 19)
GROUP_STAGE_END = dt.date(2026, 6, 27)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")


def fetch_results(use_cache: bool = False) -> pd.DataFrame:
    """All international results, canonical names, played matches only."""
    path = os.path.join(CACHE_DIR, "results.csv")
    if not (use_cache and os.path.exists(path)):
        os.makedirs(CACHE_DIR, exist_ok=True)
        r = requests.get(RESULTS_URL, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["home_team"] = df["home_team"].map(canon)
    df["away_team"] = df["away_team"].map(canon)
    missing = [t for t in WC_TEAMS
               if t not in set(df.home_team) | set(df.away_team)]
    if missing:
        raise ValueError(f"WC teams missing from results dataset: {missing}")
    return df.sort_values("date").reset_index(drop=True)


def fetch_espn_schedule() -> pd.DataFrame:
    """Every WC 2026 match from ESPN: teams, kickoff (UTC), venue, status, score.

    Knockout matches appear with placeholder team names ("TBD") until decided.
    """
    rows = []
    s = requests.Session()
    r = s.get(ESPN_SCOREBOARD,
              params={"dates": f"{WC_START:%Y%m%d}-{WC_END:%Y%m%d}",
                      "limit": "200"},
              timeout=60)
    r.raise_for_status()
    events = r.json().get("events", [])
    if len(events) < 100:  # range query unsupported or truncated: fall back
        events, seen = [], set()
        day = WC_START
        while day <= WC_END:
            r = s.get(ESPN_SCOREBOARD, params={"dates": f"{day:%Y%m%d}"},
                      timeout=60)
            r.raise_for_status()
            for e in r.json().get("events", []):
                if e["id"] not in seen:
                    seen.add(e["id"])
                    events.append(e)
            day += dt.timedelta(days=1)

    for e in events:
        comp = e["competitions"][0]
        home = away = None
        scores = {}
        for c in comp["competitors"]:
            name = canon(c["team"]["displayName"])
            side = c.get("homeAway")
            scores[side] = int(c["score"]) if c.get("score") not in (None, "") else None
            if side == "home":
                home = name
            else:
                away = name
        status = comp.get("status", {}).get("type", {})
        venue = comp.get("venue", {})
        kickoff = pd.Timestamp(e["date"]).tz_convert("UTC")
        rows.append({
            "event_id": e["id"],
            "kickoff_utc": kickoff,
            "home": home, "away": away,
            "home_score": scores.get("home"), "away_score": scores.get("away"),
            "completed": bool(status.get("completed", False)),
            "state": status.get("state", "pre"),
            "venue": venue.get("fullName", ""),
            "city": venue.get("address", {}).get("city", ""),
            "stage": ("group" if kickoff.date() <= GROUP_STAGE_END
                      else "knockout"),
        })
    df = pd.DataFrame(rows).sort_values("kickoff_utc").reset_index(drop=True)
    df["venue_country"] = [venue_country(v, c)
                           for v, c in zip(df["venue"], df["city"])]
    df["group"] = df["home"].map(TEAM_GROUP)
    df.loc[df["stage"] != "group", "group"] = None
    return df


def _devig(prices):
    """Decimal odds -> fair probabilities (proportional vig removal)."""
    inv = [1.0 / p for p in prices]
    s = sum(inv)
    return [x / s for x in inv]


def fetch_match_odds(api_key: str) -> dict:
    """Median bookmaker h2h odds per upcoming fixture, de-vigged.

    Returns {(home, away): {"home": p, "draw": p, "away": p, "books": n}}.
    """
    if not api_key:
        return {}
    r = requests.get(f"{ODDS_BASE}/soccer_fifa_world_cup/odds",
                     params={"apiKey": api_key, "regions": "eu",
                             "markets": "h2h", "oddsFormat": "decimal"},
                     timeout=60)
    r.raise_for_status()
    out = {}
    for ev in r.json():
        home, away = canon(ev["home_team"]), canon(ev["away_team"])
        prices = {"home": [], "draw": [], "away": []}
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] != "h2h":
                    continue
                quote = {}
                for o in mk.get("outcomes", []):
                    name = canon(o["name"])
                    key = ("home" if name == home else
                           "away" if name == away else
                           "draw" if o["name"].lower() == "draw" else None)
                    if key:
                        quote[key] = o["price"]
                if len(quote) == 3:
                    for k, v in quote.items():
                        prices[k].append(v)
        if prices["home"]:
            med = [statistics.median(prices[k]) for k in ("home", "draw", "away")]
            ph, pd_, pa = _devig(med)
            out[(home, away)] = {"home": ph, "draw": pd_, "away": pa,
                                 "books": len(prices["home"])}
    return out


def fetch_outright_odds(api_key: str) -> dict:
    """Market-implied tournament winner probabilities {team: p}."""
    if not api_key:
        return {}
    r = requests.get(f"{ODDS_BASE}/soccer_fifa_world_cup_winner/odds",
                     params={"apiKey": api_key, "regions": "eu",
                             "markets": "outrights", "oddsFormat": "decimal"},
                     timeout=60)
    r.raise_for_status()
    quotes = {}
    for ev in r.json():
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] != "outrights":
                    continue
                for o in mk.get("outcomes", []):
                    quotes.setdefault(canon(o["name"]), []).append(o["price"])
    if not quotes:
        return {}
    med = {t: statistics.median(v) for t, v in quotes.items()}
    inv = {t: 1.0 / p for t, p in med.items()}
    s = sum(inv.values())
    return {t: x / s for t, x in inv.items()}
