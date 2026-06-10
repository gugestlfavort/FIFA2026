"""Daily orchestrator: fetch fresh data, refit, simulate, publish JSON.

Writes docs/data/predictions.json (everything the dashboard renders) and
maintains docs/data/predictions_log.json (frozen pre-kickoff predictions,
scored once results are final — the live KPI track record).

Usage: ODDS_API_KEY=... python -m pipeline.run
"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo

from .model import (fit_dixon_coles, fit_elo_goal_model, elo_history,
                    compute_elo, BlendModel, score_matrix, outcome_probs,
                    reweight_matrix)
from .simulate import simulate_tournament
from .sources import (fetch_results, fetch_espn_schedule, fetch_match_odds,
                      fetch_outright_odds)
from .teams import GROUPS, TEAM_GROUP, FLAGS, WC_TEAMS

TZ = ZoneInfo("Europe/Zurich")
W_MARKET = 0.5          # weight on de-vigged market 1x2 when odds exist
DEFAULT_W_DC = 0.6      # overridden by backtest.json's best_w_dc
N_SIMS = 10000
DOCS_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "docs", "data")
HOST_HFA = 1.0          # scale on fitted home coefficient for host nations


def merge_espn_results(results: pd.DataFrame, sched: pd.DataFrame):
    """Add completed WC matches from ESPN that the dataset lacks yet."""
    have = set(zip(results["date"].dt.date, results["home_team"],
                   results["away_team"]))
    add = []
    for r in sched[sched["completed"]].itertuples(index=False):
        d = r.kickoff_utc.date()
        if (d, r.home, r.away) in have or (d, r.away, r.home) in have:
            continue
        add.append({"date": pd.Timestamp(d), "home_team": r.home,
                    "away_team": r.away, "home_score": r.home_score,
                    "away_score": r.away_score, "tournament": "FIFA World Cup",
                    "city": r.city, "country": r.venue_country,
                    "neutral": r.home != r.venue_country})
    if add:
        results = pd.concat([results, pd.DataFrame(add)], ignore_index=True)
        results = results.sort_values("date").reset_index(drop=True)
    return results, len(add)


def build_models(results: pd.DataFrame, ref_date):
    w_dc = DEFAULT_W_DC
    bt_path = os.path.join(DOCS_DATA, "backtest.json")
    if os.path.exists(bt_path):
        with open(bt_path) as f:
            w_dc = float(json.load(f)["best_w_dc"])
    dc = fit_dixon_coles(results, ref_date)
    eloh = elo_history(results)
    eg = fit_elo_goal_model(results, eloh, ref_date)
    eg.elo = compute_elo(results)
    return BlendModel(dc, eg, w_dc), w_dc


def match_matrix(bm: BlendModel, home, away, venue_ctry):
    """Score matrix oriented (home rows, away cols) with host advantage."""
    if home == venue_ctry:
        lh, la = bm.lambdas(home, away, HOST_HFA)
    elif away == venue_ctry:
        la2, lh2 = bm.lambdas(away, home, HOST_HFA)
        lh, la = lh2, la2
    else:
        lh, la = bm.lambdas(home, away, 0.0)
    return score_matrix(lh, la, bm.rho)


def lookup_odds(odds, home, away):
    if (home, away) in odds:
        return odds[(home, away)]
    if (away, home) in odds:
        o = odds[(away, home)]
        return {"home": o["away"], "draw": o["draw"], "away": o["home"],
                "books": o["books"]}
    return None


def top_scorelines(m: np.ndarray, k=3):
    flat = [(float(m[i, j]), i, j)
            for i in range(m.shape[0]) for j in range(m.shape[1])]
    flat.sort(reverse=True)
    return [{"score": f"{i}-{j}", "p": round(p, 4)} for p, i, j in flat[:k]]


def update_log(sched, fixtures_probs, now_utc):
    """Freeze pre-kickoff predictions; score completed ones; return KPIs."""
    path = os.path.join(DOCS_DATA, "predictions_log.json")
    log = {}
    if os.path.exists(path):
        with open(path) as f:
            log = json.load(f)

    for r in sched.itertuples(index=False):
        eid = str(r.event_id)
        started = r.kickoff_utc <= now_utc
        entry = log.get(eid)
        if entry is None and not started and eid in fixtures_probs:
            log[eid] = {"home": r.home, "away": r.away,
                        "kickoff_utc": r.kickoff_utc.isoformat(),
                        "stage": r.stage, **fixtures_probs[eid]}
        elif entry is not None and not started and eid in fixtures_probs:
            entry.update(fixtures_probs[eid])  # refresh until kickoff
        if entry is not None and r.completed and "result" not in entry:
            entry["result"] = [int(r.home_score), int(r.away_score)]

    # score frozen predictions that have results
    scored = []
    for e in log.values():
        if "result" not in e:
            continue
        gh, ga = e["result"]
        outcome = 0 if gh > ga else 2 if ga > gh else 1
        for key in ("blend", "market"):
            if key not in e:
                continue
            p = e[key]
            probs = np.array([p["home"], p["draw"], p["away"]])
            obs = np.zeros(3)
            obs[outcome] = 1
            cp, co = np.cumsum(probs), np.cumsum(obs)
            scored.append({
                "model": key,
                "rps": float(np.sum((cp - co) ** 2) / 2),
                "logloss": -float(np.log(max(probs[outcome], 1e-12))),
                "correct": int(int(np.argmax(probs)) == outcome),
            })
    live = {}
    sdf = pd.DataFrame(scored)
    if len(sdf):
        for key, g in sdf.groupby("model"):
            live[key] = {"rps": float(g["rps"].mean()),
                         "logloss": float(g["logloss"].mean()),
                         "accuracy": float(g["correct"].mean()),
                         "n": int(len(g))}
    with open(path, "w") as f:
        json.dump(log, f, indent=1)
    return live


def main():
    api_key = os.environ.get("ODDS_API_KEY", "")
    now_utc = pd.Timestamp.now(tz="UTC")
    today_local = now_utc.tz_convert(TZ).date()

    results = fetch_results(use_cache=False)
    sched = fetch_espn_schedule()
    results, n_merged = merge_espn_results(results, sched)
    ref = pd.Timestamp(now_utc.date() + dt.timedelta(days=1))
    bm, w_dc = build_models(results, ref)
    odds = fetch_match_odds(api_key)
    outright = fetch_outright_odds(api_key)
    print(f"results rows: {len(results)} (+{n_merged} from ESPN), "
          f"odds for {len(odds)} fixtures, w_dc={w_dc}")

    # --- per-fixture probabilities (group stage now; KO once teams known) ---
    fixtures_probs = {}
    group_fixtures = []
    matrices = {}
    for r in sched.itertuples(index=False):
        eid = str(r.event_id)
        known = r.home in WC_TEAMS and r.away in WC_TEAMS
        if not known:
            continue
        m = match_matrix(bm, r.home, r.away, r.venue_country)
        model_p = outcome_probs(m)
        rec = {"model": {"home": round(model_p[0], 4),
                         "draw": round(model_p[1], 4),
                         "away": round(model_p[2], 4)}}
        mk = lookup_odds(odds, r.home, r.away)
        blend_p = model_p
        if mk:
            rec["market"] = {"home": round(mk["home"], 4),
                             "draw": round(mk["draw"], 4),
                             "away": round(mk["away"], 4),
                             "books": mk["books"]}
            blend_p = tuple(W_MARKET * mk[k] + (1 - W_MARKET) * model_p[i]
                            for i, k in enumerate(("home", "draw", "away")))
            m = reweight_matrix(m, blend_p)
        rec["blend"] = {"home": round(blend_p[0], 4),
                        "draw": round(blend_p[1], 4),
                        "away": round(blend_p[2], 4)}
        fixtures_probs[eid] = rec
        matrices[eid] = m
        if r.stage == "group":
            fx = {"home": r.home, "away": r.away, "group": r.group}
            if r.completed:
                fx["result"] = (int(r.home_score), int(r.away_score))
            else:
                fx["matrix"] = m
            group_fixtures.append(fx)

    # --- knockout venue map (chronological ESPN order -> match numbers) ---
    ko = sched[sched["stage"] == "knockout"].sort_values("kickoff_utc")
    ko_venue = {}
    if len(ko) >= 31:
        nums = list(range(73, 105))  # includes 103 (3rd place)
        for num, r in zip(nums, ko.itertuples(index=False)):
            ko_venue[num] = r.venue_country

    def ko_prob(t1, t2, match_no):
        vc = ko_venue.get(match_no, "United States")
        return outcome_probs(match_matrix(bm, t1, t2, vc))

    sim = simulate_tournament(group_fixtures, ko_prob, n_sims=N_SIMS)

    # --- group tables from played matches ---
    tables = {}
    for g, teams in GROUPS.items():
        rows = {t: {"team": t, "flag": FLAGS.get(t, ""), "p": 0, "w": 0,
                    "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0} for t in teams}
        for fx in group_fixtures:
            if fx["group"] != g or "result" not in fx:
                continue
            gh, ga = fx["result"]
            h, a = fx["home"], fx["away"]
            rows[h]["p"] += 1; rows[a]["p"] += 1
            rows[h]["gf"] += gh; rows[h]["ga"] += ga
            rows[a]["gf"] += ga; rows[a]["ga"] += gh
            if gh > ga:
                rows[h]["w"] += 1; rows[a]["l"] += 1; rows[h]["pts"] += 3
            elif ga > gh:
                rows[a]["w"] += 1; rows[h]["l"] += 1; rows[a]["pts"] += 3
            else:
                rows[h]["d"] += 1; rows[a]["d"] += 1
                rows[h]["pts"] += 1; rows[a]["pts"] += 1
        for t in teams:
            rows[t]["advance"] = round(sim[t]["advance"], 4)
            rows[t]["group_win"] = round(sim[t]["group_win"], 4)
        tables[g] = sorted(rows.values(),
                           key=lambda x: (-x["pts"], -(x["gf"] - x["ga"]),
                                          -x["gf"], -x["advance"]))

    # --- today's matches (or next matchday if none today) ---
    local_dates = sched["kickoff_utc"].dt.tz_convert(TZ).dt.date
    show_date = today_local
    if not (local_dates == today_local).any():
        upcoming = sorted(d for d in local_dates if d > today_local)
        if upcoming:
            show_date = upcoming[0]
    today = []
    for r in sched.itertuples(index=False):
        local = r.kickoff_utc.tz_convert(TZ)
        if local.date() != show_date:
            continue
        eid = str(r.event_id)
        entry = {
            "home": r.home, "away": r.away,
            "home_flag": FLAGS.get(r.home, ""), "away_flag": FLAGS.get(r.away, ""),
            "kickoff_local": local.strftime("%H:%M"),
            "kickoff_utc": r.kickoff_utc.isoformat(),
            "venue": r.venue, "city": r.city, "group": r.group,
            "stage": r.stage, "state": r.state,
            "completed": bool(r.completed),
        }
        if r.completed:
            entry["result"] = [int(r.home_score), int(r.away_score)]
        if eid in fixtures_probs:
            entry.update(fixtures_probs[eid])
            entry["top_scores"] = top_scorelines(matrices[eid])
            if r.home in TEAM_GROUP:
                entry["context"] = {
                    "home_advance": round(sim[r.home]["advance"], 4),
                    "away_advance": round(sim[r.away]["advance"], 4),
                    "home_champion": round(sim[r.home]["champion"], 4),
                    "away_champion": round(sim[r.away]["champion"], 4),
                }
        today.append(entry)

    live_kpis = update_log(sched, fixtures_probs, now_utc)

    # --- tournament odds table ---
    standings = []
    for t in WC_TEAMS:
        standings.append({
            "team": t, "flag": FLAGS.get(t, ""), "group": TEAM_GROUP[t],
            "elo": round(bm.eg.elo.get(t, 1500.0)),
            **{k: round(sim[t][k], 4)
               for k in ("advance", "r16", "qf", "sf", "final", "champion")},
            "market_champion": round(outright.get(t, 0.0), 4) if outright else None,
        })
    standings.sort(key=lambda x: -x["champion"])

    backtest = None
    bt_path = os.path.join(DOCS_DATA, "backtest.json")
    if os.path.exists(bt_path):
        with open(bt_path) as f:
            backtest = json.load(f)

    out = {
        "generated_utc": now_utc.isoformat(),
        "generated_local": now_utc.tz_convert(TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "today_date": str(today_local),
        "show_date": str(show_date),
        "is_next_matchday": show_date != today_local,
        "today": today,
        "standings": standings,
        "groups": tables,
        "live_kpis": live_kpis,
        "backtest": backtest,
        "meta": {
            "w_dc": w_dc, "w_market": W_MARKET, "n_sims": N_SIMS,
            "rho": round(bm.dc.rho, 4),
            "results_rows": int(len(results)),
            "espn_merged": n_merged,
            "odds_fixtures": len(odds),
            "has_outright": bool(outright),
        },
    }
    os.makedirs(DOCS_DATA, exist_ok=True)
    with open(os.path.join(DOCS_DATA, "predictions.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote predictions.json: {len(today)} matches today, "
          f"top pick {standings[0]['team']} "
          f"({standings[0]['champion']:.1%} champion)")


if __name__ == "__main__":
    main()
