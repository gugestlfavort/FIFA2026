"""Backtest: fit models as-of past tournaments, score group-stage matches.

Validation sets (group stages only — clean 1x2 outcomes, no extra time):
  FIFA World Cup 2022, UEFA Euro 2024, Copa América 2024.

Reports RPS / log loss / Brier / accuracy for Elo-only, DC-only and the
geometric blends in between; picks the best blend weight by mean RPS and
writes everything to docs/data/backtest.json for the dashboard.

Usage: python -m pipeline.backtest
"""
import json
import os

import numpy as np
import pandas as pd

from .model import (fit_dixon_coles, fit_elo_goal_model, elo_history,
                    compute_elo, BlendModel, score_matrix, outcome_probs)
from .sources import fetch_results

TOURNAMENTS = [
    ("World Cup 2022", "FIFA World Cup", "2022-11-20", "2022-12-03"),
    ("Euro 2024", "UEFA Euro", "2024-06-14", "2024-06-27"),
    ("Copa América 2024", "Copa América", "2024-06-20", "2024-07-03"),
]
WEIGHTS = [0.0, 0.25, 0.5, 0.6, 0.75, 1.0]  # w_dc; 0 = pure Elo model
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "docs", "data", "backtest.json")


def rps(probs, outcome):
    """Ranked probability score; probs = (pH, pD, pA), outcome in {0,1,2}."""
    obs = np.zeros(3)
    obs[outcome] = 1.0
    cp, co = np.cumsum(probs), np.cumsum(obs)
    return float(np.sum((cp - co) ** 2) / 2.0)


def evaluate(results: pd.DataFrame):
    eloh = elo_history(results)
    rows = []
    for label, tname, start, end in TOURNAMENTS:
        ref = pd.Timestamp(start)
        hist = results[results["date"] < ref]
        dc = fit_dixon_coles(hist, ref)
        eg = fit_elo_goal_model(hist, eloh[eloh["date"] < ref], ref)
        eg.elo = compute_elo(hist)  # ratings as of the eve of the tournament

        matches = results[(results["tournament"] == tname)
                          & (results["date"] >= start)
                          & (results["date"] <= end)]
        for r in matches.itertuples(index=False):
            hfa = 0.0 if r.neutral else 1.0
            outcome = (0 if r.home_score > r.away_score
                       else 2 if r.away_score > r.home_score else 1)
            for w in WEIGHTS:
                bm = BlendModel(dc, eg, w)
                lh, la = bm.lambdas(r.home_team, r.away_team, hfa)
                p = outcome_probs(score_matrix(lh, la, dc.rho))
                rows.append({
                    "tournament": label, "w": w,
                    "home": r.home_team, "away": r.away_team,
                    "ph": p[0], "pd": p[1], "pa": p[2],
                    "outcome": outcome,
                    "rps": rps(p, outcome),
                    "logloss": -float(np.log(max(p[outcome], 1e-12))),
                    "brier": float(sum((p[i] - (1.0 if i == outcome else 0.0)) ** 2
                                       for i in range(3))),
                    "correct": int(np.argmax(p) == outcome),
                })
    return pd.DataFrame(rows)


def calibration_bins(df: pd.DataFrame, n_bins=8):
    """Pool every (predicted prob, did it happen) pair across outcomes."""
    preds, obs = [], []
    for r in df.itertuples(index=False):
        for i, p in enumerate((r.ph, r.pd, r.pa)):
            preds.append(p)
            obs.append(1.0 if r.outcome == i else 0.0)
    preds, obs = np.array(preds), np.array(obs)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (preds >= lo) & (preds < hi)
        if m.sum() >= 5:
            bins.append({"mid": float((lo + hi) / 2),
                         "predicted": float(preds[m].mean()),
                         "observed": float(obs[m].mean()),
                         "n": int(m.sum())})
    return bins


def main():
    results = fetch_results(use_cache=True)
    df = evaluate(results)

    summary = {}
    for w in WEIGHTS:
        sub = df[df["w"] == w]
        summary[str(w)] = {
            "rps": float(sub["rps"].mean()),
            "logloss": float(sub["logloss"].mean()),
            "brier": float(sub["brier"].mean()),
            "accuracy": float(sub["correct"].mean()),
            "n": int(len(sub)),
            "per_tournament": {
                t: {"rps": float(g["rps"].mean()),
                    "accuracy": float(g["correct"].mean()), "n": int(len(g))}
                for t, g in sub.groupby("tournament")
            },
        }
    best_w = min(summary, key=lambda w: summary[w]["rps"])

    out = {
        "weights_tested": WEIGHTS,
        "best_w_dc": float(best_w),
        "summary": summary,
        "calibration": calibration_bins(df[df["w"] == float(best_w)]),
        "baselines_note": ("w=0.0 is the pure Elo goal model; w=1.0 is pure "
                           "Dixon-Coles; intermediate values blend expected "
                           "goals geometrically."),
        "matches_evaluated": int(len(df) / len(WEIGHTS)),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)

    print(f"matches evaluated: {out['matches_evaluated']}")
    for w in map(str, WEIGHTS):
        s = summary[w]
        print(f"  w_dc={w:>4}: RPS={s['rps']:.4f} logloss={s['logloss']:.4f} "
              f"acc={s['accuracy']:.3f}")
    print(f"best w_dc by RPS: {best_w}")


if __name__ == "__main__":
    main()
