"""Monte Carlo simulation of the 2026 World Cup (48 teams, 104 matches).

Format: 12 groups of 4; top two per group plus the 8 best third-placed
teams reach a round of 32, then single-elimination to the final.

Third-placed teams are assigned to their bracket slots by solving a
bipartite matching against each slot's allowed-group constraint (FIFA's
Annex C picks one specific assignment per combination; any constraint-
respecting matching is an excellent approximation for forecasting).
"""
import math
import random
from collections import defaultdict

import numpy as np

from .teams import GROUPS, R32, R16, QF, SF, THIRD_SLOTS


def _sample_scores(matrix, n, rng):
    """n samples of (home_goals, away_goals) from a score matrix."""
    side = matrix.shape[0]
    flat = matrix.ravel()
    idx = rng.choice(len(flat), size=n, p=flat / flat.sum())
    return idx // side, idx % side


def _rank_group(teams, stats, h2h_results, rng):
    """Order 4 teams by FIFA tiebreakers: pts, GD, GF, head-to-head
    (pts/GD/GF among the tied subset), then random (drawing of lots)."""

    def overall_key(t):
        s = stats[t]
        return (-s[0], -(s[1] - s[2]), -s[1])

    ordered = sorted(teams, key=lambda t: (overall_key(t), rng.random()))
    # refine ties via head-to-head among tied subsets
    out = []
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and overall_key(ordered[j]) == overall_key(ordered[i]):
            j += 1
        tied = ordered[i:j]
        if len(tied) > 1:
            sub = defaultdict(lambda: [0, 0, 0])  # pts, gf, ga within subset
            tied_set = set(tied)
            for (a, b), (ga_, gb_) in h2h_results.items():
                if a in tied_set and b in tied_set:
                    sub[a][1] += ga_; sub[a][2] += gb_
                    sub[b][1] += gb_; sub[b][2] += ga_
                    if ga_ > gb_:
                        sub[a][0] += 3
                    elif gb_ > ga_:
                        sub[b][0] += 3
                    else:
                        sub[a][0] += 1; sub[b][0] += 1
            tied = sorted(tied, key=lambda t: (-sub[t][0],
                                               -(sub[t][1] - sub[t][2]),
                                               -sub[t][1], rng.random()))
        out.extend(tied)
        i = j
    return out


def _assign_thirds(qualified_groups, rng):
    """Match the 8 qualified third-place groups to the 8 constrained slots.

    Returns {match_number: group_letter} or None if no perfect matching.
    Backtracking over slots ordered by most-constrained-first.
    """
    slots = sorted(THIRD_SLOTS, key=lambda m: len(THIRD_SLOTS[m] & set(qualified_groups)))
    assignment = {}
    used = set()

    def bt(i):
        if i == len(slots):
            return True
        m = slots[i]
        opts = [g for g in qualified_groups
                if g in THIRD_SLOTS[m] and g not in used]
        rng.shuffle(opts)
        for g in opts:
            assignment[m] = g
            used.add(g)
            if bt(i + 1):
                return True
            used.discard(g)
            del assignment[m]
        return False

    return assignment if bt(0) else None


def _ko_winner(home, away, match_no, prob_fn, rng):
    """Sample the winner of a knockout tie (90' + ET/pens for draws)."""
    ph, pd_, pa = prob_fn(home, away, match_no)
    u = rng.random()
    if u < ph:
        return home
    if u < ph + pa:
        return away
    # drawn after 90': winner leans on relative strength, pulled toward 50/50
    q = ph / (ph + pa) if (ph + pa) > 0 else 0.5
    q = 0.5 + 0.8 * (q - 0.5)
    return home if rng.random() < q else away


def simulate_tournament(group_fixtures, prob_fn, n_sims=10000, seed=42):
    """Run the full-tournament Monte Carlo.

    group_fixtures: list of dicts {home, away, group, matrix (score matrix)
                    or result (gh, ga) when already played}.
    prob_fn(home, away, match_no) -> (ph, pd, pa) for a knockout match in
    90 minutes; match_no lets the caller apply venue-specific home advantage.

    Returns {team: {"group_win": p, "advance": p, "r16": p, "qf": p,
                    "sf": p, "final": p, "champion": p}} and
            per-group position distributions.
    """
    rng_master = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # Pre-sample scores for all unplayed group matches.
    samples = {}
    for k, fx in enumerate(group_fixtures):
        if fx.get("result") is None:
            samples[k] = _sample_scores(fx["matrix"], n_sims, rng_master)

    counters = defaultdict(lambda: defaultdict(float))
    pos_counts = defaultdict(lambda: np.zeros(4))

    all_teams = [t for g in GROUPS.values() for t in g]

    for s in range(n_sims):
        # --- group stage ---
        stats = {t: [0, 0, 0] for t in all_teams}  # pts, gf, ga
        h2h = {}
        for k, fx in enumerate(group_fixtures):
            if fx.get("result") is not None:
                gh, ga = fx["result"]
            else:
                gh, ga = samples[k][0][s], samples[k][1][s]
            h, a = fx["home"], fx["away"]
            h2h[(h, a)] = (gh, ga)
            stats[h][1] += gh; stats[h][2] += ga
            stats[a][1] += ga; stats[a][2] += gh
            if gh > ga:
                stats[h][0] += 3
            elif ga > gh:
                stats[a][0] += 3
            else:
                stats[h][0] += 1; stats[a][0] += 1

        placements = {}  # "1A" -> team, etc.
        thirds = []      # (group, team)
        for g, teams in GROUPS.items():
            order = _rank_group(teams, stats, h2h, py_rng)
            placements[f"1{g}"] = order[0]
            placements[f"2{g}"] = order[1]
            thirds.append((g, order[2]))
            for pos, t in enumerate(order):
                pos_counts[t][pos] += 1

        # rank thirds: pts, GD, GF, lots
        thirds.sort(key=lambda gt: (-stats[gt[1]][0],
                                    -(stats[gt[1]][1] - stats[gt[1]][2]),
                                    -stats[gt[1]][1], py_rng.random()))
        qualified = thirds[:8]
        third_groups = [g for g, _ in qualified]
        third_team = dict(qualified)
        assign = _assign_thirds(third_groups, py_rng)
        if assign is None:  # no valid matching: relax constraints
            assign = dict(zip(sorted(THIRD_SLOTS), sorted(third_groups)))

        for g, t in qualified:
            counters[t]["advance"] += 1
        for g in GROUPS:
            counters[placements[f"1{g}"]]["group_win"] += 1
            counters[placements[f"1{g}"]]["advance"] += 1
            counters[placements[f"2{g}"]]["advance"] += 1

        # --- knockout ---
        winners = {}
        for m, (s1, s2) in R32.items():
            t1 = third_team[assign[m]] if s1.startswith("3") else placements[s1]
            t2 = third_team[assign[m]] if s2.startswith("3") else placements[s2]
            winners[m] = _ko_winner(t1, t2, m, prob_fn, py_rng)
        for rnd, label in ((R16, "r16"), (QF, "qf"), (SF, "sf")):
            for m, (m1, m2) in rnd.items():
                t1, t2 = winners[m1], winners[m2]
                counters[t1][label] += 1
                counters[t2][label] += 1
                winners[m] = _ko_winner(t1, t2, m, prob_fn, py_rng)
        f1, f2 = winners[101], winners[102]
        counters[f1]["final"] += 1
        counters[f2]["final"] += 1
        champ = _ko_winner(f1, f2, 104, prob_fn, py_rng)
        counters[champ]["champion"] += 1

    out = {}
    for t in all_teams:
        c = counters[t]
        out[t] = {k: c.get(k, 0.0) / n_sims
                  for k in ("group_win", "advance", "r16", "qf", "sf",
                            "final", "champion")}
        out[t]["positions"] = list(pos_counts[t] / n_sims)
    return out
