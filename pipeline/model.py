"""Rating and match-prediction models.

Three layers, blended:
1. Elo ratings computed from the full results history (World Football Elo
   formula: K by match importance, goal-difference multiplier, +100 home).
2. Time-decayed Dixon-Coles: per-team attack/defence via weighted Poisson
   GLM, plus the low-score correlation correction (rho).
3. Market odds (handled in run.py): de-vigged 1x2 probabilities blended
   into the model's, with the score matrix reweighted to stay consistent.
"""
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

MAX_GOALS = 10


# --- Elo ---------------------------------------------------------------------

def _k_factor(tournament: str) -> float:
    t = tournament.lower()
    if "fifa world cup" in t and "qualification" not in t:
        return 60.0
    finals = ("uefa euro", "copa américa", "copa america", "african cup",
              "africa cup", "afc asian cup", "gold cup", "oceania nations",
              "confederations cup")
    if any(f in t for f in finals) and "qualification" not in t:
        return 50.0
    if "qualification" in t or "nations league" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 30.0


def compute_elo(results: pd.DataFrame, start: float = 1500.0) -> dict:
    """Elo rating per team after the last match in `results`."""
    ratings = {}
    home_adv = 100.0
    for row in results.itertuples(index=False):
        rh = ratings.get(row.home_team, start)
        ra = ratings.get(row.away_team, start)
        dr = rh - ra + (0.0 if row.neutral else home_adv)
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        gh, ga = row.home_score, row.away_score
        w = 1.0 if gh > ga else 0.0 if gh < ga else 0.5
        diff = abs(gh - ga)
        g = 1.0 if diff <= 1 else 1.5 if diff == 2 else 1.75 + (diff - 3) / 8.0
        delta = _k_factor(row.tournament) * g * (w - we)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
    return ratings


# --- Dixon-Coles -------------------------------------------------------------

def _long_format(df: pd.DataFrame, ref_date, half_life_days: float):
    age = (ref_date - df["date"]).dt.days.clip(lower=0)
    w = 0.5 ** (age / half_life_days)
    home_flag = (~df["neutral"]).astype(float)
    a = pd.DataFrame({"team": df["home_team"], "opp": df["away_team"],
                      "goals": df["home_score"], "home": home_flag, "w": w})
    b = pd.DataFrame({"team": df["away_team"], "opp": df["home_team"],
                      "goals": df["away_score"], "home": 0.0, "w": w})
    return pd.concat([a, b], ignore_index=True)


class DixonColes:
    def __init__(self, attack, defence, intercept, home_adv, rho, teams):
        self.attack = attack            # dict team -> coef
        self.defence = defence          # dict team -> coef (higher = leakier)
        self.intercept = intercept
        self.home_adv = home_adv
        self.rho = rho
        self.teams = teams

    def lambdas(self, home, away, home_advantage: float = 0.0):
        """Expected goals (lh, la); home_advantage in [0,1] scales the
        fitted home coefficient (0 = neutral venue)."""
        lh = math.exp(self.intercept + self.attack[home] + self.defence[away]
                      + home_advantage * self.home_adv)
        la = math.exp(self.intercept + self.attack[away] + self.defence[home])
        return lh, la


def fit_dixon_coles(results: pd.DataFrame, ref_date, years: float = 10.0,
                    half_life_days: float = 730.0,
                    min_matches: int = 15) -> DixonColes:
    cutoff = ref_date - pd.Timedelta(days=int(365.25 * years))
    df = results[(results["date"] >= cutoff) & (results["date"] < ref_date)]
    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    keep = set(counts[counts >= min_matches].index)
    df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)].copy()

    long = _long_format(df, ref_date, half_life_days)
    teams = sorted(set(long["team"]))
    t_idx = {t: i for i, t in enumerate(teams)}
    n, k = len(long), len(teams)

    # Design: intercept | home | attack dummies (ref team 0) | defence dummies
    X = np.zeros((n, 2 + 2 * (k - 1)))
    X[:, 0] = 1.0
    X[:, 1] = long["home"].to_numpy()
    ti = long["team"].map(t_idx).to_numpy()
    oi = long["opp"].map(t_idx).to_numpy()
    rows = np.arange(n)
    m_att = ti > 0
    X[rows[m_att], 1 + ti[m_att]] = 1.0
    m_def = oi > 0
    X[rows[m_def], k + oi[m_def]] = 1.0

    glm = sm.GLM(long["goals"].to_numpy(), X, family=sm.families.Poisson(),
                 freq_weights=long["w"].to_numpy())
    res = glm.fit()
    coef = res.params

    attack = {teams[0]: 0.0}
    defence = {teams[0]: 0.0}
    for t, i in t_idx.items():
        if i > 0:
            attack[t] = coef[1 + i]
            defence[t] = coef[k + i]
    model = DixonColes(attack, defence, coef[0], coef[1], 0.0, set(teams))
    model.rho = _fit_rho(df, model, ref_date, half_life_days)
    return model


def _tau(x, y, lh, la, rho):
    if x == 0 and y == 0:
        return 1.0 - lh * la * rho
    if x == 0 and y == 1:
        return 1.0 + lh * rho
    if x == 1 and y == 0:
        return 1.0 + la * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _fit_rho(df, model: DixonColes, ref_date, half_life_days):
    sub = df[(df["home_score"] <= 1) & (df["away_score"] <= 1)]
    age = (ref_date - sub["date"]).dt.days.clip(lower=0)
    ws = (0.5 ** (age / half_life_days)).to_numpy()
    lams = np.array([model.lambdas(r.home_team, r.away_team,
                                   0.0 if r.neutral else 1.0)
                     for r in sub.itertuples(index=False)])
    gh = sub["home_score"].to_numpy()
    ga = sub["away_score"].to_numpy()

    def nll(rho):
        taus = np.array([_tau(x, y, lh, la, rho)
                         for x, y, (lh, la) in zip(gh, ga, lams)])
        taus = np.clip(taus, 1e-10, None)
        return -np.sum(ws * np.log(taus))

    res = minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)


# --- Elo-based goal model ----------------------------------------------------

class EloGoalModel:
    """lambda = exp(c + b * elo_diff/400 + h * home)."""

    def __init__(self, c, b, h, elo):
        self.c, self.b, self.h = c, b, h
        self.elo = elo

    def lambdas(self, home, away, home_advantage: float = 0.0):
        d = (self.elo.get(home, 1500.0) - self.elo.get(away, 1500.0)) / 400.0
        lh = math.exp(self.c + self.b * d + home_advantage * self.h)
        la = math.exp(self.c - self.b * d)
        return lh, la


def fit_elo_goal_model(results: pd.DataFrame, elo_at_match: pd.DataFrame,
                       ref_date, years: float = 8.0,
                       half_life_days: float = 1095.0) -> EloGoalModel:
    """elo_at_match must carry pre-match elo_home/elo_away per result row."""
    cutoff = ref_date - pd.Timedelta(days=int(365.25 * years))
    m = (elo_at_match["date"] >= cutoff) & (elo_at_match["date"] < ref_date)
    df = elo_at_match[m]
    age = (ref_date - df["date"]).dt.days.clip(lower=0)
    w = (0.5 ** (age / half_life_days)).to_numpy()
    d = ((df["elo_home"] - df["elo_away"]) / 400.0).to_numpy()
    home = (~df["neutral"]).astype(float).to_numpy()

    y = np.concatenate([df["home_score"], df["away_score"]])
    X = np.column_stack([
        np.ones(2 * len(df)),
        np.concatenate([d, -d]),
        np.concatenate([home, np.zeros(len(df))]),
    ])
    res = sm.GLM(y, X, family=sm.families.Poisson(),
                 freq_weights=np.concatenate([w, w])).fit()
    c, b, h = res.params
    final_elo = {}
    return EloGoalModel(c, b, h, final_elo)


def elo_history(results: pd.DataFrame, start: float = 1500.0) -> pd.DataFrame:
    """Results frame + pre-match elo_home/elo_away columns (single pass)."""
    ratings = {}
    eh, ea = [], []
    home_adv = 100.0
    for row in results.itertuples(index=False):
        rh = ratings.get(row.home_team, start)
        ra = ratings.get(row.away_team, start)
        eh.append(rh)
        ea.append(ra)
        dr = rh - ra + (0.0 if row.neutral else home_adv)
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        gh, ga = row.home_score, row.away_score
        w = 1.0 if gh > ga else 0.0 if gh < ga else 0.5
        diff = abs(gh - ga)
        g = 1.0 if diff <= 1 else 1.5 if diff == 2 else 1.75 + (diff - 3) / 8.0
        delta = _k_factor(row.tournament) * g * (w - we)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
    out = results.copy()
    out["elo_home"] = eh
    out["elo_away"] = ea
    return out


# --- Blending and probabilities ----------------------------------------------

class BlendModel:
    """Geometric blend of Dixon-Coles and Elo-model expected goals."""

    def __init__(self, dc: DixonColes, eg: EloGoalModel, w_dc: float):
        self.dc, self.eg, self.w_dc = dc, eg, w_dc

    def lambdas(self, home, away, home_advantage: float = 0.0):
        le = self.eg.lambdas(home, away, home_advantage)
        if home in self.dc.teams and away in self.dc.teams:
            ld = self.dc.lambdas(home, away, home_advantage)
            w = self.w_dc
            return (math.exp(w * math.log(ld[0]) + (1 - w) * math.log(le[0])),
                    math.exp(w * math.log(ld[1]) + (1 - w) * math.log(le[1])))
        return le

    @property
    def rho(self):
        return self.dc.rho


def score_matrix(lh, la, rho, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(home goals = i, away goals = j), Dixon-Coles corrected."""
    ph = poisson.pmf(np.arange(max_goals + 1), lh)
    pa = poisson.pmf(np.arange(max_goals + 1), la)
    m = np.outer(ph, pa)
    m[0, 0] *= max(1.0 - lh * la * rho, 1e-10)
    m[0, 1] *= 1.0 + lh * rho
    m[1, 0] *= 1.0 + la * rho
    m[1, 1] *= max(1.0 - rho, 1e-10)
    return m / m.sum()


def outcome_probs(m: np.ndarray):
    """(p_home, p_draw, p_away) from a score matrix."""
    return (float(np.tril(m, -1).sum()), float(np.trace(m)),
            float(np.triu(m, 1).sum()))


def reweight_matrix(m: np.ndarray, target):
    """Rescale H/D/A regions of the score matrix to match target 1x2 probs."""
    ph, pd_, pa = outcome_probs(m)
    th, td, ta = target
    out = m.copy()
    il = np.tril_indices_from(m, -1)
    iu = np.triu_indices_from(m, 1)
    id_ = np.diag_indices_from(m)
    if ph > 0:
        out[il] *= th / ph
    if pd_ > 0:
        out[id_] *= td / pd_
    if pa > 0:
        out[iu] *= ta / pa
    return out / out.sum()
