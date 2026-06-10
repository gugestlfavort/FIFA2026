# World Cup 2026 prediction pipeline

Predicts every match of the 2026 FIFA World Cup daily; dashboard on GitHub
Pages, refreshed by GitHub Actions at 04:30 UTC and on demand
(Actions → "Daily predictions" → Run workflow).

## Commands
- `./.venv/bin/python -m pipeline.run` — full daily run (needs
  `ODDS_API_KEY` env var; without it, runs model-only, no market blend).
  Writes `docs/data/predictions.json` + updates `docs/data/predictions_log.json`.
- `./.venv/bin/python -m pipeline.backtest` — refit + score on WC22/Euro24/
  Copa24 group stages; writes `docs/data/backtest.json` (committed; `run.py`
  reads `best_w_dc` from it). Only needs re-running if the model changes.
- Local dashboard: `python3 -m http.server -d docs 8000`.

## Architecture
- `pipeline/teams.py` — groups A–L, name aliases (canonical = martj42
  dataset spelling), venues, knockout bracket (FIFA match numbers 73–104),
  third-place slot constraints.
- `pipeline/sources.py` — fetchers: martj42 results CSV (training data),
  ESPN scoreboard (schedule/results/venues), The Odds API (h2h + outright,
  median across EU books, de-vigged).
- `pipeline/model.py` — self-computed World Football Elo; time-decayed
  Dixon-Coles via weighted Poisson GLM (statsmodels) + rho correction;
  Elo-based goal model; geometric lambda blend (`w_dc` from backtest).
- `pipeline/simulate.py` — 10k Monte Carlo: group tiebreakers (FIFA order
  incl. head-to-head), best-8 thirds via constraint matching (approximates
  FIFA Annex C), knockout with ET/pens draw split.
- `pipeline/run.py` — orchestrator. Market blend weight `W_MARKET=0.5`.
  Freezes pre-kickoff predictions in the log; scores them when final
  (live RPS/accuracy KPIs on the dashboard).

## Gotchas
- The results dataset contains scheduled (unplayed) rows — `fetch_results`
  drops NaN scores. ESPN final scores are merged in for matches the dataset
  hasn't picked up yet (`merge_espn_results`).
- Team names differ per source; always go through `teams.canon()`.
- `predictions_log.json` is the persistent track record — never delete it;
  CI commits it back each run.
- Knockout venue→match-number mapping assumes ESPN lists KO matches
  chronologically (only used for host home advantage, low stakes).
