# Running the pipeline

Python 3.11, in a venv:

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

| Step | Command | Output |
| --- | --- | --- |
| Pull nflverse tables | `.venv\Scripts\python src/pull_nflverse.py` | `data/raw/schedules.parquet` (gitignored cache) |
| Unified game table | `.venv\Scripts\python src/build_games_all.py` | `data/processed/games_all.csv` (rebuilt each run, gitignored) |
| Elo floor + outcome v1 | `.venv\Scripts\python src/backtest_elo.py`, `src/train_outcome_model.py` | `model_metrics.json`, `outcome_predictions.csv` (gitignored) |
| Player value + projections | `.venv\Scripts\python src/build_player_value.py`, `src/train_value_model.py` | `player_season_value.csv` (with `teams` path), `player_team_stints.csv`, `value_projections.csv`, `aging_curves.csv` (gitignored) |
| Season grades (QB, RB, WR, TE, K, D/ST) | `.venv\Scripts\python src/build_player_grades.py` | `player_grades.csv` (gitignored): rank, percentile, letter, WAR, per-game impact, `teams` path per position-season |
| Regression gate | `.venv\Scripts\python src/check_model_metrics.py [--accept]` | fails if a model got worse than HEAD or `accepted_metrics.json` |
| Ledger + site | `.venv\Scripts\python src/prediction_ledger.py`, `src/export_site.py` | `predictions_ledger.csv` (append-only), `docs/index.html` |
| Checks | `.venv\Scripts\python -m pytest -q tests` | data invariants + unit tests |

**All at once:** `src/update.py` runs every step plus the tests; `--push` (or double-click
`update.bat`) also commits the published files and pushes. GitHub Pages serves `docs/` on `master`.
Run it Tuesday and Friday mornings, after nflverse's nightly refresh.
