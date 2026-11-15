# Running the pipeline

Python 3.11, in a venv:

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

| # | Step | Command | Output |
| --- | --- | --- | --- |
| 1 | Pull nflverse tables | `.venv\Scripts\python src/pull_nflverse.py` | `data/raw/schedules.parquet` (gitignored cache) |
| 4 | Unified game table | `.venv\Scripts\python src/build_games_all.py` | `data/processed/games_all.csv` (rebuilt each run, gitignored) |
| 10 | Elo baseline backtest | `.venv\Scripts\python src/backtest_elo.py` | `data/processed/model_metrics.json` (floor in `accepted_metrics.json`) |
| 12 | Ledger + site | `.venv\Scripts\python src/prediction_ledger.py`, `src/export_site.py` | `predictions_ledger.csv` (append-only), `docs/index.html` |
| — | Checks | `.venv\Scripts\python -m pytest -q tests` | data invariants + unit tests |

**All at once:** `src/update.py` runs every step plus the tests; `--push` (or double-click
`update.bat`) also commits the published files and pushes. GitHub Pages serves `docs/` on `master`.
Run it Tuesday and Friday mornings, after nflverse's nightly refresh.

Step numbers follow the full plan; the missing steps arrive with later milestones.
