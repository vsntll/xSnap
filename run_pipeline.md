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
| — | Checks | `.venv\Scripts\python -m pytest -q tests` | data invariants + unit tests |

Step numbers follow the full plan; the missing steps arrive with later milestones.
