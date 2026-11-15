"""Local refresh, in place of the GitHub Actions pipeline: pull, rebuild, backtest,
freeze predictions, write docs/index.html, run the tests. With --push, commit the
published files and push so GitHub Pages (master /docs) serves the new page.

Run:  .venv\\Scripts\\python src/update.py [--push]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# check_model_metrics.py is the regression gate: a worse model stops the run before the ledger and site
STEPS = ["pull_nflverse.py", "build_games_all.py", "backtest_elo.py", "train_outcome_model.py",
         "build_player_value.py", "train_value_model.py", "build_player_grades.py",
         "check_model_metrics.py", "prediction_ledger.py", "export_site.py"]
PUBLISHED = ["data/processed/model_metrics.json", "data/processed/predictions_ledger.csv", "docs/index.html"]


def run(*cmd: str) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="commit the published files and git push")
    push = ap.parse_args().push
    for step in STEPS:
        run(sys.executable, f"src/{step}")
    run(sys.executable, "-m", "pytest", "-q", "tests")  # nothing is published if a check fails
    if not push:
        return
    run("git", "add", *PUBLISHED)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print("nothing changed, not committing")
        return
    run("git", "commit", "-m", f"Refresh predictions and site for {date.today()}")
    run("git", "push")


if __name__ == "__main__":
    main()
