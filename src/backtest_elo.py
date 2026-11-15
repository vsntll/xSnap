"""Elo-only baseline, back-tested against the closing line.

Tune K and home-field advantage on 2002-14 (1999-2001 is rating burn-in), score
every played game from 2015 on next to the de-vigged closing moneyline and the
closing spread. Playoffs included. Ties count as a 0.5 outcome for model and
market alike. Writes data/processed/model_metrics.json.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from elo import run_elo

ROOT = Path(__file__).resolve().parent.parent
GAMES = ROOT / "data" / "processed" / "games_all.csv"
OUT = ROOT / "data" / "processed" / "model_metrics.json"
TUNE, TEST_FROM = (2002, 2014), 2015


def ml_prob(ml: pd.Series) -> pd.Series:
    """American moneyline -> implied probability (with vig)."""
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def log_loss(p: pd.Series, y: pd.Series) -> float:
    p = p.clip(1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def score(df: pd.DataFrame) -> dict:
    df = df[df["completed"]]
    return {"games": len(df), "log_loss": log_loss(df["p_home"], df["home_win"]),
            "brier": float(((df["p_home"] - df["home_win"]) ** 2).mean()),
            "margin_mae": float((df["elo_margin"] - df["home_margin"]).abs().mean())}


def main() -> None:
    games = pd.read_csv(GAMES, low_memory=False)
    tune = games["season"].between(*TUNE)
    best = min(itertools.product([10, 15, 20, 25, 30], [30, 40, 48, 55, 65]),
               key=lambda kh: score(games.merge(run_elo(games, *kh), on="game_id")[tune])["log_loss"])
    df = games.merge(run_elo(games, *best), on="game_id")
    # score model and market on the identical game set
    test = df[df["completed"] & (df["season"] >= TEST_FROM)].dropna(
        subset=["close_ml_home", "close_ml_away", "close_spread_home"])

    ph, pa = ml_prob(test["close_ml_home"]), ml_prob(test["close_ml_away"])
    market = test.assign(p_home=ph / (ph + pa), elo_margin=test["close_spread_home"])
    metrics = {
        "elo": {"k": best[0], "hfa": best[1], "test_seasons": f"{TEST_FROM}-{int(test['season'].max())}",
                **score(test)},
        "market_close": score(market),
    }
    # keep other models' sections (train_outcome_model.py writes outcome_v1)
    metrics = {**(json.loads(OUT.read_text()) if OUT.exists() else {}), **metrics}
    OUT.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
