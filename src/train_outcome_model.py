"""Outcome v1 - ridge on home margin over Elo + rolling EPA + QB features,
normal error around the margin for win probability.

Walk-forward: each season is predicted by a model fit on every earlier season
(2002 on; 1999-2001 is form burn-in). Half-life and ridge alpha are picked on
2008-14 only; 2015 on is scored once, on the same games as the Elo floor.
The live fit uses every finished game and predicts the unplayed ones ->
data/processed/outcome_predictions.csv. Metrics go into model_metrics.json["outcome_v1"].
The backtest refits once per season; the live fit refits every run.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_elo import TEST_FROM, log_loss
from build_game_model_table import build, load_raw
from elo import FRANCHISE

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
METRICS = PROC / "model_metrics.json"
FIRST_TRAIN, TUNE = 2002, (2008, 2014)
FORM = [f"{s}{u}_{c}" for s in ("h_", "a_") for u in ("off", "def") for c in ("epa", "success", "pass_epa", "rush_epa")]
FEATURES = ["elo_margin", "neutral", "rest_diff", "h_qb", "a_qb", *FORM]


def fit(train: pd.DataFrame, alpha: float):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train[FEATURES], train["home_margin"])
    sigma = float(np.std(train["home_margin"] - model.predict(train[FEATURES])))
    return model, sigma


def predict(model, sigma: float, rows: pd.DataFrame) -> pd.DataFrame:
    margin = model.predict(rows[FEATURES])
    return rows.assign(pred_margin=margin, p_home=norm.cdf(margin / sigma))


def walk_forward(table: pd.DataFrame, seasons, alpha: float) -> pd.DataFrame:
    done = table[table["completed"] & (table["season"] >= FIRST_TRAIN)].dropna(subset=FEATURES)
    return pd.concat(predict(*fit(done[done["season"] < s], alpha), done[done["season"] == s]) for s in seasons)


def score(df: pd.DataFrame, p: str = "p_home", m: str = "pred_margin") -> dict:
    return {"games": len(df), "log_loss": log_loss(df[p], df["home_win"]),
            "brier": float(((df[p] - df["home_win"]) ** 2).mean()),
            "margin_mae": float((df[m] - df["home_margin"]).abs().mean())}


def main() -> None:
    games = pd.read_csv(PROC / "games_all.csv", low_memory=False)
    metrics = json.loads(METRICS.read_text())
    k, hfa = metrics["elo"]["k"], metrics["elo"]["hfa"]
    team_raw, qb_raw = load_raw("team"), load_raw("qb")
    tables = {h: build(games, team_raw, qb_raw, k, hfa, halflife=h) for h in (1, 2, 3, 4, 6, 8)}
    tune_seasons = range(TUNE[0], TUNE[1] + 1)
    halflife, alpha = min(itertools.product(tables, [10.0, 100.0, 300.0, 1000.0, 3000.0]),
                          key=lambda ha: score(walk_forward(tables[ha[0]], tune_seasons, ha[1]))["log_loss"])
    table = tables[halflife]

    last = int(table.loc[table["completed"], "season"].max())
    test = walk_forward(table, range(TEST_FROM, last + 1), alpha).dropna(
        subset=["close_ml_home", "close_ml_away", "close_spread_home"])
    # games where either starter differs from his team's previous starter: where the QB feature should matter
    test = test.assign(p_elo=1 / (1 + 10 ** (-test["elo_margin"] * 25 / 400)))
    changed = test[test["game_id"].isin(qb_changed(games))]
    metrics["outcome_v1"] = {
        "halflife": halflife, "alpha": alpha, "test_seasons": f"{TEST_FROM}-{last}", **score(test),
        "qb_change_games": {"games": len(changed), "v1_log_loss": score(changed)["log_loss"],
                            "elo_log_loss": score(changed, "p_elo", "elo_margin")["log_loss"]},
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics["outcome_v1"], indent=2))

    done = table[table["completed"] & (table["season"] >= FIRST_TRAIN)].dropna(subset=FEATURES)
    live = predict(*fit(done, alpha), table[~table["completed"]].dropna(subset=FEATURES))
    from prediction_ledger import next_week  # noqa: PLC0415 - avoids a cycle at import time
    missing = set(next_week(games)["game_id"]) - set(live["game_id"])
    assert not missing, f"next-week games with no v1 prediction (null features): {sorted(missing)}"
    live[["game_id", "p_home", "pred_margin"]].to_csv(PROC / "outcome_predictions.csv", index=False)
    print(f"outcome_predictions: {len(live)} unplayed games")


def qb_changed(games: pd.DataFrame) -> set:
    cols = ["game_id", "date", "team", "qb"]
    long = pd.concat([games[["game_id", "date", "home_team", "home_qb_id"]].set_axis(cols, axis=1),
                      games[["game_id", "date", "away_team", "away_qb_id"]].set_axis(cols, axis=1)])
    long = long.assign(team=long["team"].replace(FRANCHISE)).sort_values(["date", "game_id"])
    prev = long.groupby("team")["qb"].shift()
    return set(long.loc[prev.notna() & (prev != long["qb"]), "game_id"])


if __name__ == "__main__":
    main()
