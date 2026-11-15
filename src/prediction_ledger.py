"""Step 12a: freeze the next week's predictions into an append-only ledger,
data/processed/predictions_ledger.csv, for every published model (elo, outcome_v1).
A (model, game) gets a new row only when its prediction changed since its last
snapshot, and never once the game has kicked off, so re-running is idempotent."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from elo import run_elo

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
LEDGER = PROC / "predictions_ledger.csv"
COLS = ["snapshot_utc", "model", "game_id", "season", "week", "home_team", "away_team",
        "p_home", "pred_margin", "close_spread_home"]


def kickoff_utc(df: pd.DataFrame) -> pd.Series:
    """nflverse gameday + gametime are US Eastern."""
    et = pd.to_datetime(df["date"] + " " + df["gametime"].fillna("00:00"))
    return et.dt.tz_localize("America/New_York").dt.tz_convert("UTC")


def next_week(games: pd.DataFrame) -> pd.DataFrame:
    """Unplayed games of the earliest week that still has unplayed games."""
    todo = games[~games["completed"]]
    if todo.empty:
        return todo
    season, week = todo.sort_values(["season", "week"]).iloc[0][["season", "week"]]
    return todo[(todo["season"] == season) & (todo["week"] == week)]


def week_predictions(games: pd.DataFrame, k: float, hfa: float) -> pd.DataFrame:
    """Next week's games with elo_* and v1_* predictions (v1 empty if not trained)."""
    elo = run_elo(games, k, hfa).rename(columns={"p_home": "elo_p_home", "elo_margin": "elo_pred_margin"})
    week = next_week(games).merge(elo[["game_id", "elo_p_home", "elo_pred_margin"]], on="game_id")
    v1 = PROC / "outcome_predictions.csv"
    if v1.exists():
        week = week.merge(pd.read_csv(v1).add_prefix("v1_").rename(columns={"v1_game_id": "game_id"}),
                          on="game_id", how="left")
    return week


def append(ledger: pd.DataFrame, preds: pd.DataFrame, now: pd.Timestamp, model: str) -> pd.DataFrame:
    preds = preds[kickoff_utc(preds) > now]  # a started game's prediction is frozen
    new = preds.assign(snapshot_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"), model=model)[COLS]
    new = new.dropna(subset=["p_home"]).round({"p_home": 4, "pred_margin": 2})
    last = ledger.sort_values("snapshot_utc").drop_duplicates(["model", "game_id"], keep="last")
    seen = new.merge(last[["model", "game_id", "p_home"]], on=["model", "game_id", "p_home"], how="left",
                     indicator=True)
    return pd.concat([ledger, new[(seen["_merge"] == "left_only").to_numpy()]], ignore_index=True)


def main() -> None:
    games = pd.read_csv(PROC / "games_all.csv", low_memory=False)
    elo = json.loads((PROC / "model_metrics.json").read_text())["elo"]
    week = week_predictions(games, elo["k"], elo["hfa"])
    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=COLS)
    now, n = pd.Timestamp.now(tz="UTC"), len(ledger)
    for model, prefix in (("elo", "elo_"), ("outcome_v1", "v1_")):
        if f"{prefix}p_home" in week:
            preds = week.rename(columns={f"{prefix}p_home": "p_home", f"{prefix}pred_margin": "pred_margin"})
            ledger = append(ledger, preds, now, model)
    ledger.to_csv(LEDGER, index=False)
    print(f"ledger: +{len(ledger) - n} rows ({len(ledger)} total) -> {LEDGER}")


if __name__ == "__main__":
    main()
