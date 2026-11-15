"""Step 12a: freeze the next week's predictions into an append-only ledger,
data/processed/predictions_ledger.csv. A game gets a new row only when its
prediction changed since its last snapshot, so re-running is idempotent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from elo import run_elo

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
LEDGER = PROC / "predictions_ledger.csv"
COLS = ["snapshot_utc", "model", "game_id", "season", "week", "home_team", "away_team",
        "p_home", "elo_margin", "close_spread_home"]


def week_predictions(games: pd.DataFrame, k: float, hfa: float) -> pd.DataFrame:
    """Elo predictions for the earliest week that still has unplayed games."""
    df = games.merge(run_elo(games, k, hfa), on="game_id")
    todo = df[~df["completed"]]
    if todo.empty:
        return todo
    season, week = todo.sort_values(["season", "week"]).iloc[0][["season", "week"]]
    return todo[(todo["season"] == season) & (todo["week"] == week)]


def append(ledger: pd.DataFrame, preds: pd.DataFrame, now: str, model: str = "elo") -> pd.DataFrame:
    new = preds.assign(snapshot_utc=now, model=model)[COLS].round({"p_home": 4, "elo_margin": 2})
    last = ledger.sort_values("snapshot_utc").drop_duplicates("game_id", keep="last")
    seen = new.merge(last[["game_id", "p_home"]], on=["game_id", "p_home"], how="left", indicator=True)
    return pd.concat([ledger, new[(seen["_merge"] == "left_only").to_numpy()]], ignore_index=True)


def main() -> None:
    games = pd.read_csv(PROC / "games_all.csv", low_memory=False)
    elo = json.loads((PROC / "model_metrics.json").read_text())["elo"]
    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=COLS)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = append(ledger, week_predictions(games, elo["k"], elo["hfa"]), now)
    out.to_csv(LEDGER, index=False)
    print(f"ledger: +{len(out) - len(ledger)} rows ({len(out)} total) -> {LEDGER}")


if __name__ == "__main__":
    main()
