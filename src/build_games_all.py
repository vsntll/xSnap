"""Step 4: one row per NFL game, results plus pre-game context, from the cached
nflverse schedules -> data/processed/games_all.csv.

Market columns are closing-type lines (nflverse schedules), so they are named
close_*. close_spread_home is the home team's expected margin (positive = home
favoured; it correlates positively with home_margin).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "schedules.parquet"
OUT = ROOT / "data" / "processed" / "games_all.csv"

RENAME = {
    "gameday": "date",
    "result": "home_margin",
    "spread_line": "close_spread_home",
    "total_line": "close_total",
    "home_moneyline": "close_ml_home",
    "away_moneyline": "close_ml_away",
}
KEEP = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "home_team", "away_team", "home_score", "away_score", "result", "total", "overtime",
    "location", "div_game", "home_rest", "away_rest", "roof", "surface", "temp", "wind",
    "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name", "home_coach", "away_coach",
    "stadium_id", "spread_line", "total_line", "home_moneyline", "away_moneyline",
]


def build(sched: pd.DataFrame) -> pd.DataFrame:
    df = sched[KEEP].rename(columns=RENAME)
    df["completed"] = df["home_score"].notna() & df["away_score"].notna()
    # 1 home win, 0 away win, 0.5 tie; NaN until played
    df["home_win"] = (df["home_margin"] > 0).astype(float) + 0.5 * (df["home_margin"] == 0)
    df.loc[~df["completed"], "home_win"] = float("nan")
    return df.sort_values(["date", "gametime", "game_id"]).reset_index(drop=True)


def main() -> None:
    df = build(pd.read_parquet(RAW))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"games_all: {len(df)} games ({int(df['completed'].sum())} played) -> {OUT}")


if __name__ == "__main__":
    main()
