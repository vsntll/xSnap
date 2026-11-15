"""Step 1: pull nflverse tables into the data/raw/ parquet cache.

shortcut: schedules only - that is all build_games_all.py needs; add play-by-play,
player stats and snap counts when the outcome v1 / value models need them.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    df = nfl.load_schedules()  # every season since 1999, incl. the current one
    out = RAW / "schedules.parquet"
    df.write_parquet(out)
    print(f"schedules: {df.height} games, seasons {df['season'].min()}-{df['season'].max()} -> {out}")


if __name__ == "__main__":
    main()
