"""Step 1: pull nflverse tables into the data/raw/ parquet cache.

- schedules.parquet: every game since 1999.
- pbp_team_<season>.parquet / pbp_qb_<season>.parquet: play-by-play reduced to
  offensive team-game and QB-game EPA totals (a season of raw pbp is ~100 MB in
  memory; the reduced files are tiny). Finished seasons are pulled once; the
  latest season is re-pulled every run.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl
import polars as pl

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
FIRST_PBP = 1999


def reduce_pbp(pbp: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Scrimmage plays only: pass/run rows, which excludes kneels, spikes, no-plays
    (penalties), special teams; two-point tries dropped."""
    plays = pbp.filter(pl.col("play_type").is_in(["pass", "run"]) & (pl.col("two_point_attempt") == 0)
                       & pl.col("epa").is_not_null() & pl.col("posteam").is_not_null())
    is_pass = pl.col("play_type") == "pass"
    team = plays.group_by("game_id", "posteam").agg(
        plays=pl.len(), epa=pl.col("epa").sum(), success=pl.col("success").sum(),
        pass_plays=is_pass.sum(), pass_epa=pl.col("epa").filter(is_pass).sum(),
        rush_plays=(~is_pass).sum(), rush_epa=pl.col("epa").filter(~is_pass).sum(),
    )
    qb = plays.filter((pl.col("qb_dropback") == 1) & pl.col("id").is_not_null()).group_by(
        "game_id", "posteam", pl.col("id").alias("qb_id")).agg(dropbacks=pl.len(), qb_epa=pl.col("qb_epa").sum())
    return team, qb


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sched = nfl.load_schedules()
    sched.write_parquet(RAW / "schedules.parquet")
    latest = int(sched.filter(pl.col("result").is_not_null())["season"].max())
    print(f"schedules: {sched.height} games, seasons {sched['season'].min()}-{sched['season'].max()}")

    for season in range(FIRST_PBP, latest + 1):
        team_path, qb_path = RAW / f"pbp_team_{season}.parquet", RAW / f"pbp_qb_{season}.parquet"
        if season < latest and team_path.exists() and qb_path.exists():
            continue
        team, qb = reduce_pbp(nfl.load_pbp([season]))
        team.write_parquet(team_path)
        qb.write_parquet(qb_path)
        print(f"pbp {season}: {team.height} team-games, {qb.height} qb-games")


if __name__ == "__main__":
    main()
