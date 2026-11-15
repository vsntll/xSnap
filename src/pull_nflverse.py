"""Pull nflverse tables into the data/raw/ parquet cache.

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
PLAYER_COLS = ["player_id", "player_display_name", "position", "position_group", "season", "week", "season_type",
               "game_id", "team", "opponent_team", "attempts", "completions", "sacks_suffered", "carries", "targets",
               "receptions", "passing_yards", "rushing_yards", "receiving_yards", "passing_epa", "rushing_epa",
               "receiving_epa", "passing_cpoe"]
BIO_COLS = ["gsis_id", "display_name", "birth_date", "position", "position_group", "rookie_season",
            "draft_year", "draft_round", "draft_pick", "height", "weight", "headshot"]
TEAM_COLS = ["team_abbr", "team_name", "team_logo_espn", "team_color"]


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


def reduce_special_teams(pbp: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Kicker-game EPA on field goals + extra points, and team-game EPA on punts and
    kickoffs from both sides (posteam = kicking team on punts, receiving team on kickoffs)."""
    st = pbp.filter(pl.col("epa").is_not_null() & pl.col("posteam").is_not_null())
    kick = st.filter(pl.col("play_type").is_in(["field_goal", "extra_point"]) & pl.col("kicker_player_id").is_not_null())
    k = kick.group_by("game_id", "posteam", pl.col("kicker_player_id").alias("kicker_id")).agg(
        name=pl.col("kicker_player_name").last(), kicks=pl.len(),
        fg_att=(pl.col("play_type") == "field_goal").sum(), k_epa=pl.col("epa").sum())
    pk = st.filter(pl.col("play_type").is_in(["punt", "kickoff"]))
    side = [pk.select("game_id", team=pl.col("posteam"), st_epa=pl.col("epa")),
            pk.select("game_id", team=pl.col("defteam"), st_epa=-pl.col("epa"))]
    return k, pl.concat(side).group_by("game_id", "team").agg(pl.col("st_epa").sum())


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sched = nfl.load_schedules()
    sched.write_parquet(RAW / "schedules.parquet")
    latest = int(sched.filter(pl.col("result").is_not_null())["season"].max())
    print(f"schedules: {sched.height} games, seasons {sched['season'].min()}-{sched['season'].max()}")

    for season in range(FIRST_PBP, latest + 1):
        paths = {kind: RAW / f"pbp_{kind}_{season}.parquet" for kind in ("team", "qb", "k", "st")}
        if season < latest and all(p.exists() for p in paths.values()):
            continue
        pbp = nfl.load_pbp([season])
        team, qb = reduce_pbp(pbp)
        k, st = reduce_special_teams(pbp)
        for kind, df in (("team", team), ("qb", qb), ("k", k), ("st", st)):
            df.write_parquet(paths[kind])
        print(f"pbp {season}: {team.height} team-games, {qb.height} qb-games")

    for season in range(FIRST_PBP, latest + 1):
        path = RAW / f"player_week_{season}.parquet"
        if season < latest and path.exists():
            continue
        nfl.load_player_stats([season], summary_level="week").select(PLAYER_COLS).write_parquet(path)
    nfl.load_players().select(BIO_COLS).write_parquet(RAW / "players.parquet")
    nfl.load_teams().select(TEAM_COLS).write_parquet(RAW / "teams.parquet")
    print(f"player stats {FIRST_PBP}-{latest} + players bio + teams cached")


if __name__ == "__main__":
    main()
