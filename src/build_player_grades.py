"""Season grades for QB, RB, WR, TE, K and D/ST -> data/processed/player_grades.csv.

Each position-season is ranked on one season-total number, regular season only:
  QB/RB/WR/TE  value above replacement (build_player_value.py), 2009 on
  K            EPA on field goals + extra points (EPA already prices kick distance), 1999 on
  D/ST         EPA the defense saved on scrimmage plays + punt/kickoff EPA, 1999 on
Qualified players only (minimum volume); percentile within the position-season,
letter grade from it. Impact: WAR = value / points per win (fit on
team-season point differential vs wins), per game, and per 17 games. K and D/ST EPA
is measured against the league average, so their WAR is wins above average.
A traded player has one row summed over his teams; `teams` lists them in order. The current season is graded to date (partial=1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_game_model_table import load_raw
from build_player_value import build, load_weeks

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
MIN_VOLUME = {"QB": 200, "RB": 100, "WR": 50, "TE": 40, "K": 10, "D/ST": 1}  # opps, opps, ..., FG att, games
LETTERS = [(90, "A"), (70, "B"), (40, "C"), (15, "D"), (0, "F")]
COLS = ["season", "pos", "player_id", "name", "team", "teams", "games", "volume", "value"]


def kickers(k: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    # sorted so "last" is his most recent team (game_id = season_week_away_home)
    k = k.merge(reg[["game_id", "season"]], on="game_id").sort_values("game_id")
    out = k.groupby(["season", "kicker_id"]).agg(name=("name", "last"), team=("posteam", "last"),
                                                  teams=("posteam", lambda s: "/".join(dict.fromkeys(s))),
                                                  games=("game_id", "nunique"), volume=("fg_att", "sum"),
                                                  value=("k_epa", "sum")).reset_index()
    return out.rename(columns={"kicker_id": "player_id"}).assign(pos="K")[COLS]


def defenses(team_raw: pd.DataFrame, st: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    off = team_raw.merge(reg[["game_id", "season"]], on="game_id")
    # a defense's game = the opposing offense's row in that game
    opp = off.merge(off[["game_id", "posteam"]].rename(columns={"posteam": "team"}), on="game_id")
    d = opp[opp["team"] != opp["posteam"]].assign(def_epa=lambda x: -x["epa"])
    d = d.groupby(["season", "team"]).agg(volume=("game_id", "nunique"), def_epa=("def_epa", "sum"))
    s = st.merge(reg[["game_id", "season"]], on="game_id").groupby(["season", "team"])["st_epa"].sum()
    d = d.join(s).fillna({"st_epa": 0.0}).reset_index()
    return d.assign(pos="D/ST", player_id=d["team"], name=d["team"] + " D/ST", teams=d["team"], games=d["volume"],
                    value=d["def_epa"] + d["st_epa"])[COLS]


def points_per_win(reg: pd.DataFrame) -> float:
    """Slope of point differential on wins over every regular-season team-season."""
    sides = [reg.assign(team=reg["home_team"], pd=reg["home_margin"], w=reg["home_win"]),
             reg.assign(team=reg["away_team"], pd=-reg["home_margin"], w=1 - reg["home_win"])]
    ts = pd.concat(sides).groupby(["season", "team"])[["pd", "w"]].sum()
    return float(1 / np.polyfit(ts["pd"], ts["w"], 1)[0])


def impact(q: pd.DataFrame, ppw: float) -> pd.DataFrame:
    """Wins above replacement (value is in points), plus per-game and per-17-game rates."""
    war = q["value"] / ppw
    return q.assign(war=war.round(2), war_per_17=(war / q["games"] * 17).round(2),
                    value_per_game=(q["value"] / q["games"]).round(2))


def grade(df: pd.DataFrame, current: int) -> pd.DataFrame:
    q = df[df["volume"] >= df["pos"].map(MIN_VOLUME)].copy()
    g = q.groupby(["season", "pos"])["value"]
    q["rank"] = g.rank(ascending=False, method="min").astype(int)
    q["pct"] = (g.rank(pct=True) * 100).round(1)
    q["grade"] = q["pct"].map(lambda p: next(letter for cut, letter in LETTERS if p >= cut))
    q["partial"] = (q["season"] >= current).astype(int)
    return q.sort_values(["season", "pos", "rank"], ascending=[False, True, True])[
        ["season", "pos", "rank", "grade", "pct", "name", "team", "teams", "games", "volume", "value", "partial", "player_id"]]


def main() -> None:
    games = pd.read_csv(PROC / "games_all.csv", low_memory=False)
    reg = games[games["completed"] & (games["game_type"] == "REG")]
    current = int(games.loc[~games["completed"], "season"].min())
    skill, _ = build(load_weeks(), pd.read_parquet(RAW / "players.parquet"))
    skill = skill.rename(columns={"opp": "volume"})[COLS]
    k = kickers(load_raw("k"), reg)
    full = pd.read_parquet(RAW / "players.parquet").drop_duplicates("gsis_id").set_index("gsis_id")["display_name"]
    k["name"] = k["player_id"].map(full).fillna(k["name"])  # pbp only has "W.Reichard"
    allpos = pd.concat([skill, k, defenses(load_raw("team"), load_raw("st"), reg)])
    ppw = points_per_win(reg)
    print(f"points per win: {ppw:.1f}")
    out = impact(grade(allpos, current), ppw).round({"value": 1})
    out.to_csv(PROC / "player_grades.csv", index=False)
    print(f"player_grades: {len(out)} graded player-seasons -> {PROC / 'player_grades.csv'}")


if __name__ == "__main__":
    main()
