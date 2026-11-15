"""Step 6: on-field value per player-season, QB and skill positions (v1)
-> data/processed/player_season_value.csv.

value = EPA - replacement EPA rate x opportunities, regular season only.
  QB:      passing + rushing EPA over dropbacks (attempts + sacks) + carries
  RB/WR/TE: rushing + receiving EPA over carries + targets
Replacement rate: EPA per opportunity of the players ranked N+1..2N by volume at
the position that season (N = starters league-wide).
shortcut: passer and receiver both get the full EPA of a completed pass (each is
judged only against its own position's replacement); split the credit when values
must sum across positions (roster availability, milestone 5).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from elo import FRANCHISE

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
METRICS = PROC / "model_metrics.json"
STARTERS = {"QB": 32, "RB": 64, "WR": 96, "TE": 48}
FIRST_SEASON = 2009  # nflverse player stats have ~no targets for 2003-08


def opportunities(w: pd.DataFrame) -> pd.DataFrame:
    w = w.fillna({c: 0 for c in ["attempts", "sacks_suffered", "carries", "targets",
                                 "passing_epa", "rushing_epa", "receiving_epa"]})
    qb = w["pos"] == "QB"
    return w.assign(
        opp=(w["attempts"] + w["sacks_suffered"] + w["carries"]).where(qb, w["carries"] + w["targets"]),
        epa=(w["passing_epa"] + w["rushing_epa"]).where(qb, w["rushing_epa"] + w["receiving_epa"]),
    )


def replacement_rates(seasons: pd.DataFrame) -> pd.Series:
    """EPA per opportunity of the (N, 2N] tier by volume, per (season, pos)."""
    ranked = seasons.assign(rank=seasons.groupby(["season", "pos"])["opp"].rank(ascending=False, method="first"))
    n = ranked["pos"].map(STARTERS)
    tier = ranked[(ranked["rank"] > n) & (ranked["rank"] <= 2 * n)].groupby(["season", "pos"])[["epa", "opp"]].sum()
    return (tier["epa"] / tier["opp"]).rename("repl_rate")


def build(weeks: pd.DataFrame, bio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (player_season, player_team_season); both carry value."""
    pos = bio.set_index("gsis_id")["position"]
    weeks = weeks[(weeks["season_type"] == "REG") & (weeks["season"] >= FIRST_SEASON)]
    w = weeks.assign(pos=weeks["player_id"].map(pos).fillna(weeks["position"]))
    w = opportunities(w[w["pos"].isin(STARTERS)])
    keys = ["season", "player_id", "pos"]
    ps = w.groupby(keys).agg(name=("player_display_name", "last"), team=("team", "last"), games=("game_id", "nunique"),
                             opp=("opp", "sum"), epa=("epa", "sum")).reset_index()
    rates = replacement_rates(ps)
    pts = w.groupby([*keys, "team"])[["opp", "epa"]].sum().reset_index()
    out = []
    for df in (ps, pts):
        df = df.join(rates, on=["season", "pos"])
        out.append(df.assign(rate=df["epa"] / df["opp"].where(df["opp"] > 0),
                             value=df["epa"] - df["repl_rate"] * df["opp"]))
    return out[0], out[1]


def load_weeks() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "player_week_*.parquet")))
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def validate(ps: pd.DataFrame, pts: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Year-to-year stability per position (a value that doesn't persist is noise)
    and team-season value totals vs regular-season point differential."""
    min_opp = {"QB": 200, "RB": 100, "WR": 50, "TE": 40}
    q = ps[ps["opp"] >= ps["pos"].map(min_opp)]
    pairs = q.merge(q.assign(season=q["season"] - 1), on=["player_id", "pos", "season"], suffixes=("", "_next"))
    yoy = {p: {"pairs": len(g), "rate_r": round(g["rate"].corr(g["rate_next"]), 3),
               "value_r": round(g["value"].corr(g["value_next"]), 3)} for p, g in pairs.groupby("pos")}

    reg = games[games["completed"] & (games["game_type"] == "REG")]
    diff = pd.concat([reg.assign(team=reg["home_team"], pd=reg["home_margin"]),
                      reg.assign(team=reg["away_team"], pd=-reg["home_margin"])])
    diff = diff.assign(team=diff["team"].replace(FRANCHISE)).groupby(["season", "team"])["pd"].sum()
    teams = pts.assign(team=pts["team"].replace(FRANCHISE))
    team_val = teams.groupby(["season", "team"])["value"].sum().to_frame("all").join(
        teams[teams["pos"] == "QB"].groupby(["season", "team"])["value"].sum().rename("qb"))
    t = team_val.join(diff, how="inner")
    return {"seasons": f"{int(ps['season'].min())}-{int(ps['season'].max())}", "yoy": yoy,
            "team_point_diff_r": {"all": round(t["all"].corr(t["pd"]), 3), "qb": round(t["qb"].corr(t["pd"]), 3),
                                  "team_seasons": len(t)}}


def main() -> None:
    ps, pts = build(load_weeks(), pd.read_parquet(RAW / "players.parquet"))
    ps.to_csv(PROC / "player_season_value.csv", index=False)
    print(f"player_season_value: {len(ps)} player-seasons -> {PROC / 'player_season_value.csv'}")
    games = pd.read_csv(PROC / "games_all.csv", low_memory=False)
    done = ps["season"] < games.loc[~games["completed"], "season"].min()  # finished seasons only
    check = validate(ps[done], pts[pts["season"].isin(ps.loc[done, "season"])], games)
    metrics = json.loads(METRICS.read_text())
    metrics["value_definition"] = check
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
