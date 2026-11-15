"""Steps 9-10a: pre-kickoff features for every game -> data/processed/game_model_table.csv.

Per team: exponentially weighted EPA/play and success rate, offense and defense,
pass and rush. Per starting QB: dropback EPA over his last 16 games, shrunk to a
replacement-level prior. Every feature is the state after the team's / QB's
previous game (as-of merge strictly before the kickoff date), so played and
upcoming games are built the same way.
shortcut: form carries straight across seasons (Elo does the off-season regression);
add a season-boundary decay if it helps on pre-2015 data.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from elo import FRANCHISE, run_elo

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
RATES = {"epa": "plays", "success": "plays", "pass_epa": "pass_plays", "rush_epa": "rush_plays"}
QB_PRIOR, QB_PRIOR_N, QB_WINDOW = -0.10, 150, 16  # EPA/dropback, dropbacks of prior weight, games
HALFLIFE = 8


def _long(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, modern team codes."""
    g = games.assign(date=pd.to_datetime(games["date"]))
    sides = [g.assign(team=g["home_team"], opp=g["away_team"], qb_id=g["home_qb_id"]),
             g.assign(team=g["away_team"], opp=g["home_team"], qb_id=g["away_qb_id"])]
    df = pd.concat(sides)[["game_id", "season", "date", "completed", "team", "opp", "qb_id"]]
    df[["team", "opp"]] = df[["team", "opp"]].replace(FRANCHISE)
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def team_form(long: pd.DataFrame, team_raw: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Post-game EWM state per team: off_* (own offense) and def_* (EPA allowed)."""
    raw = team_raw.rename(columns={"posteam": "team"})
    rates = raw[["game_id", "team"]].assign(**{c: raw[c] / raw[n] for c, n in RATES.items()})
    played = long[long["completed"]][["game_id", "date", "team", "opp"]]
    df = played.merge(rates.add_prefix("off_").rename(columns={"off_game_id": "game_id", "off_team": "team"}),
                      on=["game_id", "team"])
    df = df.merge(rates.add_prefix("def_").rename(columns={"def_game_id": "game_id", "def_team": "opp"}),
                  on=["game_id", "opp"]).sort_values("date")
    cols = [c for c in df.columns if c.startswith(("off_", "def_"))]
    df[cols] = df.groupby("team")[cols].transform(lambda s: s.ewm(halflife=halflife).mean())
    return df[["team", "date", *cols]]


def qb_form(long: pd.DataFrame, qb_raw: pd.DataFrame) -> pd.DataFrame:
    """Post-game shrunk EPA/dropback per QB over his last QB_WINDOW games."""
    dates = long.drop_duplicates("game_id")[["game_id", "date"]]
    df = qb_raw.merge(dates, on="game_id").sort_values("date")
    roll = df.groupby("qb_id")[["qb_epa", "dropbacks"]].transform(lambda s: s.rolling(QB_WINDOW, 1).sum())
    df["qb"] = (roll["qb_epa"] + QB_PRIOR * QB_PRIOR_N) / (roll["dropbacks"] + QB_PRIOR_N)
    return df[["qb_id", "date", "qb"]]


def build(games: pd.DataFrame, team_raw: pd.DataFrame, qb_raw: pd.DataFrame,
          k: float, hfa: float, halflife: float = HALFLIFE) -> pd.DataFrame:
    long = _long(games)
    # unknown starter (weeks past the next one): fall back to the team's last listed starter
    long["qb_id"] = long.groupby("team")["qb_id"].ffill()
    feats = pd.merge_asof(long, team_form(long, team_raw, halflife), on="date", by="team",
                          allow_exact_matches=False)
    feats = pd.merge_asof(feats.sort_values("date"), qb_form(long, qb_raw), on="date", by="qb_id",
                          allow_exact_matches=False)
    feats["qb"] = feats["qb"].fillna(QB_PRIOR)  # no history: rookie / first start
    cols = [c for c in feats.columns if c.startswith(("off_", "def_")) or c == "qb"]

    out = games.merge(run_elo(games, k, hfa)[["game_id", "elo_margin"]], on="game_id")
    for side, team in (("h_", "home_team"), ("a_", "away_team")):
        f = feats[["game_id", "team", *cols]].rename(columns={"team": "_team", **{c: side + c for c in cols}})
        out = out.assign(_team=out[team].replace(FRANCHISE)).merge(f, on=["game_id", "_team"], how="left")
    out = out.drop(columns="_team")
    out["neutral"] = (out["location"] == "Neutral").astype(int)
    out["rest_diff"] = (out["home_rest"] - out["away_rest"]).clip(-7, 7)
    return out


def load_raw(kind: str) -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(RAW / f"pbp_{kind}_*.parquet"))))


def main() -> None:
    games = pd.read_csv(PROC / "games_all.csv", low_memory=False)
    elo = json.loads((PROC / "model_metrics.json").read_text())["elo"]
    out = build(games, load_raw("team"), load_raw("qb"), elo["k"], elo["hfa"])
    out.to_csv(PROC / "game_model_table.csv", index=False)
    print(f"game_model_table: {len(out)} games -> {PROC / 'game_model_table.csv'}")


if __name__ == "__main__":
    main()
