"""Invariants on the processed tables (xValue's pattern): a silent upstream
change - renamed column, half a pull, a join duplicating games - fails here.
Skips when the table hasn't been built (CI without data)."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import PROC


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    path = PROC / "games_all.csv"
    if not path.exists():
        pytest.skip("games_all.csv not built")
    return pd.read_csv(path, low_memory=False)


def test_game_ids_unique(games):
    assert not games["game_id"].duplicated().any()


def test_played_games_have_consistent_scores(games):
    done = games[games["completed"]]
    assert (done["home_margin"] == done["home_score"] - done["away_score"]).all()
    assert (done["total"] == done["home_score"] + done["away_score"]).all()
    assert done["home_win"].isin([0.0, 0.5, 1.0]).all()


def test_season_game_counts_plausible(games):
    n = games[games["season"] < games["season"].max()].groupby("season").size()
    # 1999-2001: 259 (31 teams), 2002-20: 267, 2021 on: 284-285 (17 games)
    assert n.between(255, 290).all(), n[~n.between(255, 290)]
    assert games["season"].min() == 1999


def test_closing_spread_is_home_margin_sign(games):
    done = games[games["completed"] & games["close_spread_home"].notna()]
    assert done["close_spread_home"].corr(done["home_margin"]) > 0.3
