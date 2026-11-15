import pandas as pd
import pytest

from elo import run_elo


def _games():
    return pd.DataFrame({
        "game_id": ["a", "b", "c"], "season": [2024, 2024, 2024],
        "date": ["2024-09-08", "2024-09-15", "2024-09-22"], "gametime": "13:00",
        "home_team": ["KC", "BUF", "KC"], "away_team": ["BUF", "KC", "BUF"],
        "location": ["Home", "Home", "Neutral"], "home_margin": [10.0, 3.0, None],
    })


def test_no_look_ahead():
    base = run_elo(_games()).set_index("game_id")
    later = _games()
    later.loc[1, "home_margin"] = -30.0  # change game b's result
    alt = run_elo(later).set_index("game_id")
    pd.testing.assert_series_equal(base.loc[["a", "b"], "p_home"], alt.loc[["a", "b"], "p_home"])
    assert base.loc["c", "p_home"] != alt.loc["c", "p_home"]


def test_home_edge_and_neutral_site():
    r = run_elo(_games()).set_index("game_id")
    assert r.loc["a", "p_home"] > 0.5  # equal ratings, home edge
    # neutral game c: KC won a, lost b by less, so is rated higher, no home edge added
    assert r.loc["c", "elo_margin"] * 25 == pytest.approx(r.loc["c", "elo_home"] - r.loc["c", "elo_away"])
