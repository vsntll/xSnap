import pandas as pd

from train_value_model import panel


def test_panel_lags_and_targets():
    ps = pd.DataFrame({"season": [2020, 2021, 2022], "player_id": "p1", "name": "A", "pos": "WR",
                       "team": ["KC", "KC", "BUF"], "value": [10.0, 20.0, 30.0], "rate": 0.1,
                       "opp": 100, "games": 16})
    bio = pd.DataFrame({"gsis_id": ["p1"], "birth_date": ["1995-09-01"], "draft_pick": [None]})
    df = panel(ps, bio, last_finished=2022).set_index("season")
    assert df.loc[2020, "target"] == 20.0 and df.loc[2021, "target"] == 30.0
    assert pd.isna(df.loc[2022, "target"])  # 2023 not finished: no label
    assert df.loc[2022, "value_1"] == 20.0 and df.loc[2022, "value_2"] == 10.0
    assert df.loc[2022, "team_changed"] == 1 and df.loc[2021, "team_changed"] == 0
    assert df.loc[2020, "draft_pick"] == 300 and round(df.loc[2020, "age"]) == 25


def test_absent_next_season_is_zero():
    ps = pd.DataFrame({"season": [2020], "player_id": "p1", "name": "A", "pos": "RB", "team": "KC",
                       "value": [5.0], "rate": 0.0, "opp": 50, "games": 8})
    bio = pd.DataFrame({"gsis_id": ["p1"], "birth_date": ["1995-01-01"], "draft_pick": [10]})
    assert panel(ps, bio, last_finished=2022)["target"].tolist() == [0.0]
