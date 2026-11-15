import pandas as pd
import pytest

import build_player_value as bpv


def test_replacement_tier_and_value(monkeypatch):
    monkeypatch.setitem(bpv.STARTERS, "QB", 2)  # tier = volume ranks 3-4
    ps = pd.DataFrame({"season": 2024, "pos": "QB", "opp": [500, 400, 300, 200, 100],
                       "epa": [100.0, 40.0, -15.0, -45.0, -50.0]})
    rate = bpv.replacement_rates(ps).loc[(2024, "QB")]
    assert rate == pytest.approx((-15 - 45) / (300 + 200))  # rank 5 excluded


def test_traded_player_keeps_one_season_row_and_per_team_stints():
    weeks = pd.DataFrame({"player_id": "p", "player_display_name": "P", "position": "WR", "season": 2024,
                          "season_type": "REG", "week": [1, 2, 9, 10], "game_id": list("abcd"),
                          "team": ["CHI", "CHI", "WAS", "WAS"], "targets": [5, 5, 8, 8], "carries": 0,
                          "attempts": 0, "sacks_suffered": 0, "passing_epa": 0.0, "rushing_epa": 0.0,
                          "receiving_epa": [1.0, 1.0, 2.0, 2.0]})
    bio = pd.DataFrame({"gsis_id": ["p"], "position": ["WR"]})
    ps, pts = bpv.build(weeks, bio)
    row = ps.iloc[0]
    assert len(ps) == 1 and row["teams"] == "CHI/WAS" and row["team"] == "WAS"
    assert row["opp"] == 26 and row["games"] == 4
    assert pts.set_index("team")[["first_week", "last_week", "opp"]].to_dict("index") == {
        "CHI": {"first_week": 1, "last_week": 2, "opp": 10}, "WAS": {"first_week": 9, "last_week": 10, "opp": 16}}


def test_opportunities_by_position():
    w = pd.DataFrame({"pos": ["QB", "WR"], "attempts": [30, 0], "sacks_suffered": [2, 0], "carries": [3, 1],
                      "targets": [0, 8], "passing_epa": [5.0, None], "rushing_epa": [1.0, 0.5],
                      "receiving_epa": [None, 4.0]})
    out = bpv.opportunities(w)
    assert out["opp"].tolist() == [35, 9]
    assert out["epa"].tolist() == [6.0, 4.5]
