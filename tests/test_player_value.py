import pandas as pd
import pytest

import build_player_value as bpv


def test_replacement_tier_and_value(monkeypatch):
    monkeypatch.setitem(bpv.STARTERS, "QB", 2)  # tier = volume ranks 3-4
    ps = pd.DataFrame({"season": 2024, "pos": "QB", "opp": [500, 400, 300, 200, 100],
                       "epa": [100.0, 40.0, -15.0, -45.0, -50.0]})
    rate = bpv.replacement_rates(ps).loc[(2024, "QB")]
    assert rate == pytest.approx((-15 - 45) / (300 + 200))  # rank 5 excluded


def test_opportunities_by_position():
    w = pd.DataFrame({"pos": ["QB", "WR"], "attempts": [30, 0], "sacks_suffered": [2, 0], "carries": [3, 1],
                      "targets": [0, 8], "passing_epa": [5.0, None], "rushing_epa": [1.0, 0.5],
                      "receiving_epa": [None, 4.0]})
    out = bpv.opportunities(w)
    assert out["opp"].tolist() == [35, 9]
    assert out["epa"].tolist() == [6.0, 4.5]
