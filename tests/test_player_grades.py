import pandas as pd
import pytest

from build_player_grades import defenses, grade, impact, points_per_win


def test_grade_ranks_within_position_season_and_drops_unqualified():
    df = pd.DataFrame({"season": 2025, "pos": ["QB"] * 4 + ["K"], "player_id": list("abcde"), "name": list("abcde"),
                       "team": "KC", "teams": "KC", "games": 17, "volume": [500, 400, 300, 50, 20],
                       "value": [90.0, 10.0, 50.0, 200.0, 5.0]})
    g = grade(df, current=2026).set_index("player_id")
    assert "d" not in g.index  # 50 dropbacks: unqualified
    assert g.loc[["a", "c", "b"], "rank"].tolist() == [1, 2, 3]
    assert g.loc["a", "grade"] == "A" and g.loc["b", "pct"] == 33.3 and g.loc["b", "grade"] == "D"
    assert g.loc["e", "rank"] == 1 and (g["partial"] == 0).all()


def test_points_per_win_and_war():
    # two teams, two games each way: team A +10 per game and wins both -> 20 pts = 2 wins over B's -20 / 0
    reg = pd.DataFrame({"season": 2025, "home_team": ["A", "B"], "away_team": ["B", "A"],
                        "home_margin": [10.0, -10.0], "home_win": [1.0, 0.0]})
    ppw = points_per_win(reg)
    assert ppw == pytest.approx(20.0)
    q = impact(pd.DataFrame({"value": [40.0], "games": [8]}), ppw)
    assert q["war"].iloc[0] == 2.0 and q["war_per_17"].iloc[0] == 4.25 and q["value_per_game"].iloc[0] == 5.0


def test_defense_is_opponent_offense_negated_plus_special_teams():
    reg = pd.DataFrame({"game_id": ["g1"], "season": [2025]})
    off = pd.DataFrame({"game_id": ["g1", "g1"], "posteam": ["KC", "BUF"], "epa": [5.0, -3.0]})
    st = pd.DataFrame({"game_id": ["g1", "g1"], "team": ["KC", "BUF"], "st_epa": [1.0, -1.0]})
    d = defenses(off, st, reg).set_index("team")["value"]
    assert d["KC"] == 3.0 + 1.0 and d["BUF"] == -5.0 - 1.0
