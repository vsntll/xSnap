import pandas as pd
import pytest

from build_game_model_table import QB_PRIOR, build

IDS = ["g1", "g2", "g3"]


def _inputs():
    games = pd.DataFrame({
        "game_id": IDS, "season": 2024, "date": ["2024-09-08", "2024-09-15", "2024-09-22"], "gametime": "13:00",
        "home_team": ["KC", "OAK", "KC"], "away_team": ["OAK", "KC", "OAK"], "location": "Home",
        "home_margin": [7.0, -3.0, None], "completed": [True, True, False], "home_rest": 7, "away_rest": 7,
        "home_qb_id": ["q1", "q2", "q1"], "away_qb_id": ["q2", "q1", None],
    })
    team = pd.DataFrame([(g, t, 60, 6.0, 30, 40, 4.0, 20, 2.0) for g in IDS[:2] for t in ("KC", "LV")],
                        columns=["game_id", "posteam", "plays", "epa", "success", "pass_plays", "pass_epa",
                                 "rush_plays", "rush_epa"])
    qb = pd.DataFrame([(g, t, q, 40, 4.0) for g in IDS[:2] for t, q in (("KC", "q1"), ("LV", "q2"))],
                      columns=["game_id", "posteam", "qb_id", "dropbacks", "qb_epa"])
    return games, team, qb


def test_features_use_only_earlier_games():
    games, team, qb = _inputs()
    base = build(games, team, qb, k=20, hfa=50).set_index("game_id")
    team.loc[team["game_id"] == "g2", "epa"] = -30.0  # change game 2's play-by-play
    alt = build(games, team, qb, k=20, hfa=50).set_index("game_id")
    assert base.loc["g2", "h_off_epa"] == alt.loc["g2", "h_off_epa"]
    assert base.loc["g3", "h_off_epa"] != alt.loc["g3", "h_off_epa"]


def test_first_game_has_no_form_and_prior_qb():
    games, team, qb = _inputs()
    out = build(games, team, qb, k=20, hfa=50).set_index("game_id")
    assert pd.isna(out.loc["g1", "h_off_epa"])
    assert out.loc["g1", "h_qb"] == QB_PRIOR
    # OAK (schedule code) joins LV (pbp code); unknown g3 away QB falls back to OAK's last starter q2
    assert out.loc["g3", "a_off_epa"] == pytest.approx(0.1)
    assert out.loc["g3", "a_qb"] == pytest.approx((8.0 + QB_PRIOR * 150) / (80 + 150))
