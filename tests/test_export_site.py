import math
from types import SimpleNamespace

from export_site import _game, game_view, market_p


def _g(**kw):
    base = dict(away_team="NE", home_team="DET", date="2026-11-15", gametime="13:00", elo_p_home=0.54,
                close_spread_home=2.5, v1_p_home=0.6, v1_pred_margin=4.0)
    return SimpleNamespace(**{**base, **kw})


def test_market_p_is_symmetric_around_pickem():
    assert market_p(0) == 0.5
    assert math.isclose(market_p(3) + market_p(-3), 1)
    assert market_p(7) > 0.65


def test_view_favorite_and_split_from_line():
    v = game_view(_g(v1_p_home=0.3, v1_pred_margin=-6.0))
    assert v["fav"] == "NE" and math.isclose(v["fav_p"], 0.7) and math.isclose(v["edge"], -8.5)
    assert "NE 8.5 pts better" in _game(v) and "1:00 PM" in _game(v)


def test_missing_v1_and_line_fall_back_to_elo_without_crashing():
    v = game_view(_g(v1_p_home=float("nan"), v1_pred_margin=float("nan"), close_spread_home=float("nan")))
    assert v["p"] == 0.54 and v["edge"] is None and v["mkt"] is None
    html = _game(v)
    assert "Elo only" in html and "--mkt" not in html and 'data-edge="0.00"' in html


def test_assets_fall_back_to_last_starter_and_handle_missing_headshot(tmp_path, monkeypatch):
    import pandas as pd
    import export_site
    monkeypatch.setattr(export_site, "RAW", tmp_path)
    pd.DataFrame({"team_abbr": ["NE", "DET"], "team_name": ["", ""], "team_logo_espn": ["ne.png", "det.png"],
                  "team_color": ["", ""]}).to_parquet(tmp_path / "teams.parquet")
    pd.DataFrame({"gsis_id": ["q1", "q2"], "display_name": ["Drake Maye", "Jared Goff"],
                  "headshot": ["https://x/image/upload/f_auto,q_auto/league/a", None]}).to_parquet(tmp_path / "players.parquet")
    games = pd.DataFrame({"game_id": ["g1", "g2"], "date": ["2026-11-01", "2026-11-15"], "home_team": "DET",
                          "away_team": "NE", "home_qb_id": ["q2", None], "away_qb_id": ["q1", None]})
    logos, shots, qbs = export_site.load_assets(games)
    assert qbs[("g2", "NE")] == ("Drake Maye", "https://x/image/upload/" + export_site.FACE + "/league/a", "q1")
    assert qbs[("g2", "DET")] == ("Jared Goff", None, "q2")
    html = _game(game_view(_g(game_id="g2"), logos, qbs, {"q1": (3, "A")}))
    assert 'src="ne.png"' in html and 'alt="Drake Maye"' in html and 'class="noshot"' in html
    assert "#3 QB this season" in html and 'class="grade g-A"' in html


def test_player_panels_rank_by_position_and_tolerate_missing_war():
    import pandas as pd
    from export_site import _players
    g = pd.DataFrame({"pos": ["QB", "QB", "D/ST"], "rank": [2, 1, 1], "grade": ["B", "A", "A"], "pct": [50.0, 100.0, 100.0],
                      "name": ["B Guy", "A Guy", "MIN D/ST"], "team": ["NE", "DET", "MIN"], "volume": [300, 250, 9],
                      "value": [10.0, 50.0, 80.0], "player_id": ["q1", "q2", "MIN"]})
    tabs, panels = _players(g, {"MIN": "min.png"}, {"q2": "a.png"})
    assert tabs.count('role="tab"') == 2 and panels.index("A Guy") < panels.index("B Guy")
    assert 'src="min.png"' in panels and "wins added" not in panels
    assert "wins added" in _players(g.assign(war=1.0, value_per_game=2.0), {}, {})[1]
    assert "build_player_grades.py" in _players(pd.DataFrame(), {}, {})[1]


def test_american_odds_and_spread_text():
    from export_site import american, spread_text
    assert american(0.5) == "EVEN" and american(0.75) == "-300" and american(0.25) == "+300"
    assert american(1.0) == "-9900" and american(None) == "–"
    v = game_view(_g(close_spread_home=2.5))
    assert spread_text(v) == "DET -2.5"
    assert spread_text(game_view(_g(close_spread_home=-7.0))) == "NE -7.0"
    assert spread_text(game_view(_g(close_spread_home=0.0))) == "PK"


def test_team_seasons_and_season_record():
    import pandas as pd
    from elo import run_elo
    from export_site import season_record, team_seasons
    games = pd.DataFrame({"game_id": ["a", "b", "c"], "season": [2025, 2025, 2025], "game_type": ["REG", "REG", "WC"],
                          "date": ["2025-09-07", "2025-09-14", "2026-01-10"], "gametime": "13:00",
                          "home_team": ["NE", "DET", "NE"], "away_team": ["DET", "NE", "DET"], "location": "Home",
                          "home_margin": [7.0, 3.0, 10.0], "completed": True, "close_spread_home": [-3.0, 0.0, 2.0]})
    ts = team_seasons(games, run_elo(games)).set_index("team")
    assert (ts.loc["NE", ["w", "l", "t", "diff"]] == [1, 1, 0, 4]).all()  # the playoff win isn't in the record
    assert ts.loc["NE", "rank"] == 1 and ts.loc["NE", "elo"] > 1500  # but it is in the rating
    rec = season_record(games, run_elo(games)).loc[2025]
    assert rec["games"] == 3 and rec["line"] == 0.5  # pick'em game b left out of the line share
