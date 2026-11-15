import pandas as pd

from preseason_value import preseason


def _data():
    rows = [(s, p, 10.0 * i + s - 2009) for s in range(2009, 2014) for i, p in enumerate(["a", "b", "c", "d"])]
    ps = pd.DataFrame(rows, columns=["season", "player_id", "value"]).assign(
        name="x", pos="WR", team="KC", rate=0.1, opp=100, games=16)
    bio = pd.DataFrame({"gsis_id": list("abcd"), "display_name": list("abcd"), "position": "WR",
                        "birth_date": "1990-01-01", "draft_year": [2005] * 4, "draft_pick": [10, 50, 100, 200]})
    return ps, bio


def test_projection_ignores_its_own_and_later_seasons():
    ps, bio = _data()
    base = preseason(ps, bio, current=2014).set_index(["season", "player_id"])["proj"]
    ps.loc[ps["season"] >= 2012, "value"] = -500.0  # rewrite 2012 on
    alt = preseason(ps, bio, current=2014).set_index(["season", "player_id"])["proj"]
    assert base.loc[2012].equals(alt.loc[2012])  # 2012 projection uses <= 2011 only
    assert not base.loc[2013].equals(alt.loc[2013])
