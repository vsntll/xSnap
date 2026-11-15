import pandas as pd

from prediction_ledger import COLS, append


def _preds(p):
    return pd.DataFrame({"game_id": ["g1", "g2"], "season": 2026, "week": 11, "home_team": ["KC", "BUF"],
                         "away_team": ["DEN", "MIA"], "p_home": p, "elo_margin": [3.0, 1.0],
                         "close_spread_home": [3.5, 2.5]})


def test_rerun_unchanged_adds_nothing_and_change_appends():
    empty = pd.DataFrame(columns=COLS)
    first = append(empty, _preds([0.6, 0.55]), "2026-11-14T10:00:00Z")
    assert len(first) == 2
    assert len(append(first, _preds([0.6, 0.55]), "2026-11-15T10:00:00Z")) == 2
    moved = append(first, _preds([0.6, 0.58]), "2026-11-15T10:00:00Z")
    assert len(moved) == 3 and moved.iloc[:2].equals(first)  # old rows untouched
