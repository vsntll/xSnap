import pandas as pd

from prediction_ledger import COLS, append

NOW = pd.Timestamp("2026-11-14T10:00:00Z")


def _preds(p, gametime=("13:00", "13:00")):
    return pd.DataFrame({"game_id": ["g1", "g2"], "season": 2026, "week": 11, "home_team": ["KC", "BUF"],
                         "away_team": ["DEN", "MIA"], "p_home": p, "pred_margin": [3.0, 1.0],
                         "close_spread_home": [3.5, 2.5], "date": "2026-11-15", "gametime": list(gametime)})


def test_rerun_unchanged_adds_nothing_and_change_appends():
    first = append(pd.DataFrame(columns=COLS), _preds([0.6, 0.55]), NOW, "elo")
    assert len(first) == 2
    assert len(append(first, _preds([0.6, 0.55]), NOW + pd.Timedelta("1D"), "elo")) == 2
    moved = append(first, _preds([0.6, 0.58]), NOW + pd.Timedelta("1D"), "elo")
    assert len(moved) == 3 and moved.iloc[:2].equals(first)  # old rows untouched


def test_models_dedupe_separately():
    both = append(append(pd.DataFrame(columns=COLS), _preds([0.6, 0.55]), NOW, "elo"), _preds([0.6, 0.55]), NOW, "v1")
    assert len(both) == 4
    assert len(append(both, _preds([0.6, 0.55]), NOW, "elo")) == 4


def test_no_snapshot_after_kickoff():
    # 13:00 ET on 2026-11-15 is 18:00 UTC
    out = append(pd.DataFrame(columns=COLS), _preds([0.6, 0.55], ("13:00", "20:20")),
                 pd.Timestamp("2026-11-15T19:00:00Z"), "elo")
    assert out["game_id"].tolist() == ["g2"]
