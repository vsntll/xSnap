"""Unit test of build_games_all.build on a tiny synthetic schedule - runs in CI
without any pulled data."""

import math

import pandas as pd

from build_games_all import KEEP, build


def test_build_outcomes_and_names():
    rows = [  # home win, tie, away win, unplayed
        ("g1", 7.0, 24.0, 17.0), ("g2", 0.0, 20.0, 20.0), ("g3", -3.0, 10.0, 13.0), ("g4", None, None, None),
    ]
    sched = pd.DataFrame({c: [None] * len(rows) for c in KEEP})
    sched["game_id"] = [r[0] for r in rows]
    sched["result"] = [r[1] for r in rows]
    sched["home_score"] = [r[2] for r in rows]
    sched["away_score"] = [r[3] for r in rows]
    sched["gameday"], sched["gametime"] = "2026-09-10", "20:20"
    sched["spread_line"] = 3.5

    df = build(sched).set_index("game_id")
    assert df["completed"].tolist() == [True, True, True, False]
    assert df.loc[["g1", "g2", "g3"], "home_win"].tolist() == [1.0, 0.5, 0.0]
    assert math.isnan(df.loc["g4", "home_win"])
    assert (df["close_spread_home"] == 3.5).all()
    assert "spread_line" not in df.columns and "date" in df.columns
