import numpy as np
import pandas as pd

import rookie_value as rv


def test_frame_targets_and_prediction_order():
    bio = pd.DataFrame({"gsis_id": ["a", "b", "c", "d"], "display_name": list("abcd"), "position": "WR",
                        "draft_year": [2020, 2020, 2021, 2026], "draft_pick": [5, 200, 40, 10],
                        "birth_date": "1999-01-01"})
    ps = pd.DataFrame({"season": [2020, 2021], "player_id": ["a", "c"], "value": [30.0, 10.0]})
    r = rv.rookie_frame(ps, bio, last_finished=2025).set_index("player_id")
    assert r.loc["a", "target"] == 30.0 and r.loc["b", "target"] == 0.0  # no snaps = 0
    assert np.isnan(r.loc["d", "target"])  # 2026 class: season not finished
    labeled = r.dropna(subset=["target"]).reset_index()
    pred = rv.predict(rv.fit(labeled), labeled)
    assert pred[0] > pred[1]  # earlier pick projects higher
