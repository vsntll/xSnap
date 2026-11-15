"""Pre-season value projection of every QB/RB/WR/TE for every season, walk-forward:
season S uses only labels from seasons before S (veterans: shrunk 3-2-1 baseline
from S-1; rookies: draft-pick line on classes before S). This is the only value
the game model sees, so a game never uses value from its own season.
"""

from __future__ import annotations

import pandas as pd

import rookie_value
from train_value_model import aging_curves, panel, scaled_baseline

FIRST = 2011  # needs two finished value seasons (2009-10) to fit on


def preseason(ps: pd.DataFrame, bio: pd.DataFrame, current: int) -> pd.DataFrame:
    """Rows: season, player_id, pos, proj (pre-season projected value for that season)."""
    last_finished = current - 1
    df, rookies = panel(ps, bio, last_finished), rookie_value.rookie_frame(ps, bio, last_finished)
    out = []
    for s in range(FIRST, current + 1):
        train = df[(df["season"] + 1 < s) & df["target"].notna()]  # labels known before season s
        rows = df[df["season"] == s - 1]
        vet = rows.assign(season=s, proj=scaled_baseline(train, rows, aging_curves(train)))
        rk_train = rookies[(rookies["season"] < s) & rookies["target"].notna()]
        rk = rookies[(rookies["season"] == s) & ~rookies["player_id"].isin(rows["player_id"])]
        rk = rk.assign(proj=rookie_value.predict(rookie_value.fit(rk_train), rk))
        out += [vet[["season", "player_id", "pos", "proj"]], rk[["season", "player_id", "pos", "proj"]]]
    return pd.concat(out, ignore_index=True)
