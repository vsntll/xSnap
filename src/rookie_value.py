"""Rookie (cold-start) value: first-season value from draft position and age,
one least-squares line per position on [1, log(pick), age]. Drafted players only;
a drafted player with no snaps in his first season counts as 0.
Scored on the 2024-25 draft classes, fit on 2009-23, against the per-position mean."""

from __future__ import annotations

import numpy as np
import pandas as pd

POS = ["QB", "RB", "WR", "TE"]
FIRST_CLASS, HOLDOUT = 2009, (2024, 2025)


def rookie_frame(ps: pd.DataFrame, bio: pd.DataFrame, last_finished: int) -> pd.DataFrame:
    b = bio[bio["position"].isin(POS) & bio["draft_year"].notna() & bio["draft_pick"].notna()]
    r = pd.DataFrame({"player_id": b["gsis_id"], "name": b["display_name"], "pos": b["position"],
                      "season": b["draft_year"].astype(int), "draft_pick": b["draft_pick"]})
    r = r[r["season"] >= FIRST_CLASS].drop_duplicates("player_id")
    birth = pd.to_datetime(b.set_index("gsis_id")["birth_date"], errors="coerce")
    r["age"] = ((pd.to_datetime(r["season"].astype(str) + "-09-01") - r["player_id"].map(birth)).dt.days
                / 365.25).fillna(22.5)
    val = ps.set_index(["season", "player_id"])["value"]
    r["target"] = [val.get((s, p), 0.0) if s <= last_finished else np.nan for s, p in zip(r["season"], r["player_id"])]
    return r


def _x(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), np.log(df["draft_pick"]), df["age"]])


def fit(train: pd.DataFrame) -> dict:
    return {p: np.linalg.lstsq(_x(g), g["target"], rcond=None)[0] for p, g in train.groupby("pos")}


def predict(coefs: dict, rows: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(rows))
    for p, idx in rows.groupby("pos").indices.items():
        out[idx] = _x(rows.iloc[idx]) @ coefs[p]
    return out


def evaluate(r: pd.DataFrame, score) -> dict:
    labeled = r.dropna(subset=["target"])
    train, test = labeled[labeled["season"] < HOLDOUT[0]], labeled[labeled["season"].between(*HOLDOUT)]
    test = test.assign(pred=predict(fit(train), test), mean=test["pos"].map(train.groupby("pos")["target"].mean()))
    return {"classes": f"{HOLDOUT[0]}-{HOLDOUT[1]}", "model": score(test["target"], test["pred"]),
            "pos_mean": score(test["target"], test["mean"])}
