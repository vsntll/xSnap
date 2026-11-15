"""Steps 7-8: position aging curves and next-season value projections.

Rows: a player's season t (QB/RB/WR/TE with any volume) -> target value in t+1
(0 if he didn't play: value above replacement of an absent player). Features:
three seasons of value / rate / volume / games, age, draft pick, team change.
  baseline: 3-2-1 weighted value + the position aging delta for his age
  model:    gradient boosting, a direct predictor blended with a change-from-last one
Blend weight picked on targets 2021-23 (fit on earlier); scored once on targets
2024-25, fit on everything before. Live: fit on all, project the current season
from the last finished one -> value_projections.csv, aging_curves.csv.
shortcut: no rookie cold-start sub-model yet (rookies get no projection); no
rate x volume split; both are the next value-model steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
METRICS = PROC / "model_metrics.json"
POS = ["QB", "RB", "WR", "TE"]
LAG = ["value", "rate", "opp", "games"]
FEATURES = [f"{c}_{k}" for k in range(3) for c in LAG] + ["age", "draft_pick", "team_changed", "pos_code"]
VALID, HOLDOUT = (2021, 2023), (2024, 2025)  # target seasons


def panel(ps: pd.DataFrame, bio: pd.DataFrame, last_finished: int) -> pd.DataFrame:
    base = ps[["season", "player_id", "name", "pos", "team", *LAG]]
    df = base.rename(columns={c: f"{c}_0" for c in LAG})
    for k in (1, 2):
        lag = base.assign(season=base["season"] + k).drop(columns=["name", "pos"])
        df = df.merge(lag.rename(columns={c: f"{c}_{k}" for c in [*LAG, "team"]}), on=["season", "player_id"],
                      how="left")
    nxt = base.assign(season=base["season"] - 1)[["season", "player_id", "value"]].rename(columns={"value": "target"})
    df = df.merge(nxt, on=["season", "player_id"], how="left")
    df.loc[df["season"] < last_finished, "target"] = df["target"].fillna(0.0)  # absent next year = 0
    b = bio.drop_duplicates("gsis_id").set_index("gsis_id")
    birth = pd.to_datetime(df["player_id"].map(b["birth_date"]), errors="coerce")
    df["age"] = (pd.to_datetime(df["season"].astype(str) + "-09-01") - birth).dt.days / 365.25
    df["draft_pick"] = df["player_id"].map(b["draft_pick"]).fillna(300)  # undrafted
    df["team_changed"] = (df["team_1"].notna() & (df["team"] != df["team_1"])).astype(int)
    df["pos_code"] = df["pos"].map({p: i for i, p in enumerate(POS)})
    return df.drop(columns=["team_1", "team_2"])


def aging_curves(df: pd.DataFrame) -> pd.Series:
    """Delta method: mean change in value from age a to a+1, per position, smoothed."""
    pairs = df.dropna(subset=["target", "age"]).query("opp_0 > 0")
    d = pairs.assign(age=pairs["age"].round().astype(int), delta=pairs["target"] - pairs["value_0"])
    curve = d[d["age"].between(21, 40)].groupby(["pos", "age"])["delta"].mean()
    return curve.groupby("pos").transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())


def baseline(df: pd.DataFrame, curve: pd.Series) -> pd.Series:
    vals, w = df[["value_0", "value_1", "value_2"]], np.array([3.0, 2.0, 1.0])
    wavg = (vals.fillna(0) * w).sum(axis=1) / (vals.notna() * w).sum(axis=1)
    age = df["age"].round().clip(21, 40).fillna(27).astype(int)
    delta = pd.Series(list(zip(df["pos"], age)), index=df.index).map(curve).fillna(0.0)
    return wavg + delta


def fit_predict(train: pd.DataFrame, rows: pd.DataFrame, blend: float) -> np.ndarray:
    gb = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
    direct = HistGradientBoostingRegressor(**gb).fit(train[FEATURES], train["target"])
    change = HistGradientBoostingRegressor(**gb).fit(train[FEATURES], train["target"] - train["value_0"])
    return blend * direct.predict(rows[FEATURES]) + (1 - blend) * (rows["value_0"] + change.predict(rows[FEATURES]))


def score(y: pd.Series, p: pd.Series) -> dict:
    return {"n": len(y), "r2": round(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3),
            "mae": round(float((y - p).abs().mean()), 2), "spearman": round(y.corr(p, method="spearman"), 3)}


def main() -> None:
    ps = pd.read_csv(PROC / "player_season_value.csv")
    games = pd.read_csv(PROC / "games_all.csv", low_memory=False)
    last_finished = int(games.loc[~games["completed"], "season"].min()) - 1
    df = panel(ps, pd.read_parquet(RAW / "players.parquet"), last_finished)
    labeled = df.dropna(subset=["target"])
    tgt = labeled["season"] + 1

    curve = aging_curves(labeled[tgt < HOLDOUT[0]])
    vt, vtr = labeled[tgt.between(*VALID)], labeled[tgt < VALID[0]]
    blend = max([0.0, 0.25, 0.5, 0.75, 1.0],
                key=lambda b: score(vt["target"], pd.Series(fit_predict(vtr, vt, b), index=vt.index))["r2"])
    test, train = labeled[tgt.between(*HOLDOUT)], labeled[tgt < HOLDOUT[0]]
    test = test.assign(pred=fit_predict(train, test, blend), base=baseline(test, curve))
    by_pos = {p: {"model": score(g["target"], g["pred"]), "baseline": score(g["target"], g["base"])}
              for p, g in test.groupby("pos")}
    bins = test.groupby(pd.qcut(test["pred"], 5, labels=False))[["pred", "target"]].mean().round(1)
    metrics = json.loads(METRICS.read_text())
    metrics["value"] = {"blend_direct": blend, "holdout_targets": f"{HOLDOUT[0]}-{HOLDOUT[1]}",
                        "model": score(test["target"], test["pred"]),
                        "baseline": score(test["target"], test["base"]), "by_pos": by_pos,
                        "calibration_by_pred_quintile": bins.to_dict("list")}
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics["value"][k] for k in ("blend_direct", "model", "baseline")}, indent=2))

    full_curve = aging_curves(labeled)
    full_curve.rename("delta_value").reset_index().to_csv(PROC / "aging_curves.csv", index=False)
    live = df[df["season"] == last_finished]
    live = live.assign(proj_season=last_finished + 1, proj_value=fit_predict(labeled, live, blend),
                       baseline=baseline(live, full_curve))
    live.sort_values("proj_value", ascending=False)[
        ["proj_season", "player_id", "name", "pos", "team", "age", "value_0", "proj_value", "baseline"]
    ].round(2).to_csv(PROC / "value_projections.csv", index=False)
    print(f"value_projections: {len(live)} players for {last_finished + 1}")


if __name__ == "__main__":
    main()
