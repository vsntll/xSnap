"""Team Elo updated on margin (538-style NFL Elo) - the outcome-model floor.

Each game's p_home / elo_margin use only ratings from earlier games.
"""

from __future__ import annotations

import math

import pandas as pd

# relocated franchises keep their rating
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
ELO_PER_POINT = 25.0  # 538's conversion: 25 Elo points ~ 1 point of margin


def run_elo(games: pd.DataFrame, k: float = 20.0, hfa: float = 48.0, revert: float = 1 / 3) -> pd.DataFrame:
    """games: games_all rows. Returns game_id, elo_home, elo_away, p_home, elo_margin (pre-game)."""
    games = games.sort_values(["date", "gametime", "game_id"])
    rating: dict[str, float] = {}
    season = None
    out = []
    for g in games.itertuples(index=False):
        if g.season != season:  # off-season: pull every team a third of the way back to 1500
            season = g.season
            rating = {t: 1500 + (1 - revert) * (r - 1500) for t, r in rating.items()}
        h, a = FRANCHISE.get(g.home_team, g.home_team), FRANCHISE.get(g.away_team, g.away_team)
        rh, ra = rating.get(h, 1500.0), rating.get(a, 1500.0)
        diff = rh - ra + (0.0 if g.location == "Neutral" else hfa)
        p = 1 / (1 + 10 ** (-diff / 400))
        out.append((g.game_id, rh, ra, p, diff / ELO_PER_POINT))
        if pd.isna(g.home_margin):
            continue
        m = g.home_margin
        result = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
        winner_diff = diff if m >= 0 else -diff  # damp blowouts by expected favourites
        mult = math.log(max(abs(m), 1) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        delta = k * mult * (result - p)
        rating[h], rating[a] = rh + delta, ra - delta
    return pd.DataFrame(out, columns=["game_id", "elo_home", "elo_away", "p_home", "elo_margin"])
