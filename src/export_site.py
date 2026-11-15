"""Step 12b: write the static site, docs/index.html (served by GitHub Pages from
master /docs). Shows the same Elo predictions the ledger froze, plus the backtest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd

from prediction_ledger import week_predictions

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "index.html"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>xSnap NFL Predictions</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e3e3e3; --accent:#1f6feb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111; --fg:#eee; --muted:#999; --line:#333; --accent:#58a6ff; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:760px; margin:0 auto; padding:24px 16px; }}
h1 {{ margin:0 0 4px; }} .muted {{ color:var(--muted); font-size:13px; }}
.wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; margin:12px 0 28px; }}
th, td {{ padding:6px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; }} th {{ font-weight:600; color:var(--muted); font-size:13px; }}
td.p {{ font-variant-numeric:tabular-nums; }}
</style></head><body><main>
<h1>xSnap</h1>
<p class="muted">Team Elo baseline. Updated {updated}. Lines are closing-type lines from nflverse.</p>
<h2>{season} week {week}</h2>
<div class="wrap"><table>
<tr><th>Game (away @ home)</th><th>Kickoff (ET)</th><th>Home win</th><th>Elo margin</th><th>Line (home)</th></tr>
{rows}
</table></div>
<h2>Track record</h2>
<p class="muted">Every played game {test_seasons}, playoffs included. Lower is better.</p>
<div class="wrap"><table>
<tr><th>Model</th><th>Games</th><th>Log-loss</th><th>Brier</th><th>Margin MAE</th></tr>
{metrics}
</table></div>
<p class="muted">Data: <a href="https://github.com/nflverse">nflverse</a>.</p>
</main></body></html>
"""


def _row(g) -> str:
    cells = [f"{escape(g.away_team)} @ {escape(g.home_team)}", f"{g.date} {g.gametime}",
             f"{g.p_home:.0%}", f"{g.elo_margin:+.1f}",
             "" if pd.isna(g.close_spread_home) else f"{g.close_spread_home:+.1f}"]
    return "<tr>" + "".join(f'<td class="p">{escape(c)}</td>' for c in cells) + "</tr>"


def main() -> None:
    games = pd.read_csv(PROC / "games_all.csv", low_memory=False)
    m = json.loads((PROC / "model_metrics.json").read_text())
    week = week_predictions(games, m["elo"]["k"], m["elo"]["hfa"])
    metrics = "".join(
        f"<tr><td>{name}</td><td>{s['games']}</td><td>{s['log_loss']:.3f}</td>"
        f"<td>{s['brier']:.3f}</td><td>{s['margin_mae']:.2f}</td></tr>"
        for name, s in [("Elo", m["elo"]), ("Closing line", m["market_close"])])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.format(
        updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        season=int(week["season"].iloc[0]) if len(week) else "", week=int(week["week"].iloc[0]) if len(week) else "",
        rows="".join(_row(g) for g in week.itertuples()), test_seasons=m["elo"]["test_seasons"], metrics=metrics,
    ), encoding="utf-8")
    print(f"site: {len(week)} games -> {OUT}")


if __name__ == "__main__":
    main()
