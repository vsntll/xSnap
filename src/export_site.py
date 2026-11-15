"""Write the static site, docs/index.html (served by GitHub Pages from
master /docs). Shows the same predictions the ledger froze, plus the backtest."""

from __future__ import annotations

import json
import math
from datetime import datetime
from html import escape
from pathlib import Path
from string import Template

import pandas as pd

from prediction_ledger import week_predictions

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
FACE = "f_auto,q_auto,c_thumb,g_face,w_200,h_200"  # NFL CDN crop: ~20 KB instead of the 4 MB original
OUT = ROOT / "docs" / "index.html"
TEMPLATE = Path(__file__).with_name("site_template.html")
MARGIN_SD = 13.45  # NFL result-vs-spread spread in points; turns a line into win odds


def market_p(spread_home: float) -> float:
    """Home win probability implied by the home margin the line expects."""
    return 0.5 * (1 + math.erf(spread_home / (MARGIN_SD * math.sqrt(2))))


def _num(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def load_assets(games: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Team -> logo URL, player id -> headshot URL, and (game_id, team) -> (QB name, headshot URL, QB id).
    Empty when the raw tables are missing."""
    logos, shots, qbs = {}, {}, {}
    if (RAW / "teams.parquet").exists():
        t = pd.read_parquet(RAW / "teams.parquet")
        logos = dict(zip(t["team_abbr"], t["team_logo_espn"]))
    players = pd.read_parquet(RAW / "players.parquet") if (RAW / "players.parquet").exists() else None
    if players is None or "headshot" not in players:
        return logos, shots, qbs
    bio = players.drop_duplicates("gsis_id").set_index("gsis_id")
    shots = {pid: s.replace("f_auto,q_auto", FACE, 1) for pid, s in bio["headshot"].dropna().items()}
    cols = ["game_id", "date", "team", "qb"]
    long = pd.concat([games[["game_id", "date", f"{s}_team", f"{s}_qb_id"]].set_axis(cols, axis=1)
                      for s in ("home", "away")]).sort_values("date", kind="stable")
    long["qb"] = long.groupby("team")["qb"].ffill()  # unlisted starter: last listed one, as the model does
    for r in long.dropna(subset=["qb"]).itertuples():
        if r.qb in bio.index:
            qbs[(r.game_id, r.team)] = (bio.at[r.qb, "display_name"], shots.get(r.qb), r.qb)
    return logos, shots, qbs


def load_grades() -> pd.DataFrame:
    """Latest season of player_grades.csv (empty if it hasn't been built)."""
    path = PROC / "player_grades.csv"
    if not path.exists():
        return pd.DataFrame()
    g = pd.read_csv(path)
    return g[g["season"] == g["season"].max()]


def game_view(g, logos: dict | None = None, qbs: dict | None = None, qb_grades: dict | None = None) -> dict:
    """Plain numbers the page needs for one game; v1 falls back to Elo when missing."""
    elo, line = _num(g.elo_p_home), _num(g.close_spread_home)
    p = _num(getattr(g, "v1_p_home", None))
    margin = _num(getattr(g, "v1_pred_margin", None))
    kick = datetime.strptime(f"{g.date} {g.gametime}", "%Y-%m-%d %H:%M")
    v = {"away": g.away_team, "home": g.home_team, "p": p if p is not None else elo, "v1": p is not None,
         "elo": elo, "line": line, "margin": margin, "mkt": None if line is None else market_p(line),
         "kick": kick, "edge": None if margin is None or line is None else margin - line}
    v["fav"] = v["home"] if v["p"] >= 0.5 else v["away"]
    v["fav_p"] = max(v["p"], 1 - v["p"])
    v["dog"] = v["away"] if v["fav"] == v["home"] else v["home"]
    gid = getattr(g, "game_id", None)
    v["logo"] = {t: (logos or {}).get(t) for t in (v["away"], v["home"])}
    v["qb"] = {t: (qbs or {}).get((gid, t)) for t in (v["away"], v["home"])}
    v["qb_grade"] = {t: (qb_grades or {}).get(q[2]) if q else None for t, q in v["qb"].items()}
    return v


def _logo(v, team, cls="logo") -> str:
    url = v["logo"].get(team)
    return f'<img class="{cls}" src="{escape(url)}" alt="" width="40" height="40" loading="lazy">' if url else ""


def _qb(v, team) -> str:
    if not v["qb"].get(team):
        return ""
    name, shot, _ = v["qb"][team]
    img = (f'<img src="{escape(shot)}" alt="{escape(name)}" width="72" height="72" loading="lazy">' if shot
           else f'<span class="noshot" aria-hidden="true">{escape(team)}</span>')
    gr = v["qb_grade"].get(team)
    badge = f'<span class="grade g-{gr[1]}" title="Season grade">{gr[1]}</span>' if gr else ""
    rank = f", #{gr[0]} QB this season" if gr else ""
    return (f'<figure class="qb">{img}<figcaption>{escape(name)}<small>{escape(team)} starting QB{rank}</small>'
            f'</figcaption>{badge}</figure>')


VOLUME = {"QB": "dropbacks + carries", "RB": "carries + targets", "WR": "carries + targets",
          "TE": "carries + targets", "K": "field goal tries", "D/ST": "games"}


def _players(grades: pd.DataFrame, logos: dict, shots: dict) -> tuple[str, str]:
    """Position tabs and one ranked panel per position."""
    if grades.empty:
        return "", '<p class="empty">Player grades appear once build_player_grades.py has run.</p>'
    has_war = "war" in grades and "value_per_game" in grades
    tabs, panels = [], []
    for i, pos in enumerate(p for p in VOLUME if p in set(grades["pos"])):
        rows = []
        for r in grades[grades["pos"] == pos].sort_values("rank").itertuples():
            logo = logos.get(r.team)
            shot = logo if pos == "D/ST" else shots.get(r.player_id)
            img = (f'<img class="pshot{" dst" if pos == "D/ST" else ""}" src="{escape(shot)}" alt="" loading="lazy">'
                   if shot else f'<span class="pshot noshot">{escape(r.team)}</span>')
            tlogo = f'<img class="tlogo" src="{escape(logo)}" alt="" loading="lazy">' if logo and pos != "D/ST" else ""
            impact = (f'<span class="pnum"><span class="cnt">{r.war:+.2f}</span><small>wins added</small></span>'
                      f'<span class="pnum"><span class="cnt">{r.value_per_game:+.1f}</span><small>pts / game</small></span>' if has_war else "")
            rows.append(
                f'<li class="prow" data-q="{escape(f"{r.name} {r.team}".upper())}"><span class="rk">{r.rank}</span>{img}'
                f'<span class="pname">{escape(r.name)}<small>{tlogo}{escape(r.team)}, {int(r.volume)} {VOLUME[pos]}</small></span>'
                f'<span class="grade g-{r.grade}">{r.grade}</span>'
                f'<span class="pbar" style="--pct:{r.pct / 100:.3f}" title="Better than {r.pct:.0f}% at {pos}"><i></i></span>'
                f'<span class="pnum"><span class="cnt">{r.value:+.1f}</span><small>pts vs replacement</small></span>{impact}</li>')
        sel = "true" if i == 0 else "false"
        tabs.append(f'<button class="pill" type="button" role="tab" aria-selected="{sel}" aria-controls="pp-{i}" '
                    f'data-panel="pp-{i}">{pos}</button>')
        more = (f'<button class="pill more" type="button">Show all {len(rows)}</button>' if len(rows) > 10 else "")
        panels.append(f'<div class="ppanel" id="pp-{i}" data-pos="{pos}" role="tabpanel" aria-label="{pos} rankings"'
                      f'{"" if i == 0 else " hidden"}><ol class="plist">{"".join(rows)}</ol>{more}</div>')
    return "".join(tabs), "".join(panels)


def american(p: float | None) -> str:
    """Fair American moneyline for a win probability (no bookmaker margin)."""
    if p is None:
        return "–"
    p = min(max(p, 0.01), 0.99)  # keep 99%+ picks from printing absurd numbers
    if abs(p - 0.5) < 0.005:
        return "EVEN"
    return f"-{round(100 * p / (1 - p))}" if p > 0.5 else f"+{round(100 * (1 - p) / p)}"


def spread_text(v) -> str:
    """Sportsbook-style spread: favourite minus points, e.g. 'DET -2.5'."""
    if v["line"] is None:
        return "No line"
    if abs(v["line"]) < 0.25:
        return "PK"
    return f"{v['home']} -{v['line']:.1f}" if v["line"] > 0 else f"{v['away']} -{-v['line']:.1f}"


def _ml(v, team, p) -> str:
    side = "fav" if team == v["fav"] else "dog"
    return f'<span class="ml {side}" title="Model fair moneyline">{american(p)}</span>'


EDGE_PTS = 3.0  # model vs line gap (points) that earns an edge tag on the odds board


def _split_text(v) -> str:
    if v["edge"] is None:
        return "No line or model margin yet."
    if abs(v["edge"]) < 0.5:
        return "Model and line agree."
    side = v["home"] if v["edge"] > 0 else v["away"]
    return f"Model rates {side} {abs(v['edge']):.1f} pts better than the line does."


def _game(v) -> str:
    pct = lambda x: "–" if x is None else f"{x:.0%}"
    pts = lambda x: "–" if x is None else f"{x:+.1f}"
    style = f"--p:{v['p']:.3f}" + "".join(f";--{k}:{v[k]:.3f}" for k in ("elo", "mkt") if v[k] is not None)
    marks = "".join(f'<i class="mk {k}"></i>' for k in ("elo", "mkt") if v[k] is not None)
    label = f"{v['home']} {v['p']:.0%} to win at home; Elo {pct(v['elo'])}; line implies {pct(v['mkt'])}"
    kick = v["kick"]
    when = f"{kick:%a %b} {kick.day}, {kick:%I:%M %p}".replace(" 0", " ")
    facts = [("Model margin (home)", pts(v["margin"])), (f"Model moneyline ({v['home']})", american(v["p"])),
             ("Elo home win", pct(v["elo"])), ("Spread", spread_text(v)),
             (f"Line moneyline ({v['home']})", american(v["mkt"]))]
    dl = "".join(f"<div><dt>{escape(t)}</dt><dd>{escape(d)}</dd></div>" for t, d in facts)
    src = "" if v["v1"] else "<small>Elo only</small>"
    return (f'<li class="game" data-kick="{kick:%Y-%m-%dT%H:%M}" data-conf="{v["fav_p"]:.3f}" '
            f'data-edge="{abs(v["edge"] or 0):.2f}" data-day="{kick:%a}" data-teams="{escape(v["away"])} {escape(v["home"])}">'
            f'<details><summary><span class="kick">{when} ET</span>'
            f'<span class="team away">{_logo(v, v["away"])}{escape(v["away"])}{_ml(v, v["away"], 1 - v["p"])}</span>'
            f'<span class="field" style="{style}" role="img" aria-label="{escape(label)}">'
            f'<span class="fill"></span><span class="mid"></span>{marks}</span>'
            f'<span class="team home">{_logo(v, v["home"])}<span class="ab">{escape(v["home"])}</span>'
            f'{_ml(v, v["home"], v["p"])}</span>'
            f'<span class="pct"><span class="num cnt" data-from="50">{v["fav_p"]:.0%}</span>'
            f'<small>{escape(v["fav"])}</small>{src}</span></summary>'
            f'<div class="qbs">{_qb(v, v["away"])}{_qb(v, v["home"])}</div>'
            f'<dl>{dl}<div class="split"><dt>Model vs line</dt><dd>{escape(_split_text(v))}</dd></div></dl>'
            f'</details></li>')


def _hero(views) -> str:
    if not views:
        return "No games scheduled yet. Predictions appear once next week's schedule is set."
    top = max(views, key=lambda v: v["fav_p"])
    under = top["away"] if top["fav"] == top["home"] else top["home"]
    text = (f"{len(views)} games. Strongest pick: <b>{escape(top['fav'])}</b> over {escape(under)}, "
            f"{top['fav_p']:.0%} to win.")
    split = max((v for v in views if v["edge"] is not None), key=lambda v: abs(v["edge"]), default=None)
    if split and abs(split["edge"]) >= 0.5:
        text += f" Biggest split from the line: <b>{escape(split['away'])} @ {escape(split['home'])}</b>."
    return text


SUITS = [("&hearts;", "red"), ("&spades;", "black"), ("&diams;", "red"), ("&clubs;", "black")]


def _floats(views) -> str:
    """Hero playing cards: the six most confident picks, win % as the card index."""
    def card(i, v):
        suit, ink = SUITS[i % 4]
        qb = v["qb"].get(v["fav"])
        shot = f'<img class="shot" src="{escape(qb[1])}" alt="">' if qb and qb[1] else ""
        idx = f'{v["fav_p"] * 100:.0f}<i>{suit}</i>'
        return (f'<div class="card {ink}"><span class="idx">{idx}</span>{shot}{_logo(v, v["fav"], "badge")}'
                f'<b>{escape(v["fav"])}</b><span class="ml">{american(v["fav_p"])} vs {escape(v["dog"])}</span>'
                f'<span class="idx r">{idx}</span><span class="back"></span></div>')
    return "".join(card(i, v) for i, v in enumerate(sorted(views, key=lambda v: -v["fav_p"])[:6]))


def _ticker(views) -> str:
    """One pass of the odds board; the template repeats it so the loop is seamless."""
    def tick(v):
        side = lambda t, p: (f'{_logo(v, t, "tlogo")}<span class="tm">{escape(t)}</span>'
                             f'<b class="{"fav" if t == v["fav"] else "dog"}">{american(p)}</b>')
        edge = ""
        if v["edge"] is not None and abs(v["edge"]) >= EDGE_PTS:
            edge = f'<em>&#9733; Edge {escape(v["home"] if v["edge"] > 0 else v["away"])}</em>'
        return (f'<span class="tick">{side(v["away"], 1 - v["p"])}<i>@</i>{side(v["home"], v["p"])}'
                f'<span class="sp">{escape(spread_text(v))}</span>{edge}</span>')
    return "".join(tick(v) for v in sorted(views, key=lambda v: v["kick"]))


def _metrics(m) -> str:
    models = [(n, s) for n, s in [("Outcome v1", m.get("outcome_v1")), ("Elo", m["elo"]),
                                  ("Closing line", m["market_close"])] if s]
    keys = [("log_loss", ".3f"), ("brier", ".3f"), ("margin_mae", ".2f")]
    best = {k: min(s[k] for _, s in models) for k, _ in keys}
    return "".join(
        f"<tr class=\"rrow\"><td>{n}</td><td class=\"cnt\">{s['games']:,}</td>"
        + "".join(f'<td class="cnt{" best" if s[k] == best[k] else ""}">{s[k]:{f}}</td>' for k, f in keys) + "</tr>"
        for n, s in models)


def main() -> None:
    games = pd.read_csv(PROC / "games_all.csv", low_memory=False)
    m = json.loads((PROC / "model_metrics.json").read_text())
    week = week_predictions(games, m["elo"]["k"], m["elo"]["hfa"])
    logos, shots, qbs = load_assets(games)
    grades = load_grades()
    qb = grades[grades["pos"] == "QB"] if len(grades) else grades
    qb_grades = {r.player_id: (r.rank, r.grade) for r in qb.itertuples()}
    views = [game_view(g, logos, qbs, qb_grades) for g in week.itertuples()]
    ptabs, ppanels = _players(grades, logos, shots)
    days = list(dict.fromkeys(f"{v['kick']:%a}" for v in sorted(views, key=lambda v: v["kick"])))
    chips = '<button class="pill" type="button" data-day="all" aria-pressed="true">All games</button>' + "".join(
        f'<button class="pill" type="button" data-day="{d}" aria-pressed="false">{d}</button>' for d in days)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(Template(TEMPLATE.read_text(encoding="utf-8")).substitute(
        # data date, not wall-clock time, so an unchanged rerun leaves the page byte-identical
        updated=games.loc[games["completed"], "date"].max(),
        season=int(week["season"].iloc[0]) if len(week) else "", week=int(week["week"].iloc[0]) if len(week) else "",
        hero=_hero(views), floats=_floats(views), ticker=_ticker(views), chips=chips, games="\n".join(_game(v) for v in views),
        test_seasons=m["elo"]["test_seasons"], metrics=_metrics(m), ptabs=ptabs, ppanels=ppanels,
        pseason=int(grades["season"].iloc[0]) if len(grades) else "",
    ), encoding="utf-8")
    print(f"site: {len(views)} games -> {OUT}")


if __name__ == "__main__":
    main()
