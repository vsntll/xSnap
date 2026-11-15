"""Regression gate (ported from xValue): compare the scores just written to
data/processed/model_metrics.json against git HEAD's copy and against the
hand-pinned accepted_metrics.json, and exit non-zero if a model got materially
worse, so update.py stops before publishing. The pinned baseline stops a model
being walked down a little per run.

A deliberate model change that trips it: check the numbers, run with --accept,
and commit model_metrics.json + accepted_metrics.json.

Run:  .venv\\Scripts\\python src/check_model_metrics.py [--accept]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "data" / "processed" / "model_metrics.json"
ACCEPTED = ROOT / "accepted_metrics.json"
LOG_LOSS_RISE, MAE_RISE, R2_DROP = 0.01, 0.10, 0.02
# (section, key, tolerance): positive = may rise at most this much, negative = may fall at most this much
CHECKS = [(m, key, tol) for m in ("elo", "outcome_v1") for key, tol in (("log_loss", LOG_LOSS_RISE),
                                                                         ("margin_mae", MAE_RISE))]
CHECKS.append(("value", "r2", -R2_DROP))


def _get(d: dict, section: str, key: str):
    s = d.get(section, {})
    return s.get("model", {}).get(key) if section == "value" else s.get(key)


def compare(old: dict, new: dict) -> list[str]:
    """Failures (empty = pass); a model missing on either side is skipped."""
    fails = []
    for model, key, tol in CHECKS:
        a, b = _get(old, model, key), _get(new, model, key)
        if a is None or b is None:
            continue
        bad = b < a + tol if tol < 0 else b > a + tol
        print(f"  {model} {key}: {a:.4f} -> {b:.4f} ({b - a:+.4f}, limit +{tol})  {'FAIL' if bad else 'ok'}")
        if bad:
            fails.append(f"{model} {key} {a:.4f} -> {b:.4f}")
    return fails


def _committed() -> dict | None:
    try:
        out = subprocess.run(["git", "show", "HEAD:data/processed/model_metrics.json"], cwd=ROOT,
                             capture_output=True, check=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return json.loads(out.stdout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept", action="store_true", help="pin the current scores as accepted_metrics.json")
    new = json.loads(METRICS.read_text())
    if ap.parse_args().accept:
        ACCEPTED.write_text(json.dumps(new, indent=2) + "\n")
        print(f"pinned current scores as {ACCEPTED.name} - commit it with the model change")
        return
    fails = []
    for label, old in (("last commit", _committed()), ("accepted", json.loads(ACCEPTED.read_text()))):
        if old:
            print(f"vs {label}:")
            fails += [f"{f} (vs {label})" for f in compare(old, new)]
    if fails:
        raise SystemExit("models got worse past tolerance - not publishing:\n  " + "\n  ".join(fails))
    print("gate passed")


if __name__ == "__main__":
    main()
