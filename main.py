#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target instant.

Reads every ``*.in`` case from $MAKEFUN_INPUT_DIR (else /input) and writes
one ``<id>.out`` forecast per case to $MAKEFUN_OUTPUT_DIR (else /output).
Python 3.13, standard library only, deterministic, no network.

Output per case: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}
with one row per trophy, five finite quantiles in percent, non-decreasing.

Model, per trophy
---------------
The page the house reads is a snapshot: Exophase refreshes its counts on
its own schedule, so consecutive reads often show the same page, and a
refresh moves every rate at once. Two regimes follow from that.

1. Moving history. When the trophy's rate actually moved between reads,
   fit a least-squares line of logit(rate) against ln(1 + hours since the
   first read) and extrapolate it to the target, shrunk toward the last
   observed rate by n / (n + 6) where n is the number of distinct reads the
   fit rests on. Rates saturate in log time (players finish the early
   trophies first, then the curve flattens), which is why log time, not
   linear time, is the axis. The spread combines the random-walk term
   (0.5 * sqrt(horizon in days)), the fit's residual spread, and a small
   refresh term.

2. Flat history. When every read shows the same page, the page is stale:
   a refresh is due before the target, and it brings in the players who
   joined since the page froze. Those newcomers have earned each trophy at
   only a fraction of the early cohort's rate -- on launch night the
   refresh took the tracked count from 638 to 3,868 while the trophy
   totals show the newcomers had earned the non-100% trophies at roughly
   7% of the launch-night rate (they had mostly just finished Gone
   Fishing, the one trophy that stayed at 100%). So the median is the
   last rate times a dilution factor (1 + k * a) / (1 + a), where a is the
   expected growth multiple of the tracked population at the refresh
   (about 5 on launch night, smaller for later refreshes) and k is the
   newcomers' earn fraction for that trophy: early story beats are earned
   quickly, so new players catch up on them (Chapter 1: 0.35, Chapter 2:
   0.20, later chapters less); grinds, 100% completion and the platinum
   take far longer than a newcomer's first hours, so k is near 0 there;
   other trophies sit in between. The spread is wide here on purpose:
   a refresh is a jump, not a drift, so the 5%-95% band must be able to
   hold it. The trophy already at 100% stays at 100%: its median is
   99.99 and its band collapses to it.

Everything is clamped to [0.01, 99.99] for the logit, and quantiles are
rounded to 4 decimals and forced non-decreasing.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Newcomers' earn fraction at a refresh, by how fast the trophy is earned.
# Early story beats are picked up in a newcomer's first hours; grinds,
# 100% completion and the platinum are not.
K_EARLY_CHAPTER = (0.35, 0.20, 0.12, 0.08)  # Chapter 1..4
K_STORY_MISSION = 0.08
K_DEFAULT = 0.05
K_GRIND = 0.02
K_PLATINUM = 0.0

GRIND_WORDS = ("250 stunts", "100 locations", "collectible card",
               "bank vaults", "mission strands", "gold medal",
               "100% completion")


def clamp(x, lo=0.01, hi=99.99):
    return lo if x < lo else hi if x > hi else x


def logit(x):
    p = clamp(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l > 20.0:
        return 99.99
    if l < -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-l))


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def newcomer_fraction(trophy):
    """k: the fraction of the old cohort's rate at which newcomers earn it."""
    tier = trophy.get("tier", "")
    desc = (trophy.get("description", "") or "").lower()
    name = (trophy.get("name", "") or "").lower()
    if tier == "platinum" or "100% completion" in desc:
        return K_PLATINUM
    if "chapter 1" in desc:
        return K_EARLY_CHAPTER[0]
    if "chapter 2" in desc:
        return K_EARLY_CHAPTER[1]
    if "chapter 3" in desc:
        return K_EARLY_CHAPTER[2]
    if "chapter 4" in desc:
        return K_EARLY_CHAPTER[3]
    if "final mission" in desc:
        return K_GRIND
    if any(w in desc for w in GRIND_WORDS):
        return K_GRIND
    if desc.startswith("complete ") and ("mission" in desc
                                         or "heist" in desc
                                         or "run" in desc):
        return K_STORY_MISSION
    return K_DEFAULT


def is_early_story(trophy):
    desc = (trophy.get("description", "") or "").lower()
    return "chapter" in desc or "final mission" in desc


def fit_trend(points):
    """OLS of logit(rate) on ln(1 + hours since first read).

    points: [(hours, rate)] with distinct rates only, time-ordered.
    Returns (intercept, slope, residual_std) or None if unusable.
    """
    if len(points) < 3:
        return None
    xs = [math.log1p(h) for h, _ in points]
    ys = [logit(r) for _, r in points]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    slope = max(-2.0, min(2.0, slope))
    intercept = my - slope * mx
    if n > 2:
        resid = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2
                for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.3
    return intercept, slope, resid


def forecast_trophy(trophy, series, horizon_minutes, target_hours):
    """series: [(hours_since_first_read, rate)] time-ordered, non-null.

    Returns the five quantiles for this trophy.
    """
    last_rate = series[-1][1] if series else None
    if last_rate is None:
        last_rate = 1.0
    days = horizon_minutes / 1440.0

    if last_rate >= 99.99:
        # Already at 100%: the refresh left it there, and it stays there.
        return [99.99] * 5

    last_logit = logit(last_rate)

    # Collapse consecutive reads showing the same rate: a frozen page is
    # one observation, not many, and a flat tail must not flatten a trend.
    distinct = []
    for h, r in series:
        if not distinct or abs(r - distinct[-1][1]) > 1e-9:
            distinct.append((h, r))

    trend = fit_trend(distinct) if len(distinct) >= 3 else None
    if trend is not None:
        intercept, slope, resid = trend
        x_target = math.log1p(target_hours)
        trend_logit = intercept + slope * x_target
        w = len(distinct) / (len(distinct) + 6.0)
        pred = last_logit + w * (trend_logit - last_logit)
        pred = max(last_logit - 2.5, min(last_logit + 2.5, pred))
        resid = max(0.1, min(1.0, resid))
        sigma = math.sqrt((0.5 * math.sqrt(days)) ** 2 + resid ** 2
                          + 0.15 ** 2)
        median_logit = pred
    else:
        # Flat history: the page is stale and a refresh is due.
        horizon_h = horizon_minutes / 60.0
        # Expected growth multiple of the tracked population at the
        # refresh: about 5 on launch night, decaying for later refreshes.
        alpha = 1.5 + 3.5 * math.exp(-max(0.0, horizon_h - 6.0) / 24.0)
        k = newcomer_fraction(trophy)
        dil = (1.0 + k * alpha) / (1.0 + alpha)
        median = last_rate * dil
        # Over a long horizon the early cohort keeps playing and earning,
        # which offsets dilution for early story trophies in particular.
        if horizon_h > 6.0:
            early = 1.0 if is_early_story(trophy) else 0.5
            prog = 1.0 + 0.10 * min(5.0, days) * early
            median = min(median * prog, 99.99)
        median_logit = logit(median)
        sigma = math.sqrt((0.5 * math.sqrt(days)) ** 2 + 0.60 ** 2)

    qs = []
    prev = 0.01
    for z in Z:
        q = round(clamp(inv_logit(median_logit + z * sigma)), 4)
        q = max(q, prev)
        qs.append(q)
        prev = q
    return qs


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_minutes = case.get("horizon_minutes")
    if horizon_minutes is None:
        horizon_minutes = (target - checkpoint).total_seconds() / 60.0

    t0 = None
    if history:
        t0 = parse_time(history[0]["at"])
    target_hours = ((target - t0).total_seconds() / 3600.0
                    if t0 is not None else horizon_minutes / 60.0)

    rows = []
    for idx, trophy in enumerate(trophies):
        series = []
        for read in history:
            rates = read.get("rates") or []
            if idx >= len(rates):
                continue
            r = rates[idx]
            if r is None:
                continue
            h = (parse_time(read["at"]) - t0).total_seconds() / 3600.0
            series.append((h, float(r)))
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(trophy, series,
                                          horizon_minutes, target_hours)})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
