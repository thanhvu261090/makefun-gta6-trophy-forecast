#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target instant.

Reads every ``*.in`` case from $MAKEFUN_INPUT_DIR (else /input) and writes
one ``<id>.out`` forecast per case to $MAKEFUN_OUTPUT_DIR (else /output).
Python 3.13, standard library only, deterministic, no network.

Output per case: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}
with one row per trophy, five finite quantiles in percent, non-decreasing.

Model
-----
Everything is done in HOLDER COUNTS, not rates. A trophy's holders only
ever increase; its rate moves because the tracked population jumps at a
refresh (dilution) and because holders keep earning (progression). Rates
confound the two, counts separate them.

The page the house reads is a snapshot: Exophase refreshes its counts on
its own schedule, so consecutive reads often show the same page, and a
refresh moves every rate at once. Two regimes follow from that.

A. No refresh in the case's history (the launch-night practice cases).
   The page has been frozen for hours, so a refresh is due before the
   target. The refresh brings in new players: the tracked population
   grows by a multiple alpha (the first refresh after launch took it from
   638 to 3,868, alpha about 5), and the newcomers have earned each
   trophy at only a fraction of the old cohort's rate. The forecast is a
   count model: newcomers' holdings of trophy i are Poisson with mean
   lam_i = k_i * r_i * N_new, where r_i is the old rate, N_new the number
   of newcomers, and k_i the newcomers' earn fraction for that trophy.
   k_i varies only mildly by how quickly the trophy is earned (a
   chapter-one trophy is earned no faster by a newcomer than a side
   trophy is, once the totals are respected): k_i is proportional to a
   speed score to the power 1/4, fitted so the newcomers' total holdings
   match the launch-night aggregate of about 0.032 non-intro trophies
   per newcomer. The quantiles are the quantiles of the resulting count
   distribution, mixed over a small grid of population multiples -- a
   discrete count distribution, not a symmetric logit-normal, because for
   the rare trophies the answer is a small integer count of newcomers
   (0, 1, 2, ...), and a symmetric spread around a median is far wider
   than the truth. The one trophy already at 100% stays at 100%: every
   newcomer has it, so its band collapses to it.

B. Refreshes in the case's history (the final case, two days in). The
   population and the counts are projected in log time, which saturates:
   log P and log n_i are each fitted by least squares against
   ln(1 + hours since the first read), the population on the refresh
   points, the counts on the inter-refresh segments where the population
   is flat and every new holder is cohort progression. Newcomers'
   holdings at future refreshes are estimated from the observed refresh
   jumps (jump minus the cohort's growth in that interval, over the
   population jump), shrunk toward the same type prior as in A. The
   target holders are the cohort's projected holders plus the future
   newcomers' holdings, over the projected population. The spread
   combines the fit's residual, the Poisson noise on the counts and the
   population uncertainty, and is floored so it never claims more
   precision than the data supports.

If either fit is unusable (too few distinct reads), the forecast falls
back to a logit log-time trend on the last rate, as the house's own
curve baseline does.

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

# Newcomers' earn fraction, by how fast the trophy is earned. Only the
# relative speeds matter: the scale is fitted to the expected total.
SPEED_PLATINUM = 0.05
SPEED_CHAPTER = (1.0, 0.7, 0.5, 0.35)  # Chapter 1..4
SPEED_FINAL_MISSION = 0.15
SPEED_GRIND = 0.08
SPEED_STORY_MISSION = 0.4
SPEED_DEFAULT = 0.25
SPEED_EXPONENT = 0.25  # mild concentration: k ~ speed ** 1/4

GRIND_WORDS = ("250 stunts", "100 locations", "collectible card",
               "bank vaults", "mission strands", "gold medal",
               "100% completion")

# A newcomer in their first hours earns the intro trophy plus about this
# many other trophies on average (launch-night aggregate: 103 non-intro
# instances over 3,230 newcomers).
OTHER_PER_NEWCOMER = 0.032

# First-refresh population multiple after launch (638 -> 3,868), with a
# small grid around it for the mixture. Later refreshes use the observed
# growth instead (regime B).
ALPHA_GRID = (4.0, 5.0, 6.25)
ALPHA_WEIGHTS = (0.25, 0.5, 0.25)


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


def speed_score(trophy):
    """How quickly a newcomer earns this trophy, relative to others."""
    tier = trophy.get("tier", "")
    desc = (trophy.get("description", "") or "").lower()
    if tier == "platinum" or "100% completion" in desc:
        return SPEED_PLATINUM
    if "chapter 1" in desc:
        return SPEED_CHAPTER[0]
    if "chapter 2" in desc:
        return SPEED_CHAPTER[1]
    if "chapter 3" in desc:
        return SPEED_CHAPTER[2]
    if "chapter 4" in desc:
        return SPEED_CHAPTER[3]
    if "final mission" in desc:
        return SPEED_FINAL_MISSION
    if any(w in desc for w in GRIND_WORDS):
        return SPEED_GRIND
    if desc.startswith("complete ") and ("mission" in desc
                                         or "heist" in desc
                                         or "run" in desc):
        return SPEED_STORY_MISSION
    return SPEED_DEFAULT


def poisson_pmf(lam, kmax=120):
    """Poisson pmf for k = 0..kmax, renormalised after truncation."""
    if lam <= 0.0:
        return [1.0]
    p0 = math.exp(-lam)
    pmf = [p0]
    s = p0
    for k in range(1, kmax + 1):
        pk = pmf[-1] * lam / k
        pmf.append(pk)
        s += pk
        if pk < 1e-13 and k > lam + 12:
            break
    return [p / s for p in pmf]


def mixture_quantiles(rate_points):
    """Invert a mixture CDF given as {rate: weight} for the five levels.

    rate_points maps a possible rate (percent) to its probability mass.
    Returns five non-decreasing quantiles rounded to 4 decimals.
    """
    items = sorted(rate_points.items())
    qs = []
    cum = 0.0
    idx = 0
    for tau in LEVELS:
        while idx < len(items) and cum + items[idx][1] < tau:
            cum += items[idx][1]
            idx += 1
        qs.append(items[min(idx, len(items) - 1)][0])
    out = []
    prev = 0.01
    for q in qs:
        q = max(round(clamp(q), 4), prev)
        out.append(q)
        prev = q
    return out


def fit_line(xs, ys):
    """Least-squares line y = a + b*x. Returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(
            sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.3
    return a, b, resid


def refresh_forecast(trophy, n_old, rate_old, pop_now, horizon_hours,
                     elapsed_hours):
    """Regime A: no refresh observed yet. A refresh is due before the
    target; model it as a count process mixed over population multiples."""
    if rate_old >= 99.99:
        return [99.99] * 5
    # Newcomers' earn fraction for this trophy, up to a fitted scale.
    # The scale is set by the caller through k_scale (shared across
    # trophies); here it is passed via the trophy dict's _k entry.
    k = trophy.get("_k", 0.05)
    points = {}
    # Population multiple grows mildly with how long the page has been
    # flat: a longer flat stretch has accumulated more newcomers.
    stretch = max(1.0, (elapsed_hours + horizon_hours) / 4.7)
    for alpha, w in zip(ALPHA_GRID, ALPHA_WEIGHTS):
        a = alpha * (stretch ** 0.3)
        a = max(2.5, min(10.0, a))
        p_new = pop_now * (1.0 + a)
        n_new = pop_now * a
        lam = k * (rate_old / 100.0) * n_new
        for m, p in enumerate(poisson_pmf(lam)):
            rate = (n_old + m) / p_new * 100.0
            rate = round(clamp(rate), 4)
            points[rate] = points.get(rate, 0.0) + w * p
    return mixture_quantiles(points)


def project_counts(history, trophies, checkpoint, target):
    """Regime B: project holder counts and population to the target.

    Returns (pop_target, holders_target, sigma) where holders_target is a
    list of projected holder counts per trophy and sigma a per-trophy
    logit spread. Returns None if the history cannot support it.
    """
    if len(history) < 4:
        return None
    t0 = parse_time(history[0]["at"])
    hours = [(parse_time(r["at"]) - t0).total_seconds() / 3600.0
             for r in history]
    pops = [float(r["players_tracked"]) for r in history]
    if any(p <= 0 for p in pops) or pops[-1] <= 0:
        return None
    target_h = (target - t0).total_seconds() / 3600.0
    if target_h <= hours[-1]:
        return None

    # --- population: log P against ln(1 + hours), on refresh points ---
    refresh_idx = [0]
    for j in range(1, len(history)):
        if pops[j] != pops[j - 1]:
            refresh_idx.append(j)
    if len(refresh_idx) < 2:
        return None
    px = [math.log1p(hours[j]) for j in refresh_idx]
    py = [math.log(pops[j]) for j in refresh_idx]
    pop_fit = fit_line(px, py)
    if pop_fit is None:
        return None
    pa, pb, presid = pop_fit
    pb = max(0.0, min(2.0, pb))
    pop_target = math.exp(pa + pb * math.log1p(target_h))
    pop_target = max(pops[-1], pop_target)

    # --- newcomer earn probabilities from the observed refresh jumps ---
    # For each refresh, the count jump minus the cohort's growth in that
    # interval, over the population jump. Cohort growth in the interval
    # is interpolated from the surrounding flat reads.
    n_hist = []
    for r in history:
        row = []
        for v in (r.get("rates") or []):
            row.append(None if v is None else v / 100.0 * r["players_tracked"])
        n_hist.append(row)
    p_est = []
    for idx, trophy in enumerate(trophies):
        num = 0.0
        den = 0.0
        for j in refresh_idx[1:]:
            before = n_hist[j - 1][idx] if idx < len(n_hist[j - 1]) else None
            after = n_hist[j][idx] if idx < len(n_hist[j]) else None
            if before is None or after is None:
                continue
            # cohort growth during the interval: from the flat segment
            # before the refresh, if it spans at least two reads
            cohort = 0.0
            seg = [k for k in range(j - 1, -1, -1)
                   if pops[k] == pops[j - 1]]
            if len(seg) >= 2:
                k0, k1 = seg[-1], seg[0]
                c0 = n_hist[k0][idx]
                c1 = n_hist[k1][idx]
                if c0 is not None and c1 is not None and hours[k1] > hours[k0]:
                    rate = (c1 - c0) / (hours[k1] - hours[k0])
                    cohort = max(0.0, rate * (hours[j] - hours[j - 1]))
            jump = after - before - cohort
            dp = pops[j] - pops[j - 1]
            if dp > 0:
                num += max(0.0, jump)
                den += dp
        p_data = (num / den) if den > 0 else None
        p_est.append(p_data)

    # --- cohort holder projection: log n against ln(1 + hours) ---
    # Use only reads where the population was flat since the previous
    # read (no newcomer jump in between) plus the first read, so the
    # fit measures the cohort alone.
    holders = []
    sigmas = []
    x_target = math.log1p(target_h)
    for idx, trophy in enumerate(trophies):
        xs = []
        ys = []
        for j, r in enumerate(history):
            if idx >= len(n_hist[j]) or n_hist[j][idx] is None:
                continue
            if j > 0 and pops[j] != pops[j - 1]:
                continue  # refresh read: skip, its jump is newcomers
            n = max(1.0, n_hist[j][idx])
            xs.append(math.log1p(hours[j]))
            ys.append(math.log(n))
        # de-duplicate identical x values (flat reads repeat the count)
        pts = {}
        for x, y in zip(xs, ys):
            pts[x] = y
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        fit = fit_line(xs, ys) if len(xs) >= 3 else None
        last_n = max(1.0, n_hist[-1][idx]) if idx < len(n_hist[-1]) \
            and n_hist[-1][idx] is not None else 1.0
        if fit is None:
            # No usable trend: holders stay where they are.
            proj = last_n
            resid = 0.3
        else:
            a, b, resid = fit
            b = max(-0.5, min(3.0, b))
            proj = math.exp(a + b * x_target)
            proj = max(last_n, proj)
        # Future newcomers' holdings, shrunk toward the type prior.
        p_prior = speed_score(trophy) ** SPEED_EXPONENT
        # prior in probability units: scale so the prior total matches
        # OTHER_PER_NEWCOMER per newcomer (same normalisation as A).
        p_data = p_est[idx]
        if p_data is None:
            p_new = None  # filled in by the caller-scale pass
        else:
            p_new = p_data
        holders.append(proj)
        sigmas.append((resid, p_new, p_prior, last_n))
    return pop_target, pops[-1], holders, sigmas, x_target


def trend_fallback(trophy, series, horizon_minutes, target_hours):
    """Last-resort forecast: logit log-time trend on the last rate."""
    last_rate = series[-1][1] if series else 1.0
    if last_rate >= 99.99:
        return [99.99] * 5
    distinct = []
    for h, r in series:
        if not distinct or abs(r - distinct[-1][1]) > 1e-9:
            distinct.append((h, r))
    days = horizon_minutes / 1440.0
    if len(distinct) >= 3:
        xs = [math.log1p(h) for h, _ in distinct]
        ys = [logit(r) for _, r in distinct]
        fit = fit_line(xs, ys)
        if fit is not None:
            a, b, resid = fit
            b = max(-2.0, min(2.0, b))
            pred = a + b * math.log1p(target_hours)
            w = len(distinct) / (len(distinct) + 6.0)
            med = logit(last_rate) + w * (pred - logit(last_rate))
            med = max(logit(last_rate) - 2.5,
                      min(logit(last_rate) + 2.5, med))
            sigma = math.sqrt((0.5 * math.sqrt(days)) ** 2
                              + max(0.1, min(1.0, resid)) ** 2 + 0.15 ** 2)
            return [round(clamp(inv_logit(med + z * sigma)), 4)
                    for z in Z]
    sigma = math.sqrt((0.5 * math.sqrt(days)) ** 2 + 0.6 ** 2)
    med = logit(last_rate)
    return [round(clamp(inv_logit(med + z * sigma)), 4) for z in Z]


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_minutes = case.get("horizon_minutes")
    if horizon_minutes is None:
        horizon_minutes = (target - checkpoint).total_seconds() / 60.0

    t0 = parse_time(history[0]["at"]) if history else checkpoint
    target_hours = (target - t0).total_seconds() / 3600.0
    elapsed_hours = max(0.0, (checkpoint - t0).total_seconds() / 3600.0)

    # Per-trophy series of (hours since first read, rate).
    series = []
    for idx, trophy in enumerate(trophies):
        pts = []
        for read in history:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            h = (parse_time(read["at"]) - t0).total_seconds() / 3600.0
            pts.append((h, float(rates[idx])))
        series.append(pts)

    last_rates = [s[-1][1] if s else 1.0 for s in series]
    pops = [float(r["players_tracked"]) for r in history
            if r.get("players_tracked")]
    pop_now = pops[-1] if pops else 638.0
    n_refreshes = sum(1 for j in range(1, len(pops))
                      if pops[j] != pops[j - 1])

    # Newcomer earn fractions, fitted so the newcomers' total holdings
    # match OTHER_PER_NEWCOMER per newcomer (non-intro trophies only).
    denom = 0.0
    for idx, trophy in enumerate(trophies):
        if last_rates[idx] >= 99.99:
            continue
        denom += (speed_score(trophy) ** SPEED_EXPONENT) * last_rates[idx]
    k_scale = (OTHER_PER_NEWCOMER * 100.0 / denom) if denom > 0 else 0.0667
    k_list = []
    for idx, trophy in enumerate(trophies):
        if last_rates[idx] >= 99.99:
            k_list.append(0.0)
        else:
            k_list.append(k_scale * (speed_score(trophy) ** SPEED_EXPONENT))

    rows = []
    if n_refreshes == 0:
        # Regime A: the page is frozen; a refresh is due.
        for idx, trophy in enumerate(trophies):
            t = dict(trophy)
            t["_k"] = k_list[idx]
            n_old = (round(last_rates[idx] / 100.0 * pop_now)
                     if series[idx] else 0)
            rows.append({"id": trophy["id"],
                         "q": refresh_forecast(
                             t, n_old, last_rates[idx], pop_now,
                             horizon_minutes / 60.0, elapsed_hours)})
        return {"trophies": rows}

    # Regime B: refreshes observed; project counts and population.
    proj = project_counts(history, trophies, checkpoint, target)
    if proj is None:
        for idx, trophy in enumerate(trophies):
            rows.append({"id": trophy["id"],
                         "q": trend_fallback(
                             trophy, series[idx], horizon_minutes,
                             target_hours)})
        return {"trophies": rows}
    pop_target, pop_last, holders, sigmas, x_target = proj
    hours = [(parse_time(r["at"]) - t0).total_seconds() / 3600.0
             for r in history]
    x_first = math.log1p(hours[0]) if hours else 0.0
    x_last = math.log1p(hours[-1]) if hours else 0.0
    span = max(1e-9, x_last - x_first)
    extrap = math.sqrt(1.0 + max(0.0, x_target - x_last) / span)
    for idx, trophy in enumerate(trophies):
        if last_rates[idx] >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.99] * 5})
            continue
        resid, p_data, p_prior_speed, last_n = sigmas[idx]
        # Newcomer earn probability: data estimate shrunk to the prior.
        p_prior = k_list[idx] * last_rates[idx] / 100.0
        if p_data is None:
            p_i = p_prior
        else:
            # Weight the data by how many newcomers it rests on; the
            # prior counts as 5,000 newcomers' worth of evidence.
            den = max(0.0, pop_target - pop_last)
            w_data = den / (den + 5000.0) if den > 0 else 0.0
            p_i = w_data * p_data + (1.0 - w_data) * p_prior
        holders_t = holders[idx] + p_i * max(0.0, pop_target - pop_last)
        rate = holders_t / pop_target * 100.0
        sigma = math.sqrt((resid * extrap) ** 2
                          + 1.0 / max(1.0, holders_t) + 0.12 ** 2)
        sigma = max(0.15, min(1.0, sigma))
        med = logit(rate)
        qs = []
        prev = 0.01
        for z in Z:
            q = max(round(clamp(inv_logit(med + z * sigma)), 4), prev)
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
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

