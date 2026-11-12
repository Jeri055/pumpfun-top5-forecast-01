#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: calibrated Monte Carlo forecaster.

Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json.
Python 3.13, standard library only, deterministic, offline.

Model
-----
Pump.fun market caps are extremely persistent in *level* but churn fast in
*identity*: young leaders graduate, stall, or get displaced by coins born
minutes earlier, and coins born after the history cut can take top-5 places.
We therefore forecast by explicit simulation of the future chart:

1.  For every plausible candidate coin we estimate a projected log10 market
    cap (in SOL) at the answer horizon: last size plus a *shrunk, damped*
    recent growth rate (growth persistence decays with a ~1.25 h time
    constant, so a hot 10 minutes cannot extrapolate for 10 hours), plus a
    common market-level drift estimated from how the top of the book itself
    inflated during the history (square-root-in-horizon damped).
2.  Each coin's future size is log-normal around that projection.  The spread
    grows with sqrt(horizon), with the coin's own observed volatility, and is
    wider for coins still on the bonding curve (they either graduate and run,
    or die) than for graduated pool coins.
3.  Coins born after the cut are modelled as anonymous challenger draws:
    a horizon-growing number of draws anchored near the projected top of the
    book (every 10-minute cohort in the practice history produced a coin that
    reached near the then-current top size), plus a few "field" draws for
    known coins outside the candidate set.  When a challenger wins a place in
    a simulation, that place's probability mass stays *unnamed*, exactly how
    the scorer treats post-cut coins.
4.  We run many seeded simulations, rank each simulated chart, and read off
    place probabilities and place size distributions from the frequencies.
    A final calibration step caps each place's total named mass at a
    horizon-dependent level (longer horizon -> more mass left unnamed for
    newcomers and movers) and smooths the size bins with a normal prior so
    no plausible bin is ever given a near-zero probability.

All horizon dependence is read from task.json (the reads), never hardcoded,
so the same code serves the 1-hour practice miniature and the 24-hour final.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

SEED = 20261112
N_SIMS = 20000

# --- structural parameters (kept horizon-aware, not fitted to the practice) ---
GROWTH_TAU_H = 1.25        # growth-persistence time constant, hours
GROWTH_SHRINK_H = 0.75     # observation span (h) at which growth is half trusted
GROWTH_CLAMP = 1.1         # max |log10 growth| per hour trusted for a coin
SIGMA_BASE = 0.12          # log10 size spread at zero horizon
SIGMA_PER_SQRTH = 0.21     # extra spread per sqrt(hour of horizon)
SIGMA_CAP = 0.85
COMMON_FRAC = 0.55         # share of core sd that is a common market shock
VOL_WEIGHT = 0.40          # weight of a coin's own per-hour volatility
CURVE_EXTRA_SIGMA = 0.08   # un-graduated coins are more uncertain
YOUNG_EXTRA_SIGMA = 0.06   # coins < 30 min old at the cut
YOUNG_AGE_H = 0.5
DRIFT_CAP = 0.45           # cap on the common market-level drift (log10)
NAMED_SUM_INTERCEPT = 0.80 # max named mass per place at zero horizon
NAMED_SUM_DECAY = 0.20     # how much that cap falls as horizon grows
NAMED_SUM_TAU_H = 5.0
MIN_ROW_P = 0.004          # smallest named place probability worth a row
MAX_ROWS_PER_PLACE = 60
SIZE_PRIOR_WEIGHT = 0.25   # weight of the anchored normal prior in size smoothing
SIZE_MIN_ROW_P = 0.0015    # smallest named size probability worth a row


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def size_of(lamports):
    return math.log10(int(lamports)) - 9.0


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 5))
    edges = [float(e) for e in task["size"]["edges"]]
    n_bins = len(edges) + 1

    marks = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise SystemExit("history holds no ok mark")
    marks.sort(key=lambda m: m["at"])

    t0 = parse_time(task["t0"])
    last_time = parse_time(marks[-1]["at"])

    # Answer horizon: the single forecast is scored at every read, so aim at
    # the mean read time; spread of reads also widens the size distribution.
    read_times = [parse_time(r) for r in task["reads"]]
    horizon_h = max(
        0.05,
        sum((rt - last_time).total_seconds() for rt in read_times)
        / len(read_times)
        / 3600.0,
    )
    read_span_h = (
        (max(read_times) - min(read_times)).total_seconds() / 3600.0
        if len(read_times) > 1
        else 0.0
    )

    # ---- per-coin trajectories -------------------------------------------
    series = {}   # mint -> list of (hours since t0, size)
    info = {}     # mint -> dict from its most recent row
    for m in marks:
        t_h = (parse_time(m["at"]) - t0).total_seconds() / 3600.0
        for r in m["rows"]:
            mint = r["mint"]
            series.setdefault(mint, []).append((t_h, size_of(r["market_cap_lamports"])))
            info[mint] = r
    last_t_h = (last_time - t0).total_seconds() / 3600.0

    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    last_size = {}
    for mint, pts in series.items():
        last_size[mint] = pts[-1][1]
    if not last_rows:
        # fall back to the most recent non-empty mark for the chart level
        for m in reversed(marks):
            if m["rows"]:
                last_rows = sorted(m["rows"], key=lambda r: r["rank"])
                break

    # ---- candidate set -----------------------------------------------------
    cand = set(r["mint"] for r in last_rows[:120])
    if len(last_rows) >= places:
        place5_size = size_of(last_rows[places - 1]["market_cap_lamports"])
    elif last_rows:
        place5_size = size_of(last_rows[-1]["market_cap_lamports"])
    else:
        place5_size = 2.0
    recent_cut = last_t_h - 0.5
    for mint, pts in series.items():
        mx_recent = max(s for (t, s) in pts if t >= recent_cut) if any(
            t >= recent_cut for (t, s) in pts
        ) else -1e9
        if mx_recent >= place5_size - 0.9:
            cand.add(mint)
    # rank candidates by last size, keep the strongest 90
    cand = sorted(cand, key=lambda mt: (-last_size.get(mt, -1e9), mt))[:90]

    # ---- cross-sectional growth (for shrinkage target) ---------------------
    def coin_growth(pts, span_h):
        """Per-hour log10 growth over the trailing span, and obs length (h)."""
        if len(pts) < 2:
            return None, 0.0
        t_end, s_end = pts[-1]
        start = None
        for (t, s) in pts:
            if t >= t_end - span_h:
                start = (t, s)
                break
        if start is None or t_end - start[0] < 1e-9:
            # use the earliest point
            start = pts[0]
        dt = t_end - start[0]
        if dt < 1e-9:
            return None, 0.0
        return (s_end - start[1]) / dt, dt

    growths = []
    for mint in cand:
        g, dt = coin_growth(series[mint], 1.0)
        if g is not None and dt >= 0.15:
            growths.append(g)
    growths.sort()
    g_cross = growths[len(growths) // 2] if growths else 0.0
    g_cross = max(-0.3, min(0.5, g_cross))

    # ---- common market drift: how fast the top of the book inflated --------
    top_slopes = []
    top_pts = []
    for m in marks:
        if len(m["rows"]) >= 3:
            t_h = (parse_time(m["at"]) - t0).total_seconds() / 3600.0
            top3 = sum(size_of(r["market_cap_lamports"]) for r in m["rows"][:3]) / 3.0
            top_pts.append((t_h, top3))
    if len(top_pts) >= 2 and top_pts[-1][0] - top_pts[0][0] > 0.05:
        # recent slope only: the second half of the history (the opening
        # minutes of a young market inflate far faster than it can sustain)
        half = top_pts[len(top_pts) // 2:]
        if len(half) >= 2 and half[-1][0] - half[0][0] > 0.03:
            slope = (half[-1][1] - half[0][1]) / (half[-1][0] - half[0][0])
        else:
            slope = (top_pts[-1][1] - top_pts[0][1]) / (top_pts[-1][0] - top_pts[0][0])
        slope = max(0.0, min(0.5, slope))
    else:
        slope = 0.05
    market_drift = min(DRIFT_CAP, slope * 0.7 * math.sqrt(horizon_h))

    # ---- per-coin projection ----------------------------------------------
    damp = GROWTH_TAU_H * (1.0 - math.exp(-horizon_h / GROWTH_TAU_H))
    core_sigma = min(SIGMA_CAP, SIGMA_BASE + SIGMA_PER_SQRTH * math.sqrt(horizon_h)
                     + 0.05 * math.sqrt(read_span_h))
    # Split the spread into a common market shock (moves every coin together,
    # so the simulated maximum is not inflated by ~90 independent draws) and
    # a per-coin idiosyncratic part.  Only the idiosyncratic part is stored
    # per coin; the common part is drawn once per simulation.
    sigma_common = COMMON_FRAC * core_sigma
    idio_base = math.sqrt(max(core_sigma ** 2 - sigma_common ** 2, 1e-6))
    vol_scale = min(1.0, math.sqrt(horizon_h))
    coins = []  # (mint, mu, idio_sigma)
    for mint in cand:
        pts = series[mint]
        s_last = pts[-1][1]
        g, dt = coin_growth(pts, 1.0)
        if g is None:
            g_shrunk = g_cross
        else:
            w = dt / (dt + GROWTH_SHRINK_H)
            g_shrunk = w * max(-GROWTH_CLAMP, min(GROWTH_CLAMP, g)) + (1 - w) * g_cross
        # own volatility: sd of step diffs scaled to per-hour
        vol = 0.0
        if len(pts) >= 3:
            diffs = []
            for (t1, s1), (t2, s2) in zip(pts, pts[1:]):
                if t2 > t1:
                    diffs.append((s2 - s1) / math.sqrt(t2 - t1))
            if len(diffs) >= 2:
                mean_d = sum(diffs) / len(diffs)
                vol = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (len(diffs) - 1))
                vol = min(vol, 0.9)
        sigma = math.sqrt(idio_base ** 2 + (VOL_WEIGHT * vol * vol_scale) ** 2)
        r = info[mint]
        if r.get("venue") != "pool":
            sigma += CURVE_EXTRA_SIGMA
        created = parse_time(r["created_at"])
        age_h = (last_time - created).total_seconds() / 3600.0
        if age_h < YOUNG_AGE_H:
            sigma += YOUNG_EXTRA_SIGMA
        mu = s_last + market_drift + g_shrunk * damp
        coins.append((mint, mu, min(sigma, SIGMA_CAP)))

    coins.sort(key=lambda c: c[0])  # stable RNG order

    # ---- challengers: post-cut newcomers + the unmodelled field ------------
    proj_sorted = sorted((mu for (_, mu, _) in coins), reverse=True)
    anchor_top = proj_sorted[min(1, len(proj_sorted) - 1)] - 0.08 if proj_sorted else 3.0
    anchor_field = proj_sorted[min(14, len(proj_sorted) - 1)] - 0.05 if proj_sorted else 2.3
    n_new = 1.5 + 1.8 * (horizon_h ** 0.7)
    n_field = 4.0
    sigma_new = 0.45
    sigma_field = 0.48

    # ---- simulation ---------------------------------------------------------
    rng = random.Random(SEED)
    gauss = rng.gauss
    place_counts = [dict() for _ in range(places)]       # mint or None(new/field)
    top5_counts = {}
    size_counts = [[0] * n_bins for _ in range(places)]
    mus = [c[1] for c in coins]
    sigs = [c[2] for c in coins]
    mints = [c[0] for c in coins]
    n_coins = len(coins)

    def bin_of(size):
        # bin 0: < edges[0]; bin i (1..len-1): [edges[i-1], edges[i]); last: >= edges[-1]
        lo, hi = 0, len(edges)
        # binary search: first edge > size
        while lo < hi:
            mid = (lo + hi) // 2
            if edges[mid] <= size:
                lo = mid + 1
            else:
                hi = mid
        return lo  # 0..len(edges) == bin index

    for _ in range(N_SIMS):
        z = gauss(0.0, sigma_common)
        draws = [mus[i] + z + gauss(0.0, sigs[i]) for i in range(n_coins)]
        # challenger draws: post-cut newcomers + the unmodelled field; they
        # share the same market shock, then draw idiosyncratically
        n_new_i = int(n_new) + (1 if rng.random() < (n_new - int(n_new)) else 0)
        extra = [anchor_top + z + gauss(0.0, sigma_new) for _ in range(n_new_i)]
        extra += [anchor_field + z + gauss(0.0, sigma_field) for _ in range(int(n_field))]
        # top-`places` among coins + challengers
        # partial selection: keep indices of the best `places`
        cand_idx = list(range(n_coins))
        # simple approach: sort coin draws desc but only need top places;
        # n_coins <= 90 so a full sort is cheap enough.
        order = sorted(cand_idx, key=lambda i: -draws[i])
        merged = []  # (size, mint or None)
        ei = 0
        extra_sorted = sorted(extra, reverse=True)
        oi = 0
        while len(merged) < places and (oi < len(order) or ei < len(extra_sorted)):
            take_coin = False
            if oi < len(order):
                if ei >= len(extra_sorted) or draws[order[oi]] >= extra_sorted[ei]:
                    take_coin = True
            if take_coin:
                i = order[oi]
                oi += 1
                merged.append((draws[i], mints[i]))
            else:
                merged.append((extra_sorted[ei], None))
                ei += 1
        for k in range(places):
            if k < len(merged):
                sz, mt = merged[k]
                size_counts[k][bin_of(sz)] += 1
                if mt is not None:
                    place_counts[k][mt] = place_counts[k].get(mt, 0) + 1
                    top5_counts[mt] = top5_counts.get(mt, 0) + 1

    # ---- place probabilities -------------------------------------------------
    named_cap = NAMED_SUM_INTERCEPT - NAMED_SUM_DECAY * (
        1.0 - math.exp(-horizon_h / NAMED_SUM_TAU_H)
    )
    probabilities = []
    for k in range(places):
        rows = [(mt, c / N_SIMS) for mt, c in place_counts[k].items()]
        rows = [(mt, p) for mt, p in rows if p >= MIN_ROW_P]
        total = sum(p for _, p in rows)
        if total > named_cap:
            scale = named_cap / total
            rows = [(mt, p * scale) for mt, p in rows]
        rows.sort(key=lambda x: (-x[1], x[0]))
        for mt, p in rows[:MAX_ROWS_PER_PLACE]:
            probabilities.append({"mint": mt, "place": k + 1, "p": round(p, 6)})

    # ---- top5 -----------------------------------------------------------------
    mu_of = {c[0]: c[1] for c in coins}
    by_top5 = sorted(top5_counts.items(), key=lambda kv: (-kv[1], -mu_of.get(kv[0], 0), kv[0]))
    chosen = [mt for mt, _ in by_top5[:places]]
    if len(chosen) < places:
        for mt, _, _ in sorted(coins, key=lambda c: -c[1]):
            if mt not in chosen:
                chosen.append(mt)
            if len(chosen) >= places:
                break
    top5 = sorted(chosen, key=lambda mt: -mu_of.get(mt, 0))

    # ---- sizes ------------------------------------------------------------------
    last_place_sizes = []
    for idx in range(places):
        if idx < len(last_rows):
            last_place_sizes.append(size_of(last_rows[idx]["market_cap_lamports"]))
        elif last_place_sizes:
            last_place_sizes.append(last_place_sizes[-1] - 0.1)
        else:
            last_place_sizes.append(2.5)
    sizes = []
    for k in range(places):
        counts = size_counts[k]
        # prior: normal anchored between the simulated mean and the last
        # observed size at this place (plus drift), hedging projection error
        sim_sizes_mean = 0.0
        # recover mean/sd of the place's size from bin midpoints approximately
        mids = []
        for b in range(n_bins):
            if b == 0:
                mids.append(edges[0] - 0.25)
            elif b == n_bins - 1:
                mids.append(edges[-1] + 0.25)
            else:
                mids.append((edges[b - 1] + edges[b]) / 2.0)
        tot = sum(counts) or 1
        sim_mean = sum(c * mids[b] for b, c in enumerate(counts)) / tot
        var_sz = sum(c * (mids[b] - sim_mean) ** 2 for b, c in enumerate(counts)) / tot
        mean_sz = 0.6 * sim_mean + 0.4 * (last_place_sizes[k] + 0.7 * market_drift)
        sd_sz = max(0.30, math.sqrt(var_sz)) * 1.15
        prior = []
        for b in range(n_bins):
            lo_e = -1e18 if b == 0 else edges[b - 1]
            hi_e = 1e18 if b == n_bins - 1 else edges[b]
            lo_c = 0.0 if b == 0 else norm_cdf((lo_e - mean_sz) / sd_sz)
            hi_c = 1.0 if b == n_bins - 1 else norm_cdf((hi_e - mean_sz) / sd_sz)
            prior.append(max(hi_c - lo_c, 0.0))
        ps = []
        for b in range(n_bins):
            p = (counts[b] + SIZE_PRIOR_WEIGHT * N_SIMS * prior[b]) / (
                N_SIMS * (1.0 + SIZE_PRIOR_WEIGHT)
            )
            ps.append(p)
        # keep sum <= 1 by construction (both parts sum to ~1); trim rows
        for b in range(n_bins):
            if ps[b] >= SIZE_MIN_ROW_P:
                sizes.append({"place": k + 1, "bin": b, "p": round(ps[b], 6)})
        # safety: if a place somehow sums above 1, scale its rows down
        ssum = sum(r["p"] for r in sizes if r["place"] == k + 1)
        if ssum > 1.0:
            for r in sizes:
                if r["place"] == k + 1:
                    r["p"] = round(r["p"] / ssum * 0.999, 6)

    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
