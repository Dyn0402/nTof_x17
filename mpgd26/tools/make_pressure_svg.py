#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pressure_svg.py -- the ³He capsule pressure record, as MARKUP.

    ../../.venv/bin/python tools/make_pressure_svg.py            # print the SVG
    ../../.venv/bin/python tools/make_pressure_svg.py --numbers  # just the stats

Prints an inline <svg> block for the deck's setup slide 15.  Paste the output
between the two marker comments in slides/index.html:

    <!-- BEGIN he3-pressure (tools/make_pressure_svg.py) -->
    ...
    <!-- END he3-pressure -->

Why markup and not a PNG (Dylan, 2026-08-18: "can we pull the 3He capsule
pressure from the run and put it on this slide formatted nicely as an html plot
rather than just python plot?"): the same reason the efficiency slide's loss
budget is markup.  A small matplotlib panel dropped beside bullet text arrives
in matplotlib's font, at whatever size the PNG was saved at, and reads smaller
and greyer than the body text next to it.  As SVG it inherits the deck's own
type scale and colours, stays sharp at any projector resolution, and costs
~6 kB instead of ~90 kB.

DATA.  ntof_run_report/data/he3_pressure_5min.csv -- the five-minute reduction
(median/min/max/count per bin) of the campaign pressure log.  A Keithley 2000
on the capsule's transducer over GPIB, one sample every ~2 s, converted with
the transducer calibration P = (V - 1) x 400 bar; the raw 1.08 M samples are on
EOS under /eos/experiment/ntof/data/x17/july_beam/slow_control/he3_pressure/.
See ntof_run_report/figures_local.py:capsule_pressure for the full provenance
and for the three things to know about the record; the two that shape this
figure are that it starts at the 14 July MOUNT (an 8 July bench stub at 507.1
bar is in the CSV and is not part of the campaign trace) and that the
end-of-run VENT to 7.8 bar is annotated rather than plotted, because drawing it
would compress the whole run into one flat line.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(os.path.join(
    HERE, '..', '..', 'ntof_run_report', 'data', 'he3_pressure_5min.csv'))

T0 = dt.datetime(2026, 7, 14)          # the capsule goes on the beam axis
T1 = dt.datetime(2026, 8, 10, 12)      # dismount
PRODUCTION = dt.datetime(2026, 7, 26)  # the start of the production period
VENTED_TO = 7.8                        # bar, after the valve was opened

# canvas.  The viewBox is the only geometry: the slide scales it.
W, H = 640.0, 196.0
L, R, TOP, BOT = 52.0, 8.0, 16.0, 46.0
PLOT_W, PLOT_H = W - L - R, H - TOP - BOT
BIN_MINUTES = 60                       # ~1.1 drawn points per pixel of width
# a break is a MISSING BIN, so the threshold has to be read against the bin
# width and not against the raw 2 s cadence: at 1800 s every hourly point looked
# like a gap and the trace came out as 640 one-point polylines
GAP_SECONDS = 2.5 * BIN_MINUTES * 60   # logger down longer than this -> break


def load():
    rows = list(csv.DictReader(open(CSV)))
    out = []
    for r in rows:
        t = dt.datetime.fromisoformat(r['timestamp'])
        p = float(r['p_med_bar'])
        if t >= T0 and p > 400.0:       # drops the bench stub and the vent
            out.append((t, p, int(r['n'])))
    return out


def rebin(series, minutes=BIN_MINUTES):
    """Median within each ``minutes`` slot -- the five-minute file is 12x more
    points than the plot has pixels, and every one of them is markup."""
    buckets = {}
    for t, p, _n in series:
        k = int((t - T0).total_seconds() // (minutes * 60))
        buckets.setdefault(k, []).append(p)
    return [(T0 + dt.timedelta(minutes=minutes * k + minutes / 2),
             statistics.median(v)) for k, v in sorted(buckets.items())]


def svg(series, lo, hi):
    def X(t):
        return L + PLOT_W * (t - T0).total_seconds() / (T1 - T0).total_seconds()

    def Y(p):
        return TOP + PLOT_H * (hi - p) / (hi - lo)

    pts = rebin(series)
    # one <polyline> per continuous stretch, so a gap in the log is a gap on
    # the page rather than a straight line across it
    runs, cur = [], [pts[0]]
    for prev, nxt in zip(pts, pts[1:]):
        if (nxt[0] - prev[0]).total_seconds() > GAP_SECONDS:
            runs.append(cur); cur = []
        cur.append(nxt)
    runs.append(cur)

    o = []
    A = o.append
    A(f'<svg class="he3-plot" viewBox="0 0 {W:.0f} {H:.0f}" '
      f'role="img" aria-label="The helium-3 capsule pressure gauge for the '
      f'whole time the capsule was mounted: 504.8 bar when it went on the beam '
      f'axis on 14 July, falling steadily to 494.7 bar by the dismount on '
      f'10 August, with a day-night breathing cycle of about half a bar on '
      f'top of the trend.">')

    # --- y grid and labels ---
    for p in range(int(lo) + 1, int(hi) + 1):
        if p % 5:
            continue
        y = Y(p)
        A(f'<line class="g" x1="{L:.1f}" x2="{W - R:.1f}" '
          f'y1="{y:.1f}" y2="{y:.1f}"/>')
        A(f'<text class="ax yl" x="{L - 8:.1f}" y="{y + 4:.1f}">{p}</text>')
    A(f'<text class="ax unit" x="{L - 8:.1f}" y="{TOP - 4:.1f}">bar</text>')

    # --- x ticks, one a week ---
    t = T0
    while t <= T1:
        x = X(t)
        A(f'<line class="t" x1="{x:.1f}" x2="{x:.1f}" '
          f'y1="{TOP + PLOT_H:.1f}" y2="{TOP + PLOT_H + 4:.1f}"/>')
        A(f'<text class="ax xl" x="{x:.1f}" y="{TOP + PLOT_H + 17:.1f}">'
          f'{t.strftime("%-d %b")}</text>')
        t += dt.timedelta(days=7)

    # --- the production line ---
    xp = X(PRODUCTION)
    A(f'<line class="mark" x1="{xp:.1f}" x2="{xp:.1f}" '
      f'y1="{TOP:.1f}" y2="{TOP + PLOT_H:.1f}"/>')
    A(f'<text class="ax note" x="{xp + 4:.1f}" y="{TOP + 11:.1f}">'
      f'production starts</text>')

    # --- the trace ---
    for run in runs:
        d = ' '.join(f'{X(t):.1f},{Y(p):.1f}' for t, p in run)
        A(f'<polyline class="trace" points="{d}"/>')

    # --- the two ends, named ---
    t_a, p_a = pts[0]
    t_b, p_b = pts[-1]
    A(f'<circle class="dot" cx="{X(t_a):.1f}" cy="{Y(p_a):.1f}" r="3"/>')
    A(f'<text class="ax end" x="{X(t_a) + 7:.1f}" y="{Y(p_a) - 6:.1f}">'
      f'{p_a:.1f} bar at mount</text>')
    A(f'<circle class="dot" cx="{X(t_b):.1f}" cy="{Y(p_b):.1f}" r="3"/>')
    A(f'<text class="ax end r" x="{X(t_b) - 7:.1f}" y="{Y(p_b) + 15:.1f}">'
      f'{p_b:.1f} bar at dismount</text>')
    A(f'<line class="axis" x1="{L:.1f}" x2="{W - R:.1f}" '
      f'y1="{TOP + PLOT_H:.1f}" y2="{TOP + PLOT_H:.1f}"/>')
    A('</svg>')
    return '\n'.join(o)


def stats(series):
    p = [x[1] for x in series]
    days = (series[-1][0] - series[0][0]).total_seconds() / 86400.0
    return dict(n_bins=len(series), n_samples=sum(x[2] for x in series),
                days=days, first=p[0], last=p[-1], lo=min(p), hi=max(p),
                mean=sum(p) / len(p), drop=p[0] - p[-1],
                per_day=(p[0] - p[-1]) / days, vented_to=VENTED_TO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--numbers', action='store_true')
    args = ap.parse_args()
    series = load()
    s = stats(series)
    if args.numbers:
        for k, v in s.items():
            print(f'{k:12s} {v}')
        return
    print(svg(series, lo=492.0, hi=506.0))


if __name__ == '__main__':
    main()
