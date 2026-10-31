#!/usr/bin/env python3
"""make_fine_jobs.py [--have DIR] {bench,mid} > jobs.txt -- the fine Magboltz grids of MB_FINE_2026-10-31.md
(same recipe as make_hs_jobs.py: one (mixture, field) per job, 3e8 collisions).

  bench  the bench gas Ar/iso 95/5 on a COMPLETE (water, air, field) cube -> results/air_fine.  Water steps of
         0.05 % through 0.4-2.0 % (where the bench sits and where det6's slow 6-26 night sits: v = 17 um/ns at
         243 V/cm needs ~1.3 % water), fields dense below 150 V/cm (v(E, water) is most curved there) and every
         measured drift field in both conventions (V / 2.88 cm and V / 3.0 cm).  Air 0 / 0.01 / 0.04 %: the
         drop-one test moves the fitted O2 by < 3 ppm at that spacing, so finer air steps buy nothing.
  mid    validation points at mixture midpoints of the existing SPS grids (results/air_hs), 3 fields each ->
         results/air_mid; never part of a fit grid (they would leave holes in the cube).

Points already present (same file name) under --have are skipped; they are copied into air_fine instead.
"""
import argparse
import os

NCOLL = 30


def tag(g, w, a):
    f = lambda x: f'{x:g}'.replace('.', 'p')
    return f'{g}_w{f(w)}_a{f(a)}'


def frange(a, b, s):
    n = int(round((b - a) / s))
    return [round(a + i * s, 4) for i in range(n + 1)]


BENCH_W = [0, 0.1, 0.2, 0.3] + frange(0.4, 2.0, 0.05) + [2.25, 2.5, 3.0]
BENCH_A = [0, 0.01, 0.04]
BENCH_E = [33, 35, 45, 60, 80, 100, 104, 125, 150, 167, 174, 208, 233, 243, 275, 300, 312, 333, 347, 367, 382, 400]

MID = {   # (water, air) midpoints of the SPS grids, at the fields the fits use most
    'beam': ([(1.45, 0.05), (1.525, 0.075), (1.65, 0.09), (1.75, 0.115)], [92, 150, 243]),
    'co2':  ([(1.3, 0.025), (1.5, 0.075), (1.7, 0.125), (1.9, 0.175)], [139, 200, 243]),
}


def jobs(which):
    if which == 'bench':
        for w in BENCH_W:
            for a in BENCH_A:
                for e in BENCH_E:
                    yield tag('bench', w, a), e
    else:
        for g, (pts, E) in MID.items():
            for w, a in pts:
                for e in E:
                    yield tag(g, w, a), e


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('which', choices=('bench', 'mid')); ap.add_argument('--have', default=None)
    a = ap.parse_args()
    have = set(os.listdir(a.have)) if a.have else set()
    for t, e in jobs(a.which):
        if f'magboltz_{t}_E{e}_c{NCOLL}.json' not in have:
            print(t, e, NCOLL)
