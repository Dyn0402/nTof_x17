#!/usr/bin/env python3
"""make_hs_jobs.py -- the high-statistics Magboltz grid (one job per mixture and field).

Fine grids around each gas's first-pass solution (§18), 3e8 collisions per point
(the first pass used 5e7 and its eta jittered +-15-20 % between neighbouring fields).
Fields: the measured ones plus enough in between to draw smooth curves.
Bench fields use the beam convention (700 V <-> 243 V/cm): 600 V 208, 700 V 243,
1000 V 347 V/cm.

    make_hs_jobs.py > hs_jobs.txt
"""

NCOLL = 30


def tag(g, w, a):
    f = lambda x: f'{x:g}'.replace('.', 'p')
    return f'{g}_w{f(w)}_a{f(a)}'


GRIDS = {
    'beam':  ([1.40, 1.50, 1.55, 1.60, 1.70], [0, 0.04, 0.06, 0.07, 0.08, 0.10, 0.13],
              [60, 75, 92, 108, 125, 142, 150, 175, 200, 243, 275]),
    'co2':   ([1.4, 1.6, 1.8], [0, 0.05, 0.10, 0.15, 0.20, 0.25], [150, 200, 243, 275]),
    'bench': ([0, 0.1, 0.2, 0.4, 0.7, 1.0], [0, 0.005, 0.01, 0.02, 0.04],
              [150, 175, 208, 243, 275, 300, 347, 400]),
}

if __name__ == '__main__':
    for g, (W, A, E) in GRIDS.items():
        for w in W:
            for a in A:
                for e in E:
                    print(tag(g, w, a), e, NCOLL)
