#!/usr/bin/env python3
"""make_mb_ext.py > hs_jobs_gasfit.txt -- extend the high-statistics Magboltz grid (cloud_basics make_hs_jobs.py, 3e8
collisions per point) for the gas fit: the drift gap is profiled (27.0-30.0 mm), and a shorter gap needs a slower gas
(more water), so the beam grid's 1.4-1.7 % water is too narrow; the CO2 grid had only 150-275 V/cm and 1.4-1.8 % water,
while the CO2 rows run from 139 V/cm (run55 d400) and their water is unconstrained a priori.  Points already in
cloud_basics/results/air_hs are skipped.  Run with cloud_basics' condor_mb_hs.sub in lxplus ~/cloud_basics_condor/att.
"""
import os

NCOLL = 30
HERE = os.path.dirname(os.path.abspath(__file__))
HS = os.path.join(HERE, '..', '..', 'sps_beam_test_26', 'analysis', 'cloud_basics', 'results', 'air_hs')


def tag(g, w, a):
    f = lambda x: f'{x:g}'.replace('.', 'p')
    return f'{g}_w{f(w)}_a{f(a)}'


GRIDS = {
    'beam': ([1.2, 1.3, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9, 2.0], [0, 0.04, 0.06, 0.07, 0.08, 0.10, 0.13],
             [60, 75, 92, 108, 125, 142, 150, 175, 200, 243, 275]),
    'co2': ([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2], [0, 0.05, 0.10, 0.15, 0.20, 0.25],
            [100, 125, 150, 175, 200, 225, 243, 275]),
}

if __name__ == '__main__':
    have = set(os.listdir(HS))
    for g, (W, A, E) in GRIDS.items():
        for w in W:
            for a in A:
                for e in E:
                    if f'magboltz_{tag(g, w, a)}_E{e}_c{NCOLL}.json' not in have:
                        print(tag(g, w, a), e, NCOLL)
