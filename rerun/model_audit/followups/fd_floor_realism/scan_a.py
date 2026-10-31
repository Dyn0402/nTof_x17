#!/usr/bin/env python3
"""scan_a.py -- FD (model-audit follow-up 10a): why the perfect-model resolution with 60 ns depth bins jumps above
~25 deg.  The ws_floor toy (exact mean model, start at the truth) run over the five bench chambers (drift velocity
24-39 um/ns), four depth-bin widths (20/30/60/90 ns), with and without the clustering of the ionisation, plus two
discriminating variants on det3: a different strip pitch and a different prompt footprint.

The hypothesis (FOLLOWUPS 10): one depth bin spans v*dt*tan(theta) across the strips; the fit model spreads each bin's
charge uniformly over that segment, the real charge is lumpy (a few clusters per bin), so once the segment is longer
than ~ a strip the per-bin charge is mis-modelled.  If so, the excess over the floor is a function of
x = v*dt*tan(theta)/pitch alone, with its onset near x ~ 1, it shrinks without clustering, and it moves with the pitch.

    scan_a.py job --det det3 --dtbin 60 --config full [--pitch 0.78] [--s0 x] [--n 100] --out DIR
    scan_a.py collect --inp DIR                                     -> results/scan_a.json
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, '..', '..', 'ws_floor'), HERE):
    sys.path.insert(0, p)
import floor_toy as ft  # noqa: E402

if os.environ.get('FD_AUDIT'):
    ft.AUDIT = os.environ['FD_AUDIT']
import pass1_basis as pb  # noqa: E402  (FD_BASIS=pass: pass-1 bundles and inputs; off = 10-27)
pb.apply(None, ft)
ANGLES = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)


def summ(rows):
    rows = np.array(rows)
    out = {}
    for ang in ANGLES:
        r = rows[rows[:, 0] == ang]
        if len(r):
            out[str(ang)] = dict(n=len(r), s68=ft.s68(r[:, 1]), bias=float(np.median(r[:, 1])),
                                 chi2dof=float(np.median(r[:, 8])), oracle_s68=ft.s68(r[:, 4]))
    return out


def job(a):
    ft.DT_BIN = float(a.dtbin)
    ft.NSUB = max(int(round(a.dtbin / 10.0)), 2)
    ft.PITCH = float(a.pitch)
    if a.s0 > 0:
        for d in ft.CHAMBERS.values():
            d['s0'] = float(a.s0)
    tag = f'{a.det}_{a.config}_bin{int(a.dtbin)}_p{int(round(a.pitch * 100))}' + (f'_s{int(round(a.s0 * 100))}' if a.s0 > 0 else '')
    rows = []
    for k, ang in enumerate(ANGLES):
        rows += ft.one((a.det, a.config, ang, 9000 + 100 * k + int(a.dtbin), a.n, 'noise'))
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, tag + '.npy'), np.array(rows))
    json.dump(dict(det=a.det, config=a.config, dt_bin=a.dtbin, pitch=a.pitch, s0=a.s0 if a.s0 > 0 else
                   ft.CHAMBERS[a.det]['s0'], v=ft.CHAMBERS[a.det]['v'], by_angle=summ(rows)),
              open(os.path.join(a.out, tag + '.json'), 'w'), indent=1)
    print(tag, 'done')


def collect(a):
    out = {}
    for f in sorted(glob.glob(os.path.join(a.inp, '*.json'))):
        d = json.load(open(f))
        rows = np.load(f.replace('.json', '.npy'))
        tag = os.path.basename(f)[:-5]
        ba = {}
        rs = np.random.default_rng(3)
        for ang in ANGLES:
            r = rows[rows[:, 0] == ang][:, 1]
            if not len(r):
                continue
            boots = [ft.s68(rs.choice(r, len(r))) for _ in range(200)]
            x = d['v'] * d['dt_bin'] * np.tan(np.radians(ang)) / d['pitch']
            ba[str(ang)] = dict(s68=ft.s68(r), s68_err=float(np.std(boots)), bias=float(np.median(r)), x=float(x),
                                chi2dof=d['by_angle'][str(ang)]['chi2dof'])
        out[tag] = dict(det=d['det'], config=d['config'], dt_bin=d['dt_bin'], pitch=d['pitch'], s0=d['s0'],
                        v=d['v'], by_angle=ba)
    json.dump(out, open(os.path.join(HERE, 'results', f"scan_a{'_' + pb.TAG if pb.ON else ''}.json"), 'w'), indent=1)
    print(len(out), 'runs collected')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('job', 'collect'))
    ap.add_argument('--det', default='det3')
    ap.add_argument('--dtbin', type=float, default=60.0)
    ap.add_argument('--config', default='full')
    ap.add_argument('--pitch', type=float, default=0.78)
    ap.add_argument('--s0', type=float, default=0.0)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'scan_a'))
    ap.add_argument('--inp', default=os.path.join(HERE, 'results', 'scan_a'))
    a = ap.parse_args()
    {'job': job, 'collect': collect}[a.cmd](a)


if __name__ == '__main__':
    main()
