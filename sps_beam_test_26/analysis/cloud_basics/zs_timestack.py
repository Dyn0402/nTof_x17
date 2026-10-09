#!/usr/bin/env python3
"""zs_timestack.py -- the late-charge observable on every zero-suppressed beam arm, both views.

Arms (all det4 at H4, ZS, time = DREAM sample index = the beam trigger, sums
centred on the uRWELL prediction):
  r63_flat700   Ar/CF4/iso, flat, 243 V/cm, ZS 4 sigma   X +-2, Y +-8 (both head-on)
  r63_d425/325  Ar/CF4/iso, 25.64 deg, 142/108 V/cm      X +-2 (head-on), Y all +-12 (ladder)
  r56_625/590   Ar/CO2/iso, flat, resist 625/590 V, 5 sigma  X +-2, Y +-8

In a tilted view the sum over ALL strips at a sample is the same arriving current a
head-on view sees, so attachment must give the same decline in time in both.

ZS makes the absolute ratio selection-dependent: an absent sample counts 0, so
faint events are censored, and the brightest fifth holds discharges and pile-up.
Two selections are kept -- 'all' events and 'mid' (the middle 60 % by X+Y total
charge) -- and their difference is quoted as the systematic on R.

    zs_timestack.py
Output: results/zs_timestack.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zs_headon as Z                                        # noqa: E402
from det4_sps_map import VIEW, PITCH_MM                      # noqa: E402

ARMS = [  # name, file, plateau, Y half-width, label
    ('r63_flat700', 'run_63/wf_run63_flat.npz', 'flat700', 8, 'CF4 flat 243 V/cm'),
    ('r63_d425', 'run_63/wf_run63_operating.npz', 'd425', 12, 'CF4 25.64 deg 142 V/cm'),
    ('r63_d325', 'run_63/wf_run63_operating.npz', 'd325', 12, 'CF4 25.64 deg 108 V/cm'),
    ('r56_625V', 'run_56_m70V/wf_m70V.npz', '625V', 8, 'CO2 flat, resist 625 V'),
    ('r56_590V', 'run_56_m70V/wf_m70V.npz', '590V', 8, 'CO2 flat, resist 590 V'),
]
T = np.arange(64) * 60.0


def curve(M):
    S = M.mean(0); S = S - S[:9].mean()
    return S / S[(T >= 1080) & (T <= 1260)].mean()


def R(c):
    return float(c[(T >= 2400) & (T < 2700)].mean())


def main():
    rng = np.random.default_rng(13)
    res = {'t': T.tolist()}
    cache = {}
    for name, f, lab, hy, label in ARMS:
        if f not in cache:
            D = np.load(Z.ST + f, allow_pickle=True)
            cache = {f: (D['ev'], D['ch'].astype(int), D['samp'].astype(int), D['amp'],
                         D['ev_pX'], D['ev_pY'], D['ev_plateau'])}
        ev, ch, s, a, pX, pY, plat = cache[f]
        m = plat[ev] == lab
        e_, c_, s_, a_ = ev[m], ch[m], s[m], a[m]
        events = np.unique(e_)
        events = events[np.isfinite(pX[events]) & np.isfinite(pY[events])]
        k0 = np.isin(e_, events); e_, c_, s_, a_ = e_[k0], c_[k0], s_[k0], a_[k0]
        ei = np.searchsorted(events, e_)
        C = {}
        for v, p, h in (('x', pX, 2), ('y', pY, hy)):
            o = Z.SIDX[c_] - np.round(p[e_] / PITCH_MM).astype(int)
            k = (VIEW[c_] == v) & (np.abs(o) <= h)
            M = np.zeros((len(events), 64)); np.add.at(M, (ei[k], s_[k]), a_[k])
            C[v] = M
        tot = C['x'][:, 12:58].sum(1) + C['y'][:, 12:58].sum(1)
        q = np.quantile(tot[tot > 0], [0.2, 0.8])
        sel = {'all': np.ones(len(events), bool), 'mid': (tot > q[0]) & (tot < q[1])}
        r = {'label': label, 'n': int(len(events)), 'y_half': hy}
        for sk, sm in sel.items():
            for v in C:
                Mv = C[v][sm]
                c = curve(Mv)
                bs = [curve(Mv[rng.integers(0, len(Mv), len(Mv))]) for _ in range(60)]
                r[f'{v}_{sk}'] = c.tolist()
                r[f'{v}_{sk}_band'] = np.std(bs, axis=0).tolist()
                r[f'R_{v}_{sk}'] = [R(c), float(np.std([R(b) for b in bs]))]
        res[name] = r
        print(f'{name:12s} n={r["n"]:5d}  R_x all {r["R_x_all"][0]:.3f} mid {r["R_x_mid"][0]:.3f}   '
              f'R_y all {r["R_y_all"][0]:.3f} mid {r["R_y_mid"][0]:.3f}')
    json.dump(res, open(os.path.join(HERE, 'results', 'zs_timestack.json'), 'w'))


if __name__ == '__main__':
    main()
