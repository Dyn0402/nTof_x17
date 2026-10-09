#!/usr/bin/env python3
"""zs_headon.py -- the head-on late-charge observable on the zero-suppressed beam runs.

Same observable as headon_stack.py (time = DREAM sample index, i.e. the beam
trigger; strip sums centred on the uRWELL prediction), for the ZS caches:
  run_63 flat700            (CF4, flat, both views head-on)
  run_63 rot d425/d325/d225 (CF4, 25.64 deg: only X is head-on)
  run_56 m70V 590V/625V     (CO2, flat, both views head-on)

In ZS data an absent sample is a real sample below threshold (4-5 sigma), not a
lost packet, so it counts as 0.  That censors small signals: a late-time decline
of a strip whose samples sit near threshold is exaggerated.  The censoring is
reported per sample as the fraction of events in which the centre strip is
present; a sum is only trusted where that fraction stays ~1.

    zs_headon.py
Output: results/zs_headon.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.dirname(HERE)
sys.path[:0] = [ANA, os.path.join(os.path.dirname(ANA), 'det4_sps_assessment')]
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM         # noqa: E402

ST = '/media/dylan/data/x17/sps_run53_det4_check/staging/'
ARMS = [  # name, file, plateau, views head-on, gas, drift V
    ('r63_flat700', 'run_63/wf_run63_flat.npz', 'flat700', 'xy', 'CF4', 700.4),
    ('r63_rot_d425', 'run_63/wf_run63_operating.npz', 'd425', 'x', 'CF4', 425.2),
    ('r63_rot_d325', 'run_63/wf_run63_operating.npz', 'd325', 'x', 'CF4', 325.1),
    ('r56_co2_625V', 'run_56_m70V/wf_m70V.npz', '625V', 'xy', 'CO2', None),
    ('r56_co2_590V', 'run_56_m70V/wf_m70V.npz', '590V', 'xy', 'CO2', None),
]
NSMP, SNS = 64, 60.0
HALF = 8
WIDTHS = (0, 1, 2, 4, 8)
SIDX = np.round(POSITION_MM / PITCH_MM).astype(int)


def run(f, lab, views, max_ev=20000):
    Z = np.load(ST + f, allow_pickle=True)
    ev, ch, s, a = Z['ev'], Z['ch'].astype(int), Z['samp'].astype(int), Z['amp']
    plat = Z['ev_plateau']
    pred = {'x': Z['ev_pX'], 'y': Z['ev_pY']}
    m = plat[ev] == lab
    ev, ch, s, a = ev[m], ch[m], s[m], a[m]
    events = np.unique(ev)
    events = events[np.isfinite(pred['x'][events]) & np.isfinite(pred['y'][events])][:max_ev]
    out = {'n': int(len(events))}
    for v in views:
        k = (VIEW[ch] == v) & np.isin(ev, events)
        ei = np.searchsorted(events, ev[k])
        o = SIDX[ch[k]] - np.round(pred[v][ev[k]] / PITCH_MM).astype(int) + HALF
        ok = (o >= 0) & (o <= 2 * HALF)
        C = np.zeros((len(events), 2 * HALF + 1, NSMP), np.float32)
        P = np.zeros((len(events), 2 * HALF + 1, NSMP), bool)
        C[ei[ok], o[ok], s[k][ok]] = a[k][ok]
        P[ei[ok], o[ok], s[k][ok]] = True
        M = C.mean(0)
        M -= M[:, :9].mean(1, keepdims=True)
        sums = {h: M[HALF - h:HALF + h + 1].sum(0) for h in WIDTHS}
        # centre strip = the offset with the largest mean signal (uRWELL-to-det4 offset may be nonzero)
        oc = int(np.argmax(M.sum(1)))
        sums_c = {h: M[max(oc - h, 0):oc + h + 1].sum(0) for h in WIDTHS}
        out[v] = dict(centre_offset=oc - HALF,
                      sum={str(h): sums_c[h].tolist() for h in WIDTHS},
                      present_centre=P[:, oc, :].mean(0).tolist(),
                      present_pm1=P[:, [oc - 1, oc + 1], :].mean((0, 1)).tolist())
    return out


def main():
    t = np.arange(NSMP) * SNS
    res = {'t': t.tolist()}
    for name, f, lab, views, gas, dv in ARMS:
        r = run(f, lab, views)
        r.update(gas=gas, drift_V=dv)
        res[name] = r
        for v in views:
            S = np.array(r[v]['sum']['4']); pc = np.array(r[v]['present_centre'])
            ip = int(np.argmax(S))
            print(f'{name} {v} n={r["n"]} centre_off {r[v]["centre_offset"]:+d} peak@{t[ip]:.0f}  '
                  f'±4 / peak every 240 ns: ' + ' '.join(f'{x:.2f}' for x in (S / S[ip])[::4]))
            print(f'{"":>14} centre present: ' + ' '.join(f'{x:.2f}' for x in pc[::4]))
    json.dump(res, open(os.path.join(HERE, 'results', 'zs_headon.json'), 'w'))


if __name__ == '__main__':
    main()
