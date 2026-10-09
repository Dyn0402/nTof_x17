#!/usr/bin/env python3
"""att_compare.py -- can O2 attachment in the wet beam gas explain the run_71 decline?

Attachment at a constant rate removes drifting electrons as exp(-eta * v * t),
so the post-spike log-slope of the head-on wide (+-4) strip sums
(`results/beam_xy_pulse.json`) is eta*v.  Magboltz gives eta*v per O2 level
(`results/magboltz_beam_w1p7_o*.json`, condor 4410533); eta is linear in O2, so
each plateau's slope converts to the O2 level it would need.  All three plateaus
are one 30-min block of run_71 (05:22-05:52, one gas fill): one O2 level must
serve all three."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
PLATEAUS = (('sps700', 243), ('sps450', 150), ('sps275', 92))
WIN = (660.0, 1300.0)   # after the leading spike, before any ladder end


def mb(tag):
    return {p['E_Vcm']: p for p in json.load(open(os.path.join(R, f'magboltz_{tag}.json')))['points']}


def main():
    d = json.load(open(os.path.join(R, 'beam_xy_pulse.json')))
    t = np.array(d['t'])
    wet = mb('beam_w1p7')
    ref = mb('beam_w1p7_o0p1')
    print('Magboltz 1.7 % H2O, no O2: eta =', {E: wet[E]['eta_per_cm'] for _, E in PLATEAUS})
    out = {}
    for name, E in PLATEAUS:
        rate01 = ref[E]['eta_per_cm'] * ref[E]['v_true_um_ns'] * 1e-4   # /ns at 0.1 % O2
        m = (t >= WIN[0]) & (t <= WIN[1])
        for v in ('x4', 'y4'):
            y = np.array(d[name][v])
            r = -np.polyfit(t[m], np.log(y[m]), 1)[0]
            ppm = 1000.0 * r / rate01
            out[f'{E}_{v}'] = dict(rate_per_ns=r, rate_0p1pct_O2=rate01, o2_ppm=ppm)
            print(f'{E:4d} V/cm {v}: slope {r * 1e4:5.2f}e-4/ns   0.1 % O2 gives {rate01 * 1e4:5.2f}e-4/ns'
                  f'   -> {ppm:5.0f} ppm O2')
    json.dump(out, open(os.path.join(R, 'att_compare.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
