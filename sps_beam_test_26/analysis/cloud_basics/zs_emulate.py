#!/usr/bin/env python3
"""zs_emulate.py -- how much does zero suppression distort the late/early ratio?

Takes the clean RAW run_71 cube (headon_stack.py), measures the per-sample noise
sigma from the pre-trigger samples, and applies sample-level ZS at k sigma (a
sample below k*sigma is set to 0, as the FEU drops it).  R = level(2.4-2.7 us) /
level(1.08-1.26 us) for X+-1 (what zs_headon / xy_same_events used) and Y+-8,
RAW against ZS-emulated.  The shift is the ZS bias at run_71's gain; run_63
flat700 is the same gas, field and resist voltage 5 h earlier with real 4 sigma
ZS, so it should match the 4 sigma emulation if both runs lose charge alike.

    zs_emulate.py <cache.npz>
Output: results/zs_emulate.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import headon_stack as H                                     # noqa: E402

T = np.arange(H.NSMP) * H.SNS


def R(C, h):
    M = np.nanmean(C, axis=0)
    M = M - np.nanmean(M[:, H.PRE], axis=1, keepdims=True)
    S = np.nansum(M[H.KEEP - h:H.KEEP + h + 1], axis=0)
    return float(S[(T >= 2400) & (T < 2700)].mean() / S[(T >= 1080) & (T <= 1260)].mean())


def main():
    ev, ch, s, amp, lead, plat, tw = H.load(sys.argv[1])
    pcode = np.full(len(plat), -1, np.int8)
    for i, (l, _) in enumerate(H.PLATEAUS):
        pcode[plat == l] = i
    res = {}
    for i, (lab, E) in enumerate(H.PLATEAUS):
        events = np.unique(ev[pcode[ev] == i])[:15000]
        res[lab] = {}
        for v, h in (('x', 1), ('y', 8)):
            C = H.cube(ev, ch, s, amp, lead, events, v)
            C -= np.nanmean(C[:, :, H.PRE], axis=2, keepdims=True)      # per-event, per-strip baseline
            sig = float(np.nanstd(C[:, :, H.PRE]))
            row = {'sigma_adc': sig, 'raw': R(C, h)}
            for k in (4, 5):
                Z = np.where(np.isnan(C), np.nan, np.where(C > k * sig, C, 0.0))
                row[f'zs{k}'] = R(Z, h)
            res[lab][v] = row
            print(f'{lab} ({E} V/cm) {v}±{h}: noise {sig:.1f} ADC   R raw {row["raw"]:.3f}   '
                  f'ZS4 {row["zs4"]:.3f}   ZS5 {row["zs5"]:.3f}')
            del C
    json.dump(res, open(os.path.join(HERE, 'results', 'zs_emulate.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
