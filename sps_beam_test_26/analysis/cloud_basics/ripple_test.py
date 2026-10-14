#!/usr/bin/env python3
"""ripple_test.py -- is the ~0.7 us ripple a trigger-locked pickup?

In the rotated run_63 ZS stacks a ripple of ~0.7 us rides on both views, anti-phased
between X and Y, at the same sample times at 142 and 108 V/cm (so locked to the
trigger, not to the ladder).  RAW run_71 lets us look at strips with no signal:
offsets |o| = 9..12 from the track, averaged over events at each sample (missing
samples excluded).  A trigger-locked pickup shows as a coherent pattern there; its
spectrum gives the period.  Also: the same pattern in the signal-free PRE-trigger
part cannot be removed by any per-event baseline.

    ripple_test.py <cache.npz>
Output: results/ripple_test.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import headon_stack as H                                     # noqa: E402

FAR = [o for o in range(2 * H.KEEP + 1) if abs(o - H.KEEP) >= 9]


def main():
    ev, ch, s, amp, lead, plat, tw = H.load(sys.argv[1])
    pcode = np.full(len(plat), -1, np.int8)
    for i, (l, _) in enumerate(H.PLATEAUS):
        pcode[plat == l] = i
    t = np.arange(H.NSMP) * H.SNS
    res = {'t': t.tolist()}
    for i, (lab, E) in enumerate(H.PLATEAUS):
        events = np.unique(ev[pcode[ev] == i])[:15000]
        res[lab] = {}
        for v in ('x', 'y'):
            C = H.cube(ev, ch, s, amp, lead, events, v)
            C -= np.nanmean(C[:, :, H.PRE], axis=2, keepdims=True)
            far = np.nanmean(C[:, FAR, :], axis=(0, 1))                  # ADC per strip
            # the same, split into two random halves of events: is it reproducible?
            h = np.random.default_rng(1).permutation(len(C)) < len(C) // 2
            f1 = np.nanmean(C[h][:, FAR, :], axis=(0, 1)); f2 = np.nanmean(C[~h][:, FAR, :], axis=(0, 1))
            d = far - np.convolve(far, np.ones(9) / 9, mode='same')
            spec = np.abs(np.fft.rfft(d[9:-9] * np.hanning(len(d) - 18)))
            fr = np.fft.rfftfreq(len(d) - 18, H.SNS)                      # 1/ns
            k = int(np.argmax(spec[2:])) + 2
            corr = float(np.corrcoef(f1[9:-9], f2[9:-9])[0, 1])
            res[lab][v] = dict(far=far.tolist(), half1=f1.tolist(), half2=f2.tolist(),
                               period_ns=float(1 / fr[k]), half_corr=corr, rms=float(np.std(d[9:-9])))
            print(f'{lab} ({E} V/cm) {v}: far-strip mean, every 2nd sample [ADC]: ' +
                  ' '.join(f'{x:+.1f}' for x in far[8::2]) +
                  f'\n      dominant period {1 / fr[k]:.0f} ns, rms {np.std(d[9:-9]):.2f} ADC, '
                  f'half-vs-half corr {corr:.2f}')
            del C
    json.dump(res, open(os.path.join(HERE, 'results', 'ripple_test.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
