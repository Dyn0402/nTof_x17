#!/usr/bin/env python3
"""gain_vs_loss.py -- attachment or resistive-layer charging?  Use det4's own gain map.

Both remove late charge in X and Y alike without an undershoot.  They differ in
what they scale with: attachment acts on drifting electrons and does not know
the gain; charging of the resistive layer lowers the field for later avalanches
in proportion to the avalanche charge already deposited, i.e. to the gain.

det4's gain varies strongly across the chamber (amplification stripes).  Events
are binned by the position of the track (det4's lead strip in each view); the
mean X+-2 charge of a bin, averaged over its events, is that place's gain --
independent of any one event's cluster pattern, so the selection cannot bias the
late/early ratio the way a per-event charge split does.  R per bin against the
bin's gain: flat = attachment, falling with gain = charging.

    gain_vs_loss.py <cache.npz>
Output: results/gain_vs_loss.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import headon_stack as H                                     # noqa: E402
from headon_split import ratio                               # noqa: E402

NB = 8


def main():
    ev, ch, s, amp, lead, plat, tw = H.load(sys.argv[1])
    rng = np.random.default_rng(11)
    pcode = np.full(len(plat), -1, np.int8)
    for i, (l, _) in enumerate(H.PLATEAUS):
        pcode[plat == l] = i
    res = {}
    for i, (lab, E) in enumerate(H.PLATEAUS):
        events = np.unique(ev[pcode[ev] == i])
        Cx = H.cube(ev, ch, s, amp, lead, events, 'x')
        Cy = H.cube(ev, ch, s, amp, lead, events, 'y')
        res[lab] = {}
        for key in ('x', 'y'):
            pos = lead[key][events]
            edges = np.quantile(pos, np.linspace(0, 1, NB + 1))
            rows = []
            for b in range(NB):
                m = (pos >= edges[b]) & (pos <= edges[b + 1])
                M = np.nanmean(Cx[m], axis=0)
                M = M - np.nanmean(M[:, H.PRE], axis=1, keepdims=True)
                gain = float(np.nansum(M[H.KEEP - 2:H.KEEP + 3, 12:58]))   # mean total, ADC
                rx = ratio(Cx[m], 2); ry = ratio(Cy[m], 8)
                bx = [ratio(Cx[m][rng.integers(0, m.sum(), m.sum())], 2) for _ in range(40)]
                by = [ratio(Cy[m][rng.integers(0, m.sum(), m.sum())], 8) for _ in range(40)]
                rows.append(dict(pos=[float(edges[b]), float(edges[b + 1])], n=int(m.sum()), gain=gain,
                                 Rx=[float(rx), float(np.std(bx))], Ry=[float(ry), float(np.std(by))]))
            g = np.array([r['gain'] for r in rows]); rx = np.array([r['Rx'][0] for r in rows])
            ex = np.array([r['Rx'][1] for r in rows])
            # weighted slope of R_x against gain / mean gain
            w = 1 / ex ** 2; x = g / g.mean()
            A = np.vstack([np.ones_like(x), x]).T
            cov = np.linalg.inv(A.T @ (A * w[:, None]))
            beta = cov @ (A.T @ (w * rx))
            res[lab][f'by_{key}'] = dict(rows=rows, slope_Rx_per_relgain=[float(beta[1]), float(np.sqrt(cov[1, 1]))])
            print(f'{lab} ({E} V/cm) binned by {key}: gain range {g.min():.0f}-{g.max():.0f} '
                  f'(x{g.max() / g.min():.1f});  dR_x / d(gain/mean) = {beta[1]:+.3f} ± {np.sqrt(cov[1, 1]):.3f}')
            for r in rows:
                print(f'    pos {r["pos"][0]:6.1f}-{r["pos"][1]:6.1f}  n {r["n"]:5d}  gain {r["gain"]:6.0f}  '
                      f'Rx {r["Rx"][0]:.3f}±{r["Rx"][1]:.3f}  Ry {r["Ry"][0]:.3f}±{r["Ry"][1]:.3f}')
        del Cx, Cy
    json.dump(res, open(os.path.join(HERE, 'results', 'gain_vs_loss.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
