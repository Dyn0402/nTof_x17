#!/usr/bin/env python3
"""faint_test.py -- why do the faintest run_71 events lose less late charge than the toy?

headon_split.py splits by each event's total X+-2 charge.  That total mixes (i) the
event's own cluster fluctuations, which make the attachment selection effect, and
(ii) det4's gain at the track's position (amplification stripes), a pure per-event
scale that dilutes it.  Here the total is divided by the mean total of events at the
same position (16 bins in each of x and y), so the split selects cluster fluctuations
only, and the result is compared with the toy WITHOUT a gain scatter.

Also reported per class: mean position-bin gain, fraction of samples present, and the
pre-trigger baseline level -- to see whether the faint class is a different population.

    faint_test.py <cache.npz>
Output: results/faint_test.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import headon_stack as H                                     # noqa: E402
from headon_split import ratio                               # noqa: E402
import split_toy as T                                        # noqa: E402


def main():
    ev, ch, s, amp, lead, plat, tw = H.load(sys.argv[1])
    pcode = np.full(len(plat), -1, np.int8)
    for i, (l, _) in enumerate(H.PLATEAUS):
        pcode[plat == l] = i
    rng = np.random.default_rng(23)
    res = {}
    for i, (lab, E) in enumerate(H.PLATEAUS):
        events = np.unique(ev[pcode[ev] == i])
        C = H.cube(ev, ch, s, amp, lead, events, 'x')
        base = np.nanmean(C[:, :, H.PRE], axis=2)
        B = C - base[:, :, None]
        q = np.nansum(B[:, H.KEEP - 2:H.KEEP + 3, 12:58], axis=(1, 2))
        # local gain: mean q over events in the same (x, y) position cell
        px, py = lead['x'][events], lead['y'][events]
        bx = np.searchsorted(np.quantile(px, np.linspace(0, 1, 17)[1:-1]), px)
        by = np.searchsorted(np.quantile(py, np.linspace(0, 1, 17)[1:-1]), py)
        cell = bx * 16 + by
        g = np.array([q[cell == c].mean() for c in range(256)])
        qn = q / g[cell]
        # missing-sample-proof charge: mean over the PRESENT samples of the +-2 sum (only samples
        # where all five strips are present), baseline from pre-trigger samples 0-3 only, so the
        # baseline used to subtract and the one used by the stack (0-8) share only half their noise
        b03 = np.nanmean(C[:, :, 0:4], axis=2)
        X5 = C[:, H.KEEP - 2:H.KEEP + 3, 12:58] - b03[:, H.KEEP - 2:H.KEEP + 3, None]
        allp = np.isfinite(X5).all(1)
        S5 = np.where(allp, np.nansum(X5, axis=1), np.nan)
        qm = np.nanmean(S5, axis=1)
        qm = np.where(np.isfinite(qm), qm, np.nanmedian(qm))
        out = {}
        for key, val in (('raw', q), ('gain_normalised', qn), ('mean_present', qm),
                         ('mean_present_gain_norm', qm / np.array([qm[cell == c].mean() for c in range(256)])[cell])):
            lo, hi = np.quantile(val, [0.05, 0.95]); good = (val > lo) & (val < hi)
            edges = np.quantile(val[good], [0, 1 / 3, 2 / 3, 1])
            rows = []
            for c in range(3):
                m = good & (val >= edges[c]) & (val <= edges[c + 1])
                r = ratio(C[m], 2)
                bs = [ratio(C[m][rng.integers(0, m.sum(), m.sum())], 2) for _ in range(40)]
                rows.append(dict(R=float(r), err=float(np.std(bs)), n=int(m.sum()),
                                 mean_cell_gain=float(g[cell[m]].mean()),
                                 present=float(np.isfinite(C[m][:, H.KEEP]).mean()),
                                 pre_level=float(np.nanmean(base[m][:, H.KEEP - 2:H.KEEP + 3].sum(1)))))
            out[key] = rows
            print(f'{lab} ({E} V/cm) split by {key:16s}: ' + ' | '.join(
                f'R {r["R"]:.3f}±{r["err"]:.3f} gain {r["mean_cell_gain"]:.0f} pres {r["present"]:.2f} '
                f'pre {r["pre_level"]:+.1f}' for r in rows))
        res[lab] = out
        del C, B
    # the toy without per-event gain scatter (243 V/cm geometry)
    terc, allR = T.run(1.9e-4, gsig=0.0)
    res['toy_no_gain_scatter'] = dict(terciles=terc, all=allR)
    terc2, allR2 = T.run(1.9e-4, gsig=0.35)
    res['toy_gain_scatter_0p35'] = dict(terciles=terc2, all=allR2)
    print('toy r=1.9e-4, no gain scatter : ' + ' / '.join(f'{x:.3f}' for x in terc) + f'  all {allR:.3f}')
    print('toy r=1.9e-4, gain scatter .35: ' + ' / '.join(f'{x:.3f}' for x in terc2) + f'  all {allR2:.3f}')
    json.dump(res, open(os.path.join(HERE, 'results', 'faint_test.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
