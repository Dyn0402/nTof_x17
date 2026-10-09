#!/usr/bin/env python3
"""spike_test.py -- is the run_71 leading spike in the charge, or in the stacking?

§11 stacks align each event on the time its X +-4 sum crosses THR = 60 ADC.  At
low drift field the drift current per ns is small (it scales with v), so a
fixed threshold is crossed preferentially when a large ionisation cluster
arrives -- the alignment then places that cluster at t = 0 in every event, and
the stack grows a leading 'spike' whose relative size rises as v falls.  The
same mechanism makes the stack fall from its peak, i.e. the field-ordered
early 'drop' of §11/§13.

Two stacks per plateau, both from the RAW samples (no zero suppression), both
views' 9-strip sums centred on the uRWELL prediction (not on the pulse):
  trig : time = sample index (the DREAM window is opened by the beam trigger;
         the trigger is asynchronous to the 60 ns clock, so this smears by one
         sample and cannot make a spike)
  thrN : time from the X sum's crossing of N ADC (the §11 method), N = 30/60/120
An alignment artefact vanishes in 'trig' and grows with N.  A physical spike
(e.g. primary ionisation inside the amplification gap) is in both, and does
not depend on N.

    spike_test.py [--max-events 20000]
Output: results/spike_test.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.dirname(HERE)
sys.path[:0] = [ANA, os.path.join(os.path.dirname(ANA), 'det4_sps_assessment')]
import datasets                                              # noqa: E402
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM         # noqa: E402

PLATEAUS = (('raw700', 243), ('raw450', 150), ('raw275', 92))
HALF = 4
THRS = (30.0, 60.0, 120.0)
GRID = np.arange(-300.0, 3600.0, 60.0)


def view_sum(c, s, a, view, p0, nsmp):
    """9-strip sum centred on the strip nearest the prediction."""
    k = VIEW[c] == view
    if not k.any():
        return None
    d = POSITION_MM[c[k]] - p0
    m = np.abs(d) <= (HALF + 0.5) * PITCH_MM
    out = np.zeros(nsmp)
    ss = s[k][m].astype(int)
    ok = (ss >= 0) & (ss < nsmp)
    np.add.at(out, ss[ok], a[k][m][ok])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-events', type=int, default=20000)
    a = ap.parse_args()
    D = datasets.get('run71_raw')
    sns, nsmp = float(D['sample_ns']), int(D['n_samples'])
    Z = np.load(D['stage'] + 'wf_run71_raw_det4only.npz', allow_pickle=True)
    ev, ch, samp, amp = Z['ev'], Z['ch'].astype(int), Z['samp'], Z['amp']
    plat, pX, pY = Z['ev_plateau'], Z['ev_pX'], Z['ev_pY']
    order = np.argsort(ev, kind='stable')
    ev, ch, samp, amp = ev[order], ch[order], samp[order], amp[order]
    starts = np.r_[0, np.flatnonzero(ev[1:] != ev[:-1]) + 1, len(ev)]

    sums = {lab: [] for lab, _ in PLATEAUS}
    for s, e in zip(starts[:-1], starts[1:]):
        eid = int(ev[s])
        lab = str(plat[eid])
        if lab not in sums or len(sums[lab]) >= a.max_events:
            continue
        if not (np.isfinite(pX[eid]) and np.isfinite(pY[eid])):
            continue
        x = view_sum(ch[s:e], samp[s:e], amp[s:e], 'x', pX[eid], nsmp)
        y = view_sum(ch[s:e], samp[s:e], amp[s:e], 'y', pY[eid], nsmp)
        if x is None or y is None:
            continue
        sums[lab].append((x, y))

    t_s = np.arange(nsmp) * sns
    res = {'grid': GRID.tolist(), 't_trig': t_s.tolist()}
    for lab, E in PLATEAUS:
        X = np.array([p[0] for p in sums[lab]]); Y = np.array([p[1] for p in sums[lab]])
        r = dict(E_Vcm=E, n=len(X), trig_x=X.mean(0).tolist(), trig_y=Y.mean(0).tolist())
        # how many events have any signal at all (the trig stack keeps the rest as zeros)
        r['frac_signal'] = float((X.max(1) > 60).mean())
        for thr in THRS:
            acc = np.zeros(len(GRID)); cnt = np.zeros(len(GRID)); n = 0
            for x, y in zip(X, Y):
                k = next((k for k in range(1, nsmp) if x[k - 1] < thr <= x[k]), None)
                if k is None or k < 3:
                    continue
                t = (np.arange(nsmp) - (k - 1 + (thr - x[k - 1]) / (x[k] - x[k - 1]))) * sns
                w = x + y
                yy = np.interp(GRID, t, w, left=np.nan, right=np.nan)
                ok = np.isfinite(yy)
                acc[ok] += yy[ok]; cnt[ok] += 1; n += 1
            r[f'thr{int(thr)}'] = (acc / np.maximum(cnt, 1)).tolist()
            r[f'thr{int(thr)}_n'] = n
        res[lab] = r

    # summaries: spike = peak / level 360 ns after the peak; late = level at +900 ns / +360 ns
    print('stack        E   n      peak/+360ns   (+900)/(+360)')
    for lab, E in PLATEAUS:
        r = res[lab]
        for key, tt in [('trig', t_s)] + [(f'thr{int(t)}', GRID) for t in THRS]:
            w = (np.array(r['trig_x']) + np.array(r['trig_y'])) if key == 'trig' else np.array(r[key])
            ip = int(np.nanargmax(w)); tp = tt[ip]
            l360 = np.interp(tp + 360, tt, w); l900 = np.interp(tp + 900, tt, w)
            r[f'{key}_spike'] = float(w[ip] / l360)
            r[f'{key}_late'] = float(l900 / l360)
            n = r['n'] if key == 'trig' else r[f'{key}_n']
            print(f'{key:7s} {E:6d} {n:6d}   {w[ip] / l360:8.3f}   {l900 / l360:8.3f}')
    json.dump(res, open(os.path.join(HERE, 'results', 'spike_test.json'), 'w'))


if __name__ == '__main__':
    main()
