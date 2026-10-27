#!/usr/bin/env python3
"""tilt_map_xval.py -- is the measured excess over the toy a local tilt that depends on BOTH coordinates?

The angle residual d = theta_fit - theta_M3 of each view is corrected by the median residual of its cell (signed, so
a tilt, not a spread) measured on the OTHER half of the events (even/odd event id), for
  1D: cells along the view's own coordinate (what ws_widthtilt's angle_map does), and
  2D: cells in (x, y) of the chamber,
at several cell sizes; s68 before and after.  A drift-field tilt that varies over the chamber (cathode dish, mounting)
is invisible to the waveform chi2 and to the charge; it shows up here as a 2D gain over 1D.

    tilt_map_xval.py   -> results/tilt_map_xval.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.expanduser('~/x17/cosmic_bench/model_audit')


def s68(a):
    a = np.asarray(a)
    a = a[np.abs(a) <= 5]
    return float(np.percentile(np.abs(a - np.median(a)), 68))


def xval(d, keys, half):
    """cross-validated cell-median correction: cells keyed by `keys` (tuple of int arrays)."""
    df = pd.DataFrame({'d': d, 'h': half, **{f'k{i}': k for i, k in enumerate(keys)}})
    kc = [f'k{i}' for i in range(len(keys))]
    corr = np.zeros(len(d))
    for h in (0, 1):
        tr = df[(df.h != h) & (df.d.abs() <= 5)]
        med = tr.groupby(kc).d.agg(['median', 'size'])
        med = med[med['size'] >= 8]['median']
        te = df[df.h == h]
        idx = pd.MultiIndex.from_frame(te[kc]) if len(kc) > 1 else pd.Index(te[kc[0]])
        corr[te.index] = med.reindex(idx).fillna(0.0).to_numpy()
    return d - corr


def main():
    out = {}
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        z = np.load(os.path.join(AUDIT, det, 'final_physmask', 'angles', 'per_event.npz'))
        b = json.load(open(os.path.join(AUDIT, det, 'calib_bundle', 'bundle.json')))
        ev = pd.read_parquet(os.path.join(AUDIT, det, 'final_physmask', 'events.parquet')).set_index('event_id')
        for v, o in (('x', 'y'), ('y', 'x')):
            eid = z[f'{v}_eid']
            tf = (z[f'{v}_w'] * 1e3 - b['w0'][v]) / (b['kw'][v] * float(b['v_drift']))
            d = np.degrees(np.arctan(tf) - np.arctan(z[f'{v}_tan_ref']))
            e = ev.reindex(eid)
            pv, po = e[f'{v}_p0'].to_numpy(), e[f'{o}_p0'].to_numpy()
            tref = z[f'{v}_tan_ref']
            ok = np.isfinite(d) & np.isfinite(pv) & np.isfinite(po)
            d, pv, po, half, tref = d[ok], pv[ok], po[ok], (eid[ok] % 2).astype(int), tref[ok]
            # the angle response (slope, from the bow/kw) is not a tilt: fit d against tan_ref first, globally
            d = d - np.polyval(np.polyfit(tref[np.abs(d) < 5], d[np.abs(d) < 5], 1), tref)
            res = dict(n=int(ok.sum()), s68=s68(d))
            for cs in (20, 40, 80):
                k1 = (pv // cs).astype(int)
                k2 = (po // cs).astype(int)
                res[f'1d_{cs}'] = s68(xval(d, (k1,), half))
                res[f'2d_{cs}'] = s68(xval(d, (k1, k2), half))
            out[f'{det}_{v}'] = res
            print(det, v, ' '.join(f'{k} {val:.2f}' if isinstance(val, float) else f'{k} {val}' for k, val in res.items()))
    json.dump(out, open(os.path.join(HERE, 'results', 'tilt_map_xval.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
