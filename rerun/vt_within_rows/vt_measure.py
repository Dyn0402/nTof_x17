#!/usr/bin/env python3
"""vt_measure.py [data root] -- the M3-ladder drift velocity per time block within every det6/det7 6-26 row.

Per row and view: the row's own kw (set_w0kw's estimator, on the row's own reconstruction) and the same estimator on
time blocks of ~N_PER events in the kw window.  The ladder v of a block is v_drift x kw (what the reconstruction's angle
mapping effectively uses), averaged over the two views when both exist.  Also on the even / odd event_id halves, for
the held-out test in vt_apply.py.  Writes results/blocks.json.
"""
import json
import os
import sys

import numpy as np

from vt_common import RES, rows, load_row, ladder, kw_window, blocks_equal_count, data_root

N_PER = 300            # events in the kw window per block and view: ~1 % on kw (det6 per-event spread ~15 %)


def measure(R, half=None):
    """blocks of one row; half = None (all), 0 (even event_id), 1 (odd)."""
    cal = R['cal']
    V = {}
    for p in ('x', 'y'):
        v = R['views'][p]
        k = np.ones(len(v['eid']), bool) if half is None else (v['eid'] % 2 == half)
        V[p] = {q: np.asarray(a)[k] for q, a in v.items()}
        V[p]['tp'] = (V[p]['tan_ref'] + V[p]['a']) * V[p]['S']
    # block edges on the clock: the view with more events in the kw window, ~N_PER of them per block
    pb = max(('x', 'y'), key=lambda p: kw_window(V[p]['tp']).sum())
    edges = blocks_equal_count(V[pb]['t_h'], kw_window(V[pb]['tp']), N_PER)
    out = dict(row={}, blocks=[])
    for p in ('x', 'y'):
        w0, kw, n, e = ladder(cal, V[p]['w'], V[p]['tp'])
        out['row'][p] = dict(w0=w0, kw=kw, n=n, err=e, v_lad=cal.v_drift * kw, v_lad_err=cal.v_drift * e)
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = dict(lo=float(lo), hi=float(hi))
        tt = []
        for p in ('x', 'y'):
            k = (V[p]['t_h'] >= lo) & (V[p]['t_h'] < hi)
            w0, kw, n, e = ladder(cal, V[p]['w'][k], V[p]['tp'][k])
            b[p] = dict(w0=w0, kw=kw, n=n, err=e, v_lad=cal.v_drift * kw, v_lad_err=cal.v_drift * e)
            tt.append(V[p]['t_h'][k])
        tt = np.concatenate(tt)
        b['t'] = float(np.median(tt)) if len(tt) else np.nan
        b['t_lo'], b['t_hi'] = (float(tt.min()), float(tt.max())) if len(tt) else (np.nan, np.nan)
        out['blocks'].append(b)
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else data_root()
    res = []
    for run, sub, det, wdir in rows(root):
        R = load_row(wdir, root)
        df = R['df']
        r = dict(run=run, subrun=sub, det=det, key=R['key'], resist_V=R['resist_V'], v_drift=R['cal'].v_drift,
                 kw_applied=R['cal'].kw, w0_applied=R['cal'].w0, t_start=float(df.t_h.min()),
                 t_end=float(df.t_h.max()), n_events=int(len(df)),
                 acq_tags=sorted(df.acq_tag.unique().tolist()))
        r['all'] = measure(R)
        r['even'], r['odd'] = measure(R, 0), measure(R, 1)
        res.append(r)
        bl = r['all']['blocks']
        print(f"{det} {sub[:28]:28s} {r['t_start']:5.2f}-{r['t_end']:5.2f} h  v {r['v_drift']:5.2f}  "
              f"v_lad x {r['all']['row']['x']['v_lad']:5.2f} y {r['all']['row']['y']['v_lad']:5.2f}  blocks "
              + ' '.join(f"{b['t']:.2f}:{0.5 * (b['x']['v_lad'] + b['y']['v_lad']):.2f}" for b in bl))
    json.dump(res, open(os.path.join(RES, 'blocks.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
