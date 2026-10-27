#!/usr/bin/env python3
"""set_angle_map.py --bundle <src bundle> --final <finalize dir> --out <bundle dir> [--planes x y]

Measure the position-dependent angle mapping (CalibrationBundle.angle_map, ws_widthtilt 2026-10-27) from a bench
reconstruction made WITH the source bundle, and write a copy of the bundle carrying it.  The finalize dir is the one
holding events.parquet and angles/per_event.npz (fitted w, the M3 tan_ref).

Per plane, in 12 bins of the track's mid-gap position along the view's own axis — taken from the FIT
(p0 + 15 mm · tan on the constant mapping), i.e. the coordinate production has — a robust (Huber) line
    w·1e3 = w0_j + s_j · kw·v · tan_ref      over 0.04 < |tan_ref| < 0.40, outliers beyond 15 % of v removed.
Bins with < 40 tracks or < 10 of either sign are dropped (the map is clamped beyond the outermost kept bin).
Cross-validated effect on the five golden rows (w0map_xval.py): angle s68 −12…−27 % on det2/det3/det4 Y/det6/det7 Y.
"""
import argparse, json, os, shutil, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
EDGES = np.linspace(0, 400, 13)


def robust(X, y, scale):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(12):
        r = np.abs(X @ beta - y)
        sw = np.sqrt(np.where(r <= scale, 1.0, scale / np.maximum(r, 1e-12)))
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return beta


def measure(p, t, w, vk):
    """(knots, w0, s, n) of the map; w [um/ns], vk = kw*v."""
    knots, a, s, n = [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (p >= lo) & (p < hi) & (np.abs(t) > 0.04) & (np.abs(t) < 0.40) & (np.abs(w - vk * t) < 0.15 * vk)
        if m.sum() < 40 or (t[m] > 0.04).sum() < 10 or (t[m] < -0.04).sum() < 10:
            continue
        b = robust(np.c_[np.ones(m.sum()), vk * t[m]], w[m], 0.03 * vk)
        knots.append(0.5 * (lo + hi)); a.append(float(b[0])); s.append(float(b[1])); n.append(int(m.sum()))
    return knots, a, s, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True); ap.add_argument('--final', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--planes', nargs='*', default=['x', 'y'])
    a = ap.parse_args()
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(a.bundle)
    z = np.load(os.path.join(a.final, 'angles', 'per_event.npz'))
    ev = pd.read_parquet(os.path.join(a.final, 'events.parquet')).drop_duplicates('event_id', keep=False).set_index('event_id')
    amap, rec = {}, {}
    for p in a.planes:
        eid = z[f'{p}_eid']; keep = np.isin(eid, ev.index)
        eid, t, w = eid[keep], z[f'{p}_tan_ref'][keep], z[f'{p}_w'][keep] * 1e3
        vk = cal.kw.get(p, 1.0) * cal.v_drift
        p0 = ev.loc[eid, f'{p}_p0'].to_numpy()
        pos = p0 + 15.0 * (w - cal.w0.get(p, 0.0)) / vk
        ok = np.isfinite(pos) & np.isfinite(t) & np.isfinite(w)
        k, w0, s, n = measure(pos[ok], t[ok], w[ok], vk)
        if len(k) < 4:
            print(f'{p}: too few populated bins ({len(k)}), no map'); continue
        amap[p] = dict(p=k, w0=w0, s=s)
        rec[p] = dict(n=n)
        print(f'{p}: w0(p) ' + ' '.join(f'{x:+.2f}' for x in w0) + '\n   s(p)  ' + ' '.join(f'{x:.3f}' for x in s))
    if os.path.abspath(a.out) != os.path.abspath(a.bundle):
        if os.path.exists(a.out):
            shutil.rmtree(a.out)
        shutil.copytree(a.bundle, a.out)
    cal = CalibrationBundle.load(a.out)
    cal.angle_map = amap
    prov = dict(cal.provenance)
    prov['angle_map'] = dict(tool='rerun/model_audit/ws_widthtilt/set_angle_map.py', final=os.path.abspath(a.final),
                             n_per_bin=rec, mid_gap_mm=15.0)
    cal.provenance = prov
    cal.save(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
