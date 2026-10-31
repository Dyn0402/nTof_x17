#!/usr/bin/env python3
"""mb_interp_test.py <dir with magboltz_*_c30.json> [--out results/mb_interp_test.json]
-- how fine must the (water, air) Magboltz grid be for GasGrid's linear interpolation?

Drop-one test: for every interior water node (and every interior air node) of each gas, rebuild the grid without
it, predict v, eta, D_L, D_T there by GasGrid's trilinear interpolation, and compare with the real point.  Linear
interpolation error grows as the square of the step, so the error of the half-as-fine grid (the actual one) at
its midpoints is the drop-one error / 4.  Each error is turned into the composition shift a fit would make to
absorb it: delta_v / (dv/dw) as water, delta_eta / (deta/da) as air (O2 = 20.95 % of air), with the slopes from
the full grid at the node.  Magboltz's own statistical floor on eta is the rms of raw log eta about the smooth
quadratic in log E (GasGrid._smooth_eta).
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gasmodel as M  # noqa: E402

O2_PER_AIR = 2095.0          # ppm O2 per % air


def grid_from(files, gas):
    """A GasGrid built from a list of files (copied into a temporary source dir)."""
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, 'x'))
    for f in files:
        os.symlink(f, os.path.join(d, 'x', os.path.basename(f)))
    old = M.RES; M.RES = d
    try:
        return M.GasGrid(gas, 'x')
    finally:
        M.RES = old; shutil.rmtree(d)


def node_of(f):
    import re
    m = re.search(r'_w([0-9p]+)_a([0-9p]+)_E', os.path.basename(f))
    return tuple(float(x.replace('p', '.')) for x in m.groups())


def stat_floor(G):
    """rms of raw log eta about its smooth quadratic, per gas (all mixtures pooled)."""
    r = []
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.log(G.raw[..., 1]) - np.log(G.cube[..., 1])
    r = lr[np.isfinite(lr)]
    return float(np.sqrt(np.mean(r ** 2))) if len(r) else None


def test(files, gas):
    F = [f for f in files if os.path.basename(f).startswith(f'magboltz_{gas}_w')]
    G = grid_from(F, gas)
    out = dict(gas=gas, W=G.W.tolist(), A=G.A.tolist(), E=G.E.tolist(), complete=G.complete,
               eta_stat_rel=stat_floor(G), water=[], air=[])
    for axis, nodes in (('water', G.W), ('air', G.A)):
        for k in range(1, len(nodes) - 1):
            x0 = nodes[k]
            keep = [f for f in F if node_of(f)[0 if axis == 'water' else 1] != x0]
            Gd = grid_from(keep, gas)
            other = G.A if axis == 'water' else G.W
            for y in other:
                w, a = (x0, y) if axis == 'water' else (y, x0)
                for E in G.E:
                    t = G(w, a, E); p = Gd(w, a, E)
                    if not (np.isfinite(t['v']) and np.isfinite(p['v'])):
                        continue
                    # slopes on the full grid at the node: dv/dw (water) and deta/da (air), central differences
                    lo, hi = nodes[k - 1], nodes[k + 1]
                    if axis == 'water':
                        dv = (G(hi, a, E)['v'] - G(lo, a, E)['v']) / (hi - lo)
                    ja = int(np.clip(np.searchsorted(G.A, a), 1, len(G.A) - 2))
                    alo, ahi = G.A[ja - 1], G.A[ja + 1]
                    de = (G(w, ahi, E)['eta'] - G(w, alo, E)['eta']) / (ahi - alo)
                    rec = dict(node=x0, other=float(y), E=float(E), step2=float(nodes[k + 1] - nodes[k - 1]),
                               dv_rel=float((p['v'] - t['v']) / t['v']),
                               deta_rel=float((p['eta'] - t['eta']) / t['eta']) if t['eta'] > 0 else None,
                               dDT_rel=float((p['DT'] - t['DT']) / t['DT']), dDL_rel=float((p['DL'] - t['DL']) / t['DL']))
                    if axis == 'water' and dv:
                        rec['water_shift'] = float((p['v'] - t['v']) / dv)          # % water a fit would move
                    if de > 0:
                        rec['o2_shift_ppm'] = float((p['eta'] - t['eta']) / de * O2_PER_AIR)   # ppm O2 a fit would move
                    out[axis].append(rec)
    return out


def summary(R):
    S = {}
    for g, r in R.items():
        s = dict(eta_stat_rel=r['eta_stat_rel'], W=r['W'], A=r['A'])
        for axis in ('water', 'air'):
            rows = r[axis]
            if not rows:
                continue
            by = {}
            for q in rows:
                by.setdefault(q['node'], []).append(q)
            s[axis] = []
            for node, qs in sorted(by.items()):
                v = np.array([abs(q['dv_rel']) for q in qs])
                e = np.array([abs(q['deta_rel']) for q in qs if q['deta_rel'] is not None])
                ws = np.array([abs(q.get('water_shift', np.nan)) for q in qs]); ws = ws[np.isfinite(ws)]
                os_ = np.array([abs(q.get('o2_shift_ppm', np.nan)) for q in qs]); os_ = os_[np.isfinite(os_)]
                sh = ws if axis == 'water' else os_
                s[axis].append(dict(node=node, step_dropped=qs[0]['step2'],
                                    dv_rel_med=float(np.median(v)), dv_rel_max=float(v.max()),
                                    deta_rel_med=float(np.median(e)) if len(e) else None,
                                    shift_med=float(np.median(sh)) if len(sh) else None,
                                    shift_p90=float(np.percentile(sh, 90)) if len(sh) else None,
                                    # the actual (half) step's midpoint error is ~ 1/4 of the drop-one error
                                    shift_actual_p90=float(np.percentile(sh, 90) / 4) if len(sh) else None,
                                    o2_shift_actual_p90=float(np.percentile(os_, 90) / 4) if len(os_) else None))
        S[g] = s
    return S


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'mb_interp_test.json'))
    a = ap.parse_args()
    import glob
    files = sorted(glob.glob(os.path.join(os.path.abspath(a.dir), 'magboltz_*_c30.json')))
    R = {g: test(files, g) for g in ('bench', 'co2', 'beam')}
    S = summary(R)
    json.dump(dict(summary=S), open(a.out, 'w'), indent=1)
    for g, s in S.items():
        print(f'== {g}: eta stat (rel) {s["eta_stat_rel"]:.3f}  W {s["W"]}  A {s["A"]}')
        for axis in ('water', 'air'):
            for q in s.get(axis, []):
                de = q['deta_rel_med']
                print(f'  {axis:5s} node {q["node"]:6g} drop-step {q["step_dropped"]:.3f}: dv med/max {q["dv_rel_med"]*100:.2f}/'
                      f'{q["dv_rel_max"]*100:.2f} %  deta med {de*100 if de is not None else float("nan"):.1f} %  '
                      f'actual-step p90 shift {q["shift_actual_p90"]}  (O2-equivalent {q["o2_shift_actual_p90"]} ppm)')
