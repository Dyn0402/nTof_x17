#!/usr/bin/env python3
"""s4r_gap.py <dir> [--nboot 20] -- the S4 gap (gap_extent: half-height edges of the inclined-track X charge against z,
mean of the default and --satmask selections) on det2/det3/det6, with
    audit       strip_gain_audit's flags only (S4 as run: s4_stacks.fit_gap)
    mask        the pass bundle's dead list united with them (the stack re-check's mask)
    mask_in30   that, and the M3 reference at the mesh >= 30 mm from the active-area edge
<dir> as s4r_specs.py (rr1_<det>/incl.pkl, mask_<det>.json).  -> <dir>/gap_r.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_stacks  # noqa: E402  (sys.path for cloud_basics)
import gap_extent as GE  # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
from common.mx17_active_area import TRUE_ACTIVE_BY_DET  # noqa: E402


def fit(ev, bad, nboot):
    """s4_stacks.fit_gap with an explicit bad set."""
    out = {}
    for lab, sm in (('default', False), ('satmask', True)):
        GE.SATMASK = sm
        n, P, E = GE.profile(ev, 'x', bad, qmode='int')
        if n < 50:
            out[lab] = dict(n=int(n)); continue
        hh = GE.half(P, E)
        hh['gap_err'] = GE.boot(ev, 'x', bad, nboot, est=GE.half, qmode='int')
        sg = {}
        for s in (1, -1):
            ns, Ps, Es = GE.profile(ev, 'x', bad, qmode='int', tsign=s)
            if ns >= 40:
                sg[str(s)] = dict(n=int(ns), **GE.half(Ps, Es))
        out[lab] = dict(n=int(n), half=hh, by_sign=sg)
    GE.SATMASK = False
    a, b = out.get('default', {}).get('half'), out.get('satmask', {}).get('half')
    if a and b and np.isfinite(a['gap']) and np.isfinite(b['gap']):
        out['gap'] = 0.5 * (a['gap'] + b['gap'])
        out['gap_err'] = float(np.sqrt(0.25 * (a['gap_err'] ** 2 + b['gap_err'] ** 2) + (0.5 * (a['gap'] - b['gap'])) ** 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('--nboot', type=int, default=20)
    a = ap.parse_args()
    out = {}
    for det in ('det2', 'det3', 'det6'):
        ev = list(pickle.load(open(f'{a.dir}/rr1_{det}/incl.pkl', 'rb')).values())
        M = json.load(open(f'{a.dir}/mask_{det}.json'))
        (x0, x1), (y0, y1) = TRUE_ACTIVE_BY_DET['mx17_' + det[-1]]['x'], TRUE_ACTIVE_BY_DET['mx17_' + det[-1]]['y']
        inner = [e for e in ev if min(e['ref_mesh_x'] - x0, x1 - e['ref_mesh_x'], e['ref_mesh_y'] - y0, y1 - e['ref_mesh_y']) >= 30]
        out[det] = {}
        for lab, sub, bad in (('audit', ev, set(M['audit'])), ('mask', ev, set(M['union'])), ('mask_in30', inner, set(M['union']))):
            r = fit(sub, bad, a.nboot)
            out[det][lab] = r
            sg = {k: round(v['gap'], 2) for k, v in r.get('default', {}).get('by_sign', {}).items()}
            print(f'{det} {lab:9s} n {r.get("default", {}).get("n", 0):4d}  gap {r.get("gap", float("nan")):6.2f} ± '
                  f'{r.get("gap_err", float("nan")):.2f}  (default {r.get("default", {}).get("half", {}).get("gap", float("nan")):.2f}, '
                  f'satmask {r.get("satmask", {}).get("half", {}).get("gap", float("nan")):.2f}; by tan sign {sg})', flush=True)
        json.dump(out, open(f'{a.dir}/gap_r.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
