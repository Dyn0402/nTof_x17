"""q19_curl_cens.py -- outstanding question 19: does the curl excess of chamber C (and D longer) come from tracks whose
drift column is censored by the depth grid?  Repeats audit D part 4 (n = 6 reference-indexed cells, plane gradients of
tau = w0/v, bootstrap 200) with all tracks and with the censored ("railed") tracks removed (event removed if its
profile reaches the last depth bin in either view).  -> $WORK/audit/q16/q19_curl.json
"""
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from d_lib import load, view, path, cells, UL, WL  # noqa: E402

NB = 200
N = 6
OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/q16'


def centres(n):
    eu, ew = np.linspace(*UL, n + 1), np.linspace(*WL, n + 1)
    return 0.5 * (eu[1:] + eu[:-1]), 0.5 * (ew[1:] + ew[:-1])


def tau_grid(U, W, w, tr):
    C = cells(U, W, w, tr, N, minn=40)
    return np.array([c['w0'] / c['v'] if np.isfinite(c['v']) else np.nan for c in C]).reshape(N, N)


def plane(g):
    Uc, Wc = centres(N)
    UU, WW = np.meshgrid(Uc, Wc, indexing='ij')
    ok = np.isfinite(g)
    if ok.sum() < 6:
        return None
    return np.linalg.lstsq(np.c_[np.ones(ok.sum()), UU[ok], WW[ok]], g[ok], rcond=None)[0]


def curl(dx, dy):
    """dx, dy: tuples (U, W, w, tr) for the two views -> (dtx/dW - dty/dU) in 1e-3 per 100 mm"""
    cx, cy = plane(tau_grid(*dx)), plane(tau_grid(*dy))
    if cx is None or cy is None:
        return np.nan, np.nan, np.nan
    return 1e3 * (cx[2] - cy[1]), 1e3 * cx[1], 1e3 * cy[2]


rng = np.random.default_rng(19)
res = {}
for key, xh in (('det6:long', False), ('det6:longer', False), ('det6:short', False), ('det7:longer', True),
                ('det7:long', True), ('det2:long', False), ('det3:p2', False)):
    d = load(key, xh)
    full = pd.read_parquet(path(key, 'bow_gate_table.parquet'))
    mx = np.nanmax(np.r_[full.x_q_uend, full.y_q_uend])
    rail = ((d.x_q_uend >= mx - 1) | (d.y_q_uend >= mx - 1)).values
    out = dict(rail=float(rail.mean()), N=len(d))
    for lab, m in (('all', np.ones(len(d), bool)), ('uncensored', ~rail)):
        dd = d[m].reset_index(drop=True)
        vx, vy = view(dd, 'x', idx='ref')[:4], view(dd, 'y', idx='ref')[:4]
        c0 = curl(vx, vy)
        bs = []
        for _ in range(NB):
            ix = rng.integers(0, len(vx[0]), len(vx[0])); iy = rng.integers(0, len(vy[0]), len(vy[0]))
            bs.append(curl(tuple(a[ix] for a in vx), tuple(a[iy] for a in vy))[0])
        bs = np.array(bs)
        out[lab] = dict(N=int(m.sum()), curl=float(c0[0]), err=float(np.nanstd(bs)), sig=float(c0[0] / np.nanstd(bs)),
                        dtx_dU=float(c0[1]), dty_dW=float(c0[2]))
    res[key + ('|xhalf' if xh else '')] = out
    print(key, xh, json.dumps(out), flush=True)
json.dump(res, open(f'{OUT}/q19_curl.json', 'w'), indent=1)
