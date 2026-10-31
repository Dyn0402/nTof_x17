#!/usr/bin/env python3
"""xasym_sys.py [--out results/xasym_sys.json] -- the wft.xasym +- systematic on the Paper I bench numbers (PASS_PLAN §4).

The production X position carries K x dx(x, y) of the board's raw centroid-bias map (wft/xasym.py: K = -0.77 +- 0.13,
maps for det2 and det3 June-15 only; det4/6/7 excluded as too sparse).  Arms on the X view of the det2/det3 golden and
headline rows, 03_angles' interior_good events:
  none    the pass position as it is
  K       x - K dx scale (scale: det3's session_scale for the row's session, det2 1)
  K-1s / K+1s   K -+ 0.13
Each arm is re-aligned the way the pass alignment would absorb it: the M3 residual res = p0 - mesh_x (the effbudget's raw
strip frame) has an offset + rotation + scale removed by a robust (3 MAD-trimmed) linear fit in (mesh_x, mesh_y), so only
the pattern is left.  Quoted: core sigma (Gaussian in a FIXED window median +- 2 s68, res_rows.core_sigma's recipe) and
s68 of the mesh residual, and the same at mid-gap (p0 + 15 tan_fit against mesh + 15 tan_ref).  The angle is untouched by
the correction (it moves p0 only); the X copy's angle cost is the FD toy's L3 rung (0.06-0.19 deg).
"""
import argparse
import json
import os
import sys

import numpy as np

import floorlib as C

sys.path.insert(0, C.REPO)
from wft import xasym  # noqa: E402

ROWS = {'det2': [('calib', C.CALIB['det2'], 1.0), ('headline', C.HEADLINE['det2'], 1.0)],
        'det3': [('calib', C.CALIB['det3'], 0.44), ('headline', C.HEADLINE['det3'], 0.44)]}
BUILD = {'det2': 'det2', 'det3': 'det3 June-15'}
K, K_ERR = xasym.maps()['K'], xasym.maps()['K_err']


def core_sigma(d):
    d = d[np.isfinite(d)]
    med = np.median(d)
    s = np.percentile(np.abs(d - med), 68.27)
    k = np.abs(d - med) < 2 * s
    h, e = np.histogram(d[k], bins=40)
    c = 0.5 * (e[1:] + e[:-1])
    from scipy.optimize import curve_fit
    try:
        p, _ = curve_fit(lambda x, a, m, sg: a * np.exp(-0.5 * ((x - m) / sg) ** 2), c, h, p0=(h.max(), med, s))
        return float(abs(p[2])), float(s)
    except Exception:
        return float('nan'), float(s)


def realign(res, mx, my):
    A = np.vstack([np.ones_like(mx), mx, my]).T
    k = np.isfinite(res) & (np.abs(res) < 15)
    for _ in range(3):
        c, *_ = np.linalg.lstsq(A[k], res[k], rcond=None)
        r = res - A @ c
        mad = np.median(np.abs(r[k] - np.median(r[k])))
        k = np.isfinite(res) & (np.abs(r) < 3 * 1.4826 * mad + 1e-9)
    return res - A @ c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(C.HERE, 'results', 'xasym_sys.json'))
    a = ap.parse_args()
    out = {}
    for det, rows in ROWS.items():
        for role, key, scale in rows:
            if role == 'calib' and det == 'det2':
                scale = 1.0
            F = C.load(key)
            g = F[(F.view == 'x') & F.ig & np.isfinite(F.p0) & np.isfinite(F.o_p0) & np.isfinite(F.r_mesh_x)]
            x, y, mx, my = g.p0.to_numpy(), g.o_p0.to_numpy(), g.r_mesh_x.to_numpy(), g.r_mesh_y.to_numpy()
            tf, tr = g.tan_theta.to_numpy(), g.tan_ref.to_numpy()
            dxm = xasym.dx_mm(x, y, BUILD[det])
            arms = {}
            for name, k in (('none', 0.0), ('K', K), ('K-1s', K - K_ERR), ('K+1s', K + K_ERR)):
                xc = x - k * dxm * scale
                mesh = realign(xc - mx, mx, my)
                mid = realign((xc + 15 * tf) - (mx + 15 * tr), mx, my)
                cm, sm = core_sigma(mesh[np.abs(mesh) < 15])
                cd, sd = core_sigma(mid[np.abs(mid) < 15])
                arms[name] = dict(mesh_core_um=1e3 * cm, mesh_s68_um=1e3 * sm, mid_core_um=1e3 * cd, mid_s68_um=1e3 * sd)
            shift = np.abs(K * dxm * scale)
            out[f'{det}|{role}'] = dict(key=key, build=BUILD[det], scale=scale, n=int(len(g)),
                                        shift_um_rms=float(1e3 * np.sqrt(np.mean((K * dxm * scale) ** 2))),
                                        shift_um_max=float(1e3 * shift.max()), arms=arms)
            A = arms
            print(f"{det} {role} n={len(g)} scale {scale}: shift rms {out[f'{det}|{role}']['shift_um_rms']:.0f} um | "
                  + ' | '.join(f"{n}: mesh {v['mesh_core_um']:.0f}/{v['mesh_s68_um']:.0f} mid {v['mid_core_um']:.0f}"
                               for n, v in A.items()))
    out['K'], out['K_err'] = K, K_ERR
    out['excluded'] = xasym.maps().get('excluded', {})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
