#!/usr/bin/env python3
"""set_bow.py [dets...] -- the proposed per-chamber `bow` bundle block (prototype for FE; no wft change here).

Measured on the golden row of each chamber, in PRODUCTION coordinates (the fit's own mid-gap position of the track,
u = x_p0 + 15 tan_x, w = y_p0 + 15 tan_y, constant-mapping tangents), U = (u - 200)/100, W = (w - 200)/100:
  h    mesh height surface [mm] from positions:  r = a + bu U + bw W + tan * (h0 + gu U + gw W + kuu U^2/2 + kww W^2/2 + kuw U W)
       (both views fitted; stored per view and as the error-weighted mean)
  A_p  drift-tilt surface [um/ns] per view, CONSTANT REMOVED (the constant is the per-run rigid tilt = bundle w0):
       w = c0 + A_p(U, W) + B tan (1 + su U + sw W),   A_p = cu U + cw W + auu U^2/2 + aww W^2/2 + auw U W
  s_p  angle-scale gradient (su, sw) relative to B
Writes results/bow_block_<det>.json.  See REPORT.md "Spec for FE" for how tan_of() and the track point use it.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbcommon import DETS, RES, DATASETS, load
from gate_bow import prep, fit_bow2d, fit_h


def block(det):
    T = load(det)
    out = dict(chamber=det, source_row=DATASETS[det][2] + ' slot, ' + DATASETS[det][3], frame=dict(
        coords='strip frame of each view; U = (u - 200 mm)/100 mm, W = (w - 200 mm)/100 mm, u/w = fitted p0 + 15 mm * tan',
        z='M3 z up; h = mesh height along z_up (negative = lower = further from the cathode)'),
        valid='bench only (cosmic bench mounting); never transfer to n_TOF without an in-situ measurement')
    hs, he = [], []
    for p in 'xy':
        D = prep(T, p)
        b = fit_bow2d(D, D['m'])
        out[f'A_{p}'] = dict(zip(['cu', 'cw', 'auu', 'aww', 'auw'], map(float, b[1:6])))
        # tan form (2026-10-29, rerun/calib/bow/BOW_TRANSFER.md): the tilt as a geometric tan offset, A / B; wft uses it
        # in preference to A, so the shape transfers to rows of the same chamber at another drift field
        out[f'a_{p}'] = dict(zip(['cu', 'cw', 'auu', 'aww', 'auw'], map(float, b[1:6] / b[6])))
        out[f'w0_const_{p}'] = float(b[0])
        out[f'B_{p}'] = float(b[6])
        out[f's_{p}'] = dict(su=float(b[7] / b[6]), sw=float(b[8] / b[6]))
        hb = fit_h(D, D['m'])
        out[f'h_{p}'] = dict(zip(['gu', 'gw', 'kuu', 'kww', 'kuw'], map(float, hb[4:9])))
        hs.append(hb[4:9])
    out['h'] = dict(zip(['gu', 'gw', 'kuu', 'kww', 'kuw'], map(float, np.mean(hs, axis=0))))
    json.dump(out, open(f'{RES}/bow_block_{det}.json', 'w'), indent=1)
    print(det, 'h', {k: round(v, 2) for k, v in out['h'].items()})


if __name__ == '__main__':
    for d in (sys.argv[1:] or DETS):
        block(d)
