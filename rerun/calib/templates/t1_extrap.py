"""T1b: extrapolate the template width to an infinitely inclined track (no drift segment on the strip).
FWHM^2 = A + B / tan^2 over the |tan| >= 0.08 bins (per-bin median of 1/tan^2 from the pulses themselves).
sqrt(A) is the width of the response with the strip's own segment removed; the production template's excess over it
is what the model counts twice.  Usage: python -I t1_extrap.py -> results/t1_extrap.json"""
import json, pickle, os
import numpy as np
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_tan_amp import CACHE, TAN_BINS, pulses, shape_stats, HERE

V = dict(det2=39.3, det3=35.8, det4=33.0, det6=24.2, det7=36.8)   # timing v [um/ns], PASS_PLAN physics inputs
out = {}
for det in V:
    ev = pickle.load(open(CACHE.format(det), 'rb'))
    out[det] = {}
    for plane in 'xy':
        P = pulses(ev, plane)
        tn = np.array([p[0] for p in P]); S = np.array([p[2] for p in P])
        xs, ys = [], []
        for lo, hi in TAN_BINS[1:]:
            m = (tn >= lo) & (tn < hi)
            if m.sum() < 25:
                continue
            xs.append(np.median(1 / tn[m] ** 2)); ys.append(shape_stats(list(S[m]))[1]['fwhm'] ** 2)
        B, A = np.polyfit(xs, ys, 1)
        prod = shape_stats(list(S[tn >= 0.22]))[1]['fwhm']
        # effective lateral width seen by the brightest strip: B = c * (L / v)^2, c = 2.355^2/12 for a box
        L = np.sqrt(B / (2.355 ** 2 / 12)) * V[det] / 1000
        out[det][plane] = dict(fwhm_inf=float(np.sqrt(max(A, 0))), fwhm_prod=float(prod), B=float(B),
                               box_width_mm=float(L), n_bins=len(xs))
        print(f'{det} {plane}: FWHM(tan->inf) {np.sqrt(max(A,0)):5.0f} ns, production template {prod:5.0f} ns, '
              f'excess {prod-np.sqrt(max(A,0)):4.0f} ns; equivalent box seen by the strip {L:.2f} mm (pitch 0.78)')
json.dump(out, open(f'{HERE}/results/t1_extrap.json', 'w'), indent=1)
