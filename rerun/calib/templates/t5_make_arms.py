"""T5: bundle arms for the held-out A/B of the template (det3 by default).

Each arm = the fe_nobow bundle with tmpl_x replaced (rc_tmpl='x', so it drives both views) and shifted in time by
delta ns (a template shift is the same as moving the pinned t0 by -delta: it scans the template-zero vs latency
convention).  Templates: 'cur' = the bundle's own; 'heff' = t4's segment-free fit for this chamber's X view
(unit peak, t50 at 0, the bundle grid).  Writes $OUT/<det>/<tmpl>_d<delta>/ (bundle.json + arrays.npz).
Usage: python -I t5_make_arms.py det3 OUTDIR"""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t4_closure import response, T

SRC = os.path.expanduser('~/x17/cosmic_bench/model_audit/fe_unify2/bundles/{}/calib_bundle_fe_nobow')
SHIFTS = (-40, -20, 0, 20, 40)


def heff_on_grid(det, grid):
    p = json.load(open(os.path.join(os.path.dirname(__file__), 'results/t4_closure.json')))[f'{det}_x']
    h = response(p['t_peak'], p['beta'], p['f_ion'], p['ion_stretch']); h = h / h.max()
    i = np.argmax(h >= 0.5); t50 = T[i - 1] + (0.5 - h[i - 1]) / (h[i] - h[i - 1]) * (T[1] - T[0])
    return np.interp(grid, T - t50, h, left=0.0, right=0.0)


def main(det, out):
    src = SRC.format(det); a = dict(np.load(f'{src}/arrays.npz')); grid = a['grid']
    tm = dict(cur=a['tmpl_x'].astype(float), heff=heff_on_grid(det, grid))
    for name, t in tm.items():
        for d in SHIFTS:
            dst = f'{out}/{det}/{name}_d{d:+d}'; os.makedirs(dst, exist_ok=True)
            shutil.copy(f'{src}/bundle.json', dst)
            b = dict(a); b['tmpl_x'] = np.interp(grid - d, grid, t, left=0.0, right=t[-1])
            np.savez(f'{dst}/arrays.npz', **b)
    print('arms in', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
