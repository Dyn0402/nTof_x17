#!/usr/bin/env python3
"""bench_water_e.py -- bench water fraction and drift diffusion from v, at each run's real drift field.

bench_pinned.py inferred each chamber's water from its bundle v with Magboltz at 250 V/cm for all
chambers, from the old single-field Magboltz files.  The runs were at drift 1000 V (det2, det3),
700 V (det6, det7) and 600 V (det4) = 347 / 243 / 208 V/cm (V / 2.88 cm, bench_driftscan.py).  Here
the same inversion on the high-statistics air_hs grid (gasmodel.GasGrid, no air), at 250 V/cm and at
the run field, with the Dd = D_T^2 v / 2 it implies and the dry value for comparison.  bench_pinned.json
is left as it was (its fits predate the unbiased stacks, and arms were built from it).

    bench_water_e.py  -> results/bench_water_e.json
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gasmodel import GasGrid

V_BUNDLE = {'det2': 39.94, 'det3': 36.6, 'det4': 34.16, 'det6': 26.7, 'det7': 36.6}
DRIFT_HV = {'det2': 1000, 'det3': 1000, 'det4': 600, 'det6': 700, 'det7': 700}     # run_config.json


def invert(g, v, E, step=0.005):
    """Water fraction(s) whose Magboltz v equals v at field E, and the D_T there.

    v(water) is not monotonic above ~240 V/cm (at 347 V/cm it rises from 38.8 dry to 41.1 at 0.4 %
    and falls to 35.1 at 1 %), so a v can have two solutions or, below the dry value, only the wet
    one.  All are returned in 'branches' (low water first); water_pct / DT_um_rtcm are the lowest-water
    one and 'ambiguous' says whether there is another.  (Until 2026-10-16 this sorted v and
    interpolated, which at 347 V/cm returned a wrong low-water value for v below dry: det2 0.25 %
    instead of 0.81 %, det3 bundle 0.59 % instead of 0.89 %.)"""
    W = np.arange(g.W[0], g.W[-1] + step / 2, step)
    vs = np.array([g(w, 0.0, E)['v'] for w in W])
    dts = np.array([g(w, 0.0, E)['DT'] for w in W])
    Dd = lambda d: 0.5 * (d * 1e-3 / np.sqrt(10)) ** 2 * v * 1e-3          # mm^2/ns
    br = []
    for i in range(len(W) - 1):
        a, b = vs[i] - v, vs[i + 1] - v
        if a == 0 or a * b < 0:
            t = a / (a - b)
            w, dt = W[i] + t * (W[i + 1] - W[i]), dts[i] + t * (dts[i + 1] - dts[i])
            br.append(dict(water_pct=float(w), DT_um_rtcm=float(dt), Dd=Dd(dt)))
    first = br[0] if br else dict(water_pct=float('nan'), DT_um_rtcm=float('nan'), Dd=float('nan'))
    return dict(E_Vcm=E, v_dry=float(vs[0]), **first, DT_dry=float(dts[0]), Dd_dry=Dd(dts[0]),
                ambiguous=len(br) > 1, branches=br)


def main():
    g = GasGrid('bench', 'air_hs')
    out = {}
    for det, v in V_BUNDLE.items():
        out[det] = dict(v=v, drift_hv=DRIFT_HV[det], at250=invert(g, v, 250.0), run=invert(g, v, DRIFT_HV[det] / 2.88))
        for lab in ('at250', 'run'):
            r = out[det][lab]
            br = ' | '.join(f'water {b["water_pct"]:.2f} %, D_T {b["DT_um_rtcm"]:.0f}' for b in r['branches']) or 'no solution'
            print(f'{det} {lab:5s} E {r["E_Vcm"]:4.0f}: v {v:5.2f} (dry {r["v_dry"]:4.1f}) -> {br}  (dry D_T {r["DT_dry"]:.0f})')
    json.dump(out, open(f'{HERE}/results/bench_water_e.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
