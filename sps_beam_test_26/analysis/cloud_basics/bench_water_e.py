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


def invert(g, v, E):
    vs = np.array([g(w, 0.0, E)['v'] for w in g.W])
    dts = np.array([g(w, 0.0, E)['DT'] for w in g.W])
    o = np.argsort(vs)
    w = float(np.interp(v, vs[o], g.W[o])) if vs.min() <= v <= vs.max() else float('nan')
    dt = float(np.interp(w, g.W, dts)) if np.isfinite(w) else float('nan')
    Dd = lambda d: 0.5 * (d * 1e-3 / np.sqrt(10)) ** 2 * v * 1e-3          # mm^2/ns
    return dict(E_Vcm=E, v_dry=float(vs[0]), water_pct=w, DT_um_rtcm=dt, DT_dry=float(dts[0]),
                Dd=Dd(dt), Dd_dry=Dd(dts[0]))


def main():
    g = GasGrid('bench', 'air_hs')
    out = {}
    for det, v in V_BUNDLE.items():
        out[det] = dict(v=v, drift_hv=DRIFT_HV[det], at250=invert(g, v, 250.0), run=invert(g, v, DRIFT_HV[det] / 2.88))
        for lab in ('at250', 'run'):
            r = out[det][lab]
            print(f'{det} {lab:5s} E {r["E_Vcm"]:4.0f}: v {v:5.2f} (dry {r["v_dry"]:4.1f}) -> water {r["water_pct"]:.2f} %, '
                  f'D_T {r["DT_um_rtcm"]:.0f} (dry {r["DT_dry"]:.0f}), Dd {r["Dd"]:.2e} (dry {r["Dd_dry"]:.2e})')
    json.dump(out, open(f'{HERE}/results/bench_water_e.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
