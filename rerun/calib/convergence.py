#!/usr/bin/env python3
"""convergence.py <run_key> --calib <row>/wft/calib --final <row>/wft/calib_bundle --out CALIB_CONVERGENCE.json -- E3's
convergence criteria (PASS_PLAN_2026-10-28 §2) between the iterations of one chamber's calib phase.

  bundle_0 -> bundle_1 (c0/bundle vs c1/bundle):  template shape  L2 difference < 1 % of the peak (per view)
                                                  latency C       |dC| < 2 ns (per view, t0_abs[ftst 0])
                                                  D_rc,y          |d| / D < 5 %
  bundle_2 -> final    (c2/bundle vs calib_bundle): w0             |dw0| < 0.02 um/ns (per view)
                                                  kw              |dkw| / kw < 0.5 % (per view)
  BOW2 -> BOW3 (c2/bow_block.json vs r3/bow_block.json): tan-form tilt terms a_p |d| < 0.002 (~0.1 deg) and mesh
                                                  height terms h |d| < 0.4 mm (the skeptic review's systematic); the
                                                  blocks carry no errors, so these stated tolerances stand in for "1 sigma"
  ALIGN2 -> ALIGN3 (r2 vs r3 alignment.json):     x/y offset |d| < 0.1 mm, theta |d| < 0.05 deg, z equal (1 mm grid)

A chamber that fails gets one more iteration (by hand: the DAG has no loop) and the overrun is written down.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)


def _crit(name, value, tol, unit=''):
    ok = bool(np.isfinite(value) and abs(value) < tol)
    return dict(name=name, value=float(value), tol=tol, unit=unit, ok=ok)


def bundles_01(b0, b1):
    out = []
    for p in ('x', 'y'):
        t0 = np.interp(np.asarray(b1.grid), np.asarray(b0.grid), np.asarray(b0.tmpl[p]))
        t1 = np.asarray(b1.tmpl[p])
        out.append(_crit(f'template_{p}_L2_over_peak', np.sqrt(np.mean((t1 - t0) ** 2)) / max(t1.max(), 1e-9), 0.01))
        out.append(_crit(f'latency_{p}_dC', float(b1.t0_abs[p][0]) - float(b0.t0_abs[p][0]), 2.0, 'ns'))
    d0, d1 = float(b0.hyper['rc_D_y']), float(b1.hyper['rc_D_y'])
    out.append(_crit('rc_D_y_rel', (d1 - d0) / d0, 0.05))
    return out


def bundles_2f(b2, bf):
    out = []
    for p in ('x', 'y'):
        out.append(_crit(f'w0_{p}', bf.w0.get(p, 0.0) - b2.w0.get(p, 0.0), 0.02, 'um/ns'))
        out.append(_crit(f'kw_{p}_rel', bf.kw.get(p, 1.0) / b2.kw.get(p, 1.0) - 1.0, 0.005))
    return out


def bow(a, b):
    out = []
    for p in ('x', 'y'):
        for k, v in (a.get(f'a_{p}') or {}).items():
            out.append(_crit(f'bow_a_{p}_{k}', (b.get(f'a_{p}') or {}).get(k, np.nan) - v, 0.002, 'tan'))
    for k, v in (a.get('h') or {}).items():
        out.append(_crit(f'bow_h_{k}', (b.get('h') or {}).get(k, np.nan) - v, 0.4, 'mm'))
    return out


def align(a, b):
    out = [_crit(f'align_{k}', b[k] - a[k], 0.1, 'mm') for k in ('x_offset', 'y_offset')]
    out.append(_crit('align_theta', b['theta_deg'] - a['theta_deg'], 0.05, 'deg'))
    out += [_crit(f'align_{k}', b[k] - a[k], 0.5, 'mm') for k in ('z_x', 'z_y')]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--calib', required=True, help='<row>/wft/calib (c0, c1, c2, r2, r3)')
    ap.add_argument('--final', required=True, help='<row>/wft/calib_bundle')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    from wft.calib import CalibrationBundle
    L = lambda *p: CalibrationBundle.load(os.path.join(a.calib, *p))
    J = lambda *p: json.load(open(os.path.join(a.calib, *p)))
    groups = dict(bundle_0_to_1=bundles_01(L('c0', 'bundle'), L('c1', 'bundle')),
                  bundle_2_to_final=bundles_2f(L('c2', 'bundle'), CalibrationBundle.load(a.final)),
                  bow_2_to_3=bow(J('c2', 'bow_block.json'), J('r3', 'bow_block.json')),
                  align_2_to_3=align(J('r2', 'alignment', 'alignment.json'), J('r3', 'alignment', 'alignment.json')))
    res = dict(run_key=a.run_key, converged=all(c['ok'] for g in groups.values() for c in g),
               failed=[c['name'] for g in groups.values() for c in g if not c['ok']], groups=groups,
               rule='PASS_PLAN_2026-10-28 §2 (convergence criteria); bow tolerances stated (blocks carry no errors)')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f"[convergence] {a.run_key}: {'CONVERGED' if res['converged'] else 'NOT converged: ' + ', '.join(res['failed'])}")
    for g, cs in groups.items():
        print(f'  {g}: ' + ', '.join(f"{c['name']} {c['value']:+.4g}{'' if c['ok'] else ' FAIL'}" for c in cs))


if __name__ == '__main__':
    main()
