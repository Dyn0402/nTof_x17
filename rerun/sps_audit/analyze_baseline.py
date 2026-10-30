#!/usr/bin/env python3
"""analyze_baseline.py --hits hits_run53.npz [...] [--out results/baseline.json]

Reproduce the run-53 176 um from hits_baseline.py output with analyze.py's machinery (telescope clusters, 2-D robust
affine, fixed-window core, zone decomposition, free-slope fit), then apply the pass's det4 selection to the SAME
estimator to see what the selection alone does:
  all       resolution.py's selection: one cluster in each uRWELL plane-view and in each det4 view
  amax300   + the det4 hit maximum >= 300 ADC in both views (the pass seed's peak >= 300 ADC, on hits)
  amax<300  the complement
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import resid2d, resid2d_seeded, decompose, Z_DET4, ZGRID, gauss_sigma, FID_MM   # noqa: E402


def run(D, tag, seeded=False):
    """seeded=False: resolution.py exactly (robust_lstsq, no fiducial).  seeded=True: analyze.py's alignment and
    cross-view fiducial, to show that the machinery used on the pass rows does not move the run-53 answer."""
    if seeded:
        rx, ry, A = resid2d_seeded(D, D.dx.values, D.dy.values, Z_DET4)
        fy, fx = np.abs(ry) < FID_MM, np.abs(rx) < FID_MM
        R = dict(n=int(len(D)), det_A=float(np.linalg.det(A)), det4_Y=decompose(rx[fy], D.bx_p.values[fy], 'x', Z_DET4),
                 det4_X=decompose(ry[fx], D.by_p.values[fx], 'y', Z_DET4))
        _print(R, tag)
        return R
    rx, ry, A = resid2d(D, D.dx.values, D.dy.values, Z_DET4)
    zs = []
    for z in ZGRID:
        a, b, _ = resid2d(D, D.dx.values, D.dy.values, z)
        zs.append((float(z), float(gauss_sigma(a)[0]), float(gauss_sigma(b)[0])))
    R = dict(n=int(len(D)), det_A=float(np.linalg.det(A)), zscan=zs,
             det4_Y=decompose(rx, D.bx_p.values, 'x', Z_DET4), det4_X=decompose(ry, D.by_p.values, 'y', Z_DET4),
             strips_x=float(D.dx_size.mean()), strips_y=float(D.dy_size.mean()))
    _print(R, tag)
    return R


def _print(R, tag):
    line = f'  {tag:9s} n {R["n"]:6d} det(A) {R["det_A"]:+.4f}'
    for view in ('det4_X', 'det4_Y'):
        V = R[view]
        zz = ' '.join(f'{q["pitch"]}:{q["sigma_res"] * 1e3:.0f}->{q["sigma_det4"] * 1e3:.0f}' for q in V['zones'])
        fit = f' FIT f={V["fit"]["f_back"]:.3f} det4={V["fit"]["sigma_det4"] * 1e3:.0f}+-{V["fit"]["sigma_det4_err"] * 1e3:.0f}' if V['fit'] else ''
        line += f' | {view[-1]}: core {V["core_frac"]:.2f} [{zz}]{fit}'
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hits', nargs='+', required=True); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'baseline.json'))
    a = ap.parse_args()
    out = {}
    for f in a.hits:
        H = dict(np.load(f))
        D = pd.DataFrame({k: v for k, v in H.items()})
        sel = np.all([(D[f'{p}_n'] == 1).values for p in ('fx', 'fy', 'bx', 'by')], axis=0) & (D.dx_n == 1).values & (D.dy_n == 1).values
        D = D[sel].reset_index(drop=True)
        name = os.path.basename(f)[:-4]
        print(f'== {name}: {len(D)} clean coincidences')
        hi = (D.dx_amax >= 300) & (D.dy_amax >= 300)
        out[name] = dict(all=run(D, 'all'), all_seeded=run(D, 'all/seed', True), amax300=run(D[hi].reset_index(drop=True), 'amax300'),
                         amax_lt300=run(D[~hi].reset_index(drop=True), 'amax<300'), frac_amax300=float(hi.mean()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
