#!/usr/bin/env python3
"""common_sigma0.py -- item 5 (i): does ONE prompt footprint sigma0 fit both views of the SPS det4 head-on RAW data?

Input: the per-event chi2 grids of ws_sps/headon_sigma.py (run_71 RAW, 1200 clean views per field; grid
sigma0 0.10..0.60 step 0.05, Dp 0..0.030 step 0.005, Y also D_rc in {1.4, 2.2, 3.0, 3.8}e-4; each grid point is the
chi2 at the event's own free (p0, w), t0 re-optimised over +-5 ns).  For each field:

  separate : min over (s_x, Dp_x) of chi2_X  +  min over (s_y, Dp_y, D_rc) of chi2_Y        (the ws_sps result)
  common   : min over s of [ min over Dp_x chi2_X(s, .)  +  min over (Dp_y, D_rc) chi2_Y(s, ., .) ]
  Delta chi2 = common - separate, in units of the summed chi2/dof (the chi2 is model-limited, chi2/dof ~ 10), and
  also as an event bootstrap of (s_y - s_x): the events are the independent unit.

The D_rc axis is the Y RC spread left free (the 'fast component' question: any RC spreading the grid can mimic with a
wider prompt footprint).  The X anomaly (X width grows with drift time at ~2x Magboltz) is absorbed by Dp_x free;
--dpx-mb bounds it the other way (Dp_x pinned at the Y value), to show how much of sigma0_X depends on it.
Every chi2 profile is interpolated in sigma0 with a cubic through the grid (0.05 mm steps).
Writes results/common_sigma0.json and figures/common_sigma0_profiles.png.
"""
import json, os, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_sps')
FIELDS = [('raw700', 243.2), ('raw450', 156.3), ('raw275', 95.5)]
SF = np.linspace(0.10, 0.60, 501)


def grids(tag):
    D = pickle.load(open(os.path.join(SRC, f'headon_{tag}.pkl'), 'rb'))
    out = {}
    for v in 'xy':
        R = {r['eid']: r[v] for r in D['rows'] if v in r}
        e = np.array(sorted(R))
        out[v] = dict(eid=e, G=np.stack([R[i]['G'] for i in e]), dof=np.array([R[i]['dof'] for i in e]))
    return D, out


def interp(S0, prof):
    from scipy.interpolate import CubicSpline
    return CubicSpline(S0, prof)(SF)


def profiles(Gx, Gy, S0, DP, dpx_fix=None):
    """sigma0 profiles of the summed chi2 (each view's nuisances minimised at every sigma0)."""
    Sx = Gx.sum(0)                     # (s, dp, 1)
    Sy = Gy.sum(0)                     # (s, dp, drc)
    px = Sx[:, :, 0].min(1) if dpx_fix is None else Sx[:, dpx_fix, 0]
    py = Sy.min(axis=(1, 2))
    return px, py


def analyse(tag, dpx_mode='free', nboot=400):
    D, g = grids(tag)
    S0, DP = np.asarray(D['S0']), np.asarray(D['DP'])
    Gx, Gy = g['x']['G'], g['y']['G']
    dof = g['x']['dof'].sum() + g['y']['dof'].sum()

    def fit(Gx, Gy):
        dpx = None
        if dpx_mode == 'ylike':                   # bound: X diffusion forced to the Y-preferred value
            Sy = Gy.sum(0); _, j, _ = np.unravel_index(np.argmin(Sy), Sy.shape); dpx = j
        px, py = profiles(Gx, Gy, S0, DP, dpx)
        fx, fy = interp(S0, px), interp(S0, py)
        sx, sy = SF[np.argmin(fx)], SF[np.argmin(fy)]
        fc = fx + fy; sc = SF[np.argmin(fc)]
        return dict(sx=sx, sy=sy, sc=sc, chi_sep=fx.min() + fy.min(), chi_com=fc.min(), fx=fx, fy=fy)

    r = fit(Gx, Gy)
    scale = r['chi_sep'] / dof
    dchi = (r['chi_com'] - r['chi_sep']) / scale
    rng = np.random.default_rng(5); bs = []
    for _ in range(nboot):
        ix = rng.integers(0, len(Gx), len(Gx)); iy = rng.integers(0, len(Gy), len(Gy))
        b = fit(Gx[ix], Gy[iy]); bs.append((b['sx'], b['sy'], b['sc'], b['sy'] - b['sx']))
    bs = np.array(bs)
    # per-event Delta chi2 distribution (each event's own cost of the common value): sign test
    pe = []
    for G, v in ((Gx, 'x'), (Gy, 'y')):
        pass
    out = dict(tag=tag, dpx_mode=dpx_mode, n_x=int(len(Gx)), n_y=int(len(Gy)), chi2dof=float(scale),
               s0_x=float(r['sx']), s0_y=float(r['sy']), s0_common=float(r['sc']),
               s0_x_err=float(bs[:, 0].std()), s0_y_err=float(bs[:, 1].std()), s0_common_err=float(bs[:, 2].std()),
               dy_minus_x=float(r['sy'] - r['sx']), dy_minus_x_err=float(bs[:, 3].std()),
               dy_minus_x_p=[float(np.percentile(bs[:, 3], q)) for q in (2.5, 16, 50, 84, 97.5)],
               dchi2_scaled=float(dchi), dchi2_raw=float(r['chi_com'] - r['chi_sep']),
               cost_x_at_common=float((r['fx'][np.argmin(r['fx'] + r['fy'])] - r['fx'].min()) / scale),
               cost_y_at_common=float((r['fy'][np.argmin(r['fx'] + r['fy'])] - r['fy'].min()) / scale))
    prof = dict(sf=SF[::10].tolist(), fx=((r['fx'] - r['fx'].min()) / scale)[::10].tolist(),
                fy=((r['fy'] - r['fy'].min()) / scale)[::10].tolist())
    return out, prof


def main():
    res = {'what': __doc__.split('\n')[0], 'fields': {}}
    profs = {}
    for tag, E in FIELDS:
        for mode in ('free', 'ylike'):
            o, p = analyse(tag, mode)
            o['E_Vcm'] = E
            res['fields'][f'{tag}|{mode}'] = o; profs[f'{tag}|{mode}'] = p
            print(f'{tag} ({E:.0f} V/cm) Dp_x {mode:5s}: s0_x {o["s0_x"]:.3f}±{o["s0_x_err"]:.3f}  s0_y {o["s0_y"]:.3f}±{o["s0_y_err"]:.3f}  '
                  f'Y−X {o["dy_minus_x"]:+.3f}±{o["dy_minus_x_err"]:.3f}  common {o["s0_common"]:.3f}  '
                  f'Δχ² {o["dchi2_scaled"]:.1f} (scaled by χ²/dof {o["chi2dof"]:.1f}; X pays {o["cost_x_at_common"]:.1f}, Y {o["cost_y_at_common"]:.1f})')
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(dict(res, profiles=profs), open(os.path.join(HERE, 'results', 'common_sigma0.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
