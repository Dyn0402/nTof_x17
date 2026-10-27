#!/usr/bin/env python3
"""analyse_fc_headon.py [--dir DIR] -- item 5: one prompt footprint or two?  Nested width-law models on the per-event
chi2 grids of fc_headon_grid.py (SPS det4 run_71 RAW at three fields; bench head-on on five chambers).

Each view's summed chi2 S_v(sigma0, Dp, D_v) (D_y = the Y RC spread, D_x = the BOUNDED across-strip RC spread, D_x = 0
is the production X model).  Models, from the production one to the fully physical one:

  prod      X without RC (D_x = 0), sigma0 / Dp per view, D_y free           (ws_sps's model)
  prod_c    prod with ONE sigma0                                              (the question as first asked)
  free      sigma0 / Dp / D per view, X with bounded RC
  c_s       one sigma0;                      Dp, D per view
  c_sdp     one sigma0, one Dp (isotropic avalanche + gas);  D per view
  c_all     one sigma0, one Dp, one sheet D (X bounded by the strip, Y not)  -- the physical model, 3 parameters

Reported per model: parameters, summed chi2 difference to 'free' (raw, relative to the total, and scaled by chi2/dof),
and an event bootstrap (events are the independent unit; chi2 is model-limited) of the parameters and of the
X - Y differences.  Writes results/fc_headon.json.
"""
import argparse, glob, json, os, pickle
import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.expanduser('~/x17/cosmic_bench/model_audit/fc/headon')
TAGS = ['sps_raw700', 'sps_raw450', 'sps_raw275', 'sps_raw700_asym',
        'bench_det2', 'bench_det3', 'bench_det4', 'bench_det6', 'bench_det7',
        'bench05_det2', 'bench05_det3', 'bench05_det4', 'bench05_det6', 'bench05_det7']


def load(d, tag):
    fs = sorted(glob.glob(os.path.join(d, f'{tag}_c*.pkl')))
    if not fs:
        return None
    rows = []; meta = None
    for f in fs:
        D = pickle.load(open(f, 'rb')); rows += D['rows']; meta = D
    out = dict(S0=np.asarray(meta['S0']), DP=np.asarray(meta['DP']), DV=np.asarray(meta['DV']), nfiles=len(fs))
    for v in 'xy':
        R = [r[v] for r in rows if v in r and np.all(np.isfinite(r[v]['G']))]
        if R:
            out[v] = dict(G=np.stack([r['G'] for r in R]).astype(float), dof=np.array([r['dof'] for r in R]),
                          amax=np.array([r['amax'] for r in R]))
    return out


def refine(x, y):
    """minimum of a 1-D profile: cubic spline on a fine grid (grid endpoints allowed)."""
    xf = np.linspace(x[0], x[-1], 401); yf = CubicSpline(x, y)(xf)
    i = int(np.argmin(yf)); return float(xf[i]), float(yf[i])


def fit_models(Gx, Gy, S0, DP, DV):
    Sx, Sy = Gx.sum(0), Gy.sum(0)
    out = {}
    # per-view free (X bounded RC) and production (X D = 0)
    def view_best(S):
        i, j, k = np.unravel_index(np.argmin(S), S.shape)
        s, c = refine(S0, S.min(axis=(1, 2)))
        return dict(s0=s, dp=float(DP[j]), d=float(DV[k]), chi=c)
    bx, by = view_best(Sx), view_best(Sy)
    bx0 = view_best(Sx[:, :, :1])
    out['free'] = dict(s0_x=bx['s0'], s0_y=by['s0'], dp_x=bx['dp'], dp_y=by['dp'], d_x=bx['d'], d_y=by['d'], chi=bx['chi'] + by['chi'])
    out['prod'] = dict(s0_x=bx0['s0'], s0_y=by['s0'], dp_x=bx0['dp'], dp_y=by['dp'], d_x=0.0, d_y=by['d'], chi=bx0['chi'] + by['chi'])
    # common sigma0
    for name, SX in (('c_s', Sx), ('prod_c', Sx[:, :, :1])):
        s, c = refine(S0, SX.min(axis=(1, 2)) + Sy.min(axis=(1, 2)))
        out[name] = dict(s0=s, chi=c)
    # common sigma0 + Dp
    T = Sx.min(axis=2) + Sy.min(axis=2)                       # (s, dp)
    i, j = np.unravel_index(np.argmin(T), T.shape)
    s, c = refine(S0, T[:, j])
    out['c_sdp'] = dict(s0=s, dp=float(DP[j]), d_x=float(DV[np.argmin(Sx[i, j])]), d_y=float(DV[np.argmin(Sy[i, j])]),
                        chi=min(c, float(T.min())))
    T3 = Sx + Sy
    i, j, k = np.unravel_index(np.argmin(T3), T3.shape)
    s, c = refine(S0, T3[:, j, k])
    out['c_all'] = dict(s0=s, dp=float(DP[j]), d=float(DV[k]), chi=min(c, float(T3.min())))
    return out


def analyse(D, nboot=300):
    S0, DP, DV = D['S0'], D['DP'], D['DV']
    Gx, Gy = D['x']['G'], D['y']['G']
    M = fit_models(Gx, Gy, S0, DP, DV)
    dof = D['x']['dof'].sum() + D['y']['dof'].sum()
    S = M['free']['chi']; scale = S / dof
    for k, m in M.items():
        m['dchi'] = m['chi'] - S; m['dchi_rel_pct'] = 100 * m['dchi'] / S; m['dchi_scaled'] = m['dchi'] / scale
    rng = np.random.default_rng(11); B = []
    for _ in range(nboot):
        bx = rng.integers(0, len(Gx), len(Gx)); by = rng.integers(0, len(Gy), len(Gy))
        m = fit_models(Gx[bx], Gy[by], S0, DP, DV)
        B.append([m['free']['s0_x'], m['free']['s0_y'], m['prod']['s0_x'], m['prod']['s0_y'], m['c_s']['s0'],
                  m['c_sdp']['s0'], m['c_all']['s0'], m['free']['dp_x'], m['free']['dp_y'], m['prod']['dp_x'],
                  m['free']['d_x'], m['free']['d_y'], m['c_all']['dp'], m['c_all']['d']])
    B = np.array(B)
    names = ['free_s0_x', 'free_s0_y', 'prod_s0_x', 'prod_s0_y', 'c_s_s0', 'c_sdp_s0', 'c_all_s0', 'free_dp_x',
             'free_dp_y', 'prod_dp_x', 'free_d_x', 'free_d_y', 'c_all_dp', 'c_all_d']
    err = {n: float(B[:, i].std()) for i, n in enumerate(names)}
    diffs = dict(prod_y_minus_x=(B[:, 3] - B[:, 2]), free_y_minus_x=(B[:, 1] - B[:, 0]), free_dpx_minus_dpy=(B[:, 7] - B[:, 8]))
    dstat = {k: dict(mean=float(v.mean()), err=float(v.std()), p=[float(np.percentile(v, q)) for q in (2.5, 50, 97.5)]) for k, v in diffs.items()}
    return dict(n_x=int(len(Gx)), n_y=int(len(Gy)), chi2dof=float(scale), models=M, boot_err=err, diffs=dstat,
                grid=dict(S0=S0.tolist(), DP=DP.tolist(), DV=DV.tolist()),
                prof=dict(x=(Gx.sum(0).min(axis=(1, 2)) - Gx.sum(0).min()).tolist(), y=(Gy.sum(0).min(axis=(1, 2)) - Gy.sum(0).min()).tolist(),
                          x_prod=(Gx.sum(0)[:, :, 0].min(1) - Gx.sum(0)[:, :, 0].min()).tolist()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default=DIR); ap.add_argument('--nboot', type=int, default=300)
    a = ap.parse_args(); res = {}
    for tag in TAGS:
        D = load(a.dir, tag)
        if D is None or 'x' not in D:
            continue
        if 'y' not in D:                 # X-only arm (the one-sided copy): pair with the plain run's Y
            base = load(a.dir, tag.replace('_asym', ''))
            if base is None or 'y' not in base:
                continue
            D['y'] = base['y']
        r = analyse(D, a.nboot); res[tag] = r
        M, E, Df = r['models'], r['boot_err'], r['diffs']
        print(f'== {tag}: n {r["n_x"]}/{r["n_y"]}, chi2/dof {r["chi2dof"]:.1f}')
        print(f'   prod  : s0 X {M["prod"]["s0_x"]:.3f}±{E["prod_s0_x"]:.3f} Y {M["prod"]["s0_y"]:.3f}±{E["prod_s0_y"]:.3f}  '
              f'Dp X {M["prod"]["dp_x"]:.4f} Y {M["prod"]["dp_y"]:.4f}  D_y {M["prod"]["d_y"]:.1e}   Y−X {Df["prod_y_minus_x"]["mean"]:+.3f}±{Df["prod_y_minus_x"]["err"]:.3f}  dchi {M["prod"]["dchi_rel_pct"]:+.3f}%')
        print(f'   free  : s0 X {M["free"]["s0_x"]:.3f}±{E["free_s0_x"]:.3f} Y {M["free"]["s0_y"]:.3f}±{E["free_s0_y"]:.3f}  '
              f'Dp X {M["free"]["dp_x"]:.4f} Y {M["free"]["dp_y"]:.4f}  D X {M["free"]["d_x"]:.1e} Y {M["free"]["d_y"]:.1e}   '
              f'Y−X {Df["free_y_minus_x"]["mean"]:+.3f}±{Df["free_y_minus_x"]["err"]:.3f}  DpX−DpY {Df["free_dpx_minus_dpy"]["mean"]:+.4f}±{Df["free_dpx_minus_dpy"]["err"]:.4f}')
        for k in ('prod_c', 'c_s', 'c_sdp', 'c_all'):
            m = M[k]
            extra = ' '.join(f'{p} {m[p]:.4g}' for p in ('dp', 'd', 'd_x', 'd_y') if p in m)
            print(f'   {k:6s}: s0 {m["s0"]:.3f}  {extra}  dchi {m["dchi"]:.0f} = {m["dchi_rel_pct"]:+.3f}% of chi2 = {m["dchi_scaled"]:.0f} scaled')
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, 'results', 'fc_headon.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
