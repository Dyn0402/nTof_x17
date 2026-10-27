#!/usr/bin/env python3
"""fc_headon_grid.py -- item 5: per-event chi2 grids of the prompt footprint on head-on tracks, BOTH views with the
same width law and a view-specific readout term, for the SPS det4 RAW head-on runs and the five bench chambers.

Physics being tested (FOLLOWUPS item 5).  The avalanche (~25 um across a 150 um gap) and the drift diffusion are
isotropic; both views read the SAME pads (0.68 mm on a 0.78 mm pitch, layer L4, 70 um under the resistive strips;
each pad is routed to an X or a Y strip by a via -- docs/detector_spec/spec.py), so there is no readout-depth
difference between the views.  The only anisotropic element is the resistive layer: strips 550 um wide on 800 um
running along the coordinate the Y view measures.  Landed charge spreads in the resistive sheet by RC diffusion
(sigma^2 += 2 D s, D = 1/(R_s C)) in BOTH directions, but across the strips (the X view) it stops at the strip edges.
So the physical model is ONE sigma0 and ONE Dp for both views, ONE sheet D, with Y spreading without bound and X
bounded by the strip width.  This script measures, per event, chi2 on a grid of

    sigma0 (common law, applied to the view being fitted)  x  Dp  x  D_view

where D_view is rc_D_y for Y (the production RC kernel) and, for X, a BOUNDED RC spread
sigma^2(s) = sigma0^2 + Dp^2 u + b^2 (1 - exp(-2 D s / b^2)), b = 0.55 mm / sqrt(12) (the strip box), installed for the
fit as a study override of wft.model.build_matrix_rc in this process only (no wft default changes).  D_x = 0 is the
production X model.  --xasym A adds the one-sided delayed copy (rc_asym_x, cloud_basics §42) at every grid point.

Per event: one free (p0, w, t0) production fit at a reference hyper point, then the grid at fixed (p0, w), t0
re-optimised over {-5, 0, +5} ns (as ws_sps/headon_sigma.py).  Output: npz with per-event grids.

    fc_headon_grid.py --kind sps --cache sps_run71_raw700.pkl --bundle <det4 bundle> --health health.json --out o.npz
    fc_headon_grid.py --kind bench --cache <w12 it1 cache.pkl> --bundle <calib_bundle> --out o.npz [--tanmax 0.03]
    [--chunk I N] [--nev 600] [--jobs 8] [--xasym 0.13]
"""
import argparse, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)

S0 = np.round(np.arange(0.10, 0.501, 0.04), 3)                    # 11
DP_SPS = np.round(np.arange(0.0, 0.02001, 0.0025), 5)               # 9  (Magboltz beam gas 0.005)
DP_BENCH = np.round(np.arange(0.010, 0.03401, 0.003), 5)            # 9  (Magboltz Ar/iso 95/5 ~0.022)
DV = np.array([0.0, 1.0e-4, 2.0e-4, 3.0e-4, 4.5e-4, 6.5e-4])       # 6, rc_D_y (Y) / bounded D_x (X) [mm^2/ns]
PKMAX_BENCH = 3200.0                                                 # bench lead-strip peak cap [ADC] (--pkmax)
BOX_W = 0.55                                                         # resistive strip width [mm] (spec.py: 'open')
REF_SPS = dict(sigma_p0_x=0.29, sigma_p0_y=0.32, Dp=0.008, rc_D_y=2.2e-4)
_EV = None; _CAL = None; _KIND = None; _XASYM = 0.0


def _bounded_rc(plane, pos, p0, w, t0, hyper):
    """build_matrix_rc with an optional bounded RC spread on X (hyper 'rcb_D_x', 'rcb_b2_x'); otherwise production."""
    from wft import model as wm
    Db = float(hyper.get(f'rcb_D_{plane}', 0.0))
    if Db <= 0:
        return wm._build_matrix_rc_prod(plane, pos, p0, w, t0, hyper)
    b2 = float(hyper.get(f'rcb_b2_{plane}', BOX_W ** 2 / 12.0))
    h0 = dict(hyper); h0[f'rc_D_{plane}'] = 0.0; h0.pop(f'rcb_D_{plane}', None)
    # prompt part + any asymmetric copy from the production code (D = 0 there), then add the bounded increments
    M = wm._build_matrix_rc_prod(plane, pos, p0, w, t0, h0).reshape(len(pos), wm.NSAMP, wm.K)
    tp = 'x' if hyper.get('rc_tmpl', 'x') in ('x', 1, 1.0) else plane
    if wm.RC_T0_INTERP:
        t0a = np.floor(t0 / wm.T0_STEP) * wm.T0_STEP; f = (t0 - t0a) / wm.T0_STEP
        Ha = wm._rc_tensors(plane, t0a, tp)
        Hs = Ha if f < 1e-9 else (1.0 - f) * Ha + f * wm._rc_tensors(plane, t0a + wm.T0_STEP, tp)
    else:
        Hs = wm._rc_tensors(plane, round(t0 / wm.T0_STEP) * wm.T0_STEP, tp)
    ua = np.arange(wm.K) * wm.DT
    pa, pb = p0 + w * ua, p0 + w * (ua + wm.DT)
    pc = 0.5 * (pa + pb); half = 0.5 * np.abs(pb - pa)
    s0 = float(hyper.get(f'sigma_p0_{plane}', hyper['sigma_p0']))
    var0 = s0 ** 2 + hyper['Dp'] ** 2 * wm.UK + half ** 2 / 3.0
    Fprev = wm._fractions_sig(pos, pc, np.sqrt(var0))
    for j, s in enumerate(wm.RC_EDGES[1:]):
        Fj = wm._fractions_sig(pos, pc, np.sqrt(var0 + b2 * (1.0 - np.exp(-2.0 * Db * s / b2))))
        M += np.einsum('ik,tk->itk', Fj - Fprev, Hs[j + 1])
        Fprev = Fj
    return M.reshape(len(pos) * wm.NSAMP, wm.K)


def _install():
    from wft import model as wm
    if not hasattr(wm, '_build_matrix_rc_prod'):
        wm._build_matrix_rc_prod = wm.build_matrix_rc
        wm.build_matrix_rc = _bounded_rc


def _init(cache, bundle, kind, xasym):
    global _EV, _CAL, _KIND, _XASYM
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _install()
    _EV = pickle.load(open(cache, 'rb'))
    cal = CalibrationBundle.load(bundle)
    for k in [k for k in cal.hyper if k.startswith('sigma_p0_prof')]:
        cal.hyper[k] = []
    if kind == 'sps':
        cal.dead, cal.hot = {}, {}                  # bench channel numbering does not apply at H4 (F12)
        cal.hyper.update(REF_SPS)
        wm.ZS_CENSOR = True                         # RAW drops are stored as exact zeros: censored (ws_sps)
        wm.use_calibration(cal); wm.set_depth_bins(60); wm.set_nsamp(64)
    else:
        wm.use_calibration(cal); wm.set_depth_bins(int(cal.n_depth_bins))
    wm.COL_GUARD = 0.01
    _CAL, _KIND, _XASYM = cal, kind, xasym


def select_sps(e, v, bad):
    W = np.asarray(e[v]['W']); ch = np.asarray(e[v]['ch'])
    if W.shape[0] != 9:
        return False
    pk = W.max(1); c = int(np.argmax(pk))
    if abs(c - 4) > 1 or not (250 <= pk[c] <= 3200):
        return False
    return not np.isin(ch[max(c - 3, 0):c + 4], bad).any()


def select_bench(e, v, dead, tanmax):
    if v not in e or not np.isfinite(e.get(f'tan_{v}', np.nan)) or abs(e[f'tan_{v}']) > tanmax:
        return False
    W = np.asarray(e[v]['W']); ch = np.asarray(e[v]['ch'])
    pk = W.max(1); c = int(np.argmax(pk))
    if c < 3 or c > W.shape[0] - 4 or not (250 <= pk[c] <= PKMAX_BENCH) or W.max() >= 3550:
        return False
    return not np.isin(ch[c - 3:c + 4], dead).any()


def one(args):
    eid, views = args
    from wft import model as wm
    e = _EV[eid]; out = {'eid': int(eid)}
    DP = DP_SPS if _KIND == 'sps' else DP_BENCH
    for v in views:
        P = dict(e[v]); P['W'] = np.asarray(P['W'], float)
        if P['W'].shape[1] != wm.NSAMP:
            wm.set_nsamp(P['W'].shape[1])
        hy = dict(_CAL.hyper)
        if _XASYM and v == 'x':
            hy['rc_asym_x'] = float(_XASYM)
        p0i, _, t0i = wm.init_guess(P, v)
        try:
            r = wm.fit_plane_raw(P, v, p0i, 0.0, t0i, hyper=hy)
        except Exception:
            continue
        if not np.isfinite(r['chi2']):
            continue
        Wp, noise, pos, sat = wm.prep_plane(P, v)
        G = np.full((len(S0), len(DP), len(DV)), np.inf)
        for i, s in enumerate(S0):
            for j, dp in enumerate(DP):
                for k, d in enumerate(DV):
                    h = dict(hy); h[f'sigma_p0_{v}'] = float(s); h['Dp'] = float(dp)
                    if v == 'y':
                        h['rc_D_y'] = float(d)
                    else:
                        h['rc_D_x'] = 0.0; h['rcb_D_x'] = float(d)
                    best = np.inf
                    for dt in (-5.0, 0.0, 5.0):
                        c, _ = wm.chi2_plane(v, Wp, noise, pos, sat, r['p0'], r['w'], r['t0'] + dt, h)
                        best = min(best, c)
                    G[i, j, k] = best
        out[v] = dict(G=G.astype(np.float32), p0=r['p0'], w=r['w'], t0=r['t0'], chi2=r['chi2'], dof=r['dof'],
                      amax=float(np.asarray(e[v]['W']).max()), qsum=float(np.sum(r['q'])),
                      tan_ref=float(e.get(f'tan_{v}', np.nan)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['sps', 'bench'], required=True)
    ap.add_argument('--cache', required=True); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--health', default=None, help='SPS: ws_sps health.json (flagged channels)')
    ap.add_argument('--tanmax', type=float, default=0.03); ap.add_argument('--nev', type=int, default=600)
    ap.add_argument('--chunk', type=int, nargs=2, default=None); ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--xasym', type=float, default=0.0); ap.add_argument('--views', default='xy')
    ap.add_argument('--pkmax', type=float, default=3200.0, help='bench: lead-strip peak cap [ADC]')
    a = ap.parse_args()
    global PKMAX_BENCH
    PKMAX_BENCH = a.pkmax
    E = pickle.load(open(a.cache, 'rb'))
    if a.kind == 'sps':
        H = json.load(open(a.health)); sel = lambda e, v: select_sps(e, v, H[v]['flagged'])
    else:
        b = json.load(open(os.path.join(a.bundle, 'bundle.json')))
        dead = {p: list(b.get('dead', {}).get(p, [])) for p in 'xy'}
        sel = lambda e, v: select_bench(e, v, dead[v], a.tanmax)
    tasks = []; n = {'x': 0, 'y': 0}
    for eid in sorted(E):
        vs = [v for v in a.views if n[v] < a.nev and sel(E[eid], v)]
        if vs:
            tasks.append((eid, vs))
            for v in vs:
                n[v] += 1
    del E
    if a.chunk:
        tasks = tasks[a.chunk[0]::a.chunk[1]]
    print(f'{a.kind} {os.path.basename(a.cache)}: selected x {n["x"]}, y {n["y"]}; this job {len(tasks)} events', flush=True)
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.cache, a.bundle, a.kind, a.xasym)) as pool:
        R = list(pool.map(one, tasks, chunksize=2))
    DP = DP_SPS if a.kind == 'sps' else DP_BENCH
    pickle.dump(dict(S0=S0, DP=DP, DV=DV, box_w=BOX_W, kind=a.kind, xasym=a.xasym, args=vars(a), rows=R),
                open(a.out, 'wb'), protocol=4)
    print('wrote', a.out, len(R), flush=True)


if __name__ == '__main__':
    main()
