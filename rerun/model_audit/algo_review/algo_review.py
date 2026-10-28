#!/usr/bin/env python3
"""Algorithm review (2026-10-28): is the grid + Nelder-Mead search over (p0, w, t0) the right outer solver for
the NNLS-profiled forward model, and how degenerate is the 60 ns lattice shift?

Arms, per plane of each det3 bench event (M3 reference available in the cache):
  prod   wft.reco.fit_plane as shipped (pivot scan -> fit_plane_raw grid + Nelder-Mead -> 1-D curvature errors)
  lm     same pivot scan for the global start, then a Levenberg-Marquardt refinement of (p0, w, t0) with the
         variable-projection (Kaufman) Jacobian of the NNLS-profiled residual; full 3x3 covariance for free
  shift  the prod minimum re-solved at (p0 - 60 w, w, t0 - 60) and (p0 + 60 w, w, t0 + 60): the lattice symmetry
  kgap   prod with the depth grid cut to the drift gap (K = ceil(T_gap / 60 ns)) instead of the bundle's K
"""
import os, sys, json, time, pickle, argparse
import numpy as np
from multiprocessing import Pool

REPO = '/home/dylan/PycharmProjects/nTof_x17'
sys.path.insert(0, REPO)
CACHE = '/media/dylan/data/x17/cosmic_bench/condor_plane_ratio/inputs/det3/big_cache.pkl'
BUNDLE = '/media/dylan/data/x17/cosmic_bench/condor_campaign_r06/bundles/mx17_3/calib_bundle_r06'
ARM = os.path.join(REPO, 'rerun/precheck_rcv/out/arm_det3_rcvk24.json')

_EV = None
_K = None
_KGAP = None
_NSOLVE = [0]


def _init(k_gap):
    global _EV, _K, _KGAP
    from wft.calib import CalibrationBundle
    from wft import model as wm, reco as wr
    _EV = pickle.load(open(CACHE, 'rb'))
    cal = CalibrationBundle.load(BUNDLE)
    arm = json.load(open(ARM))
    cal.hyper.update(arm['hyper'])
    cal.t0_prior_sigma = 0.0
    wm.use_calibration(cal)
    _K = int(arm['depth_bins'])
    _KGAP = int(k_gap)
    wm.set_depth_bins(_K)
    wr.PIVOT_START = True
    wr.TWO_SIDED = False
    # count NNLS solves
    orig = wm._solve_nnls

    def counted(*a, **kw):
        _NSOLVE[0] += 1
        return orig(*a, **kw)
    wm._solve_nnls = counted


# ------------------------------------------------------------------ LM / VarPro
def _chi(plane, W, noise, pos, sat, th, hyper):
    from wft import model as wm
    return wm.chi2_plane(plane, W, noise, pos, sat, th[0], th[1], th[2], hyper, snap_t0=False)


def _resid_jac(plane, W, noise, pos, sat, th, q, hyper, h=(0.02, 1e-4, 1.0)):
    """Weighted residual and the Kaufman (variable-projection) Jacobian at theta, q = NNLS solution there."""
    from wft import model as wm
    Wt = np.repeat(1.0 / noise, wm.NSAMP)
    ok = ~sat.reshape(-1)
    M = wm.build_matrix(plane, pos, th[0], th[1], th[2], hyper)
    y = W.reshape(-1)
    m = M @ q
    r = (m - y) * Wt
    pen_on = sat.reshape(-1) & (m < y)            # one-sided penalty rows active
    cols = []
    for j in range(3):
        tp, tm = np.array(th, float), np.array(th, float)
        tp[j] += h[j]; tm[j] -= h[j]
        dM = (wm.build_matrix(plane, pos, *tp, hyper) - wm.build_matrix(plane, pos, *tm, hyper)) / (2 * h[j])
        cols.append((dM @ q) * Wt)
    J = np.stack(cols, 1)
    F = q > 0
    if F.any():
        AF = (M * Wt[:, None])[ok][:, F]
        Qf, _ = np.linalg.qr(AF)
        J_ok = J[ok] - Qf @ (Qf.T @ J[ok])
    else:
        J_ok = J[ok]
    rows_r = np.concatenate([r[ok], r[pen_on]])
    rows_J = np.vstack([J_ok, J[pen_on]])
    return rows_r, rows_J


def lm_refine(plane, W, noise, pos, sat, th0, hyper, maxit=40):
    """Levenberg-Marquardt on the NNLS-profiled chi2. Returns (chi2, theta, cov, n_iter)."""
    th = np.array(th0, float)
    chi, q = _chi(plane, W, noise, pos, sat, th, hyper)
    if not np.isfinite(chi) or q is None:
        return np.inf, th, None, 0
    lam = 1e-2
    n_it = 0
    H = None
    for n_it in range(1, maxit + 1):
        r, J = _resid_jac(plane, W, noise, pos, sat, th, q, hyper)
        g = J.T @ r
        H = J.T @ J
        D = np.diag(np.maximum(np.diag(H), 1e-12))
        accepted = False
        for _ in range(8):
            try:
                step = -np.linalg.solve(H + lam * D, g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            c2, q2 = _chi(plane, W, noise, pos, sat, th + step, hyper)
            if np.isfinite(c2) and q2 is not None and c2 < chi:
                dchi = chi - c2
                th, chi, q = th + step, c2, q2
                lam = max(lam / 3, 1e-6)
                accepted = True
                break
            lam *= 10
        if not accepted:
            break
        if (abs(step[0]) < 2e-4 and abs(step[1]) < 2e-6 and abs(step[2]) < 0.2) or dchi < 0.05:
            break
    cov = None
    if H is not None:
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = None
    return float(chi), th, cov, n_it


def lm_fit(P, plane, hyper):
    """Pivot scan (as production) for the start, LM for the refinement of the best PIVOT_NREF wells."""
    from wft import model as wm, reco as wr
    W, noise, pos, sat = wm.prep_plane(P, plane)
    c = wr.charge_centroid(W, noise, pos, plane, hyper)
    if c is None:
        return None
    xc, tc = c
    _p0, _w, t0 = wm.init_guess(P, plane)
    ws = np.arange(-wr.PIVOT_W_HALF, wr.PIVOT_W_HALF + 1e-12, wr.PIVOT_W_STEP)
    chis = np.array([wm.chi2_plane(plane, W, noise, pos, sat, xc - w * (tc - t0), w, t0, hyper)[0] for w in ws])
    n = len(ws)
    wells = [i for i in range(n) if (i == 0 or chis[i] <= chis[i - 1]) and (i == n - 1 or chis[i] <= chis[i + 1])]
    best = None
    for i in sorted(wells, key=lambda i: chis[i])[:wr.PIVOT_NREF]:
        w = float(ws[i])
        # the same coarse t0 ladder fit_plane_raw scans (7 points, 40 ns), along the pivot line
        t0s = t0 + np.arange(-120.0, 121.0, 40.0)
        ct = [wm.chi2_plane(plane, W, noise, pos, sat, xc - w * (tc - tt), w, tt, hyper)[0] for tt in t0s]
        tb = float(t0s[int(np.argmin(ct))])
        th0 = (xc - w * (tc - tb), w, tb)
        chi, th, cov, nit = lm_refine(plane, W, noise, pos, sat, th0, hyper)
        if np.isfinite(chi) and (best is None or chi < best[0]):
            best = (chi, th, cov, nit)
    if best is None:
        return None
    chi, th, cov, nit = best
    dof = int((~sat).sum())
    scale = max(chi / max(dof, 1), 1.0)
    err = np.sqrt(np.maximum(np.diag(cov), 0) * scale) if cov is not None else np.full(3, np.nan)
    rho = cov[0, 2] / np.sqrt(cov[0, 0] * cov[2, 2]) if cov is not None and cov[0, 0] > 0 and cov[2, 2] > 0 else np.nan
    return dict(p0=float(th[0]), w=float(th[1]), t0=float(th[2]), chi2=chi, dof=dof,
                p0_err=float(err[0]), w_err=float(err[1]), t0_err=float(err[2]), rho_p0t0=float(rho), nit=nit)


def one(eid):
    from wft import model as wm, reco as wr
    cal = wm.CAL
    hyper = dict(cal.hyper)
    ev = _EV[eid]
    out = []
    for plane in 'xy':
        if plane not in ev:
            continue
        P = ev[plane]
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        row = dict(eid=int(eid), plane=plane, tan_ref=float(ev[f'tan_{plane}']),
                   ref_mesh=float(ev.get(f'ref_mesh_{plane}', np.nan)), ftst=ev.get(f'ftst_{plane}'))
        # ---- prod
        wm.set_depth_bins(_K)
        _NSOLVE[0] = 0; t = time.perf_counter()
        try:
            f = wr.fit_plane(P, plane, cal)
        except Exception:
            f = None
        row['prod_time'] = time.perf_counter() - t; row['prod_nsolve'] = _NSOLVE[0]
        if f is None:
            out.append(row); continue
        for k in ('p0', 'w', 't0', 'chi2', 'dof', 'p0_err', 'w_err', 't0_err', 'q_sum', 'q_uend'):
            row[f'prod_{k}'] = float(getattr(f, k))
        # ---- shift (lattice symmetry), re-solved and LM-refined in the shifted basin
        Wp, noise, pos, sat = wm.prep_plane(P, plane)
        best_shift = np.inf; best_shift_ref = (np.inf, np.nan, np.nan)
        for s in (-1.0, 1.0):
            th = (f.p0 + s * 60.0 * f.w, f.w, f.t0 + s * 60.0)
            c, _ = _chi(plane, Wp, noise, pos, sat, th, hyper)
            best_shift = min(best_shift, c)
            c2, th2, _cov, _n = lm_refine(plane, Wp, noise, pos, sat, th, hyper, maxit=15)
            if c2 < best_shift_ref[0]:
                best_shift_ref = (c2, float(th2[2]), float(th2[0]))
        row['shift_chi2'] = float(best_shift); row['shift_chi2_ref'] = float(best_shift_ref[0])
        row['shift_t0_ref'] = best_shift_ref[1]; row['shift_p0_ref'] = best_shift_ref[2]
        # ---- lm
        _NSOLVE[0] = 0; t = time.perf_counter()
        try:
            g = lm_fit(P, plane, hyper)
        except Exception:
            g = None
        row['lm_time'] = time.perf_counter() - t; row['lm_nsolve'] = _NSOLVE[0]
        if g is not None:
            for k, v in g.items():
                row[f'lm_{k}'] = float(v)
        # ---- kgap
        wm.set_depth_bins(_KGAP)
        _NSOLVE[0] = 0; t = time.perf_counter()
        try:
            fk = wr.fit_plane(P, plane, cal)
        except Exception:
            fk = None
        row['kgap_time'] = time.perf_counter() - t; row['kgap_nsolve'] = _NSOLVE[0]
        if fk is not None:
            for k in ('p0', 'w', 't0', 'chi2', 'dof', 'q_sum', 'q_uend'):
                row[f'kgap_{k}'] = float(getattr(fk, k))
        wm.set_depth_bins(_K)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--kgap', type=int, default=13)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'algo_review_rows.pkl'))
    a = ap.parse_args()
    C = pickle.load(open(CACHE, 'rb'))
    eids = sorted(C)[:a.n]
    del C
    t = time.time()
    with Pool(a.jobs, initializer=_init, initargs=(a.kgap,)) as pool:
        rows = [r for rr in pool.imap_unordered(one, eids, chunksize=4) for r in rr]
    print(f'{len(rows)} plane rows from {len(eids)} events in {time.time() - t:.0f} s')
    pickle.dump(rows, open(a.out, 'wb'))


if __name__ == '__main__':
    main()
