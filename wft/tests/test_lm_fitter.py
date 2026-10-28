#!/usr/bin/env python3
"""
FITTER = 'lm' (variable projection + Levenberg-Marquardt, branch lm-varpro): the analytic d(Mq)/d(p0, w, t0) of the
RC kernel matches central differences of build_matrix_rc on every kernel branch, the residual vector reproduces
_solve_nnls's chi2, and on real det3 bench planes the LM fit lands on the grid + Nelder-Mead minimum (same w within
1 um/ns; chi2 not materially above NM -- it may be below, LM_HOPS).

    ../../.venv/bin/python wft/tests/test_lm_fitter.py [n_events]
"""
import os
import sys
import pickle

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft.calib import CalibrationBundle              # noqa: E402
from wft import model as wm, reco as wr              # noqa: E402

VL = '/media/dylan/data/x17/cosmic_bench/vladder/det3'
BUNDLE = os.path.join(VL, 'bundle_phys_mask')
CACHE = os.path.join(VL, 'cache.pkl')
HAVE_DATA = os.path.exists(os.path.join(BUNDLE, 'bundle.json')) and os.path.exists(CACHE)


def _setup():
    cal = CalibrationBundle.load(BUNDLE)
    wm.use_calibration(cal)
    return cal, pickle.load(open(CACHE, 'rb'))


def _fd_dmq(plane, pos, th, hyper, q, h=(1e-4, 1e-7, 0.25)):
    out = []
    for j in range(3):
        tp, tm = np.array(th, float), np.array(th, float)
        tp[j] += h[j]
        tm[j] -= h[j]
        out.append((wm.build_matrix(plane, pos, *tp, hyper) @ q - wm.build_matrix(plane, pos, *tm, hyper) @ q)
                   / (2 * h[j]))
    return np.stack(out, 1)


def check_jacobian(cal, C):
    """Analytic vs FD on the bundle's kernel and on the Lorentzian / one-sided-copy / drain branches."""
    ev = C[sorted(C)[5]]
    worst = 0.0
    variants = [{}, {'lor_frac_x': 0.2, 'lor_gamma_x': 0.3, 'lor_frac_y': 0.2, 'lor_gamma_y': 0.3},
                {'rc_asym_x': 0.13, 'rc_asym_y': -0.1}, {'rc_drain': 400.0}, {'rc_D_x': 2e-4}]
    for extra in variants:
        hyper = dict(cal.hyper, **extra)
        for plane in 'xy':
            W, noise, pos, sat = wm.prep_plane(ev[plane], plane)
            p0, _w, t0 = wm.init_guess(ev[plane], plane)
            # off the 5 ns nodes so the interpolated t0 derivative is the one-sided slope on both sides
            th = (p0 + 0.13, 0.004, t0 + 1.7)
            _c, q = wm.chi2_plane(plane, W, noise, pos, sat, *th, hyper, snap_t0=False)
            Ja = wm._rc_dmq(plane, pos, *th, hyper, q)
            Jn = _fd_dmq(plane, pos, th, hyper, q)
            for j in range(3):
                rel = np.abs(Ja[:, j] - Jn[:, j]).max() / max(np.abs(Jn[:, j]).max(), 1e-30)
                worst = max(worst, rel)
                assert rel < 2e-3, (extra, plane, j, rel)
    print(f'  analytic Jacobian vs FD: worst relative error {worst:.1e} over {len(variants)} kernel variants')


def check_residual(cal, C):
    """r @ r reproduces chi2_plane (prior included) on planes with and without clipped samples."""
    n = 0
    for eid in sorted(C)[:60]:
        ev = C[eid]
        for plane in 'xy':
            W, noise, pos, sat = wm.prep_plane(ev[plane], plane)
            p0, w, t0 = wm.init_guess(ev[plane], plane)
            prior = wr.t0_prior_for(cal, plane, ev.get(f'ftst_{plane}'))
            th = (p0, w, t0 + 2.5)
            chi, q, M = wm._vp_eval(plane, W, noise, pos, sat, th, cal.hyper, True, prior)
            if q is None:
                continue
            r, J = wm._vp_resid_jac(plane, W, noise, pos, sat, th, q, M, cal.hyper, True, prior)
            assert abs(r @ r - chi) <= 1e-6 * chi + 1e-6, (eid, plane, r @ r, chi)
            assert J.shape == (len(r), 3)
            n += 1
    print(f'  residual reproduces chi2 on {n} planes')


def check_fit(cal, C, n_events):
    """LM vs NM through the production entry point (wr.fit_plane: pivot start, prior, errors)."""
    rows = []
    for eid in sorted(C)[:n_events]:
        ev = C[eid]
        for plane in 'xy':
            prior = wr.t0_prior_for(cal, plane, ev.get(f'ftst_{plane}'))
            wm.FITTER = 'nm'
            a = wr.fit_plane(ev[plane], plane, cal, t0_prior=prior)
            wm.FITTER = 'lm'
            b = wr.fit_plane(ev[plane], plane, cal, t0_prior=prior)
            wm.FITTER = 'nm'
            if a is None or b is None:
                continue
            rows.append((abs(a.w - b.w) * 1e3, (b.chi2 - a.chi2) / max(a.chi2, 1.0), abs(a.t0 - b.t0)))
    d = np.array(rows)
    same_w = float((d[:, 0] < 1.0).mean())
    # LM may end LOWER (a deeper lattice basin, LM_HOPS); it must not end materially higher
    worse = float((d[:, 1] > 1e-3).mean())
    print(f'  {len(d)} planes: same w (1 um/ns) {same_w:.1%}, chi2 > NM by > 0.1 % on {worse:.1%}, '
          f'lower by > 0.1 % on {(d[:, 1] < -1e-3).mean():.1%}, median |dt0| {np.median(d[:, 2]):.2f} ns')
    assert same_w >= 0.90 and worse <= 0.05


def main(n_events=15):
    if not HAVE_DATA:
        print('SKIP: no det3 vladder cache/bundle on this machine')
        return 0
    cal, C = _setup()
    check_jacobian(cal, C)
    check_residual(cal, C)
    check_fit(cal, C, n_events)
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main(*(int(a) for a in sys.argv[1:])))
