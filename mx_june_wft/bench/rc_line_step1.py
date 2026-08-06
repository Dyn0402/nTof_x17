#!/usr/bin/env python3
"""
rc_line_step1.py — can the continuum RC line predict the Y response from X?

Physics: charge deposited on a resistive strip (which runs along Y) diffuses
along the line, sigma(t) = pitch * sqrt(t / t_p), and drains with exp(-t/tau_g).
The charge fraction sitting over readout strip Delta at time t is

    q_D(t) = [Phi((D+1/2)p, s(t)) - Phi((D-1/2)p, s(t))] * exp(-t/tau_g)

averaged over the deposit position within the source strip. The strip output
is the electronics response T_e convolved with the induced-charge rate:

    resp_0 = T_e (x) [delta + dq_0/dt]      (starts with the full charge)
    resp_D = T_e (x) dq_D/dt                (D != 0)

Taking T_e = the measured X template (same FEU electronics; X's own sharing
and undershoot are small), fit (t_p, tau_g) so that resp_0 matches the
measured Y template. The same two numbers then PREDICT the +-1/+-2 responses
— compare them to the calibrated empirical kernel (c1*kY at tau_s, smeared).

    ../../.venv/bin/python mx_june_wft/bench/rc_line_step1.py sat_det3
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

DT = 2.0                       # fine time grid [ns]
T_MAX = 1800.0


def charge_fractions(t, t_p, tau_g, pitch=0.78, n_delta=3, n_src=9):
    """q_D(t): charge fraction over strip D, deposit averaged over the source
    strip width. t in ns; sigma(t) = pitch*sqrt(t/t_p)."""
    t = np.asarray(t, float)
    sig = pitch * np.sqrt(np.maximum(t, 1e-9) / t_p)
    y0 = (np.arange(n_src) + 0.5) / n_src * pitch - pitch / 2   # deposit offsets
    out = {}
    for D in range(0, n_delta + 1):
        lo = (D - 0.5) * pitch - y0[:, None]
        hi = (D + 0.5) * pitch - y0[:, None]
        z = 1.0 / (np.sqrt(2.0) * sig)[None, :]
        q = 0.5 * (erf(hi * z) - erf(lo * z))
        out[D] = q.mean(axis=0) * np.exp(-t / tau_g)
    return out


def responses(tmpl_grid, tmpl_x, t_p, tau_g, n_delta=3):
    """Predicted per-strip responses on the fine grid, normalised to the
    direct strip's peak."""
    tg = np.arange(0.0, T_MAX, DT)
    q = charge_fractions(tg, t_p, tau_g, n_delta=n_delta)
    # electronics template on the fine grid (defined for t >= grid start)
    te_t = np.arange(tmpl_grid[0], T_MAX, DT)
    te = np.interp(te_t, tmpl_grid, tmpl_x, left=0, right=0)
    resp = {}
    for D, qD in q.items():
        rate = np.gradient(qD, DT)                 # dq/dt
        if D == 0:
            g = rate.copy()
            g[0] += 1.0 / DT                       # + delta(t): initial charge
        else:
            g = rate
        # convolution on the fine grid; re-anchor to the template's grid start
        full = np.convolve(te, g)[:len(te)] * DT
        resp[D] = (te_t, full)
    pk = resp[0][1].max()
    return {D: (t, r / pk) for D, (t, r) in resp.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--bundle', default=None)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    cfg = get_config(args.run_key)
    bundle = args.bundle or os.path.join(cfg.OUT_BASE, 'wft', 'calib_bundle')
    cal = CalibrationBundle.load(bundle)
    grid = np.asarray(cal.grid, float)
    tx, ty = np.asarray(cal.tmpl['x'], float), np.asarray(cal.tmpl['y'], float)
    ty = ty / ty.max()
    tx = tx / tx.max()

    # fit (t_p, tau_g) so resp_0 matches the measured Y template
    fit_t = np.arange(-100.0, 1400.0, 10.0)
    ty_meas = np.interp(fit_t, grid, ty)

    def loss(v):
        t_p, tau_g = v
        if t_p < 2 or tau_g < 100:
            return 1e6
        r = responses(grid, tx, t_p, tau_g, n_delta=1)
        t0g, r0 = r[0]
        pred = np.interp(fit_t, t0g, r0, left=0, right=0)
        return float(((pred - ty_meas) ** 2).sum())

    best = None
    for t_p0 in (20.0, 60.0, 150.0, 400.0):
        for tg0 in (300.0, 1000.0, 4000.0):
            res = minimize(loss, np.array([t_p0, tg0]), method='Nelder-Mead',
                           options=dict(xatol=0.5, fatol=1e-5, maxiter=400))
            if best is None or res.fun < best.fun:
                best = res
    t_p, tau_g = best.x
    base = float(((ty_meas - np.interp(fit_t, grid, tx)) ** 2).sum())
    print(f'fit: t_p = {t_p:.1f} ns (time for sigma to reach one pitch), '
          f'tau_g = {tau_g:.0f} ns')
    print(f'residual ||pred - Y||^2 = {best.fun:.4f}  '
          f'(X-as-Y baseline {base:.4f}; improvement x{base / best.fun:.1f})')

    r = responses(grid, tx, t_p, tau_g, n_delta=3)
    # compare the predicted +-1 response to the calibrated empirical copy
    h = cal.hyper
    from scipy.ndimage import gaussian_filter1d
    sm = gaussian_filter1d(ty, max(h['sigma_s'], 1.0) / (grid[1] - grid[0]))
    emp_t = grid + h['tau_s']
    c1y = h['c1'] * h.get('kY', 1.0)
    c2y = h['c2'] * h.get('kY', 1.0)

    def peak_area(t, y):
        return float(np.max(y)), float(np.trapz(np.clip(y, 0, None), t))

    p0 = peak_area(*r[0])
    for D in (1, 2, 3):
        pk, ar = peak_area(*r[D])
        tpk = r[D][0][int(np.argmax(r[D][1]))]
        print(f'pred D={D}: rel peak {pk:.3f}  rel area {ar / p0[1]:.3f}  '
              f'peak time {tpk:.0f} ns')
    print(f'empirical  +-1: c1*kY = {c1y:.3f} at tau_s = {h["tau_s"]:.0f} ns '
          f'(smear {h["sigma_s"]:.0f});  +-2: {c2y:.3f} at 2tau')

    # dump curves for plotting / later use
    out = os.path.join(cfg.OUT_BASE, 'wft', 'rc_line_step1.npz')
    np.savez(out, grid=grid, tmpl_x=tx, tmpl_y=ty, t_p=t_p, tau_g=tau_g,
             **{f'resp{D}_t': r[D][0] for D in r},
             **{f'resp{D}': r[D][1] for D in r})
    print('wrote', out)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].plot(grid, tx, 'k--', lw=1, label='measured X (= T_e)')
    axs[0].plot(grid, ty, 'b', lw=2, label='measured Y')
    axs[0].plot(*r[0], 'r', lw=1.5,
                label=f'RC pred (t_p={t_p:.0f}, tau_g={tau_g:.0f})')
    axs[0].set_xlim(-300, 1400); axs[0].legend(); axs[0].grid(alpha=0.3)
    axs[0].set_title('direct strip: Y template from X + RC diffusion')
    axs[1].plot(emp_t, c1y * sm, 'b--', lw=1.5, label='empirical +-1 (c1*kY)')
    axs[1].plot(*r[1], 'r', lw=1.5, label='RC pred +-1')
    axs[1].plot(grid + 2 * h['tau_s'], c2y * sm, 'c--', lw=1, label='empirical +-2')
    axs[1].plot(*r[2], 'm', lw=1, label='RC pred +-2')
    axs[1].set_xlim(-300, 1400); axs[1].legend(); axs[1].grid(alpha=0.3)
    axs[1].set_title('neighbour responses: prediction vs calibrated kernel')
    fig.tight_layout()
    png = out.replace('.npz', '.png')
    fig.savefig(png, dpi=110)
    print('wrote', png)


if __name__ == '__main__':
    main()
