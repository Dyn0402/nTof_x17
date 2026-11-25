#!/usr/bin/env python3
"""toy.py <data root> -- what the head-on stack fit returns for T when the record ends close to (or before) the end of
the drift column, and when the column loses charge to attachment.

The toy uses the fit's own model (x_width.predict_x: a uniform column of duration T, diffusion sigma^2 = sigma_0^2 +
2 Dd u, convolved with the row's electronics response h) as the truth, so any bias it finds is a property of the fit
and of the truncation, not of a model mismatch -- except in the attachment arm, where the truth has the charge of the
slice drifting for u weighted by exp(-u / tau) and the fit is still the uniform column.

Per case (a template row, a true T, a record end, tau): the stack is the model on rc_diffusion.GRID, NaN after the
record end (as rc_diffusion.stack leaves it when fewer than 60 % of the events reach that time), plus white Gaussian
noise of the row's fit rms (0 for the pure bias), and it is refitted with x_width.fit_x from the row's T0 (the same
bounds 0.6-1.5 T0 and the same four dt starts).  Reported: T_fit - T_true against the margin = record end - (dt + T).

Template rows (inventory.csv): A 700 V (6-27), E 6-24 long run (g_det4), C 6-26 long run, B 6-22 long run.
Writes results/toy.csv, results/toy_attach.csv.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'rerun', 'calib', 's4'), os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')]
import s4_stacks  # noqa: E402
import x_width as xw  # noqa: E402
from rc_diffusion import GRID, STEP, OFFS, F  # noqa: E402

RES = os.path.join(HERE, 'results')
ROWS = {'A700': ('mx17_det3_saturday_scan_6-27-26', 'drift_scan_resist_490V_drift_700V', 'det3'),
        'E0624': ('mx17_det4_day_6-24-26', 'long_run', 'det4'),
        'C0626': ('mx17_det6_det7_overnight_6-26-26', 'long_run', 'det6'),
        'B0622': ('mx17_det2_det3_overnight_6-22-26', 'long_run', 'det2')}
MARGINS = [-300, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250, 300, 400, 500, 700, 900]
TAUS = [np.inf, 4000, 2000, 1500, 1000]
NOISE_REP = 6
OFF = 60.0          # stack time of the column start = dt + OFF (GRID[0] - template grid start = -300 - (-360))


def predict_att(h, sig0, Dd, T, A, t_shift, n, tau):
    """x_width.predict_x with the charge of the slice drifting for u weighted by exp(-u / tau) (attachment)."""
    if not np.isfinite(tau):
        return xw.predict_x(h, sig0, Dd, T, A, t_shift, n)
    nf = T / STEP
    nu = int(np.ceil(nf))
    w = np.ones(nu); w[-1] = nf - (nu - 1)
    u = (np.arange(nu) + 0.5) * STEP
    w = w * np.exp(-u / tau)
    Fo = np.array([F(np.sqrt(sig0 ** 2 + 2 * Dd * u), o) for o in OFFS]) * w
    H = np.zeros((nu, n))
    for k in range(nu):
        m = min(len(h), n - k)
        if m > 0:
            H[k, k:k + m] = h[:m]
    P = A * (Fo @ H)
    t = np.arange(n) * STEP
    return np.array([np.interp(t - t_shift, t, p, left=0.0, right=p[-1]) for p in P])


def one(case):
    h, p, T_true, t_end, tau, sigma, seed = case
    n = len(GRID)
    P = predict_att(h, p['sig0'], p['Dd'], T_true, 1.0, p['dt'], n, tau)     # dt on the fit's own clock, as fit_x
    # normalise as rc_diffusion.stack does (max of the strip sum = 1)
    P = P / np.nanmax(P.sum(axis=0))
    rng = np.random.default_rng(seed)
    M = P + (rng.normal(0, sigma, P.shape) if sigma > 0 else 0.0)
    M[:, GRID > t_end] = np.nan
    try:
        r = xw.fit_x(h, M, p['T0'])
    except Exception as ex:     # noqa: BLE001
        return dict(T_fit=np.nan, dt_fit=np.nan, err=str(ex)[:80])
    return dict(T_fit=r['T'], dt_fit=r['dt'], sig0_fit=r['sig0'], Dd_fit=r['Dd'], rms_fit=r['rms'])


def main():
    root = sys.argv[1]
    inv = pd.read_csv(os.path.join(RES, 'inventory.csv'))
    cases, meta = [], []
    for name, (run, sub, det) in ROWS.items():
        q = inv[(inv.run == run) & (inv.subrun == sub) & (inv.det == det)].iloc[0]
        tdir = os.path.join(root, 'Analysis_pass1', run, sub, 'mx17_' + det[3:], 'wft', 'calib_bundle')
        h = s4_stacks.pass_template(tdir)
        # clocks: predict_x's index 0 is GRID[0] = -300 ns and the template's index 0 is its grid start, -360 ns from
        # the pulse's own t0, so the column's charge starts reaching the strips (pulse t0) at stack time dt + OFF
        p = dict(sig0=float(q['sig0']), Dd=float(q['Dd']), dt=float(q['dt']), T0=float(q['T0']))
        sig = float(q['rms'])                 # the row's fit residual rms (white noise of that size)
        for T_true in (q['T'],):
            for tau in TAUS:
                for m in MARGINS:
                    t_end = p['dt'] + OFF + T_true + m
                    for rep in range(NOISE_REP + 1):
                        cases.append((h, p, T_true, t_end, tau, 0.0 if rep == 0 else sig, 100000 * rep + int(m) + 1000))
                        meta.append(dict(row=name, T_true=T_true, dt_true=p['dt'], t_end=t_end, margin=m, tau=tau,
                                         noise=rep > 0, rep=rep, sigma=sig))
    print(len(cases), 'fits', flush=True)
    with ProcessPoolExecutor(14) as pool:
        out = list(pool.map(one, cases, chunksize=4))
    D = pd.concat([pd.DataFrame(meta), pd.DataFrame(out)], axis=1)
    D['bias'] = D.T_fit - D.T_true
    D.to_csv(os.path.join(RES, 'toy.csv'), index=False)
    s = D[~D.noise].pivot_table(index='margin', columns=['row', 'tau'], values='bias')
    pd.set_option('display.width', 250)
    print(s.round(0))


if __name__ == '__main__':
    main()
