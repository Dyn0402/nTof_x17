#!/usr/bin/env python3
"""attfit.py <data root> -- the head-on stack fit with attachment: x_width.fit_x's uniform column, with the charge of the
slice drifting for u weighted by exp(-u / tau) and 1/tau free (>= 0).  Per row: tau, T with tau free, and T with tau
fixed at infinity (the production fit), on the full stack and on the stack cut at 116 ns after the column end (chamber E's
6-24 margin).  First a closure on toys (toy.predict_att as the truth: does the fit recover T and tau?).

Writes results/attfit.csv, results/attfit_closure.csv.
"""
import json
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'rerun', 'calib', 's4'), os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')]
import inventory as I  # noqa: E402
import toy  # noqa: E402
import s4_stacks  # noqa: E402
import x_width as xw  # noqa: E402
from rc_diffusion import GRID  # noqa: E402

RES = os.path.join(HERE, 'results')
KMAX = 1 / 300.0          # 1/tau upper bound [1/ns]


def fit_att(h, M, T0, free_k=True):
    """x_width.fit_x with a fifth/sixth parameter k = 1/tau (free_k) -- same bounds, same dt starts."""
    sel = np.all(np.isfinite(M), axis=0)
    if not sel.any():
        raise ValueError('empty stack')
    n = len(GRID)

    def res(q):
        sig0, Dd, A, dt, T = q[:5]
        k = q[5] if free_k else 0.0
        tau = 1 / k if k > 0 else np.inf
        return (toy.predict_att(h, sig0, Dd, T, A, dt, n, tau)[:, sel] - M[:, sel]).ravel()

    x0 = [0.4, 3e-4, 0.02, 0.0, T0]; lo = [0.02, 0.0, 1e-4, -600, 0.6 * T0]; hi = [1.5, 0.005, 10, 600, 1.5 * T0]
    sc = [0.1, 1e-4, 0.01, 30, 30]
    if free_k:
        x0.append(1e-4); lo.append(0.0); hi.append(KMAX); sc.append(1e-4)
    best = None
    for d0 in (100.0, 200.0, 300.0, 400.0):
        x0[3] = d0
        r = least_squares(res, x0, bounds=(lo, hi), diff_step=1e-3, x_scale=sc)
        if best is None or r.cost < best.cost:
            best = r
    q = best.x
    k = q[5] if free_k else 0.0
    return dict(sig0=q[0], Dd=q[1], A=q[2], dt=q[3], T=q[4], k=k, tau=(1 / k if k > 0 else np.inf),
                cost=float(best.cost), rms=float(np.sqrt((best.fun ** 2).mean())))


def closure_case(c):
    h, p, T_true, tau, margin, sigma, seed = c
    n = len(GRID)
    P = toy.predict_att(h, p['sig0'], p['Dd'], T_true, 1.0, p['dt'], n, tau)
    P = P / np.nanmax(P.sum(axis=0))
    rng = np.random.default_rng(seed)
    M = P + (rng.normal(0, sigma, P.shape) if sigma > 0 else 0.0)
    if margin is not None:
        M[:, GRID > p['dt'] + toy.OFF + T_true + margin] = np.nan
    a = fit_att(h, M, p['T0'], True)
    b = fit_att(h, M, p['T0'], False)
    return dict(T_att=a['T'], tau_fit=a['tau'], T_uni=b['T'])


def row_case(c):
    key, h, M, T0, col_end = c
    out = dict(key=key)
    for tag, Mc in (('full', M), ('cut116', None)):
        if Mc is None:
            Mc = M.copy(); Mc[:, GRID > col_end + 116] = np.nan
        a = fit_att(h, Mc, T0, True)
        b = fit_att(h, Mc, T0, False)
        out.update({f'T_att_{tag}': a['T'], f'tau_{tag}': a['tau'], f'T_uni_{tag}': b['T'], f'rms_att_{tag}': a['rms'],
                    f'rms_uni_{tag}': b['rms'], f'dcost_{tag}': b['cost'] - a['cost']})
    return out


def main():
    root = sys.argv[1]
    inv = pd.read_csv(os.path.join(RES, 'inventory.csv'))
    # closure: the toy rows, tau in {inf, 4000, 1500}, full record and margin 116, noise-free and 4 noisy
    cases, meta = [], []
    for name, (run, sub, det) in toy.ROWS.items():
        q = inv[(inv.run == run) & (inv.subrun == sub) & (inv.det == det)].iloc[0]
        h = s4_stacks.pass_template(os.path.join(root, 'Analysis_pass1', run, sub, 'mx17_' + det[3:], 'wft', 'calib_bundle'))
        p = dict(sig0=float(q['sig0']), Dd=float(q['Dd']), dt=float(q['dt']), T0=float(q['T0']))
        for tau in (np.inf, 4000.0, 1500.0):
            for margin in (None, 116):
                for rep in range(5):
                    cases.append((h, p, float(q['T']), tau, margin, 0.0 if rep == 0 else float(q['rms']), 31 + rep))
                    meta.append(dict(row=name, T_true=float(q['T']), tau_true=tau, margin=('full' if margin is None else margin), rep=rep))
    F = {}
    for f in glob.glob(os.path.join(root, 'x17rerun/paper1_gas', os.environ.get('FIT_DIR', 'fit_v2'), 'fit_T*.json')):
        F.update(json.load(open(f)))
    rcases = []
    for q in inv[inv.n >= 40].itertuples():
        arm = dict(F[q.key + '|in30'])
        if 'replaces' in arm:
            arm['edge'] = None
        ev = I.arm_events(arm, root)
        n, M = xw.stack(ev, 'x', tan_max=xw.TAN)
        h = s4_stacks.pass_template(os.path.dirname(I.local(arm['template'], root)))
        rcases.append((q.key, h, M, q.T0, q.col_end))
    with ProcessPoolExecutor(14) as pool:
        cl = list(pool.map(closure_case, cases, chunksize=2))
        rr = list(pool.map(row_case, rcases, chunksize=1))
    C = pd.concat([pd.DataFrame(meta), pd.DataFrame(cl)], axis=1)
    C.to_csv(os.path.join(RES, 'attfit_closure.csv'), index=False)
    R = inv.merge(pd.DataFrame(rr), on='key')
    R.to_csv(os.path.join(RES, 'attfit.csv'), index=False)
    pd.set_option('display.width', 250)
    C['bias_att'] = C.T_att - C.T_true; C['bias_uni'] = C.T_uni - C.T_true
    print(C.groupby(['row', 'tau_true', 'margin'])[['bias_att', 'bias_uni', 'tau_fit']].agg(['mean', 'std']).round(0).to_string())
    print(R[['chamber', 'run', 'subrun', 'n', 'margin', 'T', 'T_uni_full', 'T_att_full', 'tau_full', 'dcost_full',
             'T_uni_cut116', 'T_att_cut116', 'tau_cut116']].round(1).to_string())


if __name__ == '__main__':
    main()
