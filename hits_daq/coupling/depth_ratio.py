#!/usr/bin/env python3
"""depth_ratio.py <window pickles> --bundle B --out DIR [--wmax 1.2e-3] [--jobs N] -- lateral sharing against drift depth.

A same-time sharing (prompt footprint, capacitive coupling, electronics cross-talk) puts the same fraction of every depth's
charge on the neighbours; drift diffusion adds a share that grows with the drift time u (sigma^2 = sigma0^2 + Dp^2 u), and on
Y the RC spreading adds a delayed share.  On near-normal planes (|w| < wmax: the track stays within a fraction of a pitch over
the gap) the strip waveforms are stacked at t - t0 and the neighbour-to-centre ratio is read sample by sample, i.e. against
drift time smeared by the ~200 ns template.  The model arms are regenerated on the plane's own (p0, w, t0), their charge
profile q by NNLS under the arm's kernel (as observables.py), and stacked identically, so the smearing cancels in the
comparison.  Strips are ordered (far2, far, centre, near, near2) with 'near' the side the pass p0 leans to.

Outputs in DIR: planes.parquet (per plane: phase, |w|, amplitude, window charges), stacks.npz (per view/arm: the summed
5-strip stacks per |phase| bin and |w| class, with bootstrap errors of the ratios), summary.json, figures/*.png.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..'), os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))]
import emulate as E  # noqa: E402
from observables import design, padded  # noqa: E402

G = {}
PITCH = 0.78
GRID = np.arange(-120.0, 1081.0, 20.0)          # ns after t0
RC = 3.81875e-4
# name: (sigma0 [mm], Dp [mm/sqrt(ns)], rc_D_y [mm^2/ns], k1, k2)
ARMS = {
    'pass': (0.30, 0.023, RC, 0.0, 0.0),
    'metal': (0.07, 0.023, 1e-12, 0.0, 0.0),
    'nodiff': (0.30, 1e-6, RC, 0.0, 0.0),
    'norc': (0.30, 0.023, 1e-12, 0.0, 0.0),
    'Dp18': (0.30, 0.018, RC, 0.0, 0.0),
    'Dp20': (0.30, 0.020, RC, 0.0, 0.0),
    's025': (0.25, 0.023, RC, 0.0, 0.0),
    's035': (0.35, 0.023, RC, 0.0, 0.0),
    'pureK': {'x': (0.033, 0.0195, RC, 0.131, 0.010), 'y': (0.033, 0.0185, RC, 0.111, 0.020)},
    'free': {'x': (0.246, 0.0181, RC, 0.061, 0.018), 'y': (0.117, 0.0184, RC, 0.092, 0.021)},
}
PHB = (0.0, 0.17, 0.33, 0.5001)
T_GAP = 784.0


def arm_params(name, view):
    a = ARMS[name]
    return a[view] if isinstance(a, dict) else a


def resample(y, t0):
    ts = np.arange(len(y)) * E.SNS
    return np.interp(t0 + GRID, ts, y, left=np.nan, right=np.nan)


def one(p):
    wm = G['wm']
    v = p['view']
    W, nz, g, pos = (p[k].astype(float) for k in ('W', 'noise', 'gain', 'pos'))
    w, p0, t0 = float(p['w']), float(p['p0']), float(p['t0'])
    if abs(w) > G['wmax'] or len(pos) < 7:
        return None
    pc = p0 + w * T_GAP / 2
    c = int(np.argmin(np.abs(pos - pc)))
    if c < 2 or c >= len(pos) - 2:
        return None
    phi = (pc - pos[c]) / PITCH
    s = 1 if phi >= 0 else -1
    order = [c - 2 * s, c - s, c, c + s, c + 2 * s]
    ts = np.arange(W.shape[1]) * E.SNS
    win = (ts >= t0 - 120) & (ts <= t0 + 1080)
    sat = p['censor'].astype(bool)
    if sat[order][:, win].any():
        return None
    A = float(np.max(W[c]))
    if A < 8 * nz[c] or A < 150:
        return None
    out = dict(event_id=int(p['event_id']), view=v, phi=float(abs(phi)), sphi=float(phi), w=abs(w), A=A, p0=p0, t0=t0,
               c_pos=float(pos[c]), noise=float(nz[c]))
    stacks = {'data': np.array([resample(W[i], t0) for i in order], np.float32)}
    pos2, g2 = padded(pos, g)
    for name in G['arms']:
        s0, dp, rc, k1, k2 = arm_params(name, v)
        hy = dict(G['hyper'], sigma_p0=s0, sigma_p0_x=s0, sigma_p0_y=s0, Dp=dp, rc_D_y=rc)
        B = design(wm, v, pos2, g2, p, hy, k1, k2)
        chi, q = wm._solve_nnls(B, W, nz, sat, True)
        if q is None:
            return None
        M = (B @ q).reshape(W.shape)
        stacks[name] = np.array([resample(M[i], t0) for i in order], np.float32)
        out[f'chi2_{name}'] = float(chi)
    # window charges (three drift windows) for the per-plane table
    for lab, (lo, hi) in dict(early=(-60, 180), mid=(180, 480), late=(480, 900)).items():
        k = (ts >= t0 + lo) & (ts <= t0 + hi)
        Q = W[:, k].sum(1)
        out[f'Qc_{lab}'] = float(Q[c])
        out[f'rfar_{lab}'] = float(Q[order[1]] / Q[c]) if Q[c] > 0 else np.nan
        out[f'rnear_{lab}'] = float(Q[order[3]] / Q[c]) if Q[c] > 0 else np.nan
        out[f'rfar2_{lab}'] = float(Q[order[0]] / Q[c]) if Q[c] > 0 else np.nan
        out[f'rnear2_{lab}'] = float(Q[order[4]] / Q[c]) if Q[c] > 0 else np.nan
    return out, stacks


def sigma_fit(S, xs):
    """lateral width of a 5-strip pattern S (5, nt) per time: erf-integrated Gaussian, (centroid, sigma) by least squares."""
    sig = np.full(S.shape[1], np.nan)
    cen = np.full(S.shape[1], np.nan)
    for j in range(S.shape[1]):
        y = S[:, j]
        if not np.all(np.isfinite(y)) or y.sum() <= 0 or y[2] <= 0:
            continue
        f = y / y.sum()
        x0 = float(np.sum(f * xs))

        def res(par):
            m, sg = par
            z = 1.0 / (np.sqrt(2) * max(sg, 1e-3))
            F = 0.5 * (erf((xs + PITCH / 2 - m) * z) - erf((xs - PITCH / 2 - m) * z))
            return (F - f)

        try:
            r = least_squares(res, [x0, 0.4], bounds=([-1.0, 0.02], [1.0, 3.0]))
            cen[j], sig[j] = r.x
        except Exception:
            pass
    return cen, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wmax', type=float, default=1.2e-3)
    ap.add_argument('--wtight', type=float, default=3.0e-4)
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--nboot', type=int, default=200)
    a = ap.parse_args()
    from wft import model as wm
    from wft.calib import CalibrationBundle
    planes = []
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        planes += D['planes']
    cal = CalibrationBundle.load(a.bundle)
    wm.use_calibration(cal)
    wm.set_depth_bins(len(planes[0]['q']))
    G.update(wm=wm, hyper=dict(cal.hyper), arms=a.arms.split(','), wmax=a.wmax)
    os.makedirs(os.path.join(a.out, 'figures'), exist_ok=True)
    with mp.Pool(a.jobs) as P:
        R = [r for r in P.imap_unordered(one, planes, chunksize=16) if r is not None]
    rows = [r for r, _ in R]
    T = pd.DataFrame(rows)
    T.to_parquet(os.path.join(a.out, 'planes.parquet'))
    print(f'[depth] {len(planes)} planes in, {len(T)} near-normal kept')
    names = ['data'] + G['arms']
    xs = PITCH * np.arange(-2, 3)
    summary = dict(grid=GRID.tolist(), xs=xs.tolist(), phase_bins=list(PHB), wmax=a.wmax, wtight=a.wtight, curves={})
    npz = {}
    rng = np.random.default_rng(1)
    for view in ('x', 'y'):
        for wcls, wsel in (('tight', lambda r: r['w'] < a.wtight), ('all', lambda r: True)):
            for ib in range(len(PHB) - 1 + 1):
                if ib < len(PHB) - 1:
                    lo, hi = PHB[ib], PHB[ib + 1]
                    plab = f'ph{ib}'
                else:
                    lo, hi, plab = 0.0, 1.0, 'phall'
                idx = [i for i, r in enumerate(rows) if r['view'] == view and wsel(r) and lo <= r['phi'] < hi]
                if len(idx) < 20:
                    continue
                key = f'{view}|{wcls}|{plab}'
                summary['curves'][key] = dict(n=len(idx))
                for name in names:
                    S = np.array([R[i][1][name] for i in idx])          # (n, 5, nt)
                    tot = np.nansum(S, 0)                                 # (5, nt)
                    npz[f'{key}|{name}'] = tot.astype(np.float32)
                    cen = tot[2]
                    with np.errstate(invalid='ignore', divide='ignore'):
                        r1 = (tot[1] + tot[3]) / cen
                        rfar, rnear = tot[1] / cen, tot[3] / cen
                        r2 = (tot[0] + tot[4]) / cen
                    cu, sg = sigma_fit(tot, xs)
                    o = dict(r1=r1.tolist(), rfar=rfar.tolist(), rnear=rnear.tolist(), r2=r2.tolist(), sigma=sg.tolist(),
                             centroid=cu.tolist(), centre=cen.tolist())
                    if name == 'data' and a.nboot:
                        bs = np.empty((a.nboot, 4, S.shape[2]))
                        for b in range(a.nboot):
                            k = rng.integers(0, len(idx), len(idx))
                            tb = np.nansum(S[k], 0)
                            with np.errstate(invalid='ignore', divide='ignore'):
                                bs[b, 0] = (tb[1] + tb[3]) / tb[2]
                                bs[b, 1] = tb[1] / tb[2]
                                bs[b, 2] = tb[3] / tb[2]
                                bs[b, 3] = (tb[0] + tb[4]) / tb[2]
                        e = np.nanstd(bs, 0)
                        o.update(r1_err=e[0].tolist(), rfar_err=e[1].tolist(), rnear_err=e[2].tolist(), r2_err=e[3].tolist())
                    summary['curves'][key][name] = o
    np.savez_compressed(os.path.join(a.out, 'stacks.npz'), **npz)
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'))
    # ---- figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tgrid = GRID
    for view in ('x', 'y'):
        key = f'{view}|tight|phall'
        if key not in summary['curves']:
            continue
        C = summary['curves'][key]
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        for name in names:
            if name not in C:
                continue
            o = C[name]
            lw = 2.2 if name == 'data' else 1.2
            ax[0, 0].plot(tgrid, o['r1'], lw=lw, label=name)
            ax[0, 1].plot(tgrid, o['r2'], lw=lw, label=name)
            ax[1, 0].plot(tgrid, o['sigma'], lw=lw, label=name)
            ax[1, 1].plot(tgrid, np.array(o['centre']) / np.nanmax(o['centre']), lw=lw, label=name)
        if 'r1_err' in C['data']:
            d = C['data']
            ax[0, 0].fill_between(tgrid, np.array(d['r1']) - np.array(d['r1_err']), np.array(d['r1']) + np.array(d['r1_err']), color='k', alpha=0.15)
            ax[0, 1].fill_between(tgrid, np.array(d['r2']) - np.array(d['r2_err']), np.array(d['r2']) + np.array(d['r2_err']), color='k', alpha=0.15)
        ax[0, 0].set(ylabel='(W[-1]+W[+1]) / W[0]', ylim=(0, 1.2), title=f'{view}: first neighbours, n={C["n"]} planes |tan|<{a.wtight / 0.0358:.3f}')
        ax[0, 1].set(ylabel='(W[-2]+W[+2]) / W[0]', ylim=(0, 0.3), title='second neighbours')
        ax[1, 0].set(ylabel='sigma_eff of the 5-strip pattern [mm]', xlabel='t - t0 [ns]', ylim=(0, 1.2), title='lateral width per sample (erf fit)')
        ax[1, 1].set(ylabel='centre strip (normalised)', xlabel='t - t0 [ns]', title='stacked centre waveform')
        for axx in ax.ravel():
            axx.grid(alpha=0.3)
        ax[0, 0].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, 'figures', f'depth_ratio_{view}.png'), dpi=130)
        plt.close(fig)
        # near/far by phase, data vs pass vs pureK
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
        for ib in range(3):
            key = f'{view}|all|ph{ib}'
            if key not in summary['curves']:
                continue
            C = summary['curves'][key]
            for name, col in (('data', 'k'), ('pass', 'C0'), ('pureK', 'C2'), ('metal', 'C3'), ('free', 'C1')):
                if name not in C:
                    continue
                o = C[name]
                ax[ib].plot(tgrid, o['rnear'], color=col, lw=2 if name == 'data' else 1, label=f'{name} near')
                ax[ib].plot(tgrid, o['rfar'], color=col, lw=2 if name == 'data' else 1, ls='--', label=f'{name} far')
            ax[ib].set(title=f'{view}: |phase| {PHB[ib]:.2f}-{PHB[ib + 1]:.2f}, n={C["n"]}', xlabel='t - t0 [ns]', ylim=(0, 0.9))
            ax[ib].grid(alpha=0.3)
        ax[0].set_ylabel('neighbour / centre')
        ax[0].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, 'figures', f'nearfar_{view}.png'), dpi=130)
        plt.close(fig)
    print(f'[depth] wrote {a.out}')


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
