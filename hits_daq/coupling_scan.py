#!/usr/bin/env python3
"""coupling_scan.py scan|resim <window pickle(s)> --out OUT.parquet [--nmax N] [--bundle DIR] -- same-time neighbour coupling.

The literature's explanation of the micro-TPC angle bias is capacitive coupling: a strip sees a fixed fraction kappa of its
neighbours' signal at the neighbours' time.  The pass model has no such term; its only same-time sharing is the Gaussian
prompt footprint sigma_p0.  The two differ: the footprint share depends on where the charge lands in the strip, kappa does not.

Coupled model: M_i -> M_i + kappa (M_{i-1} + M_{i+1}), with the strip list padded by PAD on each side so that the window's
edge strips also receive it.  The charge profile q stays free, so a global amplitude cannot fake kappa.

  scan   chi2(kappa, sigma_p0) on data W, per plane, q by NNLS with the pass weighting; track (p0, w, t0) fixed at the pass
         values, or refitted per grid point with --refit (Nelder-Mead, 3 parameters).
  phase  model-free discriminator on near-normal planes: neighbour / centre early charge against where the track lands in the
         strip, for data and for coupled models (--grid); the footprint share depends on the phase, kappa does not.
  decomp pure-model ladders (white noise, no fit residual) at --grid points, each with kappa, diffusion, RC switched off in turn.
  resim  W' = M(hyper') coupled by kappa + (W - M), truth = the pass track; V0 ladder u for each (arm, kappa): how much
         compression coupling alone can make, with diffusion/footprint/RC on or off.
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, '..', '..', '..'))]
import emulate as E  # noqa: E402
from tiers import ladder_u  # noqa: E402

PAD = 4
G = {}          # worker state


def couple(B, kappa):
    """B (n, ...): add kappa x each direct neighbour's row"""
    if kappa == 0:
        return B
    C = B.copy()
    C[1:] += kappa * B[:-1]
    C[:-1] += kappa * B[1:]
    return C


def padded(pos, g, nz):
    pitch = np.median(np.diff(pos))
    return (np.r_[pos[0] - pitch * np.arange(PAD, 0, -1), pos, pos[-1] + pitch * np.arange(1, PAD + 1)],
            np.r_[np.full(PAD, np.median(g)), g, np.full(PAD, np.median(g))],
            np.r_[np.full(PAD, np.median(nz)), nz, np.full(PAD, np.median(nz))])


def couple_d(B, kd):
    """add kd x the time derivative (per 60 ns sample, central difference) of each direct neighbour's row: the bipolar,
    zero-net form expected for strip-to-strip coupling into a charge amplifier.  B (n, nsamp, ...)"""
    if kd == 0:
        return B
    D = np.zeros_like(B)
    D[:, 1:-1] = 0.5 * (B[:, 2:] - B[:, :-2])
    C = B.copy()
    C[1:] += kd * D[:-1]
    C[:-1] += kd * D[1:]
    return C


def design(wm, v, pos2, g2, p0, w, t0, hyper, kappa, kd=0.0):
    """the coupled design matrix on the observed strips, (n * NSAMP, K)"""
    B = wm.build_matrix(v, pos2, p0, w, t0, hyper)
    n2 = len(pos2)
    B = B.reshape(n2, -1, B.shape[1]) * g2[:, None, None]
    B = couple_d(couple(B, kappa), kd)[PAD:-PAD]
    return B.reshape(-1, B.shape[2])


def chi2_at(x, wm, v, W, nz, sat, pos2, g2, hyper, kappa, kd=0.0):
    B = design(wm, v, pos2, g2, x[0], x[1], x[2], hyper, kappa, kd)
    chi, q = wm._solve_nnls(B, W, nz, sat, True)
    return chi if q is not None else 1e12


def hyp(s0, **kw):
    return dict(G['hyper'], sigma_p0=s0, sigma_p0_x=s0, sigma_p0_y=s0, **kw)


def grid_points():
    return G['grid'] or [(s, k) for s in G['sigmas'] for k in G['kappas']]


def scan_one(p):
    wm = G['wm']
    v = p['view']
    W, nz, g, pos = (p[k].astype(float) for k in ('W', 'noise', 'gain', 'pos'))
    if len(pos) < 3 or np.any(np.abs(np.diff(pos) - np.median(np.diff(pos))) > 1e-3):
        return []
    sat = p['censor'].astype(bool)
    pos2, g2, _ = padded(pos, g, nz)
    x0 = np.array([p['p0'], p['w'], p['t0']], float)
    rows = []
    for s0, ka in grid_points():
        hy = hyp(s0)
        for kd in G['kds']:
            if G['refit']:
                simplex = np.array([x0, x0 + [0.2, 0, 0], x0 + [0, 0.002, 0], x0 + [0, 0, 20.0]])
                r = minimize(chi2_at, x0, args=(wm, v, W, nz, sat, pos2, g2, hy, ka, kd), method='Nelder-Mead',
                             options=dict(initial_simplex=simplex, maxfev=120, xatol=1e-3, fatol=1e-2))
                chi, x = r.fun, r.x
            else:
                chi, x = chi2_at(x0, wm, v, W, nz, sat, pos2, g2, hy, ka, kd), x0
            rows.append(dict(event_id=int(p['event_id']), view=v, sigma=s0, kappa=ka, kd=kd, chi2=float(chi), w=float(x[1]),
                             w_pass=float(p['w']), p0=float(x[0]), dof=int(p['dof'])))
    return rows


def resim_one(p):
    wm = G['wm']
    v = p['view']
    W, nz, g, pos, M = (p[k].astype(float) for k in ('W', 'noise', 'gain', 'pos', 'M'))
    if len(pos) < 3:
        return []
    rng = np.random.default_rng(int(p['event_id']) * 2 + (v == 'y'))
    R = W - M
    pos2, g2, nz2 = padded(pos, g, nz)
    R2 = np.vstack([rng.normal(0, 1, (PAD, R.shape[1])) * nz2[0], R, rng.normal(0, 1, (PAD, R.shape[1])) * nz2[-1]])
    q = p['q'].astype(float)
    rows = []
    for arm, ov in G['arms'].items():
        hy = dict(G['hyper'], **ov)
        M2 = wm.model_waveforms(v, pos2, p['p0'], p['w'], p['t0'], q, hy) * g2[:, None]
        for ka in G['kappas']:
            h = E.dream(couple(M2, ka) + R2, nz2, nsig=5.0)
            u, n = ladder_u(pos2, h['t'], h['a'])
            ur, _ = ladder_u(pos2, h['t'], h['a'], robust=True)
            rows.append(dict(event_id=int(p['event_id']), view=v, arm=arm, kappa=ka, u=u, ur=ur, n=n, w_true=float(p['w'])))
    return rows


EARLY = (-60.0, 180.0)          # ns around t0: the mesh-end charge, little diffusion


def early(W, t0):
    ts = np.arange(W.shape[1]) * E.SNS
    k = (ts >= t0 + EARLY[0]) & (ts <= t0 + EARLY[1])
    return W[:, k].sum(1)


def phase_one(p):
    wm = G['wm']
    v = p['view']
    W, nz, g, pos = (p[k].astype(float) for k in ('W', 'noise', 'gain', 'pos'))
    if abs(p['w']) > 1.2e-3 or len(pos) < 5:          # tan < 0.033: the track moves < 0.3 mm during the early window
        return []
    pitch = np.median(np.diff(pos))
    pc = p['p0'] + p['w'] * 90.0                      # the track at the early window's charge
    c = int(np.argmin(np.abs(pos - pc)))
    if c < 1 or c >= len(pos) - 1:
        return []
    phi = (pc - pos[c]) / pitch
    sgn = 1 if phi >= 0 else -1                  # the neighbour on the side the charge is displaced to: 'near'
    near, far = c + sgn, c - sgn
    if not (0 <= near < len(pos) and 0 <= far < len(pos)):
        return []
    pos2, g2, _ = padded(pos, g, nz)
    sat = p['censor'].astype(bool)
    ts = np.arange(W.shape[1]) * E.SNS
    ek = (ts >= p['t0'] + EARLY[0]) & (ts <= p['t0'] + EARLY[1])
    if sat[[c, near, far]][:, ek].any():              # saturated or zero-suppressed in the window that is compared
        return []
    Q = early(W, p['t0'])
    if Q[c] < 300:
        return []
    r = dict(event_id=int(p['event_id']), view=v, phi=abs(phi), Qc=Q[c], near_data=Q[near] / Q[c], far_data=Q[far] / Q[c])
    rng = np.random.default_rng(int(p['event_id']) * 2 + (v == 'y'))
    for s0, ka in grid_points():
        for dp in G['dps']:                           # the fitted position is wrong by ~dp: evaluate the model at the true one
            dd = rng.normal(0, dp) if dp > 0 else 0.0
            B = design(wm, v, pos2, g2, p['p0'] + dd, p['w'], p['t0'], hyp(s0), ka)
            chi, q = wm._solve_nnls(B, W, nz, sat, True)
            if q is None:
                return []
            Qm = early((B @ q).reshape(W.shape), p['t0'])
            r[f'near_{s0}_{ka}_{dp}'], r[f'far_{s0}_{ka}_{dp}'] = Qm[near] / Qm[c], Qm[far] / Qm[c]
    return [r]


WINS = ((-60.0, 180.0), (180.0, 480.0), (480.0, 900.0))    # ns after t0: mesh end, middle, cathode end of the column


def wsum(W, t0, lo, hi):
    ts = np.arange(W.shape[1]) * E.SNS
    return W[:, (ts >= t0 + lo) & (ts <= t0 + hi)].sum(1)


def reach_one(p):
    """near-normal planes: far-side strips -1 and -2 (and near-side +2) over the centre strip, in three drift-time windows, for
    data and models: coupling reaches +-1 (kappa) and +-2 only as kappa^2; real long-range charge falls smoothly."""
    wm = G['wm']
    v = p['view']
    W, nz, g, pos = (p[k].astype(float) for k in ('W', 'noise', 'gain', 'pos'))
    if abs(p['w']) > 1.2e-3 or len(pos) < 7:
        return []
    pitch = np.median(np.diff(pos))
    pc = p['p0'] + p['w'] * 90.0
    c = int(np.argmin(np.abs(pos - pc)))
    phi = (pc - pos[c]) / pitch
    sgn = 1 if phi >= 0 else -1
    idx = dict(m1=c - sgn, m2=c - 2 * sgn, p2=c + 2 * sgn)
    ctrl = dict(m3=c - 3 * sgn, m4=c - 4 * sgn)              # controls: no charge can reach them early
    if abs(phi) < 0.25 or any(not 0 <= i < len(pos) for i in idx.values()):
        return []                                     # phase >= 0.25: the far side is outside any Gaussian core
    sat = p['censor'].astype(bool)
    pos2, g2, _ = padded(pos, g, nz)
    models = {}
    for s0, ka in grid_points():
        B = design(wm, v, pos2, g2, p['p0'], p['w'], p['t0'], hyp(s0), ka)
        chi, q = wm._solve_nnls(B, W, nz, sat, True)
        if q is not None:
            models[f'{s0}_{ka}'] = (B @ q).reshape(W.shape)
    for name, ov in G['lor'].items():
        B = design(wm, v, pos2, g2, p['p0'], p['w'], p['t0'], dict(hyp(ov['s']), **{f'lor_frac_{v}': ov['f'], f'lor_gamma_{v}': ov['g']}), 0.0)
        chi, q = wm._solve_nnls(B, W, nz, sat, True)
        if q is not None:
            models[name] = (B @ q).reshape(W.shape)
    rows = []
    for iw, (lo, hi) in enumerate(WINS):
        ts = np.arange(W.shape[1]) * E.SNS
        ek = (ts >= p['t0'] + lo) & (ts <= p['t0'] + hi)
        if sat[[c] + list(idx.values())][:, ek].any():
            continue
        Qc = wsum(W, p['t0'], lo, hi)[c]
        if Qc < 300:
            continue
        r = dict(event_id=int(p['event_id']), view=v, win=iw, phi=abs(phi), p0=float(p['p0']))
        S = wsum(W, p['t0'], lo, hi)
        for lab, i in ctrl.items():
            if 0 <= i < len(pos):
                r[f'{lab}_data'] = S[i] / Qc
                r[f'{lab}_cens'] = float(sat[i][ek].mean())
        for lab, i in idx.items():
            r[f'{lab}_data'] = wsum(W, p['t0'], lo, hi)[i] / Qc
            for name, M in models.items():
                r[f'{lab}_{name}'] = wsum(M, p['t0'], lo, hi)[i] / wsum(M, p['t0'], lo, hi)[c]
        rows.append(r)
    return rows


def decomp_one(p):
    wm = G['wm']
    v = p['view']
    nz, g, pos = (p[k].astype(float) for k in ('noise', 'gain', 'pos'))
    if len(pos) < 3:
        return []
    rng = np.random.default_rng(int(p['event_id']) * 2 + (v == 'y'))
    pos2, g2, nz2 = padded(pos, g, nz)
    q = p['q'].astype(float)
    rows = []
    for s0, ka in grid_points():
        for arm, (kk, ov) in dict(full=(ka, {}), no_kappa=(0.0, {}), no_diff=(ka, dict(Dp=0.0)), no_rc=(ka, dict(rc_D_y=1e-12)),
                                  no_diff_rc=(ka, dict(Dp=0.0, rc_D_y=1e-12)), floor=(0.0, dict(Dp=0.0, rc_D_y=1e-12))).items():
            hy = hyp(s0, **ov) if 'Dp' not in ov else dict(hyp(s0), **ov)
            M2 = wm.model_waveforms(v, pos2, p['p0'], p['w'], p['t0'], q, hy) * g2[:, None]
            W2 = couple(M2, kk) + rng.normal(size=M2.shape) * nz2[:, None]
            h = E.dream(W2, nz2, nsig=5.0)
            u, n = ladder_u(pos2, h['t'], h['a'])
            rows.append(dict(event_id=int(p['event_id']), view=v, sigma=s0, kappa=ka, arm=arm, u=u, n=n, w_true=float(p['w'])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('scan', 'resim', 'phase', 'decomp', 'reach'))
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bundle')
    ap.add_argument('--nmax', type=int, default=0)
    ap.add_argument('--kappas', default='0,0.03,0.06,0.1,0.15,0.2,0.3')
    ap.add_argument('--sigmas', default='0.07,0.15,0.2,0.25,0.3,0.35')
    ap.add_argument('--grid', default='', help='sigma:kappa points instead of the product, comma separated')
    ap.add_argument('--dps', default='0', help='phase: position-error sigmas [mm] injected in the model')
    ap.add_argument('--lor', default='', help='reach: Lorentzian-tail models name:sigma:frac:gamma, comma separated')
    ap.add_argument('--kds', default='0', help='scan: derivative-coupling values')
    ap.add_argument('--refit', action='store_true')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    from wft import model as wm
    from wft.calib import CalibrationBundle
    planes, meta = [], None
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = meta or D['meta']
        planes += D['planes']
    if a.nmax:
        planes = planes[:a.nmax]
    cal = CalibrationBundle.load(a.bundle or E.bundle_path(meta))
    wm.use_calibration(cal)
    wm.set_depth_bins(len(planes[0]['q']))
    hyper = dict(cal.hyper)
    G.update(wm=wm, hyper=hyper, refit=a.refit, kappas=[float(x) for x in a.kappas.split(',')],
             sigmas=[float(x) for x in a.sigmas.split(',')],
             lor={x.split(':')[0]: dict(zip('sfg', (float(y) for y in x.split(':')[1:]))) for x in a.lor.split(',')} if a.lor else {},
             kds=[float(x) for x in a.kds.split(',')],
             dps=[float(x) for x in a.dps.split(',')],
             grid=[tuple(float(y) for y in x.split(':')) for x in a.grid.split(',')] if a.grid else [])
    G['arms'] = dict(base={}, metal=dict(rc_D_y=1e-12, sigma_p0=0.07, sigma_p0_x=0.07, sigma_p0_y=0.07),
                     point=dict(rc_D_y=1e-12, sigma_p0=0.07, sigma_p0_x=0.07, sigma_p0_y=0.07, Dp=0.0))
    fn = dict(scan=scan_one, resim=resim_one, phase=phase_one, decomp=decomp_one, reach=reach_one)[a.mode]
    t0 = time.time()
    with mp.Pool(a.jobs) as P:
        rows = [r for rr in P.imap_unordered(fn, planes, chunksize=8) for r in rr]
    T = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    T.to_parquet(a.out)
    print(f'[coupling] {a.mode}: {len(planes)} planes, {len(T)} rows, {time.time() - t0:.0f} s -> {a.out}')


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
