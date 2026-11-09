#!/usr/bin/env python3
"""phase_closure.py <window pickles> --bundle B --out DIR [--wmax 1.2e-3] [--jobs N] -- does the fit's own p0 make the near/far asymmetry?

The report's position-in-strip test (HANDOFF Q1) splits the centre strip's neighbours into 'near' (the side the fitted p0
leans to) and 'far', and at small phase finds near 0.23 vs far 0.07-0.09, where a pure nearest-neighbour coupling predicts
near = far.  But p0 is fitted on the same waveforms: on symmetric data the fit leans toward the neighbour that fluctuated up,
so 'near' is brighter than 'far' by construction, and a phase-dependence can appear in data that has none.  Closure: each
near-normal plane is regenerated under a known truth (noK: the fitted no-coupling kernel sigma0 0.318/0.276, k = 0; pureK: sigma0
0.033 with k1 0.131/0.111, k2 0.010/0.020; both with the plane's own track and NNLS q), the plane's real residual W - M is added
back, (p0, w, t0) are REFITTED with the pass fitter (pass hyper, k = 0, as the pass did on the data), and the early-charge
near/far ratios against |phase| are recomputed from the refitted p0 (and, for reference, from the true p0).  The data (pass
p0) are processed the same way.  If pureK + refit shows the data's asymmetry and floor, the test does not discriminate.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..'), os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))]
import emulate as E  # noqa: E402
from observables import design, padded  # noqa: E402

G = {}
PITCH = 0.78
RC = 3.81875e-4
TRUTHS = {
    'noK': {'x': (0.318, 0.0201, RC, 0.0, 0.0), 'y': (0.276, 0.0208, RC, 0.0, 0.0)},
    'pureK': {'x': (0.033, 0.0195, RC, 0.131, 0.010), 'y': (0.033, 0.0185, RC, 0.111, 0.020)},
}
EARLY = (-60.0, 180.0)
PHB = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5001])


def ratios(W, pos, p0, w, t0, sat):
    """early-window near/far/near2/far2 over the centre, with the phase from (p0, w, t0); None if not usable"""
    pc = p0 + w * 90.0
    c = int(np.argmin(np.abs(pos - pc)))
    if c < 2 or c >= len(pos) - 2:
        return None
    phi = (pc - pos[c]) / PITCH
    s = 1 if phi >= 0 else -1
    ts = np.arange(W.shape[1]) * E.SNS
    ek = (ts >= t0 + EARLY[0]) & (ts <= t0 + EARLY[1])
    idx = [c - 2 * s, c - s, c, c + s, c + 2 * s]
    if sat[idx][:, ek].any():
        return None
    Q = W[:, ek].sum(1)
    if Q[c] < 300:
        return None
    return dict(phi=float(abs(phi)), sphi=float(phi), Qc=float(Q[c]), far2=float(Q[idx[0]] / Q[c]), far=float(Q[idx[1]] / Q[c]),
                near=float(Q[idx[3]] / Q[c]), near2=float(Q[idx[4]] / Q[c]), c_pos=float(pos[c]))


def one(p):
    wm, wr, cal = G['wm'], G['wr'], G['cal']
    v = p['view']
    W, nz, g, pos, M = (p[k].astype(float) for k in ('W', 'noise', 'gain', 'pos', 'M'))
    w, p0, t0 = float(p['w']), float(p['p0']), float(p['t0'])
    if abs(w) > G['wmax'] or len(pos) < 7:
        return []
    sat = p['censor'].astype(bool)
    base = dict(event_id=int(p['event_id']), view=v, w_pass=w, p0_pass=p0)
    rows = []
    r = ratios(W, pos, p0, w, t0, sat)
    if r is not None:
        rows.append(dict(base, truth='data', which='pass', dp0=0.0, dw=0.0, **r))
    R = W - M
    pos2, g2 = padded(pos, g)
    for name, tr in G['truths'].items():
        s0, dp, rc, k1, k2 = tr[v]
        hy = dict(G['hyper'], sigma_p0=s0, sigma_p0_x=s0, sigma_p0_y=s0, Dp=dp, rc_D_y=rc)
        B = design(wm, v, pos2, g2, p, hy, k1, k2)
        chi, q = wm._solve_nnls(B, W, nz, sat, True)
        if q is None:
            continue
        Mt = (B @ q).reshape(W.shape)
        rng = np.random.default_rng(int(p['event_id']) * 4 + (v == 'y'))
        for nm in G['noise']:
            # 'own': the plane's real residual (correlated noise and model error, but the same realisation the pass p0 saw);
            # 'white': fresh Gaussian noise at the strip sigma (the fit's lean toward a fluctuating neighbour is then independent of
            # the data's p0, which is what the circularity question needs)
            Wp = Mt + (R if nm == 'own' else rng.normal(size=W.shape) * nz[:, None])
            tn = f'{name}_{nm}'
            r = ratios(Wp, pos, p0, w, t0, sat)
            if r is not None:
                rows.append(dict(base, truth=tn, which='true', dp0=0.0, dw=0.0, **r))
            P = dict(W=Wp.astype(np.float32), noise=nz.astype(np.float32), pos=pos.astype(np.float32), ch=p['ch'].astype(int))
            try:
                f = wr.fit_plane(P, v, cal, t0_prior=tuple(p['t0_prior']))
            except Exception:
                f = None
            if f is None:
                continue
            r = ratios(Wp, pos, f.p0, f.w, f.t0, sat)
            if r is not None:
                rows.append(dict(base, truth=tn, which='refit', dp0=float(f.p0 - p0), dw=float(f.w - w), dt0=float(f.t0 - t0), **r))
    return rows


def summarise(T, out):
    S = {}
    rng = np.random.default_rng(2)
    for (view, truth, which), g in T.groupby(['view', 'truth', 'which']):
        b = np.digitize(g.phi.values, PHB) - 1
        rec = dict(n=len(g), bins=[])
        for j in range(len(PHB) - 1):
            k = b == j
            if k.sum() < 15:
                rec['bins'].append(None)
                continue
            o = dict(n=int(k.sum()))
            for col in ('near', 'far', 'near2', 'far2'):
                x = g[col].values[k]
                med = float(np.median(x))
                e = float(np.std([np.median(rng.choice(x, len(x))) for _ in range(200)]))
                o[col], o[col + '_err'] = med, e
            rec['bins'].append(o)
        if which == 'refit':
            rec['dp0_mad_mm'] = float(1.4826 * np.median(np.abs(g.dp0 - np.median(g.dp0))))
            rec['dp0_med_mm'] = float(np.median(g.dp0))
            rec['dw_mad'] = float(1.4826 * np.median(np.abs(g.dw - np.median(g.dw))))
        S[f'{view}|{truth}|{which}'] = rec
    json.dump(dict(phase_bins=PHB.tolist(), early=EARLY, groups=S), open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    return S


def plot(S, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xc = 0.5 * (PHB[:-1] + PHB[1:])
    for view in ('x', 'y'):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
        for ia, col in enumerate(('near', 'far')):
            for key, style in (('data|pass', dict(color='k', lw=2.5, marker='o')),
                               ('noK_white|true', dict(color='C0', ls=':', marker='s')), ('noK_white|refit', dict(color='C0', lw=1.5, marker='s')),
                               ('pureK_white|true', dict(color='C2', ls=':', marker='^')), ('pureK_white|refit', dict(color='C2', lw=1.5, marker='^')),
                               ('noK_own|refit', dict(color='C9', lw=1.0, marker='s', alpha=0.6)), ('pureK_own|refit', dict(color='C8', lw=1.0, marker='^', alpha=0.6))):
                rec = S.get(f'{view}|{key}')
                if not rec:
                    continue
                y = [b[col] if b else np.nan for b in rec['bins']]
                e = [b[col + '_err'] if b else np.nan for b in rec['bins']]
                ax[ia].errorbar(xc, y, e, label=f'{key} (n={rec["n"]})', **style)
            ax[ia].set(xlabel='|phase| (pitch)', ylabel=f'{col} neighbour / centre, early window', title=f'{view}: {col}')
            ax[ia].grid(alpha=0.3)
        ax[1].set_ylim(0, 0.25)
        ax[0].legend(fontsize=7)
        fig.suptitle('Near/far early-charge ratio against position in strip: data, and known-truth windows with the pass refit of p0')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'figures', f'phase_closure_{view}.png'), dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wmax', type=float, default=1.2e-3)
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--nmax', type=int, default=0)
    ap.add_argument('--noise', default='own,white')
    a = ap.parse_args()
    from wft import model as wm, reco as wr
    from wft.calib import CalibrationBundle
    planes = []
    for f in a.pickles:
        planes += pickle.load(open(f, 'rb'))['planes']
    planes = [p for p in planes if abs(p['w']) <= a.wmax]
    if a.nmax:
        planes = planes[:a.nmax]
    cal = CalibrationBundle.load(a.bundle)
    wm.use_calibration(cal)
    wm.set_depth_bins(len(planes[0]['q']))
    G.update(wm=wm, wr=wr, cal=cal, hyper=dict(cal.hyper), truths=TRUTHS, wmax=a.wmax, noise=a.noise.split(','))
    os.makedirs(os.path.join(a.out, 'figures'), exist_ok=True)
    with mp.Pool(a.jobs) as P:
        rows = [r for rr in P.imap_unordered(one, planes, chunksize=4) for r in rr]
    T = pd.DataFrame(rows)
    T.to_parquet(os.path.join(a.out, 'rows.parquet'))
    S = summarise(T, a.out)
    plot(S, a.out)
    for k, rec in S.items():
        line = '  '.join(f"{b['near']:.3f}/{b['far']:.3f}" if b else '  -  ' for b in rec['bins'])
        extra = f"  dp0 mad {rec['dp0_mad_mm']:.3f} mm" if 'dp0_mad_mm' in rec else ''
        print(f'{k:16s} n {rec["n"]:5d}  near/far by |phase| bin: {line}{extra}')
    print(f'[phase_closure] {len(planes)} near-normal planes -> {a.out}')


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
