#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ac_tof_beamoff.py -- the arm-to-arm time resolution sigma_t of the SiPM walls (and plastics), from cosmic muons
crossing arms A and C while n_TOF recorded with NO protons (run_149, n_TOF 224678-224687, 8.6 h).

Why beam-off: in the beam runs the n_TOF pulse-shape fit times the SiPM wall with a floor of 2.5-4.5 ns per hit at
5-50 us (injected test pulses, ntof_scint_recovery/ANALYSIS_PLAN.md), and the A-C muons of the beam runs give a
~6 ns wide t_A - t_C whatever is corrected (ac_tof.py).  Without beam the same injected pulses give 0.4 ns (wall)
and 0.07 ns (plastic), so only these runs can see the detector.

Why a model fit: only 17 of these muons also have an A-C Micromegas line (n_TOF records ~16 % of the time,
HANDOFF_TRACKING_2026-10-06.md §10d), too few to subtract each muon's own time of flight.  The muon TOF between the
walls is +-T with T = L/c = 2.2-2.6 ns (A and C at the same height: the sign is which wall is higher on the track).
So t_A - t_C is a two-peak mixture,

    x = o(g_A) - o(g_C) + w_A log a_A + w_C log a_C + s T + eps,    eps ~ N(0, sigma),   s = +-1,

with P(s = +1) = logistic(alpha + beta (d_C - d_A)) learnt from each bar's top-bottom difference d (the hit height
along the vertical bar: the higher crossing should fire first), T drawn from the geometry of the beam-run A-C muons
with a Micromegas line (ac_tof.py events, |pred_w|), plus a flat accidental fraction over the +-25 ns window.
sigma / sqrt(2) is the per-arm resolution if the arms are equal.  A toy (same fitted parameters, same T template,
same N) checks the fit returns its input.

    .venv/bin/python ntof_cosmics/ac_tof_beamoff.py         -> $WORK_DATA/nTof_x17/ac_tof/beamoff_*.csv, figures/
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from scipy.optimize import minimize
from scipy.special import expit

NTOF = Path('/media/dylan/data/x17/beam_july/ntof_data')
RUNS = list(range(224678, 224688))
OUT = Path(os.environ.get('WORK_DATA', '/media/dylan/data/x17')) / 'nTof_x17' / 'ac_tof'
TOF_MIN = 5e4               # drop the first 50 us (the ~12 us pickup)
WIN = 25.0                  # A-C coincidence window [ns]
PLAS_WIN = 100.0            # plastic hit within this of its arm's wall time [ns] (the PSS trees sit ~35-42 ns off the walls)


def s68(x):
    x = np.asarray(x)[np.isfinite(x)]
    q = np.percentile(x, [15.865, 84.135])
    return 0.5 * (q[1] - q[0])


def bars(a):
    """wall: pair the two ends of each group (detn 2g+1, 2g+2) within 15 ns -> (bunch, mean t, geo-mean amp, t1-t2, g)."""
    out = []
    k = a['BunchNumber'].astype(np.int64)
    for g in range(4):
        e1, e2 = a['detn'] == 2 * g + 1, a['detn'] == 2 * g + 2
        k1, t1, a1 = k[e1], a['tof'][e1], a['amp'][e1]
        k2, t2, a2 = k[e2], a['tof'][e2], a['amp'][e2]
        if len(k2) < 2 or len(k1) == 0:
            continue
        o = np.lexsort((t2, k2))
        k2, t2, a2 = k2[o], t2[o], a2[o]
        key2, key1 = k2 * 1e9 + t2, k1 * 1e9 + t1
        j = np.clip(np.searchsorted(key2, key1), 1, len(key2) - 1)
        jj = np.where(np.abs(key2[j] - key1) < np.abs(key2[j - 1] - key1), j, j - 1)
        d = t1 - t2[jj]
        ok = (k2[jj] == k1) & (np.abs(d) < 15)
        out.append(np.c_[k1[ok], 0.5 * (t1[ok] + t2[jj][ok]), np.sqrt(a1[ok] * a2[jj][ok]), d[ok], np.full(ok.sum(), g)])
    return np.concatenate(out) if out else np.zeros((0, 5))


def build() -> pd.DataFrame:
    rows = []
    for r in RUNS:
        for fn in sorted(glob.glob(str(NTOF / f'run{r}.parts' / '*.root'))):
            f = uproot.open(fn)
            W, P = {}, {}
            for arm in 'AC':
                a = f['WAL' + arm].arrays(['BunchNumber', 'detn', 'tof', 'amp', 'satuflag'], library='np')
                m = (a['tof'] > TOF_MIN) & (a['satuflag'] == 0)
                W[arm] = bars({q: v[m] for q, v in a.items()})
                p = f['PSS' + arm].arrays(['BunchNumber', 'detn', 'tof', 'amp', 'satuflag'], library='np')
                m = (p['tof'] > TOF_MIN) & (p['satuflag'] == 0)
                P[arm] = {q: v[m] for q, v in p.items()}
            A, C = W['A'], W['C']
            if not len(A) or not len(C):
                continue
            kC = C[:, 0] * 1e9 + C[:, 1]
            o = np.argsort(kC)
            C, kC = C[o], kC[o]
            kA = A[:, 0] * 1e9 + A[:, 1]
            lo, hi = np.searchsorted(kC, kA - WIN), np.searchsorted(kC, kA + WIN)
            # plastics: per arm, sorted keys
            pk = {}
            for arm in 'AC':
                q = P[arm]
                key = q['BunchNumber'].astype(np.int64) * 1e9 + q['tof']
                oo = np.argsort(key)
                pk[arm] = (key[oo], q['amp'][oo], q['detn'][oo], q['tof'][oo])

            def plastic(arm, key):
                kk, aa, dd, tt = pk[arm]
                i0, i1 = np.searchsorted(kk, key - PLAS_WIN), np.searchsorted(kk, key + PLAS_WIN)
                if i1 <= i0:
                    return np.nan, np.nan, -1
                j = i0 + int(np.argmax(aa[i0:i1]))
                return tt[j], aa[j], int(dd[j])
            for i in np.flatnonzero(hi - lo == 1):
                c = C[lo[i]]
                tpA, apA, bA = plastic('A', kA[i])
                tpC, apC, bC = plastic('C', kC[lo[i]])
                rows.append(dict(ntof_run=r, file=Path(fn).name, bunch=int(A[i, 0]), tA=A[i, 1], tC=c[1], aA=A[i, 2], aC=c[2],
                                 dA=A[i, 3], dC=c[3], gA=int(A[i, 4]), gC=int(c[4]),
                                 tpA=tpA, apA=apA, bA=bA, tpC=tpC, apC=apC, bC=bC))
    D = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT / 'beamoff_coinc.parquet', index=False)
    print(f'{len(D)} single A-C wall coincidences (|dt| < {WIN} ns, both ends, unsaturated)', flush=True)
    return D


def t_template(n=200):
    """|T| = L/c between the walls, from the beam-run A-C muons with a Micromegas line (ac_tof.py events)."""
    E = pd.read_parquet(OUT / 'ac_tof_events.parquet')
    m = (E.sep < 20) & (E.jdca > 60) & (E.ms >= 10)
    T = np.abs(E.pred_w[m].to_numpy())
    return np.quantile(T, (np.arange(n) + 0.5) / n)


# --------------------------------------------------------------------------- #
# the mixture fit
# --------------------------------------------------------------------------- #
def design(D):
    """columns: 3 A-group offsets, 3 C-group offsets, walk A, walk C (constant separate)."""
    gA, gC = D.gA.to_numpy(int), D.gC.to_numpy(int)
    cols = [(gA == g).astype(float) for g in (1, 2, 3)] + [-(gC == g).astype(float) for g in (1, 2, 3)]
    la, lc = np.log(D.aA.to_numpy(float)), np.log(D.aC.to_numpy(float))
    cols += [la - la.mean(), lc - lc.mean()]
    return np.c_[tuple(cols)]


def centred_d(D):
    """top-bottom difference, centred per arm and group (removes each group's end-to-end cable offset)."""
    dA = D.dA - D.groupby('gA').dA.transform('median')
    dC = D.dC - D.groupby('gC').dC.transform('median')
    return dA.to_numpy(float), dC.to_numpy(float)


def nll(p, x, X, dd, T):
    mu0, lsig, lf, al, be = p[:5]
    beta = p[5:]
    sig, fbg = np.exp(lsig), expit(lf)
    y = x - mu0 - X @ beta
    pp = expit(al + be * dd)[:, None]
    g = lambda m: np.exp(-0.5 * ((y[:, None] - m) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)
    sigl = (pp * g(T[None, :]) + (1 - pp) * g(-T[None, :])).mean(1)
    L = (1 - fbg) * sigl + fbg / (2 * WIN)
    return -np.sum(np.log(L + 1e-300))


def fit(x, X, dd, T, p0=None):
    if p0 is None:
        p0 = np.r_[np.median(x), np.log(1.0), -3.0, 0.0, 0.1, np.zeros(X.shape[1])]
    best = None
    for s0 in (0.5, 1.0, 2.0):
        q = p0.copy()
        q[1] = np.log(s0)
        r = minimize(nll, q, args=(x, X, dd, T), method='L-BFGS-B',
                     bounds=[(None, None), (np.log(0.05), np.log(10)), (-12, 0), (-5, 5), (-5, 5)] + [(None, None)] * X.shape[1])
        if best is None or r.fun < best.fun:
            best = r
    return best


def errors(r, x, X, dd, T):
    """sigma's error from the numerical Hessian of the NLL at the minimum."""
    p = r.x
    n = len(p)
    h = 1e-3
    H = np.zeros((n, n))
    f0 = nll(p, x, X, dd, T)
    for i in range(n):
        for j in range(i, n):
            e_i, e_j = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = H[j, i] = (nll(p + e_i + e_j, x, X, dd, T) - nll(p + e_i - e_j, x, X, dd, T)
                                 - nll(p - e_i + e_j, x, X, dd, T) + nll(p - e_i - e_j, x, X, dd, T)) / (4 * h * h)
    C = np.linalg.pinv(H)
    return np.sqrt(np.diag(C))


def toy(r, X, dd, T, rng):
    p = r.x
    mu0, sig, fbg, al, be = p[0], np.exp(p[1]), expit(p[2]), p[3], p[4]
    n = len(dd)
    s = np.where(rng.random(n) < expit(al + be * dd), 1.0, -1.0)
    Tt = rng.choice(T, n)
    x = mu0 + X @ p[5:] + s * Tt + rng.normal(0, sig, n)
    bg = rng.random(n) < fbg
    x[bg] = rng.uniform(-WIN, WIN, bg.sum()) + mu0
    return x


#: the machine-readable result other repos read (x17_facility_search lnl/sim/lnl_geant.py TOF_SIGMA_NS, ...)
TIMING_JSON = Path(__file__).resolve().parent / 'results' / 'ac_tof' / 'timing.json'


def write_timing(S: pd.DataFrame | None = None) -> dict:
    """results/ac_tof/timing.json from beamoff_summary.csv: per-arm sigma_t of the walls (range = direction-tagged
    peak cores .. two-peak fit), the reference value (their mean, rounded to 0.05 ns), and the plastics."""
    if S is None:
        S = pd.read_csv(OUT / 'beamoff_summary.csv')
    fitv = float(S.loc[S['sample'] == 'walls, all', 'per_arm_ns'].iloc[0])
    tag = S[S['sample'].str.startswith('direction-tagged')].per_arm_ns.astype(float)
    lo, hi = float(tag.mean()), fitv
    pl = S[S['sample'].str.startswith('plastics')].iloc[0]
    d = dict(
        quantity='arm-to-arm time resolution sigma_t PER ARM (sigma(t_A - t_C) / sqrt 2, arms assumed equal), ns',
        wall_sipm=dict(reference=round(round(0.5 * (lo + hi) / 0.05) * 0.05, 2), low=round(lo, 3), high=round(hi, 3)),
        plastic=dict(reference=round(float(pl.per_arm_ns), 2), err=round(float(pl.per_arm_err_ns), 2), n=int(pl.n)),
        conditions='as built at n_TOF: SiPM walls (4 bars ganged per channel, both ends), n_TOF digitiser 1 GS/s and '
                   'official pulse fit, NO beam (run_149, n_TOF 224678-224687). Under beam the pulse fit floors at '
                   '~4 ns per arm. Not a prediction for another readout.',
        method='ntof_cosmics/ac_tof_beamoff.py: A-C cosmic muons, two-peak fit (muon TOF +-T) and direction-tagged peak '
               'cores; report results/ac_tof/report.html',
        n_wall=int(S.loc[S['sample'] == 'walls, all', 'n'].iloc[0]), date='2026-11-19')
    TIMING_JSON.parent.mkdir(parents=True, exist_ok=True)
    TIMING_JSON.write_text(json.dumps(d, indent=1) + '\n')
    return d


def main() -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = OUT / 'beamoff_coinc.parquet'
    D = pd.read_parquet(f) if f.exists() and '--rebuild' not in sys.argv else build()
    T = t_template()
    print(f'{len(D)} coincidences; muon TOF template |T|: median {np.median(T):.2f} ns, 5-95 % {np.percentile(T, 5):.2f}-{np.percentile(T, 95):.2f}')
    x = (D.tA - D.tC).to_numpy(float)
    X = design(D)
    dA, dC = centred_d(D)
    dd = dC - dA
    rows = []
    res = {}

    def run(name, m, Tt=T, note=''):
        r = fit(x[m], X[m], dd[m], Tt)
        e = errors(r, x[m], X[m], dd[m], Tt)
        sig, esig = np.exp(r.x[1]), np.exp(r.x[1]) * e[1]
        row = dict(sample=name, n=int(m.sum()), sigma_AC_ns=sig, err_ns=esig, per_arm_ns=sig / np.sqrt(2),
                   per_arm_err_ns=esig / np.sqrt(2), f_bg=float(expit(r.x[2])), alpha=r.x[3], beta=r.x[4], note=note)
        rows.append(row)
        res[name] = (r, m)
        print(f'{name:52s} n {row["n"]:5d}  sigma(A-C) {sig:.3f} ± {esig:.3f} ns  per arm {sig / np.sqrt(2):.3f}  '
              f'f_bg {row["f_bg"]:.3f}  beta {r.x[4]:+.3f}/ns', flush=True)
        return r
    allm = np.ones(len(D), bool)
    r0 = run('walls, all', allm)
    # stability
    big = (D.aA > D.aA.median()) & (D.aC > D.aC.median())
    run('walls, both amplitudes above median', big.to_numpy())
    run('walls, T x 0.9', allm, T * 0.9, 'template scale')
    run('walls, T x 1.1', allm, T * 1.1, 'template scale')
    run('walls, no direction info (beta = 0 forced via shuffled d)', allm, T, 'top-bottom shuffled')  # placeholder, replaced below
    # really shuffle d for that row
    rng = np.random.default_rng(1)
    dd_s = rng.permutation(dd)
    r = fit(x, X, dd_s, T)
    rows[-1].update(sigma_AC_ns=np.exp(r.x[1]), per_arm_ns=np.exp(r.x[1]) / np.sqrt(2), beta=r.x[4], err_ns=np.nan, per_arm_err_ns=np.nan)
    print(f'  (shuffled d: sigma(A-C) {np.exp(r.x[1]):.3f} ns, beta {r.x[4]:+.3f})')
    # by n_TOF run halves (time stability)
    early = D.ntof_run.to_numpy() <= 224682
    run('walls, n_TOF 224678-682', early)
    run('walls, n_TOF 224683-687', ~early)
    # toy closure
    trng = np.random.default_rng(7)
    ts = []
    for _ in range(5):
        xt = toy(r0, X, dd, T, trng)
        rt = fit(xt, X, dd, T, p0=r0.x.copy())
        ts.append(np.exp(rt.x[1]))
    print(f'toy closure: input sigma {np.exp(r0.x[1]):.3f}, refit {np.mean(ts):.3f} ± {np.std(ts):.3f} (5 toys)')
    rows.append(dict(sample='toy closure (5 toys of the "walls, all" fit)', n=len(D), sigma_AC_ns=float(np.mean(ts)),
                     err_ns=float(np.std(ts)), per_arm_ns=float(np.mean(ts)) / np.sqrt(2), note=f'input {np.exp(r0.x[1]):.3f}'))
    # simple width estimators for comparison (what the first look quoted)
    y = x - r0.x[0] - X @ r0.x[5:]
    rows.append(dict(sample='s68 of t_A - t_C after the fitted offsets + walk (muon TOF NOT removed)', n=len(D),
                     sigma_AC_ns=s68(y), per_arm_ns=s68(y) / np.sqrt(2)))
    # direction-tagged peaks: |d_C - d_A| > 4 ns picks the direction almost every time (the figure, right); each peak's
    # Gaussian core (iterated 2 sigma) needs no T template.  It still holds the spread of T within the tag (~0.1 ns).
    from scipy.stats import norm
    for lab, m in (('dd < -4', dd < -4), ('dd > +4', dd > 4)):
        z = y[m]
        mu, sd = np.median(z), s68(z)
        for _ in range(20):
            k = np.abs(z - mu) < 2 * sd
            mu, sd = z[k].mean(), z[k].std() / np.sqrt(1 - 4 * norm.pdf(2) / (2 * norm.cdf(2) - 1))
        rows.append(dict(sample=f'direction-tagged peak ({lab}), Gaussian core', n=int(m.sum()), sigma_AC_ns=sd,
                         per_arm_ns=sd / np.sqrt(2), note=f'peak at {mu:+.2f} ns'))
        print(f'tagged {lab}: n {m.sum()}  peak {mu:+.2f}  core sigma(A-C) {sd:.3f}  per arm {sd / np.sqrt(2):.3f}')
    # plastics: same model on events with a plastic hit in both arms, offsets per bar
    pm = (D.bA >= 0) & (D.bC >= 0) & np.isfinite(D.tpA) & np.isfinite(D.tpC)
    P = D[pm].reset_index(drop=True)
    xp = (P.tpA - P.tpC).to_numpy(float)
    bA, bC = P.bA.to_numpy(int), P.bC.to_numpy(int)
    la, lc = np.log(P.apA.to_numpy(float)), np.log(P.apC.to_numpy(float))
    # the plastic bars have one PMT: the light path along the bar is not cancelled.  The wall's top-bottom gives the
    # height, the wall group the transverse position: offsets per (bar, wall group) and a linear term in d.
    Xp = np.c_[(bA == 2).astype(float), -(bC == 2).astype(float), la - la.mean(), lc - lc.mean(),
               *[(P.gA.to_numpy() == g).astype(float) for g in (1, 2, 3)], *[-(P.gC.to_numpy() == g).astype(float) for g in (1, 2, 3)],
               dA[pm.to_numpy()], dC[pm.to_numpy()]]
    keep = np.abs(xp - np.median(xp)) < WIN
    rp = fit(xp[keep], Xp[keep], dd[pm.to_numpy()][keep], T)
    ep = errors(rp, xp[keep], Xp[keep], dd[pm.to_numpy()][keep], T)
    sp = np.exp(rp.x[1])
    rows.append(dict(sample='plastics (bar, group offsets, walk, height terms)', n=int(keep.sum()), sigma_AC_ns=sp, err_ns=sp * ep[1],
                     per_arm_ns=sp / np.sqrt(2), per_arm_err_ns=sp * ep[1] / np.sqrt(2), f_bg=float(expit(rp.x[2])), beta=rp.x[4]))
    print(f'plastics: n {keep.sum()}  sigma(A-C) {sp:.3f} ± {sp * ep[1]:.3f} ns  per arm {sp / np.sqrt(2):.3f}  f_bg {expit(rp.x[2]):.3f}')
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'beamoff_summary.csv', index=False)
    write_timing(S)
    (OUT / 'beamoff_meta.json').write_text(json.dumps(dict(runs=RUNS, tof_min_ns=TOF_MIN, win_ns=WIN, n=len(D),
                                                           T_median=float(np.median(T)), fit_walls_all=list(map(float, r0.x))), indent=1))
    # figure: residual with offsets + walk removed, split by the direction estimate, model overlaid
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.arange(-10, 10.01, 0.4)
    q = np.abs(dd) > np.percentile(np.abs(dd), 50)
    sg, mu0 = np.exp(r0.x[1]), r0.x[0]
    for lab, m, c in (('d_C − d_A > 0 (half with larger |d|)', q & (dd > 0), 'C0'), ('d_C − d_A < 0', q & (dd < 0), 'C3'),
                      ('all', allm, 'k')):
        ax[0].hist(y[m], bins, histtype='step', color=c, label=f'{lab}: {m.sum()}')
    xx = np.linspace(-10, 10, 400)
    pp = expit(r0.x[3] + r0.x[4] * dd)
    fb = expit(r0.x[2])
    dens = np.zeros_like(xx)
    for s, w in ((1, pp.mean()), (-1, 1 - pp.mean())):
        dens += w * np.mean(np.exp(-0.5 * ((xx[:, None] - s * T[None, :]) / sg) ** 2), 1) / (np.sqrt(2 * np.pi) * sg)
    ax[0].plot(xx, len(D) * 0.4 * ((1 - fb) * dens + fb / (2 * WIN)), 'k--', lw=1, label=f'model, σ = {sg:.2f} ns')
    ax[0].set(xlabel='t_A − t_C − offsets − walk [ns]', ylabel='coincidences / 0.4 ns',
              title=f'SiPM walls A–C, beam off (run_149): {len(D)} muons')
    ax[0].legend(fontsize=7)
    ax[1].scatter(dd, y, s=2, alpha=0.3)
    ax[1].set(xlabel='d_C − d_A (top−bottom, centred) [ns]', ylabel='t_A − t_C − offsets − walk [ns]', ylim=(-10, 10),
              xlim=(-15, 15), title='direction from the bar heights')
    fig.tight_layout()
    (OUT / 'figures').mkdir(exist_ok=True)
    fig.savefig(OUT / 'figures' / 'beamoff_walls.png', dpi=120)
    print('->', OUT / 'beamoff_summary.csv', OUT / 'figures' / 'beamoff_walls.png')
    return 0


if __name__ == '__main__':
    if sys.argv[1:2] == ['timing']:          # rewrite timing.json from the existing summary
        print(json.dumps(write_timing(), indent=1))
        raise SystemExit(0)
    raise SystemExit(main())
