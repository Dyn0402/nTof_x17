#!/usr/bin/env python3
"""floor_toy.py -- the physics floor on the micro-TPC angle of one MX17 plane (F3, workstream E, 2026-10-27).

A first-principles toy, independent of the wft code: a minimum-ionising track crosses the drift gap; its
ionisation, drift, amplification and readout are generated electron by electron; the waveforms are then
fitted with the EXACT mean model of the toy (same template, footprint, diffusion, v, pitch, sampling) by the
same estimator class as the production reconstruction (free non-negative charge per 60 ns depth bin, free
p0 / tan / t0, chi2 weighted by the electronic noise, saturated samples censored), started at the truth.
What it returns is therefore the resolution of a reconstruction whose model is perfect: the physics floor
for this readout, with no model error, no seeding failure and no reference.

Physics, per event (X view: across the resistive strips, no RC spreading):
  ionisation   primary clusters Poisson along the path, LAMBDA_CL / mm (PDG: Ar 25, iC4H10 90 /cm ->
               95/5 = 28 /cm); cluster size from Fischle et al. (NIM A301 (1991) 202) for argon, 1/n^2 tail
               (delta rays) to N_MAX; clusters of >= N_DELTA electrons are delta rays whose electrons lie
               along a straight segment of the practical range R = 0.71 E^1.72 g/cm2 (E = n W, W = 26 eV),
               emitted perpendicular to the track at a random azimuth
  drift        v; transverse sigma_T^2 = Dp^2 u (u = drift time, Dp the bundle's, = D_T sqrt(v));
               longitudinal sigma_L = D_L sqrt(z) (Magboltz, bench Ar/iso 95/5 + ~0.5 % H2O)
  avalanche    single-electron gain Polya, theta = THETA (relative variance 1/(1+theta))
  readout      each electron's charge lands with a Gaussian footprint sigma0 (resistive spread +
               induction: a deterministic readout response, not a fluctuation), strips at 0.78 mm, the
               chamber's measured X pulse template (FWHM 300 ns), 60 ns sampling with a random phase,
               white noise sigma_n per sample, ADC ceiling 3550 (saturated samples censored in the fit)
  scale        ADC per electron set so the median total charge matches the chamber's measured q_sum

Also: the electron oracle (every electron's landing x and arrival time known, weighted straight-line fit:
the information limit of the ionisation + diffusion alone, any readout) and the Cramer-Rao bound of the
electronic noise alone on the mean waveform (deterministic cloud).

    floor_toy.py run   [--det det3] [--config full] [--n 200] [--jobs 4] [--est noise|poisson]
                                                    -> results/<det>_<config>[_poisson].json
    floor_toy.py crb   [--det det3]                                         -> results/<det>_crb.json
"""
import argparse
import json
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
AUDIT = os.path.expanduser('~/x17/cosmic_bench/model_audit')

PITCH = 0.78           # mm
DT_SAMP = 60.0         # ns, DREAM sampling period on the bench
NSAMP = 32
SAT = 3550.0           # ADC ceiling
DT_BIN = 60.0          # ns, depth bin of the fitted charge profile (as wft)
NSUB = 6               # sub-steps per depth bin in the fit model (10 ns)
W_ION = 26e-6          # MeV per ion pair, argon
RHO_GAS = 1.66e-3      # g/cm3, Ar/iso 95/5 at NTP
ANGLES = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)

# Fischle et al. 1991, argon, P(n) in % for n = 1..19; n >= 20: 21.6 / n^2 %
FISCHLE = [65.6, 15.0, 6.4, 3.5, 2.25, 1.55, 1.05, 0.81, 0.61, 0.49, 0.39, 0.30, 0.25, 0.20, 0.16, 0.12, 0.095,
           0.075, 0.063]
N_MAX = 1500           # largest cluster kept (~40 keV): beyond that the delta leaves the column
N_DELTA = 20           # clusters this large are treated as delta-ray tracks with a range

# per chamber: v [mm/ns] (bundle timing v), Dp [mm/sqrt(ns)] (phys bundle), sigma0_x [mm] (phys bundle),
# G [mm] (charge-column length = v * median uend), q50 [ADC] (median measured x_q_sum), noise [ADC]
CHAMBERS = {
    'det2': dict(v=0.0393, Dp=0.025, s0=0.238, G=30.0, q50=16558, noise=8.0, DL=0.250),
    'det3': dict(v=0.0358, Dp=0.023, s0=0.238, G=28.0, q50=11470, noise=8.0, DL=0.250),
    'det4': dict(v=0.0330, Dp=0.020, s0=0.238, G=28.0, q50=4314, noise=8.0, DL=0.250),
    'det6': dict(v=0.0242, Dp=0.018, s0=0.238, G=28.0, q50=22847, noise=8.0, DL=0.280),
    'det7': dict(v=0.0368, Dp=0.0225, s0=0.238, G=29.0, q50=25340, noise=8.5, DL=0.280),
}
THETA = 1.0            # Polya
KAPPA_FRAC = 0.19      # signal variance per unit prediction / ADC-per-electron, measured on the toy (det3, 20 deg,
                       # q profiled at the truth: <r^2 - noise^2> / mu = 11.5 ADC flat in mu from 20 to 3000 ADC)
LAMBDA_CL = 2.8        # clusters / mm

# configurations: what each one switches off
CONFIGS = {
    'full': {},
    'no_noise': dict(noise=0.0),
    'no_gainfl': dict(polya=False),
    'no_delta': dict(delta_range=False),
    'no_landau': dict(landau=False),          # Poisson electrons at the same mean density, no clustering
    'no_diff': dict(Dp=0.0, DL=0.0),
    'no_DL': dict(DL=0.0),
    'only_diff': dict(noise=0.0, polya=False, landau=False),   # diffusion + Poisson count only
    'only_ion': dict(noise=0.0, polya=False, Dp=0.0, DL=0.0),  # Landau clusters + delta rays only
}


def cluster_pdf():
    n = np.arange(1, N_MAX + 1)
    p = np.empty(N_MAX)
    p[:19] = FISCHLE
    p[19:] = 21.6 / n[19:] ** 2
    return n, p / p.sum()


CL_N, CL_P = cluster_pdf()
MEAN_CL = float((CL_N * CL_P).sum())


def load_template(det):
    z = np.load(os.path.join(AUDIT, det, 'calib_bundle', 'arrays.npz'))
    g, h = z['grid'].astype(float), z['tmpl_x'].astype(float)
    return g, h / h.max()


def params(det, config):
    P = dict(CHAMBERS[det])
    P.update(polya=True, delta_range=True, landau=True)
    P.update(CONFIGS[config])
    return P


# ---------------------------------------------------------------- generation
def electrons(tan, P, rng):
    """(z [mm, 0 = mesh], x [mm, track at x(z) = z tan]) of every ionisation electron."""
    G = P['G']
    th = np.arctan(tan)
    L = G / np.cos(th)
    if P['landau']:
        nc = rng.poisson(LAMBDA_CL * L)
        s = rng.uniform(0, L, nc)
        size = rng.choice(CL_N, nc, p=CL_P)
    else:
        ne = rng.poisson(LAMBDA_CL * MEAN_CL * L)
        s, size = rng.uniform(0, L, ne), np.ones(ne, int)
    z0, x0 = s * np.cos(th), s * np.sin(th)
    z, x = np.repeat(z0, size), np.repeat(x0, size)
    if P['delta_range'] and P['landau']:
        big = np.flatnonzero(size >= N_DELTA)
        if len(big):
            start = np.r_[0, np.cumsum(size)[:-1]]
            for i in big:
                E = size[i] * W_ION
                R = 0.71 * E ** 1.72 / RHO_GAS * 10.0          # mm
                phi = rng.uniform(0, 2 * np.pi)
                dx, dz = np.cos(phi) * np.cos(th), -np.cos(phi) * np.sin(th)
                f = rng.uniform(0, 1, size[i]) * R
                sl = slice(start[i], start[i] + size[i])
                x[sl] += f * dx
                z[sl] += f * dz
    ok = (z >= 0) & (z <= G)
    return z[ok], x[ok]


def drift(z, x, P, rng):
    """landing x and arrival time u [ns] after transverse/longitudinal diffusion."""
    v = P['v']
    u = z / v
    xl = x + rng.normal(0, 1, len(z)) * P['Dp'] * np.sqrt(u)
    sL = P['DL'] * np.sqrt(z / 10.0)                        # mm (D_L in mm/sqrt(cm))
    ul = u + rng.normal(0, 1, len(z)) * sL / v
    return xl, ul


def footprint(pos, xc, sig):
    """fraction of a Gaussian (centre xc, width sig) on each strip (pos, PITCH)."""
    if np.ndim(sig) == 0:
        sig = np.full(np.shape(xc), float(sig))
    zz = 1.0 / (np.sqrt(2) * np.maximum(sig, 1e-6))
    hi = (pos[:, None] + PITCH / 2 - xc[None, :]) * zz[None, :]
    lo = (pos[:, None] - PITCH / 2 - xc[None, :]) * zz[None, :]
    return 0.5 * (erf(hi) - erf(lo))                       # (n_strip, n)


def gen_event(tan, P, adc_per_e, tmpl, rng):
    g_t, h_t = tmpl
    z, x = electrons(tan, P, rng)
    xoff = rng.uniform(-PITCH / 2, PITCH / 2)
    x = x + xoff
    xl, ul = drift(z, x, P, rng)
    gain = rng.gamma(1 + THETA, 1.0 / (1 + THETA), len(z)) if P['polya'] else np.ones(len(z))
    t0 = 300.0 + rng.uniform(0, DT_SAMP)                   # arrival time of mesh charge on the sample clock
    xmax = P['G'] * tan
    lo, hi = min(0, xmax) - 6 * PITCH, max(0, xmax) + 6 * PITCH
    pos = np.arange(np.floor(lo / PITCH), np.ceil(hi / PITCH) + 1) * PITCH
    ts = np.arange(NSAMP) * DT_SAMP
    A = footprint(pos, xl, P['s0']) * (gain * adc_per_e)[None, :]          # (S, E)
    H = np.interp(ts[None, :] - (t0 + ul)[:, None], g_t, h_t, left=0, right=0)  # (E, T)
    W = A @ H
    W = W + rng.normal(0, 1, W.shape) * P['noise']
    sat = W >= SAT
    W = np.minimum(W, SAT)
    truth = dict(tan=tan, p0=xoff, t0=t0, ne=len(z), z=z, xl=xl, ul=ul, gain=gain)
    return W, sat, pos, truth


# ---------------------------------------------------------------- the fit (exact mean model)
class Model:
    def __init__(self, P, tmpl, pos):
        self.P, self.pos = P, pos
        self.g_t, self.h_t = tmpl
        self.K = int(np.ceil(P['G'] / P['v'] / DT_BIN)) + 2
        self.us = (np.arange(self.K * NSUB) + 0.5) * (DT_BIN / NSUB)       # sub-step drift times
        self.ts = np.arange(NSAMP) * DT_SAMP

    def matrix(self, p0, tan, t0):
        P = self.P
        xc = p0 + tan * P['v'] * self.us
        sig = np.sqrt(P['s0'] ** 2 + P['Dp'] ** 2 * self.us)
        F = footprint(self.pos, xc, sig)                                  # (S, K*NSUB)
        H = np.interp(self.ts[None, :] - (t0 + self.us)[:, None], self.g_t, self.h_t, left=0, right=0)
        S, K = len(self.pos), self.K
        M = np.matmul(F.reshape(S, K, NSUB).transpose(1, 0, 2), H.reshape(K, NSUB, NSAMP))  # (K, S, T)
        return M.reshape(K, S * NSAMP).T / NSUB                            # (S*T, K)


def fit_event(W, sat, pos, P, tmpl, start, kappa=0.0):
    """chi2 fit of (p0, tan, t0) with the charge profile q >= 0 profiled.  kappa = 0: weights from the electronic
    noise only (the production estimator).  kappa > 0: a second pass with each sample's variance
    noise^2 + kappa * mu (mu = the first pass's prediction): the electron-counting variance of the signal itself,
    the weighting a likelihood fit would use."""
    m = Model(P, tmpl, pos)
    keep = ~sat.ravel()
    scale = np.array([0.1, 0.005, 5.0])                     # p0 [mm], tan, t0 [ns]: one unit each
    x0 = np.array([start['p0'], start['tan'], start['t0']])
    sig = np.full(keep.sum(), max(P['noise'], 1.0))
    out = None
    for it in range(2 if kappa > 0 else 1):
        y = W.ravel()[keep] / sig
        yy = y @ y

        def solve(xs):
            M = m.matrix(*(x0 + xs * scale))[keep] / sig[:, None]
            # NNLS in Gram form (K x K): same minimiser, 50x faster than on the (samples x K) system
            A = M.T @ M
            R = np.linalg.cholesky(A + 1e-9 * np.trace(A) * np.eye(len(A))).T
            d = np.linalg.solve(R.T, M.T @ y)
            q, r = nnls(R, d)
            return yy - d @ d + r * r, M, q

        simplex = np.vstack([np.zeros(3), 1.5 * np.eye(3)])
        res = minimize(lambda xs: solve(xs)[0], np.zeros(3), method='Nelder-Mead',
                       options=dict(initial_simplex=simplex, xatol=0.01, fatol=0.05, maxiter=400))
        out = (x0 + res.x * scale, res.fun, res.nfev)
        if kappa > 0 and it == 0:
            _, M, q = solve(res.x)
            mu = (M * sig[:, None]) @ q
            sig = np.sqrt(max(P['noise'], 1.0) ** 2 + kappa * np.maximum(mu, 0.0))
            x0 = out[0]
    return out


def oracle(truth, P, charge=False):
    """weighted straight-line fit to every electron's (arrival depth, landing x): the information limit."""
    zr = truth['ul'] * P['v']
    x = truth['xl']
    tan = truth['tan']
    # per-electron variance about the line: transverse diffusion + the longitudinal one projected by tan
    var = P['Dp'] ** 2 * np.maximum(zr, 0.05) / P['v'] + (tan * P['DL']) ** 2 * np.maximum(zr, .05) / 10 + 1e-4
    w = 1.0 / var * (truth['gain'] if charge else 1.0)
    A = np.c_[np.ones_like(zr), zr]
    WA = A * w[:, None]
    beta = np.linalg.solve(A.T @ WA, WA.T @ x)
    return beta[1]


def adc_scale(P, rng, n=400):
    """ADC per electron that puts the median total charge at the chamber's measured q_sum (at the bench angle mix,
    |tan| ~ 0.25; with the footprint fractions summing to 1, q_sum = sum of gain x ADC/e)."""
    tot = []
    for _ in range(n):
        z, x = electrons(0.25, P | dict(landau=True, delta_range=True), rng)
        tot.append(rng.gamma(1 + THETA, 1.0 / (1 + THETA), len(z)).sum())
    return P['q50'] / np.median(tot)


def one(args):
    det, config, ang, seed, n, est = args
    P = params(det, config)
    rng = np.random.default_rng(seed)
    tmpl = load_template(det)
    adc = adc_scale(params(det, 'full'), np.random.default_rng(12345))
    tan = np.tan(np.radians(ang))
    rows = []
    for i in range(n):
        W, sat, pos, tr = gen_event(tan, P, adc, tmpl, rng)
        par, c2, _ = fit_event(W, sat, pos, P, tmpl, tr, kappa=KAPPA_FRAC * adc if est == 'poisson' else 0.0)
        rows.append((ang, np.degrees(np.arctan(par[1])) - ang, par[0] - tr['p0'], par[2] - tr['t0'],
                     np.degrees(np.arctan(oracle(tr, P))) - ang,
                     np.degrees(np.arctan(oracle(tr, P, charge=True))) - ang,
                     tr['ne'], float(sat.mean()), c2 / max(sat.size - sat.sum() - 3, 1)))
    return rows


def s68(a):
    """the fleet-table convention (fleet_tables.py, measured.py): 68th percentile of |d - median| with the far
    (> 5 deg) tail set aside, which is quoted separately"""
    a = np.asarray(a)
    a = a[np.abs(a) <= 5]
    return float(np.percentile(np.abs(a - np.median(a)), 68))


def summarise(rows):
    rows = np.array(rows)
    out = {}
    for ang in ANGLES:
        r = rows[rows[:, 0] == ang]
        if not len(r):
            continue
        out[str(ang)] = dict(n=len(r), s68=s68(r[:, 1]), rms=float(np.std(r[:, 1])), bias=float(np.median(r[:, 1])),
                             far5=float(np.mean(np.abs(r[:, 1]) > 5)),
                             oracle_s68=s68(r[:, 4]), oracle_q_s68=s68(r[:, 5]),
                             p0_s68=s68(r[:, 2]), t0_s68=s68(r[:, 3]), ne_med=float(np.median(r[:, 6])),
                             sat_frac=float(r[:, 7].mean()), chi2dof=float(np.median(r[:, 8])))
    return out


def run(a):
    global DT_BIN, NSUB, SAT
    os.makedirs(OUT, exist_ok=True)
    if a.dtbin:
        DT_BIN, NSUB = float(a.dtbin), max(int(round(a.dtbin / 10.0)), 2)   # workers fork after this
    if a.sat:
        SAT = float(a.sat)              # ADC ceiling (clip + censor), same seeds -> event-paired across ceilings
    jobs = []
    chunk = 25
    for k, ang in enumerate(ANGLES):
        if a.angles and str(ang) not in a.angles.split(','):
            continue
        for c in range(a.n // chunk):
            jobs.append((a.det, a.config, ang, 1000 * k + c + 100000 * list(CONFIGS).index(a.config), chunk, a.est))
    t = time.time()
    with Pool(a.jobs) as pool:
        rows = [r for part in pool.map(one, jobs) for r in part]
    res = dict(det=a.det, config=a.config, estimator=a.est, dt_bin=DT_BIN, params=params(a.det, a.config), theta=THETA, lambda_cl=LAMBDA_CL,
               mean_cluster=MEAN_CL, n_max=N_MAX, n_delta=N_DELTA, seconds=time.time() - t,
               by_angle=summarise(rows))
    path = os.path.join(OUT, f'{a.det}_{a.config}' + ('_poisson' if a.est == 'poisson' else '')
                        + (f'_bin{int(a.dtbin)}' if a.dtbin else '') + (f'_sat{int(a.sat)}' if a.sat else '') + '.json')
    json.dump(res, open(path, 'w'), indent=1)
    np.save(path.replace('.json', '.npy'), np.array(rows))
    print(path, f'{time.time() - t:.0f} s')
    for k, v in res['by_angle'].items():
        print(f"{k:>3}  s68 {v['s68']:.3f}  rms {v['rms']:.3f}  bias {v['bias']:+.3f}  oracle {v['oracle_s68']:.3f}"
              f"  oracle_q {v['oracle_q_s68']:.3f}  ne {v['ne_med']:.0f}  sat {v['sat_frac']:.4f}  chi2/dof {v['chi2dof']:.2f}")


def crb(a):
    """Cramer-Rao bound of the electronic noise alone: Fisher information of the mean waveform (uniform charge
    column, the chamber's median charge) in (tan, p0, t0, q_1..q_K), noise sigma_n per sample."""
    P = params(a.det, 'full')
    tmpl = load_template(a.det)
    out = {}
    for ang in ANGLES:
        tan = np.tan(np.radians(ang))
        xmax = P['G'] * tan
        pos = np.arange(np.floor((min(0, xmax) - 6 * PITCH) / PITCH), np.ceil((max(0, xmax) + 6 * PITCH) / PITCH) + 1) * PITCH
        vals = []
        for ph in np.linspace(0, DT_SAMP, 6, endpoint=False):
            m = Model(P, tmpl, pos)
            t0 = 300 + ph
            nb = P['G'] / P['v'] / DT_BIN
            q = np.where(np.arange(m.K) < np.floor(nb), 1.0, 0.0)
            q[int(np.floor(nb))] = nb - np.floor(nb)
            q *= P['q50'] / q.sum() / np.cos(np.radians(ang)) * np.cos(np.radians(14))
            def mu(p0, tn, tt):
                return m.matrix(p0, tn, tt) @ q
            e = dict(p0=1e-3, tan=1e-4, t0=0.05)
            J = [(mu(e['p0'], tan, t0) - mu(-e['p0'], tan, t0)) / (2 * e['p0']),
                 (mu(0, tan + e['tan'], t0) - mu(0, tan - e['tan'], t0)) / (2 * e['tan']),
                 (mu(0, tan, t0 + e['t0']) - mu(0, tan, t0 - e['t0'])) / (2 * e['t0'])]
            M = m.matrix(0, tan, t0)
            used = q > 0
            J = np.c_[np.array(J).T, M[:, used]] / P['noise']
            C = np.linalg.pinv(J.T @ J)
            vals.append(np.sqrt(C[1, 1]))
        st = float(np.mean(vals))
        out[str(ang)] = dict(sig_tan=st, sig_deg=float(np.degrees(st) * np.cos(np.radians(ang)) ** 2))
        print(ang, out[str(ang)])
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(det=a.det, params=P, by_angle=out), open(os.path.join(OUT, f'{a.det}_crb.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('run', 'crb'))
    ap.add_argument('--det', default='det3')
    ap.add_argument('--config', default='full', choices=list(CONFIGS))
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--dtbin', type=float, default=0.0, help='depth-bin width of the fitted charge profile [ns] '
                    '(default 60, as wft)')
    ap.add_argument('--angles', default='', help='comma list, default all')
    ap.add_argument('--sat', type=float, default=0.0, help='ADC ceiling (default 3550): the clip-information test '
                    '(rerun/calib/saturation/ERROR_MODEL.md)')
    ap.add_argument('--est', default='noise', choices=('noise', 'poisson'),
                    help='noise: chi2 weighted by the electronic noise (production); poisson: + the signal variance')
    a = ap.parse_args()
    {'run': run, 'crb': crb}[a.cmd](a)


if __name__ == '__main__':
    main()
