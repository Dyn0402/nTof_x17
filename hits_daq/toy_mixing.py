#!/usr/bin/env python3
"""toy_mixing.py -- a transparent toy of the single-time-per-strip angle bias: geometry, lateral spreads, template, estimator.

One inclined track crosses a drift gap L at tan(theta); charge from drift time u lands at x = x0 + v tan(theta) u with a Gaussian
lateral spread sigma^2(u) = sigma0^2 + Dp^2 u (prompt footprint + drift diffusion), is integrated over strips of pitch p, and
every increment is seen through the electronics template T(t) (the measured det3 impulse response, or an analytic CR-RC^n),
sampled every SNS ns with white noise.  Ionisation is either uniform or clustered (Poisson clusters along the track, 1/n^2
cluster sizes, exponential gain per electron).  The hit emulators of emulate.py (DREAM leading edge, charge-weighted time,
template fit, peak) and the ladder fits of remedies.py (plain, robust, edge cut, edge shift) give u_h = dx/dt; the ratio
u_h / w (> 1 = too steep) is scored per angle as a median over tracks.

The toy exists to show the scaling: ratio - 1 = C / N^2 + b, N = L tan / p the strips in the column, C the strip-end
coefficient (geometry + lateral spreads), b the interior compression from the depth-growing spread.  Scans: the det3 arms
(validation against resim), pitch, sigma0, Dp, estimator, template width and sampling, and other detectors' parameters.

Usage: toy_mixing.py --scan {arms,pitch,estimators,detectors,universal} --out results/toy_<scan>.json [--ntr 300]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emulate as E  # noqa: E402
from remedies import edge_fix, line_u  # noqa: E402

TEMPLATE_NPZ = '/home/dylan/.claude/jobs/bb4311c1/tmp/det3_template.npz'


def det3_template():
    z = np.load(TEMPLATE_NPZ)
    return np.asarray(z['grid'], float), np.asarray(z['tmpl_x'], float)


def crrc_template(tp, n=2, step=2.0, span=2000.0):
    """CR-RC^n with peaking time tp [ns], unit peak, on a fine grid starting at 0"""
    g = np.arange(0.0, span, step)
    tau = tp / n
    h = (g / tau) ** n * np.exp(-g / tau)
    return g, h / h.max()


class Toy:
    def __init__(self, L=28.08, pitch=0.78, v=0.0358, sigma0=0.30, Dp=0.021, template='det3', tp=120.0, sns=60.0, nsamp=32,
                 noise=20.0, mip_adc=700.0, ion='cluster', t0_sample=5, du=5.0, seed=0):
        self.L, self.p, self.v, self.s0, self.Dp = L, pitch, v, sigma0, Dp
        self.sns, self.nsamp, self.noise, self.mip = sns, nsamp, noise, mip_adc
        self.ion, self.du = ion, du
        E.SNS = float(sns)                       # the emulators read the sampling period from the module constant
        self.T = L / v
        self.t0 = t0_sample * sns
        if template == 'det3':
            self.tg, self.tt = det3_template()
        else:
            self.tg, self.tt = crrc_template(tp)
        self.ts = np.arange(nsamp) * sns
        self.rng = np.random.default_rng(seed)
        # the template value at (sample - arrival): matrix over fine arrival times
        self.uk = np.arange(0.0, self.T, du) + du / 2
        base = self.ts[:, None] - (self.t0 + self.uk[None, :])                  # (nsamp, K)
        self.H = np.interp(base, self.tg, self.tt, left=0.0, right=0.0)
        # charge scale: a MIP's charge per fine step such that a normal track's centre strip peaks near mip_adc at tan=0.15
        self.q_step = mip_adc / (0.9 * self.H.sum(0).max() * (self.p / (v * 0.15 * du)) * 0.5) if v > 0 else 1.0
        self._calib_scale(mip_adc)
        self.ideal = E.Ideal(self.tg, self.tt, nsamp) if template == 'det3' else E.Ideal(self.tg - self.tg[np.argmax(self.tt)] + 140.0, self.tt, nsamp)

    def _calib_scale(self, mip_adc):
        # set q_step so that the median centre-strip peak at tan 0.15 is mip_adc (uniform ionisation, no noise)
        self.q_step = 1.0
        W, _, _ = self.waveforms(0.15, 0.0, uniform=True)
        pk = W.max(1).max()
        self.q_step = mip_adc / pk

    def charges(self, tan):
        K = len(self.uk)
        if self.ion == 'uniform':
            return np.full(K, self.q_step)
        # clustered: Poisson clusters per fine step along the track (25/cm in Ar at 1 bar), sizes P(n) ~ 1/n^2 (n = 1..150),
        # exponential gain per electron
        path_cm = self.du * self.v * np.sqrt(1 + tan ** 2) / 10.0
        ncl = self.rng.poisson(25.0 * path_cm, K)
        q = np.zeros(K)
        sizes = np.arange(1, 151)
        pn = 1.0 / sizes ** 2
        pn /= pn.sum()
        for k in np.flatnonzero(ncl):
            n_e = self.rng.choice(sizes, ncl[k], p=pn).sum()
            q[k] = self.rng.gamma(n_e, 1.0)
        mean_e = (sizes * pn).sum() * 25.0 * path_cm            # electrons per step on average
        return q * self.q_step / mean_e

    def waveforms(self, tan, x0, uniform=False, nstrip=None):
        w = self.v * tan
        xk = x0 + w * self.uk
        sig = np.sqrt(self.s0 ** 2 + self.Dp ** 2 * self.uk + (w * self.du) ** 2 / 12.0)
        lo_x, hi_x = min(xk.min(), x0) - 4 * self.p, max(xk.max(), x0) + 4 * self.p
        pos = np.arange(np.floor(lo_x / self.p) * self.p, hi_x + self.p, self.p)
        z = 1.0 / (np.sqrt(2) * sig)[None, :]
        F = 0.5 * (erf((pos[:, None] + self.p / 2 - xk[None, :]) * z) - erf((pos[:, None] - self.p / 2 - xk[None, :]) * z))
        q = np.full(len(self.uk), self.q_step) if (uniform or self.ion == 'uniform') else self.charges(tan)
        W = (F * q[None, :]) @ self.H.T                                          # (nstrip, nsamp)
        return W, pos, w

    def one(self, tan, estimators=('dream', 'cw', 'ideal'), corrections=('plain', 'rob', 'e17', 'shift')):
        x0 = self.rng.uniform(-self.p / 2, self.p / 2)
        W, pos, w = self.waveforms(tan, x0)
        W = W + self.rng.normal(size=W.shape) * self.noise
        nz = np.full(len(pos), self.noise)
        out = {}
        for est in estimators:
            if est == 'dream':
                h = E.dream(W, nz)
            elif est == 'cw':
                h = E.cw(W, nz)
            elif est == 'ideal':
                h = self.ideal(W, nz)
            elif est == 'peak':
                h = E.peak(W, nz, neighbour=False)
            elif est == 'fd':
                h = E.fd(W, nz)
            else:
                raise ValueError(est)
            t, a = h['t'], h['a']
            for cor in corrections:
                if cor == 'plain':
                    u = line_u(pos, t, a)
                elif cor == 'rob':
                    u = line_u(pos, t, a, robust=True)
                elif cor == 'e17':
                    pp, tt = edge_fix(pos, t, a, 1 / 6, False)
                    u = line_u(pp, tt, a, robust=True)
                elif cor == 'shift':
                    pp, tt = edge_fix(pos, t, a, 1 / 6, True)
                    u = line_u(pp, tt, a, robust=True)
                else:
                    raise ValueError(cor)
                out[f'{est}_{cor}'] = u / w if w != 0 else np.nan
        return out

    def scan(self, tans, ntr, **kw):
        res = {}
        for tan in tans:
            rows = [self.one(tan, **kw) for _ in range(ntr)]
            keys = rows[0].keys()
            res[float(tan)] = {}
            for k in keys:
                r = np.array([x[k] for x in rows])
                r = r[np.isfinite(r)]
                if len(r) < 10:
                    continue
                med = float(np.median(r))
                err = float(1.2533 * 1.4826 * np.median(np.abs(r - med)) / np.sqrt(len(r)))
                res[float(tan)][k] = dict(ratio=med, err=err, n=int(len(r)))
        return res


TANS = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65)


def fit_Cb(res, key, L, p):
    t, y, e = [], [], []
    for tan, d in res.items():
        if key in d:
            t.append(tan)
            y.append((d[key]['ratio'] - 1) * tan)
            e.append(max(d[key]['err'] * tan, 1e-4))
    t, y, e = np.array(t), np.array(y), np.array(e)
    if len(t) < 3:
        return None
    A = np.vstack([1 / t, t]).T / e[:, None]
    c, *_ = np.linalg.lstsq(A, y / e, rcond=None)
    return dict(a=float(c[0]), b=float(c[1]), C=float(c[0] * (L / p) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', required=True, choices=('arms', 'pitch', 'estimators', 'detectors'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--ntr', type=int, default=300)
    ap.add_argument('--ion', default='cluster')
    a = ap.parse_args()
    out = dict(scan=a.scan, ntr=a.ntr, tans=list(TANS), configs={})

    def run(name, toy, **kw):
        res = toy.scan(TANS, a.ntr, **kw)
        fits = {k: fit_Cb(res, k, toy.L, toy.p) for k in next(iter(res.values())).keys()}
        out['configs'][name] = dict(params=dict(L=toy.L, pitch=toy.p, v=toy.v, sigma0=toy.s0, Dp=toy.Dp, sns=toy.sns, noise=toy.noise),
                                    curves={str(k): v for k, v in res.items()}, fits=fits)
        print(f'[{name}] ' + '  '.join(f'{k}: C={f["C"]:.2f} b={f["b"]:+.4f}' for k, f in fits.items() if f and k in ('dream_plain', 'dream_rob', 'cw_plain', 'cw_rob', 'cw_shift', 'ideal_plain', 'dream_shift')), flush=True)
        for tan in (0.1, 0.2, 0.3, 0.5):
            d = res.get(tan, {})
            print('     tan %.2f: ' % tan + ' '.join(f'{k}={v["ratio"]:.3f}' for k, v in d.items() if k in ('dream_plain', 'dream_rob', 'dream_shift', 'cw_plain', 'cw_rob', 'cw_shift', 'ideal_plain')), flush=True)

    if a.scan == 'arms':
        for name, s0, dp in (('point', 0.07, 0.0), ('footprint', 0.30, 0.0), ('diffusion', 0.07, 0.021), ('base', 0.30, 0.021),
                             ('base_Dp023', 0.30, 0.023), ('wide', 0.45, 0.021)):
            run(name, Toy(sigma0=s0, Dp=dp, ion=a.ion, seed=1))
    elif a.scan == 'pitch':
        for p in (0.39, 0.78, 1.17, 1.56, 2.34, 3.12):
            run(f'p{p:.2f}', Toy(pitch=p, ion=a.ion, seed=2))
            run(f'p{p:.2f}_point', Toy(pitch=p, sigma0=0.07, Dp=0.0, ion=a.ion, seed=3))
    elif a.scan == 'estimators':
        toy = Toy(ion=a.ion, seed=4)
        res = toy.scan(TANS, a.ntr, estimators=('dream', 'cw', 'ideal', 'peak', 'fd'), corrections=('plain', 'rob', 'e17', 'shift'))
        fits = {k: fit_Cb(res, k, toy.L, toy.p) for k in next(iter(res.values())).keys()}
        out['configs']['base'] = dict(params=dict(L=toy.L, pitch=toy.p, v=toy.v, sigma0=toy.s0, Dp=toy.Dp), curves={str(k): v for k, v in res.items()}, fits=fits)
        for k, f in fits.items():
            if f:
                print(f'   {k:14s} C={f["C"]:6.2f}  b={f["b"]:+.4f}')
    elif a.scan == 'detectors':
        # other geometries, APV25-like 25 ns sampling with a 50 ns CR-RC2 shaper, Fermi rise-time estimator (LMU/ATLAS practice) and peak (VMM)
        # Dp = D_T sqrt(v / 10) with D_T in mm/sqrt(cm)
        for name, L, p, v, DT, s0 in (('NSW-like 5mm/0.4', 5.0, 0.40, 0.047, 0.365, 0.10), ('FSM-like 6mm/0.5', 6.0, 0.50, 0.047, 0.365, 0.10),
                                       ('GEM-like 8mm/0.4', 8.0, 0.40, 0.047, 0.365, 0.15), ('MX17 X', 28.08, 0.78, 0.0358, 0.351, 0.30),
                                       ('MX17-like metallic', 28.08, 0.78, 0.0358, 0.351, 0.07)):
            dp = DT * np.sqrt(v / 10.0)
            if 'MX17' in name:
                toy = Toy(L=L, pitch=p, v=v, sigma0=s0, Dp=dp, ion=a.ion, seed=5)
                ests = ('dream', 'cw', 'ideal')
            else:
                toy = Toy(L=L, pitch=p, v=v, sigma0=s0, Dp=dp, template='crrc', tp=50.0, sns=25.0, nsamp=27, noise=20.0, t0_sample=4, du=2.0, ion=a.ion, seed=5)
                ests = ('fd', 'peak', 'cw')
            res = toy.scan((0.1, 0.176, 0.268, 0.364, 0.466, 0.577, 0.7, 0.839), a.ntr, estimators=ests, corrections=('plain', 'rob', 'e17', 'shift'))
            fits = {k: fit_Cb(res, k, toy.L, toy.p) for k in next(iter(res.values())).keys()}
            out['configs'][name] = dict(params=dict(L=L, pitch=p, v=v, D_T=DT, sigma0=s0, Dp=dp), curves={str(k): v for k, v in res.items()}, fits=fits)
            print(f'[{name}]  sigma_T(L)/p = {DT * np.sqrt(L / 10) / p:.2f}')
            for tan, d in res.items():
                print(f'     tan {tan:.3f} ({np.degrees(np.arctan(tan)):.0f} deg): ' + ' '.join(f'{k}={v["ratio"]:.3f}' for k, v in d.items() if k.endswith('_plain') or k.endswith('_rob')))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
