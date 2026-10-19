#!/usr/bin/env python3
"""xw_sim.py -- does the X-width estimator have an amplitude-dependent bias?  A closure test.

FINDINGS §32: on det4 and det7 the head-on X D_T (x_width.fit_x on the unbiased rc_diffusion.stack)
falls by ~100 um/sqrt(cm) as the cut on the event's window maximum tightens from 3500 to 1500 ADC,
while det3 stays flat.  Here synthetic head-on events with a KNOWN D_T and v are pushed through the
same stack and fit, with the same cuts, to see whether the estimator alone makes that trend.

Each toy event reuses one real event of the dataset's head-on cache for everything that is layout:
strip positions, per-channel noise, the reference track position and tan_x (as the true track).
What is generated:

  ionisation   Poisson primary clusters along the 30 mm gap (CL_PER_MM, Ar/iso 95/5), Fischle Ar
               cluster sizes with a 1/n^2 tail (NMAX); 'delta' arm: large clusters spread along a
               delta-ray segment of the Katz-Penfold range in a random direction
  drift        t = z / v, lateral N(0, D_T sqrt z), longitudinal N(0, D_L sqrt z) (Magboltz D_L)
  gain         per electron Polya (theta), per event lognormal (gain map / Landau-tail mismatch);
               g_adc and sig_g tuned so the toy's window-maximum quantiles match the data
  footprint    prompt Gaussian sigma_0 onto 0.78 mm strips (erf fractions)
  electronics  the chamber's bundle X template (10 ns grid) -- the same one the fit uses, resampled;
               'share' arm adds the bundle's lp neighbour copies (c1, c2 = effective_c2, tau_s)
  readout      60 ns sampling with a random phase, 32 samples; noise = per-channel rms with the
               data's lag-1 autocorrelation and common-mode fraction; per-channel ADC ceiling drawn
               from the data's saturated maxima ('noclip' arm: none)

Then x_width.fit_x on rc_diffusion.stack (|tan_x| < 0.03), for all events and for the unsat cuts
SATS, one replica = one pass over the cache's layouts (the data's statistics).

    xw_sim.py prep               -> results/xw_sim/inputs.pkl   (local: caches, bundles, unsat_width.json)
    xw_sim.py tune [--sets ...]  -> tuned gain/timing written into inputs.pkl
    xw_sim.py run <set> <arm> <seed0> <nrep> [--inputs P] [--out F]     (condor job)
    xw_sim.py collect            -> results/xw_sim.json  (+ comparison with results/unsat_width.json)
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUTD = os.path.join(HERE, 'results', 'xw_sim')
INPUTS = os.path.join(OUTD, 'inputs.pkl')
PITCH, SAMPLE_NS, NSAMP, GAP = 0.78, 60.0, 32, 30.0
DTG = 10.0                                   # generation grid (ns), the template's own
SATS = [3500.0, 3000.0, 2500.0, 2000.0, 1500.0]
TAN = 0.03
CL_PER_MM = 2.8                              # Ar 25/cm, isobutane 90/cm, 95/5
W_EV = 26.0
NMAX = 400
# Fischle et al. (NIM A301 (1991) 202), argon: P(n) for n = 1..10, then ~ 1/n^2 (21.6 % above 10)
FISCHLE = np.array([65.6, 15.0, 6.4, 3.5, 2.25, 1.55, 1.05, 0.81, 0.61, 0.49])

# set -> (cache key, chamber).  Truth: v and sigma_0 = the unsat fit (GAP-30 frame), D_T = Magboltz at
# the v-implied water scaled into the frame (x sqrt(gap/30)), D_L Magboltz at the same point.
SETS = {'det3L': ('sat_det3__long_run_resist_490V_drift_1000V', 'det3'),
        'det4': ('g_det4', 'det4'),
        'det7R': ('g_det7_long__longer_run', 'det7'),
        'det7H': ('g_det7_long__hvpool_det7_480-505V_drift_700V', 'det7')}
ARMS = {'base': {}, 'delta': dict(delta=True), 'share': dict(share=True), 'full': dict(delta=True, share=True),
        'noclip': dict(clip=False), 'flatgain': dict(sig_g=0.0), 'refc': dict(centre='ref'),
        'dtlo': dict(dt_scale=0.75), 'dthi': dict(dt_scale=1.25)}


# ---------------------------------------------------------------- prep (local)
def noise_stats(ev):
    """lag-1 autocorrelation and common-mode fraction of the X noise, from strips > 4 mm from the
    reference track (signal-free at |tan| < 0.06) in every sample."""
    rho, cm = [], []
    for e in ev.values():
        X = e['x']
        far = np.abs(X['pos'] - e['ref_mesh_x']) > 4.0
        if far.sum() < 3:
            continue
        N = np.asarray(X['W'], float)[far]
        N = N - N.mean(1, keepdims=True)
        rho.append((N[:, 1:] * N[:, :-1]).sum() / (N ** 2).sum())
        c = N.mean(0)                                      # common part over the far strips
        k = far.sum()
        # var(mean) = f s^2 + (1-f) s^2 / k  ->  f
        s2 = (N ** 2).mean()
        cm.append(((c ** 2).mean() - s2 / k) / (s2 * (1 - 1 / k)))
    return float(np.median(rho)), float(np.clip(np.median(cm), 0, 1)), int(len(rho))


def prep(a):
    import x_width as xw
    from bench_attach import bundle_dir, template
    from gasmodel import GasGrid
    from wft.calib import effective_c2
    uw = json.load(open(f'{HERE}/results/unsat_width.json'))
    gg = GasGrid('bench', 'air_hs')
    out = {}
    for name, (key, det) in SETS.items():
        ev = pickle.load(open(f'{xw.CACHES}/headon_{key}.pkl', 'rb'))
        ev = {k: e for k, e in ev.items() if 'x' in e}
        u = uw[key]['unsat']
        g = u['gapcorr']
        br = g['magboltz']['branches'][0]
        gap = g['gap_mm']
        mb = gg(br['water_pct'], 0.0, uw[key]['E_Vcm'])
        z = np.load(f'{bundle_dir(det)}/arrays.npz')
        hyp = json.load(open(f'{bundle_dir(det)}/bundle.json'))['hyper']
        h_fit, _ = template(det)
        T0 = xw.GAP / (xw.V_BUNDLE[det] * 1e-3)
        if key.startswith(xw.SCAN):
            T0 = xw.GAP / (xw.V_SCAN[int(xw.drift_hv(key))] * 1e-3)
        mx = np.array([e['x']['W'].max() for e in ev.values()])
        ho = np.array([abs(e['tan_x']) < TAN for e in ev.values()])
        ons = []
        for e in ev.values():
            if abs(e['tan_x']) >= TAN:
                continue
            W = e['x']['W']; ic = int(W.sum(1).argmax())
            if 4 <= ic <= len(W) - 5:
                s = W[ic - 4:ic + 5].sum(0)
                if (s > 60).any():
                    ons.append(int(np.argmax(s > 60)))
        rho, cmf, nn = noise_stats(ev)
        layouts = [dict(pos=np.asarray(e['x']['pos'], float), noise=np.asarray(e['x']['noise'], float),
                        ch=e['x']['ch'], tan_x=float(e['tan_x']), x0=float(e['ref_mesh_x']), nsamp=e['x']['W'].shape[1])
                   for e in ev.values()]
        out[name] = dict(
            key=key, det=det, E_Vcm=uw[key]['E_Vcm'], gap_mm=gap, T0=T0,
            v=u['v'], sig0=u['sig0'], DT=br['DT_um_rtcm'] * np.sqrt(gap / GAP), DL=mb['DL'] * np.sqrt(gap / GAP),
            DT_data_frame=u['DT'], water_pct=br['water_pct'],
            tmpl_grid=z['grid'].astype(float), tmpl=z['tmpl_x'].astype(float), h_fit=h_fit,
            share=dict(c1=float(hyp['c1']), c2=effective_c2(hyp, 'x'), tau_s=float(hyp['tau_s']),
                       sigma_s=float(hyp['sigma_s']), sigma_p0=float(hyp['sigma_p0'])),
            sat_max=np.sort(mx[mx > 3500]), data_max_q=np.quantile(mx[ho], [0.1, 0.25, 0.5, 0.75, 0.9]),
            data_frac_sat=float((mx[ho] > 3500).mean()), data_onset_q=np.quantile(ons, [0.1, 0.25, 0.5, 0.75, 0.9]),
            noise_rho=rho, noise_cm=cmf, layouts=layouts)
        o = out[name]
        print(f'{name}: {len(layouts)} layouts ({ho.sum()} head-on), v {o["v"]:.2f}, D_T truth {o["DT"]:.0f} '
              f'(data {o["DT_data_frame"]:.0f}), D_L {o["DL"]:.0f}, sig0 {o["sig0"]:.3f}, noise rho {rho:.2f} '
              f'cm {cmf:.2f} (n {nn}), share c1 {o["share"]["c1"]:.3f} c2 {o["share"]["c2"]:.3f}, '
              f'max q {np.round(o["data_max_q"])}, frac sat {o["data_frac_sat"]:.2f}', flush=True)
    os.makedirs(OUTD, exist_ok=True)
    pickle.dump(out, open(INPUTS, 'wb'), protocol=4)


# ---------------------------------------------------------------- generator
def cluster_sizes(rng, n):
    p = np.r_[FISCHLE / 100.0, 0.216 * (1 / np.arange(11, NMAX + 1) ** 2) / (1 / np.arange(11, NMAX + 1) ** 2).sum()]
    return rng.choice(np.arange(1, NMAX + 1), size=n, p=p / p.sum())


def delta_range_mm(n):
    """Katz-Penfold practical range of a delta of n * W in argon at 1 atm (extrapolated below 10 keV)."""
    E = np.maximum(n * W_EV * 1e-6, 1e-5)                                # MeV
    R = 0.412 * E ** (1.265 - 0.0954 * np.log(E))                       # g/cm^2
    return R / 1.66e-3 * 10.0                                            # mm


class Gen:
    def __init__(self, S, arm, seed):
        self.S, self.rng = S, np.random.default_rng(seed)
        self.o = dict(delta=False, share=False, clip=True, sig_g=S.get('sig_g', 0.8), centre='charge',
                      dt_scale=1.0, theta=0.5)
        self.o.update(ARMS[arm] if isinstance(arm, str) else arm)
        self.g_adc = S.get('g_adc', 20.0)
        self.t_a = S.get('t_a', 150.0)
        self.f_low = S.get('f_low', 0.0)
        g, h = S['tmpl_grid'], S['tmpl']
        self.g0 = g[0]
        self.h = h
        if self.o['share']:
            from scipy.ndimage import gaussian_filter1d
            sh = S['share']
            x = np.r_[gaussian_filter1d(h, max(sh['sigma_s'], 1.0) / DTG), np.zeros(460)]
            a = np.exp(-DTG / max(sh['tau_s'], 1.0))
            from scipy.signal import lfilter
            l1 = lfilter([1 - a], [1, -a], x)
            l2 = lfilter([1 - a], [1, -a], l1)
            self.l1, self.l2 = l1, l2
        self.ntg = int((NSAMP * SAMPLE_NS + 600) / DTG) + 1

    def electrons(self):
        """(depth z mm, lateral offset from the track at the mesh mm) of every primary electron."""
        r = self.rng
        nc = r.poisson(CL_PER_MM * GAP)
        zc = r.uniform(0, GAP, nc)
        size = cluster_sizes(r, nc)
        z = np.repeat(zc, size)
        dx = np.zeros(len(z))
        if self.o['delta']:
            big = np.where(size >= 3)[0]
            starts = np.r_[0, np.cumsum(size)[:-1]]
            for i in big:
                R = delta_range_mm(size[i])
                c = r.uniform(-1, 1); ph = r.uniform(0, 2 * np.pi)
                d = np.array([np.sqrt(1 - c * c) * np.cos(ph), c])      # (x, z) components
                f = r.uniform(0, 1, size[i]) * R
                sl = slice(starts[i], starts[i] + size[i])
                dx[sl] += f * d[0]
                z[sl] = z[sl] + f * d[1]
            keep = (z >= 0) & (z <= GAP)
            z, dx = z[keep], dx[keep]
        return z, dx

    def event(self, L):
        r, S, o = self.rng, self.S, self.o
        z, dx = self.electrons()
        zc = z / 10.0
        DT = S['DT'] * o['dt_scale'] * 1e-3                               # mm / sqrt(cm)
        x = L['x0'] + L['tan_x'] * z + dx + r.normal(0, 1, len(z)) * DT * np.sqrt(zc)
        t = z / (S['v'] * 1e-3) + r.normal(0, 1, len(z)) * S['DL'] * 1e-3 * np.sqrt(zc) / (S['v'] * 1e-3)
        th = o['theta']
        q = r.gamma(1 + th, 1 / (1 + th), len(z)) if th is not None else np.ones(len(z))
        G = self.g_adc * (r.lognormal(-0.5 * o['sig_g'] ** 2, o['sig_g']) if o['sig_g'] > 0 else 1.0)
        if self.f_low > 0 and r.uniform() < self.f_low:          # faint population (det7): 1-100 % of nominal
            G = self.g_adc * 10 ** r.uniform(-2, 0)
        pos = L['pos']
        # with the copies on, the prompt footprint is the bundle's own sigma_p0 (the data-fitted sig0
        # already contains the copies)
        s0 = np.sqrt(2) * (S['share']['sigma_p0'] if o['share'] else S['sig0'])
        fr = 0.5 * (erf((pos[:, None] + PITCH / 2 - x[None, :]) / s0) - erf((pos[:, None] - PITCH / 2 - x[None, :]) / s0))
        tm = r.uniform(self.t_a, self.t_a + SAMPLE_NS)                    # mesh arrival in the window
        ib = np.clip(((tm + t) / DTG).astype(int), 0, self.ntg - 1)
        I = np.zeros((len(pos), self.ntg))
        for j in range(len(pos)):
            I[j] = np.bincount(ib, weights=fr[j] * q, minlength=self.ntg)
        I *= G
        ns = L['nsamp']
        ts = np.arange(ns) * SAMPLE_NS
        tg = np.arange(self.ntg) * DTG + self.g0                          # response time axis
        R = np.array([np.convolve(I[j], self.h)[:self.ntg] for j in range(len(pos))])
        if o['share']:
            c1, c2 = S['share']['c1'], S['share']['c2']
            C1 = np.array([np.convolve(I[j], self.l1)[:self.ntg] for j in range(len(pos))])
            C2 = np.array([np.convolve(I[j], self.l2)[:self.ntg] for j in range(len(pos))])
            add = np.zeros_like(R)
            add[1:] += c1 * C1[:-1]; add[:-1] += c1 * C1[1:]
            add[2:] += c2 * C2[:-2]; add[:-2] += c2 * C2[2:]
            R = R + add
        W = np.array([np.interp(ts, tg, R[j], left=0.0, right=R[j, -1]) for j in range(len(pos))])
        # noise: AR(1) in time, common-mode fraction across strips
        rho, f = S['noise_rho'], S['noise_cm']
        def ar(n):
            w = r.normal(0, 1, n)
            out = np.empty_like(w)
            out[..., 0] = w[..., 0]
            for k in range(1, w.shape[-1]):
                out[..., k] = rho * out[..., k - 1] + np.sqrt(1 - rho ** 2) * w[..., k]
            return out
        nz = L['noise'][:, None]
        W = W + nz * (np.sqrt(1 - f) * ar((len(pos), ns)) + np.sqrt(f) * ar((1, ns)))
        if o['clip']:
            ceil = r.choice(S['sat_max'], len(pos))[:, None]
            W = np.minimum(W, ceil)
        return dict(pos=pos.astype(np.float32), W=W.astype(np.float32), noise=L['noise'].astype(np.float32),
                    ch=L['ch'])

    def sample(self):
        out = {}
        for i, L in enumerate(self.S['layouts']):
            if abs(L['tan_x']) >= TAN:                # the stack and the cuts only use |tan_x| < TAN
                continue
            out[i] = dict(eid=i, tan_x=L['tan_x'], ref_mesh_x=L['x0'], x=self.event(L))
        return out


# ---------------------------------------------------------------- tune (local)
def summary(ev):
    mx = np.array([e['x']['W'].max() for e in ev.values() if abs(e['tan_x']) < TAN])
    ons = []
    for e in ev.values():
        if abs(e['tan_x']) >= TAN:
            continue
        W = e['x']['W']; ic = int(W.sum(1).argmax())
        if 4 <= ic <= len(W) - 5:
            s = W[ic - 4:ic + 5].sum(0)
            if (s > 60).any():
                ons.append(int(np.argmax(s > 60)))
    return np.quantile(mx, [0.1, 0.25, 0.5, 0.75, 0.9]), float((mx > 3500).mean()), np.quantile(ons, [0.1, 0.25, 0.5, 0.75, 0.9])


def tune(a):
    from scipy.optimize import minimize
    inp = pickle.load(open(INPUTS, 'rb'))
    for name in a.sets.split(','):
        S = inp[name]
        lay = S['layouts']
        sub = dict(S, layouts=[L for L in lay if abs(L['tan_x']) < TAN])
        dq = S['data_max_q']

        mix = name.startswith('det7')          # bimodal spectrum: lognormal + faint fraction

        def loss(p):
            sub['g_adc'], sub['sig_g'] = np.exp(p[0]), min(abs(p[1]), 1.5)
            if mix:
                sub['f_low'] = 1 / (1 + np.exp(-p[2]))
            q, fs, _ = summary(Gen(sub, 'base', 7).sample())      # common random numbers
            return float(np.sum(np.log(np.maximum(q, 30) / np.maximum(dq, 30)) ** 2) + 4 * (fs - S['data_frac_sat']) ** 2)

        if mix:
            # Nelder-Mead stalls on the bimodal det7 spectrum (2026-10-16): coarse grid instead
            import itertools
            grid = list(itertools.product([30, 40, 55, 75, 100], [0.3, 0.6, 0.9], [0.08, 0.15, 0.22, 0.3]))
            ls = [loss([np.log(g), sg, np.log(fl / (1 - fl))]) for g, sg, fl in grid]
            g, sg, fl = grid[int(np.argmin(ls))]
            best = type('R', (), dict(x=[np.log(g), sg, np.log(fl / (1 - fl))], fun=min(ls)))
        else:
            best = None
            for g0 in (np.log(10.0),):
                for s0 in (0.5, 1.2):
                    r = minimize(loss, [g0, s0], method='Nelder-Mead', options=dict(xatol=0.02, fatol=1e-3, maxiter=60))
                    if best is None or r.fun < best.fun:
                        best = r
        sub['g_adc'], sub['sig_g'] = float(np.exp(best.x[0])), float(min(abs(best.x[1]), 1.5))
        sub['f_low'] = float(1 / (1 + np.exp(-best.x[2]))) if mix else 0.0
        sub['t_a'] = 270.0
        # timing: shift t_a so the median onset sample matches
        for _ in range(3):
            _, _, oq = summary(Gen(sub, 'base', 8).sample())
            sub['t_a'] = sub.get('t_a', 150.0) + (S['data_onset_q'][2] - oq[2]) * SAMPLE_NS
        q, fs, oq = summary(Gen(sub, 'base', 9).sample())
        S.update(g_adc=sub['g_adc'], sig_g=sub['sig_g'], t_a=float(sub['t_a']), f_low=sub['f_low'],
                 tune=dict(loss=float(best.fun), toy_max_q=q, toy_frac_sat=fs, toy_onset_q=oq))
        print(f'{name}: g_adc {S["g_adc"]:.2f} sig_g {S["sig_g"]:.2f} f_low {S["f_low"]:.2f} t_a {S["t_a"]:.0f}  loss {best.fun:.3f}\n'
              f'   max q data {np.round(dq)} toy {np.round(q)}; sat {S["data_frac_sat"]:.2f} / {fs:.2f}; '
              f'onset data {S["data_onset_q"]} toy {oq}', flush=True)
        json.dump(dict(g_adc=S['g_adc'], sig_g=S['sig_g'], t_a=S['t_a'], f_low=S['f_low'], tune=S['tune']),
                  open(f'{OUTD}/tune_{name}.json', 'w'), indent=1, default=lambda x: np.asarray(x).tolist())


def merge_tune():
    inp = pickle.load(open(INPUTS, 'rb'))
    for name in inp:
        p = f'{OUTD}/tune_{name}.json'
        if os.path.exists(p):
            inp[name].update(json.load(open(p)))
    pickle.dump(inp, open(INPUTS, 'wb'), protocol=4)


# ---------------------------------------------------------------- run (condor)
def fit_all(ev, S, centre):
    import x_width as xw
    os.environ['CB_CENTRE'] = centre
    res = {}
    for lab, cut in [('all', None)] + [(f'sat{s:.0f}', s) for s in SATS]:
        sub = ev if cut is None else {k: e for k, e in ev.items() if e['x']['W'].max() < cut}
        n = sum(abs(e['tan_x']) < TAN for e in sub.values())
        if n < 30:
            res[lab] = dict(n=int(n))
            continue
        from rc_diffusion import stack
        nst, M = stack(sub, 'x', tan_max=TAN)
        try:
            f = xw.fit_x(S['h_fit'], M, S['T0'])
        except Exception as ex:                       # a degenerate stack (too few events) -> skip
            res[lab] = dict(n=int(n), error=str(ex))
            continue
        res[lab] = dict(f, n=int(n), nstack=int(nst))
    return res


def run(a):
    inp = pickle.load(open(a.inputs, 'rb'))
    S = inp[a.set]
    out = dict(set=a.set, arm=a.arm, truth=dict(v=S['v'], DT=S['DT'] * ARMS[a.arm].get('dt_scale', 1.0),
                                                 sig0=S['sig0'], DL=S['DL']), reps=[])
    t0 = time.time()
    for k in range(a.nrep):
        seed = a.seed0 + k
        gen = Gen(S, a.arm, seed)
        ev = gen.sample()
        r = fit_all(ev, S, gen.o['centre'])
        r['seed'] = seed
        out['reps'].append(r)
        s = r.get('sat2000', {})
        print(f'[{time.time() - t0:6.0f} s] {a.set} {a.arm} seed {seed}: all D_T {r["all"].get("DT", np.nan):.0f} '
              f'v {r["all"].get("v", np.nan):.2f} | sat2000 D_T {s.get("DT", np.nan):.0f} (n {s.get("n", 0)})', flush=True)
        json.dump(out, open(a.out, 'w'), default=float)


# ---------------------------------------------------------------- collect (local)
def collect(a):
    rows = {}
    for p in sorted(glob.glob(os.path.join(a.indir, 'sim_*.json'))):
        d = json.load(open(p))
        rows.setdefault((d['set'], d['arm']), dict(truth=d['truth'], reps=[]))['reps'] += d['reps']
    uw = json.load(open(f'{HERE}/results/unsat_width.json'))
    out = {}
    for (st, arm), d in sorted(rows.items()):
        key = SETS[st][0]
        lab_all = ['all'] + [f'sat{s:.0f}' for s in SATS]
        o = dict(truth=d['truth'], nrep=len(d['reps']), cuts={})
        for lab in lab_all:
            vals = [r[lab] for r in d['reps'] if 'DT' in r.get(lab, {})]
            if len(vals) < 3:
                continue
            DT = np.array([v['DT'] for v in vals]); V = np.array([v['v'] for v in vals])
            o['cuts'][lab] = dict(nrep=len(vals), n=float(np.mean([v['n'] for v in vals])),
                                  DT_mean=float(DT.mean()), DT_rms=float(DT.std(ddof=1)),
                                  DT_semean=float(DT.std(ddof=1) / np.sqrt(len(DT))),
                                  DT_bias=float(DT.mean() - d['truth']['DT']),
                                  v_mean=float(V.mean()), v_rms=float(V.std(ddof=1)), v_bias=float(V.mean() - d['truth']['v']),
                                  sig0_mean=float(np.mean([v['sig0'] for v in vals])))
        # the data at the same cuts
        dat = {}
        for s in SATS:
            k = key if s == 3500.0 else f'{key}@sat{s:.0f}'
            if k in uw and 'unsat' in uw[k]:
                u = uw[k]['unsat']
                dat[f'sat{s:.0f}'] = dict(DT=u['DT'], DT_err=u['DT_err'], v=u['v'], n=u['n'])
        if key in uw and 'all' in uw[key]:
            u = uw[key]['all']; dat['all'] = dict(DT=u['DT'], DT_err=u['DT_err'], v=u['v'], n=u['n'])
        o['data'] = dat
        out[f'{st}/{arm}'] = o
        line = ' '.join(f'{lab}: {c["DT_mean"]:4.0f}±{c["DT_semean"]:2.0f} (rms {c["DT_rms"]:3.0f})' for lab, c in o['cuts'].items())
        print(f'{st:6s} {arm:8s} truth {d["truth"]["DT"]:4.0f} n_rep {o["nrep"]:3d} | {line}')
        if arm == 'base':
            print('       data            | ' + ' '.join(f'{lab}: {c["DT"]:4.0f}±{c["DT_err"]:2.0f}' for lab, c in dat.items()))
    json.dump(out, open(os.path.join(HERE, 'results', 'xw_sim.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('prep')
    t = sp.add_parser('tune'); t.add_argument('--sets', default=','.join(SETS)); t.add_argument('--merge', action='store_true')
    r = sp.add_parser('run')
    r.add_argument('set'); r.add_argument('arm'); r.add_argument('seed0', type=int); r.add_argument('nrep', type=int)
    r.add_argument('--inputs', default=INPUTS); r.add_argument('--out', default='sim.json')
    c = sp.add_parser('collect'); c.add_argument('--indir', default=os.path.join(OUTD, 'out'))
    a = ap.parse_args()
    if a.cmd == 'tune' and a.merge:
        return merge_tune()
    if a.cmd == 'prep':
        sys.path[:0] = [os.path.abspath(os.path.join(HERE, '..', '..', '..'))]
        prep(a)
    else:
        {'tune': tune, 'run': run, 'collect': collect}[a.cmd](a)


if __name__ == '__main__':
    main()
