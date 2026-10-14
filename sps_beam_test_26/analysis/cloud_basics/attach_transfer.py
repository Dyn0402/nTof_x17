#!/usr/bin/env python3
"""attach_transfer.py -- does the beam-gas attachment bias the SPS-measured model constants?

The constants the bench model takes from det4 at SPS were measured on run_71, whose gas lost
late drift charge (1.5 % water + ~160 ppm O2, eta*v ~ 1.9e-4/ns; FINDINGS §16, §25).  This
injection test simulates head-on beam events with KNOWN readout constants, with and without
that loss, and passes them through the SAME measurement code the SPS numbers came from:

  rc_diffusion.stack + fit        sigma_0 (X, Y) and D_rc (Y), X's D_rc check     (§3, §10)
  neighbour_vs_time.profiles      leading-edge +-1/centre and its sigma_eq          (§1, §10)
  decompose.fit                   prompt a1, delayed b1, tau                        (§2)
  sharing_kernel fit_kernel.fit   cascade c / tau and delay c2/c1, cross-relation   (sharing_kernel/)

Attachment on/off uses common random numbers (same tracks, clusters, positions, gain draws;
only the electrons' survival differs), so the on-off difference is the attachment effect with
most of the statistical noise cancelled.  Run at the three run_71 fields, at twice the run_71
loss (the CO2-period level), and in the bench gas (no loss) for scale: the bench-vs-beam
difference that comes from the GAS (drift diffusion, v) is reported next to the one that comes
from attachment.

Toy event (both views of one head-on track, |tan| = 0):
  primary clusters 3 /mm over the 30 mm gap, sizes P(n) ~ 1/n^2 (n <= 60);
  each electron survives exp(-eta*v * t_drift); cluster gain Gamma(n_surv);
  lateral: centroid + N(0, D_T sqrt(z)/sqrt(n)), width^2 = sigma_0^2 + D_T^2 z;
  time: z / v + N(0, D_L sqrt(z) / v);
  X: prompt footprint only.  Y: then 1-D RC spread, width^2 += 2 D_rc s;
  det3 bench X template, 60 ns sampling with random phase, 6.5 ADC white noise, 9 strips x 64
  samples; amplitude scaled so the median 9-strip peak is ~1000 ADC (run_71: 980-1100).

    attach_transfer.py [--nev 3000] [--seeds 4] [--workers 12]     -> results/attach_transfer.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'sharing_kernel'))
import rc_diffusion                                   # noqa: E402
import neighbour_vs_time as nvt                       # noqa: E402
import fit_kernel                                     # noqa: E402
from gasmodel import GasGrid                          # noqa: E402

from decompose import fit as decompose_fit               # noqa: E402

PITCH = 0.78
GAP = 30.0
NSTRIP, NSAMP, SNS = 9, 64, 60.0
DT = 10.0
L = int(NSAMP * SNS / DT) + 200                       # fine grid, 10 ns
_T = json.load(open(os.path.join(HERE, 'results', 'template_det3_x.json')))
TG, TM = np.array(_T['grid']), np.array(_T['tmpl_x'])
TM = TM / (TM.sum() * DT)
T_OFF = int(round(TG[0] / DT))                        # template starts at -360 ns
NOISE = 6.5

TRUTH = dict(sig0_x=0.45, sig0_y=0.37, drc_y=2.1e-4)


def simulate(nev, gas, etav, seed, truth=TRUTH, gain=None):
    """events in the plane_ratio cache format: {eid: {'tan_x', 'tan_y', 'x': {'W','pos','noise'}, 'y'}}"""
    rng = np.random.default_rng(seed)
    rng_att = np.random.default_rng(seed + 7919)      # survival draws: the only thing etav changes
    v, DTd, DLd = gas['v'], gas['DT'] * 1e-3, gas['DL']
    lags = np.arange(L) * DT
    sizes = np.arange(1, 61)
    psz = 1.0 / sizes ** 2
    psz /= psz.sum()
    off = np.arange(NSTRIP) - NSTRIP // 2
    gain = gain if gain is not None else 1.0
    ev = {}
    for i in range(nev):
        nc = rng.poisson(3.0 * GAP)
        z = rng.uniform(0, GAP, nc)
        n = rng.choice(sizes, nc, p=psz)
        tz = z * 1e3 / v + rng.normal(0, 1, nc) * DLd * np.sqrt(z / 10.0) / v
        u_surv = rng_att.random((nc, 60))
        surv = (u_surv < np.exp(-etav * np.maximum(tz, 0))[:, None]) & (np.arange(60)[None, :] < n[:, None])
        # per-electron gain (exponential), drawn for every electron slot so the stream is the same on/off
        q = (surv * rng.exponential(1.0, (nc, 60))).sum(1)
        t_trig = rng.uniform(600.0, 660.0)
        phase = rng.integers(0, 6)
        out = {'eid': i, 'tan_x': 0.0, 'tan_y': 0.0}
        for view in ('x', 'y'):
            u = rng.uniform(-PITCH / 2, PITCH / 2)
            sigd = DTd * np.sqrt(z / 10.0)
            xc = u + rng.normal(0, 1, nc) * sigd / np.sqrt(np.maximum(n, 1))
            w2 = truth[f'sig0_{view}'] ** 2 + sigd ** 2
            I = np.zeros((NSTRIP, L))
            for c in np.flatnonzero(q > 0):
                k0 = int(round((tz[c] + t_trig) / DT))
                if k0 >= L:
                    continue
                d = off * PITCH - xc[c]
                if view == 'x':
                    s = np.sqrt(w2[c])
                    f = 0.5 * (erf((d + PITCH / 2) / (np.sqrt(2) * s)) - erf((d - PITCH / 2) / (np.sqrt(2) * s)))
                    I[:, k0] += q[c] * f
                else:
                    m = L - k0
                    s = np.sqrt(w2[c] + 2 * truth['drc_y'] * lags[:m])[None, :]
                    F = 0.5 * (erf((d[:, None] + PITCH / 2) / (np.sqrt(2) * s))
                               - erf((d[:, None] - PITCH / 2) / (np.sqrt(2) * s)))
                    I[:, k0:] += q[c] * np.diff(np.c_[np.zeros(NSTRIP), F], axis=1)
            S = np.array([np.convolve(r, TM)[-T_OFF:-T_OFF + L] for r in I]) * DT
            W = S[:, phase::6][:, :NSAMP] * gain + rng.normal(0, NOISE, (NSTRIP, NSAMP))
            out[view] = {'W': W, 'pos': off * PITCH, 'noise': np.full(NSTRIP, NOISE), 'u_true': u}
        ev[i] = out
    return ev


def kernel_stacks(ev, view):
    """sharing_kernel/stacks.py convention: per event, centre = leading strip, traces aligned on
    its peak sample, +-30 samples, normalised to its peak; 20 %-trimmed mean per offset."""
    rows = {d: [] for d in (0, 1, -1, 2, -2)}
    for e in ev.values():
        W = np.asarray(e[view]['W'], float)
        ic = int(np.argmax(W.max(1)))
        if ic < 2 or ic > NSTRIP - 3:
            continue
        pk = int(np.argmax(W[ic])); a = W[ic, pk]
        if a <= 0:
            continue
        cols = pk + np.arange(-30, 31)
        okc = (cols >= 0) & (cols < NSAMP)
        for d in rows:
            r = np.full(61, np.nan)
            r[okc] = W[ic + d, cols[okc]] / a
            rows[d].append(r)
    return {d: fit_kernel.trim_mean(np.array(v)) for d, v in rows.items()}


def measure(ev):
    out = {}
    for view in ('x', 'y'):
        n, M = rc_diffusion.stack(ev, view)
        r = {'n': n}
        x0, e0, rms0, *_ = rc_diffusion.fit(M, 'x0')
        r['rc_sig0_prompt_only'] = float(x0[0])
        x1, e1, rms1, *_ = rc_diffusion.fit(M, 'x' if view == 'x' else 'y')
        r['rc_sig0'], r['rc_D'] = float(x1[0]), float(x1[1])
        if view == 'y':
            r['rc_taud'] = float(x1[2])
        # leading edge (neighbour_vs_time), and its decomposition
        R = nvt.profiles(ev, view, 0.03)
        m = np.full((5, len(nvt.GRID)), np.nan)
        from scipy.stats import trim_mean
        for o in range(5):
            for j in range(len(nvt.GRID)):
                col = R[:, o, j]; col = col[np.isfinite(col)]
                if len(col) > 0.5 * len(R):
                    m[o, j] = trim_mean(col, 0.2)
        c = m[2]
        r1 = 0.5 * (m[1] + m[3]) / np.where(c > 0.1, c, np.nan)
        jl = int(np.argmax(c > 0.3))
        r['lead_r1'] = float(r1[jl]); r['lead_sig_eq'] = nvt.sigma_eq(r1[jl])
        dz = decompose_fit(nvt.GRID, m.tolist(), 1300.0)
        r.update({f'dec_{k}': dz[k] for k in ('a1', 'b1', 'tau', 'a2', 'b2')})
        # sharing-kernel cross relation
        Wk = kernel_stacks(ev, view)
        fc = fit_kernel.fit('cascade', Wk); fd = fit_kernel.fit('delay', Wk)
        r['kern_cascade_c'] = fc['par']['c']; r['kern_cascade_tau'] = fc['par']['tau']
        r['kern_delay_c2_over_c1'] = fd['par']['c2'] / max(fd['par']['c1'], 1e-9)
        r['kern_delay_c1'] = fd['par']['c1']
        out[view] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nev', type=int, default=3000)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--only', default='', help='comma list of case-name substrings')
    ap.add_argument('--out', default='attach_transfer.json')
    a = ap.parse_args()
    beam = GasGrid('beam', 'air_hs')
    bench = GasGrid('bench', 'air_hs')
    W_BEAM, A_BEAM = 1.51, 0.078                      # run_71 composition (FINDINGS §25)
    cases = []
    for E in (243.0, 150.0, 92.0):
        g = beam(W_BEAM, A_BEAM, E)
        cases += [(f'beam{E:.0f}_noatt', g, 0.0), (f'beam{E:.0f}_att', g, g['etav']),
                  (f'beam{E:.0f}_att2x', g, 2 * g['etav'])]
        if E in (243.0, 92.0):                        # robustness: half the Magboltz D_T
            gh = dict(g, DT=0.5 * g['DT'])
            cases += [(f'beam{E:.0f}_halfDT_noatt', gh, 0.0), (f'beam{E:.0f}_halfDT_att', gh, g['etav'])]
    gb = bench(0.95, 0.0, 267.0)
    cases.append(('bench267_noatt', gb, 0.0))
    # the bench's own width growth fits dry-Magboltz diffusion (FINDINGS §10), so also the dry gas
    cases.append(('bench267_dry_noatt', bench(0.0, 0.0, 267.0), 0.0))
    if a.only:
        cases = [c for c in cases if any(o in c[0] for o in a.only.split(','))]
    # amplitude: run_71's median 9-strip X peak is ~1000 ADC at EVERY field (980-1100), although the
    # drift current per ns falls 3x from 243 to 92 V/cm.  So the gain is set per gas point, on its
    # no-loss case, and the attachment cases of that point keep it (the loss then costs signal).
    gains = {}
    for name, g, etav in cases:
        key = name.split('_att')[0].replace('_noatt', '')
        if key not in gains:
            probe = simulate(300, g, 0.0, 99)
            gains[key] = 1000.0 / np.median([np.asarray(e['x']['W']).sum(0).max() for e in probe.values()])
    print('gain per gas point [ADC per unit cluster charge]:',
          ', '.join(f'{k} {v:.1f}' for k, v in gains.items()), flush=True)
    res = {'truth': TRUTH, 'nev': a.nev, 'seeds': a.seeds, 'gains': gains, 'cases': {}}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {(name, s): pool.submit(_one, a.nev, g, etav, 1000 + s,
                                       gains[name.split('_att')[0].replace('_noatt', '')])
                for name, g, etav in cases for s in range(a.seeds)}
        for name, g, etav in cases:
            runs = [futs[(name, s)].result() for s in range(a.seeds)]
            res['cases'][name] = dict(gas={k: float(v) for k, v in g.items()}, etav=etav, runs=runs)
            mx, my = runs[0]['x'], runs[0]['y']
            print(f'{name:22s} etav {etav:.2e} | X sig0 {mx["rc_sig0"]:.3f} lead {mx["lead_sig_eq"]:.3f} '
                  f'| Y sig0 {my["rc_sig0"]:.3f} D {my["rc_D"]:.2e} '
                  f'lead {my["lead_sig_eq"]:.3f} a1 {my["dec_a1"]:.3f} b1 {my["dec_b1"]:.3f} tau {my["dec_tau"]:.0f} '
                  f'c {my["kern_cascade_c"]:.3f} c2/c1 {my["kern_delay_c2_over_c1"]:.2f}', flush=True)
    json.dump(res, open(os.path.join(HERE, 'results', a.out), 'w'), indent=1)


def _one(nev, g, etav, seed, gain):
    return measure(simulate(nev, g, etav, seed, gain=gain))


if __name__ == '__main__':
    main()
