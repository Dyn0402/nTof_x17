#!/usr/bin/env python3
"""ms_mc.py -- GB part 1: the M3 reference error at the DUT, with multiple scattering, by Monte Carlo.

What the reference error is: e(z_DUT) = slope of the M3 4-point unweighted fit (the producer's) - the muon's TRUE local
slope inside the DUT drift gap.  Two sources: (1) M3 point noise, per plane (M3 self-resolution study, Gaussian core:
X 0.415/0.410/0.409/0.411 mm, Y 0.511/0.503/0.445/0.458 mm); (2) multiple scattering between the planes: a kink k at
z_s moves every M3 point below z_s, so the fit picks up a fraction of k, while the local slope at the DUT has k only if
z_s is above the drift gap.  Analytically e = k * [sum_{z_k < z_s} w_k (z_k - z_s) - H(z_s above the gap)],
w_k = (z_k - zbar) / sum (z - zbar)^2.

Inputs (scanned, because no record has them):
  - per-slot material x/X0 (MX17 chamber + the 5 mm shelf board at z = 227 / 697), split top (above the gap) / bottom;
  - M3 station material 0.4 % X0 each (the M3 study's estimate), air 1278 mm (0.42 % X0) in 50 mm steps;
  - sea-level vertical muon momentum spectrum (Reyna-Bugaev above 1 GeV/c, flat continuation below), p >= p_min.
Highland on the total thickness, shared among scatterers in proportion to x/X0 (Gaussian per momentum; the mixture over
the spectrum makes the tails).  Production selection applied: 4 hits, chi2X < 1 and chi2Y < 1 mm^2 (sum of r^2).

Calibration against data: the inter-doublet kink of the sat det3 run (kink_det3.py) in the production selection,
s68 = 5.44 (M3 X) / 6.37 (M3 Y) mrad; noise-only must be compared under the same cut.

    ms_mc.py   -> results/ms_mc.json
"""
import json
import os

import numpy as np

from common import RES, M3_Z

SIG = {'X': np.array([0.415, 0.410, 0.409, 0.411]), 'Y': np.array([0.511, 0.503, 0.445, 0.458])}
MMU = 0.10566
X0_AIR = 303900.0
N = 200000


def spectrum_sample(rng, n, pmin):
    """vertical sea-level muon momenta [GeV/c]: Reyna-Bugaev for p >= 1, flat below (BESS-like plateau)."""
    p = np.logspace(np.log10(pmin), 3, 4000)
    lp = np.log10(np.maximum(p, 1.0))
    I = 0.00253 * np.maximum(p, 1.0) ** -(0.2455 + 1.288 * lp - 0.2555 * lp ** 2 + 0.0209 * lp ** 3)
    cdf = np.cumsum(I * np.gradient(p))
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, p)


def scatterers(slots, m3=0.004):
    """[(z, x/X0, tag)] top -> bottom.  slots: {z_gap_centre: (x_top, x_bottom)}; tags say whether the DUT sees it."""
    s = [(1302.0, m3, 'm3'), (1185.0, m3, 'm3'), (144.0, m3, 'm3'), (24.0, m3, 'm3')]
    for zc, (xt, xb) in slots.items():
        if xt > 0:
            s.append((zc + 16.0, xt, 'top'))          # cathode / window: above the drift gap
        if xb > 0:
            s.append((zc - 16.0, xb, 'bot'))          # readout PCB + shelf board: below the gap
    for z in np.arange(1277.0, 24.0, -50.0):
        s.append((z, 50.0 / X0_AIR, 'air'))
    return sorted(s, key=lambda a: -a[0])


def simulate(rng, slots, pmin=0.2, m3=0.004, dut_z=(702.0, 232.0), n=N, noise=True):
    p = spectrum_sample(rng, n, pmin)
    E = np.sqrt(p ** 2 + MMU ** 2)
    bp = p * p / E
    sc = scatterers(slots, m3)
    t_tot = sum(t for _, t, _ in sc)
    th_tot = 13.6e-3 / bp * np.sqrt(t_tot) * (1 + 0.038 * np.log(t_tot))
    zs = np.array([a[0] for a in sc])
    ts = np.array([a[1] for a in sc])
    out = {}
    for c in ('X', 'Y'):
        th = th_tot[:, None] * np.sqrt(ts / t_tot)[None, :] * rng.standard_normal((n, len(sc)))
        # true trajectory relative to the incoming line (top): displacement at each M3 plane and local slope at each DUT
        disp = np.zeros((n, 4))
        for j, zk in enumerate(M3_Z):
            above = zs > zk
            disp[:, j] = (th[:, above] * (zs[above] - zk)[None, :]).sum(1)   # x(z) gains k*(z_s - z) below z_s (dx/d(-z))
        meas = disp + (rng.standard_normal((n, 4)) * SIG[c][None, :] if noise else 0)
        zb = M3_Z.mean()
        wz = (M3_Z - zb) / ((M3_Z - zb) ** 2).sum()
        # slope in d/d(-z) convention (consistent with disp); intercept for the chi2
        slope = -(meas * wz[None, :]).sum(1)
        icpt = meas.mean(1)
        resid = meas - (icpt[:, None] - slope[:, None] * (M3_Z - zb)[None, :])
        chi2 = (resid ** 2).sum(1)
        st = -(meas[:, 0] - meas[:, 1]) / (M3_Z[0] - M3_Z[1])
        sb = -(meas[:, 2] - meas[:, 3]) / (M3_Z[2] - M3_Z[3])
        out[c] = dict(chi2=chi2, kink=sb - st, err={})
        for zd in dut_z:
            local = th[:, zs > zd].sum(1)                   # every kink above the gap centre
            out[c]['err'][zd] = slope - local
    return p, out


def s68(a):
    a = np.asarray(a)
    return float(np.percentile(np.abs(a - np.median(a)), 68))


def dut_effect(e_deg, rng, s_intr=1.0):
    """quadrature-equivalent effect of a reference error distribution on an s68 ~ 1 deg Gaussian DUT."""
    d = rng.normal(0, s_intr, e_deg.size)
    a = d + e_deg
    a, d = a[np.abs(a) <= 5], d[np.abs(d) <= 5]
    sa, sd = s68(a), s68(d)
    return float(np.sqrt(max(sa ** 2 - sd ** 2, 0)))


def summarise(rng, slots, pmin, m3=0.004, label=''):
    p, o = simulate(rng, slots, pmin=pmin, m3=m3)
    prod = (o['X']['chi2'] < 1) & (o['Y']['chi2'] < 1)
    r = dict(label=label, slots={str(k): v for k, v in slots.items()}, pmin=pmin, m3=m3, prod_frac=float(prod.mean()),
             p_median=float(np.median(p)), p_frac_lt1=float((p < 1).mean()))
    for c in ('X', 'Y'):
        r[f'kink_s68_prod_mrad_{c}'] = s68(o[c]['kink'][prod]) * 1e3
        for zd, e in o[c]['err'].items():
            ed = np.degrees(e[prod])
            r[f'err_{int(zd)}_{c}'] = dict(s68_deg=s68(ed), rms_deg=float(np.std(ed)),
                                          p95_deg=float(np.percentile(np.abs(ed), 95)),
                                          effect_on_1deg=dut_effect(ed, rng),
                                          effect_on_1p5deg=dut_effect(ed, rng, 1.5))
        e7, e2 = o[c]['err'][702.0][prod], o[c]['err'][232.0][prod]
        r[f'cov_232_702_deg2_{c}'] = float(np.cov(np.degrees(e7), np.degrees(e2))[0, 1])
        k = o[c]['kink'][prod]
        r[f'beta_702_{c}'] = float(np.cov(k, e7)[0, 1] / np.var(k))
    return r


def main():
    rng = np.random.default_rng(7)
    res = {'coeffs': {}, 'runs': []}
    # analytic coefficient of a kink at z_s in the DUT error, DUT at 702 / 232
    zb = M3_Z.mean()
    w = (M3_Z - zb) / ((M3_Z - zb) ** 2).sum()
    for zs in (1185, 718, 686, 467, 248, 216, 144):
        g = float(sum(w[k] * (M3_Z[k] - zs) for k in range(4) if M3_Z[k] < zs))
        res['coeffs'][zs] = {'fit_picks_up': g, 'c702': g - (zs > 702), 'c232': g - (zs > 232)}
    print('coeffs', json.dumps(res['coeffs']))
    # noise only (geometry term), then material scans
    p, o = simulate(rng, {}, pmin=0.2, m3=1e-9)
    prod = (o['X']['chi2'] < 1) & (o['Y']['chi2'] < 1)
    res['noise_only'] = {c: dict(err_702_s68_deg=s68(np.degrees(o[c]['err'][702.0][prod])),
                                 kink_s68_prod_mrad=s68(o[c]['kink'][prod]) * 1e3,
                                 kink_s68_all_mrad=s68(o[c]['kink']) * 1e3) for c in 'XY'}
    res['noise_only']['prod_frac'] = float(prod.mean())
    print('noise only', res['noise_only'])
    for pmin in (0.2, 0.5):
        for xs in (0.0, 0.01, 0.02, 0.04, 0.08, 0.16):
            for ftop in (0.0, 0.5):
                one = {702.0: (xs * ftop, xs * (1 - ftop))}
                two = {702.0: (xs * ftop, xs * (1 - ftop)), 232.0: (xs * ftop, xs * (1 - ftop))}
                for nm, sl in (('one_slot_702', one), ('two_slots', two)):
                    r = summarise(rng, sl, pmin, label=f'{nm} x={xs} ftop={ftop}')
                    res['runs'].append(r)
                    print(f"{r['label']:32s} pmin {pmin} prod {r['prod_frac']:.2f} kink X/Y {r['kink_s68_prod_mrad_X']:.2f}/"
                          f"{r['kink_s68_prod_mrad_Y']:.2f} | e702 s68 {r['err_702_X']['s68_deg']:.3f} eff {r['err_702_X']['effect_on_1deg']:.3f}"
                          f" | e232 s68 {r['err_232_X']['s68_deg']:.3f} eff {r['err_232_X']['effect_on_1deg']:.3f}"
                          f" | cov {r['cov_232_702_deg2_X']:+.4f} beta {r['beta_702_X']:+.2f}")
    json.dump(res, open(os.path.join(RES, 'ms_mc.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
