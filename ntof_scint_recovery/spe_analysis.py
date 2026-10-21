#!/usr/bin/env python3
"""
spe_analysis.py -- single-photoelectron candidates as a population-free gain
reference: small peaks in the quiet tails of zero-suppressed blocks (pulses.py
v3 ``SPE`` histograms, per channel and coarse time), production runs.

    python -m ntof_scint_recovery.spe_analysis [--hists DIR]

Per channel, the 1-10 ms small-peak spectrum is fitted with a Gaussian (the
candidate single-photoelectron peak) on two falling exponentials (noise and the
tail of real low-energy pulses), 1.0-9.0 mV.  A channel is accepted when the peak
sits at 1.5-6 mV, is narrower than its position, and its height is > 5 sigma.
For accepted channels the same fit, with the shape parameters fixed to the late
ones except the position and height, is done in every coarse time bin: the
peak position over its 1-10 ms value is the gain against time, independent of
what particles arrive -- dark and photoelectron pulses are the same population
at every time.

Writes ``out_dir('spe')/spe_curves.npz`` and ``spe_channels.csv``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from ntof_scint_recovery.common import out_dir
from ntof_scint_recovery.quick import DETS, PRODUCTION_FROM

FIT = (1.0, 9.0)
PLATEAU = (1e3, 1e4)


def model(x, a, mu, s, b1, l1, b2, l2):
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2) + b1 * np.exp(-x / l1) + b2 * np.exp(-x / l2)


def load(hdir: Path):
    acc, meta, live = None, None, None
    for f in sorted(hdir.glob('pulsehist_*.npz')):
        if int(f.stem.split('_')[1]) < PRODUCTION_FROM:
            continue
        z = np.load(f)
        if 'SPE' not in z.files:
            continue
        acc = z['SPE'].astype(np.float64) if acc is None else acc + z['SPE']
        meta = z
    return acc, meta


def fit_one(x, h, p0=None, fix=None):
    m = (x >= FIT[0]) & (x <= FIT[1]) & np.isfinite(h)
    if h[m].sum() < 300:
        return None
    sig = np.sqrt(np.maximum(h[m], 1))
    if fix is None:
        p0 = p0 or [h[m].max() * 0.3, 3.0, 0.8, h[m][0] * 3, 0.4, h[m].max() * 0.3, 3.0]
        try:
            p, cov = curve_fit(model, x[m], h[m], p0=p0, sigma=sig, maxfev=40000,
                               bounds=([0, 1.0, 0.2, 0, 0.05, 0, 0.5], [np.inf, 8, 4, np.inf, 2, np.inf, 30]))
            return p, np.sqrt(np.diag(cov))
        except Exception:                      # noqa: BLE001
            return None
    s, l1, l2 = fix

    def f2(xx, a, mu, b1, b2):
        return model(xx, a, mu, s, b1, l1, b2, l2)
    try:
        p, cov = curve_fit(f2, x[m], h[m], p0=p0, sigma=sig, maxfev=40000,
                           bounds=([0, 0.5, 0, 0], [np.inf, 10, np.inf, np.inf]))
        return p, np.sqrt(np.diag(cov))
    except Exception:                          # noqa: BLE001
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--hists', type=Path, default=out_dir('pulsehist_v3'))
    a = ap.parse_args(argv)
    S, meta = load(a.hists)
    if S is None:
        print('no SPE histograms')
        return 1
    se = meta['spe_edges']
    x = 0.5 * (se[:-1] + se[1:])
    tce = meta['tc_edges_us']
    tcc = np.sqrt(tce[:-1] * tce[1:])
    nlc = 25
    pl = (tcc[:nlc] > PLATEAU[0]) & (tcc[:nlc] < PLATEAU[1])
    chans = [tuple(c) for c in meta['channels']]
    out = {'x_mV': x, 'tcc_us': tcc[:nlc]}
    rows = []
    for k, c in enumerate(('par', 'ded')):
        for i, (d, ch) in enumerate(chans):
            det = DETS[d]
            hl = S[k, i, :nlc][pl].sum(axis=0)
            out[f'{det}{ch}_{c}_late'] = hl
            r = fit_one(x, hl)
            row = dict(det=det, ch=int(ch), cls=c, n_late=int(hl.sum()), accepted=False)
            if r is not None:
                p, e = r
                signif = p[0] / max(e[0], 1e-9)
                row.update(mu=p[1], mu_err=e[1], sigma=p[2], signif=signif)
                row['accepted'] = bool(1.5 <= p[1] <= 6 and p[2] < p[1] and signif > 5)
                if row['accepted']:
                    mus = np.full(nlc, np.nan)
                    errs = np.full(nlc, np.nan)
                    for j in range(nlc):
                        h = S[k, i, j]
                        r2 = fit_one(x, h, p0=[p[0] * h.sum() / max(hl.sum(), 1), p[1], p[3] * h.sum() / max(hl.sum(), 1),
                                               p[5] * h.sum() / max(hl.sum(), 1)], fix=(p[2], p[4], p[6]))
                        if r2 is not None and r2[1][1] < 0.5:
                            mus[j], errs[j] = r2[0][1], r2[1][1]
                    out[f'{det}{ch}_{c}_mu'] = mus / p[1]
                    out[f'{det}{ch}_{c}_mu_err'] = errs / p[1]
            rows.append(row)
    o = out_dir('spe')
    np.savez_compressed(o / 'spe_curves.npz', **out)
    keys = ['det', 'ch', 'cls', 'n_late', 'accepted', 'mu', 'mu_err', 'sigma', 'signif']
    with open(o / 'spe_channels.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in rows])
    acc = [r for r in rows if r['accepted']]
    print(f'{len(acc)} of {len(rows)} channel-classes show an accepted small-pulse peak -> {o}')
    for r in rows:
        if 'mu' in r:
            print(f"  {r['det']}{r['ch']} {r['cls']}: mu {r['mu']:.2f}±{r['mu_err']:.2f} mV, sigma {r['sigma']:.2f}, "
                  f"{r['signif']:.1f} sigma, n {r['n_late']:,}  {'ACCEPTED' if r['accepted'] else ''}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
