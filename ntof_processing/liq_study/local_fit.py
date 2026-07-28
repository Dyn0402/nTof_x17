#!/usr/bin/env python3
"""Fit real liquid-scintillator pulses locally, to find a template that works.

Every template we shipped into the PSA made the liquid fits WORSE, and no
amount of reasoning about the templates settled why. This replicates the job
the PSA does -- fit amplitude x template(t - t0) to a real pulse -- so template
choices can be compared in seconds instead of a condor round-trip, and so the
chi2 is one we compute and understand rather than one we read out.

For each candidate basis it reports the reduced chi2 over a common set of real
isolated pulses, the fitted-amplitude scale relative to the pulse maximum, and
the residual structure. "best-of-N" mirrors what a multi-shape UserInput row
appears to do: fit each shape, keep whichever fits best.

    python local_fit.py <liq_pulses.npz> [outdir]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SHIPPED = Path('/media/dylan/data/x17/ntof_processing')
REPO = Path(__file__).resolve().parent.parent.parent
PRE = 20                      # the stored pulses start 20 ns before the peak


def load_template(path):
    d = np.loadtxt(path)
    t, y = d[:, 0], d[:, 1]
    if abs(y.min()) > abs(y.max()):
        y = -y
    y = y / y.max()
    return t - t[int(np.argmax(y))], y      # time relative to the template peak


def fit_one(pulse, tmpl_t, tmpl_y, sigma, t_grid):
    """Best (chi2, amp, t0) of amplitude x template(t - t0) against one pulse."""
    n = len(pulse)
    t = np.arange(n) - PRE                  # pulse peak sits near t = 0
    best = (np.inf, 0.0, 0.0)
    for t0 in t_grid:
        m = np.interp(t - t0, tmpl_t, tmpl_y, left=0.0, right=0.0)
        den = (m * m).sum()
        if den <= 0:
            continue
        a = (pulse * m).sum() / den          # linear least squares in amplitude
        r = pulse - a * m
        chi2 = (r * r).sum() / (sigma ** 2 * max(n - 2, 1))
        if chi2 < best[0]:
            best = (chi2, a, t0)
    return best


def evaluate(name, tmpls, rows, sigma, t_grid):
    """best-of-N over the given templates, for every pulse."""
    chi2s, amps = [], []
    for p in rows:
        b = min(fit_one(p, tt, ty, sigma, t_grid) for tt, ty in tmpls)
        chi2s.append(b[0])
        amps.append(b[1])
    chi2s, amps = np.array(chi2s), np.array(amps)
    print(f'  {name:38s} chi2 p50={np.median(chi2s):7.3f}  p90={np.percentile(chi2s,90):8.3f}'
          f'   amp/peak p50={np.median(amps):5.3f}')
    return chi2s, amps


def main():
    z = np.load(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.')
    out.mkdir(parents=True, exist_ok=True)

    rows = z['LIQA_rows']
    amps = z['LIQA_amps']
    sel = amps > 3000                       # well above noise, tail is real
    rows, amps = rows[sel], amps[sel]
    # noise, in units of the normalised pulse: baseline RMS / amplitude
    sigma = float(np.median(np.std(rows[:, :12], axis=1)))
    print(f'LIQA: {len(rows)} pulses with amp>3000, normalised noise sigma={sigma:.4f}\n')

    t_grid = np.arange(-4, 4.01, 0.25)

    med = np.median(rows, axis=0)
    med_t = np.arange(len(med)) - PRE - int(np.argmax(med)) + PRE
    med_t = np.arange(len(med)) - int(np.argmax(med))

    cands = {}
    cands['shipped LIQA only'] = [load_template(SHIPPED / 'X17_LIQA_Signal_7.txt')]
    cands['shipped LIQB only'] = [load_template(SHIPPED / 'X17_LIQB_Signal_0.txt')]
    cands['shipped pair (best-of-2)'] = [
        load_template(SHIPPED / 'X17_LIQA_Signal_7.txt'),
        load_template(SHIPPED / 'X17_LIQB_Signal_0.txt')]
    cands['measured median, full 220 ns'] = [(med_t, med)]
    for cut in (40, 60, 80, 120, 160):
        m = (med_t >= -PRE) & (med_t <= cut)
        cands[f'measured median, truncated {cut} ns'] = [(med_t[m], med[m])]

    print('template basis                            reduced chi2        fitted amp')
    res = {}
    for k, v in cands.items():
        res[k] = evaluate(k, v, rows, sigma, t_grid)

    # residual of the best and the shipped, to see WHAT is unfitted
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    t = np.arange(rows.shape[1]) - PRE
    for name, col in (('shipped pair (best-of-2)', 'C3'),
                      ('measured median, full 220 ns', 'C0')):
        tmpls = cands[name]
        R = []
        for p in rows[:400]:
            c, a, t0 = min(fit_one(p, tt, ty, sigma, t_grid) for tt, ty in tmpls)
            b = min(((tt, ty) for tt, ty in tmpls),
                    key=lambda x: fit_one(p, x[0], x[1], sigma, t_grid)[0])
            m = np.interp(t - t0, b[0], b[1], left=0.0, right=0.0)
            R.append(p - a * m)
        R = np.array(R)
        axes[0].plot(t, np.median(R, axis=0), lw=1.0, color=col, label=name)
        axes[1].plot(t, np.median(np.abs(R), axis=0), lw=1.0, color=col, label=name)
    axes[0].axhline(0, color='k', lw=0.6)
    axes[0].set_xlabel('ns from peak'); axes[0].set_ylabel('median residual')
    axes[1].set_xlabel('ns from peak'); axes[1].set_ylabel('median |residual|')
    axes[1].set_yscale('log')
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=7); ax.set_xlim(-20, 200)
    plt.tight_layout(); plt.savefig(out / 'liq_fit_residuals.png', dpi=110)
    print(f'\nwrote {out}/liq_fit_residuals.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
