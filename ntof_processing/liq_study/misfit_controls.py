#!/usr/bin/env python3
"""Where does the liquid fit residual actually come from?

A single measured template already fits 3-4x better than the shipped pair, and
adding shapes binned by slow-component fraction gains only ~10 % more -- yet
the reduced chi2 is still 20-70, i.e. the residual sits 4-8x above the sample
noise. Before concluding anything about the detector, rule out the fitter:

  A  amplitude only                     (what local_fit.py did)
  B  amplitude + free baseline offset   (a residual pedestal cannot be absorbed
                                         by a scale factor)
  C  amplitude + baseline + linear slope (slow baseline wander)
  D  C, plus a free time-width scale     (allows the pulse to be slightly
                                         narrower/wider than the template)
  E  the noise model itself: compare the assumed sigma against the residual in
     a pulse-free stretch of the SAME block

If chi2 collapses at B or C the misfit was baseline, which the PSA handles
itself and which says nothing about the liquids. If it survives all of them,
the pulses genuinely vary and no fixed template basis will do.

    python misfit_controls.py <liq_pulses.npz>
"""
import sys

import numpy as np

PRE = 20


def design(n, t, tmpl_t, tmpl_y, t0, mode, width=1.0):
    tt = (t - t0) / width
    m = np.interp(tt, tmpl_t, tmpl_y, left=0.0, right=0.0)
    cols = [m]
    if mode in ('B', 'C', 'D'):
        cols.append(np.ones(n))
    if mode in ('C', 'D'):
        cols.append((t - t.mean()) / t.ptp())
    return np.vstack(cols).T


def fit(pulse, tmpl_t, tmpl_y, sigma, mode, grid, widths=(1.0,)):
    n = len(pulse)
    t = np.arange(n) - PRE
    best = np.inf
    for w in widths:
        for t0 in grid:
            X = design(n, t, tmpl_t, tmpl_y, t0, mode, w)
            try:
                beta, *_ = np.linalg.lstsq(X, pulse, rcond=None)
            except np.linalg.LinAlgError:
                continue
            r = pulse - X @ beta
            dof = max(n - X.shape[1], 1)
            chi2 = (r * r).sum() / (sigma ** 2 * dof)
            best = min(best, chi2)
    return best


def main():
    z = np.load(sys.argv[1])
    for tree in ('LIQA', 'LIQD'):
        rows, amps = z[f'{tree}_rows'], z[f'{tree}_amps']
        sel = amps > 3000
        rows, amps = rows[sel], amps[sel]
        if len(rows) < 200:
            continue
        med = np.median(rows, axis=0)
        tmpl_t = np.arange(len(med)) - int(np.argmax(med))
        tmpl_y = med / med.max()
        grid = np.arange(-3, 3.01, 0.25)

        # E: the noise model. sigma from the pre-pulse samples, and for
        # comparison the scatter of the pulses about their own median in a
        # region where the pulse is flat and small (150-200 ns after the peak)
        sig_pre = float(np.median(np.std(rows[:, :12], axis=1)))
        late = rows[:, PRE + 150:PRE + 199]
        sig_late = float(np.median(np.std(late - np.median(late, axis=0), axis=1)))
        print(f'\n{tree}: {len(rows)} pulses, amp>3000')
        print(f'  [E] noise: pre-pulse sigma={sig_pre:.5f}   '
              f'late-tail scatter={sig_late:.5f}   ratio={sig_late/sig_pre:.2f}')

        sub = rows[:250]
        for mode, lab, widths in (('A', 'amplitude only', (1.0,)),
                                  ('B', 'amp + baseline', (1.0,)),
                                  ('C', 'amp + baseline + slope', (1.0,)),
                                  ('D', 'amp + baseline + slope + width',
                                   (0.9, 0.95, 1.0, 1.05, 1.1))):
            c = np.array([fit(p, tmpl_t, tmpl_y, sig_pre, mode, grid, widths)
                          for p in sub])
            print(f'  [{mode}] {lab:32s} chi2 p50={np.median(c):8.2f}  '
                  f'p90={np.percentile(c,90):9.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
