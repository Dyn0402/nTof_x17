#!/usr/bin/env python3
"""Is the liquid fit residual systematic shape variation, or photon statistics?

The distinction decides whether a better UserInput can help at all.

  * If the slow component varies SYSTEMATICALLY (particle type, energy), the
    residual is a fixed shape difference. Its size relative to the pulse is
    constant with amplitude, so chi2 measured against fixed electronic noise
    grows as A^2, and a richer template basis can absorb it.
  * If it is PHOTON STATISTICS in the tail -- the slow component being a
    countable number of photoelectrons spread over ~100 ns -- the residual
    grows only as sqrt(A). Then chi2 against fixed noise grows as A, it is
    irreducible, and NO template basis can help, because there is no shape to
    learn.

The two predictions differ by a full power of A, so a decade of amplitude
separates them cleanly.

Also fits the tail region directly: residual_rms(tail) vs sqrt(tail area) is a
straight line through the origin for photon statistics.

    python is_it_photon_statistics.py <liq_pulses.npz>
"""
import sys

import numpy as np

PRE = 20


def main():
    z = np.load(sys.argv[1])
    for tree in ('LIQA', 'LIQD'):
        rows, amps = z[f'{tree}_rows'], z[f'{tree}_amps']
        m = amps > 800
        rows, amps = rows[m], amps[m]
        if len(rows) < 300:
            continue
        # one template from the whole sample; residual measured per pulse
        med = np.median(rows, axis=0)
        tmpl = med / med.max()
        sig_norm = float(np.median(np.std(rows[:, :12], axis=1)))

        print(f'\n{tree}: {len(rows)} pulses, amp {amps.min():.0f}-{amps.max():.0f}')
        print(f'{"amp bin":>16s} {"n":>5s} {"resid RMS":>10s} {"chi2 vs":>9s} '
              f'{"resid/peak":>11s} {"resid/sqrt(A)":>14s}')
        print(f'{"":>16s} {"":>5s} {"[ADC]":>10s} {"e-noise":>9s} {"":>11s} {"":>14s}')
        edges = np.percentile(amps, [0, 20, 40, 60, 80, 95, 100])
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (amps >= lo) & (amps < hi)
            if s.sum() < 40:
                continue
            R = []
            for p, a in zip(rows[s], amps[s]):
                # amplitude-only fit at the best integer shift (fine enough here)
                best, br = np.inf, None
                for t0 in (-1, 0, 1):
                    mm = np.roll(tmpl, t0)
                    den = (mm * mm).sum()
                    sc = (p * mm).sum() / den
                    r = p - sc * mm
                    v = (r * r).sum()
                    if v < best:
                        best, br = v, r
                # tail region only: this is where the slow component lives
                R.append(br[PRE + 12:PRE + 180] * a)      # back to ADC units
            R = np.array(R)
            rms = float(np.sqrt(np.mean(R ** 2)))
            amid = float(np.median(amps[s]))
            e_noise = sig_norm * amid                      # fixed electronic noise
            print(f'{lo:7.0f}-{hi:<8.0f} {s.sum():5d} {rms:10.1f} '
                  f'{(rms / e_noise) ** 2:9.1f} {rms / amid:11.4f} '
                  f'{rms / np.sqrt(amid):14.2f}')
        print('   photon statistics -> "resid/sqrt(A)" flat, "resid/peak" falling')
        print('   systematic shape  -> "resid/peak" flat, "resid/sqrt(A)" rising')
    return 0


if __name__ == '__main__':
    sys.exit(main())
