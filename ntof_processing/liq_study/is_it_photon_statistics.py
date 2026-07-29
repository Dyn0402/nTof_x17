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

    python is_it_photon_statistics.py <liq_pulses.npz> [TREE ...]

With no tree arguments every LIQ* family present in the npz is measured. The
scaling was originally quoted on LIQA/LIQD only; LIQB/LIQC carry fewer isolated
pulses but enough to check that the floor is a property of the whole family
(PRE_SHIP_TESTS.md T6).
"""
import sys

import numpy as np

PRE = 20


def main():
    z = np.load(sys.argv[1])
    trees = sys.argv[2:] or sorted(k[:-5] for k in z.files if k.endswith('_rows'))
    summary = []
    for tree in trees:
        rows, amps = z[f'{tree}_rows'], z[f'{tree}_amps']
        # Drop under-range pulses. These do NOT clip flat -- the samples that
        # would go below ADC 0 wrap to ~65535, so a normalised row picks up a
        # spike near -1 just after the peak, and the reported `amp` is whatever
        # the last un-wrapped sample on the rising edge happened to be. Both the
        # shape and the amplitude are meaningless, so they cannot be part of a
        # residual-scaling measurement. See WRAPAROUND in FINDINGS_liquids.md.
        wrapped = rows.min(axis=1) < -0.5
        m = (amps > 800) & ~wrapped
        nw = int((amps > 800).sum() - m.sum())
        rows, amps = rows[m], amps[m]
        if nw:
            print(f'\n[{tree}] dropped {nw} under-range (wrapped) pulses')
        if len(rows) < 100:
            print(f'\n{tree}: only {len(rows)} pulses above amp 800 -- skipped')
            continue
        # one template from the whole sample; residual measured per pulse
        med = np.median(rows, axis=0)
        tmpl = med / med.max()
        sig_norm = float(np.median(np.std(rows[:, :12], axis=1)))

        print(f'\n{tree}: {len(rows)} pulses, amp {amps.min():.0f}-{amps.max():.0f}')
        print(f'{"amp bin":>16s} {"n":>5s} {"resid RMS":>10s} {"chi2 vs":>9s} '
              f'{"resid/peak":>11s} {"resid/sqrt(A)":>14s} {"coher.":>8s}')
        print(f'{"":>16s} {"":>5s} {"[ADC]":>10s} {"e-noise":>9s} {"":>11s} '
              f'{"":>14s} {"/random":>8s}')
        # Bin adaptively. LIQC carries ~10x fewer isolated pulses than LIQD, so a
        # fixed six-bin split leaves every bin under threshold and the tree drops
        # out silently -- which is how LIQB/LIQC came to be untested.
        nmin = 40
        nbin = int(np.clip(len(rows) // nmin, 2, 6))
        edges = np.percentile(amps, np.linspace(0, 100, nbin + 1))
        scal, peaks, arange = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (amps >= lo) & (amps < hi)
            if s.sum() < nmin:
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
            # COHERENCE decides what a rising resid/sqrt(A) means. A shape the
            # template is missing is the SAME shape in every pulse, so it
            # survives averaging: coherent/total -> 1. Shot noise and residual
            # pileup average away: coherent/total -> 1/sqrt(n). Only the first
            # is something a richer template basis could absorb.
            mean_r = R.mean(axis=0)
            coh = float(np.sqrt(np.mean(mean_r ** 2)) / rms) if rms > 0 else 0.0
            floor = 1.0 / np.sqrt(s.sum())
            print(f'{lo:7.0f}-{hi:<8.0f} {s.sum():5d} {rms:10.1f} '
                  f'{(rms / e_noise) ** 2:9.1f} {rms / amid:11.4f} '
                  f'{rms / np.sqrt(amid):14.2f} {coh / floor:8.1f}')
            scal.append(rms / np.sqrt(amid))
            peaks.append(rms / amid)
            arange.append(amid)
        print('   photon statistics -> "resid/sqrt(A)" flat, "resid/peak" falling')
        print('   systematic shape  -> "resid/peak" flat, "resid/sqrt(A)" rising')
        if len(scal) > 1:
            spread = max(scal) / min(scal) - 1
            drop = max(peaks) / min(peaks)
            span = max(arange) / min(arange)
            summary.append((tree, len(rows), span, spread, drop))
            print(f'   -> resid/sqrt(A) spans {spread:+.0%} over a factor '
                  f'{span:.0f} in amplitude; resid/peak falls {drop:.1f}x')

    if summary:
        print(f'\n{"=" * 68}\nT6 summary -- the floor is photon statistics where '
              f'resid/sqrt(A) is flat')
        print(f'{"tree":6s} {"pulses":>7s} {"amp span":>9s} '
              f'{"resid/sqrt(A) spread":>21s} {"resid/peak fall":>16s}')
        for tree, n, span, spread, drop in summary:
            verdict = 'flat' if abs(spread) < 0.15 else 'NOT FLAT'
            print(f'{tree:6s} {n:7d} {span:8.0f}x {spread:20.0%} '
                  f'{drop:15.1f}x   {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
