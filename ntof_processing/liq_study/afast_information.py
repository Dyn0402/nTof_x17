#!/usr/bin/env python3
"""T5: does the 5000/30 fast/slow boundary put any INFORMATION into `afast`?

`check_psd_output.py` answers "is it filled" -- `afast` fills, `aslow` stays 0
because the slow component lies outside the reconstructed pulse boundary. Filled
is not useful. The pulse-shape-discrimination content, if there is any, has to
show up as `(area - afast)/area`: the part of the reconstructed area that falls
after 30 ns. On raw waveforms the tail/total ratio of a liquid pulse is a tight
band at 0.21 above 3000 ADC (`liq_psd.png`), so that is the target to reproduce.

The comparison is only fair on ISOLATED pulses -- a neighbour inside the pulse
boundary contributes to `area` and to `afast` in unrelated proportions -- so
this selects late-time hits with no other hit in the same tree within +-`GAP`,
which is the same isolation the raw-waveform measurement used.

Decision this feeds (../archive/PRE_SHIP_TESTS.md T5):
  * band reproduced on the isolated subset -> keep the boundary, document that
    `aslow` is empty and `afast` is only meaningful for isolated pulses
  * `afast` degenerate (== area for everything) -> drop the boundary, because a
    filled-but-meaningless PSD field in the official output is worse than an
    empty one

    python afast_information.py <file.root> [more.root ...]
"""
import sys

import numpy as np
import uproot

TREES = ('LIQA', 'LIQB', 'LIQC', 'LIQD')
T_LO, T_HI = 1_000_000.0, 18_000_000.0     # late-time, clear of the flash
GAP = 200.0                                # ns of quiet either side = isolated
AMP_MIN = 3000.0                           # where the raw-waveform band is tight
# The 0.21 quoted in FINDINGS_liquids.md is the tail fraction beyond 12 ns after
# the peak. `afast` splits at 30 ns, so 0.21 is NOT the number to compare
# against -- using it makes the PSA look twice as wrong as it is. Re-measured on
# the same isolated raw pulses at a 30 ns split: 0.113 median, p16-p84 spread
# 0.035, i.e. a TIGHT band. That spread is the real target: reproducing the
# median is easy, reproducing the width is what makes a PSD variable.
RAW_BAND = 0.113
RAW_SPREAD = 0.035


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for path in sys.argv[1:]:
        f = uproot.open(path)
        print(f'\n=== {path.split("/")[-1]} ===')
        print(f'{"tree":6s} {"isolated":>9s} {"afast>0":>8s} {"afast==area":>12s} '
              f'{"(area-afast)/area":>28s}')
        print(f'{"":6s} {"n":>9s} {"":>8s} {"":>12s} '
              f'{"p16":>8s} {"p50":>8s} {"p84":>8s}')
        for t in TREES:
            a = f[t].arrays(['BunchNumber', 'tof', 'tflash', 'amp', 'area',
                             'afast', 'aslow'], library='np')
            tt = a['tof'] - a['tflash']
            amp, area, af = np.abs(a['amp']), np.abs(a['area']), np.abs(a['afast'])
            m = (tt > T_LO) & (tt < T_HI) & (amp > AMP_MIN) & (area > 0)
            if m.sum() < 200:
                print(f'{t:6s} {m.sum():9d}   too few')
                continue
            # isolation, per bunch: no neighbouring hit of the same tree inside GAP
            b, tsel = a['BunchNumber'][m], tt[m]
            o = np.lexsort((tsel, b))
            b, tsel = b[o], tsel[o]
            af_s, area_s = af[m][o], area[m][o]
            same = np.zeros(len(b), bool)
            same[1:] = b[1:] == b[:-1]
            dprev = np.full(len(b), np.inf)
            dnext = np.full(len(b), np.inf)
            dprev[1:] = np.where(same[1:], tsel[1:] - tsel[:-1], np.inf)
            dnext[:-1] = np.where(same[1:], tsel[1:] - tsel[:-1], np.inf)
            iso = (dprev > GAP) & (dnext > GAP)
            if iso.sum() < 100:
                print(f'{t:6s} {iso.sum():9d}   too few isolated')
                continue
            r = 1.0 - af_s[iso] / area_s[iso]
            deg = float(np.mean(np.abs(af_s[iso] - area_s[iso])
                                < 1e-6 * np.abs(area_s[iso])))
            print(f'{t:6s} {iso.sum():9d} {np.mean(af_s[iso] > 0):7.1%} '
                  f'{deg:11.1%} {np.percentile(r, 16):8.3f} '
                  f'{np.percentile(r, 50):8.3f} {np.percentile(r, 84):8.3f}')
            # A PSD variable must be amplitude-STABLE (the tail fraction of a
            # given particle type does not depend on how much light it made) and
            # must not simply track the reconstructed pulse LENGTH, which is
            # what it would do if the boundary is only cutting a jittering
            # window out of the tail.
            amp_i = amp[m][o][iso]
            ln = area_s[iso] / np.abs(f[t].arrays(['amp'], library='np')['amp'][m][o][iso])
            half = amp_i < np.median(amp_i)
            rho = float(np.corrcoef(np.log(ln), r)[0, 1])
            print(f'       median ratio small vs large pulses: '
                  f'{np.median(r[half]):.3f} vs {np.median(r[~half]):.3f}   '
                  f'corr(ratio, area/amp) = {rho:+.2f}')
    print(f'\nraw-waveform target on the same isolated pulses, split at the same '
          f'30 ns:\n  median {RAW_BAND:.3f}, p16-p84 spread {RAW_SPREAD:.3f} '
          f'(a tight band -- there is only one pulse class)')
    print('\nThe median is the easy part. A PSD variable has to reproduce the '
          'WIDTH:\nif the per-hit spread is much larger than the physical band, '
          'the field is\nmeasuring reconstruction noise and cannot discriminate '
          'anything per pulse.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
