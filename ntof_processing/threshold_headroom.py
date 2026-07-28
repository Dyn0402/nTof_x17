#!/usr/bin/env python3
"""Which elimination cuts are actually BINDING, and how much is left on the table?

Before spending a condor round-trip on "loosen everything", ask the data which
knobs still have signal behind them. For each tree this prints, for the
quantities the UserInput cuts on (amplitude, area/amp, width):

  * the low-end shape of the distribution -- a spectrum that piles up against
    the cut has signal behind it; one that dies before the cut does not
  * the fraction of hits sitting within a factor 2 of the current cut

Run against a reprocessed file (partials are fine):
    python threshold_headroom.py <file.root> [<file.root> ...]
"""
import sys
from collections import defaultdict

import numpy as np
import uproot

TREES = ([f'WAL{a}' for a in 'ABCD'] + [f'PSS{a}' for a in 'ABCD']
         + [f'LIQ{a}' for a in 'ABCD'])

# current cuts in userinputs/v4_walshapes: amp thr, (area/amp lo, hi), (width lo, hi)
CUTS = {'WAL': (50.0, (10.0, 200.0), (5.0, 4000.0)),
        'PSS': (50.0, (1.0, 60.0), (10.0, 3000.0)),
        'LIQ': (50.0, (1.0, 60.0), (1.0, 5000.0))}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    acc = defaultdict(lambda: defaultdict(list))
    for p in sys.argv[1:]:
        f = uproot.open(p)
        present = {k.split(';')[0] for k in f.keys()}
        for t in TREES:
            if t not in present:
                continue
            a = f[t].arrays(['amp', 'area', 'fwhm'], library='np')
            for k, v in a.items():
                acc[t][k].append(v)

    print(f'{"tree":6s} {"amp cut":>8s} {"p1":>7s} {"p5":>7s} {"p25":>7s} '
          f'{"p50":>7s} | {"<2x cut":>8s} {"<3x cut":>8s} | '
          f'{"a/a p1":>7s} {"a/a p99":>8s} {"a/a cut":>12s}')
    for t in TREES:
        if t not in acc:
            continue
        amp = np.concatenate(acc[t]['amp'])
        area = np.concatenate(acc[t]['area'])
        amp_cut, (al, ah), _ = CUTS[t[:3]]
        pos = np.abs(amp)
        with np.errstate(divide='ignore', invalid='ignore'):
            aoa = np.abs(area) / np.where(pos > 0, pos, np.nan)
        aoa = aoa[np.isfinite(aoa)]
        print(f'{t:6s} {amp_cut:8.0f} {np.percentile(pos, 1):7.0f} '
              f'{np.percentile(pos, 5):7.0f} {np.percentile(pos, 25):7.0f} '
              f'{np.percentile(pos, 50):7.0f} | '
              f'{np.mean(pos < 2 * amp_cut):8.1%} {np.mean(pos < 3 * amp_cut):8.1%} | '
              f'{np.percentile(aoa, 1):7.2f} {np.percentile(aoa, 99):8.1f} '
              f'{f"{al}..{ah}":>12s}')

    print('\nreading: a spectrum piling up against the cut (large "<2x cut")')
    print('has signal behind it; one whose p1 is far above the cut does not.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
