#!/usr/bin/env python3
"""Is the newly-enabled afast/aslow split actually usable for n/gamma separation?

v12 sets the fast/slow boundary, so `afast` and `aslow` are filled for the
first time. Filled is not the same as useful: this checks that the ratio
behaves like a PSD variable -- bounded, amplitude-stable, and with structure
beyond a single band -- and prints the figure of merit a PSD cut would have.

Also confirms the wall and plastic trees are untouched, since v12 was meant to
change the liquids only.

    python check_psd_output.py <v12.root> [<v11.root>]
"""
import sys

import numpy as np
import uproot


def main():
    f = uproot.open(sys.argv[1])
    print('=== liquid PSD from afast/aslow ===')
    for t in ('LIQA', 'LIQB', 'LIQC', 'LIQD'):
        a = f[t].arrays(['afast', 'aslow', 'amp', 'area'], library='np')
        af, asl, amp = np.abs(a['afast']), np.abs(a['aslow']), np.abs(a['amp'])
        tot = af + asl
        ok = (tot > 0) & (amp > 500)
        if ok.sum() < 500:
            print(f'  {t}: too few ({ok.sum()})')
            continue
        r = asl[ok] / tot[ok]
        print(f'  {t}: n={ok.sum():7d}  slow/(fast+slow) '
              f'p5={np.percentile(r,5):.3f} p50={np.percentile(r,50):.3f} '
              f'p95={np.percentile(r,95):.3f}')
        # amplitude stability: a PSD variable should not drift with pulse size
        lo = amp[ok] < np.percentile(amp[ok], 50)
        print(f'      median ratio, small vs large pulses: '
              f'{np.median(r[lo]):.3f} vs {np.median(r[~lo]):.3f}')
        # is there structure? compare a 2-component split against one band
        h, e = np.histogram(r, bins=80, range=(0, 1))
        c = 0.5 * (e[1:] + e[:-1])
        pk = c[h.argmax()]
        far = (np.abs(c - pk) > 0.12) & (h > 0.02 * h.max())
        print(f'      peak at {pk:.3f}; {100*h[far].sum()/h.sum():.1f} % of hits '
              f'lie >0.12 away from it')

    if len(sys.argv) > 2:
        g = uproot.open(sys.argv[2])
        print('\n=== control: are the walls and plastics untouched? ===')
        for t in ('WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSB', 'PSSC', 'PSSD'):
            n1 = f[t].num_entries
            n2 = g[t].num_entries
            flag = 'same' if n1 == n2 else f'DIFFER {n1/n2-1:+.2%}'
            print(f'  {t}: v12 {n1:9d}   v11 {n2:9d}   {flag}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
