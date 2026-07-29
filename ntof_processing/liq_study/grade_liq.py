#!/usr/bin/env python3
"""Grade the liquid trees between two processings.

Beyond fit quality this reports the fill fraction of `afast`/`aslow`, which is
the pulse-shape-discrimination observable: it is 0 % in every processing so far
because the fast/slow boundary was never set.

    python grade_liq.py label=file.root [label2=file.root ...]
"""
import sys

import numpy as np
import uproot

TREES = ('LIQA', 'LIQB', 'LIQC', 'LIQD')
BR = ['chi2', 'amp', 'area', 'pileup1', 'afast', 'aslow', 'satuflag']


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    specs = [s.split('=', 1) for s in sys.argv[1:]]
    data = {lab: uproot.open(p) for lab, p in specs}
    hdr = (f'{"tree":5s} {"variant":30s} {"nhits":>9s} {"chi2p50":>8s} '
           f'{"chi2p90":>8s} {"amp p50":>8s} {"pileup":>7s} {"satu":>6s} '
           f'{"afast filled":>13s}')
    print(hdr)
    print('-' * len(hdr))
    base = {}
    for t in TREES:
        for lab, f in data.items():
            if t not in {k.split(';')[0] for k in f.keys()}:
                continue
            a = f[t].arrays(BR, library='np')
            c = a['chi2']
            c = c[np.isfinite(c) & (c > 0)]
            c50 = np.percentile(c, 50) if c.size else float('nan')
            c90 = np.percentile(c, 90) if c.size else float('nan')
            n = len(a['amp'])
            if t not in base:
                base[t] = (n, c50)
            dn = n / base[t][0] - 1
            dc = c50 / base[t][1] - 1 if base[t][1] == base[t][1] else float('nan')
            print(f'{t:5s} {lab:30s} {n:9d} {c50:8.3f} {c90:8.2f} '
                  f'{np.percentile(a["amp"], 50):8.0f} '
                  f'{np.mean(a["pileup1"] != 0):7.3f} '
                  f'{np.mean(a["satuflag"] != 0):6.3f} '
                  f'{np.mean(a["afast"] != 0):12.1%}'
                  + ('' if lab == specs[0][0] else
                     f'   [n {dn:+.1%}, chi2 {dc:+.1%}]'))
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
