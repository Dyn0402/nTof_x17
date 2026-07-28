#!/usr/bin/env python3
"""Compare pulse-shape FIT QUALITY between reprocessing variants.

Hit counts alone cannot say whether a template change is an improvement: fewer
hits can mean "spurious splits removed" (good) or "real pulses lost" (bad).
The stored `chi2` of the shape fit settles it, and `amp` / `pileup1` say which
way the fit moved.  This is what showed that the v3_shapes templates helped the
walls (chi2 down, amp up) and hurt the liquids (chi2 doubled, amp down 30 %).

Runs happily on lxplus against EOS paths -- no need to pull GB of partials home:

    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
    python3 compare_fits.py old=a.root,b.root new=c.root,d.root
"""
import sys

import numpy as np
import uproot

TREES = ([f'WAL{a}' for a in 'ABCD'] + [f'LIQ{a}' for a in 'ABCD']
         + [f'PSS{a}' for a in 'ABCD'])
BRANCHES = ['chi2', 'amp', 'pileup1']


def read(files):
    acc = {}
    for p in files:
        f = uproot.open(p)
        present = {k.split(';')[0] for k in f.keys()}
        for t in TREES:
            if t not in present:
                continue
            a = f[t].arrays(BRANCHES, library='np')
            for k, v in a.items():
                acc.setdefault(t, {}).setdefault(k, []).append(v)
    return {t: {k: np.concatenate(v) for k, v in d.items()}
            for t, d in acc.items()}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    res, labs = {}, []
    for spec in sys.argv[1:]:
        lab, files = spec.split('=', 1)
        labs.append(lab)
        res[lab] = read(files.split(','))

    hdr = 'chi2p50 chi2p90  ampp50   pileup    nhits'
    print('tree   ' + '  '.join(f'{l[:16]:>{len(hdr)}s}' for l in labs))
    print('       ' + '  '.join(f'{hdr:>{len(hdr)}s}' for _ in labs))
    for t in TREES:
        cells = []
        for l in labs:
            d = res[l].get(t)
            if d is None:
                cells.append(f'{"-":>{len(hdr)}s}')
                continue
            c = d['chi2']
            c = c[np.isfinite(c) & (c > 0)]
            if not c.size:
                # AMPLITUDE OPTION != 2: no shape fit, so no chi2 (e.g. PSS)
                cells.append(f'{"no shape fit":>16s} '
                             f'{np.percentile(d["amp"], 50):7.0f} '
                             f'{np.mean(d["pileup1"] != 0):8.3f} {len(d["amp"]):8d}')
                continue
            cells.append(f'{np.percentile(c, 50):7.3f} {np.percentile(c, 90):7.3f} '
                         f'{np.percentile(d["amp"], 50):7.0f} '
                         f'{np.mean(d["pileup1"] != 0):8.3f} {len(d["amp"]):8d}')
        print(f'{t:6s} ' + '  '.join(cells))

    if len(labs) > 1:
        print(f'\nchange vs {labs[0]} (chi2 p50 down = better fit)')
        for t in TREES:
            base = res[labs[0]].get(t)
            if base is None:
                continue
            b = base['chi2'][np.isfinite(base['chi2']) & (base['chi2'] > 0)]
            bamp, bn = np.percentile(base['amp'], 50), len(base['amp'])
            b50 = np.percentile(b, 50) if b.size else None
            row = []
            for l in labs[1:]:
                d = res[l].get(t)
                if d is None:
                    row.append(f'{"-":>28s}')
                    continue
                c = d['chi2'][np.isfinite(d['chi2']) & (d['chi2'] > 0)]
                chi = (f'chi2 {np.percentile(c, 50) / b50 - 1:+6.1%}'
                       if (b50 and c.size) else f'chi2 {"n/a":>6s}')
                row.append(f'{chi}  '
                           f'amp {np.percentile(d["amp"], 50) / bamp - 1:+6.1%}  '
                           f'n {len(d["amp"]) / bn - 1:+6.1%}')
            print(f'  {t:6s} ' + '   '.join(row))
    return 0


if __name__ == '__main__':
    sys.exit(main())
