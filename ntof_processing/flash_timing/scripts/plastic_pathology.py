#!/usr/bin/env python3
"""Why the plastic (PSS) flash timing is bad, quantified.

For each run/tree/channel, characterise the flash region:
  found      fraction of bunches in which the PSA put ANY hit at the flash time
  sigma      per-bunch spread of that hit's time (largest-amplitude hit within
             +-60 ns of the flash position)
  amp, satu  its amplitude and the PSA saturation flag
  spread90   width containing 90% of ALL hits in the flash window (the
             "erratic" measure -- small means the flash is a single clean pulse)
  nhit       hits per bunch in the +-1 us flash window (pile-up / fragmentation)

Compared against LIQ and the walls in the same runs, which share the digitiser
and the PSA but not the front end.
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'

RUNS = [(224357, '2026-07-11'), (224464, '2026-07-16'), (224572, '2026-07-26 post-FIFO')]
BIG = 25000


def pkup_map(d):
    pk = d['PKUP']
    p = pk[np.abs(pk['tof'] - pk['anchor']) < 4000]
    o = np.lexsort((-p['amp'], p['BunchNumber'])); p = p[o]
    f = np.ones(len(p), bool); f[1:] = p['BunchNumber'][1:] != p['BunchNumber'][:-1]
    p = p[f]
    p = p[np.abs(p['tof'] - np.median(p['tof'])) < 200]
    return dict(zip(p['BunchNumber'].tolist(), p['tof'].astype(float).tolist()))


def characterise(d, tree, ch, pm, half=1000):
    if tree not in d.files:
        return None
    r = d[tree]
    r = r[r['detn'] == ch]
    if len(r) < 100:
        return None
    ref = np.array([pm.get(int(b), np.nan) for b in r['BunchNumber']])
    dt = r['tof'] - ref
    fin = np.isfinite(dt)
    nb = len(set(r['BunchNumber'][fin].tolist()))
    if nb < 50:
        return None
    # locate the flash: modal position of the large hits
    big = fin & (r['amp'] > BIG) & (dt > -3000) & (dt < -500)
    if big.sum() < 30:
        big = fin & (dt > -3000) & (dt < -500)
        if big.sum() < 30:
            return None
    h, e = np.histogram(dt[big], bins=np.arange(-3000, -500, 5))
    peak = e[h.argmax()] + 2.5
    win = fin & (np.abs(dt - peak) < half)
    # NEUTRAL estimator: the largest-amplitude hit within +-60 ns of the flash
    # position, whatever its size.  "found" is then simply: did the PSA put any
    # hit at the flash time in this bunch?
    sel = fin & (np.abs(dt - peak) < 60)
    if sel.sum() < 30:
        return dict(tree=tree, ch=ch, peak=peak, found=0.0, sigma=np.nan, amp=np.nan,
                    satu=np.nan, fwhm=np.nan, rise=np.nan, area=np.nan,
                    spread90=np.nan, nhit=win.sum() / nb, nb=nb)
    b, v, a = r['BunchNumber'][sel], dt[sel], r['amp'][sel]
    o = np.lexsort((-a, b)); b, v, a = b[o], v[o], a[o]
    fi = np.ones(len(b), bool); fi[1:] = b[1:] != b[:-1]
    v, a = v[fi], a[fi]
    m = np.median(v); core = np.abs(v - m) < 60
    sat = r['satuflag'][sel][o][fi]
    q = np.percentile(dt[win], [5, 95])
    idx = np.arange(len(r))[sel][o][fi]          # rows of the chosen flash hits
    return dict(tree=tree, ch=ch, peak=float(peak), found=float(core.sum() / nb),
                sigma=float(1.4826 * np.median(np.abs(v[core] - np.median(v[core])))),
                amp=float(np.median(a[core])), satu=float(np.mean(sat[core] > 0)),
                fwhm=float(np.median(r['fwhm'][idx][core])),
                rise=float(np.median(r['risetime'][idx][core])),
                area=float(np.median(r['area'][idx][core])),
                spread90=float(q[1] - q[0]), nhit=float(win.sum() / nb), nb=nb)


def main():
    rows = []
    for run, ep in RUNS:
        p = DATA / f'flash_run{run}.npz'
        if not p.exists():
            print('missing', p, file=sys.stderr); continue
        d = np.load(p)
        pm = pkup_map(d)
        for tree in ('PSSA', 'PSSB', 'PSSC', 'PSSD',
                     'LIQA', 'LIQB', 'LIQC', 'LIQD', 'WALA', 'WALB', 'WALC', 'WALD'):
            for ch in (range(1, 3) if tree[:3] in ('PSS', 'LIQ') else range(1, 9)):
                r = characterise(d, tree, ch, pm)
                if r:
                    r.update(run=run, epoch=ep)
                    rows.append(r)
    cols = ['run', 'epoch', 'tree', 'ch', 'peak', 'found', 'sigma', 'amp', 'satu',
            'fwhm', 'rise', 'area', 'spread90', 'nhit', 'nb']
    with open(DATA / 'plastic_pathology.csv', 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in rows:
            fh.write(','.join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + '\n')
    print(f'wrote {DATA / "plastic_pathology.csv"} ({len(rows)} rows)')

    # summary by tree family and epoch
    print(f"\n{'epoch':22} {'family':7} {'found':>7} {'sigma':>7} {'amp':>8} {'satu':>6} "
          f"{'fwhm':>6} {'rise':>6} {'nhit/bunch':>10}")
    for run, ep in RUNS:
        for fam in ('WAL', 'PSS', 'LIQ'):
            s = [r for r in rows if r['run'] == run and r['tree'].startswith(fam)
                 and np.isfinite(r['sigma'])]
            if not s:
                continue
            g = lambda k: np.nanmean([r[k] for r in s])
            print(f"{ep:22} {fam:7} {g('found'):6.0%} {g('sigma'):7.1f} {g('amp'):8.0f} "
                  f"{g('satu'):6.0%} {g('fwhm'):6.0f} {g('rise'):6.1f} {g('nhit'):10.1f}")


if __name__ == '__main__':
    main()
