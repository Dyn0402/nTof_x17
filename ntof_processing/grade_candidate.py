#!/usr/bin/env python3
"""Grade a reprocessed n_TOF file directly, with no caches and no repo state.

`validate_reprocessing.py` is the full acceptance test but it needs the
ntof_io bunch-index machinery and a whole-run file.  This one reads a candidate
ROOT file (a merged run OR one of the per-job partials in `<out>/completed/`)
with uproot alone, so a variant can be graded within minutes of its first job
finishing instead of after the 26 GB merge.

Checks, on whatever bunches the file happens to contain:

  1. flash identification -- per tree, the fraction of BUNCHES whose tflash
     deviates >150 ns from the tree's mode.  Broken official processing:
     PSS 37-85 %.  Target <2 %.
  2. flash consistency -- per arm, the prompt-coincidence peak of large
     (amp>1000) PSS hits and of LIQ hits against the same arm's wall hits,
     after removing each tree's modal tflash.  Broken official processing:
     -375/+25/-325/-325 ns.  Target |peak| <25 ns.
  3. hit counts per tree, normalised per bunch, so a change that silently eats
     real hits (or floods the file with junk) is visible.

Usage:
    python grade_candidate.py <file.root> [<file.root> ...]
"""
import sys
from pathlib import Path

import numpy as np
import uproot

TREES = ([f'WAL{a}' for a in 'ABCD'] + [f'PSS{a}' for a in 'ABCD']
         + [f'LIQ{a}' for a in 'ABCD'] + ['PKUP'])


def mode(v, binw=10.0):
    v = v[np.isfinite(v)]
    if not v.size:
        return np.nan
    h, e = np.histogram(v, bins=np.arange(0.0, 20000.0, binw))
    return float(e[h.argmax()] + binw / 2)


def load(path):
    f = uproot.open(path)
    out = {}
    for t in TREES:
        if t not in [k.split(';')[0] for k in f.keys()]:
            continue
        br = ['BunchNumber', 'tflash', 'tof']
        if 'amp' in f[t].keys():
            br.append('amp')
        out[t] = f[t].arrays(br, library='np')
    return out


def grade(path):
    d = load(path)
    name = Path(path).name
    print(f'\n{"=" * 78}\n{name}')
    ref = d.get('WALA', d[list(d)[0]])
    bunches = np.unique(ref['BunchNumber'])
    print(f'  {len(bunches)} bunches: {bunches.min()}-{bunches.max()}')

    # ---- 1. flash identification, per bunch ----
    print(f'\n  [1] flash id      {"mode (ns)":>11s} {"bad bunches":>12s}   verdict')
    modes, ok1 = {}, True
    for t in TREES:
        if t not in d:
            continue
        b, tf = d[t]['BunchNumber'], d[t]['tflash']
        # one tflash per (tree, bunch): take the first hit of each bunch
        _, first = np.unique(b, return_index=True)
        per_bunch = tf[first]
        m = mode(per_bunch)
        modes[t] = m
        bad = float(np.mean(np.abs(per_bunch - m) > 150.0))
        ok1 &= bad < 0.02
        print(f'  {t:>16s} {m:11.1f} {bad:11.1%}   '
              f'{"PASS" if bad < 0.02 else "FAIL"}')

    # ---- 2. cross-detector consistency ----
    print(f'\n  [2] consistency vs same-arm wall (prompt coincidence peak)')
    ok2 = True
    for arm in 'ABCD':
        w = d.get(f'WAL{arm}')
        if w is None:
            continue
        wt = w['tof'] - modes[f'WAL{arm}']
        for fam, amp_min in ((f'PSS{arm}', 1000.0), (f'LIQ{arm}', 0.0)):
            h = d.get(fam)
            if h is None:
                continue
            ht = h['tof'] - modes[fam]
            sel = h['amp'] > amp_min if 'amp' in h else np.ones(len(ht), bool)
            dts = []
            for bn in np.unique(w['BunchNumber']):
                tw = np.sort(wt[(w['BunchNumber'] == bn) & (wt > 20e6)])
                m2 = (h['BunchNumber'] == bn) & sel & (ht > 20e6)
                tp = ht[m2]
                if not tw.size or not tp.size:
                    continue
                j = np.searchsorted(tw, tp)
                j0, j1 = np.clip(j - 1, 0, tw.size - 1), np.clip(j, 0, tw.size - 1)
                d0, d1 = tp - tw[j0], tp - tw[j1]
                dts.append(np.where(np.abs(d0) <= np.abs(d1), d0, d1))
            if not dts:
                continue
            a = np.concatenate(dts)
            a = a[np.abs(a) < 1000]
            if a.size < 50:
                print(f'  {fam:>16s}  too few pairs ({a.size})')
                continue
            hist, e = np.histogram(a, bins=200, range=(-1000, 1000))
            pk = float(0.5 * (e[1:] + e[:-1])[hist.argmax()])
            core = a[np.abs(a - pk) < 30]
            off = float(np.median(core))
            ok2 &= abs(off) < 25
            print(f'  {fam:>16s} {off:+8.1f} ns   n={a.size:6d}  '
                  f'{"PASS" if abs(off) < 25 else "FAIL"}')

    # ---- 3. hit counts ----
    print(f'\n  [3] hits per bunch')
    counts = {}
    for t in TREES:
        if t not in d:
            continue
        n = len(d[t]['BunchNumber'])
        nb = len(np.unique(d[t]['BunchNumber']))
        counts[t] = n / max(nb, 1)
        print(f'  {t:>16s} {counts[t]:10.0f}')
    print(f'\n  verdict: flash-id {"PASS" if ok1 else "FAIL"}, '
          f'consistency {"PASS" if ok2 else "FAIL"}')
    return counts


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    allc = {}
    for p in sys.argv[1:]:
        allc[Path(p).name] = grade(p)
    if len(allc) > 1:
        print(f'\n{"=" * 78}\nhits per bunch, side by side')
        names = list(allc)
        print(f'  {"tree":>8s} ' + ' '.join(f'{n[:14]:>15s}' for n in names))
        for t in TREES:
            if not any(t in c for c in allc.values()):
                continue
            base = allc[names[0]].get(t)
            row = []
            for n in names:
                v = allc[n].get(t)
                if v is None:
                    row.append(f'{"-":>15s}')
                elif n == names[0] or not base:
                    row.append(f'{v:15.0f}')
                else:
                    row.append(f'{v:9.0f}({v / base - 1:+5.1%})')
            print(f'  {t:>8s} ' + ' '.join(row))
    return 0


if __name__ == '__main__':
    sys.exit(main())
