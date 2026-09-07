#!/usr/bin/env python3
"""Hit-for-hit comparison of our product against n_TOF's, on the same bunches.

For the runs that exist in BOTH processings this replaces the statistical
equivalence argument with a direct one: take a window of bunches, pull every
hit both processings recorded for them, and compare the columns.

The two processings split a run into partials differently (n_TOF's job takes a
different number of raw files than ours), so partial N is not partial N.  The
matching is therefore done on `BunchNumber`: our partial is opened, its bunch
range read, and the official partials that cover it are found by bisection on
the first/last entry of each -- one cheap read per candidate rather than a full
scan.

The first and last bunch of the overlap are dropped: a bunch that straddles a
raw-file boundary would be split across two jobs and appear truncated in one of
them, which is a bookkeeping artefact, not a difference in the reconstruction.

Usage:
    python3 -u compare_identity.py --run=224574 \\
        --ours=/eos/experiment/ntof/data/x17/reproc/prod_v11/224574/completed/224574 \\
        --official=/eos/experiment/ntof/processing/official/completed/224574 \\
        [--part=8] [--bunches=5] [--json=out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import uproot

TREES = ['WALA', 'WALB', 'WALC', 'WALD',
         'PSSA', 'PSSB', 'PSSC', 'PSSD',
         'LIQA', 'LIQB', 'LIQC', 'LIQD',
         'SILI', 'PKUP']

# every per-hit column except the waveform blob and the wall-clock stamps,
# which are recorded per job and are not a reconstruction output
COLS = ['detn', 'tflash', 'tof', 'peak_tof', 'amp', 'area', 'fwhm', 'fwtm',
        'ratio_der', 'lobe_asymmetry', 'pileup1', 'pileup2', 'polarity',
        'risetime', 'satuflag', 'chi2', 'area_0', 'amp_0', 'area2',
        'afast', 'aslow', 'isAlpha']

KEY = 'BunchNumber'


def partials(d):
    return sorted(Path(d).glob('run[0-9]*_[0-9]*.root'),
                  key=lambda p: int(p.stem.split('_')[-1]))


def bunch_span(path, tree):
    """(first, last) BunchNumber of `tree` in `path`, two one-entry reads."""
    t = uproot.open(path)[tree]
    n = t.num_entries
    if n == 0:
        return None
    lo = int(t[KEY].array(entry_start=0, entry_stop=1, library='np')[0])
    hi = int(t[KEY].array(entry_start=n - 1, entry_stop=n, library='np')[0])
    return lo, hi


def rows_for(path, tree, bunches):
    """All hits of `tree` in `path` whose bunch is in `bunches`, as a dict.

    Only `BunchNumber` is read over the whole tree; the wanted bunches are
    contiguous in it, so the 22 payload columns are then read for that entry
    range alone.  Reading all of them over an 800 MB partial and filtering
    afterwards costs minutes per tree.
    """
    t = uproot.open(path)[tree]
    have = [c for c in COLS if c in t.keys()]
    bn = t[KEY].array(library='np')
    sel = np.isin(bn, list(bunches))
    if not sel.any():
        return {c: np.empty(0) for c in [KEY] + have}, have
    idx = np.flatnonzero(sel)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    a = t.arrays(have, entry_start=lo, entry_stop=hi, library='np')
    keep = sel[lo:hi]
    out = {KEY: bn[lo:hi][keep]}
    out.update({c: a[c][keep] for c in have})
    return out, have


def order(d, cols):
    """Stable ordering inside a bunch, so the two lists line up.

    np.lexsort takes the primary key LAST, so the sequence is reversed here.
    """
    by = [c for c in ('BunchNumber', 'detn', 'tof', 'amp') if c in d]
    return np.lexsort([d[c] for c in reversed(by)])


def compare(run, ours_dir, off_dir, part_idx, n_bunches, trees=TREES):
    op = partials(ours_dir)
    fp = partials(off_dir)
    if not op or not fp:
        raise SystemExit(f'run {run}: ours={len(op)} official={len(fp)} partials')
    mine = op[min(part_idx, len(op) - 1)]

    span = bunch_span(mine, 'WALA')
    if span is None:
        raise SystemExit(f'{mine} has no WALA entries')
    lo, hi = span
    print(f'run {run}: ours {mine.name} covers bunches {lo}-{hi} '
          f'({len(op)} partials, official has {len(fp)})')

    # official partials overlapping [lo, hi]
    cover = []
    for p in fp:
        s = bunch_span(p, 'WALA')
        if s and not (s[1] < lo or s[0] > hi):
            cover.append((p, s))
    if not cover:
        raise SystemExit(f'run {run}: no official partial covers {lo}-{hi}')
    print('  official partials covering it: ' +
          ', '.join(f'{p.name}[{s[0]}-{s[1]}]' for p, s in cover))

    # bunches fully inside both, minus the boundary bunch on each side
    olo = max(lo, min(s[0] for _, s in cover))
    ohi = min(hi, max(s[1] for _, s in cover))
    common = list(range(olo + 1, ohi))
    if len(common) > n_bunches:
        mid = len(common) // 2
        common = common[max(0, mid - n_bunches // 2):][:n_bunches]
    if not common:
        raise SystemExit(f'run {run}: no interior bunch shared')
    print(f'  comparing bunches {common[0]}-{common[-1]} ({len(common)})')

    out = {}
    for tree in trees:
        try:
            mine_d, cols = rows_for(mine, tree, common)
        except Exception as e:
            out[tree] = {'error': f'ours: {type(e).__name__}'}
            continue
        offs = []
        for p, _ in cover:
            try:
                d, _ = rows_for(p, tree, common)
            except Exception:
                continue
            offs.append(d)
        if not offs:
            out[tree] = {'error': 'official: unreadable'}
            continue
        off_d = {c: np.concatenate([d[c] for d in offs]) for c in mine_d}

        n_ours, n_off = mine_d[KEY].size, off_d[KEY].size
        rec = {'n_ours': int(n_ours), 'n_official': int(n_off)}
        if n_ours != n_off:
            rec['verdict'] = 'DIFFERENT (hit count)'
            out[tree] = rec
            print(f'  {tree:5s} {n_ours:7d} vs {n_off:7d} hits   DIFFERENT (count)')
            continue
        io, ifz = order(mine_d, cols), order(off_d, cols)
        worst, worst_col, nbad = 0.0, '', 0
        per_col = {}
        for c in cols:
            x = mine_d[c][io].astype('f8')
            y = off_d[c][ifz].astype('f8')
            d = np.abs(x - y)
            d = d[np.isfinite(d)]
            n_c = int((d > 0).sum())
            if n_c:
                per_col[c] = {'cells': n_c, 'frac': n_c / max(1, d.size),
                              'max_abs': float(d.max())}
            if d.size and d.max() > worst:
                worst, worst_col = float(d.max()), c
            nbad += n_c
        rec['max_abs_diff'] = worst
        rec['max_diff_column'] = worst_col
        rec['differing_cells'] = nbad
        rec['columns_differing'] = per_col
        rec['verdict'] = 'IDENTICAL' if nbad == 0 else 'DIFFERENT (values)'
        out[tree] = rec
        print(f'  {tree:5s} {n_ours:7d} hits   {rec["verdict"]:20s} '
              f'worst |d| = {worst:g} ({worst_col or "-"})')
        if per_col:
            print('        ' + ', '.join(
                f'{c} {v["cells"]} ({100 * v["frac"]:.2f} %, max {v["max_abs"]:g})'
                for c, v in sorted(per_col.items(),
                                   key=lambda kv: -kv[1]['cells'])))
    return {'run': run, 'ours_partial': mine.name,
            'bunches': [int(common[0]), int(common[-1])], 'trees': out}


def main():
    run = None
    ours = off = None
    part_idx, n_bunches, outjson = 4, 5, None
    trees = TREES
    for a in sys.argv[1:]:
        k, _, v = a.partition('=')
        if k == '--run':
            run = int(v)
        elif k == '--ours':
            ours = v
        elif k == '--official':
            off = v
        elif k == '--part':
            part_idx = int(v)
        elif k == '--bunches':
            n_bunches = int(v)
        elif k == '--trees':
            trees = v.split(',')
        elif k == '--json':
            outjson = v
    if not (run and ours and off):
        print(__doc__)
        return 2
    res = compare(run, ours, off, part_idx, n_bunches, trees)
    if outjson:
        Path(outjson).write_text(json.dumps(res, indent=1))
        print(f'wrote {outjson}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
