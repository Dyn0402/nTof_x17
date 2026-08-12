#!/usr/bin/env python3
"""Which n_TOF runs are complete enough to slim -- decided from the data.

The cheap structural tests are not sufficient any more, for two reasons found on
2026-08-12:

  * `completed_ledger.py` calls a run SHORT when its partials are gapped, but
    n_TOF deletes partials after a successful merge, so 224566 and 224569 look
    short while their merged files are complete.
  * `slim_pipeline.config._require_complete` sizes a partial set against the raw
    files still on the EOS disk buffer. That buffer expires after two weeks, so
    for 224526 it computes want = ceil(22/4) = 6 and the truncated 6-partial
    official product PASSES. The guard cannot see the loss.

So this asks the only question that matters: **for every bunch that had beam,
did any detector record a hit?** It resolves the source the way
`slim_pipeline.config.ntof_files` does -- non-empty merged file preferred,
partial set as fallback -- and reads the same trees the slim will.

Usage (lxplus, LCG view sourced):
    python3 -u slim_ready.py --out=slim_ready.txt [--jobs=12] [--runs=a,b,c]
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import uproot

DAQ = Path('/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement')
COMPLETED = Path('/eos/experiment/ntof/processing/official/completed')
DONE = Path('/eos/experiment/ntof/processing/official/done')

# A bunch counts as recorded if ANY of these fired, so one dead detector cannot
# condemn a run.
PROBE = ['WALA', 'WALB', 'PSSA', 'PSSB']
BEAM_P = 1e12
READY_FRAC = 0.98


def source_for(run):
    """Resolve exactly as slim_pipeline.config.ntof_files does."""
    merged = DONE / f'run{run}.root'
    try:
        if merged.stat().st_size > 0:
            return [merged], 'merged'
    except OSError:
        pass
    parts = sorted((COMPLETED / str(run)).glob(f'run{run}_[0-9]*.root'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    if parts:
        return parts, 'partials'
    return [], 'none'


def check(run):
    out = {'run': run, 'source': 'none', 'nfiles': 0, 'bunches': 0,
           'beam': 0, 'covered': 0, 'frac': None, 'state': 'ABSENT',
           'contiguous': None, 'note': ''}
    files, kind = source_for(run)
    out['source'], out['nfiles'] = kind, len(files)
    if not files:
        return out
    if kind == 'partials':
        idx = [int(p.stem.split('_')[-1]) for p in files]
        out['contiguous'] = idx == list(range(1, len(idx) + 1))
    try:
        # The index tree is replicated in full in every partial, so one open
        # gives the whole run's bunch list regardless of how it was split.
        f0 = uproot.open(files[0])
        bn = f0['index']['BunchNumber'].array(library='np')
        pi = f0['index']['PulseIntensity'].array(library='np')
    except Exception as e:
        out['state'] = 'UNREADABLE'
        out['note'] = f'index: {type(e).__name__}'
        return out

    beam = pi > BEAM_P
    out['bunches'], out['beam'] = int(len(bn)), int(beam.sum())

    hits = set()
    bad = 0
    for p in files:
        try:
            g = uproot.open(p)
        except Exception:
            bad += 1
            continue
        for t in PROBE:
            if t in g:
                try:
                    # np.unique FIRST. A hit tree carries tens of millions of
                    # entries but only a few thousand distinct bunches, and
                    # set(arr.tolist()) materialises the whole thing as Python
                    # ints -- that took the process pool out with 16 workers.
                    a = g[t]['BunchNumber'].array(library='np')
                    hits |= set(np.unique(a).tolist())
                    del a
                except Exception:
                    bad += 1
    if bad:
        out['note'] = f'{bad} unreadable tree(s)'

    if out['beam'] == 0:
        out['state'] = 'NO_BEAM'
        return out
    hb = np.fromiter((b in hits for b in bn), bool, len(bn))
    cov = int((beam & hb).sum())
    out['covered'] = cov
    out['frac'] = round(cov / out['beam'], 5)
    out['state'] = 'READY' if out['frac'] >= READY_FRAC else 'INCOMPLETE'
    return out


def ranges(rs):
    rs = sorted(rs)
    if not rs:
        return []
    o, s, p = [], rs[0], rs[0]
    for r in rs[1:]:
        if r == p + 1:
            p = r
            continue
        o.append(str(s) if s == p else f'{s}-{p}')
        s = p = r
    o.append(str(s) if s == p else f'{s}-{p}')
    return o


def main():
    out_path, jobs, only = None, 12, None
    for a in sys.argv[1:]:
        if a.startswith('--out='):
            out_path = a.split('=', 1)[1]
        elif a.startswith('--jobs='):
            jobs = int(a.split('=', 1)[1])
        elif a.startswith('--runs='):
            only = [int(x) for x in a.split('=', 1)[1].split(',')]

    runs = only or sorted(int(d.name) for d in DAQ.iterdir()
                          if d.is_dir() and d.name.isdigit())
    res = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, r in enumerate(ex.map(check, runs), 1):
            res.append(r)
            if i % 25 == 0:
                print(f'  {i}/{len(runs)}', flush=True)

    res.sort(key=lambda r: r['run'])
    by = {}
    for r in res:
        by.setdefault(r['state'], []).append(r['run'])

    ready = sorted(by.get('READY', []) + by.get('NO_BEAM', []))
    bad = sorted(by.get('INCOMPLETE', []) + by.get('UNREADABLE', [])
                 + by.get('ABSENT', []))

    L = ['# n_TOF X17 EAR2 2026 -- runs ready to slim',
         f'# generated {datetime.now():%Y-%m-%d %H:%M} by campaign_qa/slim_ready.py',
         '# test: of the bunches with beam (PulseIntensity > 1e12), what fraction',
         f'#       recorded a hit in any of {"/".join(PROBE)}. READY at >= {READY_FRAC:.0%}.',
         '# source resolved as slim_pipeline.config.ntof_files does.',
         '',
         f'## READY -- {len(ready)} runs']
    rr = ranges(ready)
    L += [', '.join(rr[i:i + 8]) for i in range(0, len(rr), 8)]
    L += ['', f'## NOT READY -- {len(bad)} runs']
    if bad:
        for r in res:
            if r['run'] in bad:
                L.append(f"  {r['run']}  {r['state']:11s} {r['source']:8s} "
                         f"{r['nfiles']:3d} file(s)  "
                         f"{r['covered']}/{r['beam']} beam bunches"
                         + (f" = {100 * r['frac']:.1f} %" if r['frac'] is not None else '')
                         + (f"  [{r['note']}]" if r['note'] else ''))
    else:
        L.append('  (none)')
    nb = by.get('NO_BEAM', [])
    L += ['', f'## of the READY runs, {len(nb)} have no beam bunches at all',
          '# structurally fine, nothing to slim against protons.',
          ', '.join(ranges(nb)) or '(none)']

    text = '\n'.join(L) + '\n'
    print('\n' + text)
    if out_path:
        Path(out_path).write_text(text)
        Path(out_path).with_suffix('.json').write_text(json.dumps(res, indent=1))
        print(f'wrote {out_path} and {Path(out_path).with_suffix(".json")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
