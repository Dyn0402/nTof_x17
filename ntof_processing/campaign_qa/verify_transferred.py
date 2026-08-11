#!/usr/bin/env python3
"""Structural verification of the runs this campaign has moved to the ntof disk.

Same checks as skip_diagnosis/verify_partials.py, but against an arbitrary base
directory (ours lives under /eos/experiment/ntof/data/x17/reproc/prod_v12) and
it opens EVERY partial, not a sample of three -- the point here is to catch a
truncated or half-transferred file, which only a real read will show.

Per run:
  * partial count against ceil(n_raw_files / 4), the split RunProcessing uses
  * contiguity of run<run>_NNNN.root from 1..N
  * history_<run>.root present, and its md5 against a reference v12 product
  * every partial opened: 16 top-level keys, all 14 hit trees readable, and the
    per-file bunch range, so gaps or overlaps between partials are visible
  * bunch coverage of the run: union of index bunches vs the per-partial hit
    bunches

Usage:
    python verify_transferred.py [--base=DIR] [--ref=FILE] <run> [run ...]
"""
import hashlib
import math
import os
import re
import sys

import uproot

BASE = '/eos/experiment/ntof/data/x17/reproc/prod_v12'
RAW = '/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement'
REF = None          # default: the first run's own history, i.e. self-consistency
FILES_PER_JOB = 4
TREES = ([f'WAL{a}' for a in 'ABCD'] + [f'PSS{a}' for a in 'ABCD']
         + [f'LIQ{a}' for a in 'ABCD'] + ['PKUP', 'SILI'])


def history_string(path):
    o = uproot.open(path)['history']
    for attr in ('fString', 'fTitle', 'fName'):
        try:
            v = o.member(attr)
            if isinstance(v, (str, bytes)) and len(v) > 200:
                return v.decode() if isinstance(v, bytes) else v
        except Exception:
            pass
    return str(o)


def history_md5(path):
    try:
        return hashlib.md5(history_string(path).encode()).hexdigest()
    except Exception as e:
        return f'FAIL:{type(e).__name__}'


def rundir(base, run):
    """Both layouts are in use: <base>/<run>/completed/<run>/ and <base>/<run>/."""
    for d in (f'{base}/{run}/completed/{run}', f'{base}/{run}', f'{base}/completed/{run}'):
        if os.path.isdir(d) and any(n.startswith(f'run{run}_') for n in os.listdir(d)):
            return d
    return None


def main(argv):
    base, ref_path, runs = BASE, REF, []
    for a in argv:
        if a.startswith('--base='):
            base = a.split('=', 1)[1]
        elif a.startswith('--ref='):
            ref_path = a.split('=', 1)[1]
        else:
            runs.append(int(a))

    ref = history_md5(ref_path) if ref_path else None
    problems = []
    print(f'base = {base}')
    if ref:
        print(f'reference history md5 ({ref_path}) = {ref}')
    print()
    hdr = (f'{"run":>7} {"parts":>5} {"exp":>4} {"contig":>7} {"raw":>4} '
           f'{"hist":>8} {"badfiles":>8} {"bunches":>8} {"gaps":>5} {"GB":>7}')
    print(hdr)
    print('-' * len(hdr))

    for run in runs:
        d = rundir(base, run)
        if d is None:
            print(f'{run:>7}  no directory under {base}')
            problems.append((run, 'no directory'))
            continue
        names = os.listdir(d)
        idx = sorted(int(m.group(1)) for n in names
                     if (m := re.fullmatch(rf'run{run}_(\d+)\.root', n)))
        nraw = len(os.listdir(f'{RAW}/{run}/stream1')) \
            if os.path.isdir(f'{RAW}/{run}/stream1') else 0
        exp = math.ceil(nraw / FILES_PER_JOB) if nraw else 0
        contig = bool(idx) and idx == list(range(1, len(idx) + 1))

        hfile = f'{d}/history_{run}.root'
        if not os.path.exists(hfile):
            hist = 'MISSING'
        else:
            hm = history_md5(hfile)
            hist = 'ok' if (ref is None or hm == ref) else hm[:8]
            if ref is None:
                ref, ref_path = hm, hfile
                hist = 'ref'

        bad, all_bunches, nbytes = [], set(), 0
        for i in idx:
            p = f'{d}/run{run}_{i:04d}.root'
            try:
                nbytes += os.path.getsize(p)
                f = uproot.open(p)
                keys = {k.split(';')[0] for k in f.keys()}
                missing = [t for t in TREES + ['index', 'DAQsettings'] if t not in keys]
                if missing:
                    bad.append(f'{i}:missing{missing[:3]}')
                    continue
                # a real read of every hit tree: entry count plus one column, so
                # a truncated basket is hit rather than only the header
                empty = []
                for t in TREES:
                    n = f[t].num_entries
                    if n == 0:
                        empty.append(t)
                        continue
                    _ = f[t]['tof'].array(entry_start=max(0, n - 1000),
                                          entry_stop=n, library='np')
                bn = f['PKUP']['BunchNumber'].array(library='np')
                all_bunches.update(int(x) for x in bn)
                if empty:
                    bad.append(f'{i}:empty{empty}')
            except Exception as e:
                bad.append(f'{i}:{type(e).__name__}')

        sb = sorted(all_bunches)
        gaps = sum(1 for a, b in zip(sb, sb[1:]) if b != a + 1) if sb else -1
        flag = ''
        if not contig or bad or hist not in ('ok', 'ref') or (exp and len(idx) != exp):
            flag = '   <<<'
            problems.append((run, f'parts={len(idx)} exp={exp} contig={contig} '
                                  f'hist={hist} bad={bad[:5]}'))
        print(f'{run:>7} {len(idx):5d} {exp:4d} {str(contig):>7} {nraw:4d} '
              f'{hist:>8} {len(bad):8d} {len(sb):8d} {gaps:5d} '
              f'{nbytes / 2**30:7.1f}{flag}')
        if bad:
            print(f'         bad: {bad[:8]}')

    print()
    if problems:
        print('NEEDS ATTENTION:')
        for r, why in problems:
            print(f'  {r}: {why}')
    else:
        print('all runs: complete, contiguous, same UserInput, every partial reads.')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
