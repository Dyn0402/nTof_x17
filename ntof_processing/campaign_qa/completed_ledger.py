#!/usr/bin/env python3
"""Simplified bookkeeping: forget the merge, ask only whether the partials cover the run.

The merge is not a quality signal -- large runs are left unmerged on purpose and
the partial set is the same processing. So the only question worth asking of
`official/completed/<run>/` is: **do the partials cover the whole run?**

Counting them against `ceil(raw files / N)` does not answer it:

  * n_TOF used TWO split sizes -- 10 raw files per job before 2026-07-08 and 4
    after (224302, 224602, 224627, 224630, 224634, 224667, 224670, 224682,
    224686 all match ceil(raw/10) exactly), so a single divisor mis-flags the
    older half of the campaign;
  * the raw has aged off disk for 309 of the 445 runs, so for most of them there
    is no denominator at all.

What does answer it is inside the files. The `index` tree is replicated IN FULL
in every partial: it lists every bunch of the whole run regardless of which
partial you open. So one open gives the run's true bunch range, and the last
partial's own hits say where the processing actually stopped. A set is COVERED
when its file indices run 1..N with no gap AND its hits reach the last bunch the
run recorded.

States:
  COVERED      contiguous partials spanning the run -- usable, merged or not
  SHORT        partials present but they stop early, or an index is missing
  MERGED_ONLY  partials cleaned up; the merged file is the only product (224569)
  ABSENT       nothing at all

Usage:
    python3 -u completed_ledger.py [--csv=out.csv] [--json=out.json] [--runs=a,b]
"""
import csv
import json
import sys
from pathlib import Path

import uproot

DAQ = Path('/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement')
COMPLETED = Path('/eos/experiment/ntof/processing/official/completed')
DONE = Path('/eos/experiment/ntof/processing/official/done')
OURS = [Path('/eos/experiment/ntof/data/x17/reproc/prod_v12'),
        Path('/eos/experiment/ntof/data/x17/reproc/prod_v11')]

# a hit tree that is populated even on a quiet run; the walls see the flash
PROBE = ['WALA', 'WALB', 'PSSA', 'LIQA']


def partials(d: Path, run: int):
    return sorted(d.glob(f'run{run}_[0-9]*.root'),
                  key=lambda p: int(p.stem.split('_')[-1]))


def coverage(ps, run):
    """(n, contiguous, run_first, run_last, hits_last, state).

    Two opens: the first partial for the run-wide `index`, the last partial for
    the bunch its hits actually reach.
    """
    idx = [int(p.stem.split('_')[-1]) for p in ps]
    contig = idx == list(range(1, len(idx) + 1))
    try:
        bn = uproot.open(ps[0])['index']['BunchNumber'].array(library='np')
        run_first, run_last = int(bn.min()), int(bn.max())
    except Exception as e:
        return len(ps), contig, None, None, None, f'UNREADABLE:{type(e).__name__}'

    last = None
    for tree in PROBE:
        try:
            t = uproot.open(ps[-1])[tree]
            n = t.num_entries
            if not n:
                continue
            v = int(t['BunchNumber'].array(entry_start=n - 1, entry_stop=n,
                                           library='np')[0])
            last = v if last is None else max(last, v)
        except Exception:
            continue
    if last is None:
        # a run with no hits anywhere: contiguity is all there is to check
        return len(ps), contig, run_first, run_last, None, \
            ('COVERED' if contig else 'SHORT')

    covered = contig and last >= run_last
    return len(ps), contig, run_first, run_last, last, \
        ('COVERED' if covered else 'SHORT')


def main():
    out_csv = out_json = None
    only = None
    for a in sys.argv[1:]:
        if a.startswith('--csv='):
            out_csv = a.split('=', 1)[1]
        elif a.startswith('--json='):
            out_json = a.split('=', 1)[1]
        elif a.startswith('--runs='):
            only = {int(x) for x in a.split('=', 1)[1].split(',')}

    runs = sorted(int(d.name) for d in DAQ.iterdir()
                  if d.is_dir() and d.name.isdigit())
    if only:
        runs = [r for r in runs if r in only]

    hdr = (f'{"run":>7} {"official":>9} {"parts":>6} {"bunches":>16} '
           f'{"reach":>7} {"ours":>9} {"parts":>6} {"reach":>7} {"merged":>7}')
    print(hdr)
    print('-' * len(hdr))
    rows = []
    for run in runs:
        merged = (DONE / f'run{run}.root')
        msize = merged.stat().st_size if merged.exists() else -1

        off_ps = partials(COMPLETED / str(run), run)
        if off_ps:
            n, contig, b0, b1, last, state = coverage(off_ps, run)
        elif msize > 0:
            # the partials were cleaned up but the merged file is the product;
            # 224569 is the only run in the campaign in this state
            n, contig, b0, b1, last, state = 0, True, None, None, None, 'MERGED_ONLY'
        else:
            n, contig, b0, b1, last, state = 0, True, None, None, None, 'ABSENT'

        our_state, our_n, our_last, our_prod = 'ABSENT', 0, None, ''
        for base in OURS:
            d = base / str(run) / 'completed' / str(run)
            ps = partials(d, run) if d.is_dir() else []
            if ps:
                our_prod = base.name
                our_n, _, _, ob1, our_last, our_state = coverage(ps, run)
                if b1 is None:
                    b1 = ob1
                break

        rows.append(dict(run=run, off_state=state, off_parts=n,
                         off_contiguous=contig, run_first=b0, run_last=b1,
                         off_reach=last, ours_prod=our_prod,
                         ours_state=our_state, ours_parts=our_n,
                         ours_reach=our_last, merged_bytes=msize))
        print(f'{run:>7} {state:>9} {n:6d} {f"{b0}-{b1}" if b0 else "?":>16} '
              f'{last if last is not None else "-":>7} {our_prod or "-":>9} '
              f'{our_n:6d} {our_last if our_last is not None else "-":>7} '
              f'{"yes" if msize > 0 else ("EMPTY" if msize == 0 else "no"):>7}',
              flush=True)

    if out_csv:
        with open(out_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {out_csv}')
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=1))
        print(f'wrote {out_json}')

    from collections import Counter
    print('\nofficial:', Counter(r['off_state'] for r in rows))
    print('ours    :', Counter(r['ours_state'] for r in rows))
    orphan = [r['run'] for r in rows
              if r['off_state'] not in ('COVERED', 'MERGED_ONLY')
              and r['ours_state'] != 'COVERED']
    print(f'\ncovered by NEITHER ({len(orphan)}): {orphan}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
