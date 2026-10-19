#!/usr/bin/env python3
"""Publish recovered slim products to the DREAM tree on EOS -- ADD ONLY.

    python3 publish_missing.py [--dry-run] TREE [TREE ...]

TREEs are condor output trees (each holding runs/<run>/<subrun>/ntof_hits/),
in PRIORITY order: the first tree that has a segment wins. Typical call, newest
vintage first:

    python3 publish_missing.py ~/x17slim_wide/out_* ~/x17slim_refactor/out_* \
                               ~/x17slim_fixed/out_*

Rules, per (run, subrun, n_TOF run) segment:

  * EOS has no ntof_hits_<run>_<subrun>_<ntof>.root  -> copy the whole
    ntof_hits/ directory content for that segment (root + sidecars).
  * EOS has the root, SAME SIZE as the source          -> same vintage: copy
    only the sidecars EOS lacks (clock_qa.json, burst_map.json,
    burst_census.json, calibration.json ...) so products born before the
    2026-08-13 contract become harvestable in place.
  * EOS has the root, DIFFERENT size                   -> a different
    vintage is already published: touch NOTHING, report it.

Never deletes, never overwrites a root file -- EXCEPT the segments named with
--replace run/subrun/ntof (2026-08-16: products deliberately re-made with
burst_fixes.json overrides), whose root AND sidecars are overwritten from the
first tree that has them; the previous sizes are printed. Run on lxplus with a
token (writes to EOS need it). Verifies every copied root by size afterwards.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

DEST = Path(os.environ.get('X17_EOS_JULY',
                           '/eos/experiment/ntof/data/x17/july_beam'))


def segments(tree: Path):
    """{(run, subrun, ntof): (ntof_hits dir, root path)} in one tree."""
    out = {}
    for root in (tree / 'runs').rglob('ntof_hits_*.root'):
        stem = root.stem                       # ntof_hits_run_116_stat090_0027_224640
        ntof = stem.split('_')[-1]
        d = root.parent
        subrun, run = d.parent.name, d.parent.parent.name
        out[(run, subrun, ntof)] = (d, root)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('trees', nargs='+', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--refresh-sidecars', action='store_true',
                    help='where EOS already has the identical root (same '
                         'size), overwrite its sidecars with the source ones '
                         '-- for re-analysed clock_qa.json records. Roots '
                         'are still never touched.')
    ap.add_argument('--replace', nargs='*', default=[],
                    help='run/subrun/ntof segments whose EOS product may be '
                         'OVERWRITTEN (root and sidecars) by the source')
    a = ap.parse_args()
    replace = {tuple(r.split('/')) for r in a.replace}

    seen = {}
    for t in a.trees:
        if not (t / 'runs').is_dir():
            print(f'skip {t}: no runs/')
            continue
        for k, v in segments(t).items():
            seen.setdefault(k, v)             # first tree wins
    print(f'{len(seen)} distinct segment(s) across {len(a.trees)} tree(s)')

    added, sidecars, same, differ, verify_fail = [], [], [], [], []
    replaced = []
    for (run, subrun, ntof), (d, root) in sorted(seen.items()):
        dd = DEST / 'runs' / run / subrun / 'ntof_hits'
        droot = dd / root.name
        if (run, subrun, ntof) in replace:
            old = droot.stat().st_size if droot.exists() else None
            replaced.append((run, subrun, ntof, old, root.stat().st_size))
            if not a.dry_run:
                dd.mkdir(parents=True, exist_ok=True)
                for f in d.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dd / f.name)
                if droot.stat().st_size != root.stat().st_size:
                    verify_fail.append((run, subrun, ntof))
            continue
        if not droot.exists():
            added.append((run, subrun, ntof))
            if not a.dry_run:
                dd.mkdir(parents=True, exist_ok=True)
                for f in d.iterdir():
                    if f.is_file():
                        # sidecars are per ntof_hits dir; a sub-run that
                        # straddles two n_TOF runs shares the dir, and the
                        # second job's clock_qa.json would clobber the first's
                        # -- so sidecars are copied only when absent, roots
                        # always (their names carry the n_TOF run)
                        tgt = dd / f.name
                        if f.suffix == '.root' or not tgt.exists():
                            shutil.copy2(f, tgt)
                if droot.stat().st_size != root.stat().st_size:
                    verify_fail.append((run, subrun, ntof))
        elif droot.stat().st_size == root.stat().st_size:
            missing = [f for f in d.iterdir()
                       if f.is_file() and f.suffix != '.root'
                       and (a.refresh_sidecars or not (dd / f.name).exists())]
            if missing:
                sidecars.append((run, subrun, ntof, [f.name for f in missing]))
                if not a.dry_run:
                    for f in missing:
                        shutil.copy2(f, dd / f.name)
            else:
                same.append((run, subrun, ntof))
        else:
            differ.append((run, subrun, ntof, root.stat().st_size,
                           droot.stat().st_size))

    tag = 'WOULD ADD' if a.dry_run else 'ADDED'
    print(f'\n{tag} {len(added)} product(s):')
    for k in added:
        print('   ', '/'.join(k))
    print(f'\nsidecars {"refreshed" if a.refresh_sidecars else "completed"} '
          f'on {len(sidecars)} existing product(s)')
    for k in sidecars[:20]:
        print('   ', '/'.join(k[:3]), k[3])
    print(f'\n{"WOULD REPLACE" if a.dry_run else "REPLACED"} {len(replaced)}:')
    for k in replaced:
        print('   ', '/'.join(k[:3]), f'eos {k[3]} B -> src {k[4]} B')
    print(f'\nalready published, identical: {len(same)}')
    print(f'DIFFERENT vintage on EOS, untouched: {len(differ)}')
    for k in differ:
        print('   ', '/'.join(k[:3]), f'src {k[3]} B vs eos {k[4]} B')
    if verify_fail:
        print(f'\n!! SIZE MISMATCH after copy on {len(verify_fail)}:')
        for k in verify_fail:
            print('   ', '/'.join(k))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
