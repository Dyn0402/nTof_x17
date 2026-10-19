#!/usr/bin/env python3
"""
23_promote_r06.py -- park the frozen product tree and promote the r06 arm.

Makes one detector's wft/ directory entirely self-consistent on the c2-slaved
calibration: the reco table, its candidates sidecar and meta, the alignment and
the angles all come from `calib_bundle_r06`, and the stages that follow
(efficiency, maps, digest) are then re-run in place against them.

RULES IT ENFORCES, both learned the hard way (FREEZE_MPGD26 postmortem):
  * the park is written ONCE. If the park directory already exists the script
    refuses -- the first park is the true pre-promotion state and a second pass
    must never land on top of it.
  * nothing is deleted. The frozen tree is COPIED into the park before the
    promotion touches anything, and the promotion itself moves the r06 files
    rather than overwriting the originals in place.

    ../.venv/bin/python mx_june_wft/23_promote_r06.py sat_det3 [--dry-run]
    ../.venv/bin/python mx_june_wft/23_promote_r06.py sat_det3 --revert
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

PARK = 'pre_r06_backup_20260819'
FILES = ['events.parquet', 'events.candidates.parquet', 'events.meta.json']
DIRS = ['alignment', 'angles', 'efficiency', 'maps']
# what the r06 arm already produced, and what it maps onto
ARM_FILES = {'events_r06.parquet': 'events.parquet',
             'events_r06.candidates.parquet': 'events.candidates.parquet',
             'events_r06.meta.json': 'events.meta.json'}
ARM_DIRS = {'alignment_r06': 'alignment', 'angles_r06': 'angles'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--revert', action='store_true',
                    help='restore the park over the live tree')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    W = os.path.join(get_config(args.run_key).OUT_BASE, 'wft')
    park = os.path.join(W, PARK)

    if args.revert:
        if not os.path.isdir(park):
            sys.exit(f'nothing to revert: {park} does not exist')
        for n in FILES:
            src = os.path.join(park, n)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(W, n))
                print(f'restored {n}')
        for n in DIRS:
            src = os.path.join(park, n)
            if os.path.isdir(src):
                shutil.rmtree(os.path.join(W, n), ignore_errors=True)
                shutil.copytree(src, os.path.join(W, n))
                print(f'restored {n}/')
        print(f'reverted from {park} (the park is left in place)')
        return

    if os.path.exists(park):
        sys.exit(f'REFUSING: {park} already exists. The first park is the true '
                 'pre-promotion state -- a second pass must not overwrite it. '
                 'Use --revert, or move the park aside deliberately.')
    missing = [n for n in ARM_FILES if not os.path.exists(os.path.join(W, n))] \
        + [n for n in ARM_DIRS if not os.path.isdir(os.path.join(W, n))]
    if missing:
        sys.exit('REFUSING: the r06 arm is incomplete, missing ' +
                 ', '.join(missing) + ' -- run the reco and 21_r06_gate.sh first')

    meta = json.load(open(os.path.join(W, 'events_r06.meta.json')))
    used = os.path.basename(meta.get('calibration', '?'))
    if 'r06' not in used:
        sys.exit(f'REFUSING: events_r06.meta.json says the table was built with '
                 f'{used!r}, not an r06 bundle')
    print(f'{args.run_key}: {W}\n  arm built with {used}, '
          f'{meta.get("n_events", "?"):,} events')

    if args.dry_run:
        print('  would park:', ', '.join(FILES + [d + "/" for d in DIRS]))
        print('  would promote:', ', '.join(f'{a} -> {b}' for a, b in
                                            {**ARM_FILES, **ARM_DIRS}.items()))
        return

    os.makedirs(park)
    for n in FILES:
        src = os.path.join(W, n)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(park, n))
    for n in DIRS:
        src = os.path.join(W, n)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(park, n))
    with open(os.path.join(park, 'PARK_NOTE.md'), 'w') as f:
        f.write(
            f'# Pre-r06 state of {args.run_key}\n\n'
            'These are the FROZEN MPGD26 products (the 2026-08-12 condor\n'
            'campaign output and everything derived from it) as they stood\n'
            'before the r06 promotion on 2026-08-19. The live tree beside\n'
            'this directory is now the c2-slaved calibration.\n\n'
            'Unlike the 7-31 golden parquets lost during the campaign\n'
            'promotion, these ARE regenerable: frozen bundle + frozen code +\n'
            '`wft reco <key> --bundle <W>/calib_bundle_lp* --matched-only`.\n'
            'They are parked anyway, because a park costs 2 MB and a\n'
            'regeneration costs a reasoning chain about which code produced\n'
            'what.\n\n'
            'Restore with `23_promote_r06.py <key> --revert`.\n\n'
            'Record: `mx_june_wft/R06_GATE_2026-08-19.md`.\n')
    print(f'  parked -> {PARK}/')

    for a, b in ARM_FILES.items():
        shutil.move(os.path.join(W, a), os.path.join(W, b))
        print(f'  promoted {a} -> {b}')
    for a, b in ARM_DIRS.items():
        shutil.rmtree(os.path.join(W, b), ignore_errors=True)
        shutil.move(os.path.join(W, a), os.path.join(W, b))
        print(f'  promoted {a}/ -> {b}/')
    print('  NOW RE-RUN the stages that were not part of the arm: '
          '02_efficiency (x3), 04_maps, digest')


if __name__ == '__main__':
    main()
