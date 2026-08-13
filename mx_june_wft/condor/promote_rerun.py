#!/usr/bin/env python3
"""
promote_rerun.py — move the 2026-08-13 local re-reco into the Analysis tree.

Separate from collect_results.py --promote on purpose: this generation is not a
condor collection, and its predecessor on disk is the CAMPAIGN product, not the
pre-campaign one. Parking it in `pre_campaign_backup/` would overwrite the
meaning of that directory (and on four golden keys the 8-12 loss already made
those dirs mislabelled), so backups go to a dated dir of their own.

    ../../.venv/bin/python mx_june_wft/condor/promote_rerun.py \
        --src <rerun_out> [--exclude-row 59] [--apply]

Dry-run by default: prints what it would move. Nothing is written without
--apply, and an existing backup is NEVER overwritten (the 8-12 data loss was a
second promote pass parking campaign files on top of the originals).
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil

BENCH = '/home/dylan/x17/cosmic_bench'
FILES = ('events.parquet', 'events.candidates.parquet', 'events.meta.json')
BACKUP = 'pre_rerun_backup_20260813'


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--manifest',
                    default=f'{BENCH}/condor_campaign/campaign_manifest.csv')
    ap.add_argument('--exclude-row', type=int, action='append', default=[],
                    help='row index to reconstruct but NOT promote (row 59 = '
                         'g_det3_wknd, whose local product is the reference)')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    n_ok = n_skip = 0
    for d in sorted(glob.glob(os.path.join(args.src, '*'))):
        jr = os.path.join(d, 'job_row.json')
        if not os.path.exists(jr):
            continue
        row = json.load(open(jr))
        i = next((k for k, r in enumerate(rows)
                  if r['key'] == row['key'] and r['key']), None)
        if i in args.exclude_row:
            print(f'SKIP (excluded) row {i}: {row["key"]}')
            n_skip += 1
            continue
        if not row.get('matched_list'):
            # Every row in this generation must come from a local matched
            # list; anything else was reconstructed on the degraded recipe.
            print(f'SKIP (no matched list) {row["key"]}')
            n_skip += 1
            continue
        meta = json.load(open(os.path.join(d, 'events.meta.json')))
        if not (meta.get('angle_constants') or {}).get('applied'):
            print(f'SKIP (angle constants not applied) {row["key"]}')
            n_skip += 1
            continue
        out = os.path.join(BENCH, 'Analysis', row['run'], row['subrun'],
                           row['det'], 'wft')
        bak = os.path.join(out, BACKUP)
        print(f'{"PROMOTE" if args.apply else "would promote"} {row["key"]}')
        print(f'    -> {out}')
        for fn in FILES:
            src = os.path.join(d, fn)
            if not os.path.exists(src):
                continue
            live = os.path.join(out, fn)
            if not args.apply:
                print(f'       {fn}: {"back up live, " if os.path.exists(live) else ""}copy')
                continue
            os.makedirs(out, exist_ok=True)
            if os.path.exists(live):
                os.makedirs(bak, exist_ok=True)
                if os.path.exists(os.path.join(bak, fn)):
                    os.remove(live)          # backup already taken; do not clobber
                else:
                    shutil.move(live, os.path.join(bak, fn))
            shutil.copy2(src, live)
            assert md5(src) == md5(live), f'copy mismatch {live}'
        n_ok += 1
    print(f'\n{n_ok} rows {"promoted" if args.apply else "to promote"}, '
          f'{n_skip} skipped')
    if not args.apply:
        print('DRY RUN — nothing written. Re-run with --apply.')


if __name__ == '__main__':
    main()
