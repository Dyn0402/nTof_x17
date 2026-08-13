#!/usr/bin/env python3
"""
seed_scan_alignment.py — give a scan subrun the alignment its v-refit needs.

Tier-B rows run `wft.calibrate --vrefit`, which builds its reference corridor
from the hits-chain event cache AND `alignment_tpc_veto<V>/alignment.json`.
The six 6-27 drift-scan points have the cache but never had an alignment, so
their v-refit dies on a worker (and would die here too) with

    need the hits-chain alignment + event cache for the reference geometry

The scan points do NOT get a fresh alignment. Per the registry's own note on
`sat_det3` ("the long run seeds the alignment for the scans"), and following
10_hv_scan_efficiency.py / 43_drift_window_truncation.py, the long run supplies
z, rotation and handedness, and only the TRANSLATION is refitted per point:
re-fitting z at 100 V drift would fold that point's own (unknown, refitted-
downstream) drift velocity into the geometry, which is exactly backwards for a
v measurement.

    ../../.venv/bin/python mx_june_wft/condor/seed_scan_alignment.py \
        --rows 61,62,63,64,65,66 --seed-key sat_det3 [--apply]

Dry-run by default. An existing alignment.json is never overwritten without
--force: on these keys its absence is the whole point, and a surprise
overwrite would silently redefine a geometry someone else measured.
"""
import argparse
import csv
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import (_Config, get_config, setup_paths,      # noqa: E402
                       M3_CHI2_CUT, M3_MIN_NCLUS)
setup_paths()
import cosmic_micro_tpc_analysis as cm                        # noqa: E402
from M3RefTracking import M3RefTracking                       # noqa: E402

BENCH = '/home/dylan/x17/cosmic_bench'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest',
                    default=os.path.join(HERE, 'campaign_manifest.csv'))
    ap.add_argument('--rows', required=True, help='comma-separated row indices')
    ap.add_argument('--seed-key', default=None,
                    help='qa_config key whose alignment seeds z/rotation '
                         '(e.g. sat_det3). Default: the row run\'s long_run.')
    ap.add_argument('--seed-json', default=None,
                    help='explicit alignment.json, overrides --seed-key')
    ap.add_argument('--veto', type=int, default=50)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    if args.seed_json:
        seed_path = args.seed_json
    elif args.seed_key:
        seed_path = os.path.join(get_config(args.seed_key).OUT_BASE,
                                 f'alignment_tpc_veto{args.veto}',
                                 'alignment.json')
    else:
        sys.exit('need --seed-key or --seed-json')
    if not os.path.exists(seed_path):
        sys.exit(f'FATAL: seed alignment not found: {seed_path}')
    seed = cm.load_alignment(seed_path)
    print(f'seed {seed_path}\n     z_x={seed.z_x:.1f} z_y={seed.z_y:.1f} '
          f'theta={seed.theta_deg:.3f} ref_x_sign={seed.ref_x_sign:+.0f} '
          f'offsets=({seed.x_offset:.2f}, {seed.y_offset:.2f})\n')

    n_ok = 0
    for i in [int(x) for x in args.rows.split(',')]:
        row = rows[i]
        cfg = _Config(row['key'], row['run'], row['subrun'],
                      feus=[int(row['feu_x']), int(row['feu_y'])],
                      det_z=float(row['det_z']), det_name=row['det'],
                      base_path=os.path.join(BENCH, row['tree']) + '/')
        tag = f'alignment_tpc_veto{args.veto}'
        out = os.path.join(cfg.OUT_BASE, tag, 'alignment.json')
        cache = os.path.join(cfg.out_dir('cache'),
                             f'event_results_veto{args.veto}.pkl')
        label = f'row {i:3d} {row["subrun"][:42]:42s}'
        if not os.path.exists(cache):
            print(f'{label}  SKIP: no event cache ({cache})')
            continue
        if os.path.exists(out) and not args.force:
            print(f'{label}  SKIP: alignment already exists (use --force)')
            continue

        results = pickle.load(open(cache, 'rb'))
        rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                             min_nclus=M3_MIN_NCLUS)
        params = cm.translation_alignment(results, rays, seed)
        n_both = sum(1 for r in results if getattr(r, 'has_both', False))
        d = (params.x_offset - seed.x_offset, params.y_offset - seed.y_offset)
        print(f'{label}  {n_both:5,} X+Y  offsets '
              f'({params.x_offset:8.3f}, {params.y_offset:8.3f})  '
              f'shift vs seed ({d[0]:+.3f}, {d[1]:+.3f}) mm'
              f'{"" if args.apply else "   [dry run]"}')
        if abs(d[0]) > 5 or abs(d[1]) > 5:
            # The scan points are minutes apart on an unmoved detector; a
            # centimetre-scale shift means the seed does not belong to this
            # geometry, not that the detector moved.
            print('        WARNING: >5 mm from the seed — check the seed key')
        if args.apply:
            cm.save_alignment(params, out)
            with open(os.path.join(cfg.OUT_BASE, tag, 'seed.json'), 'w') as f:
                json.dump(dict(seeded_from=seed_path, method='translation only',
                               veto=args.veto, n_events_xy=n_both,
                               note='z/theta/ref_x_sign inherited from the seed; '
                                    'only x_offset/y_offset refitted'), f, indent=1)
        n_ok += 1
    print(f'\n{n_ok} rows {"written" if args.apply else "to write"}')
    if not args.apply:
        print('DRY RUN — nothing written. Re-run with --apply.')


if __name__ == '__main__':
    main()
