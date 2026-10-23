#!/usr/bin/env python3
"""make_dag.py <iteration> <code_dir> [det ...] -- W12 calibration iteration as a condor DAG (written to it<k>.dag).

Per chamber, iteration k:  RECO (wft.cli reco with bundle_k, matched events, first 8000)  ->  ALIGN (01_alignment on
those positions)  ->  CACHE (calibration cache with ref_mesh from that alignment)  ->  FIT (pinned RC fit, v free,
seeded from bundle_k's arm; its bundle is bundle_{k+1}).
bundle_0 / arm_0 = the W11 profiled fit (w11/w11q_cm0); bundle_k = w12/it{k-1}/fit_pinned/bundle.
"""
import os
import sys

ROWS = {'det2': ('o22_long_det2', 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2'),
        'det3': ('sat_det3', 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3'),
        'det4': ('g_det4', 'mx17_det4_day_6-24-26/long_run/mx17_4'),
        'det6': ('g_det6_long', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6'),
        'det7': ('g_det7_long', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7')}
CM = os.environ.get('W12_CM', '0')
SRC = os.environ.get('W12_SRC', 'w11q_cm0')


def main():
    k = int(sys.argv[1]); code = sys.argv[2]
    dets = sys.argv[3:] or list(ROWS)
    it = f'it{k}'
    bundle = f'{{out}}/wft/w11/{SRC}/bundle_arm' if k == 0 else f'{{out}}/wft/w12/it{k - 1}/fit_pinned/bundle'
    seed = f'w11/{SRC}/arm.json' if k == 0 else f'w12/it{k - 1}/fit_pinned/arm.json'
    L = []
    for d in dets:
        key, rel = ROWS[d]
        jobs = [
            ('RECO', 'stage1.sub', 16, 24000,
             f'{key} --module wft.cli -- reco {{key}} --bundle {bundle} --out {{out}}/wft/w12/{it}/events.parquet '
             f'--matched-only --limit 8000 --jobs 16'),
            ('ALIGN', 'stage1.sub', 1, 8000,
             f'{key} --parts run_config,m3,hits --script mx_june_wft/01_alignment.py -- '
             f'--table {{out}}/wft/w12/{it}/events.parquet --out {{out}}/wft/w12/{it}/alignment'),
            ('CACHE', 'stage1.sub', 2, 16000,
             f'{key} --script rerun/w12/build_cache_wft.py -- --align {{out}}/wft/w12/{it}/alignment/alignment.json '
             f'--out {{out}}/wft/w12/{it}/cache.pkl --cm {CM}'),
            ('FIT', 'fit1.sub', 8, 12000, f'{d} {rel} {it} {seed} pinned 24'),
        ]
        for name, sub, cpus, mem, args in jobs:
            L.append(f'JOB {name}_{d} {sub}')
            L.append(f'VARS {name}_{d} code="{code}" args="{args}" cpus="{cpus}" mem="{mem}" node="{name}_{d}_{it}"')
            L.append(f'RETRY {name}_{d} 1')
        L.append(f'PARENT RECO_{d} CHILD ALIGN_{d}')
        L.append(f'PARENT ALIGN_{d} CHILD CACHE_{d}')
        L.append(f'PARENT CACHE_{d} CHILD FIT_{d}')
    open(f'{it}.dag', 'w').write('\n'.join(L) + '\n')
    print(f'wrote {it}.dag ({len(dets)} chambers)')


if __name__ == '__main__':
    main()
