#!/usr/bin/env python3
"""make_fb_dag.py <code_dir> -- consistency rows for the chamber-bow study (FB, 2026-10-27).

The bow is claimed to be a static property of each chamber.  To test that, other runs of each chamber (another subrun
of the same night, another night, the other bench slot, other mesh-HV rows) are reconstructed with the chamber's
promoted golden calib_bundle (as `make_pass_dag.py vrefit --keep-v`: the bow observables -- positions p0 against M3
and the transverse speed w in um/ns -- do not depend on the bundle's v), then aligned with their OWN waveform
alignment (01) and given the M3 reference angles (03).  Nodes per row: RECO -> ALIGN -> ANG.
Output: /eos/user/d/dneff/x17rerun/followups/fb/Analysis/<run>/<subrun>/<det>/wft/fb/{events.parquet,alignment,angles}.
"""
import argparse

A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'
OUT = '/eos/user/d/dneff/x17rerun/followups/fb/Analysis'
GOLD = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4',
        'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
# name -> (registry key, chamber, K)
ROWS = {
    'det2_o22long': ('c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2', 'det2', 24),        # same night, top
    'det2_o17': ('c26__mx17_det1_det2_overnight_6-17-26__longer_run__mx17_2', 'det2', 24),          # 5 days earlier, top, remounted
    'det3_p2': ('c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3', 'det3', 24),  # next night
    'det3_o23': ('c26__mx17_det3_det4_overnight_6-23-26__long_run__mx17_3', 'det3', 24),             # BOTTOM slot, drift 600 V
    'det4_o23': ('c26__mx17_det3_det4_overnight_6-23-26__long_run__mx17_4', 'det4', 24),             # previous night, top
    'det6_longer': ('c26__mx17_det6_det7_overnight_6-26-26__longer_run__mx17_6', 'det6', 24),
    'det7_longer': ('c26__mx17_det6_det7_overnight_6-26-26__longer_run__mx17_7', 'det7', 24),
    'det6_hv505': ('c26__mx17_det6_det7_overnight_6-26-26__resist_det6_505V_det7_480V_drift_700V__mx17_6', 'det6', 24),
    'det7_hv480': ('c26__mx17_det6_det7_overnight_6-26-26__resist_det6_505V_det7_480V_drift_700V__mx17_7', 'det7', 24),
    'det6_hv530': ('c26__mx17_det6_det7_overnight_6-26-26__resist_det6_530V_det7_505V_drift_700V__mx17_6', 'det6', 24),
    'det7_hv505': ('c26__mx17_det6_det7_overnight_6-26-26__resist_det6_530V_det7_505V_drift_700V__mx17_7', 'det7', 24),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('code'); ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=12000)
    a = ap.parse_args(); L = []
    eos = f' --eos-analysis {OUT}'

    def node(name, args, cpus, mem, parents=()):
        L.extend([f'JOB {name} stage1.sub',
                  f'VARS {name} code="{a.code}" args="{args}" cpus="{cpus}" mem="{mem}" node="{name}"',
                  f'RETRY {name} 1'] + [f'PARENT {p} CHILD {name}' for p in parents])
    for n, (k, det, K) in ROWS.items():
        if a.only and n not in a.only:
            continue
        b = f'{A}/{GOLD[det]}/wft/calib_bundle'
        t = '{out}/wft/fb/events.parquet'
        node(f'RECO_{n}', f'{k}{eos} --module wft.cli -- reco {{key}} --bundle {b} --out {t} --matched-only '
             f'--limit {a.limit} --jobs 16 --n-depth-bins {K}', 16, 24000)
        node(f'ALIGN_{n}', f'{k}{eos} --parts run_config,m3,hits --script mx_june_wft/01_alignment.py -- --table {t} '
             f'--out {{out}}/wft/fb/alignment', 1, 8000, [f'RECO_{n}'])
        node(f'ANG_{n}', f'{k}{eos} --parts run_config,m3,hits --script mx_june_wft/03_angles.py -- --table {t} '
             f'--alignment {{out}}/wft/fb/alignment/alignment.json --out {{out}}/wft/fb/angles', 1, 8000, [f'ALIGN_{n}'])
    open('fb.dag', 'w').write('\n'.join(L) + '\n')
    print('wrote fb.dag:', sum(1 for l in L if l.startswith('JOB')), 'nodes')


if __name__ == '__main__':
    main()
