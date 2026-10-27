#!/usr/bin/env python3
"""build_fb_tables.py [names...] -- pull the FB consistency rows (make_fb_dag.py) from EOS and build per-event tables.

For each row: wft/fb/{events.parquet, alignment/alignment.json, angles/per_event.npz} from the FB Analysis tree and
the row's m3_tracking_root_v2 (the rr1 M3 generation) from the campaign tree go to
~/x17/cosmic_bench/model_audit/fb/rows/<name>/; ws_v_w0/build_table.build joins reco + M3 + the row's own waveform
alignment into ~/x17/cosmic_bench/model_audit/fb/tables/<name>.parquet (same columns as the golden ws_v_w0 tables).
"""
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'model_audit', 'ws_v_w0'))
sys.path.insert(0, HERE)
from make_fb_dag import ROWS, OUT                       # noqa: E402
from fbcommon import FB                                   # noqa: E402

KNOWN = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}


def manifest():
    return {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'mx_june_wft', 'condor', 'campaign_manifest.csv')))}


def main():
    names = sys.argv[1:] or list(ROWS)
    man = manifest()
    import build_table
    for n in names:
        key, det, _ = ROWS[n]
        r = man[key]
        rel = f"{r['run']}/{r['subrun']}/{r['det']}/wft/fb"
        dst = f'{FB}/rows/{n}'
        os.makedirs(f'{dst}/m3', exist_ok=True)
        src = f'{OUT}/{rel}'
        rc = subprocess.call(['rsync', '-a', f'lxplus:{src}/events.parquet', f'lxplus:{src}/alignment/alignment.json',
                              f'lxplus:{src}/angles/per_event.npz', f'{dst}/'])
        if rc:
            print(n, 'not ready'); continue
        m3 = f"{r['eos_run_dir']}/{r['subrun']}/m3_tracking_root_v2/"
        subprocess.check_call(['rsync', '-a', '--include=*_rays.root', '--exclude=*', f'lxplus:{m3}', f'{dst}/m3/'])
        os.makedirs(f'{FB}/tables', exist_ok=True)
        build_table.build(det, key=KNOWN[det], table=f'{dst}/events.parquet', alignment=f'{dst}/alignment.json',
                          out=f'{FB}/tables/{n}.parquet', m3dir=f'{dst}/m3/')


if __name__ == '__main__':
    main()
