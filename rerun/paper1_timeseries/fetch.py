#!/usr/bin/env python3
"""fetch.py [OUT] -- copy the inputs of timeseries.py (and of rerun/paper1_timing/walk_snr.py) from EOS over XRootD
(root://eospublic.cern.ch, Kerberos; no lxplus).  Needs the XRootD Python bindings (a throwaway venv, e.g.
/tmp/xrdvenv/bin/python -I fetch.py).  OUT defaults to $WORK_DATA/claude-jobs/1ccd2342/timeseries (~60 MB).

  ana/<rel>/wft/{events.parquet, events.meta.json, angles/per_event.npz, calib_bundle/bundle.json}   Analysis_pass1
  budget/<rel>/rays_det3.parquet                                                                   x17rerun/paper1_eff/budget
  angles/<rel>/per_event.npz                                                                       x17rerun/paper1_eff/angles
  timing_rows/<key>.{npz,json}   chamber A rows of 6-27 (HV + drift scans, Run 1) and Run 2        x17rerun/paper1_timing/rows
"""
import os
import sys

from XRootD import client

U = 'root://eospublic.cern.ch/'
E = '/eos/experiment/ntof/data/x17/cosmic_bench/'
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('WORK_DATA', '/media/ucla/work'),
                                                         'claude-jobs', '1ccd2342', 'timeseries')
RELS = ['mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3']


def cp(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = client.CopyProcess(); p.add_job(U + src, dst); p.prepare(); s, _ = p.run()
    print(('ok  ' if s.ok else 'FAIL') + ' ' + src)


def main():
    st, ls = client.FileSystem(U).dirlist(E + 'x17rerun/paper1_timing/rows')
    for e in ls:
        n = e.name
        if (('mx17_det3_saturday_scan_6-27-26' in n or 'p2_det1' in n) and '__mx17_3.' in n) or n.startswith('sat_det3.'):
            cp(E + 'x17rerun/paper1_timing/rows/' + n, os.path.join(OUT, 'timing_rows', n))
    for rel in RELS:
        for f in ('events.parquet', 'events.meta.json', 'angles/per_event.npz', 'calib_bundle/bundle.json'):
            cp(E + 'Analysis_pass1/' + rel + '/wft/' + f, os.path.join(OUT, 'ana', rel, 'wft', f))
        cp(E + 'x17rerun/paper1_eff/budget/' + rel + '/rays_det3.parquet', os.path.join(OUT, 'budget', rel, 'rays_det3.parquet'))
        cp(E + 'x17rerun/paper1_eff/angles/' + rel + '/per_event.npz', os.path.join(OUT, 'angles', rel, 'per_event.npz'))


if __name__ == '__main__':
    main()
