#!/usr/bin/env python3
"""stage_reco.py -- stage chamber A Run 1 (sat_det3) as pass 1 read it, for the reconstruction of the chi2 > 1 tracks.

Run with ~/.venvs/xrd/bin/python.  Copies (adler32-checked) from EOS june_tests the rr1 generation the pass used:
decoded_root_fa80c06 (FEUs 7, 8), combined_hits_root_rr1_cm0, m3_tracking_root_v2, run_config.json, and the row's
pass-1 calib_bundle, into $WORK/bench/det3/<run>/<subrun>/ under the canonical names (as rerun/inputs.py stages).
"""
import os
import sys
from XRootD import client

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rows import WORK  # noqa: E402

URL = 'root://eospublic.cern.ch'
RUN, SUB = 'mx17_det3_saturday_scan_6-27-26', 'long_run_resist_490V_drift_1000V'
SRC = f'/eos/experiment/ntof/data/x17/cosmic_bench/june_tests/{RUN}'
P1 = f'/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/{RUN}/{SUB}/mx17_3/wft/calib_bundle'
DST = os.path.join(WORK, 'bench', 'det3', RUN)
fs = client.FileSystem(URL)


def copy(src, dst):
    st, info = fs.stat(src)
    if os.path.exists(dst) and os.path.getsize(dst) == info.size:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = client.CopyProcess()
    p.add_job(URL + '/' + src, dst, force=True, checksummode='end', checksumtype='adler32')
    p.prepare(); st, res = p.run()
    if not (st.ok and res[0]['status'].ok):
        raise OSError(f'copy failed {src}: {res[0]["status"].message}')
    print('  ', os.path.basename(dst), info.size // 1000000, 'MB', flush=True)


copy(f'{SRC}/run_config.json', f'{DST}/run_config.json')
for src, canon, keep in (('decoded_root_fa80c06', 'decoded_root', lambda n: n.endswith(('_07.root', '_08.root'))),
                         ('combined_hits_root_rr1_cm0', 'combined_hits_root', lambda n: n.endswith('.root')),
                         ('m3_tracking_root_v2', 'm3_tracking_root_v2', lambda n: n.endswith('_rays.root'))):
    st, ls = fs.dirlist(f'{SRC}/{SUB}/{src}')
    for e in sorted(ls, key=lambda e: e.name):
        if keep(e.name):
            copy(f'{SRC}/{SUB}/{src}/{e.name}', f'{DST}/{SUB}/{canon}/{e.name}')
for n in ('arrays.npz', 'bundle.json', 'w0kw.json'):
    copy(f'{P1}/{n}', os.path.join(WORK, 'reco', 'calib_bundle', n))
print('staged', DST)
