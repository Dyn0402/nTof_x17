#!/usr/bin/env python3
"""fetch.py -- copy the inputs of the M3 per-track pointing study from EOS (read-only there).

Run with the XRootD venv (~/.venvs/xrd/bin/python).  For every row in ROWS: the pass-1 alignment.json and the
production M3 rays (m3_tracking_root_v2, the files the pass used).  Output: $WORK_DATA/nTof_x17/m3_pointing/inputs/
<run>/<subrun>/{<det>/alignment.json, m3_tracking_root_v2/*.root}.  Files already present with the right size are kept.
"""
import os
import sys
from XRootD import client

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rows import ROWS, INPUTS  # noqa: E402

URL = 'root://eospublic.cern.ch'
P1 = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
EXCL = {('mx17_det3_p2_det1_overnight_6-27-26', 'long_run_p2_det1_sanity_check'): '260628_01H29'}
fs = client.FileSystem(URL)


def copy(src, dst):
    st, info = fs.stat(src)
    if not st.ok:
        raise OSError(f'{src}: {st.message}')
    if os.path.exists(dst) and os.path.getsize(dst) == info.size:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = client.CopyProcess()
    p.add_job(URL + '/' + src, dst, force=True, checksummode='end', checksumtype='adler32')
    p.prepare()
    st, res = p.run()
    if not (st.ok and res[0]['status'].ok):
        raise OSError(f'copy failed {src}: {res[0]["status"].message}')


for r in ROWS:
    rel = f"{r['run']}/{r['subrun']}"
    copy(f"{P1}/{rel}/{r['det']}/wft/alignment/alignment.json", f"{INPUTS}/{rel}/{r['det']}/alignment.json")
    m3d = f"{r['eos_run_dir']}/{r['subrun']}/m3_tracking_root_v2"
    st, ls = fs.dirlist(m3d)
    if not st.ok:
        raise OSError(f'{m3d}: {st.message}')
    names = sorted(e.name for e in ls if e.name.endswith('_rays.root'))
    ex = EXCL.get((r['run'], r['subrun']))
    names = [n for n in names if not (ex and ex in n)]
    for n in names:
        copy(f'{m3d}/{n}', f'{INPUTS}/{rel}/m3_tracking_root_v2/{n}')
    print(rel, r['det'], len(names), 'M3 files', flush=True)
