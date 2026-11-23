#!/usr/bin/env python3
"""build_incl.py <key> <mirror_root> <work> -- the S4 inclined-track and head-on caches of one det3 row, built on the
desktop from an EOS mirror (mirror.py) exactly as rerun/calib/s4/s4_stacks.py builds them in a condor job:
rerun/inputs.py stages the rr1 decode (decoded_root_fa80c06) + M3 v2 rays into <work>/bench, the reference is the M3
rays through the PASS-1 alignment, and s4_stacks.build() reads the windows.  -> <work>/out/<key>/{incl,headon}.pkl

Keys: sat_det3 (Run 1, 6-27) and g_det3_wknd (Run 2, 6-28 p2 overnight).  sat_det3 also exists as S4's own cache, so
rebuilding it checks this staging against the pass's.
"""
import os
import pickle
import shutil
import sys

key, mirror, work = sys.argv[1], sys.argv[2], sys.argv[3]
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ['X17_BENCH_ROOT'] = os.path.join(work, 'bench')
os.environ['X17_ANALYSIS_ROOT'] = os.path.join(work, 'Analysis')
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun'), os.path.join(REPO, 'rerun', 'calib', 's4')]
import inputs  # noqa: E402

inputs.AUDIT_DIR = mirror + inputs.AUDIT_DIR
row = dict(inputs.row_for(key))
row['eos_run_dir'] = mirror + row['eos_run_dir']
rec = inputs.stage(row, os.environ['X17_BENCH_ROOT'], 'rr1', ('run_config', 'm3', 'decoded'))
print('staged', {k: len(v) for k, v in rec['files'].items()}, flush=True)

import s4_stacks  # noqa: E402  (sets sys.path for cloud_basics, qa_config, cosmic_bench_analysis)
from qa_config import get_config, setup_paths  # noqa: E402
import cosmic_micro_tpc_analysis as cm  # noqa: E402
setup_paths()
cfg = get_config(key)
ali = (f"{mirror}/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/{row['run']}/{row['subrun']}/"
       f"{row['det']}/wft/alignment/alignment.json")
best = cm.load_alignment(ali)
head, incl = s4_stacks.build(cfg, best)
out = os.path.join(work, 'out', key)
os.makedirs(out, exist_ok=True)
for name, ev in (('headon', head), ('incl', incl)):
    pickle.dump(ev, open(os.path.join(out, f'{name}.pkl'), 'wb'), protocol=4)
print(f'{key}: {len(head)} head-on, {len(incl)} inclined events -> {out}', flush=True)
# the staged waveforms are a copy of the mirror: drop them
shutil.rmtree(os.path.join(work, 'bench'), ignore_errors=True)
