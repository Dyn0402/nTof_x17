#!/usr/bin/env python3
"""make_ws_dag.py <code_dir> <spec-file> [--tag T] — condor DAG for the width–tilt workstream (ws_widthtilt).

Each non-empty line of the spec file is one node:  <name> <script.py> <args...>   where the tokens {CACHE:<det>} and
{BUNDLE:<det>} expand to the golden row's W12 it1 cache and promoted calib_bundle on EOS (as in make_heldout_dag.py),
and {OUT} to /eos/user/d/dneff/x17rerun/model_audit/ws_widthtilt.  Writes <tag>.dag, ws.sub and run_ws.sh into the
current directory (run it in ~/x17rerun/ws_widthtilt on lxplus, then condor_submit_dag <tag>.dag).
"""
import argparse, os, re

A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'
OUT = '/eos/user/d/dneff/x17rerun/model_audit/ws_widthtilt'
ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4',
        'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
SH = '''#!/bin/bash
# run_ws.sh <code_dir> <script> <args...>
CODE=$1; SCRIPT=$2; shift 2
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE MPLBACKEND=Agg WFT_COL_GUARD=0.01
cd $_CONDOR_SCRATCH_DIR 2>/dev/null || cd /tmp
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/model_audit/ws_widthtilt/$SCRIPT "$@"
'''
SUB = '''executable     = $ENV(HOME)/x17rerun/ws_widthtilt/run_ws.sh
arguments      = $(code) $(script) $(args)
output         = log/$(nodename).out
error          = log/$(nodename).err
log            = log/ws.log
request_cpus   = 8
request_memory = 12000
+JobFlavour    = "workday"
should_transfer_files = NO
queue
'''


def expand(s):
    s = re.sub(r'\{CACHE:(\w+)\}', lambda m: f'{A}/{ROWS[m.group(1)]}/wft/w12/it1/cache.pkl', s)
    s = re.sub(r'\{BUNDLE:(\w+)\}', lambda m: f'{A}/{ROWS[m.group(1)]}/wft/calib_bundle', s)
    return s.replace('{OUT}', OUT)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('code'); ap.add_argument('spec'); ap.add_argument('--tag', default='ws')
    a = ap.parse_args(); L = []
    for line in open(a.spec):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, script, *args = line.split()
        L += [f'JOB {name} ws.sub', f'VARS {name} nodename="{name}" code="{a.code}" script="{script}" args="{expand(" ".join(args))}"',
              f'RETRY {name} 1']
    open(f'{a.tag}.dag', 'w').write('\n'.join(L) + '\n'); open('ws.sub', 'w').write(SUB); open('run_ws.sh', 'w').write(SH)
    os.chmod('run_ws.sh', 0o755)
    print(f'wrote {a.tag}.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
