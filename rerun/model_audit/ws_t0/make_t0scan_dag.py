#!/usr/bin/env python3
"""make_t0scan_dag.py <code_dir> [--dets ...] [--C free,160:460:20] [--n 600] -- t0 latency scan on condor.

One node per (chamber, C): t0_scan.py on the golden row's W12 it1 cache with the promoted calib_bundle, events
[180, 180 + n), K = 30 (an early pinned t0 lengthens the column), WFT_RC_T0_INTERP=1.  Output:
/eos/user/d/dneff/x17rerun/model_audit/ws_t0/<det>_C<C>.npz.  Writes t0scan.dag + t0scan.sub + run_t0scan.sh into cwd.
"""
import argparse, os
import numpy as np

A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'
OUT = '/eos/user/d/dneff/x17rerun/model_audit/ws_t0'
ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4',
        'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
SH = '''#!/bin/bash
# run_t0scan.sh <code_dir> <t0_scan.py args...>
CODE=$1; shift
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE MPLBACKEND=Agg WFT_COL_GUARD=0.01 WFT_RC_T0_INTERP=1
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/model_audit/ws_t0/t0_scan.py "$@"
'''
SUB = '''executable     = $ENV(HOME)/x17rerun/audit_t0/run_t0scan.sh
arguments      = $(code) $(args)
output         = log/$(nodename).out
error          = log/$(nodename).err
log            = log/t0scan.log
request_cpus   = 8
request_memory = 8000
+JobFlavour    = "workday"
should_transfer_files = NO
queue
'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('code')
    ap.add_argument('--dets', nargs='*', default=list(ROWS)); ap.add_argument('--C', default='free,160:460:20')
    ap.add_argument('--n', type=int, default=600); ap.add_argument('--tag', default='t0scan')
    a = ap.parse_args(); L = []
    Cs = []
    for tok in a.C.split(','):
        if tok == 'free' or ':' not in tok:
            Cs.append(tok)
        else:
            lo, hi, st = (float(x) for x in tok.split(':'))
            Cs += [f'{c:g}' for c in np.arange(lo, hi + 1e-9, st)]
    for det in a.dets:
        row = ROWS[det]
        for c in Cs:
            name = f'T0_{det}_{c}'
            args = (f'--cache {A}/{row}/wft/w12/it1/cache.pkl --bundle {A}/{row}/wft/calib_bundle '
                    f'--out {OUT}/{det}_C{c}.npz --C {c} --start 180 --n {a.n} --jobs 8 --k 30')
            L += [f'JOB {name} t0scan.sub', f'VARS {name} nodename="{name}" code="{a.code}" args="{args}"', f'RETRY {name} 1']
    open(f'{a.tag}.dag', 'w').write('\n'.join(L) + '\n'); open('t0scan.sub', 'w').write(SUB); open('run_t0scan.sh', 'w').write(SH)
    os.chmod('run_t0scan.sh', 0o755)
    print(f'wrote {a.tag}.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
