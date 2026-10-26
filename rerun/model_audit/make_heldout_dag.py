#!/usr/bin/env python3
"""make_heldout_dag.py <code_dir> [--arms ...] [--dets ...] -- held-out model arms on condor (model audit 2026-10-27).

One node per (chamber, arm): rerun/model_audit/heldout_arm.py on the golden row's W12 it1 cache
(Analysis_rr1/<row>/wft/w12/it1/cache.pkl) with the promoted calib_bundle, events [180, 1500).  Output:
/eos/user/d/dneff/x17rerun/model_audit/heldout/<det>_<arm>.npz (+ .json).  Writes heldout.dag + heldout.sub + run_heldout.sh.
"""
import argparse, os

A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'
OUT = '/eos/user/d/dneff/x17rerun/model_audit/heldout'
ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4',
        'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
# arm -> (environment, extra heldout_arm.py args)
ARMS = {
    'base': ('', ''),
    't0i': ('WFT_RC_T0_INTERP=1', ''),
    'mf03': ('WFT_MODEL_FRAC=0.03', ''),
    'mf05': ('WFT_MODEL_FRAC=0.05', ''),
    'sat': ('WFT_SAT_SOLVE=1', ''),
    'ts': ('', '--two-sided'),
    't0fix': ('', '--t0-fix'),
    'asymp': ('', '--hyper {\\"rc_asym_x\\":0.13}'),
    'asymn': ('', '--hyper {\\"rc_asym_x\\":-0.13}'),
    'asymp07': ('', '--hyper {\\"rc_asym_x\\":0.07}'),
    's0x20': ('', '--hyper {\\"sigma_p0_x\\":0.20}'),
    'k30': ('', '--k 30'),
    't0i_mf03': ('WFT_RC_T0_INTERP=1 WFT_MODEL_FRAC=0.03', ''),
}
SH = '''#!/bin/bash
# run_heldout.sh <code_dir> <heldout_arm.py args...>
CODE=$1; shift
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE MPLBACKEND=Agg
export WFT_COL_GUARD=${WFT_COL_GUARD:-0.01}
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/model_audit/heldout_arm.py "$@"
'''
SUB = '''executable     = $ENV(HOME)/x17rerun/audit/run_heldout.sh
arguments      = $(code) $(args)
environment    = "$(env)"
output         = log/$(nodename).out
error          = log/$(nodename).err
log            = log/heldout.log
request_cpus   = $(cpus:8)
request_memory = $(mem:16000)
+JobFlavour    = "workday"
should_transfer_files = NO
queue
'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('code')
    ap.add_argument('--arms', nargs='*', default=list(ARMS)); ap.add_argument('--dets', nargs='*', default=list(ROWS))
    ap.add_argument('--n', type=int, default=1320)
    a = ap.parse_args(); L = []
    for det in a.dets:
        row = ROWS[det]
        for arm in a.arms:
            env, extra = ARMS[arm]
            name = f'HO_{det}_{arm}'
            k = '' if '--k' in extra else ' --k 24'
            args = (f'--cache {A}/{row}/wft/w12/it1/cache.pkl --bundle {A}/{row}/wft/calib_bundle --out {OUT}/{det}_{arm}.npz '
                    f'--start 180 --n {a.n} --jobs 8{k} {extra}')
            L += [f'JOB {name} heldout.sub', f'VARS {name} nodename="{name}" code="{a.code}" env="{env}" args="{args}"', f'RETRY {name} 1']
    open('heldout.dag', 'w').write('\n'.join(L) + '\n'); open('heldout.sub', 'w').write(SUB); open('run_heldout.sh', 'w').write(SH)
    os.chmod('run_heldout.sh', 0o755)
    print(f'wrote heldout.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
