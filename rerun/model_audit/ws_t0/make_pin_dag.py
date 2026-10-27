#!/usr/bin/env python3
"""make_pin_dag.py <code_dir> -- the held-out arm 'lat' on condor: each golden row's 1320 held-out events through
heldout_arm.py --t0-fix with the latency bundle (ws_t0/bundles/<det>: t0_abs = model-free trigger latency, pinned),
WFT_RC_T0_INTERP=1, K = 24 -- directly comparable with the audit's t0i arm (heldout/<det>_t0i.npz, same events).
Uses the audit's wrapper ~/x17rerun/audit/run_heldout.sh.  Writes pin.dag + pin.sub into cwd.
"""
import sys

A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'
B = '/eos/user/d/dneff/x17rerun/model_audit/ws_t0/bundles'
OUT = '/eos/user/d/dneff/x17rerun/model_audit/ws_t0/heldout'
ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4',
        'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
SUB = '''executable     = $ENV(HOME)/x17rerun/audit/run_heldout.sh
arguments      = $(code) $(env) $(args)
output         = log/$(nodename).out
error          = log/$(nodename).err
log            = log/pin.log
request_cpus   = 8
request_memory = 16000
+JobFlavour    = "workday"
should_transfer_files = NO
queue
'''


def main():
    code = sys.argv[1]; L = []
    for det, row in ROWS.items():
        name = f'PIN_{det}'
        args = (f'--cache {A}/{row}/wft/w12/it1/cache.pkl --bundle {B}/{det} --out {OUT}/{det}_lat.npz '
                f'--start 180 --n 1320 --jobs 8 --k 24 --t0-fix')
        L += [f'JOB {name} pin.sub', f'VARS {name} nodename="{name}" code="{code}" env="WFT_RC_T0_INTERP=1" args="{args}"',
              f'RETRY {name} 1']
    open('pin.dag', 'w').write('\n'.join(L) + '\n'); open('pin.sub', 'w').write(SUB)
    print('wrote pin.dag: 5 nodes')


if __name__ == '__main__':
    main()
