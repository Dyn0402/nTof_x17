#!/usr/bin/env python3
"""make_gate_dag.py <code_dir> <eos_analysis_root> [--arms ...] -- fleet gate of the prototype switches (RERUN_PLAN §12 B2/B4).

For each golden row and each arm: RECO (wft.cli reco --matched-only --limit 8000 with the promoted calib_bundle and
the arm's WFT_* environment) -> BUDGET (rerun/effbudget/budget.py) and ANGLES (mx_june_wft/03_angles.py) on the
production alignment (<row>/wft/prod_alignment/alignment.json, copied from final_physmask).  Writes gate.dag + gate.sub.
"""
import argparse

GOLDEN = [('det3', 'sat_det3'), ('det2', 'o22_long_det2'), ('det4', 'g_det4'), ('det6', 'g_det6_long'), ('det7', 'g_det7_long')]
ARMS = {
    'prod': '',
    'band': 'WFT_DROP_FLAT_HITS=1',
    'headon': 'WFT_HEADON_SIG=20 WFT_HEADON_RESCUE=1',
    'twotrack': 'WFT_TWO_TRACK_FIT=1',
    'all': 'WFT_DROP_FLAT_HITS=1 WFT_HEADON_SIG=20 WFT_HEADON_RESCUE=1 WFT_TWO_TRACK_FIT=1',
}
SH = '''#!/bin/bash
# run_gate.sh <code_dir> <envspec> <stage_job.py args...>   envspec: comma-separated K=V list, or '-'
CODE=$1; ENVSPEC=$2; shift 2
if [ "$ENVSPEC" != "-" ]; then for kv in ${ENVSPEC//,/ }; do export "$kv"; done; fi
exec $HOME/x17rerun/stage_job.sh $CODE "$@"
'''
SUB = '''# written by rerun/model_audit/make_gate_dag.py
executable     = $ENV(HOME)/x17rerun/audit/run_gate.sh
arguments      = $(code) $(env) $(args)
output         = log/$(nodename).out
error          = log/$(nodename).err
log            = log/gate.log
request_cpus   = $(cpus:1)
request_memory = $(mem:8000)
+JobFlavour    = "workday"
should_transfer_files = NO
queue
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code'); ap.add_argument('root')
    ap.add_argument('--arms', nargs='*', default=list(ARMS))
    ap.add_argument('--dets', nargs='*', default=[d for d, _ in GOLDEN])
    ap.add_argument('--limit', type=int, default=8000); ap.add_argument('--skip', nargs='*', default=[], help='det_arm already done')
    ap.add_argument('--tag', default='gate')
    a = ap.parse_args()
    L = []

    def node(name, env, args, cpus, mem, parents=()):
        L.append(f'JOB {name} gate.sub')
        L.append(f'VARS {name} nodename="{name}" code="{a.code}" env="{env}" cpus="{cpus}" mem="{mem}" args="{args}"')
        L.append(f'RETRY {name} 1')
        for p in parents:
            L.append(f'PARENT {p} CHILD {name}')

    for det, key in GOLDEN:
        if det not in a.dets:
            continue
        for arm in a.arms:
            if f'{det}_{arm}' in a.skip:
                continue
            env = ARMS[arm].replace(' ', ',') or '-'
            t = f'{{out}}/wft/{arm}/events.parquet'
            al = '{out}/wft/prod_alignment/alignment.json'
            pre = f'{key} --eos-analysis {a.root}'
            node(f'RECO_{det}_{arm}', env, f'{pre} --module wft.cli -- reco {{key}} --bundle {{out}}/wft/calib_bundle '
                 f'--out {t} --matched-only --limit {a.limit} --jobs 16', 16, 24000)
            node(f'BUD_{det}_{arm}', env, f'{pre} --parts run_config,m3,hits --script rerun/effbudget/budget.py -- {det} '
                 f'--table {t} --alignment {al} --out {{out}}/wft/{arm}/budget', 1, 8000, [f'RECO_{det}_{arm}'])
            node(f'ANG_{det}_{arm}', env, f'{pre} --parts run_config,m3,hits --script mx_june_wft/03_angles.py -- '
                 f'--table {t} --alignment {al} --out {{out}}/wft/{arm}/angles', 1, 8000, [f'RECO_{det}_{arm}'])
    open(f'{a.tag}.dag', 'w').write('\n'.join(L) + '\n')
    open('gate.sub', 'w').write(SUB); open('run_gate.sh', 'w').write(SH); import os; os.chmod('run_gate.sh', 0o755)
    print(f'wrote {a.tag}.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
