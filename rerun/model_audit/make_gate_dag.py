#!/usr/bin/env python3
"""make_gate_dag.py <code_dir> <eos_analysis_root> [--arms ...] -- fleet gate of the prototype switches (RERUN_PLAN §12 B2/B4).

For each golden row and each arm: RECO (wft.cli reco --matched-only --limit 8000 with the promoted calib_bundle and
the arm's WFT_* environment) -> BUDGET (rerun/effbudget/budget.py) and ANGLES (mx_june_wft/03_angles.py) on the
production alignment (<row>/wft/prod_alignment/alignment.json, copied from final_physmask).  Writes gate.dag + gate.sub.
"""
import argparse

GOLDEN = [('det3', 'sat_det3'), ('det2', 'o22_long_det2'), ('det4', 'g_det4'), ('det6', 'g_det6_long'), ('det7', 'g_det7_long')]
#: arm -> environment, or (environment, bundle dir name under <row>/wft).  Since 2026-10-27 the switches are part of
#: wft (wft/diag.py): 'unified' is the algorithm as it now stands, on bundles carrying the measured dt_xy / xy_pairing
#: (calib_bundle_xyp, ws_unify/make_xyp_bundles.sh); 'legacy' turns every new piece off through the retired switch
#: names and must reproduce the 2026-10-27 gate's 'prod' arm (a check of the refactor itself).  The older arms are the
#: 2026-10-27 night's; under the new code they would read as studies on top of the unified algorithm.
LEGACY = ('WFT_SAT_SOLVE=0 WFT_RC_T0_INTERP=0 WFT_TWO_TRACK_FIT=0 WFT_HEADON_SIG=0 WFT_HEADON_RESCUE=0 '
          'WFT_DROP_FLAT_HITS=0')
ARMS = {
    'unified': ('', 'calib_bundle_xyp'),
    'legacy': (LEGACY, 'calib_bundle'),
    'prod': '',
    'band': 'WFT_DROP_FLAT_HITS=1',
    'headon': 'WFT_HEADON_SIG=20 WFT_HEADON_RESCUE=1',
    'twotrack': 'WFT_TWO_TRACK_FIT=1',
    'all': 'WFT_DROP_FLAT_HITS=1 WFT_HEADON_SIG=20 WFT_HEADON_RESCUE=1 WFT_TWO_TRACK_FIT=1',
    # 2026-10-29 saturation / error-model gate (rerun/calib/saturation/ERROR_MODEL.md), on the pinned FE bundles of the
    # LM gate: the current algorithm, the recipe (cap 1200 + per-strip amplitude nuisance + plain charge re-solve), and
    # the cap alone with the plain re-solve.  A WFT_DIAG=... spec is exported whole (its overrides are comma-separated).
    'sat_base': ('', 'calib_bundle_fe'),
    'sat_cand': ('WFT_DIAG=model.SAT_CAP=1200,model.STRIP_TAU=0.05,model.QSOLVE_PLAIN=true', 'calib_bundle_fe'),
    'sat_cap': ('WFT_DIAG=model.SAT_CAP=1200,model.QSOLVE_PLAIN=true', 'calib_bundle_fe'),
}
SH = '''#!/bin/bash
# run_gate.sh <code_dir> <envspec> <key> <stage_job.py args...>   envspec: comma-separated K=V list, or '-'
# a non-empty envspec is a study arm: stage_job.py refuses wft overrides without --diag
CODE=$1; ENVSPEC=$2; KEY=$3; shift 3
DIAG=""
if [[ "$ENVSPEC" == WFT_DIAG=* ]]; then export "$ENVSPEC"; DIAG=--diag
elif [ "$ENVSPEC" != "-" ]; then for kv in ${ENVSPEC//,/ }; do export "$kv"; done; DIAG=--diag; fi
exec $HOME/x17rerun/stage_job.sh $CODE $KEY $DIAG "$@"
'''
SUB = '''# written by rerun/model_audit/make_gate_dag.py
executable     = {exe}
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
    ap.add_argument('--exe', default='$ENV(HOME)/x17rerun/audit/run_gate.sh', help='run_gate.sh the jobs execute (written next to the dag)')
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
            spec = ARMS[arm]
            env, bname = spec if isinstance(spec, tuple) else (spec, 'calib_bundle')
            env = (env if env.startswith('WFT_DIAG=') else env.replace(' ', ',')) or '-'
            t = f'{{out}}/wft/{arm}/events.parquet'
            al = '{out}/wft/prod_alignment/alignment.json'
            pre = f'{key} --eos-analysis {a.root}'
            node(f'RECO_{det}_{arm}', env, f'{pre} --module wft.cli -- reco {{key}} --bundle {{out}}/wft/{bname} '
                 f'--out {t} --matched-only --limit {a.limit} --jobs 16', 16, 24000)
            node(f'BUD_{det}_{arm}', env, f'{pre} --parts run_config,m3,hits --script rerun/effbudget/budget.py -- {det} '
                 f'--table {t} --alignment {al} --out {{out}}/wft/{arm}/budget', 1, 8000, [f'RECO_{det}_{arm}'])
            node(f'ANG_{det}_{arm}', env, f'{pre} --parts run_config,m3,hits --script mx_june_wft/03_angles.py -- '
                 f'--table {t} --alignment {al} --out {{out}}/wft/{arm}/angles', 1, 8000, [f'RECO_{det}_{arm}'])
    open(f'{a.tag}.dag', 'w').write('\n'.join(L) + '\n')
    open('gate.sub', 'w').write(SUB.format(exe=a.exe)); open('run_gate.sh', 'w').write(SH); import os; os.chmod('run_gate.sh', 0o755)
    print(f'wrote {a.tag}.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
