#!/usr/bin/env python3
"""make_fe_dag.py <code_dir> <eos_analysis_root> --arms fe1 fe1g [--dets ...] [--tag fe_r1] -- the FE fleet gate.

Same nodes as rerun/model_audit/make_gate_dag.py (gate3): per golden row and arm RECO (wft.cli reco --matched-only
--limit 8000, 16 cores) -> BUDGET (effbudget) and ANGLES (03_angles) on the production alignment.  Every FE arm is the
ONE algorithm on a different bundle (no wft overrides, so stage_job.py runs it as production):
  fe1   round 1: fleet sigma0, one-rule dead list, t0 pinned at the latency (w0/kw of the promoted bundle)
  fe1g  fe1 + the 20 ns charge grid
  fe    round 2: fe1 re-finalised (set_w0kw.py --table/--angles on fe1: w0/kw, tan_floor, tc_tol, lq per class)
  feg   round 2 of fe1g
  fe_nb fe with the band censoring off (wft.diag model.BAND_NSIG=0): a study arm, attribution only
<eos_analysis_root>/<row>/wft/calib_bundle_<arm> must exist (upload_fe.sh).  Writes <tag>.dag, gate.sub, run_gate.sh.
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from make_gate_dag import GOLDEN, SUB, SH  # noqa: E402

ARMS = {'fe1': ('', 'calib_bundle_fe1'), 'fe1g': ('', 'calib_bundle_fe1g'), 'fe': ('', 'calib_bundle_fe'),
        'feg': ('', 'calib_bundle_feg'), 'fe_nb': ('WFT_DIAG=model.BAND_NSIG=0', 'calib_bundle_fe'),
        'fe_nobow': ('', 'calib_bundle_fe_nobow')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code'); ap.add_argument('root')
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--dets', nargs='*', default=[d for d, _ in GOLDEN])
    ap.add_argument('--limit', type=int, default=8000)
    ap.add_argument('--tag', default='fe')
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
            env, bname = ARMS[arm]
            env = env.replace(' ', ',') or '-'
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
    open('gate.sub', 'w').write(SUB.replace('/x17rerun/audit/run_gate.sh', '/x17rerun/fe/run_gate.sh')
                                .replace('log/gate.log', f'log/{a.tag}.log'))
    open('run_gate.sh', 'w').write(SH)
    os.chmod('run_gate.sh', 0o755)
    print(f'wrote {a.tag}.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
