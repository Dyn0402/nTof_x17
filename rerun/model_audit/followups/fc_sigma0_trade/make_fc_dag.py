#!/usr/bin/env python3
"""make_fc_dag.py <code_dir> [--only trade|headon] -- condor DAG for follow-up FC (items 5 and 6).

Item 6 (the width-tilt trade with everything else on), one node per (chamber, arm) on the golden row's held-out events
[180, 1500) of the W12 it1 cache, through fc_arm.py (= heldout_arm.py + the --dt override):
  every arm: t0 pinned at the model-free latency (ws_t0 bundles: set_w0kw.py --latency-cache --pin-t0, --t0-fix),
  per-event sigma0 choice OFF (sigma_p0_prof_x/y = []), one fixed sigma0 per arm (no per-event choice);
  s24 / s31 / s38 / s45: common sigma0 0.24 / 0.31 / 0.38 / 0.45 mm (the trade);  split: X 0.27, Y 0.33 (two-value
  proposal of ws_sps);  s31free: s31 with the free t0 (what the pin does);  s31g20: s31 on a 20 ns charge grid (K 72),
  in three event chunks.  The position (bow) map is applied offline in trade_analysis.py, cross-validated by halves.
Item 5, fc_headon_grid.py: SPS det4 run_71 RAW head-on at 243 / 156 / 96 V/cm (6 chunks each), the X view again with
the one-sided copy (rc_asym_x 0.13, 243 V/cm), and bench head-on (|tan_ref| < 0.03, up to 2000 per view) on all five chambers (4 chunks each).
Outputs: /eos/user/d/dneff/x17rerun/followups/fc/{trade,headon}/.  Writes fc.dag, fc.sub, run_fc.sh into cwd.
"""
import argparse, os

A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'
LAT = '/eos/user/d/dneff/x17rerun/model_audit/ws_t0/bundles'
SPS = '/eos/user/d/dneff/x17rerun/ws_sps/data'
OUT = '/eos/user/d/dneff/x17rerun/followups/fc'
ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4',
        'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
NOPROF = 'sigma_p0_prof_x=,sigma_p0_prof_y='
ARMS = {  # name -> (sigma_x, sigma_y, t0fix, dt, n_chunks)
    's24': (0.24, 0.24, True, 60, 1), 's31': (0.31, 0.31, True, 60, 1), 's38': (0.38, 0.38, True, 60, 1),
    's45': (0.45, 0.45, True, 60, 1), 'split': (0.27, 0.33, True, 60, 1), 's31free': (0.31, 0.31, False, 60, 1),
    's31g20': (0.31, 0.31, True, 20, 3),
}
SH = '''#!/bin/bash
# run_fc.sh <code_dir> <script relative to the code dir> <args...>
CODE=$1; SCRIPT=$2; shift 2
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE MPLBACKEND=Agg
cd $_CONDOR_SCRATCH_DIR 2>/dev/null || cd /tmp
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/$SCRIPT "$@"
'''
SUB = '''executable     = $ENV(HOME)/x17rerun/fc/run_fc.sh
arguments      = $(code) $(script) $(args)
output         = log/$(nodename).out
error          = log/$(nodename).err
log            = log/fc.log
request_cpus   = 8
request_memory = 14000
+JobFlavour    = "$(flavour)"
should_transfer_files = NO
queue
'''
ARM_PY = 'rerun/model_audit/followups/fc_sigma0_trade/fc_arm.py'
GRID_PY = 'rerun/model_audit/followups/fc_sigma0_trade/fc_headon_grid.py'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('code'); ap.add_argument('--only', default=None)
    a = ap.parse_args(); L = []

    def node(name, script, args, flavour='workday'):
        L.extend([f'JOB {name} fc.sub', f'VARS {name} nodename="{name}" code="{a.code}" script="{script}" '
                  f'args="{args}" flavour="{flavour}"', f'RETRY {name} 1'])

    if a.only in (None, 'trade'):
        for det, row in ROWS.items():
            cache = f'{A}/{row}/wft/w12/it1/cache.pkl'
            for arm, (sx, sy, fix, dt, nch) in ARMS.items():
                k = 24 * 60 // dt
                for c in range(nch):
                    n = 1320 // nch; start = 180 + c * n
                    tag = arm if nch == 1 else f'{arm}_c{c}'
                    args = (f'--cache {cache} --bundle {LAT}/{det} --out {OUT}/trade/{det}_{tag}.npz --start {start} --n {n} '
                            f'--jobs 8 --k {k} --dt {dt} --set {NOPROF},sigma_p0_x={sx},sigma_p0_y={sy}' + (' --t0-fix' if fix else ''))
                    node(f'TR_{det}_{tag}', ARM_PY, args, 'tomorrow' if dt < 60 else 'workday')
    if a.only in (None, 'headon'):
        for fld in ('raw700', 'raw450', 'raw275'):
            for c in range(6):
                args = (f'--kind sps --cache {SPS}/sps_run71_{fld}.pkl --bundle {SPS}/calib_bundle --health {a.code}/rerun/model_audit/ws_sps/results/health.json '
                        f'--nev 800 --chunk {c} 6 --jobs 8 --out {OUT}/headon/sps_{fld}_c{c}.pkl')
                node(f'HO_sps_{fld}_c{c}', GRID_PY, args)
        for c in range(6):
            args = (f'--kind sps --cache {SPS}/sps_run71_raw700.pkl --bundle {SPS}/calib_bundle --health {a.code}/rerun/model_audit/ws_sps/results/health.json '
                    f'--nev 800 --chunk {c} 6 --jobs 8 --views x --xasym 0.13 --out {OUT}/headon/sps_raw700_asym_c{c}.pkl')
            node(f'HO_sps_raw700_asym_c{c}', GRID_PY, args)
        for det, row in ROWS.items():
            for c in range(4):
                args = (f'--kind bench --cache {A}/{row}/wft/w12/it1/cache.pkl --bundle {A}/{row}/wft/calib_bundle --tanmax 0.03 '
                        f'--nev 2000 --chunk {c} 4 --jobs 8 --out {OUT}/headon/bench_{det}_c{c}.pkl')
                node(f'HO_bench_{det}_c{c}', GRID_PY, args)
    if a.only == 'bench05':
        # the wider bench head-on sample (2026-10-27 evening: |tan| < 0.03 and peak <= 3200 left 16-157 views/chamber)
        for det, row in ROWS.items():
            for c in range(4):
                args = (f'--kind bench --cache {A}/{row}/wft/w12/it1/cache.pkl --bundle {A}/{row}/wft/calib_bundle --tanmax 0.05 '
                        f'--pkmax 3500 --nev 2000 --chunk {c} 4 --jobs 8 --out {OUT}/headon/bench05_{det}_c{c}.pkl')
                node(f'HO_bench05_{det}_c{c}', GRID_PY, args)
    open('fc.dag', 'w').write('\n'.join(L) + '\n'); open('fc.sub', 'w').write(SUB); open('run_fc.sh', 'w').write(SH)
    os.chmod('run_fc.sh', 0o755)
    print(f'wrote fc.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
