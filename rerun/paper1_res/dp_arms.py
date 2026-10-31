#!/usr/bin/env python3
"""dp_arms.py bundles|dag <code_dir> -- the X-width systematic arms of the bench resolutions (PASS_PLAN §4: det3 +100,
others +-100 um/sqrt(cm); PASS1_RESOLUTION_2026-10-31.md §4).

The forward model has ONE diffusion constant for both views (hyper Dp, mm/sqrt(ns); Dp^2 = D_T^2 v), so an X-only
width excess cannot be put into a bundle.  The arm shifts D_T by the stated amount in BOTH views: an upper bound on
what the X-width systematic does to the resolutions.  D_T [mm/sqrt(mm)] = Dp / sqrt(v [mm/ns]); 100 um/sqrt(cm) =
0.1 / sqrt(10) mm/sqrt(mm).

  bundles   (on lxplus, in the code archive) copy each golden row's pass bundle (Analysis_pass1/<rel>/wft/calib_bundle)
            to ROOT/<rel>/wft/bundle_<arm> with Dp shifted; prints the Dp values
  dag       writes dp_arms.dag (cwd): per arm RECO (wft.cli reco --matched-only --limit 8000, the G1 / pass-table events,
            code 6a0bb1f = the pass's, so an unshifted arm reproduces the pass table).  Copy rerun/condor/stage1.sub next
            to it and make log/.  Scoring is table level (dp_score.py), on exactly the pass's interior_good events and
            M3 rays: the 6a0bb1f 03_angles has no --fiducial and its budget.py predates the 10-31 fixes.
READ-ONLY on Analysis_pass1 (stage_job --eos-analysis ROOT).
"""
import csv
import math
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('RES_CODE') or os.path.abspath(os.path.join(HERE, '..', '..'))
P1 = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
ROOT = '/eos/user/d/dneff/x17rerun/paper1_res/dp_arms/Analysis'
DT_STEP = 0.1 / math.sqrt(10.0)          # 100 um/sqrt(cm) in mm/sqrt(mm)
#: arm -> (registry key, chamber short name, D_T shift in units of 100 um/sqrt(cm))
ARMS = {'det3_p': ('sat_det3', 'det3', +1), 'det2_p': ('o22_long_det2', 'det2', +1),
        'det2_m': ('o22_long_det2', 'det2', -1)}


def reg():
    return {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv')))}


def rel(r):
    return f"{r['run']}/{r['subrun']}/{r['det']}"


def bundles():
    sys.path.insert(0, REPO)
    from wft.calib import CalibrationBundle
    R = reg()
    for arm, (key, d, k) in ARMS.items():
        r = R[key]
        src = os.path.join(P1, rel(r), 'wft', 'calib_bundle')
        dst = os.path.join(ROOT, rel(r), 'wft', f'bundle_{arm}')
        b = CalibrationBundle.load(src)
        v = float(b.v_drift) * 1e-3                    # mm/ns
        dp0 = float(b.hyper['Dp'])
        dt0 = dp0 / math.sqrt(v)
        dt1 = dt0 + k * DT_STEP
        b.hyper['Dp'] = dt1 * math.sqrt(v)
        b.provenance = dict(b.provenance, paper1_res=f'X-width systematic arm {arm}: D_T {dt0 * math.sqrt(10) * 1e3:.0f}'
                                                     f' -> {dt1 * math.sqrt(10) * 1e3:.0f} um/sqrt(cm) (Dp {dp0:.5f} -> '
                                                     f'{b.hyper["Dp"]:.5f}), both views')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        b.save(dst)
        print(f'{arm}: {key} v {b.v_drift:.2f} um/ns  Dp {dp0:.5f} -> {b.hyper["Dp"]:.5f}  D_T '
              f'{dt0 * math.sqrt(10) * 1e3:.0f} -> {dt1 * math.sqrt(10) * 1e3:.0f} um/sqrt(cm)  -> {dst}')


def dag(code):
    R = reg()
    L = []
    for arm, (key, d, k) in ARMS.items():
        r = R[key]
        pre = f'{key} --eos-analysis {ROOT}'
        w = '{out}/wft'
        t = f'{w}/arm_{arm}/events.parquet'

        def node(name, args, cpus, mem, parents=()):
            L.append(f'JOB {name} stage1.sub')
            L.append(f'VARS {name} code="{code}" args="{pre} {args}" cpus="{cpus}" mem="{mem}" node="{name}"')
            L.append(f'RETRY {name} 3')
            for p in parents:
                L.append(f'PARENT {p} CHILD {name}')

        node(f'RECO_{arm}', f'--module wft.cli -- reco {{key}} --bundle {w}/bundle_{arm} --out {t} --matched-only '
                            f'--limit 8000 --jobs 16', 16, 24000)
    open('dp_arms.dag', 'w').write('\n'.join(L) + '\n')
    print(f'wrote dp_arms.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    if sys.argv[1] == 'bundles':
        bundles()
    else:
        dag(sys.argv[2])
