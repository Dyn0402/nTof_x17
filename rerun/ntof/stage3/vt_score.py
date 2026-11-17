#!/usr/bin/env python3
"""vt_score.py -- does the v(t) correction move the N-G1 numbers it should?  (NTOF_PASS_PLAN P7a, 11-17 step 3)

Two of rerun/ntof/ng1's scores, frozen v against v(t), on the same tracks:

  (kappa = v_angle / v_t per chamber and view, vt_kappa.json: run_149's 87 sub-runs are 174 of the 476 sub-run
  views kappa was fitted on, so the line after correction is not independent; the independent checks are the slope
  in vt_line_campaign.py and srcy on beam.)
  line   run_149 cosmic: ac_line.score's angle-scale ratio fitted/true per view (the A-C line is built from mesh
         positions, which do not depend on v, so it is an external truth for the angle scale).  The pass's candidates
         (ng1_pass/pass/, the EOS stage-2 copy) with tan_theta scaled per acquisition by v_bundle / v_t (cosmic
         trigger), against the same candidates unscaled.  Variants: frozen (reproduces ng1 line_run149.json), vt,
         and vt with the |tan_vt| < 0.5 acceptance on the reco gate.
  srcy   run_145 beam: y_guard.late_y_crossing (late band crossing, the stage-3 guard's estimator) on the stage-3
         tables built with v(t) (ntof_stage3.py --out <dir>): frozen columns, frozen columns inside |tan| < 0.5
         (the acceptance alone), and the `_vt` columns with gated_vt.

    vt_score.py [--stage3 $WORK_DATA/ntof_pass1/stage3_vt_test] [--out rerun/ntof/stage3/vt_score]

Expected (handoff 11-17): C's 4-5 % steep scale moves toward 1; source y unchanged within its +-3 mm systematic.
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(NT, '..', '..'))
sys.path[:0] = [HERE, NT, REPO, os.path.join(REPO, 'sept26_prelim_analysis')]
warnings.filterwarnings('ignore')

import vt_join as VT  # noqa: E402

WD = os.environ.get('WORK_DATA') or '/media/ucla/work'
PASS = f'{WD}/ntof_pass1/ng1_pass/pass'
RUNS = '/media/dylan/data/x17/beam_july/runs'


def line(vt, out):
    import ac_line
    v = vt[(vt.run == '149') & (vt.trigger == 'cosmic')].set_index(['subrun', 'tag', 'arm'])
    kap = VT.load_kappa()
    res = {}
    for variant in ('frozen', 'vt_nokappa', 'vt', 'vt_acc05'):
        with tempfile.TemporaryDirectory() as tmp:
            for arm in 'AC':
                parts = []
                for sub in sorted(os.listdir(f'{PASS}/run_149')):
                    for f in sorted(glob.glob(f'{PASS}/run_149/{sub}/{arm}/*_{arm}.candidates.parquet')):
                        c = pd.read_parquet(f)
                        tags = c['acq_tag'].unique()
                        if len(tags) != 1:
                            raise ValueError(f'{f}: {len(tags)} acq_tags')
                        if variant != 'frozen':
                            row = v.loc[(sub, tags[0], arm)]
                            k = c['plane'].map({'x': kap[(arm, 'x')], 'y': kap[(arm, 'y')]}).to_numpy(float)
                            if variant == 'vt_nokappa':
                                k = 1.0
                            c['tan_theta'] = c['tan_theta'] * row['v_bundle'] / (k * row['v_fit'])
                            if variant == 'vt_acc05':
                                c['track_gated'] = c['track_gated'] & (c['tan_theta'].abs() < VT.TAN_MAX_VT)
                        c['event_id'] = c['event_id'] + 10_000_000 * int(sub[-4:])
                        parts.append(c)
                pd.concat(parts, ignore_index=True).to_parquet(f'{tmp}/{arm}.candidates.parquet')
            o, _ = ac_line.score(f'{tmp}/A.parquet', f'{tmp}/C.parquet', f'{RUNS}/run_149/run_config.json',
                                 label=variant)
        res[variant] = {k: x for k, x in o.items() if k != 'bins'}
        pd.DataFrame(o['bins']).to_csv(f'{out}/line_bins_run149_{variant}.csv', index=False)
    vv = v.reset_index()
    res['v_t_over_v_bundle_run149'] = {a: float((g.v_fit / g.v_bundle).median()) for a, g in vv.groupby('arm')}
    return res


def srcy(stage3):
    import y_guard
    fs = sorted(glob.glob(f'{stage3}/tracks_run_145_*.parquet'))
    if not fs:
        raise FileNotFoundError(f'no run_145 stage-3 tables in {stage3}')
    cols = ['event_id', 'arm', 'gated', 'gated_reco', 'gated_vt', 'angle_calibrated', 'tanx', 'tany', 'p0_x', 'p0_y',
            'p0_z', 'd_x', 'd_y', 'd_z', 'tanx_vt', 'tany_vt', 'd_x_vt', 'd_y_vt', 'd_z_vt', 't_since_flash_ns',
            'x_t0', 'y_t0', 'v_t', 'v_drift_um_ns']
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
    d = d[d.arm.isin(['A', 'C'])]
    base = d[d.gated.astype(bool)]
    acc = base[(base.tanx.abs() < 0.5) & (base.tany.abs() < 0.5)]
    w = d[d.gated_vt.astype(bool)].copy()
    for c in ('tanx', 'tany', 'd_x', 'd_y', 'd_z'):
        w[c] = w[f'{c}_vt']
    w['gated'] = w['gated_reco'] = True
    r = {'frozen': y_guard.late_y_crossing(base), 'frozen_acc05': y_guard.late_y_crossing(acc),
         'vt': y_guard.late_y_crossing(w), 'n_files': len(fs)}
    r['v_t_over_v_used_run145'] = {a: float((g.v_t / g.v_drift_um_ns).median()) for a, g in d.groupby('arm')}
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage3', default=f'{WD}/ntof_pass1/stage3_vt_test')
    ap.add_argument('--vt', default=f'{WD}/ntof_pass1/vt/vt_smooth.parquet')
    ap.add_argument('--out', default=os.path.join(HERE, 'vt_score'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vt = VT.load_vt(a.vt)
    res = {'line_run149': line(vt, a.out), 'srcy_run145': srcy(a.stage3)}
    json.dump(res, open(f'{a.out}/vt_score.json', 'w'), indent=1, default=float)
    L = res['line_run149']
    print('A-C line ratio fitted/true (run_149):')
    for k in ('Ax', 'Ay', 'Cx', 'Cy'):
        print(f'  {k}: ' + '  '.join(f"{v}: {L[v][k]['ratio']:.4f}±{L[v][k]['ratio_err']:.4f} (n {L[v][k]['n_kw']})"
                                     for v in ('frozen', 'vt_nokappa', 'vt', 'vt_acc05')))
    print('  v_t / v_bundle run_149:', L['v_t_over_v_bundle_run149'])
    S = res['srcy_run145']
    print('source y crossing (run_145, late):')
    for v in ('frozen', 'frozen_acc05', 'vt'):
        print(f'  {v}: ' + '  '.join(f"{arm} {S[v][arm].get('y_cross_mm')}±{S[v][arm].get('err_mm')} (n {S[v][arm]['n']})"
                                     for arm in 'AC'))


if __name__ == '__main__':
    main()
