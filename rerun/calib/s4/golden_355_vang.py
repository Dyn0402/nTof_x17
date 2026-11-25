#!/usr/bin/env python3
"""golden_355_vang.py -- the angle-scale velocity v_ang of the 355 V/cm (1000 V) golden point of chamber A, on the per-mounting
angle map (bench re-finalization, 2026-11-25).

results/s4r_run2/golden_355.json holds the head-on stack fit of Run 1 (sat_det3) and Run 2 (p2 night) summed (T, written
2026-11-24) and v_ang = v_drift x mean(kw) of each run.  Until 2026-11-24 Run 1's v_ang was the hv_scan2 490 V / 1000 V
ladder of rerun/paper1_gas (a short scan row, ~340 tracks), and Run 2's the pass-1 bundle.  Here both come from the run's
own map1 bundle (the row the stack was fitted on): v_ang = v_drift x 0.5 (kw_x + kw_y), the constants the map's angles use.
The ladder of vt_common (map interior, M3 reference) is computed too, as a check of the applied kw.  The relative error of
v_ang is kept at the earlier 0.47 % (the spread between kw estimators is of that size; the ladders' statistical errors
are 0.07-0.12 %).  T is not touched.  The pass-1 values are kept as *_pass1.

    .venv/bin/python rerun/calib/s4/golden_355_vang.py <Analysis_pass1 root with the two rows' wft/> [--map1 <map1 root>]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'vt_within_rows')]
import vt_common as V  # noqa: E402

V.NAME.setdefault('mx17_3', 'det3')
OUT = os.path.join(HERE, 'results', 's4r_run2', 'golden_355.json')
ROWS = dict(run1=('mx17_det3_saturday_scan_6-27-26', 'long_run_resist_490V_drift_1000V'),
            run2=('mx17_det3_p2_det1_overnight_6-27-26', 'long_run_p2_det1_sanity_check'))
REL_ERR = 0.17286123425804875 / 36.665191381098026        # the earlier relative error (hv_scan2 ladder), 0.47 %


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--map1', default=os.path.join(os.environ.get('WORK_DATA', '/media/ucla/work'), 'claude-jobs/d006ab29/map1'))
    a = ap.parse_args()
    G = json.load(open(OUT))
    for r, (run, sub) in ROWS.items():
        R = V.load_row(os.path.join(a.root, run, sub, 'mx17_3', 'wft'), a.root, map1_root=a.map1)
        cal = R['cal']
        assert cal.map1, f'{r}: no map in the map1 bundle'
        lad = {}
        for p in 'xy':
            vw = R['views'][p]
            w0, kw, n, ke = V.ladder(cal, vw['w'], (vw['tan_ref'] + vw['a']) * vw['S'])
            lad[p] = dict(kw_est=kw, kw_err=ke, n=n, kw_applied=cal.kw[p])
        va = cal.v_drift * 0.5 * (cal.kw['x'] + cal.kw['y'])
        g = G[r]
        if 'v_ang_pass1' not in g:
            g['v_ang_pass1'], g['v_ang_err_pass1'] = g['v_ang'], g['v_ang_err']
            g['gap_pass1'], g['gap_err_pass1'] = g['gap'], g['gap_err']
        g['v_ang'], g['v_ang_err'] = va, va * REL_ERR
        g['gap'] = va * g['T'] / 1e3
        g['gap_err'] = g['gap'] * np.hypot(REL_ERR, g['T_err'] / g['T'])
        g['ladder_map1'] = lad
        g['v_ang_source'] = f'map1 bundle of {run}/{sub} (v_drift x mean kw)'
        print(r, f"v_ang {g['v_ang_pass1']:.3f} -> {va:.3f}, gap {g['gap_pass1']:.2f} -> {g['gap']:.2f}", lad)
    s, f1 = G['sum'], G['sum']['f_run1']
    if 'v_ang_pass1' not in s:
        s['v_ang_pass1'], s['v_ang_err_pass1'], s['gap_pass1'], s['gap_err_pass1'] = s['v_ang'], s['v_ang_err'], s['gap'], s['gap_err']
    s['v_ang'] = f1 * G['run1']['v_ang'] + (1 - f1) * G['run2']['v_ang']
    s['v_ang_err'] = np.hypot(f1 * G['run1']['v_ang_err'], (1 - f1) * G['run2']['v_ang_err'])
    s['gap'] = s['v_ang'] * s['T'] / 1e3
    s['gap_err'] = s['gap'] * np.hypot(s['v_ang_err'] / s['v_ang'], s['T_err'] / s['T'])
    print('sum', f"v_ang {s['v_ang_pass1']:.3f} -> {s['v_ang']:.3f}, gap {s['gap_pass1']:.2f} -> {s['gap']:.2f} +- {s['gap_err']:.2f}")
    G['note_map1'] = ('2026-11-25: v_ang of both runs from the run\'s own map1 bundle (bench re-finalization), Run 1 no longer '
                      'from the hv_scan2 ladder; relative error kept at 0.47 %; pass-1 values in *_pass1 '
                      '(rerun/calib/s4/golden_355_vang.py).')
    json.dump(G, open(OUT, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
