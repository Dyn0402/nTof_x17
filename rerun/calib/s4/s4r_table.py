#!/usr/bin/env python3
"""s4r_table.py [results/s4r] -- the per-chamber table of the S4 re-check from s4f_fit.py / s4r_gap.py / s4r_inject.py
outputs (r_<det>.json, where_<det>.json, gap_r.json, inj_<det>.json).

The quoted estimator is the head-on stack, masked (pass bundle dead + strip_gain_audit) and interior (M3 reference >= 30 mm
from the active-area edge), with the pass template, at every unsaturated amplitude ('<det>_mask_in30_pass').
v = gap_input / T; pull = (new - input) / the input's stated error (s4_inputs.py's V_ERR, DP_ERR, GAP_ERR).
-> summary.json and a printed table.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 's4r')
PHYS = json.load(open(os.path.join(HERE, '..', 'physics_inputs.json')))
V_ERR = dict(det2=1.2, det3=0.6, det4=0.7, det6=0.8 * 28.0 / 30.0)          # s4_inputs.py
DP_ERR = dict(det2=0.0027, det3=0.0007, det4=0.0011, det6=0.0044)
GAP_ERR = dict(det2=0.42, det3=0.19, det4=0.44, det6=1.0)
S4 = json.load(open(os.path.join(HERE, 'results', 's4_inputs.json')))


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def v_of(gap, arm):
    if not arm or 'T' not in arm:
        return np.nan, np.nan
    return gap / (arm['T'] * 1e-3), gap / (arm['T'] * 1e-3) * arm['T_err'] / arm['T']


out = {}
gaps = load('gap_r.json')
for det in ('det2', 'det3', 'det6', 'det4'):
    P = PHYS[det]
    W = load(f'where_{det}.json')
    r = load(f'r_{det}.json')
    q = W.get(f'{det}_mask_in30_pass', {})
    v, ve = v_of(P['gap_mm'], q)
    row = dict(v_in=P['v_timing_um_ns'], v_err_in=V_ERR[det], Dp_in=P['Dp'], Dp_err_in=DP_ERR[det],
               gap_in=P['gap_mm'], gap_err_in=GAP_ERR[det], n=q.get('n'), T=q.get('T'), T_err=q.get('T_err'),
               v=v, v_err=ve, v_pull=(v - P['v_timing_um_ns']) / V_ERR[det],
               Dp=q.get('Dp'), Dp_err=q.get('Dp_err'),
               Dp_pull=(q['Dp'] - P['Dp']) / DP_ERR[det] if 'Dp' in q else np.nan,
               s4_v=S4[det]['v_stack'], s4_Dp=S4[det]['stack_Dp'], s4_gap=S4[det].get('gap_new'))
    # the steps from S4 as run to the quoted arm (T in ns)
    T = lambda d, k: d.get(k, {}).get('T', np.nan)
    row['steps_T'] = dict(
        s4=S4[det]['stack_T'],
        mask_all=T(r, f'{det}_mask_pass'),
        none_in30=T(W, f'{det}_none_in30_pass'), mask_in30=T(W, f'{det}_mask_in30_pass'),
        mask_in30_old=T(W, f'{det}_mask_in30_old'), mask_out30=T(W, f'{det}_mask_out30_pass'),
        mask_in20=T(W, f'{det}_mask_in20_pass'), mask_in40=T(W, f'{det}_mask_in40_pass'))
    row['amp_T'] = {k: T(W, f'{det}_{k}_pass') for k in
                    ('none_amp0-1000_all', 'none_amp1000-3500_all', 'mask_amp0-1000_all', 'mask_amp1000-3500_all',
                     'mask_amp0-1000_in30', 'mask_amp1000-3500_in30', 'mask_amp0-1000_out30', 'mask_amp1000-3500_out30')}
    row['template_dT'] = row['steps_T']['mask_in30'] - row['steps_T']['mask_in30_old']
    g = gaps.get(det, {})
    if g:
        gi = g.get('mask_in30', {})
        row['gap'] = gi.get('gap'); row['gap_err'] = gi.get('gap_err')
        row['gap_pull'] = (gi['gap'] - P['gap_mm']) / GAP_ERR[det] if 'gap' in gi else np.nan
        row['gap_arms'] = {k: g[k].get('gap') for k in g}
    inj = load(f'inj_{det}.json')
    if inj:
        a = inj['arms']
        row['inject_T'] = {s: float(np.mean([m['T'] for k, m in a.items() if k.startswith(f's{s}_')]))
                           for s in ('1', '0.5', '0.3')}
    out[det] = row
    print(f'{det}: n {row["n"]}  T {row["T"]:.1f}±{row["T_err"]:.1f}  v {v:.2f}±{ve:.2f} (in {P["v_timing_um_ns"]}±{V_ERR[det]:.2f}, '
          f'pull {row["v_pull"]:+.2f}; S4 {row["s4_v"]:.2f})  Dp {row["Dp"]:.4f}±{row["Dp_err"]:.4f} (pull {row["Dp_pull"]:+.2f})  '
          f'gap {row.get("gap", np.nan):.2f}±{row.get("gap_err", np.nan):.2f} (pull {row.get("gap_pull", np.nan):+.2f})  '
          f'template dT {row["template_dT"]:+.1f} ns')
    print('   steps', {k: round(x, 1) for k, x in row['steps_T'].items()})
    print('   amp  ', {k: round(x, 1) for k, x in row['amp_T'].items()})
    if 'inject_T' in row:
        print('   inject', {k: round(x, 1) for k, x in row['inject_T'].items()})
json.dump(out, open(os.path.join(R, 'summary.json'), 'w'), indent=1, default=float)
