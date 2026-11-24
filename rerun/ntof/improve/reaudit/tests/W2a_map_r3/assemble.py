#!/usr/bin/env python3
"""assemble.py -- r3_validation.csv (long format: section, arm, view, region, model, metric, value, err, note)."""
import numpy as np, pandas as pd
from pathlib import Path
H = Path(__file__).resolve().parent
rows = []
def add(sec, arm, view, region, model, metric, value, err=np.nan, note=''):
    rows.append(dict(section=sec, arm=arm, view=view, region=region, model=model, metric=metric, value=value, err=err, note=note))
for arm in 'AC':
    T = pd.read_csv(H / f't1_models_{arm}.csv')
    for _, r in T.iterrows():
        note = 'IN-SAMPLE (r2 trained on all lines)' if r.model == 'r2_file_insample' else 'held-out, run parity'
        add('cosmic_heldout', arm, r["view"], 'all', r.model, 'sigma_clean', r.sig_clean, r.sig_clean_err, note)
        for m in ('w68', 'w95', 'sig_edge', 'w68_edge', 'w95_edge', 'sig_int', 'w68_int', 'w95_int', 'sig_all', 'tail_all'):
            add('cosmic_heldout', arm, r["view"], 'all', r.model, m, r[m], np.nan, note)
    d = pd.read_csv(H / f't1_diffs_{arm}.csv')
    for _, r in d.iterrows():
        add('cosmic_heldout_diff', arm, r["view"], r.region, f'{r.a} - {r.b}', 'd_sigma_clean', r.d_sig, r.err, 'run-block bootstrap')
for arm in 'ACD':
    T = pd.read_csv(H / f't2_models_{arm}.csv')
    for _, r in T.iterrows():
        for m in ('sig', 'w68', 'w95'):
            add('joint_pointing_heldout', arm, r["view"], r.region, r.model, m, r[m], np.nan,
                'held-out lines, run parity; covered = 25 mm cells with valid pointing; cells_current = angle_map algorithm (D 100 mm)')
    d = pd.read_csv(H / f't2_diffs_{arm}.csv')
    for _, r in d.iterrows():
        add('joint_pointing_diff', arm, r["view"], r.region, f'{r.a} - {r.b}', 'd_sigma_clean', r.d_sig, r.err, 'run-block bootstrap')
T = pd.read_csv(H / 'd_event.csv')
for _, r in T.iterrows():
    for m in ('sig', 'w68', 'w95', 'sig_edge', 'sig_int'):
        add('D_heldout_event_parity', 'D', r["view"], 'all', r.model, m, r[m], np.nan, "T6's event-parity protocol")
B = pd.read_csv(H / 'beam_angle_map_r3_candidate_r3_joint_split.csv')
for _, r in B.iterrows():
    note = 'beam late >= 25 ms; *_split = pointing trained on the other run group'
    for m in ('core_tan', 'cellrms', 'cellrms_detr', 'w68_mm', 'f15', 'exc15', 'x_cross', 'focus_s', 'focus_med',
              'core_tan_g0', 'core_tan_g1', 'cellrms_detr_g0', 'cellrms_detr_g1', 'w68_mm_g0', 'w68_mm_g1', 'n', 'n_mm'):
        err = r.x_cross_err if m == 'x_cross' else (r.exc15_err if m == 'exc15' else np.nan)
        add('beam', r.arm, 'x', 'all', r.map, m, r[m], err, note)
N = pd.read_csv(H / 'noise_floor_fit.csv')
for _, r in N.iterrows():
    add('learning_curve', r.arm, r["view"], 'all', 'spl_pot_sep', f'sig_inf ({r.model})', r.sig_inf, r.sig_inf_err, f'q={r.q:.2f}')
    add('learning_curve', r.arm, r["view"], 'all', 'spl_pot_sep', f'sig_at_2x_lines ({r.model})', r.sig_at_2x, np.nan, '')
pd.DataFrame(rows).to_csv(H / 'r3_validation.csv', index=False)
print(len(rows))
