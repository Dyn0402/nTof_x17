"""r2_01_extract.py -- pull a compact per-track table (timing quantiles included) from the n_TOF pass-1 analysis view.
Read only on ntof_pass1.  Writes $WORK/audit/r2/tr_cosmic.parquet (beam-off sub-runs, registry class cosmic) and
tr_late.parquet (beam, t_since_flash >= 20 ms).  Rows: gated (true |tan|<0.5, no censored window) tracks, all four arms."""
import sys, os
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17')
import numpy as np, pandas as pd
import pyarrow.dataset as ds
from sept26_prelim_analysis import pass1_tracks as P1
OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/r2'
COLS = ['run', 'subrun', 'tag', 'event_id', 'arm', 'gated', 'censored_window', 'x_local', 'y_local', 'x_p0', 'y_p0',
        'x_tan_theta', 'y_tan_theta', 'tanx', 'tany', 'x_w', 'y_w', 'x_t0', 'y_t0', 'x_q_u50', 'y_q_u50', 'x_q_u90', 'y_q_u90',
        'x_q_uend', 'y_q_uend', 'x_q_sum', 'y_q_sum', 'x_n_strips', 'y_n_strips', 'chi2dof_x', 'chi2dof_y',
        'x_cens_frac', 'y_cens_frac', 'x_quality_ok', 'y_quality_ok', 'x_slope_reliable', 'y_slope_reliable',
        'x_isochronous', 'y_isochronous', 'drift_len_mm', 'v_t', 'drift_t_end_ns', 'drift_railed', 'n_cand_x', 'n_cand_y',
        't_since_flash_ns', 'x_p0_err', 'x_tan_err', 'y_p0_err', 'y_tan_err', 'tan_sane']
which = sys.argv[1]
runs = sorted({f.name.split('_stat')[0].split('_cosbounce')[0].replace('tracks_', '') for f in P1.files()})
parts = []
for r in runs:
    if which == 'cosmic':
        flt = P1.cosmic_filter() & ds.field('gated')
    else:
        flt = (ds.field('t_since_flash_ns') >= 2e7) & ds.field('gated')
    try:
        d = P1.read(COLS, runs=[r], filter=flt)
    except Exception as e:
        print(r, 'ERR', e); continue
    if len(d) == 0: continue
    d = d[~d.censored_window.fillna(False).astype(bool)]
    for c in d.columns:
        if d[c].dtype == np.float64: d[c] = d[c].astype(np.float32) if c not in ('x_q_sum', 'y_q_sum') else d[c].clip(upper=1e9).astype(np.float32)
    parts.append(d); print(r, len(d), flush=True)
D = pd.concat(parts, ignore_index=True)
D.to_parquet(f'{OUT}/tr_{which}.parquet', index=False)
print(which, len(D), D.groupby('arm').size().to_dict())
