"""vt_join.py -- the P7a drift-velocity correction v(t), joined onto a stage-3 track table as `_vt` columns.

Decided 2026-11-17 (Dylan): no reconstruction rerun.  The A/B test (condor 4677861, `../vt/README.md`) showed that v
does not enter the waveform fit and that tan scales as v_used / v to 0.1-0.5 %, so the correction is exact at table
level:

    tanx_vt = tanx * v_used / (kappa_x * v_t)        (same for y with kappa_y)

with v_used the velocity the tans were computed at (the row's `v_drift_um_ns`; k = 1 on A, C, D), v_t the smooth
model's `v_fit` for the row's acquisition, arm and trigger class (`vt_smooth.parquet`, EOS Analysis_pass1/vt_table/),
and kappa = v_angle / v_t, one constant per (chamber, view).  v_t is a TIMING velocity (30 mm / column duration); the
A-C cosmic line, which measures the angle velocity from positions alone, follows it in time with slope 0.84-0.97
(+-0.02-0.08) but sits a constant 2-3 % below it (kappa A 0.968 x / 0.973 y, C 0.978 / 0.979; vt_line_campaign.py ->
vt_score/vt_kappa.json, 2026-11-17).  Without kappa the correction overshoots: run_149 line ratio C 1.04 -> 0.98.
D has no A-C line: kappa_D = 1.  One acquisition is ~5 min and the curve varies over hours, so per acquisition is
enough; no per-event time is used.

Every angle-derived column (build_tracks.ANGLE_DERIVED) gets a `_vt` twin computed by build_tracks' own functions on
the corrected tans, so the direction, the pointing, the path length and the scintillator predictions all sit on one
calibration.  Positions at the mesh (p0, x_local, y_local) do not depend on v and have no twin.  `drift_len_mm_vt`
converts the same drift time with v_t.

Acceptance: `in_acceptance_vt` = |tanx_vt|, |tany_vt| < 0.5 true, `gated_vt = gated_reco & in_acceptance_vt`.  0.5
rather than 0.6 because the reco cut is |tan_raw| < 0.6 at the frozen v: at the fastest times (C up to +15 %) its true
reach falls to ~0.52, so a 0.6 cut would move with v (the A/B: C's 0.5-0.6 band changes by +46 %).  The ~1 % track
reshuffle at the fastest times (candidate ranking via the column-end window) is not undone by this and is a stated
systematic.

B has no angle (positions only): its `_vt` columns are null and gated_vt = gated (= gated_reco).  D's v_t on beam
is the model's prediction from D's cosmic sub-runs (D has no beam v measurement), and D's absolute scale reads ~10 %
off its bundle even on its own run: it needs an angle anchor.  The sidecar says so.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

KAPPA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vt_score', 'vt_kappa.json')
TAN_MAX_VT = 0.5
VT_ARMS = ('A', 'C', 'D')
TRIGGER_OF_RCLASS = {'production': 'beam', 'cosmic': 'cosmic'}
D_CAVEAT = ('D: v_t on beam is predicted from D cosmic sub-runs (no beam v measurement on D), and D absolute '
            'angle scale reads ~10 % off its bundle even on its own run: needs an angle anchor')
SYST_NOTE = ('the frozen v also chose which candidate ranks first: ~1 % of tracks at the fastest times would differ '
             'in a v(t) re-reconstruction (A/B condor 4677861; counts change < 0.6 % inside |tan| < 0.5). '
             'Stated systematic, not corrected.')


def load_kappa(path=KAPPA_FILE) -> dict:
    """{(arm, view): kappa}; D = 1 (no A-C line)."""
    k = json.load(open(path))['kappa']
    out = {(a[0], a[1]): float(v['kappa']) for a, v in k.items()}
    out.update({('D', 'x'): 1.0, ('D', 'y'): 1.0})
    return out


def load_vt(path) -> pd.DataFrame:
    v = pd.read_parquet(path, columns=['run', 'subrun', 'tag', 'arm', 'trigger', 'v_fit', 'v_fit_err', 'v_bundle',
                                       'measured'])
    v['run'] = v['run'].astype(str)
    if v.duplicated(['run', 'subrun', 'tag', 'arm', 'trigger']).any():
        raise ValueError('vt table has duplicate (run, subrun, tag, arm, trigger) keys')
    return v


def attach_vt(tracks: pd.DataFrame, vt: pd.DataFrame, trigger: str, trs: dict,
              kappa: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Add the `_vt` columns to one sub-run's stage-3 table.  `trs` = geometry.detector_transforms(run_config)."""
    from sept26_prelim_analysis import build_tracks as BT
    from ntof_tracking.reco import geometry as G
    kappa = load_kappa() if kappa is None else kappa

    run = str(tracks['run'].iloc[0])
    sub = tracks['subrun'].iloc[0]
    v = vt[(vt['run'] == run.replace('run_', '')) & (vt['subrun'] == sub) & (vt['trigger'] == trigger)]
    t = tracks.merge(v[['tag', 'arm', 'v_fit', 'v_fit_err', 'v_bundle', 'measured']]
                     .rename(columns={'v_fit': 'v_t', 'v_fit_err': 'v_t_err', 'v_bundle': 'v_t_bundle',
                                      'measured': 'v_t_measured'}),
                     on=['tag', 'arm'], how='left', validate='many_to_one')
    if len(t) != len(tracks):
        raise AssertionError('vt join changed the row count')
    t.loc[t['arm'] == 'B', ['v_t', 'v_t_err', 'v_t_bundle']] = np.nan
    t['v_t_measured'] = t['v_t_measured'].astype('boolean')
    t.loc[t['arm'] == 'B', 'v_t_measured'] = pd.NA

    out_cols = [f'{c}_vt' for c in BT.ANGLE_DERIVED]
    for c in out_cols:
        t[c] = np.nan
    for c in ('pred_plastic_vt', 'pred_ls_vt'):       # None for "no prediction", as build_tracks writes it
        t[c] = pd.Series([None] * len(t), index=t.index, dtype=object)
    t['in_bore_vt'] = pd.array([pd.NA] * len(t), dtype='boolean')

    info = {'trigger': trigger, 'tan_max_true_vt': TAN_MAX_VT, 'by_arm': {}}
    for arm in VT_ARMS:
        m = (t['arm'] == arm).to_numpy()
        if not m.any():
            continue
        a = t.loc[m]
        miss = int(a['v_t'].isna().sum())
        if miss:
            raise KeyError(f'{run}/{sub} arm {arm}: {miss} rows have no v(t) for trigger {trigger!r}')
        v_used = a['v_drift_um_ns'].to_numpy(float)
        if not np.all(np.isfinite(v_used)):
            raise ValueError(f'{run}/{sub} arm {arm}: rows without v_drift_um_ns')
        n_bundle_diff = int((np.abs(v_used - a['v_t_bundle'].to_numpy(float)) > 1e-6).sum())
        r = v_used / a['v_t'].to_numpy(float)
        rx, ry = r / kappa[(arm, 'x')], r / kappa[(arm, 'y')]
        # build_tracks' own geometry on the corrected tans: k = 1 on the rescaled raw tans
        g = a[['x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta', 'x_q_uend', 'y_q_uend', 'x_q_sum', 'y_q_sum']].copy()
        g['x_tan_theta'] *= rx
        g['y_tan_theta'] *= ry
        g = BT.local_and_global(g, trs[G.DET_NAME[arm]], k=1.0)
        vt_mm = a['v_t'].to_numpy(float) / 1000.0
        g['drift_t_end_ns'] = a['drift_t_end_ns'].to_numpy(float)
        g['drift_len_mm'] = g['drift_t_end_ns'] * vt_mm
        sec = np.sqrt(1.0 + g['tanx'] ** 2 + g['tany'] ** 2)
        g['path_len_mm'] = g['drift_len_mm'] * sec
        with np.errstate(divide='ignore', invalid='ignore'):
            qpl = a['q_total'].to_numpy(float) / g['path_len_mm'].replace(0, np.nan).to_numpy(float)
        g['q_per_len'] = np.where(a['drift_railed'].to_numpy(bool), np.nan, qpl)
        g = BT.pointing(g)
        g = BT.predictions(g)
        for c in BT.ANGLE_DERIVED:
            t.loc[m, f'{c}_vt'] = g[c].to_numpy()
        info['by_arm'][arm] = dict(
            v_used=sorted(set(np.round(v_used, 4).tolist())), v_t_bundle=sorted(set(a['v_t_bundle'].round(4))),
            n_rows_v_used_ne_vt_bundle=n_bundle_diff,
            ratio_v_used_over_v_t=dict(min=float(r.min()), median=float(np.median(r)), max=float(r.max())),
            kappa_x=kappa[(arm, 'x')], kappa_y=kappa[(arm, 'y')],
            frac_v_t_measured=float(a['v_t_measured'].astype(float).mean()))

    with np.errstate(invalid='ignore'):
        acc = ((np.abs(t['tanx_vt'].to_numpy(float)) < TAN_MAX_VT)
               & (np.abs(t['tany_vt'].to_numpy(float)) < TAN_MAX_VT))
    t['in_acceptance_vt'] = pd.array(acc, dtype='boolean')
    b = (t['arm'] == 'B').to_numpy()
    t.loc[b, 'in_acceptance_vt'] = pd.NA
    t['gated_vt'] = t['gated_reco'].to_numpy(bool) & acc
    t.loc[b, 'gated_vt'] = t.loc[b, 'gated'].to_numpy(bool)
    t['in_bore_vt'] = t['in_bore_vt'].astype('boolean')
    info['n_gated_vt'] = int(t['gated_vt'].sum())
    info['n_gated_vt_by_arm'] = {k: int(x) for k, x in t.groupby('arm')['gated_vt'].sum().items()}
    return t, info


def meta_block(info: dict, vt_path, vt_md5: str) -> dict:
    return dict(
        source=str(vt_path), md5=vt_md5, column='v_fit (smooth model: spline + water + mixer T, vt_smooth.py)',
        formula='tan{x,y}_vt = tan{x,y} * v_drift_um_ns / (kappa_{x,y} * v_t); every ANGLE_DERIVED column has a _vt '
                'twin from build_tracks. v_t is the timing v; kappa = v_angle / v_t from the A-C cosmic line '
                '(vt_line_campaign.py), D kappa = 1',
        kappa_file=KAPPA_FILE, kappa_json=json.load(open(KAPPA_FILE)),
        acceptance=f'in_acceptance_vt = |tanx_vt|, |tany_vt| < {TAN_MAX_VT} true; gated_vt = gated_reco & '
                   'in_acceptance_vt (B: gated_vt = gated, no angle)',
        keep='tan*, in_acceptance (0.6) and gated are unchanged (frozen bundle v)',
        d_caveat=D_CAVEAT, systematic=SYST_NOTE, decided='2026-11-17 (Dylan): no rerun, table-level correction',
        **info)
